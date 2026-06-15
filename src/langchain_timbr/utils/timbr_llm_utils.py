from typing import Any, Optional
from langchain_core.language_models.llms import LLM
from datetime import datetime
import concurrent.futures
import contextvars
import json
import re

try:
    from langsmith import traceable as ls_traceable
    _LANGSMITH_AVAILABLE = True
except ImportError:
    def ls_traceable(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        def decorator(func):
            return func
        return decorator
    _LANGSMITH_AVAILABLE = False

from .general import parse_list
from .timbr_utils import get_datasources, get_tags, get_concepts, get_concept_properties, validate_sql, get_properties_description, get_relationships_description, encrypt_prompt, get_ontology_description
from .prompt_service import (
    get_determine_concept_prompt_template,
    get_generate_sql_prompt_template,
    get_generate_sql_reasoning_prompt_template,
    get_qa_prompt_template
)
from .. import config

def _clean_snowflake_prompt(prompt: Any) -> None:
    import re

    def clean_func(prompt_content: str) -> str:
        raw = prompt_content
        # 1. Normalize Windows/Mac line endings → '\n'
        raw = raw.replace('\r\n', '\n').replace('\r', '\n')

        # 2. Collapse any multiple blank lines → single '\n'
        raw = re.sub(r'\n{2,}', '\n', raw)

        # 3. Convert ALL real '\n' → literal backslash-n
        raw = raw.replace('\n', '\\n')

        # 4. Normalize curly quotes to straight ASCII
        raw = (raw
            .replace('’', "'")
            .replace('‘', "'")
            .replace('“', '"')
            .replace('”', '"'))

        # 5. Collapse any accidental double-backticks → single backtick
        raw = raw.replace('``', '`')

        # 6. Escape ALL backslashes so '\\n' survives as two chars
        raw = raw.replace('\\', '\\\\')

        # 7. Escape single-quotes for SQL string literal
        raw = raw.replace("'", "''")

        # 8. Escape double-quotes for SQL string literal
        raw = raw.replace('"', '\\"')

        return raw

    prompt[0].content = clean_func(prompt[0].content)  # System message
    prompt[1].content = clean_func(prompt[1].content)  # User message


def _call_llm_with_timeout(llm: LLM, prompt: Any, timeout: int = 120) -> Any:
    """
    Call LLM with timeout to prevent hanging.

    Args:
        llm: The LLM instance
        prompt: The prompt to send
        timeout: Timeout in seconds (default: 120)

    Returns:
        LLM response

    Raises:
        TimeoutError: If the call takes longer than timeout seconds
        Exception: Any other exception from the LLM call
    """
    ctx = contextvars.copy_context()

    def _llm_call():
        return ctx.run(llm.invoke, prompt)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(_llm_call)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"LLM call timed out after {timeout} seconds")
        except Exception as e:
            raise e

MEASURES_DESCRIPTION = "The following columns are calculated measures and can only be aggregated with an aggregate function: COUNT/SUM/AVG/MIN/MAX (count distinct is not allowed)"
TRANSITIVE_RELATIONSHIP_DESCRIPTION = "Transitive relationship columns match pattern \"<relationship>[<concept>*N].<column>\". The *N is a PLACEHOLDER you must rewrite: if the question specifies a depth set N to that number; otherwise keep the schema's N. Use the SAME N across the query. Example: if schema shows `<rel>[<concept>*2].<col>` and the question asks for 3 levels, write `<rel>[<concept>*3].<col>`. Do NOT add `<relationship>_transitivity_level BETWEEN 1 AND N` — *N already bounds traversal. Only filter `<relationship>_transitivity_level` to exclude levels: =1 for direct only, >1 for indirect only."

def _prompt_to_string(prompt: Any) -> str:
    prompt_text = ''
    if isinstance(prompt, str):
        prompt_text = prompt
    elif isinstance(prompt, list):
        for message in prompt:
            if hasattr(message, "content"):
                prompt_text += message.type + ": " + message.content + "\n"
            else:
                prompt_text += str(message)
    else:
        prompt_text = str(prompt)
    return prompt_text.strip()


def _calculate_token_count(llm: LLM, prompt: str | list[Any]) -> int:
    """
    Calculate the token count for a given prompt text using the specified LLM.
    Falls back to basic if the LLM doesn't support token counting.
    """
    import tiktoken
    token_count = 0

    encoding = None
    try:
        if hasattr(llm, 'client') and hasattr(llm.client, 'model_name') and llm.client.model_name:
            encoding = tiktoken.encoding_for_model(llm.client.model_name)
    except Exception as e:
        print(f"Error with primary token counting: {e}")
        pass

    try:
        if encoding is None:
            encoding = tiktoken.get_encoding("cl100k_base")
        if isinstance(prompt, str):
            token_count = len(encoding.encode(prompt))
        else:
            prompt_text = _prompt_to_string(prompt)
            token_count = len(encoding.encode(prompt_text))
    except Exception as e2:
        #print(f"Error calculating token count with fallback method: {e2}")
        pass

    return token_count
    

def _get_response_text(response: Any) -> str:
    if hasattr(response, "content"):
        response_text = response.content

        # Handle Databricks gpt-oss type of responses (having list of dicts with type + summary for reasoning or type + text for result)
        if isinstance(response_text, list):
            response_text = next(filter(lambda x: x.get('type') == 'text', response.content), None)
        if isinstance(response_text, dict):
            response_text = response_text.get('text', '')
    elif isinstance(response, str):
        response_text = response
    else:
        raise ValueError("Unexpected response format from LLM.")

    if "QUESTION VALIDATION ERROR:" in response_text:
        err = response_text.split("QUESTION VALIDATION ERROR:", 1)[1].strip()
        raise ValueError(err)

    return response_text


def _extract_usage_metadata(response: Any) -> dict:
    """
    Extract usage metadata from LLM response across different providers.
    
    Different providers return usage data in different formats:
    - OpenAI/AzureOpenAI: response.response_metadata['token_usage'] or response.usage_metadata
    - Anthropic: response.response_metadata['usage'] or response.usage_metadata
    - Google/VertexAI: response.usage_metadata
    - Bedrock: response.response_metadata['usage'] or response.response_metadata (direct ResponseMetadata)
    - Snowflake: response.response_metadata['usage']
    - Databricks: response.usage_metadata or response.response_metadata
    """
    usage_metadata = {}
    
    # Try to get response_metadata first (most common)
    if hasattr(response, 'response_metadata') and response.response_metadata:
        resp_meta = response.response_metadata
        
        # Check for 'usage' key (Anthropic, Bedrock, Snowflake)
        if 'usage' in resp_meta:
            usage_metadata = resp_meta['usage']
        # Check for 'token_usage' key (OpenAI/AzureOpenAI)
        elif 'token_usage' in resp_meta:
            usage_metadata = resp_meta['token_usage']
        # Check for direct token fields in response_metadata (some Bedrock responses)
        elif any(key in resp_meta for key in ['input_tokens', 'output_tokens', 'total_tokens', 
                                                'prompt_tokens', 'completion_tokens']):
            usage_metadata = {
                k: v for k, v in resp_meta.items() 
                if k in ['input_tokens', 'output_tokens', 'total_tokens', 
                        'prompt_tokens', 'completion_tokens']
            }
    
    # Try usage_metadata attribute (Google, VertexAI, some others)
    if not usage_metadata and hasattr(response, 'usage_metadata') and response.usage_metadata:
        usage_meta = response.usage_metadata
        if isinstance(usage_meta, dict):
            # If it has a nested 'usage' key
            if 'usage' in usage_meta:
                usage_metadata = usage_meta['usage']
            else:
                usage_metadata = usage_meta
        else:
            # Handle case where usage_metadata is an object with attributes
            usage_metadata = {
                k: getattr(usage_meta, k) 
                for k in dir(usage_meta) 
                if not k.startswith('_') and not callable(getattr(usage_meta, k))
            }
    
    # Try direct usage attribute (fallback)
    if not usage_metadata and hasattr(response, 'usage') and response.usage:
        usage = response.usage
        if isinstance(usage, dict):
            if 'usage' in usage:
                usage_metadata = usage['usage']
            else:
                usage_metadata = usage
        else:
            # Handle case where usage is an object with attributes
            usage_metadata = {
                k: getattr(usage, k) 
                for k in dir(usage) 
                if not k.startswith('_') and not callable(getattr(usage, k))
            }
    
    # Normalize token field names to standard format
    # Different providers use different names: input_tokens vs prompt_tokens, etc.
    if usage_metadata:
        normalized = {}
        
        # Map various input token field names
        if 'input_tokens' in usage_metadata:
            normalized['input_tokens'] = usage_metadata['input_tokens']
        elif 'prompt_tokens' in usage_metadata:
            normalized['input_tokens'] = usage_metadata['prompt_tokens']
        
        # Map various output token field names
        if 'output_tokens' in usage_metadata:
            normalized['output_tokens'] = usage_metadata['output_tokens']
        elif 'completion_tokens' in usage_metadata:
            normalized['output_tokens'] = usage_metadata['completion_tokens']
        
        # Map total tokens
        if 'total_tokens' in usage_metadata:
            normalized['total_tokens'] = usage_metadata['total_tokens']
        elif 'input_tokens' in normalized and 'output_tokens' in normalized:
            # Calculate total if not provided
            normalized['total_tokens'] = normalized['input_tokens'] + normalized['output_tokens']
        
        # Keep any other metadata fields that don't conflict
        for key, value in usage_metadata.items():
            if key not in ['input_tokens', 'prompt_tokens', 'output_tokens', 
                          'completion_tokens', 'total_tokens']:
                normalized[key] = value
        
        return normalized if normalized else usage_metadata
    
    return usage_metadata

def filter_list_by_ontology(concepts_list, ontology) -> list:
    ontology_specific_concepts = concepts_list
    
    if concepts_list is not None and isinstance(concepts_list, list):
        ontology_specific_concepts = []
        for concept in concepts_list:
            if "." not in concept:
                ontology_specific_concepts.append(concept)
            elif concept.startswith(f"{ontology}."):
                ontology_specific_concepts.append(concept.split(".", 1)[1])
    
    return ontology_specific_concepts

@ls_traceable(name="identify_concept")
def determine_concept(
    question: str,
    llm: LLM,
    conn_params: dict,
    concepts_list: Optional[list] = None,
    views_list: Optional[list] = None,
    include_logic_concepts: Optional[bool] = False,
    include_tags: Optional[str] = None,
    should_validate: Optional[bool] = False,
    retries: Optional[int] = 3,
    note: Optional[str] = '',
    debug: Optional[bool] = False,
    timeout: Optional[int] = None,
    memory_context=None,
) -> dict[str, Any]:
    _determine_concept_start = datetime.now()
    usage_metadata = {}
    determined_concept_name = None
    identify_concept_reason = None
    schema = 'dtimbr'
    
    # Use config default timeout if none provided
    if timeout is None:
        timeout = config.llm_timeout

    # Inject memory context into note
    if memory_context is not None:
        from .memory import format_memory_note_for_sql
        memory_note = format_memory_note_for_sql(memory_context)
        if memory_note:
            note = (memory_note + '\n' + note) if note else memory_note
    
    # Fix for multiple ontologies - load the prompt template using a specific ontology connection
    determine_concept_prompt = None

    ontologies_conn_params = {}
    #ontologies = [o.strip().lower() for o in conn_params.get("ontology").split(",")]
    ontologies = parse_list(conn_params.get("ontology"))

    for ontology in ontologies:
        ontology_conn_param =  conn_params.copy()
        ontology_conn_param["ontology"] = ontology
        ontologies_conn_params[ontology] = ontology_conn_param
        
        # Fix for multiple ontologies
        if not determine_concept_prompt:
            determine_concept_prompt = get_determine_concept_prompt_template(ontology_conn_param)

    concepts_desc_arr = []
    ontologies_concepts_and_views = {}
    candidates = []

    for ontology in ontologies_conn_params.keys():
        tags = get_tags(conn_params=ontologies_conn_params[ontology], include_tags=include_tags)
        ontology_description, domain_description = get_ontology_description(ontologies_conn_params[ontology])

        ontology_specific_concepts = filter_list_by_ontology(concepts_list, ontology)
        ontology_specific_views = filter_list_by_ontology(views_list, ontology)

        concepts_and_views = get_concepts(
            conn_params=ontologies_conn_params[ontology],
            concepts_list=ontology_specific_concepts,
            views_list=ontology_specific_views,
            include_logic_concepts=include_logic_concepts,
        )

        if concepts_and_views:
            ontologies_concepts_and_views[ontology] = concepts_and_views

            formatted_ontology_desc = f"-- Schema `{ontology}`" 
            
            if ontology_description != "":
                cleaned_ontology_desc = ontology_description.replace("\r\n", " ").replace("\n", " ")
                formatted_ontology_desc += f" description: {cleaned_ontology_desc}"

            if domain_description != "":
                cleaned_domain_desc = domain_description.replace("\r\n", " ").replace("\n", " ")
                formatted_ontology_desc += f". Related Domains description: {cleaned_domain_desc}"
            concepts_desc_arr.append(formatted_ontology_desc + "\n")

            for item in concepts_and_views.values():
                item_name = item.get('concept')
                item_desc = item.get('description')
                item_tags = tags.get('concept_tags').get(item_name) if item.get('is_view') == 'false' else tags.get('view_tags').get(item_name)

                if item_tags:
                    item_tags = str(item_tags).replace('{', '').replace('}', '').replace("'", '')

                clean_prefix = ""
                prefix = ""

                if len(ontologies_conn_params) > 1:
                    clean_prefix = f"{ontology}."
                    prefix = f"`{ontology}`."

                candidates.append(clean_prefix + item_name)
                concept_verbose = prefix + f"`{item_name}`"
                if item_desc:
                    concept_verbose += f" (description: {item_desc})"
                if item_tags:
                    concept_verbose += f" [tags: {item_tags}]"
                    concepts_and_views[item_name]['tags'] = f"- Annotations and constraints: {item_tags}\n"

                concepts_desc_arr.append(concept_verbose)
            
            concepts_desc_arr.append('\n')

    if len(ontologies_concepts_and_views) == 0:
        raise Exception("No relevant concepts found for the query.")

    if len(ontologies_concepts_and_views) == 1 and len(list(ontologies_concepts_and_views.values())[0]) == 1:
        # If only one concept is provided, return it directly
        determined_concept_name = list(list(ontologies_concepts_and_views.values())[0].keys())[0]
    else:
        # Use LLM to determine the concept based on the question
        iteration = 0
        error = ''
        while determined_concept_name is None and iteration < retries:
            iteration += 1
            err_txt = f"\nLast try got an error: {error}" if error else ""
            prompt = determine_concept_prompt.format_messages(
                question=question.strip(),
                concepts="\n".join(concepts_desc_arr),
                note=(note or '') + err_txt,
            )

            # temporary fix to old prompts 
            if len(prompt) == 2:
                prompt[1].content = prompt[1].content.replace("no quotes", "no backtick quotes")
            
            apx_token_count = _calculate_token_count(llm, prompt)
            if "snowflake" in llm._llm_type:
                _clean_snowflake_prompt(prompt)

            try:
                response = _call_llm_with_timeout(llm, prompt, timeout=timeout)
            except TimeoutError as e:
                error = f"LLM call timed out: {str(e)}"
                raise Exception(error)
            except Exception as e:
                error = f"LLM call failed: {str(e)}"
                continue
            usage_metadata['determine_concept'] = {
                "approximate": apx_token_count,
                **_extract_usage_metadata(response),
            }
            if debug:
                usage_metadata['determine_concept']["p_hash"] = encrypt_prompt(prompt)

            # Try to parse as JSON first (with 'result' and 'reason' keys)
            try:
                parsed_response = _parse_json_from_llm_response(response)
                if isinstance(parsed_response, dict) and 'result' in parsed_response:
                    candidate = parsed_response.get('result', '').strip().replace("`", "").replace('"', "").lower()
                    identify_concept_reason = parsed_response.get('reason', None)
                else:
                    # Fallback to plain text if JSON doesn't have expected structure
                    candidate = _get_response_text(response).strip().replace("`", "").replace('"', "").lower()
            except (json.JSONDecodeError, ValueError):
                # If not JSON, treat as plain text (backwards compatibility)
                candidate = _get_response_text(response).strip().replace("`", "").replace('"', "").lower()
            
            if candidate not in candidates:

                if len(ontologies_conn_params) > 1:
                    for existing in candidates:
                        if existing.endswith("." + candidate):
                            candidate = existing

                if candidate not in candidates:            
                    error = f"Concept '{candidate}' not found in the list of concepts."
                    continue
            
            determined_concept_name = candidate
            error = ''

        if determined_concept_name is None and error != '':
            raise Exception(f"Failed to determine concept: {error}")

    ontology = list(ontologies_concepts_and_views.keys())[0]
    concepts_and_views = list(ontologies_concepts_and_views.values())[0]
    
    if "." in determined_concept_name:
        ontology = determined_concept_name.split(".")[0]
        determined_concept_name = determined_concept_name.split(".")[1]
        concepts_and_views = ontologies_concepts_and_views.get(ontology)    

    schema = 'vtimbr' if concepts_and_views.get(determined_concept_name).get('is_view') == 'true' else 'dtimbr'

    return {
        "concept_metadata": concepts_and_views.get(determined_concept_name) if determined_concept_name else None,
        "concept": determined_concept_name,
        "identify_concept_reason": identify_concept_reason,
        "schema": schema,
        "usage_metadata": usage_metadata,
        "ontology": ontology,
        "conn_params": ontologies_conn_params.get(ontology),
        "duration_ms": int((datetime.now() - _determine_concept_start).total_seconds() * 1000),
    }


def _segments_from_prefix(
    prefix: str, anchor: str,
) -> list[tuple[str, str, str]]:
    """Parse a column-name prefix like ``rel1[t1].rel2[t2*4]`` into a
    walked chain of ``(from_concept, rel_name, target_concept)`` triples
    starting from ``anchor``.

    Returns ``[]`` when the prefix is empty or malformed — caller falls
    back to leaving description unchanged. The transitivity marker
    (``*N``) on the target is stripped so the returned target_concept is
    just the concept name.
    """
    if not prefix:
        return []
    out: list[tuple[str, str, str]] = []
    current_from = anchor
    for segment in prefix.split('.'):
        # Each segment is ``rel_name[target]`` or ``rel_name[target*N]``.
        if '[' not in segment or not segment.endswith(']'):
            return []   # malformed — bail out
        rel_name, rest = segment.split('[', 1)
        target = rest[:-1]   # strip trailing ']'
        if '*' in target:
            target = target.split('*', 1)[0]
        if not rel_name or not target:
            return []
        out.append((current_from, rel_name, target))
        current_from = target
    return out


def _partition_static_relationships_by_prefix(
    top_level_dict: dict, *, ontology, anchor: str,
) -> dict:
    """Re-key the static ``relationships`` dict by FULL per-hop prefix.

    Where the input today looks like ``{'of_customer': {columns: [...all
    50 across hops...], measures: [...]}}``, the output is keyed by every
    distinct prefix the items reference:

        {'of_customer[customer]':
           {description: '<of_customer desc> (cardinality: 1:N)',
            columns: [<customer's direct cols>],
            measures: [<customer's native measures>],
            is_transitive: False},
         'of_customer[customer].received_shipment[shipment]':
           {description: '<received_shipment desc> (cardinality: 1:N)',
            columns: [<shipment cols>], measures: [<shipment measures>],
            is_transitive: False}}

    The renderer (`_build_rel_columns_str`) iterates dict.items() and
    emits one column block + one measure block per key. Measures get
    their ``measure.`` prefix stripped before the ``rsplit('.', 1)``
    grouping step so a column and a measure of the same hop land in
    the same bucket.

    Description + cardinality come from the LAST segment of each prefix
    (the hop owning that bucket's columns/measures), via the existing
    ``_lookup_relationship_description`` and ``_safe_cardinality_of``
    helpers shared with the dynamic rebuild — no logic duplication.
    """
    from ..ontology_context.context_builder.rebuild import (
        compose_rel_description_with_cardinality,
        _lookup_relationship_description,
        _safe_cardinality_of,
    )

    new_dict: dict = {}
    transitivity_map: dict = {}   # prefix -> is_transitive

    def _bucket(prefix: str) -> dict:
        if prefix not in new_dict:
            new_dict[prefix] = {
                "description": "",
                "columns": [],
                "measures": [],
                "is_transitive": "*" in prefix,
            }
        return new_dict[prefix]

    for _top_rel, top_bucket in top_level_dict.items():
        for col in top_bucket.get("columns", []) or []:
            name = col.get("name") or col.get("col_name", "")
            if "." not in name:
                continue
            prefix = name.rsplit(".", 1)[0]
            if not prefix:
                continue
            _bucket(prefix)["columns"].append(col)
        for meas in top_bucket.get("measures", []) or []:
            name = meas.get("name") or meas.get("col_name", "")
            # Strip leading ``measure.`` so the prefix-by-rsplit matches
            # the column-side prefix for the same hop.
            if name.startswith("measure."):
                name = name[len("measure."):]
            if "." not in name:
                continue
            prefix = name.rsplit(".", 1)[0]
            if not prefix:
                continue
            _bucket(prefix)["measures"].append(meas)

    # Resolve description + cardinality per prefix via the last hop's
    # (from_concept, rel_name) — same helpers the dynamic rebuild uses.
    for prefix, bucket in new_dict.items():
        chain = _segments_from_prefix(prefix, anchor)
        if not chain:
            continue
        from_concept, rel_name, _target = chain[-1]
        raw_desc = _lookup_relationship_description(
            ontology, from_concept, rel_name,
        )
        card = _safe_cardinality_of(ontology, from_concept, rel_name)
        bucket["description"] = compose_rel_description_with_cardinality(
            raw_desc, card,
        )

    return new_dict


def _build_columns_str(
    columns: list[dict],
    columns_tags: Optional[dict] = {},
    exclude: Optional[list] = None,
) -> str:
    columns_desc_arr = []
    for col in columns:
        full_name = col.get('name') or col.get('col_name') # When rel column, it can be `relationship_name[column_name]`
        col_name = col.get('col_name', '')

        if col_name.startswith("measure."):
            col_name = col_name.replace("measure.", "")

        if exclude and (col_name in exclude or any(col_name.endswith('.' + exc) for exc in exclude)):
            continue

        col_tags = str(columns_tags.get(col_name)) if columns_tags.get(col_name) else None
        if col_tags:
            col_tags = col_tags.replace('{', '').replace('}', '').replace("'", '').replace(": ", " - ").replace(",", ". ").strip()
        
        description = col.get('description') or  col.get('comment', '')

        data_type = col.get('data_type', 'string').lower() or 'string'

        col_meta = []
        if data_type:
            col_meta.append(f"type: {data_type}")
        if description:
            col_meta.append(f"description: {description}")
        if col_tags:
            col_meta.append(f"annotations and constraints: {col_tags}")

        statistics = col.get('technical_context')
        if statistics:
            col_meta.append(f"statistics: {statistics}")

        col_meta_str = ', '.join(col_meta) if col_meta else ''
        if col_meta_str:
            col_meta_str = f" ({col_meta_str})"

        columns_desc_arr.append(f"`{full_name}`{col_meta_str}")

    return ", ".join(columns_desc_arr) if columns_desc_arr else ''


def _build_rel_columns_str(relationships: list[dict], columns_tags: Optional[dict] = {}, exclude_properties: Optional[list] = None) -> str:
    if not relationships:
        return ''
    rel_str_arr = []
    for rel_name in relationships:
        rel = relationships[rel_name]
        rel_description = rel.get('description', '')
        rel_description = f" which described as \"{rel_description}\"" if rel_description else ""
        rel_columns = rel.get('columns', [])
        rel_measures = rel.get('measures', [])
        
        if rel_columns:
            joined_columns_str = _build_columns_str(rel_columns, columns_tags=columns_tags, exclude=exclude_properties)
            rel_str_arr.append(f"- The following columns are part of {rel_name} relationship{rel_description}, and must be used as is wrapped with quotes: {joined_columns_str}")
        if rel_measures:
            joined_measures_str = _build_columns_str(rel_measures, columns_tags=columns_tags, exclude=exclude_properties)
            rel_str_arr.append(f"- {MEASURES_DESCRIPTION}, are part of {rel_name} relationship{rel_description}: {joined_measures_str}")
    
    return '.\n'.join(rel_str_arr) if rel_str_arr else ''


def _parse_sql_and_reason_from_llm_response(response: Any) -> dict:
    """
    Parse SQL, reason, and (optionally) decisions from LLM response.
    Handles plain SQL strings, 2-field JSON (legacy: result+reason),
    and 3-field JSON (new: reason+decisions+result).

    Returns:
        dict with keys:
            'sql'       — always present
            'reason'    — None if not provided
            'decisions' — None if not provided (legacy API or LLM omitted it);
                          also None if present but malformed (non-list)
    """
    # Try to parse as JSON first
    try:
        parsed_json = _parse_json_from_llm_response(response)

        if isinstance(parsed_json, dict) and 'result' in parsed_json:
            sql = parsed_json.get('result', '')
            reason = parsed_json.get('reason', None)
            decisions = parsed_json.get('decisions', None)

            # Drop malformed decisions rather than failing the whole parse —
            # SQL is still valid output and shouldn't be discarded over a bad trace.
            if decisions is not None and not isinstance(decisions, list):
                decisions = None

            # Clean the SQL
            sql = (sql
                   .replace("```sql", "")
                   .replace("```", "")
                   .replace('SELECT \n', 'SELECT ')
                   .replace(';', '')
                   .strip())

            return {'sql': sql, 'reason': reason, 'decisions': decisions}
    except (json.JSONDecodeError, ValueError):
        # If not JSON, treat as plain SQL string (backwards compatibility)
        pass

    # Fallback to plain text parsing
    response_text = _get_response_text(response)
    sql = (response_text
           .replace("```sql", "")
           .replace("```", "")
           .replace('SELECT \n', 'SELECT ')
           .replace(';', '')
           .strip())

    return {'sql': sql, 'reason': None, 'decisions': None}


def _get_active_datasource(conn_params: dict) -> dict:
    datasources = get_datasources(conn_params, filter_active=True)
    return datasources[0] if datasources else None


def _parse_json_from_llm_response(response: Any) -> dict:
    """
    Parse JSON from LLM response. Handles markdown code blocks and extracts valid JSON.
    
    Args:
        response: LLM response object
        
    Returns:
        dict containing parsed JSON
        
    Raises:
        json.JSONDecodeError: If response cannot be parsed as JSON
        ValueError: If response format is unexpected
    """
    response_text = _get_response_text(response)
    
    # Remove markdown code block markers if present
    content = response_text.strip()
    if content.startswith("```json"):
        content = content[7:]  # Remove ```json
    elif content.startswith("```"):
        content = content[3:]  # Remove ```
    
    if content.endswith("```"):
        content = content[:-3]  # Remove closing ```
    
    content = content.strip()
    
    # Parse and return JSON
    return json.loads(content)


def _append_reasoning_context_blocks(
    prompt: Any,
    note: Optional[str] = None,
    generate_sql_reason: Optional[str] = None,
    decisions: Optional[list] = None,
) -> None:
    """
    Append context blocks (note, generator reasoning, decision trace) to the
    trailing HumanMessage of a rendered reasoning prompt, in place.

    Conversation memory is intentionally NOT a separate block here — by the
    time we reach the reasoning step, ``note`` already carries the memory
    block (merged in at the top of ``generate_sql`` via
    ``format_memory_note_for_sql``). Adding it again would duplicate it.

    Performed AFTER the template substitutes its own placeholders so it
    works against any version of the reasoning template — no new
    placeholders required server-side.
    """
    blocks: list[str] = []

    if note and note.strip():
        blocks.append(f"**Additional Notes:**\n{note.strip()}")

    if generate_sql_reason:
        blocks.append(f"**Generated SQL Reasoning:**\n{generate_sql_reason}")

    if decisions:
        # decisions is a list (parser already validated shape)
        decisions_str = json.dumps(decisions, indent=2)
        blocks.append(f"**Generated SQL Decision Trace:**\n{decisions_str}")

    if not blocks:
        return

    if not isinstance(prompt, list) or not prompt:
        return

    # Find trailing HumanMessage and append
    human_msg = None
    for msg in reversed(prompt):
        if getattr(msg, "type", None) == "human":
            human_msg = msg
            break
    if human_msg is None:
        human_msg = prompt[-1]

    appendix = "\n\n" + "\n\n".join(blocks)
    if hasattr(human_msg, "content"):
        human_msg.content = (human_msg.content or "") + appendix


def _evaluate_sql_enable_reasoning(
    question: str,
    sql_query: str,
    llm: LLM,
    conn_params: dict,
    timeout: int,
    note: Optional[str] = None,
    generate_sql_reason: Optional[str] = None,
    decisions: Optional[list] = None,
) -> dict:
    """
    Evaluate if the generated SQL correctly answers the business question.

    The reasoning human message is rendered from the API-hosted template,
    then context blocks (note, generator reasoning, decision trace) are
    appended on the client side before sending to the LLM. This decouples
    the client from the reasoning template's structure — no server-side
    placeholders are needed.

    Returns:
        dict with 'assessment' ('correct'|'partial'|'incorrect') and 'reasoning'
    """
    generate_sql_reasoning_template = get_generate_sql_reasoning_prompt_template(conn_params)
    prompt = generate_sql_reasoning_template.format_messages(
        question=question.strip(),
        sql_query=sql_query.strip(),
    )

    _append_reasoning_context_blocks(
        prompt,
        note=note,
        generate_sql_reason=generate_sql_reason,
        decisions=decisions,
    )

    apx_token_count = _calculate_token_count(llm, prompt)
    if hasattr(llm, "_llm_type") and "snowflake" in llm._llm_type:
        _clean_snowflake_prompt(prompt)

    response = _call_llm_with_timeout(llm, prompt, timeout=timeout)
    
    # Parse JSON response
    evaluation = _parse_json_from_llm_response(response)
    
    return {
        "evaluation": evaluation,
        "apx_token_count": apx_token_count,
        "usage_metadata": _extract_usage_metadata(response),
    }


# Matches the ``*N`` transitive-depth marker that ``build_relationships_from_paths``
# bakes into a rebuilt column name (e.g. ``has_acquired[company*3].name``). The
# marker is positional only — stats are keyed by terminal concept + final
# property and are identical across depths — so it must be stripped before
# matching a rebuilt name against a TC annotation key or feeding it to the
# statistics loader (whose ``parse_column_path`` would otherwise mis-read
# ``company*3`` as a concept name).
_TRANSITIVITY_MARKER_RE = re.compile(r"\*\d+(?=\])")


def _strip_transitivity_marker(name: Optional[str]) -> Optional[str]:
    """Drop the ``*N`` depth marker so a name matches regardless of transitive
    depth. ``rel[c*3].p`` and ``rel[c].p`` share one annotation (same concept,
    same property, same stats)."""
    if not name:
        return name
    return _TRANSITIVITY_MARKER_RE.sub("", name)


def _inject_tc_annotations_into_rebuild(
    filtered_relationships: dict,
    tc_annotations: dict,
) -> None:
    """Mutate ``filtered_relationships`` in place, copying TC annotation strings
    from ``tc_annotations`` onto each column/measure dict by name.

    The technical_context pipeline runs once upstream against the static
    column dicts (mutating them in place). The dynamic rebuild then discards
    those static relationship dicts and constructs fresh ones via
    ``build_relationships_from_paths`` — which means the rebuilt nested
    columns arrive at the SQL-gen prompt with no statistics block, even
    though stats were loaded and an annotation was computed. This pass
    re-keys the annotations onto the rebuilt dicts so they reach the prompt.

    Matching is transitivity-marker-insensitive: the constructor bakes a
    ``*N`` depth marker into rebuilt names (``has_acquired[company*3].name``)
    that the static TC keys don't carry (``has_acquired[company].name``). An
    exact match wins first; otherwise the marker is stripped from both sides.
    This is safe because two names that normalize together differ only in
    transitive depth — same concept, same property, same stats. Columns the
    TC pipeline never saw at all (a concept deeper than the static
    ``graph_depth``) are handled separately by the top-up pass in
    ``_apply_dynamic_metadata_context``.
    """
    normalized = {
        _strip_transitivity_marker(k): v for k, v in tc_annotations.items()
    }
    for rel_data in filtered_relationships.values():
        if not isinstance(rel_data, dict):
            continue
        for entries_field in ("columns", "measures"):
            entries = rel_data.get(entries_field)
            if not isinstance(entries, list):
                continue
            for col in entries:
                if not isinstance(col, dict):
                    continue
                name = col.get("name")
                if not name:
                    continue
                annotation = tc_annotations.get(name)
                if annotation is None:
                    annotation = normalized.get(_strip_transitivity_marker(name))
                if annotation is not None:
                    col["technical_context"] = annotation


def _inject_descriptions_into_flat(columns: list, properties_desc: dict | None) -> None:
    """Fill each flat column/measure dict's ``description`` from ``properties_desc``
    (the SYS_PROPERTIES source the static path uses), keyed by bare property name
    (``col_name``). Only overrides when properties_desc has a value, so any
    existing (ontology) description stays as a fallback. Used for the reanchored
    anchor block, whose columns were freshly built from ontology metadata."""
    if not properties_desc:
        return
    for col in columns or []:
        if not isinstance(col, dict):
            continue
        desc = properties_desc.get(col.get("col_name"))
        if desc:
            col["description"] = desc


def _inject_descriptions_into_rebuild(
    filtered_relationships: dict,
    properties_desc: dict | None,
) -> None:
    """Mutate ``filtered_relationships`` in place, filling each rebuilt
    relationship column/measure ``description`` from ``properties_desc``.

    ``build_relationships_from_paths`` seeds descriptions from the ontology's
    ``describe concept`` comments, which are often empty — the static path
    instead pulls richer descriptions from ``properties_desc`` (SYS_PROPERTIES)
    and bakes them onto both flat and relationship columns. This pass brings the
    rebuilt relationship columns to parity. Keyed by bare property name
    (``col_name``); no transitivity-marker handling is needed because markers
    only appear in ``name``, never ``col_name``."""
    if not properties_desc:
        return
    for rel_data in filtered_relationships.values():
        if not isinstance(rel_data, dict):
            continue
        for entries_field in ("columns", "measures"):
            entries = rel_data.get(entries_field)
            if not isinstance(entries, list):
                continue
            _inject_descriptions_into_flat(entries, properties_desc)


def _apply_dynamic_metadata_context(
    *,
    mode: str,
    question: str,
    anchor: str,
    conn_params: dict,
    graph_depth: int,
    columns: list,
    measures: list,
    tags,
    exclude_properties,
    static_columns_str: str,
    static_measures_str: str,
    static_rel_prop_str: str,
    llm,
    config_overrides: dict,
    note: str = "",
    tc_annotations: dict | None = None,
    tc_topup=None,
    tc_seen_names: set | None = None,
    properties_desc: dict | None = None,
) -> tuple[str, str, str, str | None]:
    """Decide static vs dynamic and return possibly-rebuilt context strings.

    Returns ``(columns_str, measures_str, rel_prop_str, effective_anchor)``.
    ``effective_anchor`` is non-None only when Tier 2 anchor re-evaluation
    swapped the anchor and the caller MUST use it in the SQL FROM clause.
    On any failure or when the dynamic pipeline is not triggered, returns the
    static strings unchanged and ``effective_anchor=None`` (backward-compat).
    """
    from ..ontology_context import (
        MetadataContextConfig,
        build_filtered_metadata,
        config_from_module,
        get_shared_ontology,
    )
    from ..ontology_context.context_builder.rebuild import (
        apply_transitivity_overrides,
        build_relationships_from_paths,
        filter_columns_for_concepts,
    )

    cfg: MetadataContextConfig = config_from_module(
        mode=mode, **{k: v for k, v in config_overrides.items() if v is not None}
    )

    # Trigger decision -----------------------------------------------------
    # Only 'dynamic' runs the pipeline; 'static' is a strict no-op that returns
    # the pre-computed strings unchanged. cfg.mode is already normalized by
    # config_from_module (the retired 'auto' value is coerced to 'dynamic').
    if cfg.mode != "dynamic":
        return static_columns_str, static_measures_str, static_rel_prop_str, None

    # Memoize the entire pipeline result for this (question, anchor, graph_depth)
    # within the lifetime of the shared Ontology. This prevents the Step 1 LLM
    # filter call from running twice when handle_validate_generate_sql retries
    # SQL generation: both invocations of _build_sql_generation_context use the
    # same question + anchor, so the second call reuses the cached rebuild.
    # The cache is automatically invalidated by Ontology when version_id changes.
    ontology = get_shared_ontology(conn_params)
    cache_key = (
        "dynamic_rebuild_v1",
        question,
        anchor,
        graph_depth,
        cfg.mode,
        # Bind the cache to the static strings so a different pre-state never
        # serves a stale rebuilt output. Hashing the lengths is enough — the
        # static strings are deterministic for a given (ontology_version,
        # concept, graph_depth) tuple.
        len(static_columns_str or ""),
        len(static_measures_str or ""),
        len(static_rel_prop_str or ""),
    )
    cached = ontology.get_filtered_cache(cache_key)
    if cached is not None:
        # Cache entry shape evolved (Part 3): older entries are 3-tuples;
        # newer ones are 4-tuples that include effective_anchor.
        if len(cached) == 4:
            return cached  # type: ignore[return-value]
        cached_cols, cached_meas, cached_rels = cached
        return cached_cols, cached_meas, cached_rels, None

    # Run the dynamic pipeline -------------------------------------------
    result = build_filtered_metadata(
        question=question,
        anchor=anchor,
        ontology=ontology,
        llm=llm,
        config=cfg,
        graph_depth=graph_depth,
        note=note,
    )
    # Helper to apply overrides to whichever strings we end up returning. The
    # override is question-driven (depth requested by user) and should ALWAYS
    # apply when the LLM emitted valid overrides.
    def _with_overrides(cols: str, meas: str, rels: str):
        if result.accepted_overrides:
            cols = apply_transitivity_overrides(cols, result.accepted_overrides)
            meas = apply_transitivity_overrides(meas, result.accepted_overrides)
            rels = apply_transitivity_overrides(rels, result.accepted_overrides)
        return cols, meas, rels

    import logging as _logging
    _log = _logging.getLogger(__name__)

    # In dynamic mode the pipeline NEVER reverts to the static (full,
    # unfiltered) strings. Every no-validated-paths outcome — anchor-only,
    # depth-capped, disconnected anchor ('empty'), or a handled pipeline error
    # — flows through the rebuild below, which emits the anchor's
    # columns/measures with a relationships block that is either empty (the
    # prompt was NOT cascade-trimmed, so we trust the planner's "no joins")
    # or a token-gated 1-hop safety net (the prompt WAS trimmed — see the
    # depth-1 gate near the end). The only static path left is the call-site
    # crash guard for an unexpected Python exception.
    resolved_by = (result.stats or {}).get("resolved_by")
    if result.error:
        _log.warning(
            "Dynamic metadata-context pipeline error (%r); emitting lean "
            "anchor-only output (no relationships), NOT static.",
            result.error,
        )

    # Rebuild filtered strings ----------------------------------------------
    # ALL direct properties + measures for the anchor (flat columns/measures
    # list). Non-anchor concept properties are emitted by the constructor
    # below, which walks ONLY the validated paths — off-path chains can't
    # appear in the output by construction (so the over-expansion bug that
    # the upstream `relationships` dict carries is sidestepped entirely).
    # Anchor for normalization: Tier 2 may have swapped it, in which case the
    # walk must be rooted at the new anchor so the rebuilt prefixes match the
    # SQL FROM clause downstream.
    effective_anchor_for_rebuild = result.effective_anchor or anchor
    # Reanchor refresh: the flat columns/measures passed in were loaded for the
    # ORIGINAL anchor. When Tier 2 swapped the anchor, re-source the NEW
    # anchor's DIRECT props/measures from the ontology so the flat block matches
    # the SQL FROM root (stats are topped up below). Without this the SQL-gen
    # prompt shows the pre-reanchor anchor's columns under the new FROM table.
    reanchored = bool(result.effective_anchor and result.effective_anchor != anchor)
    if reanchored:
        from ..ontology_context.context_builder.rebuild import build_anchor_columns
        columns, measures = build_anchor_columns(ontology, effective_anchor_for_rebuild)
    # Always keep the anchor's columns/measures. On a hard pipeline error
    # filtered_concepts is empty; fall back to the anchor so its flat columns
    # survive (an empty set would otherwise rely on filter_columns_for_concepts'
    # pass-through behavior — make the intent explicit and future-proof).
    concepts_for_filter = result.filtered_concepts or {effective_anchor_for_rebuild}
    filtered_columns = filter_columns_for_concepts(columns, concepts_for_filter)
    filtered_measures = filter_columns_for_concepts(measures, concepts_for_filter)
    filtered_relationships = build_relationships_from_paths(
        result.validated_paths, ontology, anchor=effective_anchor_for_rebuild,
    )

    # Top-up TC for columns the upstream (static-depth) pipeline never saw.
    # The static pass only loaded stats for columns within the configured
    # graph_depth; a validated path that reaches a concept deeper than that
    # has rebuilt columns the static pass never processed (and the
    # marker-insensitive inject below can't recover what was never computed).
    # Gather those columns, strip transitivity markers so the statistics
    # loader can resolve them, and run a targeted second TC pass.
    #
    # ``tc_seen_names`` (normalized names the static pass actually processed)
    # is the right "already covered" set — NOT the annotation keys. A column
    # the static pass saw but that had no stats produces no annotation key,
    # yet must not be re-topped-up (that would fire a needless LLM call every
    # rebuild). Genuinely-new deeper columns are absent from tc_seen_names.
    if tc_topup is not None:
        seen = set(tc_seen_names or ())
        missing: dict[str, str] = {}  # normalized name -> data_type
        for rel_data in filtered_relationships.values():
            if not isinstance(rel_data, dict):
                continue
            for entries_field in ("columns", "measures"):
                for col in rel_data.get(entries_field, []) or []:
                    if not isinstance(col, dict):
                        continue
                    name = col.get("name")
                    norm = _strip_transitivity_marker(name)
                    if norm and norm not in seen and norm not in missing:
                        missing[norm] = col.get("data_type", "")
        # Reanchored flat columns are new to TC too (the static pass processed
        # the OLD anchor's columns). Gather them so they get stats as well.
        if reanchored:
            for col in list(filtered_columns) + list(filtered_measures):
                if not isinstance(col, dict):
                    continue
                norm = _strip_transitivity_marker(col.get("name"))
                if norm and norm not in seen and norm not in missing:
                    missing[norm] = col.get("data_type", "")
        if missing:
            try:
                # After a reanchor, bare direct columns of the new anchor must
                # resolve their stats against IT (the original concept the
                # closure captured is wrong for them). Relationship columns in
                # the same batch resolve via their prefix regardless.
                extra = tc_topup(
                    [{"name": n, "type": t} for n, t in missing.items()],
                    bound_concept=effective_anchor_for_rebuild if reanchored else None,
                )
                if extra:
                    tc_annotations = {**(tc_annotations or {}), **extra}
            except Exception:
                # TC must never break SQL generation.
                pass

    # Re-inject technical_context annotations onto the freshly-constructed
    # relationship column dicts. See _inject_tc_annotations_into_rebuild for
    # the why.
    if tc_annotations:
        _inject_tc_annotations_into_rebuild(filtered_relationships, tc_annotations)
        # The reanchored flat columns were built fresh here (not by the caller),
        # so inject their annotations too. Flat names carry no transitivity
        # markers, so an exact name match against tc_annotations is correct.
        if reanchored:
            for col in list(filtered_columns) + list(filtered_measures):
                if not isinstance(col, dict):
                    continue
                name = col.get("name")
                if name and name in tc_annotations:
                    col["technical_context"] = tc_annotations[name]

    # Parity with the static path's descriptions (properties_desc / SYS_PROPERTIES);
    # the constructor only had the ontology's describe-concept comments, often
    # empty. Tags already apply at render via the bare-name columns_tags lookup.
    _inject_descriptions_into_rebuild(filtered_relationships, properties_desc)
    if reanchored:
        _inject_descriptions_into_flat(filtered_columns, properties_desc)
        _inject_descriptions_into_flat(filtered_measures, properties_desc)

    new_columns_str = _build_columns_str(
        filtered_columns, columns_tags=tags, exclude=exclude_properties
    )
    new_measures_str = _build_columns_str(
        filtered_measures, tags, exclude=exclude_properties
    )
    new_rel_prop_str = _build_rel_columns_str(
        filtered_relationships,
        columns_tags=tags,
        exclude_properties=exclude_properties,
    )

    # Empty filtered_relationships is now an INTENDED output (anchor-only when
    # the planner chose no joins, or the degraded 1-hop safety net dropped by
    # the token gate below). It is NOT a failure — emit the anchor's
    # columns/measures with an empty relationships block; never revert to
    # static. (Previously a safety net reverted to static here.)

    # Apply LLM-chosen transitivity overrides to ALL three strings BEFORE the
    # hard-ceiling check. Bakes the requested depth into the SQL-gen prompt.
    new_columns_str, new_measures_str, new_rel_prop_str = _with_overrides(
        new_columns_str, new_measures_str, new_rel_prop_str
    )

    # ---- Waypoint filter: threshold-gated, precondition-protected ---------
    # Only applies when (1) the rendered columns/measures block exceeds the
    # soft budget, AND (2) the path-selection prompt was served with FULL
    # visibility (no cascade trimming on any concept). Both gates are
    # required — the LLM's is_intermediate flags are only meaningful when
    # it could see every concept's full props/measures/descriptions.
    from ..ontology_context.context_builder.rebuild import (
        compute_waypoint_strip_set,
        is_path_prompt_degraded,
        strip_waypoint_columns,
    )

    soft_size = _count_metadata_tokens(
        new_columns_str, new_measures_str, new_rel_prop_str
    )
    if soft_size > cfg.metadata_context_max_tokens:
        if is_path_prompt_degraded(result.compact_ddl or ""):
            _log.info(
                "waypoint filter skipped: degraded prompt (cascade markers "
                "present in DDL). size=%d threshold=%d",
                soft_size, cfg.metadata_context_max_tokens,
            )
        else:
            keep_set, strip_set = compute_waypoint_strip_set(
                result.validated_paths, effective_anchor_for_rebuild,
            )
            # Surface anchor/terminal-override conflicts the spec calls out.
            for path in result.validated_paths or []:
                segs = getattr(path, "segments", []) or []
                if not segs:
                    continue
                last_seg = segs[-1]
                if getattr(last_seg, "is_intermediate", False):
                    _log.warning(
                        "Path %s terminal %r marked is_intermediate=True; "
                        "override ignored (terminals are never stripped).",
                        getattr(path, "path_id", "?"), last_seg.to_concept,
                    )
            if strip_set:
                filtered_relationships = strip_waypoint_columns(
                    filtered_relationships, strip_set,
                )
                # Re-render the relationships block + re-apply overrides.
                new_rel_prop_str = _build_rel_columns_str(
                    filtered_relationships,
                    columns_tags=tags,
                    exclude_properties=exclude_properties,
                )
                new_rel_prop_str = apply_transitivity_overrides(
                    new_rel_prop_str, result.accepted_overrides,
                )
                post_size = _count_metadata_tokens(
                    new_columns_str, new_measures_str, new_rel_prop_str
                )
                if post_size > cfg.metadata_context_max_tokens:
                    _log.warning(
                        "waypoint filter applied but output still over "
                        "threshold: %d > %d. Stripped %d concept(s): %s.",
                        post_size, cfg.metadata_context_max_tokens,
                        len(strip_set), sorted(strip_set),
                    )
            else:
                _log.warning(
                    "waypoint filter could not reduce: no intermediate-only "
                    "concepts marked (size=%d, threshold=%d).",
                    soft_size, cfg.metadata_context_max_tokens,
                )

    new_token_count = _count_metadata_tokens(
        new_columns_str, new_measures_str, new_rel_prop_str
    )
    if new_token_count > cfg.metadata_context_max_tokens:
        if resolved_by == "depth_capped_static":
            # Degraded no-paths safety net: the planner chose no joins from a
            # cascade-trimmed prompt, so we synthesized a 1-hop slice. It does
            # NOT fit the budget — drop the relationships entirely and keep
            # only the anchor's columns/measures. (Still never static.)
            _log.info(
                "Dynamic metadata-context: degraded 1-hop safety net over "
                "budget (%d > %d) — dropping relationships, keeping anchor "
                "columns/measures.",
                new_token_count, cfg.metadata_context_max_tokens,
            )
            new_rel_prop_str = ""
        else:
            # Principle: dynamic-over-budget is preferred over static-but-much-larger
            # (see .claude/plans/budget-knobs-cleanup.md). There is NO hard-revert
            # to static for the real-paths case when the rebuilt output is still
            # over the soft cap after cascade + waypoint filter. Log and emit the
            # rebuilt strings as-is — the SQL-gen prompt is still SMALLER than the
            # static alternative.
            _log.warning(
                "Dynamic metadata-context still over soft cap after cascade + "
                "waypoint filter (%d > %d). Emitting rebuilt strings as-is — "
                "no revert to static. resolved_by=%s, validated_paths=%d.",
                new_token_count,
                cfg.metadata_context_max_tokens,
                resolved_by,
                len(result.validated_paths or []),
            )

    entry = (
        new_columns_str,
        new_measures_str,
        new_rel_prop_str,
        result.effective_anchor,
    )
    ontology.set_filtered_cache(cache_key, entry)
    return entry


def _count_metadata_tokens(*parts: str) -> int:
    """Count tiktoken-cl100k tokens for the concatenated metadata strings.

    Falls back to a coarse chars/4 estimate when tiktoken is unavailable.
    """
    text = "\n".join(p or "" for p in parts)
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _build_sql_generation_context(
    question: str,
    conn_params: dict,
    schema: str,
    concept: str,
    concept_metadata: dict,
    graph_depth: int,
    include_tags: Optional[str],
    exclude_properties: Optional[list],
    db_is_case_sensitive: bool,
    max_limit: int,
    llm=None,
    enable_technical_context: bool = True,
    technical_context_mode: str = "auto",
    technical_context_max_tokens: int = 3000,
    technical_context_properties: Optional[list] = None,
    # Plan 2 — dynamic metadata-context. Defaults preserve current behavior.
    # NOTE: The DDL prompt budget knobs (metadata_context_filter_max_tokens,
    # metadata_context_filter_max_tokens_hard_ceiling) and the planner retry
    # budget (metadata_context_dynamic_retry) are config-only — they're read
    # from langchain_timbr.config inside _apply_dynamic_metadata_context, not
    # plumbed through chain constructors. See .claude/plans/budget-knobs-cleanup.md.
    metadata_context_mode: Optional[str] = None,
    metadata_context_max_tokens: Optional[int] = None,
    max_graph_depth: Optional[int] = None,
    include_logic_concepts: Optional[bool] = None,
    # Conversation memory (follow-up context) + caller-supplied notes,
    # plumbed verbatim into the dynamic context_builder LLM prompts so the
    # filter / pre-filter / anchor-reeval / not_needed-verifier all see the
    # same prior-turn context the SQL-gen prompt sees.
    note: Optional[str] = None,
) -> dict:
    """
    Prepare the complete SQL generation context by gathering all necessary metadata.
    
    This includes:
    - Datasource information
    - Concept properties (columns, measures, relationships)
    - Property tags
    - Building column/measure/relationship descriptions
    - Assembling the final context dictionary
    
    Returns:
        dict containing all context needed for SQL generation prompts
    """
    datasource_type = _get_active_datasource(conn_params).get('target_type')

    properties_desc = get_properties_description(conn_params=conn_params)
    relationships_desc = get_relationships_description(conn_params=conn_params)
  
    concept_properties_metadata = get_concept_properties(
        schema=schema,
        concept_name=concept,
        conn_params=conn_params,
        properties_desc=properties_desc,
        relationships_desc=relationships_desc,
        graph_depth=graph_depth
    )
    columns = concept_properties_metadata.get('columns', [])
    measures = concept_properties_metadata.get('measures', [])
    relationships = concept_properties_metadata.get('relationships', {})
    tags = get_tags(conn_params=conn_params, include_tags=include_tags).get('property_tags')

    # Per-hop relationship partition. The static ``get_concept_properties``
    # returns a single bucket per top-level rel — every column from every
    # nested concept on the chain piles into the same bucket. We re-key
    # by FULL prefix (``of_customer[customer]`` / ``of_customer[customer]
    # .received_shipment[shipment]`` / ...) so the SQL-gen prompt emits
    # one block per hop, each carrying its OWN description + cardinality.
    # Defensive: failures here must NOT break SQL gen.
    if relationships:
        try:
            from ..ontology_context.ontology.shared import get_shared_ontology
            _shared_ont = get_shared_ontology(conn_params)
            relationships = _partition_static_relationships_by_prefix(
                relationships, ontology=_shared_ont, anchor=concept,
            )
        except Exception:
            # Never abort SQL gen over a partition failure — fall back to
            # the original top-level-keyed dict shape.
            pass

    # Enrich column dicts with technical context annotations (stats + question matching)
    tc_annotations: dict[str, str] = {}
    # Closure the dynamic rebuild uses to top-up TC for columns the static-depth
    # pass never saw (concepts deeper than graph_depth). None when TC is off.
    tc_topup = None
    tc_seen_names: set | None = None
    if enable_technical_context:
        try:
            import copy
            from ..technical_context import build_technical_context
            from ..technical_context.config import TechnicalContextConfig
            columns = copy.deepcopy(columns)
            measures = copy.deepcopy(measures)
            relationships = copy.deepcopy(relationships)

            all_col_dicts = columns + measures
            for rel in relationships.values():
                all_col_dicts += rel.get('columns', []) + rel.get('measures', [])
            tc_columns = [{"name": c.get("name") or c.get("col_name", ""), "type": c.get("data_type", "")} for c in all_col_dicts]
            # Normalized names the static pass processes — the "already covered"
            # set the dynamic top-up uses to detect genuinely-new deeper columns.
            tc_seen_names = {
                _strip_transitivity_marker(c["name"]) for c in tc_columns if c["name"]
            }
            tc_config = TechnicalContextConfig(
                mode=technical_context_mode,
                max_tokens=technical_context_max_tokens,
                technical_context_properties=technical_context_properties or [],
                exclude_properties=exclude_properties or [],
            )

            def tc_topup(topup_columns, bound_concept=None):
                """Run a targeted TC pass for newly-revealed columns and return
                their annotations. Reuses the configured mode (so matched-value
                ranking is preserved), at the cost of a second candidate
                extraction when the mode is LLM-backed.

                A reanchor swaps the SQL FROM root: BARE direct columns of the new
                anchor must resolve their stats against IT, so the caller passes
                the effective anchor as ``bound_concept``. Relationship columns
                resolve via their prefix regardless of the bound concept, so a
                mixed batch is safe."""
                r = build_technical_context(
                    question=question,
                    columns=topup_columns,
                    schema=schema,
                    concept=bound_concept or concept,
                    conn_params=conn_params,
                    config=tc_config,
                    llm=llm,
                )
                return dict(r.column_annotations or {})

            tc_result = build_technical_context(
                question=question,
                columns=tc_columns,
                schema=schema,
                concept=concept,
                conn_params=conn_params,
                config=tc_config,
                llm=llm,
            )
            tc_annotations = dict(tc_result.column_annotations or {})
            for c in all_col_dicts:
                name = c.get('name') or c.get('col_name', '')
                if name and name in tc_annotations:
                    c['technical_context'] = tc_annotations[name]
        except Exception:
            pass  # Technical context failure must not break SQL generation

    columns_str = _build_columns_str(columns, columns_tags=tags, exclude=exclude_properties)
    measures_str = _build_columns_str(measures, tags, exclude=exclude_properties)
    rel_prop_str = _build_rel_columns_str(relationships, columns_tags=tags, exclude_properties=exclude_properties)

    # --- Plan 2 — dynamic metadata-context (safe, opt-in) -----------------
    # Schema gate: dtimbr only; non-dtimbr schemas (vtimbr views/cubes) skip.
    # Mode gate: 'static' is a strict no-op (initial release default).
    # Errors anywhere inside this block fall back to the static strings above.
    from ..ontology_context import normalize_mode
    _dynamic_mode = normalize_mode(metadata_context_mode or config.metadata_context_mode)
    if schema == 'dtimbr' and _dynamic_mode == 'dynamic':
        try:
            (
                _new_columns_str,
                _new_measures_str,
                _new_rel_prop_str,
                _effective_anchor,
            ) = _apply_dynamic_metadata_context(
                mode=_dynamic_mode,
                question=question,
                anchor=concept,
                conn_params=conn_params,
                graph_depth=graph_depth,
                columns=columns,
                measures=measures,
                tags=tags,
                exclude_properties=exclude_properties,
                static_columns_str=columns_str,
                static_measures_str=measures_str,
                static_rel_prop_str=rel_prop_str,
                llm=llm,
                note=note or '',
                tc_annotations=tc_annotations,
                tc_topup=tc_topup,
                tc_seen_names=tc_seen_names,
                properties_desc=properties_desc,
                config_overrides=dict(
                    metadata_context_max_tokens=metadata_context_max_tokens,
                    max_graph_depth=max_graph_depth,
                    include_logic_concepts=include_logic_concepts,
                ),
            )
            columns_str = _new_columns_str
            measures_str = _new_measures_str
            rel_prop_str = _new_rel_prop_str
            # Tier 2 anchor re-evaluation may have swapped the anchor — propagate
            # it downstream so the SQL FROM clause references the new concept.
            if _effective_anchor and _effective_anchor != concept:
                concept = _effective_anchor
                # Refresh the anchor-derived description/tags for the new FROM
                # root — concept_metadata was loaded for the pre-reanchor
                # concept, so leaving it stale shows the OLD concept's
                # description under the new table.
                try:
                    from ..ontology_context.ontology.shared import get_shared_ontology
                    _new_meta = get_shared_ontology(conn_params).get_concept_metadata(concept)
                    concept_metadata = {
                        **(concept_metadata or {}),
                        "description": getattr(_new_meta, "description", None),
                        "tags": getattr(_new_meta, "tags", None),
                    }
                except Exception:
                    pass
        except Exception as _dyn_exc:  # noqa: BLE001 — backward-compat fallback
            import logging
            logging.getLogger(__name__).warning(
                "Dynamic metadata-context failed (%s); falling back to static.",
                _dyn_exc,
            )

    if rel_prop_str:
        measures_str += f"\n{rel_prop_str}"

    # Determine if relationships have transitive properties
    has_transitive_relationships = any(
        rel.get('is_transitive')
        for rel in relationships.values()
    ) if relationships else False
    
    concept_description = f"- Description: {concept_metadata.get('description')}\n" if concept_metadata and concept_metadata.get('description') else ""
    concept_tags = concept_metadata.get('tags') if concept_metadata and concept_metadata.get('tags') else ""
    
    cur_date = datetime.now().strftime("%Y-%m-%d")
    
    # Build context descriptions
    sensitivity_txt = "- Ensure value comparisons are case-insensitive, e.g., use LOWER(column) = 'value'.\n" if db_is_case_sensitive else ""
    measures_context = f"\n- {MEASURES_DESCRIPTION}: {measures_str}\n" if measures_str else ""
    transitive_context = f"\n- {TRANSITIVE_RELATIONSHIP_DESCRIPTION}\n" if has_transitive_relationships else ""
    
    return {
        'cur_date': cur_date,
        'datasource_type': datasource_type or 'standard sql',
        'schema': schema,
        'concept': concept,
        'concept_description': concept_description or "",
        'concept_tags': concept_tags or "",
        'columns_str': columns_str,
        'measures_context': measures_context,
        'transitive_context': transitive_context,
        'sensitivity_txt': sensitivity_txt,
        'max_limit': max_limit,
    }


def _generate_sql_with_llm(
    question: str,
    llm: LLM,
    generate_sql_prompt: Any,
    current_context: dict,
    note: str,
    timeout: int,
    debug: bool = False,
) -> dict:
    """
    Generate SQL using LLM based on the provided context and note.
    This function is used for both initial SQL generation and regeneration with feedback.
    
    Args:
        current_context: dict containing datasource_type, schema, concept, concept_description,
                        concept_tags, columns_str, measures_context, transitive_context,
                        sensitivity_txt, max_limit, cur_date
        note: Additional instructions/feedback to include in the prompt
    
    Returns:
        dict with 'sql', 'is_valid', 'error', 'apx_token_count', 'usage_metadata', 'p_hash' (if debug)
    """
    prompt = generate_sql_prompt.format_messages(
        current_date=current_context['cur_date'],
        datasource_type=current_context['datasource_type'],
        schema=current_context['schema'],
        concept=f"`{current_context['concept']}`",
        description=current_context['concept_description'],
        tags=current_context['concept_tags'],
        question=question,
        columns=current_context['columns_str'],
        measures_context=current_context['measures_context'],
        transitive_context=current_context['transitive_context'],
        sensitivity_context=current_context['sensitivity_txt'],
        max_limit=current_context['max_limit'],
        note=note,
    )

    apx_token_count = _calculate_token_count(llm, prompt)
    if hasattr(llm, "_llm_type") and "snowflake" in llm._llm_type:
        _clean_snowflake_prompt(prompt)
    
    response = _call_llm_with_timeout(llm, prompt, timeout=timeout)
    
    # Parse response which now includes both SQL and reason
    parsed_response = _parse_sql_and_reason_from_llm_response(response)
    
    result = {
        "sql": parsed_response['sql'],
        "generate_sql_reason": parsed_response['reason'],
        "decisions": parsed_response['decisions'],
        "apx_token_count": apx_token_count,
        "usage_metadata": _extract_usage_metadata(response),
        "is_valid": True,
        "error": None,
    }
    
    if debug:
        result["p_hash"] = encrypt_prompt(prompt)
    
    
    return result

def handle_generate_sql_reasoning(
    sql_query: str,
    question: str,
    llm: LLM,
    conn_params: dict,
    schema: str,
    concept: str,
    concept_metadata: dict,
    include_tags: bool,
    exclude_properties: list,
    db_is_case_sensitive: bool,
    max_limit: int,
    reasoning_steps: int,
    note: str,
    graph_depth: int,
    usage_metadata: dict,
    timeout: int,
    debug: bool,
    previous_token_count,
    enable_technical_context: bool = True,
    technical_context_mode: str = "auto",
    technical_context_max_tokens: int = 3000,
    technical_context_properties: Optional[list] = None,
    generate_sql_reason: Optional[str] = None,
    decisions: Optional[list] = None,
    # Plan 2 — dynamic metadata-context propagation (None ⇒ inherit from config)
    metadata_context_mode: Optional[str] = None,
    metadata_context_max_tokens: Optional[int] = None,
    max_graph_depth: Optional[int] = None,
    include_logic_concepts: Optional[bool] = None,
) -> tuple[str, int, str, int]:
    import time as _time
    generate_sql_prompt = get_generate_sql_prompt_template(conn_params)
    context_graph_depth = graph_depth
    reasoned_sql = sql_query
    reasoned_sql_reason = None
    _reasoning_start = _time.monotonic()
    for step in range(reasoning_steps):
        try:
            # Step 1: Evaluate the current SQL.
            # `note` already carries the conversation-memory block (merged in
            # by generate_sql via format_memory_note_for_sql), so memory flows
            # into the reasoning appendix through the note channel.
            eval_result = _evaluate_sql_enable_reasoning(
                question=question,
                sql_query=reasoned_sql,
                llm=llm,
                conn_params=conn_params,
                timeout=timeout,
                note=note,
                generate_sql_reason=generate_sql_reason,
                decisions=decisions,
            )
            
            usage_metadata[f'sql_reasoning_step_{step + 1}'] = {
                "approximate": eval_result['apx_token_count'],
                **eval_result['usage_metadata'],
            }
            
            evaluation = eval_result['evaluation']
            reasoning_status = evaluation.get("assessment", "partial").lower()
            reasoned_sql_reason = evaluation.get("reasoning", "")
            
            if reasoning_status == "correct":
                break
            
            # Step 2: Regenerate SQL with feedback
            evaluation_note = note + f"\n\nThe previously generated SQL: `{reasoned_sql}` was assessed as '{evaluation.get('assessment')}' because: {reasoned_sql_reason or '*could not determine cause*'}. Please provide a corrected SQL query that better answers the question: '{question}'.\n\nCRITICAL: Return ONLY the SQL query without any explanation or comments."
            
            # Increase graph depth for 2nd+ reasoning attempts, up to max of 3
            max_context_graph_depth = 3
            context_graph_depth = min(max_context_graph_depth, graph_depth)

            if (step >= 1 and type(previous_token_count) == int and previous_token_count > 0 and previous_token_count < 20000):
                context_graph_depth = min(max_context_graph_depth, context_graph_depth + 1)

            regen_result = _generate_sql_with_llm(
                question=question,
                llm=llm,
                generate_sql_prompt=generate_sql_prompt,
                current_context=_build_sql_generation_context(
                    question=question,
                    conn_params=conn_params,
                    schema=schema,
                    concept=concept,
                    concept_metadata=concept_metadata,
                    graph_depth=context_graph_depth,
                    include_tags=include_tags,
                    exclude_properties=exclude_properties,
                    db_is_case_sensitive=db_is_case_sensitive,
                    max_limit=max_limit,
                    llm=llm,
                    enable_technical_context=enable_technical_context,
                    technical_context_mode=technical_context_mode,
                    technical_context_max_tokens=technical_context_max_tokens,
                    technical_context_properties=technical_context_properties,
                    metadata_context_mode=metadata_context_mode,
                    metadata_context_max_tokens=metadata_context_max_tokens,
                    max_graph_depth=max_graph_depth,
                    include_logic_concepts=include_logic_concepts,
                    note=evaluation_note,
                ),
                note=evaluation_note,
                timeout=timeout,
                debug=debug,
            )

            reasoned_sql = regen_result['sql']
            reasoned_sql_reason = regen_result['generate_sql_reason']
            error = regen_result['error']

            # Refresh generator reason + decision trace so the next iteration's
            # evaluator sees the freshest plan + trace for the SQL it evaluates.
            generate_sql_reason = regen_result.get('generate_sql_reason')
            decisions = regen_result.get('decisions')

            step_key = f'generate_sql_reasoning_step_{step + 1}'
            usage_metadata[step_key] = {
                "approximate": regen_result['apx_token_count'],
                **regen_result['usage_metadata'],
            }
            previous_token_count = regen_result['apx_token_count']

            if debug and 'p_hash' in regen_result:
                usage_metadata[step_key]['p_hash'] = regen_result['p_hash']

            if error:
                raise Exception(error)
            
        except TimeoutError as e:
            raise Exception(f"LLM call timed out: {str(e)}")
        except Exception as e:
            print(f"Warning: LLM reasoning failed: {e}")
            break
    
    _reasoning_duration_ms = int((_time.monotonic() - _reasoning_start) * 1000)
    return reasoned_sql, context_graph_depth, reasoned_sql_reason, _reasoning_duration_ms

def handle_validate_generate_sql(
    sql_query: str,
    question: str,
    llm: LLM,
    conn_params: dict,
    generate_sql_prompt: Any,
    schema: str,
    concept: str,
    concept_metadata: dict,
    include_tags: bool,
    exclude_properties: list,
    db_is_case_sensitive: bool,
    max_limit: int,
    graph_depth: int,
    retries: int,
    timeout: int,
    debug: bool,
    usage_metadata: dict,
    enable_technical_context: bool = True,
    technical_context_mode: str = "auto",
    technical_context_max_tokens: int = 3000,
    technical_context_properties: Optional[list] = None,
    # Plan 2 — dynamic metadata-context. None ⇒ inherit from config.
    metadata_context_mode: Optional[str] = None,
    metadata_context_max_tokens: Optional[int] = None,
    max_graph_depth: Optional[int] = None,
    include_logic_concepts: Optional[bool] = None,
    # Conversation memory + caller notes — forwarded to context_builder LLM
    # prompts so the validation-retry regeneration sees the same prior-turn
    # context the original generate_sql call did.
    note: Optional[str] = None,
) -> tuple[bool, str, str]:
    is_sql_valid, error, sql_query = validate_sql(sql_query, conn_params)
    validation_attempt = 0
  
    while validation_attempt < retries and not is_sql_valid:
        validation_attempt += 1
        validation_err_txt = f"\nThe generated SQL (`{sql_query}`) was invalid with error: {error}. Please generate a corrected query that achieves the intended result." if error and "snowflake" not in llm._llm_type else ""

        regen_result = _generate_sql_with_llm(
            question=question,
            llm=llm,
            generate_sql_prompt=generate_sql_prompt,
            current_context=_build_sql_generation_context(
                question=question,
                conn_params=conn_params,
                schema=schema,
                concept=concept,
                concept_metadata=concept_metadata,
                graph_depth=graph_depth,
                include_tags=include_tags,
                exclude_properties=exclude_properties,
                db_is_case_sensitive=db_is_case_sensitive,
                max_limit=max_limit,
                llm=llm,
                enable_technical_context=enable_technical_context,
                technical_context_mode=technical_context_mode,
                technical_context_max_tokens=technical_context_max_tokens,
                technical_context_properties=technical_context_properties,
                metadata_context_mode=metadata_context_mode,
                metadata_context_max_tokens=metadata_context_max_tokens,
                max_graph_depth=max_graph_depth,
                include_logic_concepts=include_logic_concepts,
                note=note,
            ),
            note=validation_err_txt,
            timeout=timeout,
            debug=debug,
        )
        
        regen_error = regen_result['error']
        sql_query = regen_result['sql']

        validation_key = f'generate_sql_validation_regen_{validation_attempt}'
        usage_metadata[validation_key] = {
            "approximate": regen_result['apx_token_count'],
            **regen_result['usage_metadata'],
        }
        if debug and 'p_hash' in regen_result:
            usage_metadata[validation_key]['p_hash'] = regen_result['p_hash']

        if regen_error:
            raise Exception(regen_error)
        
        is_sql_valid, error, sql_query = validate_sql(sql_query, conn_params)

    return is_sql_valid, error, sql_query

@ls_traceable(name="generate_sql")
def generate_sql(
        question: str,
        llm: LLM,
        conn_params: dict,
        concept: str,
        schema: Optional[str] = None,
        concepts_list: Optional[list] = None,
        views_list: Optional[list] = None,
        include_logic_concepts: Optional[bool] = False,
        include_tags: Optional[str] = None,
        exclude_properties: Optional[list] = None,
        should_validate_sql: Optional[bool] = config.should_validate_sql,
        retries: Optional[int] = 3,
        max_limit: Optional[int] = config.llm_default_limit,
        note: Optional[str] = '',
        db_is_case_sensitive: Optional[bool] = False,
        graph_depth: Optional[int] = 1,
        enable_reasoning: Optional[bool] = False,
        reasoning_steps: Optional[int] = 2,
        debug: Optional[bool] = False,
        timeout: Optional[int] = None,
        memory_context=None,
        enable_technical_context: Optional[bool] = None,
        technical_context_mode: Optional[str] = None,
        technical_context_max_tokens: Optional[int] = None,
        technical_context_properties: Optional[list] = None,
        # Plan 2 — dynamic metadata-context (None ⇒ inherit from config)
        metadata_context_mode: Optional[str] = None,
        metadata_context_max_tokens: Optional[int] = None,
        max_graph_depth: Optional[int] = None,
    ) -> dict[str, str]:
    usage_metadata = {}
    concept_metadata = None
    reasoning_status = 'correct'
    reasoning_duration = 0

    # Use config default timeout if none provided
    if timeout is None:
        timeout = config.llm_timeout

    # Inject memory context into note
    if memory_context is not None:
        from .memory import format_memory_note_for_sql
        memory_note = format_memory_note_for_sql(memory_context)
        if memory_note:
            note = (memory_note + '\n' + note) if note else memory_note
    
    if concept and concept != "" and (schema is None or schema != "vtimbr"):
        concepts_list = [concept]
    elif concept and concept != "" and schema == "vtimbr":
        views_list = [concept]

    determine_concept_res = determine_concept(
        question=question,
        llm=llm,
        conn_params=conn_params,
        concepts_list=concepts_list,
        views_list=views_list,
        include_logic_concepts=include_logic_concepts,
        include_tags=include_tags,
        should_validate=should_validate_sql,
        retries=retries,
        note=note,
        debug=debug,
        timeout=timeout,
    )

    identify_concept_chain_duration = determine_concept_res.pop("duration_ms", 0)

    if (type(conn_params.get('ontology')) == list and len(conn_params.get('ontology')) > 1) or ',' in conn_params.get('ontology'):
        conn_params = determine_concept_res.get('conn_params')

    concept = determine_concept_res.get('concept')
    identify_concept_reason = determine_concept_res.get('identify_concept_reason', None)
    schema = determine_concept_res.get('schema')
    concept_metadata = determine_concept_res.get('concept_metadata')
    usage_metadata.update(determine_concept_res.get('usage_metadata', {}))

    if not concept:
        raise Exception("No relevant concept found for the query.")

    generate_sql_prompt = get_generate_sql_prompt_template(conn_params)
    sql_query = None
    generate_sql_reason = None
    is_sql_valid = True  # Assume valid by default; set to False only if validation fails
    error = ''

    try:
        result = _generate_sql_with_llm(
            question=question,
            llm=llm,
            generate_sql_prompt=generate_sql_prompt,
            current_context=_build_sql_generation_context(
                question=question,
                conn_params=conn_params,
                schema=schema,
                concept=concept,
                concept_metadata=concept_metadata,
                graph_depth=graph_depth,
                include_tags=include_tags,
                exclude_properties=exclude_properties,
                db_is_case_sensitive=db_is_case_sensitive,
                max_limit=max_limit,
                llm=llm,
                enable_technical_context=enable_technical_context if enable_technical_context is not None else config.enable_technical_context,
                technical_context_mode=technical_context_mode or config.technical_context_mode,
                technical_context_max_tokens=technical_context_max_tokens or config.technical_context_max_tokens,
                technical_context_properties=technical_context_properties,
                metadata_context_mode=metadata_context_mode,
                metadata_context_max_tokens=metadata_context_max_tokens,
                max_graph_depth=max_graph_depth,
                include_logic_concepts=include_logic_concepts,
                note=note,
            ),
            note=note,
            timeout=timeout,
            debug=debug,
        )

        usage_metadata['generate_sql'] = {
            "approximate": result['apx_token_count'],
            **result['usage_metadata'],
        }
        if debug and 'p_hash' in result:
            usage_metadata['generate_sql']["p_hash"] = result['p_hash']
        
        sql_query = result['sql']
        generate_sql_reason = result.get('generate_sql_reason', None)
        decisions = result.get('decisions', None)
        error = result['error']

        if error:
            raise Exception(error)

        if enable_reasoning and sql_query is not None:
            sql_query, graph_depth, generate_sql_reason, reasoning_duration = handle_generate_sql_reasoning(
                sql_query=sql_query,
                question=question,
                llm=llm,
                conn_params=conn_params,
                schema=schema,
                concept=concept,
                concept_metadata=concept_metadata,
                include_tags=include_tags,
                exclude_properties=exclude_properties,
                db_is_case_sensitive=db_is_case_sensitive,
                max_limit=max_limit,
                reasoning_steps=reasoning_steps,
                note=note,
                graph_depth=graph_depth,
                usage_metadata=usage_metadata,
                timeout=timeout,
                debug=debug,
                previous_token_count=result['apx_token_count'],
                enable_technical_context=enable_technical_context if enable_technical_context is not None else config.enable_technical_context,
                technical_context_mode=technical_context_mode or config.technical_context_mode,
                technical_context_max_tokens=technical_context_max_tokens or config.technical_context_max_tokens,
                technical_context_properties=technical_context_properties,
                generate_sql_reason=generate_sql_reason,
                decisions=decisions,
                metadata_context_mode=metadata_context_mode,
                metadata_context_max_tokens=metadata_context_max_tokens,
                max_graph_depth=max_graph_depth,
                include_logic_concepts=include_logic_concepts,
            )

        if should_validate_sql or enable_reasoning:
            # Validate & regenerate only once if reasoning enabled and validation is disabled
            validate_retries = 1 if not should_validate_sql else retries
            is_sql_valid, error, sql_query = handle_validate_generate_sql(
                sql_query=sql_query,
                question=question,
                llm=llm,
                conn_params=conn_params,
                generate_sql_prompt=generate_sql_prompt,
                schema=schema,
                concept=concept,
                concept_metadata=concept_metadata,
                include_tags=include_tags,
                exclude_properties=exclude_properties,
                db_is_case_sensitive=db_is_case_sensitive,
                max_limit=max_limit,
                graph_depth=graph_depth,
                retries=validate_retries,
                timeout=timeout,
                debug=debug,
                usage_metadata=usage_metadata,
                enable_technical_context=enable_technical_context if enable_technical_context is not None else config.enable_technical_context,
                technical_context_mode=technical_context_mode or config.technical_context_mode,
                technical_context_max_tokens=technical_context_max_tokens or config.technical_context_max_tokens,
                technical_context_properties=technical_context_properties,
                metadata_context_mode=metadata_context_mode,
                metadata_context_max_tokens=metadata_context_max_tokens,
                max_graph_depth=max_graph_depth,
                include_logic_concepts=include_logic_concepts,
                note=note,
            )
    except TimeoutError as e:
        error = f"LLM call timed out: {str(e)}"
        raise Exception(error)
    except Exception as e:
        error = f"LLM call failed: {str(e)}"
        raise Exception(error)
    
    return {
        "sql": sql_query,
        "concept": concept,
        "schema": schema,
        "error": error if not is_sql_valid else None,
        "is_sql_valid": is_sql_valid if should_validate_sql else None,
        "identify_concept_reason": identify_concept_reason,
        "generate_sql_reason": generate_sql_reason,
        "reasoning_status": reasoning_status,
        "reasoning_duration": reasoning_duration,
        "identify_concept_chain_duration": identify_concept_chain_duration,
        "usage_metadata": usage_metadata,
        "ontology": conn_params.get('ontology'),
        "conn_params": conn_params
    }


@ls_traceable(name="generate_answer")
def answer_question(
    question: str,
    llm: LLM,
    conn_params: dict,
    results: str,
    sql: Optional[str] = None,
    timeout: Optional[int] = None,
    note: Optional[str] = '',
    debug: Optional[bool] = False,
    memory_context=None,
) -> dict[str, Any]:
    # Use config default timeout if none provided
    if timeout is None:
        timeout = config.llm_timeout

    qa_prompt = get_qa_prompt_template(conn_params)

    # Build additional_context with optional memory
    additional_context = f"SQL QUERY:\n{sql}\n\n" if sql else ""
    if memory_context is not None:
        from .memory import format_memory_note_for_answer
        memory_note = format_memory_note_for_answer(memory_context)
        if memory_note:
            additional_context = memory_note + "\n\n" + additional_context

    prompt = qa_prompt.format_messages(
        question=question,
        formatted_rows=results,
        additional_context=additional_context,
        note=note,
    )
    
    apx_token_count = _calculate_token_count(llm, prompt)

    if "snowflake" in llm._llm_type:
        _clean_snowflake_prompt(prompt)
    
    try:
        response = _call_llm_with_timeout(llm, prompt, timeout=timeout)
    except TimeoutError as e:
        raise TimeoutError(f"LLM call timed out while answering question: {str(e)}")
    except Exception as e:
        raise Exception(f"LLM call failed while answering question: {str(e)}")

    if hasattr(response, "content"):
        response_text = response.content
    elif isinstance(response, str):
        response_text = response
    else:
        raise ValueError("Unexpected response format from LLM.")
    
    usage_metadata = {
        "answer_question": {
            "approximate": apx_token_count,
            **_extract_usage_metadata(response),
        },
    }
    if debug:
        usage_metadata["answer_question"]["p_hash"] = encrypt_prompt(prompt)

    return {
        "answer": response_text,
        "usage_metadata": usage_metadata,
    }

