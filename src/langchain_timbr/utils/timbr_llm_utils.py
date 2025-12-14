from typing import Any, Optional
from langchain.llms.base import LLM
from datetime import datetime
import concurrent.futures
import json

from .timbr_utils import get_datasources, get_tags, get_concepts, get_concept_properties, validate_sql, get_properties_description, get_relationships_description, cache_with_version_check, encrypt_prompt
from .prompt_service import (
    get_determine_concept_prompt_template,
    get_generate_sql_prompt_template,
    get_generate_sql_reasoning_prompt_template,
    get_qa_prompt_template
)
from ..config import llm_timeout

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


def _call_llm_with_timeout(llm: LLM, prompt: Any, timeout: int = 60) -> Any:
    """
    Call LLM with timeout to prevent hanging.
    
    Args:
        llm: The LLM instance
        prompt: The prompt to send
        timeout: Timeout in seconds (default: 60)
        
    Returns:
        LLM response
        
    Raises:
        TimeoutError: If the call takes longer than timeout seconds
        Exception: Any other exception from the LLM call
    """
    def _llm_call():
        return llm(prompt)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(_llm_call)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"LLM call timed out after {timeout} seconds")
        except Exception as e:
            raise e

MEASURES_DESCRIPTION = "The following columns are calculated measures and can only be aggregated with an aggregate function: COUNT/SUM/AVG/MIN/MAX (count distinct is not allowed)"
TRANSITIVE_RELATIONSHIP_DESCRIPTION = "Transitive relationship columns allow you to access data through multiple relationship hops. These columns follow the pattern `<relationship_name>[<table_name>*<number>].<column_name>` where the number after the asterisk (*) indicates how many relationship levels to traverse. For example, `acquired_by[company*4].company_name` means 'go through up to 4 levels of the acquired_by relationship to get the company name', while columns ending with '_transitivity_level' indicate the actual relationship depth (Cannot be null or 0 - level 1 represents direct relationships, while levels 2, 3, 4, etc. represent indirect relationships through multiple hops. To filter by relationship type, use `_transitivity_level = 1` for direct relationships only, `_transitivity_level > 1` for indirect relationships only."


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
        if hasattr(llm, 'client') and hasattr(llm.client, 'model_name'):
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
) -> dict[str, Any]:
    usage_metadata = {}
    determined_concept_name = None
    schema = 'dtimbr'
    
    # Use config default timeout if none provided
    if timeout is None:
        timeout = llm_timeout
    
    determine_concept_prompt = get_determine_concept_prompt_template(conn_params)
    tags = get_tags(conn_params=conn_params, include_tags=include_tags)
    concepts = get_concepts(
        conn_params=conn_params,
        concepts_list=concepts_list,
        views_list=views_list,
        include_logic_concepts=include_logic_concepts,
    )

    if not concepts:
        raise Exception("No relevant concepts found for the query.")

    concepts_desc_arr = []
    for concept in concepts.values():
        concept_name = concept.get('concept')
        concept_desc = concept.get('description')
        concept_tags = tags.get('concept_tags').get(concept_name) if concept.get('is_view') == 'false' else tags.get('view_tags').get(concept_name)

        if concept_tags:
            concept_tags = str(concept_tags).replace('{', '').replace('}', '').replace("'", '')

        concept_verbose = f"`{concept_name}`"
        if concept_desc:
            concept_verbose += f" (description: {concept_desc})"
        if concept_tags:
            concept_verbose += f" [tags: {concept_tags}]"
            concepts[concept_name]['tags'] = f"- Annotations and constraints: {concept_tags}\n"

        concepts_desc_arr.append(concept_verbose)
    
    combined_list = concepts_list + views_list
    
    if len(combined_list) == 1 and not (combined_list[0].lower() == 'none' or combined_list[0].lower() == 'null'):
        # If only one concept is provided, return it directly
        determined_concept_name = concepts_list[0] if concepts_list else views_list[0]

        if determined_concept_name not in concepts:
            raise Exception(f"'{determined_concept_name}' was not found in the ontology.")

    else:
        # Use LLM to determine the concept based on the question
        iteration = 0
        error = ''
        while determined_concept_name is None and iteration < retries:
            iteration += 1
            err_txt = f"\nLast try got an error: {error}" if error else ""
            prompt = determine_concept_prompt.format_messages(
                question=question.strip(),
                concepts=",".join(concepts_desc_arr),
                note=(note or '') + err_txt,
            )

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

            response_text = _get_response_text(response)
            candidate = response_text.strip()
            if should_validate and candidate not in concepts.keys():
                error = f"Concept '{determined_concept_name}' not found in the list of concepts."
                continue
            
            determined_concept_name = candidate
            error = ''

        if determined_concept_name is None and error != '':
            raise Exception(f"Failed to determine concept: {error}")

    if determined_concept_name:
        schema = 'vtimbr' if concepts.get(determined_concept_name).get('is_view') == 'true' else 'dtimbr'
    return {
        "concept_metadata": concepts.get(determined_concept_name) if determined_concept_name else None,
        "concept": determined_concept_name,
        "schema": schema,
        "usage_metadata": usage_metadata,
    }


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


def _parse_sql_from_llm_response(response: Any) -> str:
    response_text = _get_response_text(response)
    return (response_text
            .replace("```sql", "")
            .replace("```", "")
            .replace('SELECT \n', 'SELECT ')
            .replace(';', '')
            .strip())


def _get_active_datasource(conn_params: dict) -> dict:
    datasources = get_datasources(conn_params, filter_active=True)
    return datasources[0] if datasources else None


def _evaluate_sql_enable_reasoning(
    question: str,
    sql_query: str,
    llm: LLM,
    conn_params: dict,
    timeout: int,
) -> dict:
    """
    Evaluate if the generated SQL correctly answers the business question.
    
    Returns:
        dict with 'assessment' ('correct'|'partial'|'incorrect') and 'reasoning'
    """
    generate_sql_reasoning_template = get_generate_sql_reasoning_prompt_template(conn_params)
    prompt = generate_sql_reasoning_template.format_messages(
        question=question.strip(),
        sql_query=sql_query.strip(),
    )

    apx_token_count = _calculate_token_count(llm, prompt)
    if hasattr(llm, "_llm_type") and "snowflake" in llm._llm_type:
        _clean_snowflake_prompt(prompt)
    
    response = _call_llm_with_timeout(llm, prompt, timeout=timeout)
    
    # Extract JSON from response content (handle markdown code blocks)
    content = response.content.strip()
    
    # Remove markdown code block markers if present
    if content.startswith("```json"):
        content = content[7:]  # Remove ```json
    elif content.startswith("```"):
        content = content[3:]  # Remove ```
    
    if content.endswith("```"):
        content = content[:-3]  # Remove closing ```
    
    content = content.strip()
    
    # Parse JSON response
    evaluation = json.loads(content)
    
    return {
        "evaluation": evaluation,
        "apx_token_count": apx_token_count,
        "usage_metadata": _extract_usage_metadata(response),
    }


@cache_with_version_check
def _build_sql_generation_context(
    conn_params: dict,
    schema: str,
    concept: str,
    concept_metadata: dict,
    graph_depth: int,
    include_tags: Optional[str],
    exclude_properties: Optional[list],
    db_is_case_sensitive: bool,
    max_limit: int,
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

    columns_str = _build_columns_str(columns, columns_tags=tags, exclude=exclude_properties)
    measures_str = _build_columns_str(measures, tags, exclude=exclude_properties)
    rel_prop_str = _build_rel_columns_str(relationships, columns_tags=tags, exclude_properties=exclude_properties)

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
    measures_context = f"- {MEASURES_DESCRIPTION}: {measures_str}\n" if measures_str else ""
    transitive_context = f"- {TRANSITIVE_RELATIONSHIP_DESCRIPTION}\n" if has_transitive_relationships else ""
    
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
    conn_params: dict,
    generate_sql_prompt: Any,
    current_context: dict,
    note: str,
    should_validate_sql: bool,
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
    
    result = {
        "sql": _parse_sql_from_llm_response(response),
        "apx_token_count": apx_token_count,
        "usage_metadata": _extract_usage_metadata(response),
        "is_valid": True,
        "error": None,
    }
    
    if debug:
        result["p_hash"] = encrypt_prompt(prompt)
    
    if should_validate_sql:
        result["is_valid"], result["error"] = validate_sql(result["sql"], conn_params)
    
    return result


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
        should_validate_sql: Optional[bool] = False,
        retries: Optional[int] = 3,
        max_limit: Optional[int] = 500,
        note: Optional[str] = '',
        db_is_case_sensitive: Optional[bool] = False,
        graph_depth: Optional[int] = 1,
        enable_reasoning: Optional[bool] = False,
        reasoning_steps: Optional[int] = 2,
        debug: Optional[bool] = False,
        timeout: Optional[int] = None,
    ) -> dict[str, str]:
    usage_metadata = {}
    concept_metadata = None
    reasoning_status = 'correct'
    
    # Use config default timeout if none provided
    if timeout is None:
        timeout = llm_timeout
    
    generate_sql_prompt = get_generate_sql_prompt_template(conn_params)
   
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
    concept, schema, concept_metadata = determine_concept_res.get('concept'), determine_concept_res.get('schema'), determine_concept_res.get('concept_metadata')
    usage_metadata.update(determine_concept_res.get('usage_metadata', {}))

    if not concept:
        raise Exception("No relevant concept found for the query.")

    sql_query = None
    iteration = 0
    is_sql_valid = True
    error = ''
    while sql_query is None or (should_validate_sql and iteration < retries and not is_sql_valid):
        iteration += 1
        err_txt = f"\nThe original SQL (`{sql_query}`) was invalid with error: {error}. Please generate a corrected query." if error and "snowflake" not in llm._llm_type else ""

        try:
            result = _generate_sql_with_llm(
                question=question,
                llm=llm,
                conn_params=conn_params,
                generate_sql_prompt=generate_sql_prompt,
                current_context=_build_sql_generation_context(
                    conn_params=conn_params,
                    schema=schema,
                    concept=concept,
                    concept_metadata=concept_metadata,
                    graph_depth=graph_depth,
                    include_tags=include_tags,
                    exclude_properties=exclude_properties,
                    db_is_case_sensitive=db_is_case_sensitive,
                    max_limit=max_limit),
                note=note + err_txt,
                should_validate_sql=should_validate_sql,
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
            is_sql_valid = result['is_valid']
            error = result['error']
            
        except TimeoutError as e:
            error = f"LLM call timed out: {str(e)}"
            raise Exception(error)
        except Exception as e:
            error = f"LLM call failed: {str(e)}"
            if should_validate_sql:
                continue
            else:
                raise Exception(error)
    
    
    if enable_reasoning and sql_query is not None:
        for step in range(reasoning_steps):
            try:
                # Step 1: Evaluate the current SQL
                eval_result = _evaluate_sql_enable_reasoning(
                    question=question,
                    sql_query=sql_query,
                    llm=llm,
                    conn_params=conn_params,
                    timeout=timeout,
                )
                
                usage_metadata[f'sql_reasoning_step_{step + 1}'] = {
                    "approximate": eval_result['apx_token_count'],
                    **eval_result['usage_metadata'],
                }
                
                evaluation = eval_result['evaluation']
                reasoning_status = evaluation.get("assessment", "partial").lower()
                
                if reasoning_status == "correct":
                    break
                
                # Step 2: Regenerate SQL with feedback
                evaluation_note = note + f"\n\nThe previously generated SQL: `{sql_query}` was assessed as '{evaluation.get('assessment')}' because: {evaluation.get('reasoning', '*could not determine cause*')}. Please provide a corrected SQL query that better answers the question: '{question}'."
                
                # Increase graph depth for 2nd+ reasoning attempts, up to max of 3
                context_graph_depth = min(3, int(graph_depth) + step) if graph_depth < 3 and step > 0 else graph_depth
                regen_result = _generate_sql_with_llm(
                    question=question,
                    llm=llm,
                    conn_params=conn_params,
                    generate_sql_prompt=generate_sql_prompt,
                    current_context=_build_sql_generation_context(
                        conn_params=conn_params,
                        schema=schema,
                        concept=concept,
                        concept_metadata=concept_metadata,
                        graph_depth=context_graph_depth,
                        include_tags=include_tags,
                        exclude_properties=exclude_properties,
                        db_is_case_sensitive=db_is_case_sensitive,
                        max_limit=max_limit),
                    note=evaluation_note,
                    should_validate_sql=should_validate_sql,
                    timeout=timeout,
                    debug=debug,
                )
                
                usage_metadata[f'generate_sql_reasoning_step_{step + 1}'] = {
                    "approximate": regen_result['apx_token_count'],
                    **regen_result['usage_metadata'],
                }
                if debug and 'p_hash' in regen_result:
                    usage_metadata[f'generate_sql_reasoning_step_{step + 1}']['p_hash'] = regen_result['p_hash']
                
                sql_query = regen_result['sql']
                is_sql_valid = regen_result['is_valid']
                error = regen_result['error']
                
            except TimeoutError as e:
                raise Exception(f"LLM call timed out: {str(e)}")
            except Exception as e:
                print(f"Warning: LLM reasoning failed: {e}")
                break
    
    return {
        "sql": sql_query,
        "concept": concept,
        "schema": schema,
        "error": error if not is_sql_valid else None,
        "is_sql_valid": is_sql_valid if should_validate_sql else None,
        "reasoning_status": reasoning_status,
        "usage_metadata": usage_metadata,
    }


def answer_question(
    question: str,
    llm: LLM,
    conn_params: dict,
    results: str,
    sql: Optional[str] = None,
    timeout: Optional[int] = None,
    note: Optional[str] = '',
    debug: Optional[bool] = False,
) -> dict[str, Any]:
    # Use config default timeout if none provided
    if timeout is None:
        timeout = llm_timeout

    qa_prompt = get_qa_prompt_template(conn_params)

    prompt = qa_prompt.format_messages(
        question=question,
        formatted_rows=results,
        additional_context=f"SQL QUERY:\n{sql}\n\n" if sql else "",
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

