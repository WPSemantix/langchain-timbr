from typing import Any, Optional
from langchain.llms.base import LLM
import base64, hashlib
from cryptography.fernet import Fernet
from datetime import datetime
import concurrent.futures
import time

from .timbr_utils import get_datasources, get_tags, get_concepts, get_concept_properties, validate_sql, get_properties_description, get_relationships_description
from .prompt_service import (
    get_determine_concept_prompt_template,
    get_generate_sql_prompt_template,
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


def generate_key() -> bytes:
    """Generate a new Fernet secret key."""
    passcode = b"lucylit2025"
    hlib = hashlib.md5()
    hlib.update(passcode)
    return base64.urlsafe_b64encode(hlib.hexdigest().encode('utf-8'))


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

ENCRYPT_KEY = generate_key()
MEASURES_DESCRIPTION = "The following columns are calculated measures and can only be aggregated with an aggregate function: COUNT/SUM/AVG/MIN/MAX (count distinct is not allowed)"
TRANSITIVE_RELATIONSHIP_DESCRIPTION = "Transitive relationship columns allow you to access data through multiple relationship hops. These columns follow the pattern `<relationship_name>[<table_name>*<number>].<column_name>` where the number after the asterisk (*) indicates how many relationship levels to traverse. For example, `acquired_by[company*4].company_name` means 'go through up to 4 levels of the acquired_by relationship to get the company name', while columns ending with '_transitivity_level' indicate the actual relationship depth (Cannot be null or 0 - level 1 represents direct relationships, while levels 2, 3, 4, etc. represent indirect relationships through multiple hops. To filter by relationship type, use `_transitivity_level = 1` for direct relationships only, `_transitivity_level > 1` for indirect relationships only."

def encrypt_prompt(prompt: Any, key: Optional[bytes] = ENCRYPT_KEY) -> bytes:
    """Serialize & encrypt the prompt; returns a URL-safe token."""
    # build prompt_text as before…
    if isinstance(prompt, str):
        text = prompt
    elif isinstance(prompt, list):
        parts = []
        for message in prompt:
            if hasattr(message, "content"):
                parts.append(f"{message.type}: {message.content}")
            else:
                parts.append(str(message))
        text = "\n".join(parts)
    else:
        text = str(prompt)

    f = Fernet(key)
    return f.encrypt(text.encode()).decode('utf-8')


def decrypt_prompt(token: bytes, key: bytes) -> str:
    """Decrypt the token and return the original prompt string."""
    f = Fernet(key)
    return f.decrypt(token).decode()


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


def _calculate_token_count(llm: LLM, prompt: str) -> int:
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
    usage_metadata = response.response_metadata

    if usage_metadata and 'usage' in usage_metadata:
        usage_metadata = usage_metadata['usage']

    if not usage_metadata and 'usage_metadata' in response:
        usage_metadata = response.usage_metadata
        if usage_metadata and 'usage' in usage_metadata:
            usage_metadata = usage_metadata['usage']

    if not usage_metadata and 'usage' in response:
        usage_metadata = response.usage
        if usage_metadata and 'usage' in usage_metadata:
            usage_metadata = usage_metadata['usage']

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
        debug: Optional[bool] = False,
        timeout: Optional[int] = None,
    ) -> dict[str, str]:
    usage_metadata = {}
    concept_metadata = None
    
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

    datasource_type = _get_active_datasource(conn_params).get('target_type')

    properties_desc = get_properties_description(conn_params=conn_params)
    relationships_desc = get_relationships_description(conn_params=conn_params)
  
    concept_properties_metadata = get_concept_properties(schema=schema, concept_name=concept, conn_params=conn_params, properties_desc=properties_desc, relationships_desc=relationships_desc, graph_depth=graph_depth)
    columns, measures, relationships = concept_properties_metadata.get('columns', []), concept_properties_metadata.get('measures', []), concept_properties_metadata.get('relationships', {})
    tags = get_tags(conn_params=conn_params, include_tags=include_tags).get('property_tags')

    columns_str = _build_columns_str(columns, columns_tags=tags, exclude=exclude_properties)
    measures_str = _build_columns_str(measures, tags, exclude=exclude_properties)
    rel_prop_str = _build_rel_columns_str(relationships, columns_tags=tags, exclude_properties=exclude_properties)

    if rel_prop_str:
        measures_str += f"\n{rel_prop_str}"

    sql_query = None
    iteration = 0
    is_sql_valid = True
    error = ''
    while sql_query is None or (should_validate_sql and iteration < retries and not is_sql_valid):
        iteration += 1
        err_txt = f"\nThe original SQL (`{sql_query}`) was invalid with error: {error}. Please generate a corrected query." if error and "snowflake" not in llm._llm_type else ""

        sensitivity_txt = "- Ensure value comparisons are case-insensitive, e.g., use LOWER(column) = 'value'.\n" if db_is_case_sensitive else ""

        measures_context = f"- {MEASURES_DESCRIPTION}: {measures_str}\n" if measures_str else ""
        has_transitive_relationships = any(
            rel.get('is_transitive')
            for rel in relationships.values()
        ) if relationships else False
        transitive_context = f"- {TRANSITIVE_RELATIONSHIP_DESCRIPTION}\n" if has_transitive_relationships else ""
        concept_description = f"- Description: {concept_metadata.get('description')}\n" if concept_metadata and concept_metadata.get('description') else ""
        concept_tags = concept_metadata.get('tags') if concept_metadata and concept_metadata.get('tags') else ""
        cur_date = datetime.now().strftime("%Y-%m-%d")
        prompt = generate_sql_prompt.format_messages(
            current_date=cur_date,
            datasource_type=datasource_type or 'standard sql',
            schema=schema,
            concept=f"`{concept}`",
            description=concept_description or "",
            tags=concept_tags or "",
            question=question,
            columns=columns_str,
            measures_context=measures_context,
            transitive_context=transitive_context,
            sensitivity_context=sensitivity_txt,
            max_limit=max_limit,
            note=note + err_txt,
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
            if should_validate_sql:
                continue
            else:
                raise Exception(error)

        usage_metadata['generate_sql'] = {
            "approximate": apx_token_count,
            **_extract_usage_metadata(response),
        }
        if debug:
            usage_metadata['generate_sql']["p_hash"] = encrypt_prompt(prompt)

        sql_query = _parse_sql_from_llm_response(response)

        if should_validate_sql:
            is_sql_valid, error = validate_sql(sql_query, conn_params)
    
    return {
        "sql": sql_query,
        "concept": concept,
        "schema": schema,
        "error": error if not is_sql_valid else None,
        "is_sql_valid": is_sql_valid if should_validate_sql else None,
        "usage_metadata": usage_metadata,
    }


def answer_question(
    question: str,
    llm: LLM,
    conn_params: dict,
    results: str,
    sql: Optional[str] = None,
    debug: Optional[bool] = False,
    timeout: Optional[int] = None,
) -> dict[str, Any]:
    # Use config default timeout if none provided
    if timeout is None:
        timeout = llm_timeout

    qa_prompt = get_qa_prompt_template(conn_params)

    prompt = qa_prompt.format_messages(
        question=question,
        formatted_rows=results,
        additional_context=f"SQL QUERY:\n{sql}\n\n" if sql else "",
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

