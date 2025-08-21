from typing import Optional, Any
import time
from pytimbr_api import timbr_http_connector
from functools import wraps

from ..config import cache_timeout, ignore_tags, ignore_tags_prefix
from .general import to_boolean

# Cache dictionary
_cache = {}
_ontology_version = None
_last_version_check = 0

def clear_cache():
    """Clear the cache and reset the ontology version."""
    global _cache, _ontology_version
    # with cache_lock:
    _cache.clear()
    _ontology_version = None


def _get_ontology_version(conn_params) -> str:
    """Fetch the current ontology version."""
    query = "SHOW VERSION"
    res = run_query(query, conn_params)
    return res[0].get("id") if res else "unknown"


def _serialize_cache_key(*args, **kwargs):
    """Serialize arguments into a hashable cache key."""
    def serialize(obj):
        if isinstance(obj, dict):
            return tuple(sorted((k, serialize(v)) for k, v in obj.items()))
        elif isinstance(obj, list):
            return tuple(serialize(x) for x in obj)
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        raise TypeError(f"Unsupported type for caching: {type(obj)}")

    return (tuple(serialize(arg) for arg in args), tuple((k, serialize(v)) for k, v in kwargs.items()))


def cache_with_version_check(func):
    """Decorator to cache function results and invalidate if ontology version changes."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        global _ontology_version, _last_version_check

        now = time.time()
        if (now - _last_version_check) > cache_timeout:
            conn_params = kwargs.get("conn_params") or args[-1]
            current_version = _get_ontology_version(conn_params)

            # If version changed, clear cache and set new version
            if _ontology_version != current_version:
                clear_cache()
                _ontology_version = current_version

            _last_version_check = now
        
        # Generate a cache key based on function name and arguments
        cache_key = (func.__name__, _serialize_cache_key(*args, **kwargs))
        if cache_key not in _cache or not _cache[cache_key]:
            # Call the function and store the result in the cache
            _cache[cache_key] = func(*args, **kwargs)

        return _cache[cache_key]

    return wrapper


def run_query(sql: str, conn_params: dict, llm_prompt: Optional[str] = None) -> list[list]:
    if not conn_params:
        raise("Please provide connection params.")

    query = sql
    if llm_prompt:
        clean_prompt = llm_prompt.replace('\r\n', ' ').replace('\n', ' ').replace('?', '')
        query = f"-- LLM: {clean_prompt}\n{sql}"

    results = timbr_http_connector.run_query(
        query=query,
        **conn_params,
    )

    return results


def get_ontologies(conn_params: dict) -> list[str]:
    query = "SELECT ontology FROM timbr.sys_ontologies"
    res = run_query(query, conn_params)
    return [row.get('ontology') for row in res]


def get_datasources(conn_params: dict, filter_active: Optional[bool] = False) -> list[dict]:
    query = "SHOW DATASOURCES"
    res = run_query(query, conn_params)
    if filter_active:
        res = [row for row in res if to_boolean(row.get('is_active'))]
    
    return res


def validate_sql(sql: str, conn_params: dict) -> tuple[bool, str]:
    if not sql:
        raise Exception("Please provide SQL to validate.")
    
    explain_res = None
    query_res = None
    error = None

    try:
        explain_sql = f"EXPLAIN {sql}"
        explain_res = run_query(explain_sql, conn_params)
        
        query_sql = f"SELECT * FROM ({sql.replace(';', '')}) explainable_query WHERE 1=0"
        query_res = run_query(query_sql, conn_params)
    except Exception as e:
        error = str(getattr(e, 'doc', e))

    return to_boolean(explain_res and explain_res[0].get('PLAN') and query_res is not None), error


def _should_ignore_tag(tag_name: str) -> bool:
    if not tag_name:
        return True
    
    tag_name_lower = tag_name.lower()
    if tag_name_lower in ignore_tags:
        return True
    
    for prefix in ignore_tags_prefix:
        if tag_name_lower.startswith(prefix.lower()):
            return True
            
    return False


def _prepare_tags_dict(
    type: Optional[str] = 'concept',
    tags_list: Optional[list] = [],
    include_tags: Optional[str] = '',
) -> dict:
    tags_dict = {}
    if not include_tags:
        return tags_dict # currently empty
    
    for tag in tags_list:
        # Make sure that the tag is of the correct type
        if type != tag.get('target_type'):
            continue

        tag_name = tag.get('tag_name')

        # Check if the tag is included
        if (include_tags != '*' and tag_name not in include_tags) or _should_ignore_tag(tag_name):
            continue
        
        key = tag.get('target_name')
        tag_value = tag.get('tag_value')

        if key not in tags_dict:
            tags_dict[key] = {}
        
        tags_dict[key][tag_name] = tag_value
        
    return tags_dict


@cache_with_version_check
def get_tags(conn_params: dict, include_tags: Optional[Any] = None) -> dict:
    if not to_boolean(include_tags):
        return {
            "concept_tags": {},
            "view_tags": {},
            "property_tags": {},
            # "relationship_tags": {},
        }

    query = "SHOW TAGS"
    ontology_tags = run_query(query, conn_params)

    return {
        "concept_tags": _prepare_tags_dict('concept', ontology_tags, include_tags),
        "view_tags": _prepare_tags_dict('ontology view', ontology_tags, include_tags),
        "property_tags": _prepare_tags_dict('property', ontology_tags, include_tags),
        # "relationship_tags": _prepare_tags_dict('relationship', ontology_tags, include_tags),
    }


def _should_ignore_list(list: list) -> bool:
    return bool(list and len(list) == 1 and (list[0] == 'none' or list[0] == 'null'))


def _should_select_all(list: list) -> bool:
    return list and len(list) == 1 and list[0] == '*'


@cache_with_version_check
def get_concepts(
    conn_params,
    concepts_list: Optional[list] = None,
    views_list: Optional[list] = None,
    include_logic_concepts: Optional[bool] = False,
) -> dict:
    """Fetch concepts (or views) from timbr.sys_concepts and/or timbr.sys_views."""
    joined_views = ','.join(f"'{v}'" for v in views_list) if views_list else ''
    should_ignore_concepts = _should_ignore_list(concepts_list)
    should_ignore_views = _should_ignore_list(views_list)

    filter_concepts = " WHERE concept IN (SELECT DISTINCT concept FROM timbr.sys_concept_properties)" if not include_logic_concepts else ""
    if concepts_list:
        if should_ignore_concepts:
            filter_concepts = " WHERE 1 = 0"
        elif _should_select_all(concepts_list):
            filter_concepts = ""
        else:
            joined_concepts = ','.join(f"'{c}'" for c in concepts_list) if concepts_list else ''
            filter_concepts = f" WHERE concept IN ({joined_concepts})" if concepts_list else ""

    filter_views = f" WHERE view_name IN ({joined_views})" if views_list else ""
    if should_ignore_views:
        filter_views = " WHERE 1 = 0"
    elif _should_select_all(views_list):
        filter_views = ""

    # if there is concepts_list and not views - filter only concepts
    # if there is views_list and not concepts - filter only views
    # if there is both or none - union the two tables
    if concepts_list and not should_ignore_concepts and not views_list:
        # Only fetch concepts
        query = f"""
            SELECT concept, description, 'false' AS is_view 
            FROM timbr.sys_concepts{filter_concepts}
            ORDER BY is_view ASC
        """.strip()
    elif views_list and not should_ignore_views and not concepts_list:
        # Only fetch views
        query = f"""
            SELECT view_name AS concept, description, 'true' AS is_view
            FROM timbr.sys_views{filter_views}
            ORDER BY is_view ASC
        """.strip()
    else:
        # Both or neither => union the two tables (existing logic)
        query = f"""
            SELECT * FROM (
                SELECT concept, description, 'false' AS is_view 
                FROM timbr.sys_concepts{filter_concepts}
                UNION ALL
                SELECT view_name AS concept, description, 'true' AS is_view 
                FROM timbr.sys_views{filter_views}
            ) AS combined
            ORDER BY is_view ASC
        """.strip()

    res = run_query(query, conn_params)
    uniq_concepts = {}
    for row in res:
        concept = row.get('concept')
        if concept not in uniq_concepts and concept != 'thing':
            uniq_concepts[concept] = row

    return uniq_concepts


def _generate_column_relationship_description(column_name):
    """
    Generates a concise description for a column used in text-to-SQL generation.

    Expected column name formats:
      relationship_name[target_concept].property_name
      or
      relationship_name[target_concept].relationship_name[target_concept].property_name
      (and potentially more nested relationships)

    For example:
      "includes_product[product].contains[material].material_name"

    Output example:
      "This column represents the material name from table material using the relationship contains from table product from relationship includes product."
    """

    try:
        # Split the column name into parts using the period as a delimiter.
        parts = column_name.split('.')
        # The final part is the property name; replace underscores with spaces.
        property_name = parts[-1].replace('_', ' ')

        # Extract relationships (each part before the final property)
        relationships = []
        for part in parts[:-1]:
            if '[' in part and ']' in part:
                relationship, target_concept = part.split('[')
                target_concept = target_concept.rstrip(']')
                # Replace underscores with spaces.
                relationship = relationship.replace('_', ' ')
                target_concept = target_concept.replace('_', ' ')
                relationships.append((relationship, target_concept))

        col_type = "column"
        if column_name.startswith("measure."):
            col_type = "measure"

        # Build the description.
        if relationships:
            # The final table is taken from the target of the last relationship.
            final_table = relationships[-1][1]
            description = f"This {col_type} represents the {property_name} from table {final_table}"
            if len(relationships) == 1:
                # Only one relationship in the chain.
                description += f" using the relationship {relationships[0][0]}."
            else:
                # For two or more relationships:
                # The last relationship is applied on the table from the previous relationship.
                # For example, for two relationships:
                #   relationships[0] = ("includes product", "product")
                #   relationships[1] = ("contains", "material")
                # We want: "using the relationship contains from table product from relationship includes product."
                last_rel, _ = relationships[-1]
                base_table = relationships[-2][1]
                derivation = f" using the relationship {last_rel} from table {base_table}"
                # For any additional relationships (if more than two), append them in order.
                for i in range(len(relationships) - 2, -1, -1):
                    derivation += f" from relationship {relationships[i][0]}"
                description += derivation + "."
        else:
            description = f"This {col_type} represents the {property_name}."

        return description
    except Exception as exp:
      return ""

@cache_with_version_check
def get_relationships_description(conn_params: dict) -> dict:
    """Fetch relationships data."""
    query = f"""
        SELECT 
            relationship_name,
            description
        FROM `timbr`.`SYS_CONCEPT_RELATIONSHIPS`
        WHERE description is not null
    """.strip()

    res = run_query(query, conn_params)
    relationships_desc = {}
    for row in res:
        relationships_desc[row['relationship_name']] = row['description']
    
    return relationships_desc

@cache_with_version_check
def get_properties_description(conn_params: dict) -> dict:
    query = f"""
        SELECT property_name, description
        FROM `timbr`.`SYS_PROPERTIES`
        WHERE description is not null
    """.strip()

    res = run_query(query, conn_params)
    properties_desc = {}
    for row in res:
        properties_desc[row['property_name']] = row['description']
    
    return properties_desc


def _add_relationship_column(
    relationship_name: str,
    relationship_desc: str,
    col_dict: dict,
    relationships: dict,
) -> None:
    """Add a column to the specified relationship."""
    col_name = col_dict.get('name')
    if col_name:
        if relationship_name not in relationships:
            is_transitive = '*' in col_name
            relationships[relationship_name] = {
                "relationship_name": relationship_name,
                "description": relationship_desc,
                "columns": [],
                "measures": [],
                "is_transitive": is_transitive,
            }

        if col_name.startswith('measure.'):
            relationships[relationship_name]['measures'].append(col_dict)
        else:
            relationships[relationship_name]['columns'].append(col_dict)

@cache_with_version_check
def get_concept_properties(
    concept_name: str,
    conn_params: dict,
    properties_desc: dict,
    relationships_desc: dict,
    schema: Optional[str] = 'dtimbr',
    graph_depth: Optional[int] = 1,
) -> dict:
    rows = []
    desc_query = f"describe concept `{schema}`.`{concept_name}`"
    if schema == 'dtimbr':
        desc_query += f" options (graph_depth='{graph_depth}')"

    try:
        rows = run_query(desc_query, conn_params)
    except Exception as e:
        # skipping new describe concept syntax
        pass

    if not rows:
        legacy_desc_query = f"desc `{schema}`.`{concept_name}`"
        try:
            rows = run_query(legacy_desc_query, conn_params)
        except Exception as e:
            print(f"Error describing concept using legacy desc stmt: {e}")

    relationships = {}
    columns = []
    measures = []
    
    for column in rows:
        col_name = column.get('col_name')
        comment = properties_desc.get(col_name)

        if col_name:
            if "_type_of_" in col_name:
                comment = f"if this value is 1, the row is of type {col_name.split('_type_of_')[1]}"
            # elif (comment is None or comment == "") and "[" in col_name and "]" in col_name:
            #     comment = _generate_column_relationship_description(col_name)
            elif col_name.startswith("~"):
                rel_name = col_name[1:].split('[')[0]
                comment = comment + "; " if comment else ''
                comment = comment + f"This columns means the inverse of `{rel_name}`"
            
            if "." in col_name and (comment is None or comment == ""): 
                    comment = properties_desc.get(col_name.split(".")[-1])

            if '[' in col_name:
                # This is a relationship column
                rel_path, rel_col_name = col_name.rsplit('.', 1) if '.' in col_name else col_name.rsplit('_', 1) if '_' in col_name else col_name
                rel_name = rel_path.split('[', 1)[0]

                if rel_name:
                    if rel_name.startswith('measure.'):
                        rel_name = rel_name.replace('measure.', '')
            
                    comment = properties_desc.get(rel_col_name, '')

                    rel_col_dict = {
                        'name': col_name,
                        'col_name': rel_col_name,
                        'type': column.get('data_type', 'string').lower(),
                        'data_type': column.get('data_type', 'string').lower(),
                        'comment': comment,
                    }
                    _add_relationship_column(
                        relationship_name=rel_name,
                        relationship_desc=relationships_desc.get(rel_name, ''),
                        col_dict=rel_col_dict,
                        relationships=relationships
                    )

            elif col_name.startswith("measure."):
                measures.append({ **column, 'comment': comment })
            else:
                columns.append({ **column, 'comment': comment })
    return {
        "columns": columns,
        "measures": measures,
        "relationships": relationships,
    }

