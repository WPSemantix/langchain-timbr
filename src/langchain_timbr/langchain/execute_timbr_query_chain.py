from typing import Optional, Union, Dict, Any
from langchain.chains.base import Chain
from langchain.llms.base import LLM

from ..utils.general import parse_list, to_boolean, to_integer, validate_timbr_connection_params
from ..utils.timbr_utils import run_query, validate_sql
from ..utils.timbr_llm_utils import generate_sql
from ..llm_wrapper.llm_wrapper import LlmWrapper
from .. import config

class ExecuteTimbrQueryChain(Chain):
    """
    LangChain chain for executing SQL queries against Timbr knowledge graph databases.
    
    This chain executes SQL queries on Timbr ontology/knowledge graph databases and 
    returns the query results, handling retries and result validation. It uses an LLM
    for query generation and connects to Timbr via URL and token.
    """
    
    def __init__(
        self,
        llm: Optional[LLM] = None,
        url: Optional[str] = None,
        token: Optional[str] = None,
        ontology: Optional[str] = None,
        schema: Optional[str] = 'dtimbr',
        concept: Optional[str] = None,
        concepts_list: Optional[Union[list[str], str]] = None,
        views_list: Optional[Union[list[str], str]] = None,
        include_logic_concepts: Optional[bool] = False,
        include_tags: Optional[Union[list[str], str]] = None,
        exclude_properties: Optional[Union[list[str], str]] = ['entity_id', 'entity_type', 'entity_label'],
        should_validate_sql: Optional[bool] = False,
        retries: Optional[int] = 3,
        max_limit: Optional[int] = 500,
        retry_if_no_results: Optional[bool] = False,
        no_results_max_retries: Optional[int] = 2,
        note: Optional[str] = '',
        db_is_case_sensitive: Optional[bool] = False,
        graph_depth: Optional[int] = 1,
        verify_ssl: Optional[bool] = True,
        is_jwt: Optional[bool] = False,
        jwt_tenant_id: Optional[str] = None,
        conn_params: Optional[dict] = None,
        debug: Optional[bool] = False,
        **kwargs,
    ):
        """
        :param llm: An LLM instance or a function that takes a prompt string and returns the LLM's response (optional, will use LlmWrapper with env variables if not provided)
        :param url: Timbr server url (optional, defaults to TIMBR_URL environment variable)
        :param token: Timbr password or token value (optional, defaults to TIMBR_TOKEN environment variable)
        :param ontology: The name of the ontology/knowledge graph (optional, defaults to ONTOLOGY/TIMBR_ONTOLOGY environment variable)
        :param schema: The name of the schema to query
        :param concept: The name of the concept to query
        :param concepts_list: Optional specific concept options to query
        :param views_list: Optional specific view options to query
        :param include_logic_concepts: Optional boolean to include logic concepts (concepts without unique properties which only inherits from an upper level concept with filter logic) in the query.
        :param include_tags: Optional specific concepts & properties tag options to use in the query (Disabled by default. Use '*' to enable all tags or a string represents a list of tags divided by commas (e.g. 'tag1,tag2')
        :param exclude_properties: Optional specific properties to exclude from the query (entity_id, entity_type & entity_label by default).
        :param should_validate_sql: Whether to validate the SQL before executing it
        :param retries: Number of retry attempts if the generated SQL is invalid
        :param max_limit: Maximum number of rows to return
        :retry_if_no_results: Whether to infer the result value from the SQL query. If the query won't return any rows, it will try to re-generate the SQL query then re-run it.
        :param no_results_max_retries: Number of retry attempts to infer the result value from the SQL query
        :param note: Optional additional note to extend our llm prompt
        :param db_is_case_sensitive: Whether the database is case sensitive (default is False).
        :param graph_depth: Maximum number of relationship hops to traverse from the source concept during schema exploration (default is 1).
        :param verify_ssl: Whether to verify SSL certificates (default is True).
        :param is_jwt: Whether to use JWT authentication (default is False).
        :param jwt_tenant_id: JWT tenant ID for multi-tenant environments (required when is_jwt=True).
        :param conn_params: Extra Timbr connection parameters sent with every request (e.g., 'x-api-impersonate-user').
        :param kwargs: Additional arguments to pass to the base
        :return: A list of rows from the Timbr query

        ## Example
        ```
        # Using explicit parameters
        execute_timbr_query_chain = ExecuteTimbrQueryChain(
            url=<url>,
            token=<token>,
            llm=<llm or timbr_llm_wrapper instance>,
            ontology=<ontology_name>,
            schema=<schema_name>,
            concept=<concept_name>,
            concepts_list=<concepts>,
            views_list=<views>,
            should_validate_sql=False,
            note=<note>,
        )

        # Using environment variables for timbr environment (TIMBR_URL, TIMBR_TOKEN, TIMBR_ONTOLOGY)
        execute_timbr_query_chain = ExecuteTimbrQueryChain(
            llm=<llm or timbr_llm_wrapper instance>,
        )

        # Using environment variables for both timbr environment & llm (TIMBR_URL, TIMBR_TOKEN, TIMBR_ONTOLOGY, LLM_TYPE, LLM_API_KEY, etc.)
        execute_timbr_query_chain = ExecuteTimbrQueryChain()

        return execute_timbr_query_chain.invoke({ "prompt": question }).get("rows", [])
        ```
        """
        super().__init__(**kwargs)
        
        # Initialize LLM - use provided one or create with LlmWrapper from env variables
        if llm is not None:
            self._llm = llm
        else:
            try:
                self._llm = LlmWrapper()
            except Exception as e:
                raise ValueError(f"Failed to initialize LLM from environment variables. Either provide an llm parameter or ensure LLM_TYPE and LLM_API_KEY environment variables are set. Error: {e}")
        
        self._url = url if url is not None else config.url
        self._token = token if token is not None else config.token
        self._ontology = ontology if ontology is not None else config.ontology
        
        # Validate required parameters
        validate_timbr_connection_params(self._url, self._token)
        
        self._schema = schema
        self._concept = concept
        self._concepts_list = parse_list(concepts_list)
        self._views_list = parse_list(views_list)
        self._include_tags = parse_list(include_tags)
        self._include_logic_concepts = to_boolean(include_logic_concepts)
        self._exclude_properties = parse_list(exclude_properties)
        self._should_validate_sql = to_boolean(should_validate_sql)
        self._retries = to_integer(retries)
        self._max_limit = to_integer(max_limit)
        self._retry_if_no_results = to_boolean(retry_if_no_results)
        self._no_results_max_retries = to_integer(no_results_max_retries)
        self._note = note
        self._db_is_case_sensitive = to_boolean(db_is_case_sensitive)
        self._graph_depth = to_integer(graph_depth)
        self._verify_ssl = to_boolean(verify_ssl)
        self._is_jwt = to_boolean(is_jwt)
        self._jwt_tenant_id = jwt_tenant_id
        self._debug = to_boolean(debug)
        self._conn_params = conn_params or {}


    @property
    def usage_metadata_key(self) -> str:
        return "execute_timbr_usage_metadata"


    @property
    def input_keys(self) -> list:
        return ["prompt"]


    @property
    def output_keys(self) -> list:
        return [
            "rows",
            "sql",
            "schema",
            "concept",
            "error",
            self.usage_metadata_key,
        ]


    def _get_conn_params(self) -> dict:
        return {
            "url": self._url,
            "token": self._token,
            "ontology": self._ontology,
            "verify_ssl": self._verify_ssl,
            "is_jwt": self._is_jwt,
            "jwt_tenant_id": self._jwt_tenant_id,
            **self._conn_params,
        }


    def _validate_inputs(self, inputs: Dict[str, Any]) -> None:
        if (not inputs.get("sql")) and (not inputs.get("prompt")):
            raise ValueError("Timbr SQL or user prompt is required for executing the chain.")


    def _generate_sql(
        self,
        prompt: str,
        sql: Optional[str] = None,
        concept_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:

        if not prompt:
            raise ValueError("Timbr SQL or user prompt is required for executing the chain.")

        err_txt = f"\nThe original SQL (`{sql}`) was invalid with error: {error}. Please generate a corrected query." if error else ""
        generate_res = generate_sql(
            prompt,
            self._llm,
            self._get_conn_params(),
            concept=concept_name,
            schema=schema_name,
            concepts_list=self._concepts_list,
            views_list=self._views_list,
            include_tags=self._include_tags,
            include_logic_concepts=self._include_logic_concepts,
            exclude_properties=self._exclude_properties,
            should_validate_sql=self._should_validate_sql,
            retries=self._retries,
            max_limit=self._max_limit,
            note=(self._note or '') + err_txt,
            db_is_case_sensitive=self._db_is_case_sensitive,
            graph_depth=self._graph_depth,
            debug=self._debug,
        )

        return generate_res


    def _has_no_meaningful_results(self, rows: list) -> bool:
        """
        Check if the rows returned from the query are empty or do not contain meaningful data.
        This can be customized based on specific criteria for what constitutes "meaningful" results.
        """
        if not rows:
            return True
        
        # Check if all rows have all None values
        for row in rows:
            if any(value is not None for value in row.values()):
                return False
        
        return True


    def _call(self, inputs: Dict[str, Any], run_manager=None) -> Dict[str, Any]:
        try:
            prompt = inputs.get("prompt")
            sql = inputs.get("sql", None)
            schema_name = inputs.get("schema", self._schema)
            concept_name = inputs.get("concept", self._concept)
            is_sql_valid = True
            error = None
            usage_metadata = {}

            if sql and self._should_validate_sql:
                is_sql_valid, error = validate_sql(sql, self._get_conn_params())

            is_infered = False
            iteration = 0
            generated = []
            while not is_infered and iteration <= self._no_results_max_retries:
                if prompt is not None and not sql or not is_sql_valid:
                    generate_res = self._generate_sql(prompt, sql, concept_name, schema_name, error)
                    
                    sql = generate_res.get("sql", "")
                    schema_name = generate_res.get("schema", schema_name)
                    concept_name = generate_res.get("concept", concept_name)
                    is_sql_valid = generate_res.get("is_sql_valid")
                    if not is_sql_valid and not self._should_validate_sql:
                        is_sql_valid = True

                    error = generate_res.get("error")
                    usage_metadata = self._summarize_usage_metadata(usage_metadata, generate_res.get("usage_metadata", {}))
                
                is_sql_not_tried = not any(sql.lower().strip() == gen.lower().strip() for gen in generated)

                rows = run_query(
                    sql,
                    self._get_conn_params(),
                    llm_prompt=prompt
                ) if is_sql_valid and is_sql_not_tried else []
                
                if iteration < self._no_results_max_retries:
                    # If no rows are returned and we should infer the result, we will try to re-generate the SQL query
                    if prompt is not None and self._retry_if_no_results and self._has_no_meaningful_results(rows):
                        if is_sql_not_tried:
                            generated.append(sql)
                            # If the SQL is valid but no rows are returned, create an error message to be sent to the LLM
                            if is_sql_valid:
                                error = "No rows returned. Please revise the SQL considering if the question was ambiguous (e.g., which ID or name to use), try use alternative columns in the WHERE clause part in a way that could match the user's intent, without adding new columns with new filters."
                                error += "\nConsider that this queries already generated and returned 0 rows:\n" + "\n".join(generated)
                                is_sql_valid = False
                        else:
                            # Generated twice the same SQL, so we will stop the loop
                            is_infered = True
                    else:
                        is_infered = True
                iteration += 1

            return {
                "rows": rows,
                "sql": sql,
                "schema": schema_name,
                "concept": concept_name,
                "error": error if not is_sql_valid else None,
                self.usage_metadata_key: usage_metadata,
            }

        except Exception as e:
            raise RuntimeError(f"Error executing the chain: {str(e)}")

    def _summarize_usage_metadata(self, current_metadata: dict, new_metadata: dict) -> dict:
        """
        Summarize usage metadata by aggregating specific numeric keys and overriding others.
        
        :param current_metadata: The existing usage metadata dictionary
        :param new_metadata: The new usage metadata to be added
        :return: Updated usage metadata dictionary
        """
        keys_to_sum = ['approximate', 'input_tokens', 'output_tokens', 'total_tokens']
        
        for outer_key, outer_value in new_metadata.items():
            if isinstance(outer_value, dict):
                if outer_key not in current_metadata:
                    current_metadata[outer_key] = {}
                
                for inner_key, inner_value in outer_value.items():
                    if inner_key in keys_to_sum:
                        # Sum the numeric values
                        current_val = current_metadata[outer_key].get(inner_key, 0)
                        if isinstance(inner_value, (int, float)) and isinstance(current_val, (int, float)):
                            current_metadata[outer_key][inner_key] = current_val + inner_value
                        else:
                            current_metadata[outer_key][inner_key] = inner_value
                    else:
                        # Override other keys
                        current_metadata[outer_key][inner_key] = inner_value
            else:
                # If the outer value is not a dict, just override it
                current_metadata[outer_key] = outer_value
        
        return current_metadata

