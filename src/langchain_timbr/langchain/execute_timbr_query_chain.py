from typing import Optional, Union, Dict, Any
from ..utils._base_chain import Chain
from langchain_core.language_models.llms import LLM

from langchain_timbr.utils.timbr_utils import get_timbr_agent_options

from ..utils.general import parse_list, to_boolean, to_integer, validate_timbr_connection_params, sanitize_results
from ..utils.timbr_utils import run_query, validate_sql, build_server_url
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

    _ontology: Optional[str] = None
    
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
        should_validate_sql: Optional[bool] = config.should_validate_sql,
        retries: Optional[int] = 3,
        max_limit: Optional[int] = config.llm_default_limit,
        retry_if_no_results: Optional[bool] = config.retry_if_no_results,
        no_results_max_retries: Optional[int] = 2,
        note: Optional[str] = '',
        db_is_case_sensitive: Optional[bool] = False,
        graph_depth: Optional[int] = 1,
        agent: Optional[str] = None,
        verify_ssl: Optional[bool] = True,
        is_jwt: Optional[bool] = False,
        jwt_tenant_id: Optional[str] = None,
        conn_params: Optional[dict] = None,
        enable_reasoning: Optional[bool] = None,
        reasoning_steps: Optional[int] = None,
        debug: Optional[bool] = False,
        enable_trace: Optional[bool] = config.enable_trace,
        conversation_id: Optional[str] = None,
        enable_memory: Optional[bool] = config.enable_memory,
        memory_window_size: Optional[int] = config.memory_window_size,
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
        :param retry_if_no_results: Whether to infer the result value from the SQL query. If the query won't return any rows, it will try to re-generate the SQL query then re-run it.
        :param no_results_max_retries: Number of retry attempts to infer the result value from the SQL query
        :param note: Optional additional note to extend our llm prompt
        :param db_is_case_sensitive: Whether the database is case sensitive (default is False).
        :param graph_depth: Maximum number of relationship hops to traverse from the source concept during schema exploration (default is 1).
        :param agent: Optional Timbr agent name for options setup.
        :param verify_ssl: Whether to verify SSL certificates (default is True).
        :param is_jwt: Whether to use JWT authentication (default is False).
        :param jwt_tenant_id: JWT tenant ID for multi-tenant environments (required when is_jwt=True).
        :param conn_params: Extra Timbr connection parameters sent with every request (e.g., 'x-api-impersonate-user').
        :param enable_reasoning: Whether to enable reasoning during SQL generation (default is False).
        :param reasoning_steps: Number of reasoning steps to perform if reasoning is enabled (default is 2).
        :param enable_trace: Whether to enable trace logging for this chain's operations (default is False).
        :param conversation_id: Optional conversation ID to associate with this chain's execution for tracking and logging in multi-turn conversations.
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
        
        # Validate required parameters
        validate_timbr_connection_params(self._url, self._token)
        
        self._verify_ssl = to_boolean(verify_ssl)
        self._is_jwt = to_boolean(is_jwt)
        self._jwt_tenant_id = jwt_tenant_id
        self._debug = to_boolean(debug)
        self._conn_params = conn_params or {}
        self._max_limit = config.llm_default_limit # use default value so the self._get_conn_params() won't fail before agent options are processed

        self._agent = agent
        if self._agent:
            agent_options = get_timbr_agent_options(self._agent, conn_params=self._get_conn_params())

            self._ontology = agent_options.get("ontology") if "ontology" in agent_options else None
            self._schema = agent_options.get("schema") if "schema" in agent_options else schema
            self._concept = agent_options.get("concept") if "concept" in agent_options else None
            self._concepts_list = parse_list(agent_options.get("concepts_list")) if "concepts_list" in agent_options else None
            self._views_list = parse_list(agent_options.get("views_list")) if "views_list" in agent_options else None
            self._include_tags = parse_list(agent_options.get("include_tags")) if "include_tags" in agent_options else None
            self._include_logic_concepts = to_boolean(agent_options.get("include_logic_concepts")) if "include_logic_concepts" in agent_options else False
            self._exclude_properties = parse_list(agent_options.get("exclude_properties")) if "exclude_properties" in agent_options else ['entity_id', 'entity_type', 'entity_label']
            self._should_validate_sql = to_boolean(agent_options.get("should_validate_sql")) if "should_validate_sql" in agent_options else config.should_validate_sql
            self._retries = to_integer(agent_options.get("retries") if "retries" in agent_options else retries)
            self._max_limit = to_integer(agent_options.get("max_limit")) if "max_limit" in agent_options else config.llm_default_limit
            self._retry_if_no_results = to_boolean(agent_options.get("retry_if_no_results")) if "retry_if_no_results" in agent_options else config.retry_if_no_results
            self._no_results_max_retries = to_integer(agent_options.get("no_results_max_retries")) if "no_results_max_retries" in agent_options else 2
            self._db_is_case_sensitive = to_boolean(agent_options.get("db_is_case_sensitive")) if "db_is_case_sensitive" in agent_options else False
            self._graph_depth = to_integer(agent_options.get("graph_depth")) if "graph_depth" in agent_options else 1
            self._note = agent_options.get("note") if "note" in agent_options else ''
            if note:
                self._note = ((self._note + '\n') if self._note else '') + note
            self._enable_reasoning = to_boolean(agent_options.get("enable_reasoning")) if "enable_reasoning" in agent_options else config.enable_reasoning
            if enable_reasoning is not None and enable_reasoning != self._enable_reasoning:
                self._enable_reasoning = to_boolean(enable_reasoning)
            self._reasoning_steps = to_integer(agent_options.get("reasoning_steps")) if "reasoning_steps" in agent_options else config.reasoning_steps
            if reasoning_steps is not None and reasoning_steps != self._reasoning_steps:
                self._reasoning_steps = to_integer(reasoning_steps)
            self._enable_trace = to_boolean(agent_options.get("enable_trace")) if "enable_trace" in agent_options else to_boolean(enable_trace)
            self._enable_memory = to_boolean(agent_options.get("enable_memory")) if "enable_memory" in agent_options else to_boolean(enable_memory)
            self._memory_window_size = to_integer(agent_options.get("memory_window_size")) if "memory_window_size" in agent_options else to_integer(memory_window_size)
        else:
            self._ontology = ontology if ontology is not None else config.ontology
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
            self._db_is_case_sensitive = to_boolean(db_is_case_sensitive)
            self._graph_depth = to_integer(graph_depth)
            self._note = note
            self._enable_reasoning = to_boolean(enable_reasoning) if enable_reasoning is not None else config.enable_reasoning
            self._reasoning_steps = to_integer(reasoning_steps) if reasoning_steps is not None else config.reasoning_steps
            self._enable_trace = to_boolean(enable_trace)
            self._enable_memory = to_boolean(enable_memory)
            self._memory_window_size = to_integer(memory_window_size)

        self._enable_logging = self._enable_trace
        self._conversation_id = conversation_id


    @property
    def usage_metadata_key(self) -> str:
        return "execute_timbr_usage_metadata"


    @property
    def input_keys(self) -> list:
        return ["prompt", "conversation_id"]


    @property
    def output_keys(self) -> list:
        base = [
            "rows",
            "sql",
            "ontology",
            "schema",
            "concept",
            "error",
            "reasoning_status",
            "identify_concept_reason",
            "generate_sql_reason",
            self.usage_metadata_key,
            "conversation_id",
        ]
        return list(dict.fromkeys(self.input_keys + base))


    def _get_conn_params(self) -> dict:
        return {
            "url": self._url,
            "token": self._token,
            "ontology": self._ontology if self._ontology is not None else config.ontology,
            "verify_ssl": self._verify_ssl,
            "is_jwt": self._is_jwt,
            "jwt_tenant_id": self._jwt_tenant_id,
            "additional_headers": {"results-limit": str(self._max_limit)},
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
        conn_params: dict = None,
        memory_context=None,
    ) -> Dict[str, Any]:

        if not prompt:
            raise ValueError("Timbr SQL or user prompt is required for executing the chain.")

        err_txt = f"\nThe original SQL (`{sql}`) was invalid with error: {error}. Please generate a corrected query." if error else ""
        generate_res = generate_sql(
            prompt,
            self._llm,
            conn_params,
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
            enable_reasoning=self._enable_reasoning,
            reasoning_steps=self._reasoning_steps,
            debug=self._debug,
            memory_context=memory_context,
        )

        return generate_res

    def _has_no_meaningful_results(self, rows: list, sql: str) -> bool:
        """
        Check if the rows returned from the query are empty or do not contain meaningful data.
        This can be customized based on specific criteria for what constitutes "meaningful" results.
        """
        if not rows:
            return True

        # Single-row aggregate returning 0/NULL — filter matched nothing.
        if sql and len(rows) == 1:
            sql_lower = sql.lower()
            if any(fn in sql_lower for fn in ('count(', 'sum(', 'avg(', 'min(', 'max(')):
                if all(v is None or v == 0 for v in rows[0].values()):
                    return True

        # Check if all rows have all None values
        for row in rows:
            if any(value is not None for value in row.values()):
                return False

        return True


    def _call(self, inputs: Dict[str, Any], run_manager=None) -> Dict[str, Any]:
        from ..utils.chain_logger import (
            AgentLogContext, new_query_id,
            log_agent_start, log_agent_step, log_chain_trace, _now, _sum_token_field,
        )
        from ..utils.memory import resolve_memory, MemoryContext, MEMORY_DISABLED

        # Variables declared before try so exception handler can reference them
        prompt = inputs.get("prompt")
        sql = inputs.get("sql", None)
        schema_name = inputs.get("schema", self._schema)
        ontology_name = inputs.get("ontology", self._ontology)
        concept_name = inputs.get("concept", self._concept)
        conversation_id = inputs.get("conversation_id") or self._conversation_id
        usage_metadata = {}
        rows = []
        reasoning_status = None
        identify_concept_reason = None
        generate_sql_reason = None
        _generate_sql_chain_duration_ms = 0

        # ---- memory resolution (once per top-level invocation) ----
        _chain_ctx = self._received_chain_context
        if _chain_ctx.get("memory") is None and self._enable_memory:
            _chain_ctx["memory"] = resolve_memory(
                llm=self._llm,
                conn_params=self._get_conn_params(),
                conversation_id=conversation_id,
                prompt=prompt or "",
                enable_memory=self._enable_memory,
                memory_window_size=self._memory_window_size,
                concept_names=self._concepts_list,
            )
        memory_ctx = _chain_ctx.get("memory")
        memory_ctx = memory_ctx if isinstance(memory_ctx, MemoryContext) else None

        # Resolve logging context: received from parent (delegated) or create standalone
        _log_ctx = self._received_log_ctx

        if _log_ctx is None and self._enable_logging:
            _query_id = new_query_id()
            _log_ctx = AgentLogContext(
                query_id=_query_id,
                agent_name=self._agent or "",
                url=build_server_url(self._url, config.thrift_host, config.thrift_port),
                token=self._token,
                chain_type="ExecuteTimbrQueryChain",
                start_time=_now(),
                prompt=prompt or "",
                enable_trace=self._enable_trace,
                is_delegated=False,
                conversation_id=conversation_id or _query_id,
            )
            log_agent_start(_log_ctx, ontology_name, schema_name)
        elif _log_ctx is not None:
            _log_ctx.retry_count = 0
            _log_ctx.no_results_retry_count = 0

        # Persist memory follow-up state
        if _log_ctx and memory_ctx and memory_ctx.is_follow_up:
            _log_ctx.is_follow_up = True
            _log_ctx.parent_query_id = memory_ctx.parent_message_id

        _chain_start = _now()
        try:
            is_sql_valid = True
            error = None
            identify_concept_reason = None
            generate_sql_reason = None

            if sql and self._should_validate_sql:
                if _log_ctx:
                    _log_ctx.current_step = "validating_sql"
                    log_agent_step(_log_ctx)
                is_sql_valid, error, sql = validate_sql(sql, self._get_conn_params())

            is_infered = False
            iteration = 0
            generated = []

            while not is_infered and iteration <= self._no_results_max_retries:
                conn_params = self._get_conn_params()
                if prompt is not None and not sql or not is_sql_valid:
                    # Show identifying_concept step on first iteration when concept is unknown
                    if _log_ctx:
                        if concept_name is None and iteration == 0:
                            _log_ctx.current_step = "identifying_concept"
                        else:
                            _log_ctx.current_step = "generating_sql"
                        log_agent_step(_log_ctx)

                    _gen_start = _now()
                    generate_res = self._generate_sql(prompt, sql, concept_name, schema_name, error, conn_params, memory_context=memory_ctx)
                    _generate_sql_chain_duration_ms += int((_now() - _gen_start).total_seconds() * 1000)
                    conn_params = generate_res.get("conn_params")
                    sql = generate_res.get("sql", "")
                    ontology_name = generate_res.get("ontology", ontology_name)
                    schema_name = generate_res.get("schema", schema_name)
                    concept_name = generate_res.get("concept", concept_name)
                    is_sql_valid = generate_res.get("is_sql_valid")
                    reasoning_status = generate_res.get("reasoning_status")
                    if not is_sql_valid and not self._should_validate_sql:
                        is_sql_valid = True

                    error = generate_res.get("error")
                    identify_concept_reason = generate_res.get("identify_concept_reason")
                    generate_sql_reason = generate_res.get("generate_sql_reason")
                    _gen_meta = generate_res.get("usage_metadata", {})
                    usage_metadata = self._summarize_usage_metadata(usage_metadata, _gen_meta)

                    if _log_ctx:
                        if concept_name:
                            _log_ctx.concept = concept_name
                        _log_ctx.current_step = "generating_sql"
                        log_agent_step(_log_ctx)

                is_sql_not_tried = not any(sql.lower().strip() == gen.lower().strip() for gen in generated)

                if _log_ctx:
                    _log_ctx.current_step = "executing_query"
                    log_agent_step(_log_ctx)

                rows = run_query(
                    sql,
                    conn_params,
                    llm_prompt=prompt,
                    use_query_limit=True,
                ) if is_sql_valid and is_sql_not_tried else []

                if iteration < self._no_results_max_retries:
                    # If no rows are returned and we should infer the result, we will try to re-generate the SQL query
                    if prompt is not None and self._retry_if_no_results and self._has_no_meaningful_results(rows, sql):
                        if is_sql_not_tried:
                            generated.append(sql)
                            # If the SQL is valid but no rows are returned, create an error message to be sent to the LLM
                            if is_sql_valid:
                                error = "The query returned no rows, or an aggregate matched nothing (zero/null). Please revise the SQL considering if the question was ambiguous (e.g., which ID or name to use), try use alternative columns in the WHERE clause part in a way that could match the user's intent, without adding new columns with new filters."
                                error += "\nConsider that these queries already generated and returned no results:\n" + "\n".join(generated)
                                is_sql_valid = False
                            if _log_ctx:
                                _log_ctx.no_results_retry_count += 1
                                _log_ctx.current_step = "retrying"
                                log_agent_step(_log_ctx)
                        else:
                            # Generated twice the same SQL, so we will stop the loop
                            is_infered = True
                    else:
                        is_infered = True
                iteration += 1

            final_error = error if not is_sql_valid else None

            _total_duration_ms = int((_now() - _chain_start).total_seconds() * 1000)
            _chain_ctx = self._received_chain_context
            _chain_ctx["duration"]["ExecuteTimbrQueryChain"] = _total_duration_ms
            if _generate_sql_chain_duration_ms:
                _chain_ctx["duration"]["GenerateTimbrSqlChain"] = _generate_sql_chain_duration_ms
            if identify_concept_reason:
                _chain_ctx["reasoning"]["identify_concept_reason"] = identify_concept_reason
            if generate_sql_reason:
                _chain_ctx["reasoning"]["generate_sql_reason"] = generate_sql_reason
            _chain_ctx["tokens"]["ExecuteTimbrQueryChain"] = {
                "total_tokens": _sum_token_field(usage_metadata, "total_tokens", "approximate"),
                "input_tokens":  _sum_token_field(usage_metadata, "input_tokens"),
                "output_tokens": _sum_token_field(usage_metadata, "output_tokens"),
            }

            result = {
                **inputs,
                "rows": rows,
                "sql": sql,
                "ontology": ontology_name,
                "schema": schema_name,
                "concept": concept_name,
                "error": final_error,
                "reasoning_status": reasoning_status,
                "identify_concept_reason": identify_concept_reason,
                "generate_sql_reason": generate_sql_reason,
                self.usage_metadata_key: usage_metadata,
                "conversation_id": conversation_id or (_log_ctx.query_id if _log_ctx else None),
            }

            if _log_ctx:
                log_chain_trace(
                    ctx=_log_ctx,
                    chain_type=_log_ctx.chain_type,
                    start_time=_chain_start,
                    status="failed" if final_error else "completed",
                    question=prompt,
                    ontology=ontology_name,
                    concept=concept_name,
                    schema=schema_name,
                    generated_sql=sql,
                    chain_output={"row_count": len(rows) if rows else 0},
                    rows_returned=len(rows) if rows else 0,
                    error=final_error,
                    reasoning_status=reasoning_status,
                    usage_metadata=usage_metadata,
                )
                
            return sanitize_results(self.output_keys, result)

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

