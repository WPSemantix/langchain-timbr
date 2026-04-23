from typing import Optional, Any, Union
from langchain_core.language_models.llms import LLM
from langchain_core.runnables import Runnable

try:
    from langsmith import trace as ls_trace
    _LANGSMITH_AVAILABLE = True
except ImportError:
    _LANGSMITH_AVAILABLE = False

from ..utils.general import parse_list, to_boolean, to_integer, sanitize_results
from .execute_timbr_query_chain import ExecuteTimbrQueryChain
from .generate_answer_chain import GenerateAnswerChain
from .. import config
from ..utils.timbr_utils import build_server_url

class TimbrSqlAgent(Runnable):
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
        generate_answer: Optional[bool] = False,
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
        enable_history: Optional[bool] = config.enable_history,
        save_results: Optional[bool] = config.history_save_results,
        conversation_id: Optional[str] = None,
        enable_memory: Optional[bool] = config.enable_memory,
        memory_window_size: Optional[int] = config.memory_window_size,
    ):
        """
        :param llm: An LLM instance or a function that takes a prompt string and returns the LLM's response (optional, will use LlmWrapper with env variables if not provided)
        :param url: Timbr server URL (optional, defaults to TIMBR_URL environment variable)
        :param token: Timbr authentication token (optional, defaults to TIMBR_TOKEN environment variable)
        :param ontology: Name of the ontology/knowledge graph (optional, defaults to ONTOLOGY/TIMBR_ONTOLOGY environment variable)
        :param schema: Optional specific schema name to query
        :param concept: Optional specific concept name to query
        :param concepts_list: Optional specific concept options to query
        :param views_list: Optional specific view options to query
        :param include_logic_concepts: Optional boolean to include logic concepts (concepts without unique properties which only inherits from an upper level concept with filter logic) in the query.
        :param include_tags: Optional specific concepts & properties tag options to use in the query (Disabled by default). Use '*' to enable all tags or a string represents a list of tags divided by commas (e.g. 'tag1,tag2')
        :param exclude_properties: Optional specific properties to exclude from the query (entity_id, entity_type & entity_label by default).
        :param should_validate_sql: Whether to validate the SQL before executing it
        :param retries: Number of retry attempts if the generated SQL is invalid
        :param max_limit: Maximum number of rows to return
        :retry_if_no_results: Whether to infer the result value from the SQL query. If the query won't return any rows, it will try to re-generate the SQL query then re-run it.
        :param no_results_max_retries: Number of retry attempts to infer the result value from the SQL query
        :param generate_answer: Whether to generate a natural language answer from the query results (default is False, which means the agent will return the SQL and rows only).
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
        :param enable_trace: Whether to enable detailed trace logging for the agent's operations (default is False).
        :param enable_history: Whether to enable query history tracking (default is False).
        :param save_results: Whether to save query results in history (default is False).
        :param conversation_id: Optional conversation ID to associate with all interactions of this agent instance (useful for tracking and logging in multi-turn conversations).
        ## Example
        ```
        # Using explicit parameters
        agent = TimbrSqlAgent(
            llm=<llm>,
            url=<url>,
            token=<token>,
            ontology=<ontology>,
            schema=<schema>,
            concept=<concept>,
            concepts_list=<concepts>,
            views_list=<views>,
            should_validate_sql=<should_validate_sql>,
            retries=<retries>,
            note=<note>,
        )

        # Using environment variables for timbr environment (TIMBR_URL, TIMBR_TOKEN, TIMBR_ONTOLOGY)
        agent = TimbrSqlAgent(
            llm=<llm>,
        )

        # Using environment variables for both timbr environment & llm (TIMBR_URL, TIMBR_TOKEN, TIMBR_ONTOLOGY, LLM_TYPE, LLM_API_KEY, etc.)
        agent = TimbrSqlAgent()
        ```
        """
        super().__init__()
        self._enable_logging = to_boolean(enable_trace)
        self._conversation_id = conversation_id
        self._generate_answer = to_boolean(generate_answer)
        self._enable_memory = to_boolean(enable_memory)
        self._memory_window_size = to_integer(memory_window_size)

        if self._generate_answer:
            self._chain = GenerateAnswerChain(
                llm=llm,
                url=url,
                token=token,
                ontology=ontology,
                schema=schema,
                concept=concept,
                concepts_list=parse_list(concepts_list),
                views_list=parse_list(views_list),
                include_logic_concepts=to_boolean(include_logic_concepts),
                include_tags=parse_list(include_tags),
                exclude_properties=parse_list(exclude_properties),
                should_validate_sql=to_boolean(should_validate_sql),
                retries=to_integer(retries),
                max_limit=to_integer(max_limit),
                retry_if_no_results=to_boolean(retry_if_no_results),
                no_results_max_retries=to_integer(no_results_max_retries),
                db_is_case_sensitive=to_boolean(db_is_case_sensitive),
                graph_depth=to_integer(graph_depth),
                enable_reasoning=to_boolean(enable_reasoning) if enable_reasoning is not None else None,
                reasoning_steps=to_integer(reasoning_steps) if reasoning_steps is not None else None,
                note=note,
                agent=agent,
                verify_ssl=to_boolean(verify_ssl),
                is_jwt=to_boolean(is_jwt),
                jwt_tenant_id=jwt_tenant_id,
                conn_params=conn_params,
                debug=to_boolean(debug),
                enable_trace=to_boolean(enable_trace),
                enable_history=to_boolean(enable_history),
                save_results=to_boolean(save_results),
                conversation_id=conversation_id,
                enable_memory=self._enable_memory,
                memory_window_size=self._memory_window_size,
            )
        else:
            self._chain = ExecuteTimbrQueryChain(
                llm=llm,
                url=url,
                token=token,
                ontology=ontology,
                schema=schema,
                concept=concept,
                concepts_list=parse_list(concepts_list),
                views_list=parse_list(views_list),
                include_logic_concepts=to_boolean(include_logic_concepts),
                include_tags=parse_list(include_tags),
                exclude_properties=parse_list(exclude_properties),
                should_validate_sql=to_boolean(should_validate_sql),
                retries=to_integer(retries),
                max_limit=to_integer(max_limit),
                retry_if_no_results=to_boolean(retry_if_no_results),
                no_results_max_retries=to_integer(no_results_max_retries),
                note=note,
                db_is_case_sensitive=to_boolean(db_is_case_sensitive),
                graph_depth=to_integer(graph_depth),
                agent=agent,
                verify_ssl=to_boolean(verify_ssl),
                is_jwt=to_boolean(is_jwt),
                jwt_tenant_id=jwt_tenant_id,
                conn_params=conn_params,
                enable_reasoning=to_boolean(enable_reasoning) if enable_reasoning is not None else None,
                reasoning_steps=to_integer(reasoning_steps) if reasoning_steps is not None else None,
                debug=to_boolean(debug),
                enable_trace=to_boolean(enable_trace),
                conversation_id=conversation_id,
                enable_memory=self._enable_memory,
                memory_window_size=self._memory_window_size,
            )


    @property
    def output_keys(self) -> list:
        return [
            "answer", "rows", "sql", "ontology", "schema", "concept",
            "error", "reasoning_status", "usage_metadata",
            "identify_concept_reason", "generate_sql_reason",
            "conversation_id", "chain_context",
        ]

    def _get_empty_input_response(self, conversation_id: Optional[str] = None) -> dict:
        """Return error response for empty/missing input."""
        return {
            "error": "No input provided or input is empty",
            "answer": None,
            "rows": None,
            "sql": None,
            "ontology": None,
            "schema": None,
            "concept": None,
            "reasoning_status": None,
            "identify_concept_reason": None,
            "generate_sql_reason": None,
            "usage_metadata": {},
            "conversation_id": conversation_id,
        }

    def _get_error_response(self, error_msg: str, conversation_id: Optional[str] = None) -> dict:
        """Return error response with exception message."""
        response = self._get_empty_input_response(conversation_id)
        response["error"] = error_msg
        return response

    def _setup_log_contexts(self, user_input: str, conversation_id: str) -> tuple:
        """Setup logging contexts if logging is enabled. Returns (_log_ctx, _delegated_ctx)."""
        from ..utils.chain_logger import AgentLogContext, new_query_id, _now, log_agent_start

        _log_ctx = None
        _delegated_ctx = None
        _query_id = new_query_id()

        if self._enable_logging:
            _log_ctx = AgentLogContext(
                query_id=_query_id,
                agent_name=self._chain._agent or "",
                url=build_server_url(self._chain._url, config.thrift_host, config.thrift_port),
                token=self._chain._token,
                chain_type="TimbrSqlAgent",
                start_time=_now(),
                prompt=user_input,
                enable_trace=self._chain._enable_trace,
                is_delegated=False,
                conversation_id=conversation_id,
            )
            log_agent_start(_log_ctx, self._chain._ontology, self._chain._schema)
            _delegated_ctx = AgentLogContext(
                query_id=_log_ctx.query_id,
                agent_name=_log_ctx.agent_name,
                url=_log_ctx.url,
                token=_log_ctx.token,
                chain_type=_log_ctx.chain_type,
                start_time=_log_ctx.start_time,
                prompt=_log_ctx.prompt,
                enable_trace=_log_ctx.enable_trace,
                is_delegated=True,
                conversation_id=conversation_id,
            )

        return _log_ctx, _delegated_ctx

    def _build_result(self, result: dict, conversation_id: str, log_ctx, delegated_ctx) -> dict:
        """Build the final result dictionary."""
        exec_meta = result.get("execute_timbr_usage_metadata", {})
        gen_meta = result.get("generate_answer_usage_metadata", {})
        usage_metadata = {**exec_meta, **gen_meta}

        if log_ctx and delegated_ctx:
            log_ctx.concept = delegated_ctx.concept
            log_ctx.retry_count = delegated_ctx.retry_count
            log_ctx.no_results_retry_count = delegated_ctx.no_results_retry_count

        return sanitize_results(self.output_keys, {
            "answer": result.get("answer"),
            "rows": result.get("rows", []),
            "sql": result.get("sql", ""),
            "ontology": result.get("ontology", ""),
            "schema": result.get("schema", ""),
            "concept": result.get("concept", ""),
            "error": result.get("error"),
            "reasoning_status": result.get("reasoning_status"),
            "usage_metadata": usage_metadata,
            "identify_concept_reason": result.get("identify_concept_reason"),
            "generate_sql_reason": result.get("generate_sql_reason"),
            "conversation_id": conversation_id,
            "chain_context": result.get("chain_context"),
        })


    def invoke(
        self, input: dict, config=None, **kwargs: Any
    ) -> dict:
        """Run the agent and return results."""
        if _LANGSMITH_AVAILABLE:
            with ls_trace(name="TimbrSqlAgent", run_type="chain", inputs={"input": input}) as rt:
                result = self._invoke_impl(input)
                rt.end(outputs=result)
                return result
        return self._invoke_impl(input)

    def _invoke_impl(self, input: dict) -> dict:
        user_input = input.get("input", "") if isinstance(input, dict) else input

        if not user_input or not user_input.strip():
            return self._get_empty_input_response()

        _conversation_id = (input.get("conversation_id") if isinstance(input, dict) else None) or self._conversation_id
        _chain_context = input.get("chain_context") if isinstance(input, dict) else None
        _log_ctx, _delegated_ctx = self._setup_log_contexts(user_input, _conversation_id)

        try:
            result = self._chain.invoke({"prompt": user_input, "conversation_id": _conversation_id, "chain_context": _chain_context}, log_ctx=_delegated_ctx)
            if result.get('conversation_id') != _conversation_id:
                _conversation_id = result.get('conversation_id')

            # Persist memory follow-up state to log context
            _result_chain_ctx = result.get("chain_context") or {}
            _mem = _result_chain_ctx.get("memory")
            if _log_ctx and _mem and hasattr(_mem, "is_follow_up") and _mem.is_follow_up:
                _log_ctx.is_follow_up = True
                _log_ctx.parent_query_id = _mem.parent_message_id

            return self._build_result(result, _conversation_id, _log_ctx, _delegated_ctx)
        except Exception as e:
            return self._get_error_response(str(e), _conversation_id)

    async def ainvoke(
        self, input: dict, config=None, **kwargs: Any
    ) -> dict:
        """Async version of invoke."""
        if _LANGSMITH_AVAILABLE:
            with ls_trace(name="TimbrSqlAgent", run_type="chain", inputs={"input": input}) as rt:
                result = await self._ainvoke_impl(input)
                rt.end(outputs=result)
                return result
        return await self._ainvoke_impl(input)

    async def _ainvoke_impl(self, input: dict) -> dict:
        user_input = input.get("input", "") if isinstance(input, dict) else input

        if not user_input or not user_input.strip():
            return self._get_empty_input_response()

        _conversation_id = (input.get("conversation_id") if isinstance(input, dict) else None) or self._conversation_id
        _chain_context = input.get("chain_context") if isinstance(input, dict) else None
        _log_ctx, _delegated_ctx = self._setup_log_contexts(user_input, _conversation_id)

        try:
            if hasattr(self._chain, 'ainvoke'):
                result = await self._chain.ainvoke({"prompt": user_input, "conversation_id": _conversation_id, "chain_context": _chain_context}, log_ctx=_delegated_ctx)
            else:
                result = self._chain.invoke({"prompt": user_input, "conversation_id": _conversation_id, "chain_context": _chain_context}, log_ctx=_delegated_ctx)

            if result.get('conversation_id') != _conversation_id:
                _conversation_id = result.get('conversation_id')

            # Persist memory follow-up state to log context
            _result_chain_ctx = result.get("chain_context") or {}
            _mem = _result_chain_ctx.get("memory")
            if _log_ctx and _mem and hasattr(_mem, "is_follow_up") and _mem.is_follow_up:
                _log_ctx.is_follow_up = True
                _log_ctx.parent_query_id = _mem.parent_message_id

            return self._build_result(result, _conversation_id, _log_ctx, _delegated_ctx)
        except Exception as e:
            return self._get_error_response(str(e), _conversation_id)


def create_timbr_sql_agent(
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
    generate_answer: Optional[bool] = False,
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
    enable_history: Optional[bool] = config.enable_history,
    save_results: Optional[bool] = config.history_save_results,
    conversation_id: Optional[str] = None,
    enable_memory: Optional[bool] = config.enable_memory,
    memory_window_size: Optional[int] = config.memory_window_size,
) -> TimbrSqlAgent:
    """
    Create and configure a Timbr agent with its executor.
    
    :param llm: An LLM instance or a function that takes a prompt string and returns the LLM's response (optional, will use LlmWrapper with env variables if not provided)
    :param url: Timbr server URL (optional, defaults to TIMBR_URL environment variable)
    :param token: Timbr authentication token (optional, defaults to TIMBR_TOKEN environment variable)
    :param ontology: Name of the ontology/knowledge graph (optional, defaults to ONTOLOGY/TIMBR_ONTOLOGY environment variable)
    :param schema: Optional specific schema name to query
    :param concept: Optional specific concept name to query
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
    :param generate_answer: Whether to generate an LLM answer based on the SQL results (default is False, which means the agent will return the SQL and rows only).
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
    :param enable_trace: Whether to enable detailed trace logging for the agent's operations (default is False).
    :param enable_history: Whether to enable query history tracking (default is False).
    :param save_results: Whether to save query results in history (default is False).
    :param conversation_id: Optional conversation ID to associate with all interactions of this agent instance (useful for tracking and logging in multi-turn conversations).

    Returns:
        TimbrSqlAgent: Configured agent ready to use
    
    ## Example
        ```
        # Using explicit parameters
        agent = create_timbr_sql_agent(
            llm=<llm>,
            url=<url>,
            token=<token>,
            ontology=<ontology>,
            schema=<schema>,
            concept=<concept>,
            concepts_list=<concepts>,
            views_list=<views>,
            include_tags=<tags>,
            exclude_properties=<properties>,
            should_validate_sql=<should_validate_sql>,
            retries=<retries>,
            note=<note>,
        )

        # Using environment variables for timbr environment (TIMBR_URL, TIMBR_TOKEN, TIMBR_ONTOLOGY)
        agent = create_timbr_sql_agent(
            llm=<llm>,
        )

        # Using environment variables for both timbr environment & llm (TIMBR_URL, TIMBR_TOKEN, TIMBR_ONTOLOGY, LLM_TYPE, LLM_API_KEY, etc.)
        agent = create_timbr_sql_agent()

        result = agent.invoke("What are the total sales for last month?")
        
        # Access the components of the result:
        rows = result["rows"]
        sql = result["sql"]
        schema = result["schema"]
        concept = result["concept"]
        error = result["error"]
        ```
    """
    timbr_agent = TimbrSqlAgent(
        llm=llm,
        url=url,
        token=token,
        ontology=ontology,
        schema=schema,
        concept=concept,
        concepts_list=concepts_list,
        views_list=views_list,
        include_logic_concepts=include_logic_concepts,
        include_tags=include_tags,
        exclude_properties=exclude_properties,
        should_validate_sql=should_validate_sql,
        retries=retries,
        max_limit=max_limit,
        retry_if_no_results=retry_if_no_results,
        no_results_max_retries=no_results_max_retries,
        generate_answer=generate_answer,
        note=note,
        db_is_case_sensitive=db_is_case_sensitive,
        graph_depth=graph_depth,
        agent=agent,
        verify_ssl=verify_ssl,
        is_jwt=is_jwt,
        jwt_tenant_id=jwt_tenant_id,
        conn_params=conn_params,
        enable_reasoning=enable_reasoning,
        reasoning_steps=reasoning_steps,
        debug=debug,
        enable_trace=enable_trace,
        enable_history=enable_history,
        save_results=save_results,
        conversation_id=conversation_id,
        enable_memory=enable_memory,
        memory_window_size=memory_window_size,
    )

    return timbr_agent
