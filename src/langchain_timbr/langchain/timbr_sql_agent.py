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
        enable_logging: Optional[bool] = False,
        chain_trace: Optional[bool] = False,
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
        self._enable_logging = to_boolean(enable_logging)
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
            enable_logging=to_boolean(enable_logging),
            chain_trace=to_boolean(chain_trace),
        )
        self._generate_answer = to_boolean(generate_answer)
        
        # Pre-initialize the answer chain to avoid creating it on every request
        self._answer_chain = GenerateAnswerChain(
            llm=llm,
            url=url,
            token=token,
            note=note,
            agent=agent,
            verify_ssl=to_boolean(verify_ssl),
            is_jwt=to_boolean(is_jwt),
            jwt_tenant_id=jwt_tenant_id,
            conn_params=conn_params,
            debug=to_boolean(debug),
        ) if self._generate_answer else None


    @property
    def output_keys(self) -> list:
        return [
            "answer", "rows", "sql", "ontology", "schema", "concept",
            "error", "reasoning_status", "usage_metadata",
            "identify_concept_reason", "generate_sql_reason",
        ]

    def _should_skip_answer_generation(self, result: dict) -> bool:
        """
        Determine if answer generation should be skipped based on result content.
        This can save LLM calls when there's an error or no meaningful data.
        """
        if not self._generate_answer:
            return True
            
        # Skip if there's an error
        if result.get("error"):
            return True
            
        # Skip if no rows returned
        rows = result.get("rows", [])
        if not rows or len(rows) == 0:
            return True
            
        return False


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
        from datetime import datetime as _dt
        from ..utils.chain_logger import (
            AgentLogContext, new_query_id,
            log_agent_start, log_agent_history, determine_status,
            get_llm_type, get_llm_model,
        )

        user_input = input.get("input", "") if isinstance(input, dict) else input

        # Enhanced input validation
        if not user_input or not user_input.strip():
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
            }

        # Build log context when logging is enabled
        _log_ctx = None
        _delegated_ctx = None
        if self._enable_logging:
            _log_ctx = AgentLogContext(
                query_id=new_query_id(),
                agent_name=self._chain._agent or "",
                url=self._chain._url,
                token=self._chain._token,
                chain_type="TimbrSqlAgent",
                start_time=_dt.now(),
                prompt=user_input,
                chain_trace_enabled=self._chain._chain_trace,
                is_delegated=False,
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
                chain_trace_enabled=_log_ctx.chain_trace_enabled,
                is_delegated=True,
            )

        try:
            result = self._chain.invoke({"prompt": user_input}, log_ctx=_delegated_ctx)
            answer = None
            _answer_chain_duration_ms = None
            usage_metadata = result.get(self._chain.usage_metadata_key, {})

            if self._answer_chain and not self._should_skip_answer_generation(result):
                _answer_start = _dt.now()
                answer_res = self._answer_chain.invoke({
                    "prompt": user_input,
                    "rows": result.get("rows"),
                    "sql": result.get("sql")
                }, log_ctx=_delegated_ctx)
                _answer_chain_duration_ms = int((_dt.now() - _answer_start).total_seconds() * 1000)
                answer = answer_res.get("answer", "")
                usage_metadata.update(answer_res.get(self._answer_chain.usage_metadata_key, {}))

            rows = result.get("rows", [])
            error = result.get("error", None)

            if _log_ctx:
                if _delegated_ctx:
                    _log_ctx.concept = _delegated_ctx.concept
                    _log_ctx.retry_count = _delegated_ctx.retry_count
                    _log_ctx.no_results_retry_count = _delegated_ctx.no_results_retry_count
                log_agent_history(
                    ctx=_log_ctx,
                    ontology=result.get("ontology"),
                    schema=result.get("schema"),
                    concept=result.get("concept"),
                    generated_sql=result.get("sql"),
                    rows_returned=len(rows) if rows else 0,
                    status=determine_status(rows, error),
                    failed_at_step=_delegated_ctx.current_step if (error and _delegated_ctx) else None,
                    error=error,
                    reasoning_status=result.get("reasoning_status"),
                    usage_metadata=usage_metadata,
                    answer_generated=bool(answer),
                    llm_type=get_llm_type(self._chain._llm),
                    llm_model=get_llm_model(self._chain._llm),
                    identify_concept_reason=result.get("identify_concept_reason"),
                    generate_sql_reason=result.get("generate_sql_reason"),
                    answer_chain_duration=_answer_chain_duration_ms,
                )

            return sanitize_results(self.output_keys, {
                "answer": answer,
                "rows": rows,
                "sql": result.get("sql", ""),
                "ontology": result.get("ontology", ""),
                "schema": result.get("schema", ""),
                "concept": result.get("concept", ""),
                "error": error,
                "reasoning_status": result.get("reasoning_status", None),
                "usage_metadata": usage_metadata,
                "identify_concept_reason": result.get("identify_concept_reason", None),
                "generate_sql_reason": result.get("generate_sql_reason", None),
            })
        except Exception as e:
            if _log_ctx:
                log_agent_history(
                    ctx=_log_ctx,
                    ontology=None,
                    schema=None,
                    concept=None,
                    generated_sql=None,
                    rows_returned=None,
                    status="timeout" if "timed out" in str(e).lower() else "failed",
                    failed_at_step=_delegated_ctx.current_step if _delegated_ctx else None,
                    error=str(e),
                    reasoning_status=None,
                    usage_metadata={},
                    answer_generated=False,
                    llm_type=get_llm_type(self._chain._llm),
                    llm_model=get_llm_model(self._chain._llm),
                )
            return sanitize_results(self.output_keys, {
                "error": str(e),
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
            })

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
        from datetime import datetime as _dt
        from ..utils.chain_logger import (
            AgentLogContext, new_query_id,
            log_agent_start, log_agent_history, determine_status,
            get_llm_type, get_llm_model,
        )

        user_input = input.get("input", "") if isinstance(input, dict) else input

        if not user_input or not user_input.strip():
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
            }

        _log_ctx = None
        _delegated_ctx = None
        if self._enable_logging:
            _log_ctx = AgentLogContext(
                query_id=new_query_id(),
                agent_name=self._chain._agent or "",
                url=self._chain._url,
                token=self._chain._token,
                chain_type="TimbrSqlAgent",
                start_time=_dt.now(),
                prompt=user_input,
                chain_trace_enabled=self._chain._chain_trace,
                is_delegated=False,
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
                chain_trace_enabled=_log_ctx.chain_trace_enabled,
                is_delegated=True,
            )

        try:
            # Use async invoke if available, fallback to sync
            if hasattr(self._chain, 'ainvoke'):
                result = await self._chain.ainvoke({"prompt": user_input}, log_ctx=_delegated_ctx)
            else:
                result = self._chain.invoke({"prompt": user_input}, log_ctx=_delegated_ctx)

            answer = None
            _answer_chain_duration_ms = None
            usage_metadata = result.get(self._chain.usage_metadata_key, {})

            if self._answer_chain and not self._should_skip_answer_generation(result):
                _answer_start = _dt.now()
                if hasattr(self._answer_chain, 'ainvoke'):
                    answer_res = await self._answer_chain.ainvoke({
                        "prompt": user_input,
                        "rows": result.get("rows"),
                        "sql": result.get("sql")
                    }, log_ctx=_delegated_ctx)
                else:
                    answer_res = self._answer_chain.invoke({
                        "prompt": user_input,
                        "rows": result.get("rows"),
                        "sql": result.get("sql")
                    }, log_ctx=_delegated_ctx)
                _answer_chain_duration_ms = int((_dt.now() - _answer_start).total_seconds() * 1000)
                answer = answer_res.get("answer", "")
                usage_metadata.update(answer_res.get(self._answer_chain.usage_metadata_key, {}))

            rows = result.get("rows", [])
            error = result.get("error", None)

            if _log_ctx:
                if _delegated_ctx:
                    _log_ctx.concept = _delegated_ctx.concept
                    _log_ctx.retry_count = _delegated_ctx.retry_count
                    _log_ctx.no_results_retry_count = _delegated_ctx.no_results_retry_count
                log_agent_history(
                    ctx=_log_ctx,
                    ontology=result.get("ontology"),
                    schema=result.get("schema"),
                    concept=result.get("concept"),
                    generated_sql=result.get("sql"),
                    rows_returned=len(rows) if rows else 0,
                    status=determine_status(rows, error),
                    failed_at_step=_delegated_ctx.current_step if (error and _delegated_ctx) else None,
                    error=error,
                    reasoning_status=result.get("reasoning_status"),
                    usage_metadata=usage_metadata,
                    answer_generated=bool(answer),
                    llm_type=get_llm_type(self._chain._llm),
                    llm_model=get_llm_model(self._chain._llm),
                    identify_concept_reason=result.get("identify_concept_reason"),
                    generate_sql_reason=result.get("generate_sql_reason"),
                    answer_chain_duration=_answer_chain_duration_ms,
                )

            return sanitize_results(self.output_keys, {
                "answer": answer,
                "rows": rows,
                "sql": result.get("sql", ""),
                "ontology": result.get("ontology", ""),
                "schema": result.get("schema", ""),
                "concept": result.get("concept", ""),
                "error": error,
                "reasoning_status": result.get("reasoning_status", None),
                "identify_concept_reason": result.get("identify_concept_reason", None),
                "generate_sql_reason": result.get("generate_sql_reason", None),
                "usage_metadata": usage_metadata,
            })
        except Exception as e:
            if _log_ctx:
                log_agent_history(
                    ctx=_log_ctx,
                    ontology=None,
                    schema=None,
                    concept=None,
                    generated_sql=None,
                    rows_returned=None,
                    status="timeout" if "timed out" in str(e).lower() else "failed",
                    failed_at_step=_delegated_ctx.current_step if _delegated_ctx else None,
                    error=str(e),
                    reasoning_status=None,
                    usage_metadata={},
                    answer_generated=False,
                    llm_type=get_llm_type(self._chain._llm),
                    llm_model=get_llm_model(self._chain._llm),
                )
            return sanitize_results(self.output_keys, {
                "error": str(e),
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
            })


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
    enable_logging: Optional[bool] = False,
    chain_trace: Optional[bool] = False,
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
        enable_logging=enable_logging,
        chain_trace=chain_trace,
    )

    return timbr_agent
