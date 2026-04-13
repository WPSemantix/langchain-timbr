from typing import Optional, Dict, Any, Union
from ..utils._base_chain import Chain
from langchain_core.language_models.llms import LLM

from langchain_timbr.utils.timbr_utils import get_timbr_agent_options, build_server_url

from ..utils.general import to_boolean, to_integer, parse_list, validate_timbr_connection_params, sanitize_results
from ..utils.timbr_llm_utils import answer_question
from ..llm_wrapper.llm_wrapper import LlmWrapper
from .. import config

class GenerateAnswerChain(Chain):
    """
    Chain that generates an answer based on a given prompt and rows of data.
    It uses the LLM to build a human-readable answer.
    
    This chain connects to a Timbr server via the provided URL and token to generate contextual answers from query results using an LLM.
    """
    def __init__(
        self,
        llm: Optional[LLM] = None,
        url: Optional[str] = None,
        token: Optional[str] = None,
        ontology: Optional[str] = None,
        schema: Optional[str] = 'dtimbr',
        concept: Optional[str] = None,
        concepts_list: Optional[Union[list, str]] = None,
        views_list: Optional[Union[list, str]] = None,
        include_logic_concepts: Optional[bool] = False,
        include_tags: Optional[Union[list, str]] = None,
        exclude_properties: Optional[Union[list, str]] = None,
        should_validate_sql: Optional[bool] = config.should_validate_sql,
        retries: Optional[int] = 3,
        max_limit: Optional[int] = config.llm_default_limit,
        retry_if_no_results: Optional[bool] = config.retry_if_no_results,
        no_results_max_retries: Optional[int] = 2,
        db_is_case_sensitive: Optional[bool] = False,
        graph_depth: Optional[int] = 1,
        enable_reasoning: Optional[bool] = None,
        reasoning_steps: Optional[int] = None,
        note: Optional[str] = '',
        agent: Optional[str] = None,
        verify_ssl: Optional[bool] = True,
        is_jwt: Optional[bool] = False,
        jwt_tenant_id: Optional[str] = None,
        conn_params: Optional[dict] = None,
        debug: Optional[bool] = False,
        enable_trace: Optional[bool] = config.enable_trace,
        enable_history: Optional[bool] = config.enable_history,
        save_results: Optional[bool] = config.history_save_results,
        conversation_id: Optional[str] = None,
        **kwargs,
    ):
        """
        :param llm: An LLM instance or a function that takes a prompt string and returns the LLM’s response (optional, will use LlmWrapper with env variables if not provided)
        :param url: Timbr server url (optional, defaults to TIMBR_URL environment variable)
        :param token: Timbr password or token value (optional, defaults to TIMBR_TOKEN environment variable)
        :param ontology: Name of the ontology/knowledge graph (optional). Required when rows are not provided so the chain can fall back to executing a query.
        :param schema: Optional specific schema name to query (default is ‘dtimbr’).
        :param concept: Optional specific concept name to query.
        :param concepts_list: Optional specific concept options to query.
        :param views_list: Optional specific view options to query.
        :param include_logic_concepts: Optional boolean to include logic concepts in the query.
        :param include_tags: Optional specific concepts & properties tag options to use in the query.
        :param exclude_properties: Optional specific properties to exclude from the query.
        :param should_validate_sql: Whether to validate the SQL before executing it.
        :param retries: Number of retry attempts if the generated SQL is invalid.
        :param max_limit: Maximum number of rows to return.
        :param retry_if_no_results: Whether to retry if the query returns no rows.
        :param no_results_max_retries: Number of retry attempts when query returns no rows.
        :param db_is_case_sensitive: Whether the database is case sensitive (default is False).
        :param graph_depth: Maximum number of relationship hops to traverse from the source concept (default is 1).
        :param enable_reasoning: Whether to enable reasoning during SQL generation.
        :param reasoning_steps: Number of reasoning steps to perform if reasoning is enabled.
        :param note: Optional additional note to extend our llm prompt
        :param agent: Optional Timbr agent name for options setup.
        :param verify_ssl: Whether to verify SSL certificates (default is True).
        :param is_jwt: Whether to use JWT authentication (default is False).
        :param jwt_tenant_id: JWT tenant ID for multi-tenant environments (required when is_jwt=True).
        :param conn_params: Extra Timbr connection parameters sent with every request (e.g., ‘x-api-impersonate-user’).
        :param enable_trace: Whether to enable trace (default is False).
        :param enable_history: Whether to enable history (default is True).
        :param save_results: Whether to save results in history when enable_history is True (default is False).
        :param conversation_id: Optional conversation ID to associate with this chain's execution for tracking and logging in multi-turn conversations.
        

        ## Example
        ```
        # Using explicit parameters
        generate_answer_chain = GenerateAnswerChain(
            llm=<llm or timbr_llm_wrapper instance>,
            url=<url>,
            token=<token>
        )

        # Using environment variables for timbr environment (TIMBR_URL, TIMBR_TOKEN)
        generate_answer_chain = GenerateAnswerChain(
            llm=<llm or timbr_llm_wrapper instance>
        )
        
        # Using environment variables for both timbr environment & llm (TIMBR_URL, TIMBR_TOKEN, LLM_TYPE, LLM_API_KEY, etc.)
        generate_answer_chain = GenerateAnswerChain()

        return generate_answer_chain.invoke({ "prompt": prompt, "rows": rows }).get("answer", [])
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

        self._agent = agent
        if self._agent:
            agent_options = get_timbr_agent_options(self._agent, conn_params=self._get_conn_params())

            self._note = agent_options.get("note") if "note" in agent_options else ''
            if note:
                self._note = ((self._note + '\n') if self._note else '') + note
            self._enable_trace = to_boolean(agent_options.get("enable_trace")) if "enable_trace" in agent_options else to_boolean(enable_trace)
            self._enable_history = to_boolean(agent_options.get("enable_history")) if "enable_history" in agent_options else to_boolean(enable_history)
            self._save_results = to_boolean(agent_options.get("history_save_results")) if "history_save_results" in agent_options else to_boolean(save_results)

        else:
            self._note = note
            self._enable_trace = to_boolean(enable_trace)
            self._enable_history = to_boolean(enable_history)
            self._save_results = to_boolean(save_results)

        self._enable_logging = self._enable_trace or self._enable_history
        self._conversation_id = conversation_id

        self._ontology = ontology
        self._schema = schema

        from .execute_timbr_query_chain import ExecuteTimbrQueryChain
        _exclude_properties = parse_list(exclude_properties) if exclude_properties is not None else ['entity_id', 'entity_type', 'entity_label']
        self._execute_chain = ExecuteTimbrQueryChain(
            llm=self._llm,
            url=self._url,
            token=self._token,
            ontology=ontology,
            schema=schema,
            concept=concept,
            concepts_list=parse_list(concepts_list),
            views_list=parse_list(views_list),
            include_logic_concepts=to_boolean(include_logic_concepts),
            include_tags=parse_list(include_tags),
            exclude_properties=_exclude_properties,
            should_validate_sql=to_boolean(should_validate_sql),
            retries=to_integer(retries),
            max_limit=to_integer(max_limit),
            retry_if_no_results=to_boolean(retry_if_no_results),
            no_results_max_retries=to_integer(no_results_max_retries),
            note=self._note,
            db_is_case_sensitive=to_boolean(db_is_case_sensitive),
            graph_depth=to_integer(graph_depth),
            agent=agent,
            verify_ssl=self._verify_ssl,
            is_jwt=self._is_jwt,
            jwt_tenant_id=self._jwt_tenant_id,
            conn_params=conn_params,
            enable_reasoning=to_boolean(enable_reasoning) if enable_reasoning is not None else None,
            reasoning_steps=to_integer(reasoning_steps) if reasoning_steps is not None else None,
            debug=self._debug,
            enable_trace=enable_trace,
            conversation_id=conversation_id,
        )


    @property
    def usage_metadata_key(self) -> str:
        return "generate_answer_usage_metadata"


    @property
    def input_keys(self) -> list:
        return ["prompt", "conversation_id"]


    @property
    def output_keys(self) -> list:
        base = [
            "answer", self.usage_metadata_key, "conversation_id",
            "rows", "sql", "ontology", "schema", "concept", "error",
            "reasoning_status", "identify_concept_reason", "generate_sql_reason",
            "execute_timbr_usage_metadata",
        ]
        return list(dict.fromkeys(self.input_keys + base))

    def _get_conn_params(self) -> dict:
        return {
            "url": self._url,
            "token": self._token,
            "ontology": config.ontology,
            "verify_ssl": self._verify_ssl,
            "is_jwt": self._is_jwt,
            "jwt_tenant_id": self._jwt_tenant_id,
            **self._conn_params,
        }
    

    def _merge_usage_metadata(self, current: dict, new: dict) -> dict:
        keys_to_sum = ['approximate', 'input_tokens', 'output_tokens', 'total_tokens']
        for outer_key, outer_value in new.items():
            if isinstance(outer_value, dict):
                if outer_key not in current:
                    current[outer_key] = {}
                for inner_key, inner_value in outer_value.items():
                    if inner_key in keys_to_sum:
                        current_val = current[outer_key].get(inner_key, 0)
                        if isinstance(inner_value, (int, float)) and isinstance(current_val, (int, float)):
                            current[outer_key][inner_key] = current_val + inner_value
                        else:
                            current[outer_key][inner_key] = inner_value
                    else:
                        current[outer_key][inner_key] = inner_value
            else:
                current[outer_key] = outer_value
        return current

    def _call(self, inputs: Dict[str, Any], run_manager=None) -> Dict[str, str]:
        from ..utils.chain_logger import (
            AgentLogContext, new_query_id, _now,
            log_agent_start, log_agent_step, log_agent_history, log_chain_trace,
            determine_status, get_llm_type, get_llm_model,
        )

        prompt = inputs["prompt"]
        rows = inputs.get("rows")
        sql = inputs.get("sql")
        conversation_id = inputs.get("conversation_id") or self._conversation_id

        execute_result = {}
        if rows is None:
            execute_result = self._execute_chain.invoke(
                {"prompt": prompt, "conversation_id": conversation_id},
                log_ctx=self._received_log_ctx,
            )
            rows = execute_result.get("rows")
            sql = execute_result.get("sql") or sql
            conversation_id = execute_result.get("conversation_id") or conversation_id

        _log_ctx = self._received_log_ctx

        if _log_ctx is None and self._enable_logging:
            _query_id = new_query_id()
            _log_ctx = AgentLogContext(
                query_id=_query_id,
                agent_name=self._agent or "",
                url=build_server_url(config.thrift_host, config.thrift_port),
                token=self._token,
                chain_type="GenerateAnswerChain",
                start_time=_now(),
                prompt=prompt,
                enable_trace=self._enable_trace,
                is_delegated=False,
                conversation_id=conversation_id or _query_id,
            )
            log_agent_start(_log_ctx, None, None)

        if _log_ctx:
            _log_ctx.current_step = "generating_answer"
            log_agent_step(_log_ctx)

        _chain_start = _now()
        res = answer_question(
            question=prompt,
            llm=self._llm,
            conn_params=self._get_conn_params(),
            results=rows,
            sql=sql,
            note=self._note,
            debug=self._debug,
        )

        answer = res.get("answer", "")
        usage_metadata = res.get("usage_metadata", {})

        if self._enable_history and _log_ctx:
            _has_results = bool(rows and any(any(v is not None for v in r.values()) for r in rows))

            _all_usage = {}
            for k, v in inputs.items():
                if k.endswith("_usage_metadata") and isinstance(v, dict):
                    _all_usage = self._merge_usage_metadata(_all_usage, v)
            _all_usage = self._merge_usage_metadata(_all_usage, usage_metadata)

            _error = inputs.get("error") or execute_result.get("error")
            log_agent_history(
                ctx=_log_ctx,
                ontology=inputs.get("ontology") or execute_result.get("ontology"),
                schema=inputs.get("schema") or execute_result.get("schema"),
                concept=inputs.get("concept") or execute_result.get("concept") or (_log_ctx.concept if _log_ctx else None),
                generated_sql=inputs.get("sql") or sql,
                rows_returned=len(rows) if rows is not None else None,
                status=determine_status(rows, _error),
                failed_at_step=None,
                error=_error,
                reasoning_status=inputs.get("reasoning_status") or execute_result.get("reasoning_status"),
                usage_metadata=_all_usage,
                answer_generated=bool(answer),
                llm_type=get_llm_type(self._llm),
                llm_model=get_llm_model(self._llm),
                identify_concept_reason=inputs.get("identify_concept_reason") or execute_result.get("identify_concept_reason"),
                generate_sql_reason=inputs.get("generate_sql_reason") or execute_result.get("generate_sql_reason"),
                answer=answer or None,
                has_results=_has_results,
                results=rows,
            )

        result = {
            **execute_result,
            **inputs,
            "rows": rows,
            "sql": sql,
            "answer": answer,
            self.usage_metadata_key: res.get("usage_metadata", {}),
            "conversation_id": conversation_id or (_log_ctx.query_id if _log_ctx else None),
        }

        if _log_ctx:
            log_chain_trace(
                ctx=_log_ctx,
                chain_type=_log_ctx.chain_type,
                start_time=_chain_start,
                status="completed",
                question=prompt,
                chain_output={"answer": answer},
                usage_metadata=usage_metadata,
            )
            
        return sanitize_results(self.output_keys, result)
