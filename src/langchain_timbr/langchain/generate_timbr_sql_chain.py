import logging
from typing import Optional, Union, Dict, Any
from ..utils._base_chain import Chain
from langchain_core.language_models.llms import LLM
from langchain_timbr.utils.timbr_utils import get_timbr_agent_options, build_server_url
from ..utils.general import parse_list, to_boolean, to_integer, validate_timbr_connection_params, sanitize_results
from ..utils.timbr_llm_utils import generate_sql
from ..llm_wrapper.llm_wrapper import LlmWrapper
from .. import config

logger = logging.getLogger(__name__)

class GenerateTimbrSqlChain(Chain):
    """
    LangChain chain for generating SQL queries from natural language prompts using Timbr knowledge graphs.
    
    This chain takes user prompts and generates corresponding SQL queries that can be executed
    against Timbr ontology/knowledge graph databases. It uses an LLM to process prompts and
    connects to Timbr via URL and token for SQL generation.
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
        include_tags: Optional[Union[list[str], str]] = None,
        include_logic_concepts: Optional[bool] = False,
        exclude_properties: Optional[Union[list[str], str]] = ['entity_id', 'entity_type', 'entity_label'],
        should_validate_sql: Optional[bool] = config.should_validate_sql,
        retries: Optional[int] = 3,
        max_limit: Optional[int] = config.llm_default_limit,
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
        :param include_tags: Optional specific concepts & properties tag options to use in the query (Disabled by default. Use '*' to enable all tags or a string represents a list of tags divided by commas (e.g. 'tag1,tag2')
        :param include_logic_concepts: Optional boolean to include logic concepts (concepts without unique properties which only inherits from an upper level concept with filter logic) in the query.
        :param exclude_properties: Optional specific properties to exclude from the query (entity_id, entity_type & entity_label by default).
        :param should_validate_sql: Whether to validate the SQL before executing it
        :param retries: Number of retry attempts if the generated SQL is invalid
        :param max_limit: Maximum number of rows to query
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
        :param debug: Whether to enable debug mode for detailed logging
        :param enable_trace: Whether to enable trace logging for this chain's operations (default is False).
        :param conversation_id: Optional conversation ID to associate with this chain's execution for tracking and logging in multi-turn conversations.
        :param kwargs: Additional arguments to pass to the base
        
        ## Example
        ```
        # Using explicit parameters
        generate_timbr_sql_chain = GenerateTimbrSqlChain(
            url=<url>,
            token=<token>,
            llm=<llm or timbr_llm_wrapper instance>,
            ontology=<ontology_name>,
            schema=<schema_name>,
            concept=<concept_name>,
            concepts_list=<concepts>,
            views_list=<views>,
            include_tags=<tags>,
            note=<note>,
        )

        # Using environment variables for timbr environment (TIMBR_URL, TIMBR_TOKEN, TIMBR_ONTOLOGY)
        generate_timbr_sql_chain = GenerateTimbrSqlChain(
            llm=<llm or timbr_llm_wrapper instance>,
        )

        # Using environment variables for both timbr environment & llm (TIMBR_URL, TIMBR_TOKEN, TIMBR_ONTOLOGY, LLM_TYPE, LLM_API_KEY, etc.)
        generate_timbr_sql_chain = GenerateTimbrSqlChain()

        return generate_timbr_sql_chain.invoke({ "prompt": question }).get("sql", [])
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
            self._db_is_case_sensitive = to_boolean(db_is_case_sensitive)
            self._graph_depth = to_integer(graph_depth)
            self._enable_reasoning = to_boolean(enable_reasoning) if enable_reasoning is not None else config.enable_reasoning
            self._reasoning_steps = to_integer(reasoning_steps) if reasoning_steps is not None else config.reasoning_steps
            self._note = note
            self._enable_trace = to_boolean(enable_trace)

        self._enable_logging = self._enable_trace
        self._conversation_id = conversation_id


    @property
    def usage_metadata_key(self) -> str:
        return "generate_sql_usage_metadata"


    @property
    def input_keys(self) -> list:
        return ["prompt", "conversation_id"]

    @property
    def output_keys(self) -> list:
        base = [
            "sql",
            "ontology",
            "schema",
            "concept",
            "is_sql_valid",
            "error",
            "identify_concept_reason",
            "generate_sql_reason",
            "reasoning_status",
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


    def _call(self, inputs: Dict[str, Any], run_manager=None) -> Dict[str, str]:
        from ..utils.chain_logger import (
            AgentLogContext, new_query_id, _now,
            log_agent_start, log_agent_step, log_chain_trace, _sum_token_field,
        )

        prompt = inputs["prompt"]
        conversation_id = inputs.get("conversation_id") or self._conversation_id

        _log_ctx = self._received_log_ctx

        if _log_ctx is None and self._enable_logging:
            _query_id = new_query_id()
            _log_ctx = AgentLogContext(
                query_id=_query_id,
                agent_name=self._agent or "",
                url=build_server_url(config.thrift_host, config.thrift_port),
                token=self._token,
                chain_type="GenerateTimbrSqlChain",
                start_time=_now(),
                prompt=prompt,
                enable_trace=self._enable_trace,
                is_delegated=False,
                conversation_id=conversation_id or _query_id,
            )
            log_agent_start(_log_ctx, self._ontology, self._schema)

        if _log_ctx:
            _log_ctx.current_step = "identifying_concept" if self._concept is None else "generating_sql"
            log_agent_step(_log_ctx)

        _chain_start = _now()
        try:
            generate_res = generate_sql(
                question=prompt,
                llm=self._llm,
                conn_params=self._get_conn_params(),
                schema=self._schema,
                concept=self._concept,
                concepts_list=self._concepts_list,
                views_list=self._views_list,
                include_tags=self._include_tags,
                include_logic_concepts=self._include_logic_concepts,
                exclude_properties=self._exclude_properties,
                should_validate_sql=self._should_validate_sql,
                retries=self._retries,
                max_limit=self._max_limit,
                note=self._note,
                db_is_case_sensitive=self._db_is_case_sensitive,
                graph_depth=self._graph_depth,
                enable_reasoning=self._enable_reasoning,
                reasoning_steps=self._reasoning_steps,
                debug=self._debug,
            )
        except Exception as exc:
            error = str(exc)
            logger.error("GenerateTimbrSqlChain generate_sql failed: %s", error)
            if _log_ctx:
                log_chain_trace(
                    ctx=_log_ctx,
                    chain_type=_log_ctx.chain_type,
                    start_time=_chain_start,
                    status="failed",
                    question=prompt,
                    ontology=self._ontology,
                    concept=self._concept,
                    schema=self._schema,
                    generated_sql="",
                    chain_output={},
                    is_sql_valid=False,
                    error=error,
                    reasoning_status=None,
                    usage_metadata={},
                )
            return sanitize_results(
                self.output_keys,
                {
                    **inputs,
                    "sql": "",
                    "ontology": self._ontology,
                    "schema": self._schema,
                    "concept": self._concept,
                    "is_sql_valid": False,
                    "error": error,
                    "identify_concept_reason": None,
                    "generate_sql_reason": None,
                    "reasoning_status": None,
                    self.usage_metadata_key: {},
                    "conversation_id": conversation_id or (_log_ctx.query_id if _log_ctx else None),
                },
            )

        sql = generate_res.get("sql", "")
        ontology = generate_res.get("ontology", self._ontology)
        schema = generate_res.get("schema", self._schema)
        concept = generate_res.get("concept", self._concept)
        is_sql_valid = generate_res.get("is_sql_valid")
        error = generate_res.get("error")
        reasoning_status = generate_res.get("reasoning_status")
        usage_metadata = generate_res.get("usage_metadata") or {}

        _duration_ms = int((_now() - _chain_start).total_seconds() * 1000)
        _chain_ctx = self._received_chain_context
        _chain_ctx["duration"]["GenerateTimbrSqlChain"] = _duration_ms
        if generate_res.get("identify_concept_reason"):
            _chain_ctx["reasoning"]["identify_concept_reason"] = generate_res["identify_concept_reason"]
        if generate_res.get("generate_sql_reason"):
            _chain_ctx["reasoning"]["generate_sql_reason"] = generate_res["generate_sql_reason"]
        _chain_ctx["tokens"]["GenerateTimbrSqlChain"] = {
            "total_tokens": _sum_token_field(usage_metadata, "total_tokens", "approximate"),
            "input_tokens":  _sum_token_field(usage_metadata, "input_tokens"),
            "output_tokens": _sum_token_field(usage_metadata, "output_tokens"),
        }

        if _log_ctx:
            if concept:
                _log_ctx.concept = concept
            _log_ctx.current_step = "generating_sql"
            log_agent_step(_log_ctx)

        result = {
            **inputs,
            "sql": sql,
            "ontology": ontology,
            "schema": schema,
            "concept": concept,
            "is_sql_valid": is_sql_valid,
            "error": error,
            "identify_concept_reason": generate_res.get("identify_concept_reason"),
            "generate_sql_reason": generate_res.get("generate_sql_reason"),
            "reasoning_status": reasoning_status,
            self.usage_metadata_key: usage_metadata,
            "conversation_id": conversation_id or (_log_ctx.query_id if _log_ctx else None),
        }
        
        if _log_ctx:
            log_chain_trace(
                ctx=_log_ctx,
                chain_type=_log_ctx.chain_type,
                start_time=_chain_start,
                status="failed" if (not is_sql_valid and error) else "completed",
                question=prompt,
                ontology=ontology,
                concept=concept,
                schema=schema,
                generated_sql=sql,
                chain_output={"generate_sql_reason": generate_res.get("generate_sql_reason")},
                is_sql_valid=is_sql_valid,
                error=error if not is_sql_valid else None,
                reasoning_status=reasoning_status,
                usage_metadata=usage_metadata,
            )

        return sanitize_results(self.output_keys, result)
