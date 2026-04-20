from typing import Optional, Union
from langchain_core.language_models.llms import LLM
from langgraph.graph import StateGraph

from ..langchain.execute_timbr_query_chain import ExecuteTimbrQueryChain
from .. import config

class ExecuteSemanticQueryNode:
    """
    Node that wraps ExecuteTimbrQueryChain functionality.
    Expects the SQL (and optionally the concept) in the payload.
    Returns the query rows along with the SQL and concept.
    """
    def __init__(
        self,
        llm: Optional[LLM] = None,
        url: Optional[str] = None,
        token: Optional[str] = None,
        ontology: Optional[str] = None,
        schema: Optional[str] = None,
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
        :param enable_trace: Whether to enable trace logging for this node's operations.
        :param conversation_id: Optional conversation ID for tracking across multi-turn conversations.
        :param enable_memory: Whether to enable conversation memory (default from TIMBR_ENABLE_MEMORY env var).
        :param memory_window_size: Number of past conversation turns to consider (default from TIMBR_MEMORY_WINDOW_SIZE env var).
        """
        self.chain = ExecuteTimbrQueryChain(
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
            conversation_id=conversation_id,
            enable_memory=enable_memory,
            memory_window_size=memory_window_size,
            **kwargs,
        )


    def run(self, state: StateGraph) -> dict:
        try:
            prompt = state.messages[-1].get('content') if state.messages[-1] and 'content' in state.messages[-1] else None
        except Exception:
            prompt = state.get('prompt', None) if hasattr(state, 'get') else None

        _state_get = state.get if hasattr(state, 'get') else lambda k, d=None: d
        conversation_id = _state_get('conversation_id', None)
        chain_context = _state_get('chain_context', None)
        return self.chain.invoke({"prompt": prompt, "conversation_id": conversation_id, "chain_context": chain_context})


    def __call__(self, state: dict) -> dict:
        return self.run(state)

