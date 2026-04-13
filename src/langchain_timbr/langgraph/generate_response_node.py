from typing import Optional, Union
from langchain_core.language_models.llms import LLM

from ..langchain import GenerateAnswerChain
from .. import config


class GenerateResponseNode:
    """
    Node that wraps GenerateAnswerChain functionality, which generates an answer based on a given prompt and rows of data.
    It uses the LLM to build a human-readable answer.

    This node connects to a Timbr server via the provided URL and token to generate contextual answers from query results using an LLM.
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
        :param llm: An LLM instance or a function that takes a prompt string and returns the LLM's response (optional, will use LlmWrapper with env variables if not provided)
        :param url: Timbr server url (optional, defaults to TIMBR_URL environment variable)
        :param token: Timbr password or token value (optional, defaults to TIMBR_TOKEN environment variable)
        :param ontology: Name of the ontology/knowledge graph (optional). Required when rows are not provided so the chain can fall back to executing a query.
        :param schema: Optional specific schema name to query (default is 'dtimbr').
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
        :param conn_params: Extra Timbr connection parameters sent with every request (e.g., 'x-api-impersonate-user').
        :param enable_trace: Whether to enable trace logging for this node's operations.
        :param enable_history: Whether to enable history logging (default is True).
        :param save_results: Whether to save results in history when enable_history is True (default is False).
        :param conversation_id: Optional conversation ID for tracking across multi-turn conversations.
        """
        self.chain = GenerateAnswerChain(
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
            db_is_case_sensitive=db_is_case_sensitive,
            graph_depth=graph_depth,
            enable_reasoning=enable_reasoning,
            reasoning_steps=reasoning_steps,
            note=note,
            agent=agent,
            verify_ssl=verify_ssl,
            is_jwt=is_jwt,
            jwt_tenant_id=jwt_tenant_id,
            conn_params=conn_params,
            debug=debug,
            enable_trace=enable_trace,
            enable_history=enable_history,
            save_results=save_results,
            conversation_id=conversation_id,
            **kwargs,
        )


    def run(self, state: dict) -> dict:
        sql = state.get("sql", "")
        rows = state.get("rows", "")
        prompt = state.get("prompt", "")
        conversation_id = state.get("conversation_id", None)

        return self.chain.invoke({"prompt": prompt, "rows": rows, "sql": sql, "conversation_id": conversation_id})


    def __call__(self, state: dict) -> dict:
        return self.run(state)
