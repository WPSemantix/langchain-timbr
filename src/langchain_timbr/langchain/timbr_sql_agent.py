from typing import Optional, Any, Union
from langchain.agents import AgentExecutor, BaseSingleActionAgent
from langchain.llms.base import LLM
from langchain.schema import AgentAction, AgentFinish

from ..utils.general import parse_list, to_boolean, to_integer
from .execute_timbr_query_chain import ExecuteTimbrQueryChain
from .generate_answer_chain import GenerateAnswerChain

class TimbrSqlAgent(BaseSingleActionAgent):
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
        should_validate_sql: Optional[bool] = True,
        retries: Optional[int] = 3,
        max_limit: Optional[int] = 500,
        retry_if_no_results: Optional[bool] = False,
        no_results_max_retries: Optional[int] = 2,
        generate_answer: Optional[bool] = False,
        note: Optional[str] = '',
        db_is_case_sensitive: Optional[bool] = False,
        graph_depth: Optional[int] = 1,
        verify_ssl: Optional[bool] = True,
        is_jwt: Optional[bool] = False,
        jwt_tenant_id: Optional[str] = None,
        conn_params: Optional[dict] = None,
        debug: Optional[bool] = False
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
        :param verify_ssl: Whether to verify SSL certificates (default is True).
        :param is_jwt: Whether to use JWT authentication (default is False).
        :param jwt_tenant_id: JWT tenant ID for multi-tenant environments (required when is_jwt=True).
        :param conn_params: Extra Timbr connection parameters sent with every request (e.g., 'x-api-impersonate-user').

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
            verify_ssl=to_boolean(verify_ssl),
            is_jwt=to_boolean(is_jwt),
            jwt_tenant_id=jwt_tenant_id,
            conn_params=conn_params,
            debug=to_boolean(debug),
        )
        self._generate_answer = to_boolean(generate_answer)
        
        # Pre-initialize the answer chain to avoid creating it on every request
        self._answer_chain = GenerateAnswerChain(
            llm=llm,
            url=url,
            token=token,
            verify_ssl=to_boolean(verify_ssl),
            is_jwt=to_boolean(is_jwt),
            jwt_tenant_id=jwt_tenant_id,
            conn_params=conn_params,
            debug=to_boolean(debug),
        ) if self._generate_answer else None


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


    @property
    def input_keys(self) -> list[str]:
        """Get the input keys required by the agent."""
        return ["input"]


    def plan(
        self, intermediate_steps: list[tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """Plan the next action based on the input."""
        user_input = kwargs.get("input", "")
        
        # Enhanced input validation
        if not user_input or not user_input.strip():
            return AgentFinish(
                return_values={
                    "error": "No input provided or input is empty",
                    "answer": None,
                    "rows": None,
                    "sql": None,
                    "schema": None,
                    "concept": None,
                    "usage_metadata": {},
                },
                log="Empty input received"
            )

        try:
            result = self._chain.invoke({ "prompt": user_input })
            answer = None
            usage_metadata = result.get(self._chain.usage_metadata_key, {})

            if self._answer_chain and not self._should_skip_answer_generation(result):
                answer_res = self._answer_chain.invoke({
                    "prompt": user_input,
                    "rows": result.get("rows"),
                    "sql": result.get("sql")
                })
                answer = answer_res.get("answer", "")
                usage_metadata.update(answer_res.get(self._answer_chain.usage_metadata_key, {}))

            return AgentFinish(
                return_values={
                    "answer": answer,
                    "rows": result.get("rows", []),
                    "sql": result.get("sql", ""),
                    "schema": result.get("schema", ""),
                    "concept": result.get("concept", ""),
                    "error": result.get("error", None),
                    "usage_metadata": usage_metadata,
                },
                log=f"Successfully executed query on concept: {result.get('concept', '')}"
            )
        except Exception as e:
            error_context = f"Error in TimbrSqlAgent.plan (sync): {str(e)}"
            return AgentFinish(
                return_values={
                    "error": str(e),
                    "answer": None,
                    "rows": None,
                    "sql": None,
                    "schema": None,
                    "concept": None,
                    "usage_metadata": {},
                },
                log=error_context
            )

    async def aplan(
        self, intermediate_steps: list[tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """Async version of the plan method."""
        user_input = kwargs.get("input", "")
        
        if not user_input or not user_input.strip():
            return AgentFinish(
                return_values={
                    "error": "No input provided or input is empty",
                    "answer": None,
                    "rows": None,
                    "sql": None,
                    "schema": None,
                    "concept": None,
                    "usage_metadata": {},
                },
                log="Empty or whitespace-only input received"
            )

        try:
            # Use async invoke if available, fallback to sync
            if hasattr(self._chain, 'ainvoke'):
                result = await self._chain.ainvoke({ "prompt": user_input })
            else:
                result = self._chain.invoke({ "prompt": user_input })
                
            answer = None
            usage_metadata = result.get("usage_metadata", {})
            
            if not self._should_skip_answer_generation(result) and self._answer_chain:
                # Use async invoke if available for answer chain too
                if hasattr(self._answer_chain, 'ainvoke'):
                    answer_res = await self._answer_chain.ainvoke({
                        "prompt": user_input,
                        "rows": result.get("rows"),
                        "sql": result.get("sql")
                    })
                else:
                    answer_res = self._answer_chain.invoke({
                        "prompt": user_input,
                        "rows": result.get("rows"),
                        "sql": result.get("sql")
                    })
                answer = answer_res.get("answer", "")
                usage_metadata.update(answer_res.get(self._answer_chain.usage_metadata_key, {}))

            return AgentFinish(
                return_values={
                    "answer": answer,
                    "rows": result.get("rows", []),
                    "sql": result.get("sql", ""),
                    "schema": result.get("schema", ""),
                    "concept": result.get("concept", ""),
                    "error": result.get("error", None),
                    "usage_metadata": usage_metadata,
                },
                log=f"Successfully executed query on concept: {result.get('concept', '')}"
            )
        except Exception as e:
            error_context = f"Error in TimbrSqlAgent.aplan (async): {str(e)}"
            return AgentFinish(
                return_values={
                    "error": str(e),
                    "answer": None,
                    "rows": None,
                    "sql": None,
                    "schema": None,
                    "concept": None,
                    "usage_metadata": {},
                },
                log=error_context
            )

    @property
    def return_values(self) -> list[str]:
        """Get the return values that this agent can produce."""
        return [
            "answer",
            "rows",
            "sql",
            "schema",
            "concept",
            "error",
            "usage_metadata",
        ]


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
    should_validate_sql: Optional[bool] = True,
    retries: Optional[int] = 3,
    max_limit: Optional[int] = 500,
    retry_if_no_results: Optional[bool] = False,
    no_results_max_retries: Optional[int] = 2,
    generate_answer: Optional[bool] = False,
    note: Optional[str] = '',
    db_is_case_sensitive: Optional[bool] = False,
    graph_depth: Optional[int] = 1,
    verify_ssl: Optional[bool] = True,
    is_jwt: Optional[bool] = False,
    jwt_tenant_id: Optional[str] = None,
    conn_params: Optional[dict] = None,
    debug: Optional[bool] = False
) -> AgentExecutor:
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
    :param verify_ssl: Whether to verify SSL certificates (default is True).
    :param is_jwt: Whether to use JWT authentication (default is False).
    :param jwt_tenant_id: JWT tenant ID for multi-tenant environments (required when is_jwt=True).
    :param conn_params: Extra Timbr connection parameters sent with every request (e.g., 'x-api-impersonate-user').

    Returns:
        AgentExecutor: Configured agent executor ready to use
    
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
    agent = TimbrSqlAgent(
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
        verify_ssl=verify_ssl,
        is_jwt=is_jwt,
        jwt_tenant_id=jwt_tenant_id,
        conn_params=conn_params,
        debug=debug,
    )
    
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=[],  # No tools needed as we're directly using the chain
        verbose=True
    )
