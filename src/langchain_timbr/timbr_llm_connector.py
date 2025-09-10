from typing import Optional, Any, Literal
from typing_extensions import TypedDict
from langchain.llms.base import LLM
from langgraph.graph import StateGraph, END

from .utils.general import to_boolean, to_integer
from .llm_wrapper.llm_wrapper import LlmWrapper
from .utils.timbr_utils import get_ontologies, get_concepts
from .langchain import IdentifyTimbrConceptChain, GenerateTimbrSqlChain, ValidateTimbrSqlChain, ExecuteTimbrQueryChain, create_timbr_sql_agent
from .langgraph import GenerateTimbrSqlNode, ValidateSemanticSqlNode, ExecuteSemanticQueryNode, GenerateResponseNode


from . import config

class TimbrLanggraphState(TypedDict):
    prompt: str
    sql: str
    concept: str
    rows: list
    response: str
    error: str
    is_sql_valid: bool
    usage_metadata: dict[str, Any]


class TimbrLlmConnector:
    def __init__(
        self,
        llm: LLM,
        url: Optional[str] = config.url,
        token: Optional[str] = config.token,
        ontology: Optional[str] = config.ontology,
        max_limit: Optional[int] = 500,
        verify_ssl: Optional[bool] = True,
        is_jwt: Optional[bool] = False,
        jwt_tenant_id: Optional[str] = None,
        conn_params: Optional[dict] = None,
    ):
        """
        :param url: Timbr server url
        :param token: Timbr password or token value
        :param ontology: The name of the ontology/knowledge graph
        :param llm: An LLM instance or a function that takes a prompt string and returns the LLM’s response
        :param max_limit: Maximum number of rows to return
        :param verify_ssl: Whether to verify SSL certificates (default is True).
        :param is_jwt: Whether to use JWT authentication (default is False).
        :param jwt_tenant_id: Tenant ID for JWT authentication (if applicable).
        :param conn_params: Extra Timbr connection parameters sent with every request (e.g., 'x-api-impersonate-user').

        ## Example
        ```
        timbr_llm_wrapper = LlmWrapper(
            llm_type=LlmTypes.OpenAI,
            model="gpt-4o"
            api_key=<openai_api_key>
        )

        llm_connector = TimbrLlmConnector(
            url=<url>,
            token=<token>,
            llm=timbr_llm_wrapper,
        )

        # Show ontology list at timbr instance from url connection
        ontologies = llm_connector.get_ontologies()

        # Find which concept & schema will be queried by the user input
        determine_concept_res = llm_connector.determine_concept(llm_input)
        query_concept, query_schema = determine_concept_res.get("concept"), determine_concept_res.get("schema")

        # Generate timbr SQL query from user input
        sql_query = llm_connector.generate_sql(llm_input).get("sql")

        # Run timbr SQL query
        results = llm_connector.run_timbr_query(sql_query).get("rows", [])
        
        # Parse & Run LLM question
        results = llm_connector.run_llm_query(llm_input).get("rows", [])
        ```
        """
        self.url = url
        self.token = token
        self.ontology = ontology
        self.max_limit = to_integer(max_limit)
        self.verify_ssl = to_boolean(verify_ssl)
        self.is_jwt = to_boolean(is_jwt)
        self.jwt_tenant_id = jwt_tenant_id
        self.conn_params = conn_params or {}
        
        if llm is not None:
            self._llm = llm
        elif config.llm_type is not None and config.llm_api_key is not None:
            llm_params = {}
            if config.llm_temperature is not None:
                llm_params["temperature"] = config.llm_temperature

            self._llm = LlmWrapper(
                llm_type=config.llm_type,
                api_key=config.llm_api_key,
                model=config.llm_model,
                **llm_params,
            )


    # TODO: Make this function a decorator and use in on relevant methods
    # def _is_ontology_set(self):
    #     return self.ontology != 'system_db'
    

    def _get_conn_params(self):
        return {
            "url": self.url,
            "token": self.token,
            "ontology": self.ontology,
            "verify_ssl": self.verify_ssl,
            "is_jwt": self.is_jwt,
            "jwt_tenant_id": self.jwt_tenant_id,
            **self.conn_params,
        }


    def get_ontologies(self) -> list[str]:
        return get_ontologies(conn_params=self._get_conn_params())


    def get_concepts(self) -> dict:
        """
        Get the list of concepts from the Timbr server.
        """
        return get_concepts(
            conn_params=self._get_conn_params(),
            concepts_list="*",
        )
    

    def get_views(self) -> dict:
        """
        Get the list of views from the Timbr server.
        """
        return get_concepts(
            conn_params=self._get_conn_params(),
            views_list="*",
        )


    def set_ontology(self, ontology: str):
        self.ontology = ontology


    def determine_concept(
        self,
        question: str,
        concepts_list: Optional[list] = None,
        views_list: Optional[list] = None,
        include_logic_concepts: Optional[bool] = False,
        include_tags: Optional[str] = None,
        should_validate: Optional[bool] = False,
        retries: Optional[int] = 3,
        note: Optional[str] = '',
        **chain_kwargs: Any,
    ) -> dict[str, Any]:
        determine_concept_chain = IdentifyTimbrConceptChain(
            **self._get_conn_params(),
            llm=self._llm,
            concepts_list=concepts_list,
            views_list=views_list,
            include_logic_concepts=include_logic_concepts,
            include_tags=include_tags,
            should_validate=should_validate,
            retries=retries,
            note=note,
            **chain_kwargs,
        )

        return determine_concept_chain.invoke({ "prompt": question })


    def generate_sql(
        self,
        question: str,
        concept_name: Optional[str] = None,
        schema: Optional[str] = None,
        concepts_list: Optional[list] = None,
        views_list: Optional[list] = None,
        include_logic_concepts: Optional[bool] = False,
        include_tags: Optional[str] = None,
        should_validate_sql: Optional[bool] = False,
        retries: Optional[int] = 3,
        note: Optional[str] = '',
        **chain_kwargs: Any,
    ) -> dict[str, Any]:
        generate_timbr_llm_chain = GenerateTimbrSqlChain(
            llm=self._llm,
            **self._get_conn_params(),
            schema=schema,
            concept=concept_name,
            concepts_list=concepts_list,
            views_list=views_list,
            include_logic_concepts=include_logic_concepts,
            include_tags=include_tags,
            should_validate_sql=should_validate_sql,
            retries=retries,
            max_limit=self.max_limit,
            note=note,
            **chain_kwargs,
        )

        return generate_timbr_llm_chain.invoke({ "prompt": question })


    def validate_sql(
        self,
        question: str,
        sql_query: str,
        retries: Optional[int] = 3,
        concepts_list: Optional[list] = None,
        views_list: Optional[list] = None,
        include_logic_concepts: Optional[bool] = False,
        include_tags: Optional[str] = None,
        note: Optional[str] = '',
        **chain_kwargs: Any,
    ) -> dict[str, Any]:
        validate_timbr_sql_chain = ValidateTimbrSqlChain(
            llm=self._llm,
            **self._get_conn_params(),
            retries=retries,
            concepts_list=concepts_list,
            views_list=views_list,
            include_logic_concepts=include_logic_concepts,
            include_tags=include_tags,
            max_limit=self.max_limit,
            note=note,
            **chain_kwargs,
        )
        return validate_timbr_sql_chain.invoke({ "sql": sql_query, "prompt": question })


    def run_timbr_query(
        self,
        sql_query: str,
        concepts_list: Optional[list] = None,
        views_list: Optional[list] = None,
        include_logic_concepts: Optional[bool] = False,
        include_tags: Optional[str] = None,
        should_validate_sql: Optional[bool] = True,
        retries: Optional[int] = 3,
        note: Optional[str] = '',
        **chain_kwargs: Any,
    ) -> dict[str, Any]:
        execute_timbr_query_chain = ExecuteTimbrQueryChain(
            llm=self._llm,
            **self._get_conn_params(),
            concepts_list=concepts_list,
            views_list=views_list,
            include_logic_concepts=include_logic_concepts,
            include_tags=include_tags,
            should_validate_sql=should_validate_sql,
            retries=retries,
            max_limit=self.max_limit,
            note=note,
            **chain_kwargs,
        )

        return execute_timbr_query_chain.invoke({ "sql": sql_query })


    def run_llm_query(
        self,
        question: str,
        concepts_list: Optional[list] = None,
        views_list: Optional[list] = None,
        include_logic_concepts: Optional[bool] = False,
        include_tags: Optional[str] = None,
        should_validate_sql: Optional[bool] = True,
        retries: Optional[int] = 3,
        note: Optional[str] = '',
        **agent_kwargs: Any,
    ) -> dict[str, Any]:
        agent = create_timbr_sql_agent(
            llm=self._llm,
            **self._get_conn_params(),
            concept=None,
            concepts_list=concepts_list,
            views_list=views_list,
            include_logic_concepts=include_logic_concepts,
            include_tags=include_tags,
            should_validate_sql=should_validate_sql,
            retries=retries,
            max_limit=self.max_limit,
            note=note,
            **agent_kwargs,
        )

        return agent.invoke(question)


    def run_llm_query_graph(
        self,
        question: str,
        concepts_list: Optional[list] = None,
        views_list: Optional[list] = None,
        include_logic_concepts: Optional[bool] = False,
        include_tags: Optional[str] = None,
        should_validate_sql: Optional[bool] = True,
        retries: Optional[int] = 3,
        note: Optional[str] = '',
        **nodes_kwargs: Any,
    ) -> dict[str, Any]:
        generate_sql_node = GenerateTimbrSqlNode(
            llm=self._llm,
            **self._get_conn_params(),
            concepts_list=concepts_list,
            views_list=views_list,
            include_logic_concepts=include_logic_concepts,
            include_tags=include_tags,
            max_limit=self.max_limit,
            note=note,
            **nodes_kwargs,
        )
        validate_sql_node = ValidateSemanticSqlNode(
            llm=self._llm,
            **self._get_conn_params(),
            retries=retries,
            concepts_list=concepts_list,
            views_list=views_list,
            include_logic_concepts=include_logic_concepts,
            include_tags=include_tags,
            max_limit=self.max_limit,
            note=note,
            **nodes_kwargs,
        )
        execute_sql_node = ExecuteSemanticQueryNode(
            llm=self._llm,
            **self._get_conn_params(),
            concepts_list=concepts_list,
            views_list=views_list,
            include_logic_concepts=include_logic_concepts,
            include_tags=include_tags,
            should_validate_sql=should_validate_sql,
            retries=retries,
            max_limit=self.max_limit,
            note=note,
            **nodes_kwargs,
        )
        generate_response_node = GenerateResponseNode()

        graph_builder = StateGraph(TimbrLanggraphState)

        graph_builder.add_node("generate_sql", generate_sql_node)
        graph_builder.add_node("validate_sql", validate_sql_node)
        graph_builder.add_node("execute_sql", execute_sql_node)
        graph_builder.add_node("generate_response", generate_response_node)

        graph_builder.add_edge("generate_sql", "validate_sql")
        
        def route_validation(state: dict) -> Literal["execute_sql", "end"]:
            # If validation is successful, proceed to execute the query.
            # Otherwise, stop the flow.
            if state.get("is_sql_valid"):
                return "execute_sql"
            else:
                return "end"
            
        graph_builder.add_conditional_edges(
            "validate_sql", 
            route_validation,
            {
                "execute_sql": "execute_sql",
                "end": END
            }
        )
        
        graph_builder.add_edge("execute_sql", "generate_response")
        graph_builder.set_entry_point("generate_sql")

        compiled_graph = graph_builder.compile()
        
        initial_state = {
            "prompt": question,
            "sql": "",
            "concept": "",
            "rows": [],
            "response": "",
            "error": "",
            "is_sql_valid": False,
            "usage_metadata": {}
        }
        
        result = compiled_graph.invoke(initial_state)
        return result
