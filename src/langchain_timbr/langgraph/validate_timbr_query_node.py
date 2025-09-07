from typing import Optional, Union
from langchain.llms.base import LLM
from langgraph.graph import StateGraph

from ..langchain.validate_timbr_sql_chain import ValidateTimbrSqlChain


class ValidateSemanticSqlNode:
    """
    Node that wraps ValidateTimbrSqlChain functionality.
    Expects an input payload with a "sql" or "prompt" key.
    Produces output with keys: "sql" and "is_sql_valid".
    """
    def __init__(
        self,
        llm: Optional[LLM] = None,
        url: Optional[str] = None,
        token: Optional[str] = None,
        ontology: Optional[str] = None,
        schema: Optional[str] = None,
        concept: Optional[str] = None,
        retries: Optional[int] = 3,
        concepts_list: Optional[Union[list[str], str]] = None,
        views_list: Optional[Union[list[str], str]] = None,
        include_logic_concepts: Optional[bool] = False,
        include_tags: Optional[Union[list[str], str]] = None,
        exclude_properties: Optional[Union[list[str], str]] = ['entity_id', 'entity_type', 'entity_label'],
        max_limit: Optional[int] = 500,
        note: Optional[str] = None,
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
        :param retries: The maximum number of retries to attempt
        :param concepts_list: Optional specific concept options to query
        :param views_list: Optional specific view options to query
        :param include_logic_concepts: Optional boolean to include logic concepts (concepts without unique properties which only inherits from an upper level concept with filter logic) in the query. 
        :param include_tags: Optional specific concepts & properties tag options to use in the query (Disabled by default. Use '*' to enable all tags or a string represents a list of tags divided by commas (e.g. 'tag1,tag2')
        :param exclude_properties: Optional specific properties to exclude from the query (entity_id, entity_type & entity_label by default).
        :param max_limit: Maximum number of rows to query
        :param note: Optional additional note to extend our llm prompt
        :param db_is_case_sensitive: Whether the database is case sensitive (default is False).
        :param graph_depth: Maximum number of relationship hops to traverse from the source concept during schema exploration (default is 1).
        :param verify_ssl: Whether to verify SSL certificates (default is True).
        :param is_jwt: Whether to use JWT authentication (default: False)
        :param jwt_tenant_id: Tenant ID for JWT authentication when using multi-tenant setup
        :param conn_params: Extra Timbr connection parameters sent with every request (e.g., 'x-api-impersonate-user').
        """
        self.chain = ValidateTimbrSqlChain(
            llm=llm,
            url=url,
            token=token,
            ontology=ontology,
            schema=schema,
            concept=concept,
            retries=retries,
            concepts_list=concepts_list,
            views_list=views_list,
            include_logic_concepts=include_logic_concepts,
            include_tags=include_tags,
            exclude_properties=exclude_properties,
            max_limit=max_limit,
            note=note,
            db_is_case_sensitive=db_is_case_sensitive,
            graph_depth=graph_depth,
            verify_ssl=verify_ssl,
            is_jwt=is_jwt,
            jwt_tenant_id=jwt_tenant_id,
            conn_params=conn_params,
            debug=debug,
            **kwargs,
        )


    def run(self, state: StateGraph) -> dict:
        try:
            sql = state.sql
            prompt = state.prompt
        except Exception:
            sql = state.get('sql', None)
            prompt = state.get('prompt', None)

        return self.chain.invoke({ "sql": sql, "prompt": prompt })


    def __call__(self, payload: dict) -> dict:
        return self.run(payload)
    