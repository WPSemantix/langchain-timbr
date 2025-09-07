from typing import Optional, Union
from langchain.llms.base import LLM
from langgraph.graph import StateGraph

from ..langchain.identify_concept_chain import IdentifyTimbrConceptChain


class IdentifyConceptNode:
    def __init__(
        self,
        llm: Optional[LLM] = None,
        url: Optional[str] = None,
        token: Optional[str] = None,
        ontology: Optional[str] = None,
        concepts_list: Optional[Union[list[str], str]] = None,
        views_list: Optional[Union[list[str], str]] = None,
        include_logic_concepts: Optional[bool] = False,
        include_tags: Optional[Union[list[str], str]] = None,
        should_validate: Optional[bool] = False,
        retries: Optional[int] = 3,
        note: Optional[str] = None,
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
        :param concepts_list: Optional specific concept options to query
        :param views_list: Optional specific view options to query
        :param include_logic_concepts: Optional boolean to include logic concepts (concepts without unique properties which only inherits from an upper level concept with filter logic) in the query.
        :param include_tags: Optional specific concepts & properties tag options to use in the query (Disabled by default. Use '*' to enable all tags or a string represents a list of tags divided by commas (e.g. 'tag1,tag2')
        :param should_validate: Whether to validate the identified concept before returning it
        :param retries: Number of retry attempts if the identified concept is invalid
        :param note: Optional additional note to extend our llm prompt
        :param verify_ssl: Whether to verify SSL certificates
        :param is_jwt: Whether to use JWT authentication (default: False)
        :param jwt_tenant_id: Tenant ID for JWT authentication when using multi-tenant setup
        :param conn_params: Extra Timbr connection parameters sent with every request (e.g., 'x-api-impersonate-user').
        """
        self.chain = IdentifyTimbrConceptChain(
            llm=llm,
            url=url,
            token=token,
            ontology=ontology,
            concepts_list=concepts_list,
            views_list=views_list,
            include_logic_concepts=include_logic_concepts,
            include_tags=include_tags,
            should_validate=should_validate,
            retries=retries,
            note=note,
            verify_ssl=verify_ssl,
            is_jwt=is_jwt,
            jwt_tenant_id=jwt_tenant_id,
            conn_params=conn_params,
            debug=debug,
            **kwargs,
        )


    def run(self, state: StateGraph) -> dict:
        try:
          prompt = state.messages[-1].content if state.messages[-1] else None
        except Exception:
          prompt = state.get('prompt', None)

        return self.chain.invoke({ "prompt": prompt })


    def __call__(self, state: dict) -> dict:
        return self.run(state)

