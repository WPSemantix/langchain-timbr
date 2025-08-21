from typing import Optional, Union, Dict, Any
from langchain.chains.base import Chain
from langchain.llms.base import LLM

from ..utils.general import parse_list, to_boolean, to_integer
from ..utils.timbr_llm_utils import determine_concept


class IdentifyTimbrConceptChain(Chain):
    """
    LangChain chain for identifying relevant concepts from user prompts using Timbr knowledge graphs.
    
    This chain analyzes natural language prompts to determine the most appropriate concept(s)
    within a Timbr ontology/knowledge graph that best matches the user's intent. It uses an LLM
    to process prompts and connects to Timbr via URL and token for concept identification.
    """
    
    def __init__(
        self,
        llm: LLM,
        url: str,
        token: str,
        ontology: str,
        concepts_list: Optional[Union[list[str], str]] = None,
        views_list: Optional[Union[list[str], str]] = None,
        include_logic_concepts: Optional[bool] = False,
        include_tags: Optional[Union[list[str], str]] = None,
        should_validate: Optional[bool] = False,
        retries: Optional[int] = 3,
        note: Optional[str] = '',
        verify_ssl: Optional[bool] = True,
        is_jwt: Optional[bool] = False,
        jwt_tenant_id: Optional[str] = None,
        conn_params: Optional[dict] = None,
        debug: Optional[bool] = False,
        **kwargs,
    ):
        """
        :param llm: An LLM instance or a function that takes a prompt string and returns the LLM’s response
        :param url: Timbr server url
        :param token: Timbr password or token value
        :param ontology: The name of the ontology/knowledge graph
        :param concepts_list: Optional specific concept options to query
        :param views_list: Optional specific view options to query
        :param include_logic_concepts: Optional boolean to include logic concepts (concepts without unique properties which only inherits from an upper level concept with filter logic) in the query.
        :param include_tags: Optional specific concepts & properties tag options to use in the query (Disabled by default. Use '*' to enable all tags or a string represents a list of tags divided by commas (e.g. 'tag1,tag2')
        :param should_validate: Whether to validate the identified concept before returning it
        :param retries: Number of retry attempts if the identified concept is invalid
        :param note: Optional additional note to extend our llm prompt
        :param verify_ssl: Whether to verify SSL certificates (default is True).
        :param is_jwt: Whether to use JWT authentication (default is False).
        :param jwt_tenant_id: JWT tenant ID for multi-tenant environments (required when is_jwt=True).
        :param conn_params: Extra Timbr connection parameters sent with every request (e.g., 'x-api-impersonate-user').
        :param kwargs: Additional arguments to pass to the base
        
        ## Example
        ```
        identify_timbr_concept_chain = IdentifyTimbrConceptChain(
            url=<url>,
            token=<token>,
            llm=<llm or timbr_llm_wrapper instance>,
            ontology=<ontology_name>,
            concepts_list=<concepts>,
            views_list=<views>,
            include_tags=<tags>,
            note=<note>,
        )

        return identify_timbr_concept_chain.invoke({ "prompt": question }).get("concept", None)
        ```
        """
        super().__init__(**kwargs)
        self._llm = llm
        self._url = url
        self._token = token
        self._ontology = ontology
        self._concepts_list = parse_list(concepts_list)
        self._views_list = parse_list(views_list)
        self._include_logic_concepts = to_boolean(include_logic_concepts)
        self._include_tags = parse_list(include_tags)
        self._should_validate = to_boolean(should_validate)
        self._retries = to_integer(retries)
        self._note = note
        self._verify_ssl = to_boolean(verify_ssl)
        self._is_jwt = to_boolean(is_jwt)
        self._jwt_tenant_id = jwt_tenant_id
        self._debug = to_boolean(debug)
        self._conn_params = conn_params or {}


    @property
    def usage_metadata_key(self) -> str:
        return "identify_concept_usage_metadata"


    @property
    def input_keys(self) -> list:
        return ["prompt"]


    @property
    def output_keys(self) -> list:
        return ["schema", "concept", "concept_metadata", self.usage_metadata_key]


    def _get_conn_params(self) -> dict:
        return {
            "url": self._url,
            "token": self._token,
            "ontology": self._ontology,
            "verify_ssl": self._verify_ssl,
            "is_jwt": self._is_jwt,
            "jwt_tenant_id": self._jwt_tenant_id,
            **self._conn_params,
        }


    def _call(self, inputs: Dict[str, Any], run_manager=None) -> Dict[str, str]:
        prompt = inputs["prompt"]
        res = determine_concept(
            question=prompt,
            llm=self._llm,
            conn_params=self._get_conn_params(),
            concepts_list=self._concepts_list,
            views_list=self._views_list,
            include_logic_concepts=self._include_logic_concepts,
            include_tags=self._include_tags,
            should_validate=self._should_validate,
            retries=self._retries,
            note=self._note,
            debug=self._debug,
        )

        usage_metadata = res.pop("usage_metadata", {})
        return {
            **res,
            self.usage_metadata_key: usage_metadata,
        }
