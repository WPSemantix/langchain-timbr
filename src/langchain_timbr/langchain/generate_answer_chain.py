from typing import Optional, Dict, Any
from langchain.chains.base import Chain
from langchain.llms.base import LLM

from ..utils.general import to_boolean, validate_timbr_connection_params
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
        verify_ssl: Optional[bool] = True,
        is_jwt: Optional[bool] = False,
        jwt_tenant_id: Optional[str] = None,
        conn_params: Optional[dict] = None,
        debug: Optional[bool] = False,
        **kwargs,
    ):
        """
        :param llm: An LLM instance or a function that takes a prompt string and returns the LLM’s response (optional, will use LlmWrapper with env variables if not provided)
        :param url: Timbr server url (optional, defaults to TIMBR_URL environment variable)
        :param token: Timbr password or token value (optional, defaults to TIMBR_TOKEN environment variable)
        :param verify_ssl: Whether to verify SSL certificates (default is True).
        :param is_jwt: Whether to use JWT authentication (default is False).
        :param jwt_tenant_id: JWT tenant ID for multi-tenant environments (required when is_jwt=True).
        :param conn_params: Extra Timbr connection parameters sent with every request (e.g., 'x-api-impersonate-user').
        
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


    @property
    def usage_metadata_key(self) -> str:
        return "generate_answer_usage_metadata"


    @property
    def input_keys(self) -> list:
        return ["prompt", "rows"]


    @property
    def output_keys(self) -> list:
        return ["answer", self.usage_metadata_key]

    def _get_conn_params(self) -> dict:
        return {
            "url": self._url,
            "token": self._token,
            # "ontology": self._ontology,
            "verify_ssl": self._verify_ssl,
            "is_jwt": self._is_jwt,
            "jwt_tenant_id": self._jwt_tenant_id,
            **self._conn_params,
        }
    

    def _call(self, inputs: Dict[str, Any], run_manager=None) -> Dict[str, str]:
        prompt = inputs["prompt"]
        rows = inputs["rows"]
        sql = inputs['sql'] if 'sql' in inputs else None

        res = answer_question(
            question=prompt,
            llm=self._llm,
            conn_params=self._get_conn_params(),
            results=rows,
            sql=sql,
            debug=self._debug,
        )

        return {
            "answer": res.get("answer", ""),
            self.usage_metadata_key: res.get("usage_metadata", {}),
        }
