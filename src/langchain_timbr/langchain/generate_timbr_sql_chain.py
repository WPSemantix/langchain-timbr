from typing import Optional, Union, Dict, Any
from langchain.chains.base import Chain
from langchain.llms.base import LLM

from ..utils.general import parse_list, to_boolean, to_integer
from ..utils.timbr_llm_utils import generate_sql

class GenerateTimbrSqlChain(Chain):
    """
    LangChain chain for generating SQL queries from natural language prompts using Timbr knowledge graphs.
    
    This chain takes user prompts and generates corresponding SQL queries that can be executed
    against Timbr ontology/knowledge graph databases. It uses an LLM to process prompts and
    connects to Timbr via URL and token for SQL generation.
    """
    
    def __init__(
        self,
        llm: LLM,
        url: str,
        token: str,
        ontology: str,
        schema: Optional[str] = 'dtimbr',
        concept: Optional[str] = None,
        concepts_list: Optional[Union[list[str], str]] = None,
        views_list: Optional[Union[list[str], str]] = None,
        include_tags: Optional[Union[list[str], str]] = None,
        include_logic_concepts: Optional[bool] = False,
        exclude_properties: Optional[Union[list[str], str]] = ['entity_id', 'entity_type', 'entity_label'],
        should_validate_sql: Optional[bool] = False,
        retries: Optional[int] = 3,
        max_limit: Optional[int] = 500,
        note: Optional[str] = '',
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
        :param llm: An LLM instance or a function that takes a prompt string and returns the LLM’s response
        :param url: Timbr server url
        :param token: Timbr password or token value
        :param ontology: The name of the ontology/knowledge graph
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
        :param verify_ssl: Whether to verify SSL certificates (default is True).
        :param is_jwt: Whether to use JWT authentication (default is False).
        :param jwt_tenant_id: JWT tenant ID for multi-tenant environments (required when is_jwt=True).
        :param conn_params: Extra Timbr connection parameters sent with every request (e.g., 'x-api-impersonate-user').
        :param kwargs: Additional arguments to pass to the base
        
        ## Example
        ```
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

        return generate_timbr_sql_chain.invoke({ "prompt": question }).get("sql", [])
        ```
        """
        super().__init__(**kwargs)
        self._llm = llm
        self._url = url
        self._token = token
        self._ontology = ontology
        self._schema = schema
        self._concept = concept
        self._concepts_list = parse_list(concepts_list)
        self._views_list = parse_list(views_list)
        self._include_tags = parse_list(include_tags)
        self._include_logic_concepts = to_boolean(include_logic_concepts)
        self._should_validate_sql = to_boolean(should_validate_sql)
        self._exclude_properties = parse_list(exclude_properties)
        self._retries = to_integer(retries)
        self._max_limit = to_integer(max_limit)
        self._note = note
        self._db_is_case_sensitive = to_boolean(db_is_case_sensitive)
        self._graph_depth = to_integer(graph_depth)
        self._verify_ssl = to_boolean(verify_ssl)
        self._is_jwt = to_boolean(is_jwt)
        self._jwt_tenant_id = jwt_tenant_id
        self._debug = to_boolean(debug)
        self._conn_params = conn_params or {}


    @property
    def usage_metadata_key(self) -> str:
        return "generate_sql_usage_metadata"


    @property
    def input_keys(self) -> list:
        return ["prompt"]

    @property
    def output_keys(self) -> list:
        return [
            "sql",
            "schema",
            "concept",
            "is_sql_valid",
            "error",
            self.usage_metadata_key,
        ]


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
            debug=self._debug,
        )
        
        sql = generate_res.get("sql", "")
        schema = generate_res.get("schema", self._schema)
        concept = generate_res.get("concept", self._concept)
        
        return {
            "sql": sql,
            "schema": schema,
            "concept": concept,
            "is_sql_valid": generate_res.get("is_sql_valid"),
            "error": generate_res.get("error"),
            self.usage_metadata_key: generate_res.get("usage_metadata"),
        }
