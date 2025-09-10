import os
from .utils.general import to_boolean, to_integer, parse_list

# MUST HAVE VARIABLES
url = os.environ.get('TIMBR_URL')
token = os.environ.get('TIMBR_TOKEN')
ontology = os.environ.get('TIMBR_ONTOLOGY', os.environ.get('ONTOLOGY', 'system_db'))

# OPTIONAL VARIABLES
is_jwt = to_boolean(os.environ.get('IS_JWT', 'false'))
jwt_tenant_id = os.environ.get('JWT_TENANT_ID', None)

cache_timeout = to_integer(os.environ.get('CACHE_TIMEOUT', 120))
ignore_tags = parse_list(os.environ.get('IGNORE_TAGS', 'icon'))
ignore_tags_prefix = parse_list(os.environ.get('IGNORE_TAGS_PREFIX', 'mdx.,bli.'))

llm_type = os.environ.get('LLM_TYPE')
llm_model = os.environ.get('LLM_MODEL')
llm_api_key = os.environ.get('LLM_API_KEY')
llm_temperature = os.environ.get('LLM_TEMPERATURE', 0.0)
llm_additional_params = os.environ.get('LLM_ADDITIONAL_PARAMS', '')
llm_timeout = to_integer(os.environ.get('LLM_TIMEOUT', 60))  # Default 60 seconds timeout

# Optional for Azure OpenAI with Service Principal authentication
llm_tenant_id = os.environ.get('LLM_TENANT_ID', None)
llm_client_id = os.environ.get('LLM_CLIENT_ID', None)
llm_client_secret = os.environ.get('LLM_CLIENT_SECRET', None)
llm_endpoint = os.environ.get('LLM_ENDPOINT', None)
llm_api_version = os.environ.get('LLM_API_VERSION', None)
llm_scope = os.environ.get('LLM_SCOPE', "https://cognitiveservices.azure.com/.default")  # e.g. "api://<your-client-id>/.default"