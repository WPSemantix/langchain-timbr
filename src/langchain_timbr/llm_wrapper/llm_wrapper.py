from enum import Enum
from typing import Optional
from langchain.llms.base import LLM
from pydantic import Field

from .timbr_llm_wrapper import TimbrLlmWrapper
from ..utils.general import is_llm_type, is_support_temperature, get_supported_models, parse_additional_params, pop_param_value
from .. import config

class LlmTypes(Enum):
  OpenAI = 'openai-chat'
  Anthropic = 'anthropic-chat'
  Google = 'chat-google-generative-ai'
  AzureOpenAI = 'azure-openai-chat'
  Snowflake = 'snowflake-cortex'
  Databricks = 'chat-databricks'
  VertexAI = 'chat-vertexai'
  Timbr = 'timbr'


class LlmWrapper(LLM):
  """
  LlmWrapper is a unified interface for connecting to various Large Language Model (LLM) providers
  (OpenAI, Anthropic, Google, Azure OpenAI, Snowflake Cortex, Databricks, etc.) using LangChain. It abstracts
  the initialization and connection logic for each provider, allowing you to switch between them
  """
  client: Optional[LLM] = Field(default=None, exclude=True)

  def __init__(
      self,
      llm_type: Optional[str] = None,
      api_key: Optional[str] = None,
      model: Optional[str] = None,
      **llm_params,
  ):
      """
      :param llm_type (str, optional): The type of LLM provider (e.g., 'openai-chat', 'anthropic-chat').
                                       If not provided, will try to get from LLM_TYPE environment variable.
      :param api_key (str, optional): The API key for authenticating with the LLM provider. 
                                     If not provided, will try to get from LLM_API_KEY environment variable.
      :param model (str, optional): The model name or deployment to use. If not provided, will try to get from LLM_MODEL environment variable.
      :param **llm_params: Additional parameters for the LLM (e.g., temperature, endpoint, etc.).
      """
      super().__init__()
      
      selected_llm_type = llm_type or config.llm_type
      selected_api_key = api_key or config.llm_api_key or config.llm_client_secret
      selected_model = model or config.llm_model
      selected_additional_params = llm_params.pop('additional_params', None)

      # Parse additional parameters from init params or config and merge with provided params
      default_additional_params = parse_additional_params(selected_additional_params or config.llm_additional_params or {})
      additional_llm_params = {**default_additional_params, **llm_params}
      
      # Validation: Ensure we have the required parameters
      if not selected_llm_type:
          raise ValueError("llm_type must be provided either as parameter or in config (LLM_TYPE environment variable)")
            
      self.client = self._connect_to_llm(
        selected_llm_type,
        selected_api_key,
        selected_model,
        **additional_llm_params,
      )


  @property
  def _llm_type(self):
    return self.client._llm_type


  def _add_temperature(self, llm_type, llm_model, **llm_params):
    """
    Add temperature to the LLM parameters if the LLM model supports it.
    """
    if "temperature" not in llm_params:
      if config.llm_temperature is not None and is_support_temperature(llm_type, llm_model):
        llm_params["temperature"] = config.llm_temperature
    return llm_params

  def _try_build_vertexai_credentials(self,params, api_key):
    from google.oauth2 import service_account
    from google.auth import default

    # Try multiple authentication methods in order of preference
    creds = None
    scope = pop_param_value(params, ['vertex_scope', 'llm_scope', 'scope'], default=config.llm_scope)
    scopes = [scope] if scope else ["https://www.googleapis.com/auth/cloud-platform"]
    
    # Method 1: Service Account File (json_path)
    json_path = pop_param_value(params, ['azure_json_path', 'llm_json_path', 'json_path'])
    if json_path:
      try:
        creds = service_account.Credentials.from_service_account_file(
          json_path,
          scopes=scopes,
        )
      except Exception as e:
        raise ValueError(f"Failed to load service account from file '{json_path}': {e}")
    
    # Method 2: Service Account Info (as dictionary)
    if not creds:
      service_account_info = pop_param_value(params, ['service_account_info', 'vertex_service_account_info'])
      if service_account_info:
        try:
          creds = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=scopes,
          )
        except Exception as e:
          raise ValueError(f"Failed to load service account from info: {e}")
    
    # Method 3: Service Account Email + Private Key
    if not creds:
      service_account_email = pop_param_value(params, ['service_account_email', 'vertex_email', 'service_account'])
      private_key = pop_param_value(params, ['private_key', 'vertex_private_key']) or api_key
      
      if service_account_email and private_key:
        try:
          service_account_info = {
            "type": "service_account",
            "client_email": service_account_email,
            "private_key": private_key,
            "token_uri": "https://oauth2.googleapis.com/token",
          }
          creds = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=scopes,
          )
        except Exception as e:
          raise ValueError(f"Failed to create service account from email and private key: {e}")
    
    # Method 4: Default Google Cloud Credentials (fallback)
    if not creds:
      try:
        creds, _ = default(scopes=scopes)
      except Exception as e:
        raise ValueError(
          "VertexAI authentication failed. Please provide one of:\n"
          "1. 'json_path' - path to service account JSON file\n"
          "2. 'service_account_info' - service account info as dictionary\n"
          "3. 'service_account_email' + 'private_key' - service account credentials\n"
          "4. Set up default Google Cloud credentials (gcloud auth application-default login)\n"
          f"Error: {e}"
        )

    return creds  

  def _connect_to_llm(self, llm_type, api_key = None, model = None, **llm_params):
    if is_llm_type(llm_type, LlmTypes.OpenAI):
      from langchain_openai import ChatOpenAI as OpenAI
      llm_model = model or "gpt-4o-2024-11-20"
      params = self._add_temperature(LlmTypes.OpenAI.name, llm_model, **llm_params)
      return OpenAI(
        openai_api_key=api_key,
        model_name=llm_model,
        **params,
      )
    elif is_llm_type(llm_type, LlmTypes.Anthropic):
      from langchain_anthropic import ChatAnthropic as Claude
      llm_model = model or "claude-3-5-sonnet-20241022"
      params = self._add_temperature(LlmTypes.Anthropic.name, llm_model, **llm_params)
      return Claude(
        anthropic_api_key=api_key,
        model=llm_model,
        **params,
      )
    elif is_llm_type(llm_type, LlmTypes.Google):
      from langchain_google_genai import ChatGoogleGenerativeAI
      llm_model = model or "gemini-2.0-flash-exp"
      params = self._add_temperature(LlmTypes.Google.name, llm_model, **llm_params)
      return ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model=llm_model,
        **params,
      )
    elif is_llm_type(llm_type, LlmTypes.Timbr):
      return TimbrLlmWrapper(
        api_key=api_key,
        **params,
      )
    elif is_llm_type(llm_type, LlmTypes.Snowflake):
      from langchain_community.chat_models import ChatSnowflakeCortex
      llm_model = model or "openai-gpt-4.1"
      params = self._add_temperature(LlmTypes.Snowflake.name, llm_model, **llm_params)
      snowflake_password = params.pop('snowflake_api_key', params.pop('snowflake_password', api_key))
      
      return ChatSnowflakeCortex(
        model=llm_model,
        snowflake_password=snowflake_password,
        **params,
      )
    elif is_llm_type(llm_type, LlmTypes.AzureOpenAI):
      from langchain_openai import AzureChatOpenAI
      llm_model = model or "gpt-4o-2024-11-20"
      params = self._add_temperature(LlmTypes.AzureOpenAI.name, llm_model, **llm_params)

      azure_endpoint = pop_param_value(params, ['azure_endpoint', 'llm_endpoint'], default=config.llm_endpoint)
      azure_api_version = pop_param_value(params, ['azure_api_version', 'llm_api_version'], default=config.llm_api_version)

      azure_client_id = pop_param_value(params, ['azure_client_id', 'llm_client_id'], default=config.llm_client_id)
      azure_tenant_id = pop_param_value(params, ['azure_tenant_id', 'llm_tenant_id'], default=config.llm_tenant_id)
      if azure_tenant_id and azure_client_id:
        from azure.identity import ClientSecretCredential, get_bearer_token_provider
        azure_client_secret = pop_param_value(params, ['azure_client_secret', 'llm_client_secret'], default=api_key)
        scope = pop_param_value(params, ['azure_scope', 'llm_scope', 'scope'], default=config.llm_scope)
        credential = ClientSecretCredential(
          tenant_id=azure_tenant_id,
          client_id=azure_client_id,
          client_secret=azure_client_secret
        )
        token_provider = get_bearer_token_provider(credential, scope)
        params['azure_ad_token_provider'] = token_provider
      else:
        params['api_key'] = api_key

      if 'openai_api_version' not in params or not params['openai_api_version']:
        params['openai_api_version'] = azure_api_version

      if 'azure_endpoint' not in params or not params['azure_endpoint']:
        params['azure_endpoint'] = azure_endpoint

      return AzureChatOpenAI(
        azure_deployment=llm_model,
        **params,
      )
    elif is_llm_type(llm_type, LlmTypes.Databricks):
      from databricks.sdk import WorkspaceClient
      from databricks_langchain import ChatDatabricks
      llm_model = model or "databricks-claude-sonnet-4"
      params = self._add_temperature(LlmTypes.Databricks.name, llm_model, **llm_params)

      host = params.pop('databricks_host', params.pop('host', None))
      w = WorkspaceClient(host=host, token=api_key)
      return ChatDatabricks(
        endpoint=llm_model,
        workspace_client=w,  # Using authenticated client
        **params,
      )    
    elif is_llm_type(llm_type, LlmTypes.VertexAI):
      from langchain_google_vertexai import ChatVertexAI
      llm_model = model or "gemini-2.5-flash-lite"
      params = self._add_temperature(LlmTypes.VertexAI.name, llm_model, **llm_params)

      project = pop_param_value(params, ['vertex_project', 'llm_project', 'project'])
      if project:
        params['project'] = project

      creds = self._try_build_vertexai_credentials(params, api_key)
      return ChatVertexAI(
        model_name=llm_model,
        credentials=creds,
        **params,
      )
    else:
      raise ValueError(f"Unsupported LLM type: {llm_type}")


  def get_model_list(self) -> list[str]:
    """Return the list of available models for the LLM."""
    models = []
    llm_type_name = None

    try:
      if is_llm_type(self._llm_type, LlmTypes.OpenAI):
        from openai import OpenAI
        client = OpenAI(api_key=self.client.openai_api_key._secret_value)
        models = [model.id for model in client.models.list()]
      elif is_llm_type(self._llm_type, LlmTypes.Anthropic):
        import anthropic
        client = anthropic.Anthropic(api_key=self.client.anthropic_api_key._secret_value)
        models = [model.id for model in client.models.list()]
      elif is_llm_type(self._llm_type, LlmTypes.Google):
        import google.generativeai as genai
        genai.configure(api_key=self.client.google_api_key._secret_value)
        models = [m.name.replace('models/', '') for m in genai.list_models()]
      elif is_llm_type(self._llm_type, LlmTypes.AzureOpenAI):
        from openai import AzureOpenAI
        # Get Azure-specific attributes from the client
        api_key = None
        azure_ad_token_provider = None
        azure_endpoint = getattr(self.client, 'azure_endpoint', config.llm_endpoint)
        api_version = getattr(self.client, 'openai_api_version', config.llm_api_version)

        params = {
          "azure_endpoint": azure_endpoint,
          "api_version": api_version,
        }

        api_key = getattr(self.client.openai_api_key, '_secret_value', None)
        if api_key:
          params['api_key'] = api_key
        else:
          azure_ad_token_provider = getattr(self.client, 'azure_ad_token_provider', None)
          if azure_ad_token_provider:
            params['azure_ad_token_provider'] = azure_ad_token_provider
          else:
            raise ValueError("Azure OpenAI requires either an API key or an Azure AD token provider for authentication.")

        if azure_endpoint and api_version and (api_key or azure_ad_token_provider):
          client = AzureOpenAI(**params)
          # For Azure, get the deployments instead of models
          try:
            models = [model.id for model in client.models.list()]
          except Exception:
            # If listing models fails, provide some common deployment names
            models = ["gpt-4o", "Other (Custom)"]
      elif is_llm_type(self._llm_type, LlmTypes.Snowflake):
        # Snowflake Cortex available models
        models = [
          "openai-gpt-4.1",
          "mistral-large2",
          "llama3.1-70b",
          "llama3.1-405b"
        ]
      elif is_llm_type(self._llm_type, LlmTypes.Databricks):
        w = getattr(self.client, 'workspace_client', None)
        if w:
          models = [ep.name for ep in w.serving_endpoints.list()]

      # elif self._is_llm_type(self._llm_type, LlmTypes.Timbr):
      elif is_llm_type(self._llm_type, LlmTypes.VertexAI):
        from google import genai
        if self.client.credentials:
          client = genai.Client(credentials=self.client.credentials, vertexai=True, project=self.client.project, location=self.client.location)
          models = [m.name.split('/')[-1] for m in client.models.list()]

    except Exception:
      # If model list fetching throws an exception, return default value using get_supported_models
      if hasattr(self, '_llm_type'):
        # Try to extract the LLM type name from the _llm_type
        for llm_enum in LlmTypes:
          if is_llm_type(self._llm_type, llm_enum):
            llm_type_name = llm_enum.name
            break

    if len(models) == 0 and llm_type_name:
      models = get_supported_models(llm_type_name)
    
    return sorted(models)


  def _call(self, prompt, **kwargs):
    return self.client(prompt, **kwargs)


  def __call__(self, prompt, **kwargs):
        """
        Override the default __call__ method to handle input preprocessing.
        I used this in order to override prompt input validation made by pydantic
        and allow sending list of AiMessages instead of string only
        """
        return self._call(prompt, **kwargs)
  

  def query(self, prompt, **kwargs):
    return self._call(prompt, **kwargs)

