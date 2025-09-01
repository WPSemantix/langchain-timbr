from enum import Enum
from langchain.llms.base import LLM
from pydantic import Field

from .timbr_llm_wrapper import TimbrLlmWrapper
from ..utils.general import is_llm_type, is_support_temperature
from ..config import llm_temperature

class LlmTypes(Enum):
  OpenAI = 'openai-chat'
  Anthropic = 'anthropic-chat'
  Google = 'chat-google-generative-ai'
  AzureOpenAI = 'azure-openai-chat'
  Snowflake = 'snowflake-cortex'
  Databricks = 'chat-databricks'
  Timbr = 'timbr'


class LlmWrapper(LLM):
  """
  LlmWrapper is a unified interface for connecting to various Large Language Model (LLM) providers
  (OpenAI, Anthropic, Google, Azure OpenAI, Snowflake Cortex, Databricks, etc.) using LangChain. It abstracts
  the initialization and connection logic for each provider, allowing you to switch between them
  with a consistent API.
  """
  client: LLM = Field(default=None, exclude=True)

  def __init__(
      self,
      llm_type: str,
      api_key: str,
      model: str = None,
      **llm_params,
  ):
      """
      :param llm_type (str): The type of LLM provider (e.g., 'openai-chat', 'anthropic-chat').
      :param api_key (str): The API key for authenticating with the LLM provider.
      :param model (str): The model name or deployment to use. Defaults to provider-specific values (Optional).
      :param **llm_params: Additional parameters for the LLM (e.g., temperature, endpoint, etc.).
      """
      super().__init__()
      self.client = self._connect_to_llm(
        llm_type,
        api_key,
        model,
        **llm_params,
      )


  @property
  def _llm_type(self):
    return self.client._llm_type


  def _add_temperature(self, llm_type, llm_model, **llm_params):
    """
    Add temperature to the LLM parameters if the LLM model supports it.
    """
    if "temperature" not in llm_params:
      if llm_temperature is not None and is_support_temperature(llm_type, llm_model):
        llm_params["temperature"] = llm_temperature
    return llm_params

  
  def _connect_to_llm(self, llm_type, api_key, model, **llm_params):
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
      azure_endpoint = params.pop('azure_endpoint', None)
      azure_api_version = params.pop('azure_openai_api_version', None)
      llm_model = model or "gpt-4o-2024-11-20"
      params = self._add_temperature(LlmTypes.AzureOpenAI.name, llm_model, **llm_params)
      return AzureChatOpenAI(
        openai_api_key=api_key,
        azure_deployment=llm_model,
        azure_endpoint=azure_endpoint,
        openai_api_version=azure_api_version,
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
    else:
      raise ValueError(f"Unsupported LLM type: {llm_type}")


  def get_model_list(self) -> list[str]:
    """Return the list of available models for the LLM."""
    models = []
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
        azure_endpoint = getattr(self.client, 'azure_endpoint', None)
        api_version = getattr(self.client, 'openai_api_version', None)
        api_key = self.client.openai_api_key._secret_value
        
        if azure_endpoint and api_version and api_key:
          client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version
          )
          # For Azure, get the deployments instead of models
          try:
            models = [model.id for model in client.models.list()]
          except:
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
        w = self.client.workspace_client
        models = [ep.name for ep in w.serving_endpoints.list()]
        
      # elif self._is_llm_type(self._llm_type, LlmTypes.Timbr):
        
    except Exception as e:
      models = []
    
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

