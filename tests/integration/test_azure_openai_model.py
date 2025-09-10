from langchain_openai import AzureChatOpenAI
from langchain_timbr import LlmWrapper, LlmTypes, ExecuteTimbrQueryChain


class TestAzureOpenAIModel:
    """Test suite for Azure OpenAI model integration with Timbr chains."""
    
    def skip_test_gpt_o3_mini(self, llm, config):
        """Test GPT-o3-mini model integration with ExecuteTimbrQueryChain."""
        timbr_token = "..."
        host = "http://localhost:5000"
        ontology = "timbr_crunchbase"

        AZURE_LLM_ENDPOINT = "..."
        AZURE_API_VERSION = "2024-12-01-preview"
        AZURE_OPENAI_API_KEY = "..."
        
        LLM_MODEL = "o3-mini"
        # LLM_MODEL = "gpt-4o"

        llm_instance = AzureChatOpenAI(
            azure_endpoint=AZURE_LLM_ENDPOINT,
            openai_api_version=AZURE_API_VERSION,
            openai_api_key=AZURE_OPENAI_API_KEY,
            azure_deployment=LLM_MODEL,
            # disabled_params={"temperature": None},
        )

        # llm_instance = LlmWrapper(
        #   llm_type=LlmTypes.OpenAI,
        #   api_key=config["llm_api_key"],
        #   model=LLM_MODEL,
        # )

        chain = ExecuteTimbrQueryChain(
            llm=llm_instance,
            url=host,
            token=timbr_token,
            ontology=ontology,
            views_list=["org1", "person"],
        )

        inputs = {
            "prompt": "who is Mark Zuckerberg",
        }
        result = chain.invoke(inputs)

        print("ExecuteTimbrQueryChain result:", result)
        assert "rows" in result, "Result should contain 'rows'"
        assert isinstance(result["rows"], list), "'rows' should be a list"
        assert result["sql"], "SQL should be present in the result"

        inputs = {
            "prompt": "What is the first company Microsoft acquired?",
        }
        result = chain.invoke(inputs)

        print("ExecuteTimbrQueryChain result:", result)
        assert "rows" in result, "Result should contain 'rows'"
        assert isinstance(result["rows"], list), "'rows' should be a list"
        assert result["sql"], "SQL should be present in the result"

    def skip_test_azure_inference(self, llm, config):
        """Test Azure inference with ExecuteTimbrQueryChain using GPT-4o."""
        timbr_token = ""
        host = "https://demo-env.timbr.ai/"
        ontology = "bom_1_new"

        AZURE_LLM_ENDPOINT = ""
        AZURE_API_VERSION = "2024-12-01-preview"
        AZURE_OPENAI_API_KEY = ""
        
        LLM_MODEL = "gpt-4o"

        llm_instance = AzureChatOpenAI(
            azure_endpoint=AZURE_LLM_ENDPOINT,
            openai_api_version=AZURE_API_VERSION,
            openai_api_key=AZURE_OPENAI_API_KEY,
            azure_deployment=LLM_MODEL,
        )

        chain = ExecuteTimbrQueryChain(
            llm=llm_instance,
            url=host,
            token=timbr_token,
            ontology=ontology,
            retry_if_no_results=True,
            no_results_max_retries=5,
        )

        inputs = {
            "prompt": "Show me delivieries for id 1",
        }
        result = chain.invoke(inputs)

        print("ExecuteTimbrQueryChain result:", result)
        assert "rows" in result, "Result should contain 'rows'"
        assert isinstance(result["rows"], list), "'rows' should be a list"
        assert result["sql"], "SQL should be present in the result"

    def skip_test_azure_service_principal_with_client_and_secret(self, llm, config):
        """Test Azure OpenAI model integration using service principal with client ID and secret."""
        ontology = "timbr_crunchbase"

        AZURE_LLM_ENDPOINT = "<Azure OpenAI endpoint>"
        AZURE_API_VERSION = "2024-12-01-preview"
        AZURE_CLIENT_ID = "..."
        AZURE_TENANT_ID = "..."
        AZURE_CLIENT_SECRET = "..."
        LLM_MODEL = "gpt-4o"

        llm_instance = LlmWrapper(
            llm_type=LlmTypes.AzureOpenAI,
            api_key=AZURE_CLIENT_SECRET,
            model=LLM_MODEL,
            azure_endpoint=AZURE_LLM_ENDPOINT,
            azure_api_version=AZURE_API_VERSION,
            azure_client_id=AZURE_CLIENT_ID,
            azure_tenant_id=AZURE_TENANT_ID,
        )

        # models = llm_instance.get_model_list()
        # print("Available models:", models)

        chain = ExecuteTimbrQueryChain(
            llm=llm_instance,
            url=config['timbr_url'],
            token=config['timbr_token'],
            ontology=ontology,
            views_list=["org1", "person"],
        )

        inputs = {
            "prompt": "who is Mark Zuckerberg",
        }
        result = chain.invoke(inputs)

        print("ExecuteTimbrQueryChain result:", result)
        assert "rows" in result, "Result should contain 'rows'"
        assert isinstance(result["rows"], list), "'rows' should be a list"
        assert result["sql"], "SQL should be present in the result"
