from langchain_timbr import LlmWrapper, LlmTypes, ExecuteTimbrQueryChain
import os

class TestAzureDatabricksProvider:
    """Test suite for Azure Databricks provider integration with Timbr chains."""
    
    def skip_test_databricks_connection(self, llm, config):
        DATABRICKS_LLM_HOST = os.getenv('databricks_host')
        AZURE_TOKEN = os.getenv('databricks_token')
        
        llm_models_test = [
          "databricks-claude-sonnet-4",
          "databricks-gpt-oss-20b",
        ]

        for llm_model in llm_models_test:
          llm_instance = LlmWrapper(
              llm_type=LlmTypes.Databricks,
              api_key=AZURE_TOKEN,
              databricks_host=DATABRICKS_LLM_HOST,
              model=llm_model,
          )

          chain = ExecuteTimbrQueryChain(
              llm=llm_instance,
              url=config["timbr_url"],
              token=config["timbr_token"],
              ontology=config["timbr_ontology"],
              verify_ssl=config["verify_ssl"],
          )

          inputs = {
              "prompt": config["test_prompt"],
          }
          result = chain.invoke(inputs)

          print("ExecuteTimbrQueryChain result:", result)
          assert "rows" in result, "Result should contain 'rows'"
          assert isinstance(result["rows"], list), "'rows' should be a list"
          assert result["sql"], "SQL should be present in the result"

   