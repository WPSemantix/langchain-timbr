from langchain_timbr import (
  GenerateTimbrSqlChain,
  ValidateTimbrSqlChain,
  ExecuteTimbrQueryChain,
  create_timbr_sql_agent
)

class TestLangchainChainsReasoningIntegration:
    def _assert_reasoning(self, chain, result, usage_metadata_key=None):
        assert 'reasoning_status' in result
        assert result['reasoning_status'] in ['correct', 'partial', 'incorrect']
        
        usage_metadata = result.get(usage_metadata_key or chain.usage_metadata_key, {})
        assert 'sql_reasoning_step_1' in usage_metadata

        # if first reasoning was incorrect, there must be a re-generating sql & a second reasoning step
        if 'sql_reasoning_step_2' in usage_metadata:
          assert 'generate_sql_reasoning_step_1' in usage_metadata

        # if the final result was incorrect - there must have two re-generation steps
        if result['reasoning_status'] == 'incorrect':
            assert 'generate_sql_reasoning_step_2' in usage_metadata

    # SKIP THIS TESTS UNTIL API WILL BE UPDATED

    def test_generate_timbr_sql_chain(self, llm, config):
        chain = GenerateTimbrSqlChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            verify_ssl=config["verify_ssl"],
            enable_reasoning=True,
        )
        result = chain.invoke({ "prompt": config["test_reasoning_prompt"] })
        print("GenerateTimbrSqlChain result:", result)
        self._assert_reasoning(chain, result)

    def test_validate_timbr_sql_chain(self, llm, config):
        chain = ValidateTimbrSqlChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            retries=1,  # Use a single retry for test speed
            verify_ssl=config["verify_ssl"],
            enable_reasoning=True,
        )
        inputs = {
            "prompt": config["test_reasoning_prompt"],
            "sql": "SELECT * FROM invalid_table",  # Intentionally invalid (or test with a valid one if available)
        }
        result = chain.invoke(inputs)
        print("ValidateTimbrSqlChain result:", result)
        self._assert_reasoning(chain, result)

    def test_execute_timbr_query_chain(self, llm, config):
        chain = ExecuteTimbrQueryChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            verify_ssl=config["verify_ssl"],
            enable_reasoning=True,
        )
        inputs = {
            "prompt": config["test_reasoning_prompt"],
        }
        result = chain.invoke(inputs)
        print("ExecuteTimbrQueryChain result:", result)
        self._assert_reasoning(chain, result)

    def test_create_timbr_sql_agent(self, llm, config):
        agent = create_timbr_sql_agent(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            verify_ssl=config["verify_ssl"],
            enable_reasoning=True,
        )
        result = agent.invoke(config["test_reasoning_prompt"])
        print("Timbr SQL Agent result:", result)
        self._assert_reasoning(agent, result, usage_metadata_key="usage_metadata")