import pytest
from langchain_timbr import create_timbr_sql_agent


class TestTimbrSqlAgentIntegration:
    """Test suite for Timbr SQL Agent integration functionality."""
    
    def test_timbr_sql_agent_integration(self, llm, config):
        """Test basic Timbr SQL Agent integration with case-sensitive database."""
        agent = create_timbr_sql_agent(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            db_is_case_sensitive=True,
            verify_ssl=config["verify_ssl"],
        )
        result = agent.invoke(config["test_prompt"])
        
        print(result['sql'])
        print(result['rows'])
        print(result['concept'])

        assert "sql" in result and result["sql"], "SQL should be generated"
        assert " LOWER(" in result["sql"], "Generated query should be match to case-sensitive database"
        assert "rows" in result, "Rows should be returned"
        assert "concept" in result, "Concept should be returned"
        assert len(result["rows"]) > 0, "Rows should be returned"
        assert "usage_metadata" in result, "Agent should return 'usage_metadata'"
        assert len(result["usage_metadata"]) == 2 and 'determine_concept' in result["usage_metadata"] and 'generate_sql' in result["usage_metadata"], "Usage metadata should contain both 'determine_concept' and 'generate_sql'"
    
    def test_timbr_sql_agent_with_generate_answer(self, llm, config):
        """Test the Timbr SQL Agent with generate_answer enabled to ensure natural language responses are generated."""
        agent = create_timbr_sql_agent(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            generate_answer=True,  # Enable answer generation
            retry_if_no_results=True,
            verify_ssl=config["verify_ssl"],
        )
        result = agent.invoke(config["test_prompt"])
        
        print("SQL:", result['sql'])
        print("Answer:", result.get('answer', 'No answer generated'))

        # Test that all expected fields are present
        assert "sql" in result and result["sql"], "SQL should be generated"
        assert "rows" in result, "Rows should be returned"
        assert "concept" in result, "Concept should be returned"
        assert "answer" in result, "Answer should be present when generate_answer=True"
        
        # Test that answer has actual content
        assert result["answer"] is not None, "Answer should not be None"
        assert isinstance(result["answer"], str), "Answer should be a string"
        assert len(result["answer"].strip()) > 0, "Answer should not be empty"

    def test_timbr_sql_agent_generate_answer_disabled(self, llm, config):
        """Test that when generate_answer is False, no answer is generated."""
        agent = create_timbr_sql_agent(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            generate_answer=False,
            verify_ssl=config["verify_ssl"],
        )
        result = agent.invoke(config["test_prompt"])
        
        # Test that answer is None when disabled
        assert "answer" in result, "Answer key should be present"
        assert result["answer"] is None, "Answer should be None when generate_answer=False"
        
        # Test that other fields still work correctly
        assert "sql" in result and result["sql"], "SQL should be generated"
        assert "rows" in result, "Rows should be returned"
        assert "concept" in result, "Concept should be returned"
        assert len(result["rows"]) > 0, "Rows should be returned"


class TestTimbrSqlAgentErrors:
    """Test suite for Timbr SQL Agent error handling scenarios."""
    
    def test_wrong_ontology_name_error(self, llm, config):
        """Test error handling for incorrect ontology name."""
        agent = create_timbr_sql_agent(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"] + 'wrong',
            verify_ssl=config["verify_ssl"],
        )
        result = agent.invoke(config["test_prompt"])
        assert "error" in result and "Unknown database" in result["error"], "Error should be returned"
        assert result["schema"] is None, "Schema should not be returned"
        assert result["concept"] is None, "Concept should not be returned"
        assert result["rows"] is None, "Rows should not be returned"
        assert result["sql"] is None, "SQL should not be returned"
        assert "usage_metadata" in result, "Chain should return 'usage_metadata'"
        assert len(result["usage_metadata"]) == 0, "Usage metadata should be empty on error"

    def test_wrong_token_error(self, llm, config):
        """Test error handling for incorrect authentication token."""
        agent = create_timbr_sql_agent(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"] + 'wrong',
            ontology=config["timbr_ontology"],
            verify_ssl=config["verify_ssl"],
        )
        result = agent.invoke(config["test_prompt"])
        assert "error" in result and "HTTP Response code: 401" in result["error"], "Error should be returned"
        assert result["schema"] is None, "Schema should not be returned"
        assert result["concept"] is None, "Concept should not be returned"
        assert result["rows"] is None, "Rows should not be returned"
        assert result["sql"] is None, "SQL should not be returned"


class TestTimbrSqlAgentDebugMode:
    """Test suite for Timbr SQL Agent debug mode functionality."""
    
    def test_debug_returns_prompt_hash(self, llm, config):
        """Test that the debug information includes the prompt hash."""
        agent = create_timbr_sql_agent(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            debug=True,
            verify_ssl=config["verify_ssl"],
        )

        result = agent.invoke(config["test_prompt"])
        usage_metadata = result.get("usage_metadata", {})
        for key, value in usage_metadata.items():
            assert "p_hash" in value, f"Prompt hash should be present in usage metadata for {key}"

    def test_no_debug_doesnt_return_prompt_hash(self, llm, config):
        """Test that the debug information does not include the prompt hash."""
        agent = create_timbr_sql_agent(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            verify_ssl=config["verify_ssl"],
        )

        result = agent.invoke(config["test_prompt"])
        usage_metadata = result.get("usage_metadata", {})
        for key, value in usage_metadata.items():
            assert "p_hash" not in value, f"Prompt hash should not be present in usage metadata for {key}"
