import pytest

from langchain_timbr import (
    IdentifyConceptNode,
    GenerateTimbrSqlNode,
    ValidateSemanticSqlNode,
    ExecuteSemanticQueryNode,
    GenerateResponseNode,
)


class TestLangGraphNodes:
    """Test suite for LangGraph node functionality."""
    
    # The following tests assume that:
    # - Your state is a dictionary.
    # - For nodes that require a prompt, we provide a 'prompt' key,
    #   and (if needed) a 'messages' list with at least one message object (a dict with a 'content' key).

    def test_identify_concept_node(self, llm, config):
        """Test the IdentifyConceptNode functionality."""
        node = IdentifyConceptNode(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            verify_ssl=config["verify_ssl"],
        )
        # Create a test state payload.
        state = {
            "prompt": config["test_prompt"],
            "messages": [{ "content": config["test_prompt"] }],
        }
        result = node(state)
        print("IdentifyConceptNode result:", result)
        assert "concept" in result, "Result should contain 'concept'"
        assert result["concept"], "Concept should not be empty"

    def test_generate_timbr_sql_node(self, llm, config):
        """Test the GenerateTimbrSqlNode functionality."""
        node = GenerateTimbrSqlNode(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            verify_ssl=config["verify_ssl"],
        )
        state = {
            "prompt": config["test_prompt"],
            "messages": [{ "content": config["test_prompt"] }],
        }
        result = node(state)
        print("GenerateTimbrSqlNode result:", result)
        assert "sql" in result and result["sql"], "SQL should be generated"
        assert "concept" in result and result["concept"], "Concept should be returned"

    def test_validate_semantic_sql_node(self, llm, config):
        """Test the ValidateSemanticSqlNode functionality."""
        node = ValidateSemanticSqlNode(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            retries=1,  # Using one retry for test speed.
            verify_ssl=config["verify_ssl"],
        )
        # For testing, provide a state with a SQL query.
        state = {
            "prompt": config["test_prompt"],
            "sql": "SELECT * FROM invalid_table",  # Intentionally invalid (or test with a valid one if available)
        }
        result = node(state)
        print("ValidateSemanticSqlNode result:", result)
        assert "is_sql_valid" in result, "Result should include 'is_sql_valid'"
        # If your test query is valid, the flag should be True.
        assert result["is_sql_valid"] is True, "SQL should be valid"

    def test_execute_semantic_query_node(self, llm, config):
        """Test the ExecuteSemanticQueryNode functionality."""
        node = ExecuteSemanticQueryNode(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            verify_ssl=config["verify_ssl"],
        )
        state = {
            "prompt": config["test_prompt"],
        }
        result = node(state)
        print("ExecuteSemanticQueryNode result:", result)
        assert "rows" in result, "Result should contain 'rows'"
        assert isinstance(result["rows"], list), "'rows' should be a list"

    def test_generate_response_node(self, llm, config):
        """Test the GenerateResponseNode functionality."""
        node = GenerateResponseNode(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            verify_ssl=config["verify_ssl"],
        )
        state = {
            "prompt": config["test_prompt"],
            "sql": "SELECT total_sales FROM dtimbr.product",
            "schema": "dtimbr",
            "concept": "product",
            "rows": [["1000"], ["2000"]],
        }
        result = node(state)
        print("GenerateResponseNode result:", result)
        assert "answer" in result, "Result should contain 'answer'"
        assert result["answer"], "Answer should not be empty"
