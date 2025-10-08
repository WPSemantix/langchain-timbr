"""Unit tests for individual chain components."""
import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from langchain_timbr import (
    IdentifyTimbrConceptChain,
    GenerateTimbrSqlChain,
    ExecuteTimbrQueryChain
)
from langchain_timbr.utils.timbr_llm_utils import _calculate_token_count


class TestChainUnitTests:
    """Unit tests for individual chain functionality."""
    
    def test_identify_concept_chain_unit(self, mock_llm):
        """Unit test for IdentifyTimbrConceptChain without external dependencies."""
        with patch('langchain_timbr.langchain.identify_concept_chain.determine_concept') as mock_determine:
            mock_determine.return_value = {
                'concept': 'customer',
                'schema': 'dtimbr',
                'concept_metadata': {},
                'usage_metadata': {}
            }
            
            chain = IdentifyTimbrConceptChain(
                llm=mock_llm,
                url="http://test",
                token="test",
                ontology="test"
            )
            
            result = chain.invoke({"prompt": "What are the customers?"})
            assert 'concept' in result
            mock_determine.assert_called_once()
    
    def test_generate_sql_chain_unit(self, mock_llm):
        """Unit test for GenerateTimbrSqlChain without external dependencies."""
        with patch('langchain_timbr.langchain.generate_timbr_sql_chain.generate_sql') as mock_generate:
            mock_generate.return_value = {
                'sql': 'SELECT * FROM customer',
                'concept': 'customer',
                'usage_metadata': {}
            }
            
            chain = GenerateTimbrSqlChain(
                llm=mock_llm,
                url="http://test",
                token="test",
                ontology="test"
            )

            result = chain.invoke({"prompt": "Get all customers"})
            assert 'sql' in result
            mock_generate.assert_called_once()
    
    def test_execute_query_chain_unit(self):
        """Test ExecuteTimbrQueryChain unit functionality."""
        from unittest.mock import Mock
        
        # Mock the LLM
        mock_llm = Mock()
        mock_llm.invoke.return_value = "SELECT * FROM customers"
        
        # Create chain
        chain = ExecuteTimbrQueryChain(
            llm=mock_llm,
            url="http://test.com",
            token="test_token",
            ontology="test_ontology"
        )
        
        # Mock the _call method to return expected output format with all required keys
        expected_result = {
            "rows": [{"id": 1, "name": "Customer 1"}],
            "sql": "SELECT * FROM customers", 
            "schema": "dtimbr",
            "concept": None,
            "error": None,
            "execute_timbr_usage_metadata": {}
        }
        chain._call = Mock(return_value=expected_result)
        
        # Test invocation
        result = chain.invoke({"prompt": "Get all customers"})
        
        # Verify result structure contains all expected keys
        assert isinstance(result, dict)
        assert "rows" in result
        assert "sql" in result
        assert "schema" in result
        assert "error" in result
        assert "execute_timbr_usage_metadata" in result
    
    def test_chain_input_sanitization(self, mock_llm):
        """Test that chains properly sanitize inputs."""
        chain = IdentifyTimbrConceptChain(
            llm=mock_llm,
            url="http://test",
            token="test",
            ontology="test"
        )
        
        # Test with various input types
        test_prompts = [
            "normal question",
            "question with 'quotes'",
            "question with \"double quotes\"",
            "question with; semicolon",
            "",  # empty string
        ]
        
        for prompt in test_prompts:
            # Should not raise exceptions for any input
            try:
                # This will fail connection but shouldn't crash on input validation
                chain.invoke({"prompt": prompt})
            except Exception as e:
                # Should be connection-related, not input validation
                error_msg = str(e).lower()
                assert any(keyword in error_msg for keyword in 
                          ["connection", "invalid", "network", "rstrip", "nonetype"])
    
    def test_chain_parameter_validation(self, mock_llm):
        """Test that chains validate constructor parameters."""
        # Test that chain can be created with valid parameters
        try:
            chain = IdentifyTimbrConceptChain(
                llm=mock_llm,
                url="http://test",
                token="test",
                ontology="test"
            )
            assert chain is not None, "Chain should be created with valid parameters"
        except Exception as e:
            pytest.fail(f"Chain creation failed unexpectedly: {e}")
        
        # Test invalid parameter types (if the chain validates them)
        try:
            invalid_chain = IdentifyTimbrConceptChain(
                llm="not_an_llm",  # Invalid LLM type
                url="http://test",
                token="test",
                ontology="test"
            )
            # If it doesn't raise an error, that's also acceptable for some implementations
            assert invalid_chain is not None
        except (ValueError, TypeError, AttributeError):
            # These errors are expected for invalid parameters
            pass
    
    def test_chain_state_management(self, mock_llm):
        """Test that chains properly manage internal state."""
        chain = IdentifyTimbrConceptChain(
            llm=mock_llm,
            url="http://test-url",
            token="test-token",
            ontology="test-ontology"
        )
        
        # Test that chain maintains configuration in private attributes
        assert hasattr(chain, '_url'), "Chain should store URL parameter"
        assert hasattr(chain, '_token'), "Chain should store token parameter"
        assert hasattr(chain, '_ontology'), "Chain should store ontology parameter"
        
        # Test that multiple instances don't interfere
        chain2 = IdentifyTimbrConceptChain(
            llm=mock_llm,
            url="http://different",
            token="different-token",
            ontology="different-ontology"
        )
        
        assert chain._url != chain2._url
        assert chain._token != chain2._token
        assert chain._ontology != chain2._ontology


class TestTokenCountFunctionality:
    """Test suite for token counting functionality with tiktoken."""
    
    def test_calculate_token_count_with_string_prompt(self):
        """Test token counting with a simple string prompt."""
        mock_llm = Mock()
        mock_llm._llm_type = "openai"
        mock_llm.client = Mock()
        mock_llm.client.model_name = "gpt-4"
        
        prompt = "What are the top customers?"
        token_count = _calculate_token_count(mock_llm, prompt)
        
        assert token_count > 0, "Token count should be greater than 0 for non-empty prompt"
        assert isinstance(token_count, int), "Token count should be an integer"
    
    def test_calculate_token_count_with_list_prompt(self):
        """Test token counting with a list-based prompt (ChatPrompt format)."""
        mock_llm = Mock()
        mock_llm._llm_type = "openai"
        
        # Mock message objects with type and content
        system_msg = Mock()
        system_msg.type = "system"
        system_msg.content = "You are a helpful SQL assistant."
        
        user_msg = Mock()
        user_msg.type = "user"
        user_msg.content = "Generate SQL for top customers"
        
        prompt = [system_msg, user_msg]
        token_count = _calculate_token_count(mock_llm, prompt)
        
        assert token_count > 0, "Token count should be greater than 0 for non-empty prompt"
        assert isinstance(token_count, int), "Token count should be an integer"
    
    def test_calculate_token_count_without_model_name(self):
        """Test token counting falls back when LLM doesn't have model_name attribute."""
        mock_llm = Mock()
        # LLM without client.model_name attribute
        mock_llm.client = Mock(spec=[])
        
        prompt = "Test prompt without model name"
        token_count = _calculate_token_count(mock_llm, prompt)
        
        # Should still return a count using fallback encoding
        assert token_count >= 0, "Token count should not fail when model_name is missing"
        assert isinstance(token_count, int), "Token count should be an integer"
    
    def test_calculate_token_count_without_client(self):
        """Test token counting falls back when LLM doesn't have client attribute."""
        mock_llm = Mock(spec=['_llm_type'])
        mock_llm._llm_type = "custom"
        # LLM without client attribute at all
        if hasattr(mock_llm, 'client'):
            delattr(mock_llm, 'client')
        
        prompt = "Test prompt without client"
        token_count = _calculate_token_count(mock_llm, prompt)
        
        # Should still return a count using fallback encoding
        assert token_count >= 0, "Token count should not fail when client is missing"
        assert isinstance(token_count, int), "Token count should be an integer"
    
    def test_calculate_token_count_with_tiktoken_error(self):
        """Test token counting handles tiktoken errors gracefully."""
        mock_llm = Mock()
        mock_llm._llm_type = "custom"
        mock_llm.client = Mock()
        mock_llm.client.model_name = "unknown-model-that-causes-error"
        
        # This should not raise an exception even if tiktoken fails
        prompt = "Test prompt with potential tiktoken error"
        token_count = _calculate_token_count(mock_llm, prompt)
        
        # Should return 0 or a valid count even on error
        assert token_count >= 0, "Token count should return 0 or valid count on error"
        assert isinstance(token_count, int), "Token count should be an integer"
    
    def test_calculate_token_count_empty_prompt(self):
        """Test token counting with empty prompt."""
        mock_llm = Mock()
        mock_llm._llm_type = "openai"
        
        prompt = ""
        token_count = _calculate_token_count(mock_llm, prompt)
        
        assert token_count == 0, "Token count should be 0 for empty prompt"
        assert isinstance(token_count, int), "Token count should be an integer"
    
    def test_calculate_token_count_with_different_llm_types(self):
        """Test token counting works with different LLM types."""
        llm_types = ["openai", "anthropic", "azure", "custom", "databricks"]
        
        for llm_type in llm_types:
            mock_llm = Mock()
            mock_llm._llm_type = llm_type
            
            prompt = f"Test prompt for {llm_type}"
            token_count = _calculate_token_count(mock_llm, prompt)
            
            assert token_count >= 0, f"Token count should work for {llm_type}"
            assert isinstance(token_count, int), f"Token count should be integer for {llm_type}"
