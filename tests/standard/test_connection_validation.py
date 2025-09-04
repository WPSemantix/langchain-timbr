"""Unit tests for Timbr connection parameter validation."""
import pytest
from unittest.mock import Mock

from langchain_timbr.langchain.execute_timbr_query_chain import ExecuteTimbrQueryChain
from langchain_timbr.utils.general import validate_timbr_connection_params


class TestConnectionValidation:
    """Test suite for Timbr connection parameter validation."""

    def test_validate_timbr_connection_params_with_valid_params(self):
        """Test that validation passes with valid URL and token."""
        # Should not raise an exception
        validate_timbr_connection_params(url="http://test-url", token="test-token")

    def test_validate_timbr_connection_params_missing_url(self):
        """Test that validation fails when URL is missing."""
        with pytest.raises(ValueError) as exc_info:
            validate_timbr_connection_params(url=None, token="test-token")
        
        assert "URL must be provided" in str(exc_info.value)
        assert "TIMBR_URL" in str(exc_info.value)

    def test_validate_timbr_connection_params_missing_token(self):
        """Test that validation fails when token is missing."""
        with pytest.raises(ValueError) as exc_info:
            validate_timbr_connection_params(url="http://test-url", token=None)
        
        assert "Token must be provided" in str(exc_info.value)
        assert "TIMBR_TOKEN" in str(exc_info.value)

    def test_validate_timbr_connection_params_empty_url(self):
        """Test that validation fails when URL is empty string."""
        with pytest.raises(ValueError) as exc_info:
            validate_timbr_connection_params(url="", token="test-token")
        
        assert "URL must be provided" in str(exc_info.value)

    def test_validate_timbr_connection_params_empty_token(self):
        """Test that validation fails when token is empty string."""
        with pytest.raises(ValueError) as exc_info:
            validate_timbr_connection_params(url="http://test-url", token="")
        
        assert "Token must be provided" in str(exc_info.value)

    def test_validate_timbr_connection_params_both_missing(self):
        """Test that validation fails when both URL and token are missing."""
        with pytest.raises(ValueError) as exc_info:
            validate_timbr_connection_params(url=None, token=None)
        
        # Should fail on the first missing parameter (URL)
        assert "URL must be provided" in str(exc_info.value)

    def test_chain_validation_with_explicit_params(self):
        """Test that ExecuteTimbrQueryChain works with explicit valid parameters."""
        mock_llm = Mock()
        
        # Should not raise an exception
        chain = ExecuteTimbrQueryChain(
            llm=mock_llm,
            url="http://test-url",
            token="test-token",
            ontology="test-ontology"
        )
        
        assert chain._url == "http://test-url"
        assert chain._token == "test-token"
        assert chain._ontology == "test-ontology"


    def test_chain_validation_missing_url(self, monkeypatch):
        """Test that ExecuteTimbrQueryChain fails when URL is not provided."""
        mock_llm = Mock()

        # Mock the config module to ensure URL is None
        monkeypatch.setattr('langchain_timbr.langchain.execute_timbr_query_chain.config.url', None)
        
        with pytest.raises(ValueError) as exc_info:
            ExecuteTimbrQueryChain(
                llm=mock_llm,
                url=None,
                token="test-token",
                ontology="test-ontology"
            )

        assert "URL must be provided" in str(exc_info.value)
        assert "TIMBR_URL" in str(exc_info.value)


    def test_chain_validation_missing_token(self, monkeypatch):
        """Test that ExecuteTimbrQueryChain fails when token is not provided."""
        mock_llm = Mock()
        
        # Mock the config module to ensure token is None
        monkeypatch.setattr('langchain_timbr.langchain.execute_timbr_query_chain.config.token', None)
        
        with pytest.raises(ValueError) as exc_info:
            ExecuteTimbrQueryChain(
                llm=mock_llm,
                url="http://test-url",
                token=None,
                ontology="test-ontology"
            )
        
        assert "Token must be provided" in str(exc_info.value)
        assert "TIMBR_TOKEN" in str(exc_info.value)

    def test_chain_validation_with_config_defaults(self, monkeypatch):
        """Test that ExecuteTimbrQueryChain uses config defaults when parameters are None."""
        mock_llm = Mock()
        
        # Mock the config module
        monkeypatch.setattr('langchain_timbr.langchain.execute_timbr_query_chain.config.url', 'http://config-url')
        monkeypatch.setattr('langchain_timbr.langchain.execute_timbr_query_chain.config.token', 'config-token')
        monkeypatch.setattr('langchain_timbr.langchain.execute_timbr_query_chain.config.ontology', 'config-ontology')
        
        chain = ExecuteTimbrQueryChain(
            llm=mock_llm,
            url=None,  # Should use config default
            token=None,  # Should use config default
            ontology=None  # Should use config default
        )
        
        assert chain._url == "http://config-url"
        assert chain._token == "config-token"
        assert chain._ontology == "config-ontology"

    def test_chain_validation_with_missing_config_defaults(self, monkeypatch):
        """Test that ExecuteTimbrQueryChain fails when config defaults are also None."""
        mock_llm = Mock()
        
        # Mock the config module with None values
        monkeypatch.setattr('langchain_timbr.langchain.execute_timbr_query_chain.config.url', None)
        monkeypatch.setattr('langchain_timbr.langchain.execute_timbr_query_chain.config.token', None)
        monkeypatch.setattr('langchain_timbr.langchain.execute_timbr_query_chain.config.ontology', None)
        
        with pytest.raises(ValueError) as exc_info:
            ExecuteTimbrQueryChain(
                llm=mock_llm,
                url=None,
                token=None,
                ontology=None
            )
        
        assert "URL must be provided" in str(exc_info.value)
