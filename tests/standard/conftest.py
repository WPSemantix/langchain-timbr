"""Conftest file for standard tests with mock fixtures."""
import pytest
from unittest.mock import Mock, MagicMock


@pytest.fixture
def mock_llm():
    """Mock LLM for standard tests that don't need real LLM functionality."""
    mock = Mock()
    mock.invoke.return_value = "mock response"
    mock.predict.return_value = "mock response"
    mock.stream.return_value = iter(["mock", " response"])
    return mock


@pytest.fixture
def minimal_config():
    """Minimal configuration for standard tests."""
    return {
        "timbr_url": "http://test-timbr-url",
        "timbr_token": "test-timbr-token",
        "timbr_ontology": "test-timbr-ontology",
        "verify_ssl": True
    }
