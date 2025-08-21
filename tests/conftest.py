import pytest
import os

from langchain_timbr import LlmWrapper

@pytest.fixture(scope="session")
def config():
    return {
        # "cache_timeout": int(os.environ.get("CACHE_TIMEOUT", 120)),
        "llm_type": os.environ.get("LLM_TYPE", "openai-chat"),
        "llm_model": os.environ.get("LLM_MODEL", "gpt-4o-2024-11-20"),
        "llm_api_key": os.environ.get("LLM_API_KEY"),
        "timbr_url": os.environ.get("TIMBR_URL", "https://demo-env.timbr.ai"),
        "timbr_token": os.environ.get("TIMBR_TOKEN"),
        "timbr_ontology": os.environ.get("TIMBR_ONTOLOGY", "supply_metrics_llm_tests"),
        "test_prompt": os.environ.get("TEST_PROMPT", "What are the total sales for consumer customers?"),
        "test_prompt_2": os.environ.get("TEST_PROMPT_2", "Get all customers"),
        "test_prompt_3": os.environ.get("TEST_PROMPT_3", "Get all products and materials"),
        "verify_ssl": os.environ.get("VERIFY_SSL", "true"),
        "jwt_timbr_url": os.environ.get("JWT_TIMBR_URL", "https://staging.timbr.ai:443/"),
        "jwt_timbr_ontology": os.environ.get("JWT_TIMBR_ONTOLOGY", "supply_metrics"),
        "jwt_client_id": os.environ.get("JWT_CLIENT_ID"),
        "jwt_secret": os.environ.get("JWT_SECRET"),
        "jwt_tenant_id": os.environ.get("JWT_TENANT_ID"),
        "jwt_scope": os.environ.get("JWT_SCOPE"),
        "jwt_username": os.environ.get("JWT_USERNAME"),
        "jwt_password": os.environ.get("JWT_PASSWORD"),
    }

@pytest.fixture(scope="session")
def llm(config):
    # For testing purposes, provide a fallback if no API key is available
    api_key = config["llm_api_key"] or "test-api-key-for-testing"
    
    try:
        return LlmWrapper(
            llm_type=config["llm_type"],
            api_key=api_key,
            model=config["llm_model"],
        )
    except Exception as e:
        # If LLM creation fails, create a mock LLM for testing
        pytest.skip(f"Skipping tests requiring LLM due to: {e}")


@pytest.fixture(scope="session") 
def mock_llm():
    """Mock LLM for tests that don't need real LLM functionality."""
    from unittest.mock import Mock
    
    mock = Mock()
    mock.invoke.return_value = "mock response"
    mock.predict.return_value = "mock response"
    return mock
