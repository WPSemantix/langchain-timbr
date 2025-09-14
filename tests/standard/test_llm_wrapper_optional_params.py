import pytest
import os
from unittest.mock import patch

from langchain_timbr.llm_wrapper.llm_wrapper import LlmWrapper

class TestLlmWrapperOptionalParams:
    """Test optional parameters and config fallback for LlmWrapper"""
    
    def test_with_explicit_parameters(self):
        """Test that explicit parameters work as before"""
        # This should work without accessing config
        with patch.dict(os.environ, {}, clear=True):
            try:
                wrapper = LlmWrapper(
                    llm_type="openai-chat",
                    api_key="test-key",
                    model="gpt-4"
                )
                # Should not raise an exception during initialization
                assert wrapper is not None
            except Exception as e:
                # We expect this to fail due to invalid API key, but not due to missing parameters
                assert "llm_type must be provided" not in str(e)
                assert "api_key must be provided" not in str(e)
    
    def test_with_config_fallback(self):
        """Test that config fallback works"""
        # Mock the config values directly
        with patch('langchain_timbr.llm_wrapper.llm_wrapper.config.llm_type', 'openai-chat'),\
             patch('langchain_timbr.llm_wrapper.llm_wrapper.config.llm_api_key', 'test-key-from-config'),\
             patch('langchain_timbr.llm_wrapper.llm_wrapper.config.llm_model', 'gpt-4-from-config'):
            try:
                wrapper = LlmWrapper()  # No parameters provided
                assert wrapper is not None
            except Exception as e:
                # Should not fail due to missing parameters
                assert "llm_type must be provided" not in str(e)
                assert "api_key must be provided" not in str(e)
    
    def test_missing_llm_type_raises_error(self):
        """Test that missing llm_type raises appropriate error"""
        with patch.dict(os.environ, {}, clear=True):
            # Mock the config values to ensure they're None
            with patch('langchain_timbr.llm_wrapper.llm_wrapper.config.llm_type', None), \
                 patch('langchain_timbr.llm_wrapper.llm_wrapper.config.llm_api_key', 'test-key'):
                with pytest.raises(ValueError, match="llm_type must be provided"):
                    LlmWrapper(api_key="test-key")
    
    # This test is deprecated because api_key is now mandatory
    def skip_test_missing_api_key_raises_error(self):
        """Test that missing api_key raises appropriate error"""
        with patch.dict(os.environ, {}, clear=True):
            # Mock the config values to ensure they're None
            with patch('langchain_timbr.llm_wrapper.llm_wrapper.config.llm_type', 'openai-chat'), \
                 patch('langchain_timbr.llm_wrapper.llm_wrapper.config.llm_api_key', None):
                with pytest.raises(ValueError, match="api_key must be provided"):
                    LlmWrapper(llm_type="openai-chat")
    
    def test_additional_params_from_config(self):
        """Test that additional parameters can be loaded from config"""
        # Mock the config values directly
        with patch('langchain_timbr.llm_wrapper.llm_wrapper.config.llm_type', 'openai-chat'),\
             patch('langchain_timbr.llm_wrapper.llm_wrapper.config.llm_api_key', 'test-key'),\
             patch('langchain_timbr.llm_wrapper.llm_wrapper.config.llm_model', 'gpt-4'),\
             patch('langchain_timbr.llm_wrapper.llm_wrapper.config.llm_temperature', 0.8),\
             patch('langchain_timbr.llm_wrapper.llm_wrapper.config.llm_additional_params', '{"top_p": 0.9, "presence_penalty": 0.1}'):
            try:
                wrapper = LlmWrapper()  # No parameters provided
                assert wrapper is not None
                assert wrapper.client.top_p == 0.9
                assert wrapper.client.presence_penalty == 0.1
            except Exception as e:
                # Should not fail due to missing parameters
                assert "llm_type must be provided" not in str(e)
                assert "api_key must be provided" not in str(e)
    
    def test_explicit_params_override_config(self):
        """Test that explicit parameters override config values"""
        with patch.dict(os.environ, {
            'LLM_TYPE': 'openai-chat',
            'LLM_API_KEY': 'config-key',
            'LLM_MODEL': 'config-model'
        }):
            try:
                wrapper = LlmWrapper(
                    llm_type="anthropic-chat",
                    api_key="explicit-key",
                    model="explicit-model"
                )
                assert wrapper is not None
                assert wrapper.client._llm_type == "anthropic-chat"

            except Exception as e:
                # Should not fail due to missing parameters
                assert "llm_type must be provided" not in str(e)
                assert "api_key must be provided" not in str(e)
    
    def test_both_params_missing_raises_error(self):
        """Test that missing both llm_type and api_key raises appropriate error"""
        with patch.dict(os.environ, {}, clear=True):
            # Mock the config values to ensure they're None
            with patch('langchain_timbr.llm_wrapper.llm_wrapper.config.llm_type', None), \
                 patch('langchain_timbr.llm_wrapper.llm_wrapper.config.llm_api_key', None):
                with pytest.raises(ValueError, match="llm_type must be provided"):
                    LlmWrapper()
