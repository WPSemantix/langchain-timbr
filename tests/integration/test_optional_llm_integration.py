"""Integration test for explicit LLM overriding environment configuration.

Moved here from tests/standard because it instantiates ExecuteTimbrQueryChain and
TimbrSqlAgent without explicit connection params, relying on TIMBR_URL/TIMBR_TOKEN
being present in the environment (sourced from launch.json for integration runs).
"""

from unittest.mock import patch

from langchain_timbr.langchain.execute_timbr_query_chain import ExecuteTimbrQueryChain
from langchain_timbr.langchain.timbr_sql_agent import TimbrSqlAgent


class TestOptionalLLMIntegration:
    """Test explicit LLM parameter behavior against a live backend."""

    def test_explicit_llm_overrides_env_variables(self):
        """Test that providing explicit LLM parameter works even with env variables"""
        from langchain_timbr.llm_wrapper.llm_wrapper import LlmWrapper

        # Mock the config values
        with patch('langchain_timbr.llm_wrapper.llm_wrapper.config.llm_type', 'openai-chat'),\
             patch('langchain_timbr.llm_wrapper.llm_wrapper.config.llm_api_key', 'env-key'):
            # Create explicit LLM
            explicit_llm = LlmWrapper(
                llm_type='openai-chat',
                api_key='explicit-key',
                model='gpt-3.5-turbo'
            )

            # Test chain with explicit LLM
            chain = ExecuteTimbrQueryChain(llm=explicit_llm)
            assert chain is not None
            assert chain._llm is explicit_llm

            # Test agent with explicit LLM
            agent = TimbrSqlAgent(llm=explicit_llm)
            assert agent is not None
            assert agent._chain._llm is explicit_llm
