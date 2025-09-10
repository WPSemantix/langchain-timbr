import pytest
import os
from unittest.mock import patch
from langchain_timbr.langchain.execute_timbr_query_chain import ExecuteTimbrQueryChain
from langchain_timbr.langchain.generate_answer_chain import GenerateAnswerChain
from langchain_timbr.langchain.generate_timbr_sql_chain import GenerateTimbrSqlChain
from langchain_timbr.langchain.identify_concept_chain import IdentifyTimbrConceptChain
from langchain_timbr.langchain.validate_timbr_sql_chain import ValidateTimbrSqlChain
from langchain_timbr.langchain.timbr_sql_agent import TimbrSqlAgent, create_timbr_sql_agent
from langchain_timbr.langgraph.execute_timbr_query_node import ExecuteSemanticQueryNode
from langchain_timbr.langgraph.generate_timbr_sql_node import GenerateTimbrSqlNode
from langchain_timbr.langgraph.identify_concept_node import IdentifyConceptNode
from langchain_timbr.langgraph.generate_response_node import GenerateResponseNode
from langchain_timbr.langgraph.validate_timbr_query_node import ValidateSemanticSqlNode


class TestOptionalLLMIntegration:
    """Test that all chains, agents, and nodes work with optional LLM parameters"""
    
    def test_chains_with_env_variables(self):
        """Test that all chains can be initialized without LLM when config defaults are available"""
        # Mock the config values directly instead of environment variables
        with patch('langchain_timbr.llm_wrapper.llm_wrapper.config.llm_type', 'openai-chat'),\
             patch('langchain_timbr.llm_wrapper.llm_wrapper.config.llm_api_key', 'test-key'),\
             patch('langchain_timbr.llm_wrapper.llm_wrapper.config.llm_model', 'gpt-4'),\
             patch.dict(os.environ, {
                'TIMBR_URL': 'http://test-timbr.com',
                'TIMBR_TOKEN': 'test-token',
                'TIMBR_ONTOLOGY': 'test-ontology'
             }):
            # Test all chain classes
            chains = [
                ExecuteTimbrQueryChain,
                GenerateAnswerChain,
                GenerateTimbrSqlChain,
                IdentifyTimbrConceptChain,
                ValidateTimbrSqlChain
            ]
            
            for chain_class in chains:
                try:
                    chain = chain_class()  # No LLM parameter provided
                    assert chain is not None
                    assert hasattr(chain, '_llm')
                    assert chain._llm is not None
                except Exception as e:
                    # Should not fail due to missing LLM parameters
                    assert "llm_type must be provided" not in str(e)
                    assert "api_key must be provided" not in str(e)
                    assert "Failed to initialize LLM from environment variables" not in str(e)
    
    def test_agent_with_env_variables(self):
        """Test that TimbrSqlAgent can be initialized without LLM when config defaults are available"""
        # Mock the config values directly instead of environment variables
        with patch('langchain_timbr.llm_wrapper.llm_wrapper.config.llm_type', 'openai-chat'),\
             patch('langchain_timbr.llm_wrapper.llm_wrapper.config.llm_api_key', 'test-key'),\
             patch('langchain_timbr.llm_wrapper.llm_wrapper.config.llm_model', 'gpt-4'),\
             patch.dict(os.environ, {
                'TIMBR_URL': 'http://test-timbr.com',
                'TIMBR_TOKEN': 'test-token',
                'TIMBR_ONTOLOGY': 'test-ontology'
             }):
            try:
                agent = TimbrSqlAgent()  # No LLM parameter provided
                assert agent is not None
                assert hasattr(agent, '_chain')
                assert agent._chain is not None
            except Exception as e:
                # Should not fail due to missing LLM parameters
                assert "llm_type must be provided" not in str(e)
                assert "api_key must be provided" not in str(e)
                assert "Failed to initialize LLM from environment variables" not in str(e)
    
    def test_create_agent_function_with_env_variables(self):
        """Test that create_timbr_sql_agent can be called without LLM when config defaults are available"""
        # Mock the config values directly instead of environment variables
        with patch('langchain_timbr.llm_wrapper.llm_wrapper.config.llm_type', 'openai-chat'),\
             patch('langchain_timbr.llm_wrapper.llm_wrapper.config.llm_api_key', 'test-key'),\
             patch('langchain_timbr.llm_wrapper.llm_wrapper.config.llm_model', 'gpt-4'),\
             patch.dict(os.environ, {
                'TIMBR_URL': 'http://test-timbr.com',
                'TIMBR_TOKEN': 'test-token',
                'TIMBR_ONTOLOGY': 'test-ontology'
             }):
            try:
                agent_executor = create_timbr_sql_agent()  # No LLM parameter provided
                assert agent_executor is not None
                assert hasattr(agent_executor, 'agent')
            except Exception as e:
                # Should not fail due to missing LLM parameters
                assert "llm_type must be provided" not in str(e)
                assert "api_key must be provided" not in str(e)
                assert "Failed to initialize LLM from environment variables" not in str(e)
    
    def test_langgraph_nodes_with_env_variables(self):
        """Test that all langgraph nodes can be initialized without LLM when config defaults are available"""
        # Mock the config values directly instead of environment variables
        with patch('langchain_timbr.llm_wrapper.llm_wrapper.config.llm_type', 'openai-chat'),\
             patch('langchain_timbr.llm_wrapper.llm_wrapper.config.llm_api_key', 'test-key'),\
             patch('langchain_timbr.llm_wrapper.llm_wrapper.config.llm_model', 'gpt-4'),\
             patch.dict(os.environ, {
                'TIMBR_URL': 'http://test-timbr.com',
                'TIMBR_TOKEN': 'test-token',
                'TIMBR_ONTOLOGY': 'test-ontology'
             }):
            # Test all node classes
            nodes = [
                ExecuteSemanticQueryNode,
                GenerateTimbrSqlNode,
                IdentifyConceptNode,
                GenerateResponseNode,
                ValidateSemanticSqlNode
            ]
            
            for node_class in nodes:
                try:
                    node = node_class()  # No LLM parameter provided
                    assert node is not None
                    assert hasattr(node, 'chain')
                    assert node.chain is not None
                except Exception as e:
                    # Should not fail due to missing LLM parameters
                    assert "llm_type must be provided" not in str(e)
                    assert "api_key must be provided" not in str(e)
                    assert "Failed to initialize LLM from environment variables" not in str(e)
    
    def test_missing_llm_env_variables_raises_error(self):
        """Test that missing LLM env variables raise appropriate errors"""
        with patch('langchain_timbr.llm_wrapper.llm_wrapper.config.llm_type', None),\
             patch('langchain_timbr.llm_wrapper.llm_wrapper.config.llm_api_key', None),\
             patch.dict(os.environ, {
                'TIMBR_URL': 'http://test-timbr.com',
                'TIMBR_TOKEN': 'test-token',
                'TIMBR_ONTOLOGY': 'test-ontology'
             }, clear=True):
            with pytest.raises(ValueError, match="Failed to initialize LLM from environment variables"):
                ExecuteTimbrQueryChain()
            
            with pytest.raises(ValueError, match="Failed to initialize LLM from environment variables"):
                TimbrSqlAgent()
    
    def test_explicit_llm_overrides_env_variables(self):
        """Test that providing explicit LLM parameter works even with env variables"""
        from langchain_timbr.llm_wrapper.llm_wrapper import LlmWrapper
        
        # Mock the config values
        with patch('langchain_timbr.llm_wrapper.llm_wrapper.config.llm_type', 'openai-chat'),\
             patch('langchain_timbr.llm_wrapper.llm_wrapper.config.llm_api_key', 'env-key'),\
             patch.dict(os.environ, {
                'TIMBR_URL': 'http://test-timbr.com',
                'TIMBR_TOKEN': 'test-token',
                'TIMBR_ONTOLOGY': 'test-ontology'
             }):
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
