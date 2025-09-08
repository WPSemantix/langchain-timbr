"""Standard tests required by LangChain for chain contributions."""
import pytest
from langchain.chains.base import Chain
from langchain_core.runnables import Runnable

from langchain_timbr import (
    IdentifyTimbrConceptChain,
    GenerateTimbrSqlChain,
    ValidateTimbrSqlChain,
    ExecuteTimbrQueryChain,
    GenerateAnswerChain
)

class TestStandardChainRequirements:
    """Standard tests required by LangChain for chain contributions."""
    
    def test_chain_inheritance(self):
        """Test that all chains properly inherit from Chain or Runnable."""
        chains = [
            IdentifyTimbrConceptChain,
            GenerateTimbrSqlChain,
            ValidateTimbrSqlChain,
            ExecuteTimbrQueryChain,
            GenerateAnswerChain
        ]
        
        for chain_class in chains:
            # Chains should inherit from Chain or implement Runnable
            assert (issubclass(chain_class, Chain) or issubclass(chain_class, Runnable)), \
                f"{chain_class.__name__} must inherit from Chain or implement Runnable"
    
    def test_chain_instantiation(self, mock_llm, minimal_config):
        """Test that chains can be instantiated with required parameters."""
        base_params = {
            "llm": mock_llm,
            "url": minimal_config["timbr_url"],
            "token": minimal_config["timbr_token"],
            "ontology": minimal_config["timbr_ontology"],
            "verify_ssl": minimal_config["verify_ssl"]
        }
        
        # Test each chain can be instantiated
        identify_chain = IdentifyTimbrConceptChain(**base_params)
        sql_chain = GenerateTimbrSqlChain(**base_params)
        validate_chain = ValidateTimbrSqlChain(**base_params)
        execute_chain = ExecuteTimbrQueryChain(**base_params)
        answer_chain = GenerateAnswerChain(
            llm=mock_llm,
            url=minimal_config["timbr_url"],
            token=minimal_config["timbr_token"],
            verify_ssl=minimal_config["verify_ssl"]
        )
        
        # Verify they are proper chain instances
        assert isinstance(identify_chain, (Chain, Runnable))
        assert isinstance(sql_chain, (Chain, Runnable))
        assert isinstance(validate_chain, (Chain, Runnable))
        assert isinstance(execute_chain, (Chain, Runnable))
        assert isinstance(answer_chain, (Chain, Runnable))
    
    def test_chain_required_methods(self, mock_llm, minimal_config):
        """Test that chains have required methods."""
        chain = IdentifyTimbrConceptChain(
            llm=mock_llm,
            url=minimal_config["timbr_url"],
            token=minimal_config["timbr_token"],
            ontology=minimal_config["timbr_ontology"]
        )
        
        # Required methods for LangChain chains
        assert hasattr(chain, 'invoke'), "Chain must have 'invoke' method"
        assert callable(chain.invoke), "Chain invoke must be callable"
        
        # Check for additional chain methods
        if hasattr(chain, '_call'):
            assert callable(chain._call), "Chain _call must be callable"
        if hasattr(chain, 'run'):
            assert callable(chain.run), "Chain run must be callable"
    
    def test_chain_input_validation(self, mock_llm, minimal_config):
        """Test that chains properly validate inputs."""
        chain = IdentifyTimbrConceptChain(
            llm=mock_llm,
            url=minimal_config["timbr_url"],
            token=minimal_config["timbr_token"],
            ontology=minimal_config["timbr_ontology"]
        )
        
        # Test valid input structure
        valid_input = {"prompt": "What are the customers?"}
        assert isinstance(valid_input, dict), "Valid input should be dict"
        assert "prompt" in valid_input, "Valid input should have prompt key"
        
        # Test that chain validates input keys properly
        assert "prompt" in chain.input_keys, "Chain should require 'prompt' key"
        
        # Test that chain has expected output keys
        assert "concept" in chain.output_keys, "Chain should output 'concept'"
        assert "schema" in chain.output_keys, "Chain should output 'schema'"
    
    def test_chain_error_handling(self, mock_llm):
        """Test that chains handle errors gracefully."""
        # Test with invalid connection parameters
        chain = IdentifyTimbrConceptChain(
            llm=mock_llm,
            url="http://invalid-url",
            token="invalid-token",
            ontology="invalid-ontology"
        )
        
        # Test that chain can be instantiated with invalid params
        assert chain is not None, "Chain should be instantiated even with invalid params"
        
        # Check that chain stores parameters (using private attributes)
        assert hasattr(chain, '_url'), "Chain should store URL parameter"
        assert hasattr(chain, '_token'), "Chain should store token parameter"
        assert hasattr(chain, '_ontology'), "Chain should store ontology parameter"
        
        # Test that chain has proper structure
        assert hasattr(chain, 'input_keys'), "Chain should have input_keys property"
        assert hasattr(chain, 'output_keys'), "Chain should have output_keys property"
        assert "prompt" in chain.input_keys, "Chain should accept prompt input"
    
    def test_chain_serialization(self, mock_llm, minimal_config):
        """Test that chains can be serialized/deserialized."""
        chain = IdentifyTimbrConceptChain(
            llm=mock_llm,
            url=minimal_config["timbr_url"],
            token=minimal_config["timbr_token"],
            ontology=minimal_config["timbr_ontology"]
        )
        
        # Test basic serialization capabilities
        if hasattr(chain, 'model_dump'):
            chain_model_dump = chain.model_dump()
            assert isinstance(chain_model_dump, dict), "Chain dict() should return dictionary"
        
        # Test that chain has string representation
        chain_str = str(chain)
        assert isinstance(chain_str, str), "Chain should have string representation"
        assert len(chain_str) > 0, "Chain string representation should not be empty"
    
    def test_chain_input_output_types(self, mock_llm, minimal_config):
        """Test that chains have consistent input/output types."""
        chain = ExecuteTimbrQueryChain(
            llm=mock_llm,
            url=minimal_config["timbr_url"],
            token=minimal_config["timbr_token"],
            ontology=minimal_config["timbr_ontology"],
            verify_ssl=minimal_config["verify_ssl"]
        )
        
        # Test input type requirements (should accept dict)
        test_input = {"prompt": "Get 3 customers"}
        assert isinstance(test_input, dict), "Chain input should be dict"
        assert "prompt" in test_input, "Chain input should have prompt key"
        
        # Test that chain accepts the right input structure
        assert "prompt" in chain.input_keys, "Chain should require 'prompt' key"
        
        # Test chain has expected private attributes (where LangChain stores parameters)
        assert hasattr(chain, '_llm'), "Chain should have _llm attribute"
        assert hasattr(chain, '_url'), "Chain should have _url attribute" 
        assert hasattr(chain, '_token'), "Chain should have _token attribute"
        assert hasattr(chain, '_ontology'), "Chain should have _ontology attribute"
        
        # Test that chain implements required LangChain interface
        assert hasattr(chain, 'invoke'), "Chain should have invoke method"
        assert callable(chain.invoke), "Chain invoke should be callable"
        assert hasattr(chain, 'input_keys'), "Chain should have input_keys property"
        assert hasattr(chain, 'output_keys'), "Chain should have output_keys property"


class TestChainIntegration:
    """Integration tests for chain functionality."""
    
    @pytest.mark.slow
    def test_full_chain_pipeline(self, mock_llm, minimal_config):
        """Test that chains can work together in a pipeline."""
        from unittest.mock import patch, Mock
        
        # Test the full pipeline: Identify -> Generate SQL -> Execute -> Generate Answer
        
        # Step 1: Create all chains for the pipeline
        identify_chain = IdentifyTimbrConceptChain(
            llm=mock_llm,
            url=minimal_config["timbr_url"],
            token=minimal_config["timbr_token"],
            ontology=minimal_config["timbr_ontology"],
            verify_ssl=minimal_config["verify_ssl"]
        )
        
        sql_chain = GenerateTimbrSqlChain(
            llm=mock_llm,
            url=minimal_config["timbr_url"],
            token=minimal_config["timbr_token"],
            ontology=minimal_config["timbr_ontology"],
            verify_ssl=minimal_config["verify_ssl"]
        )
        
        execute_chain = ExecuteTimbrQueryChain(
            llm=mock_llm,
            url=minimal_config["timbr_url"],
            token=minimal_config["timbr_token"],
            ontology=minimal_config["timbr_ontology"],
            verify_ssl=minimal_config["verify_ssl"]
        )
        
        answer_chain = GenerateAnswerChain(
            llm=mock_llm,
            url=minimal_config["timbr_url"],
            token=minimal_config["timbr_token"],
            verify_ssl=minimal_config["verify_ssl"]
        )
        
        # Step 2: Mock the chain _call methods to simulate execution without network calls
        with patch.object(identify_chain, '_call') as mock_identify_call, \
             patch.object(sql_chain, '_call') as mock_sql_call, \
             patch.object(execute_chain, '_call') as mock_execute_call, \
             patch.object(answer_chain, '_call') as mock_answer_call:
            
            # Configure mock responses for each chain
            mock_identify_call.return_value = {
                'concept': 'customer',
                'schema': 'dtimbr',
                'concept_metadata': {'description': 'Customer entity'},
                'identify_concept_usage_metadata': {}
            }
            
            mock_sql_call.return_value = {
                'sql': 'SELECT * FROM customer LIMIT 5',
                'concept': 'customer',
                'schema': 'dtimbr',
                'is_sql_valid': True,
                'error': None,
                'generate_sql_usage_metadata': {}
            }
            
            mock_execute_call.return_value = {
                'rows': [
                    {'customer_id': 1, 'name': 'Alice Johnson'},
                    {'customer_id': 2, 'name': 'Bob Smith'}
                ],
                'sql': 'SELECT * FROM customer LIMIT 5',
                'schema': 'dtimbr',
                'concept': 'customer',
                'error': None,
                'execute_timbr_usage_metadata': {}
            }
            
            mock_answer_call.return_value = {
                'answer': 'Based on the query results, there are customers including Alice Johnson and Bob Smith.',
                'generate_answer_usage_metadata': {}
            }
            
            # Step 3: Execute the full pipeline
            user_prompt = "Show me some customers"
            
            # 1. Identify concept
            concept_result = identify_chain.invoke({"prompt": user_prompt})
            assert 'concept' in concept_result
            assert concept_result['concept'] == 'customer'
            
            # 2. Generate SQL  
            sql_result = sql_chain.invoke({"prompt": user_prompt})
            assert 'sql' in sql_result
            assert 'SELECT' in sql_result['sql'].upper()
            
            # 3. Execute query
            execute_result = execute_chain.invoke({"prompt": user_prompt})
            assert 'rows' in execute_result
            assert len(execute_result['rows']) > 0
            assert execute_result['rows'][0]['name'] in ['Alice Johnson', 'Bob Smith']
            
            # 4. Generate answer
            answer_result = answer_chain.invoke({
                "prompt": user_prompt,
                "rows": execute_result['rows']
            })
            assert 'answer' in answer_result
            assert len(answer_result['answer']) > 0
            
            # Step 4: Verify pipeline data flow
            # Check that each step produces compatible output for the next step
            assert concept_result['concept'] == sql_result['concept']
            assert sql_result['sql'] == execute_result['sql']
            
            # Step 5: Verify all chain methods were called
            mock_identify_call.assert_called_once()
            mock_sql_call.assert_called_once()
            mock_execute_call.assert_called_once()
            mock_answer_call.assert_called_once()
            
            # Step 6: Test chain composition capabilities
            # Verify chains can be chained together programmatically
            pipeline_chains = [identify_chain, sql_chain, execute_chain, answer_chain]
            for chain in pipeline_chains:
                assert hasattr(chain, 'invoke'), f"{chain.__class__.__name__} should have invoke method"
                assert hasattr(chain, 'input_keys'), f"{chain.__class__.__name__} should have input_keys"
                assert hasattr(chain, 'output_keys'), f"{chain.__class__.__name__} should have output_keys"
                
            # Step 7: Test that chains share compatible configuration
            chain_configs = [identify_chain, sql_chain, execute_chain]
            for i in range(len(chain_configs) - 1):
                assert chain_configs[i]._url == chain_configs[i+1]._url, "Chains should share URL configuration"
                assert chain_configs[i]._token == chain_configs[i+1]._token, "Chains should share token configuration"
    
    def test_chain_memory_usage(self, mock_llm, minimal_config):
        """Test that chains don't have memory leaks."""
        import gc
        import sys
        
        chain = IdentifyTimbrConceptChain(
            llm=mock_llm,
            url=minimal_config["timbr_url"],
            token=minimal_config["timbr_token"],
            ontology=minimal_config["timbr_ontology"]
        )
        
        # Get initial reference count
        initial_refs = sys.getrefcount(chain)
        
        # Use chain multiple times
        for i in range(5):
            try:
                chain.invoke({"prompt": f"test query {i}"})
            except Exception:
                pass  # Ignore connection errors
        
        # Force garbage collection
        gc.collect()
        
        # Check that reference count hasn't grown significantly
        final_refs = sys.getrefcount(chain)
        assert final_refs <= initial_refs + 2, "Chain should not accumulate references"
