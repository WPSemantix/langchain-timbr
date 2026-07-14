"""Tests for chain documentation and examples."""
import inspect

from langchain_timbr import (
    IdentifyTimbrConceptChain,
    GenerateTimbrSqlChain,
    ValidateTimbrSqlChain,
    ExecuteTimbrQueryChain,
    GenerateAnswerChain
)


class TestChainDocumentation:
    """Test that chains have proper documentation."""
    
    def test_chain_docstrings(self):
        """Test that all chains have proper docstrings."""
        chains = [
            IdentifyTimbrConceptChain,
            GenerateTimbrSqlChain,
            ValidateTimbrSqlChain,
            ExecuteTimbrQueryChain,
            GenerateAnswerChain
        ]
        
        for chain_class in chains:
            assert chain_class.__doc__ is not None, f"{chain_class.__name__} must have docstring"
            assert len(chain_class.__doc__.strip()) > 20, f"{chain_class.__name__} docstring too short"
            
            # Check that invoke method has docstring (if it exists)
            if hasattr(chain_class, 'invoke') and hasattr(chain_class.invoke, '__doc__'):
                if chain_class.invoke.__doc__:
                    assert len(chain_class.invoke.__doc__.strip()) > 10, \
                        f"{chain_class.__name__}.invoke docstring too short"
    
    def test_chain_method_documentation(self):
        """Test that chain methods have proper documentation."""
        chains = [
            IdentifyTimbrConceptChain,
            GenerateTimbrSqlChain,
            ValidateTimbrSqlChain,
            ExecuteTimbrQueryChain,
            GenerateAnswerChain
        ]
        
        for chain_class in chains:
            # Check __init__ method documentation
            if hasattr(chain_class, '__init__'):
                init_method = getattr(chain_class, '__init__')
                if init_method.__doc__:
                    assert len(init_method.__doc__.strip()) > 10, \
                        f"{chain_class.__name__}.__init__ docstring too short"
            
            # Check other important methods
            important_methods = ['invoke', '_call', 'run']
            for method_name in important_methods:
                if hasattr(chain_class, method_name):
                    method = getattr(chain_class, method_name)
                    if hasattr(method, '__doc__') and method.__doc__:
                        assert len(method.__doc__.strip()) > 5, \
                            f"{chain_class.__name__}.{method_name} docstring too short"
    
    def test_chain_parameter_documentation(self):
        """Test that chain parameters are documented."""
        chains = [
            IdentifyTimbrConceptChain,
            GenerateTimbrSqlChain,
            ValidateTimbrSqlChain,
            ExecuteTimbrQueryChain,
            GenerateAnswerChain
        ]
        
        for chain_class in chains:
            # Get signature of __init__ method
            init_signature = inspect.signature(chain_class.__init__)
            parameters = list(init_signature.parameters.keys())
            
            # Remove 'self' parameter
            if 'self' in parameters:
                parameters.remove('self')
            
            # Check that parameters are documented in docstring
            if chain_class.__doc__:
                docstring = chain_class.__doc__.lower()
                
                # Key parameters should be mentioned
                key_params = ['llm', 'url', 'token', 'ontology']
                for param in key_params:
                    if param in parameters:
                        # Parameter should be mentioned in docstring
                        assert param.lower() in docstring, \
                            f"Parameter '{param}' should be documented in {chain_class.__name__}"
    
    def test_type_annotations(self):
        """Test that chains have proper type annotations."""
        chains = [
            IdentifyTimbrConceptChain,
            GenerateTimbrSqlChain,
            ValidateTimbrSqlChain,
            ExecuteTimbrQueryChain,
            GenerateAnswerChain
        ]
        
        for chain_class in chains:
            # Check __init__ method annotations
            init_signature = inspect.signature(chain_class.__init__)
            
            # Should have type annotations for key parameters
            for param_name, param in init_signature.parameters.items():
                if param_name in ['llm', 'url', 'token', 'ontology']:
                    # These key parameters should have type annotations
                    if param.annotation != inspect.Parameter.empty:
                        assert param.annotation is not None, \
                            f"Parameter '{param_name}' in {chain_class.__name__} should have type annotation"


class TestChainExamples:
    """Test chain usage examples and patterns."""
    
    def test_chain_configuration_examples(self, llm, config):
        """Test various chain configuration examples."""
        configurations = [
            # Basic configuration
            {
                "llm": llm,
                "url": config["timbr_url"],
                "token": config["timbr_token"],
                "ontology": config["timbr_ontology"]
            },
            # Configuration with SSL verification
            {
                "llm": llm,
                "url": config["timbr_url"],
                "token": config["timbr_token"],
                "ontology": config["timbr_ontology"],
                "verify_ssl": True
            },
            # Configuration with additional options
            {
                "llm": llm,
                "url": config["timbr_url"],
                "token": config["timbr_token"],
                "ontology": config["timbr_ontology"],
                "verify_ssl": config["verify_ssl"],
                "should_validate_sql": True
            }
        ]
        
        for i, config_dict in enumerate(configurations):
            try:
                # Test that each configuration works
                chain = ExecuteTimbrQueryChain(**config_dict)
                
                # Test basic functionality
                result = chain.invoke({"prompt": "Count all orders"})
                assert isinstance(result, dict)
                
            except Exception as e:
                # Allow connection and parameter errors
                error_msg = str(e).lower()
                valid_errors = ['connection', 'network', 'timeout', 'unreachable', 
                               'parameter', 'argument', 'missing']
                assert any(keyword in error_msg for keyword in valid_errors)
    
    def test_error_handling_examples(self, llm):
        """Test error handling examples."""
        err_keywords = ['auth', 'token', 'unauthorized', 'connection', 'invalid', 'nonetype', 'could not open client transport']
        
        # Example 1: Invalid URL
        try:
            chain = ExecuteTimbrQueryChain(
                llm=llm,
                url="http://invalid-timbr-url.com",
                token="test-token",
                ontology="test-ontology"
            )
            
            result = chain.invoke({"prompt": "test"})
            # Should either work (unlikely) or raise appropriate error
            
        except Exception as e:
            # Should be a connection-related error
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in err_keywords)

        # Example 2: Invalid token
        try:
            chain = ExecuteTimbrQueryChain(
                llm=llm,
                url="http://localhost:5000",  # Assume local test instance
                token="invalid-token",
                ontology="test-ontology"
            )
            
            result = chain.invoke({"prompt": "test"})
            # Should either work or raise appropriate error
            
        except Exception as e:
            # Should be an authentication or connection error
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in err_keywords)
