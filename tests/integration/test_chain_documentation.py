"""Integration tests for chain documentation usage examples.

These tests instantiate chains against a live Timbr backend (connection params
sourced from the environment / launch.json) and exercise the documented usage
patterns end-to-end. They were moved here from tests/standard because they
require a reachable backend rather than pure introspection.
"""

from langchain_timbr import (
    IdentifyTimbrConceptChain,
    GenerateTimbrSqlChain,
    ExecuteTimbrQueryChain,
)


class TestChainDocumentation:
    """Test that chains can be used as documented against a live backend."""

    def test_example_usage_completeness(self, llm, config):
        """Test that chains can be used as documented."""
        # Test that basic usage examples work
        chains_to_test = [
            (IdentifyTimbrConceptChain, {"prompt": "What are customers?"}),
            (GenerateTimbrSqlChain, {"prompt": "Get all customers"}),
            (ExecuteTimbrQueryChain, {"prompt": "Show 3 customers"}),
        ]

        for chain_class, test_input in chains_to_test:
            try:
                # Test instantiation as would be shown in examples
                chain = chain_class(
                    llm=llm,
                    url=config["timbr_url"],
                    token=config["timbr_token"],
                    ontology=config["timbr_ontology"],
                    verify_ssl=config["verify_ssl"]
                )

                # Test basic invoke usage
                result = chain.invoke(test_input)
                assert isinstance(result, dict), f"{chain_class.__name__} should return dict"

            except Exception as e:
                # Allow connection errors in test environment
                error_msg = str(e).lower()
                assert any(keyword in error_msg for keyword in
                          ['connection', 'network', 'timeout', 'unreachable']), \
                    f"Unexpected error in {chain_class.__name__}: {e}"


class TestChainExamples:
    """Test chain usage examples and patterns against a live backend."""

    def test_chain_composition_example(self, llm, config):
        """Test that chains can be composed as shown in examples."""
        try:
            # Example: Using multiple chains in sequence
            identify_chain = IdentifyTimbrConceptChain(
                llm=llm,
                url=config["timbr_url"],
                token=config["timbr_token"],
                ontology=config["timbr_ontology"],
                verify_ssl=config["verify_ssl"]
            )

            execute_chain = ExecuteTimbrQueryChain(
                llm=llm,
                url=config["timbr_url"],
                token=config["timbr_token"],
                ontology=config["timbr_ontology"],
                verify_ssl=config["verify_ssl"]
            )

            # Example composition pattern
            prompt = "Get customer information"

            # Step 1: Identify concept
            concept_result = identify_chain.invoke({"prompt": prompt})
            assert isinstance(concept_result, dict)

            # Step 2: Execute full query
            execution_result = execute_chain.invoke({"prompt": prompt})
            assert isinstance(execution_result, dict)

        except Exception as e:
            # Allow connection errors
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in
                      ['connection', 'network', 'timeout', 'unreachable'])
