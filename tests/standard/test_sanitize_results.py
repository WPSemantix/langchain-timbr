"""Tests verifying that each chain returns exactly its declared output_keys.

For every chain:
- The underlying utility functions are mocked so no real network calls are made.
- Inputs include a 'surprise_key' that is NOT in output_keys, simulating a caller
  that passes extra kwargs through the chain pipeline.
- Assertions confirm the extra key is stripped and the declared output_keys are
  present with the expected values from the mocked responses.
"""
import pytest
from unittest.mock import Mock, patch
from langchain_timbr.utils._base_chain import _init_chain_context

from langchain_timbr.langchain.execute_timbr_query_chain import ExecuteTimbrQueryChain
from langchain_timbr.langchain.generate_answer_chain import GenerateAnswerChain
from langchain_timbr.langchain.generate_timbr_sql_chain import GenerateTimbrSqlChain
from langchain_timbr.langchain.identify_concept_chain import IdentifyTimbrConceptChain
from langchain_timbr.langchain.validate_timbr_sql_chain import ValidateTimbrSqlChain

URL = "http://test-timbr"
TOKEN = "test-token"


class TestGenerateTimbrSqlChainOutput:

    @patch("langchain_timbr.langchain.generate_timbr_sql_chain.generate_sql")
    def test_returns_only_output_keys(self, mock_generate_sql):
        mock_generate_sql.return_value = {
            "sql": "SELECT * FROM cities",
            "ontology": "my_ontology",
            "schema": "dtimbr",
            "concept": "city",
            "is_sql_valid": True,
            "error": None,
            "identify_concept_reason": "matched city concept",
            "generate_sql_reason": "straightforward query",
            "reasoning_status": None,
            "usage_metadata": {"input_tokens": 10, "output_tokens": 5},
            "internal_llm_debug": "should_not_appear",  # extra key the util may return
        }
        chain = GenerateTimbrSqlChain(llm=Mock(), url=URL, token=TOKEN)
        chain._received_chain_context = _init_chain_context(None)
        inputs = {"prompt": "list all cities", "surprise_key": "strip_me"}

        result = chain._call(inputs)

        assert set(result.keys()) == set(chain.output_keys)
        assert "surprise_key" not in result
        assert "internal_llm_debug" not in result
        assert result["sql"] == "SELECT * FROM cities"
        assert result["ontology"] == "my_ontology"
        assert result["is_sql_valid"] is True
        assert result["prompt"] == "list all cities"


class TestGenerateAnswerChainOutput:

    @patch("langchain_timbr.langchain.generate_answer_chain.answer_question")
    def test_returns_only_output_keys(self, mock_answer_question):
        mock_answer_question.return_value = {
            "answer": "Paris is the capital of France.",
            "usage_metadata": {"input_tokens": 20, "output_tokens": 8},
            "raw_response": "should_not_appear",  # extra key the util may return
        }
        chain = GenerateAnswerChain(llm=Mock(), url=URL, token=TOKEN)
        # Initialize chain context so _call doesn't crash when tracking duration/tokens
        chain._received_chain_context = _init_chain_context(None)
        inputs = {
            "prompt": "What is the capital of France?",
            "rows": [{"city": "Paris"}],
            "surprise_key": "strip_me",
        }

        result = chain._call(inputs)

        assert set(result.keys()) == set(chain.output_keys)
        assert "surprise_key" not in result
        assert "raw_response" not in result
        assert result["answer"] == "Paris is the capital of France."
        assert result["prompt"] == "What is the capital of France?"
        assert result["rows"] == [{"city": "Paris"}]
        # New output keys should be present (None when not provided via execute chain)
        assert "conversation_id" in result
        assert "execute_timbr_usage_metadata" in result
        assert "ontology" in result


class TestIdentifyTimbrConceptChainOutput:

    @patch("langchain_timbr.langchain.identify_concept_chain.determine_concept")
    def test_returns_only_output_keys(self, mock_determine_concept):
        # determine_concept may return extra internal fields via **res spread —
        # this is the primary leak scenario sanitize_results was added to fix.
        mock_determine_concept.return_value = {
            "ontology": "my_ontology",
            "schema": "dtimbr",
            "concept": "city",
            "concept_metadata": {"description": "A city entity"},
            "identify_concept_reason": "city matched the question intent",
            "usage_metadata": {"input_tokens": 15, "output_tokens": 6},
            "internal_candidates": ["city", "country"],  # extra key that should be stripped
        }
        chain = IdentifyTimbrConceptChain(llm=Mock(), url=URL, token=TOKEN)
        chain._received_chain_context = _init_chain_context(None)
        inputs = {"prompt": "find cities in Europe", "surprise_key": "strip_me"}

        result = chain._call(inputs)

        assert set(result.keys()) == set(chain.output_keys)
        assert "surprise_key" not in result
        assert "internal_candidates" not in result
        assert result["concept"] == "city"
        assert result["ontology"] == "my_ontology"
        assert result["identify_concept_reason"] == "city matched the question intent"
        assert result["prompt"] == "find cities in Europe"


class TestValidateTimbrSqlChainOutput:

    @patch("langchain_timbr.langchain.validate_timbr_sql_chain.validate_sql")
    def test_returns_only_output_keys_when_sql_is_valid(self, mock_validate_sql):
        mock_validate_sql.return_value = (True, None, "SELECT * FROM cities")
        chain = ValidateTimbrSqlChain(llm=Mock(), url=URL, token=TOKEN)
        chain._received_chain_context = _init_chain_context(None)
        inputs = {
            "prompt": "list cities",
            "sql": "SELECT * FROM cities",
            "surprise_key": "strip_me",
        }

        result = chain._call(inputs)

        assert set(result.keys()) == set(chain.output_keys)
        assert "surprise_key" not in result
        assert result["sql"] == "SELECT * FROM cities"
        assert result["is_sql_valid"] is True
        assert result["error"] is None
        assert result["prompt"] == "list cities"

    @patch("langchain_timbr.langchain.validate_timbr_sql_chain.generate_sql")
    @patch("langchain_timbr.langchain.validate_timbr_sql_chain.validate_sql")
    def test_returns_only_output_keys_when_sql_is_invalid(self, mock_validate_sql, mock_generate_sql):
        mock_validate_sql.return_value = (False, "syntax error", "BAD SQL")
        mock_generate_sql.return_value = {
            "sql": "SELECT * FROM cities",
            "schema": "dtimbr",
            "concept": "city",
            "is_sql_valid": True,
            "error": None,
            "reasoning_status": None,
            "identify_concept_reason": None,
            "generate_sql_reason": "regenerated after syntax error",
            "usage_metadata": {},
            "internal_debug": "should_not_appear",
        }
        chain = ValidateTimbrSqlChain(llm=Mock(), url=URL, token=TOKEN)
        chain._received_chain_context = _init_chain_context(None)
        inputs = {
            "prompt": "list cities",
            "sql": "BAD SQL",
            "surprise_key": "strip_me",
        }

        result = chain._call(inputs)

        assert set(result.keys()) == set(chain.output_keys)
        assert "surprise_key" not in result
        assert "internal_debug" not in result
        assert result["sql"] == "SELECT * FROM cities"
        assert result["is_sql_valid"] is True


class TestExecuteTimbrQueryChainOutput:

    @patch("langchain_timbr.langchain.execute_timbr_query_chain.run_query")
    def test_returns_only_output_keys_with_provided_sql(self, mock_run_query):
        mock_run_query.return_value = [{"city": "Paris"}, {"city": "Berlin"}]
        chain = ExecuteTimbrQueryChain(
            llm=Mock(),
            url=URL,
            token=TOKEN,
            should_validate_sql=False,
            no_results_max_retries=0,
        )
        chain._received_chain_context = _init_chain_context(None)
        # Pass sql directly so no LLM generation is needed.
        # Also include an extra key that must not appear in the output.
        inputs = {
            "prompt": "list cities",
            "sql": "SELECT * FROM cities",
            "surprise_key": "strip_me",
        }

        result = chain._call(inputs)

        assert set(result.keys()) == set(chain.output_keys)
        assert "surprise_key" not in result
        assert result["rows"] == [{"city": "Paris"}, {"city": "Berlin"}]
        assert result["sql"] == "SELECT * FROM cities"
        assert result["error"] is None
        assert result["prompt"] == "list cities"
