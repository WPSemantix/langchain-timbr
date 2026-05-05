"""Tests for LLM-based candidate extraction in the Technical Context Builder."""

import json
import pytest
from unittest.mock import MagicMock, patch

from langchain_timbr.technical_context.extraction.llm import (
    extract_candidates_with_llm,
    _build_candidate_extraction_prompt,
    _parse_candidates_response,
)


class TestExtractCandidatesWithLlm:
    """Tests for extract_candidates_with_llm()."""

    def test_returns_empty_when_llm_is_none(self):
        result = extract_candidates_with_llm("Show active orders", llm=None)
        assert result == []

    def test_returns_empty_when_no_question(self):
        llm = MagicMock()
        result = extract_candidates_with_llm("", llm=llm)
        assert result == []
        llm.invoke.assert_not_called()

    def test_returns_empty_when_whitespace_question(self):
        llm = MagicMock()
        result = extract_candidates_with_llm("   ", llm=llm)
        assert result == []

    def test_valid_json_response(self):
        llm = MagicMock()
        llm.invoke.return_value = json.dumps({
            "candidates": [
                {"literal": "Active", "synonyms": []}
            ]
        })
        result = extract_candidates_with_llm("Show active orders", llm=llm)
        assert result == ["Active"]

    def test_literals_and_synonyms_flattened(self):
        llm = MagicMock()
        llm.invoke.return_value = json.dumps({
            "candidates": [
                {"literal": "California", "synonyms": ["CA", "Calif"]},
                {"literal": "Acme Corp", "synonyms": ["Acme"]},
            ]
        })
        result = extract_candidates_with_llm("Acme Corp orders in California", llm=llm)
        assert result == ["California", "CA", "Calif", "Acme Corp", "Acme"]

    def test_deduplicates_across_candidates(self):
        llm = MagicMock()
        llm.invoke.return_value = json.dumps({
            "candidates": [
                {"literal": "Active", "synonyms": ["Active"]},
            ]
        })
        result = extract_candidates_with_llm("Show active", llm=llm)
        assert result == ["Active"]

    def test_timeout_returns_empty(self):
        import concurrent.futures
        llm = MagicMock()
        llm.invoke.side_effect = concurrent.futures.TimeoutError("timed out")
        result = extract_candidates_with_llm("Show active orders", llm=llm)
        assert result == []

    def test_invalid_json_returns_empty(self):
        llm = MagicMock()
        llm.invoke.return_value = "This is not JSON at all"
        result = extract_candidates_with_llm("Show active orders", llm=llm)
        assert result == []

    def test_llm_returns_none_returns_empty(self):
        llm = MagicMock()
        llm.invoke.return_value = None
        result = extract_candidates_with_llm("Show active orders", llm=llm)
        assert result == []

    def test_llm_exception_returns_empty(self):
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("LLM is down")
        result = extract_candidates_with_llm("Show active orders", llm=llm)
        assert result == []

    def test_response_with_code_fence(self):
        llm = MagicMock()
        llm.invoke.return_value = '```json\n{"candidates": [{"literal": "Active", "synonyms": []}]}\n```'
        result = extract_candidates_with_llm("Show active orders", llm=llm)
        assert result == ["Active"]

    def test_response_object_with_content_attr(self):
        llm = MagicMock()
        response = MagicMock()
        response.content = json.dumps({
            "candidates": [{"literal": "Active", "synonyms": []}]
        })
        llm.invoke.return_value = response
        result = extract_candidates_with_llm("Show active orders", llm=llm)
        assert result == ["Active"]

    def test_non_dict_entries_skipped(self):
        llm = MagicMock()
        llm.invoke.return_value = json.dumps({
            "candidates": ["not_a_dict", {"literal": "Valid", "synonyms": []}]
        })
        result = extract_candidates_with_llm("test", llm=llm)
        assert result == ["Valid"]

    def test_empty_candidates_list(self):
        llm = MagicMock()
        llm.invoke.return_value = json.dumps({"candidates": []})
        result = extract_candidates_with_llm("test", llm=llm)
        assert result == []


class TestBuildCandidateExtractionPrompt:
    """Tests for _build_candidate_extraction_prompt()."""

    def test_includes_question(self):
        prompt = _build_candidate_extraction_prompt("Show active orders")
        assert "Show active orders" in prompt

    def test_requests_json_format(self):
        prompt = _build_candidate_extraction_prompt("test")
        assert "JSON" in prompt

    def test_mentions_candidates(self):
        prompt = _build_candidate_extraction_prompt("test")
        assert "candidates" in prompt

    def test_includes_examples(self):
        prompt = _build_candidate_extraction_prompt("test")
        assert "places" in prompt
        assert "premium tier" in prompt


class TestParseCandidatesResponse:
    """Tests for _parse_candidates_response()."""

    def test_valid_json(self):
        result = _parse_candidates_response(
            '{"candidates": [{"literal": "Active", "synonyms": ["Enabled"]}]}'
        )
        assert result == ["Active", "Enabled"]

    def test_empty_response(self):
        assert _parse_candidates_response("") == []

    def test_non_dict_json(self):
        assert _parse_candidates_response('["list"]') == []

    def test_strips_code_fence(self):
        result = _parse_candidates_response(
            '```json\n{"candidates": [{"literal": "Active", "synonyms": []}]}\n```'
        )
        assert result == ["Active"]

    def test_missing_candidates_key(self):
        result = _parse_candidates_response('{"other": "data"}')
        assert result == []

    def test_candidates_not_list(self):
        result = _parse_candidates_response('{"candidates": "not_a_list"}')
        assert result == []

    def test_strips_whitespace_from_literals(self):
        result = _parse_candidates_response(
            '{"candidates": [{"literal": "  Active  ", "synonyms": []}]}'
        )
        assert result == ["Active"]

    def test_skips_empty_literals(self):
        result = _parse_candidates_response(
            '{"candidates": [{"literal": "", "synonyms": ["Syn"]}]}'
        )
        assert result == ["Syn"]

    def test_numeric_synonyms_skipped(self):
        result = _parse_candidates_response(
            '{"candidates": [{"literal": "Active", "synonyms": [123]}]}'
        )
        assert result == ["Active"]
