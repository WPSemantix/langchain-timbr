"""Tests for LLM-based candidate extraction in the Technical Context Builder."""

import json
import pytest
from unittest.mock import MagicMock, patch

from langchain_timbr.technical_context.extraction.llm import (
    extract_candidates_with_llm,
    _build_candidate_extraction_prompt,
    _parse_candidates_response,
    _extraction_cache_clear,
)
from langchain_timbr.technical_context.extraction import llm as _llm_mod


@pytest.fixture(autouse=True)
def _clear_extraction_cache():
    """Drop the in-process LRU cache before every test so MagicMock id() reuse
    or test ordering can't leak state across tests."""
    _extraction_cache_clear()
    yield
    _extraction_cache_clear()


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


class TestExtractionCache:
    """In-process LRU cache over ``(question, id(llm))``. Removes the duplicate
    ``extract_candidates_with_llm`` call the dynamic-metadata-context
    ``tc_topup`` pass otherwise triggers."""

    def _payload(self, *, literal: str = "metal", synonyms=("metallic",)) -> str:
        return json.dumps({
            "candidates": [{"literal": literal, "synonyms": list(synonyms)}]
        })

    def test_same_question_invokes_llm_once(self):
        llm = MagicMock()
        llm.invoke.return_value = self._payload()
        first = extract_candidates_with_llm("Which metal material", llm=llm)
        second = extract_candidates_with_llm("Which metal material", llm=llm)
        assert first == ["metal", "metallic"]
        assert second == ["metal", "metallic"]
        assert llm.invoke.call_count == 1

    def test_different_question_invokes_llm_each_time(self):
        llm = MagicMock()
        llm.invoke.side_effect = [
            self._payload(literal="metal"),
            self._payload(literal="wood", synonyms=("wooden",)),
        ]
        a = extract_candidates_with_llm("Which metal material", llm=llm)
        b = extract_candidates_with_llm("Which wooden parts", llm=llm)
        assert a == ["metal", "metallic"]
        assert b == ["wood", "wooden"]
        assert llm.invoke.call_count == 2

    def test_question_strip_normalizes_key(self):
        """``question.strip()`` is the cache key — surrounding whitespace
        must not multiply the entry."""
        llm = MagicMock()
        llm.invoke.return_value = self._payload()
        extract_candidates_with_llm("Which metal material", llm=llm)
        extract_candidates_with_llm("  Which metal material  ", llm=llm)
        assert llm.invoke.call_count == 1

    def test_different_llm_instances_do_not_share_cache(self):
        """``id(llm)`` is part of the key so two LLMs in the same process
        don't serve each other's answers."""
        llm_a = MagicMock()
        llm_b = MagicMock()
        llm_a.invoke.return_value = self._payload(literal="A_metal")
        llm_b.invoke.return_value = self._payload(literal="B_metal")
        ra = extract_candidates_with_llm("same question", llm=llm_a)
        rb = extract_candidates_with_llm("same question", llm=llm_b)
        assert ra[0] == "A_metal"
        assert rb[0] == "B_metal"
        assert llm_a.invoke.call_count == 1
        assert llm_b.invoke.call_count == 1

    def test_error_is_not_cached(self):
        """First call raises; second call must re-invoke (no poisoned slot)."""
        llm = MagicMock()
        llm.invoke.side_effect = [
            RuntimeError("transient"),
            self._payload(),
        ]
        first = extract_candidates_with_llm("Which metal material", llm=llm)
        second = extract_candidates_with_llm("Which metal material", llm=llm)
        assert first == []                       # error swallowed → []
        assert second == ["metal", "metallic"]   # retry succeeds
        assert llm.invoke.call_count == 2

    def test_empty_success_is_cached(self):
        """An empty parsed list is a real answer; subsequent calls should
        NOT re-invoke (distinct from the error path above)."""
        llm = MagicMock()
        llm.invoke.return_value = '{"candidates": []}'
        a = extract_candidates_with_llm("noise question", llm=llm)
        b = extract_candidates_with_llm("noise question", llm=llm)
        assert a == [] and b == []
        assert llm.invoke.call_count == 1

    def test_cached_result_is_a_defensive_copy(self):
        """Mutating the returned list must not pollute the cached entry."""
        llm = MagicMock()
        llm.invoke.return_value = self._payload()
        first = extract_candidates_with_llm("Q", llm=llm)
        first.append("MUTATION")
        second = extract_candidates_with_llm("Q", llm=llm)
        assert "MUTATION" not in second
        assert second == ["metal", "metallic"]

    def test_lru_eviction_at_maxsize(self):
        """Fill the cache past ``_CACHE_MAXSIZE``; the LRU loser (the first
        question) must be evicted so re-calling it triggers a fresh
        invoke."""
        cap = _llm_mod._CACHE_MAXSIZE
        llm = MagicMock()
        llm.invoke.return_value = self._payload()
        # Warm the cache with ``cap`` distinct questions.
        for i in range(cap):
            extract_candidates_with_llm(f"Q{i}", llm=llm)
        assert llm.invoke.call_count == cap
        # One more push the FIRST entry (Q0) out — LRU loser.
        extract_candidates_with_llm("Q_overflow", llm=llm)
        assert llm.invoke.call_count == cap + 1
        # Re-call Q0 → it should miss the cache and re-invoke.
        extract_candidates_with_llm("Q0", llm=llm)
        assert llm.invoke.call_count == cap + 2
        # The MOST recently inserted overflow entry must still be hot.
        extract_candidates_with_llm("Q_overflow", llm=llm)
        assert llm.invoke.call_count == cap + 2
