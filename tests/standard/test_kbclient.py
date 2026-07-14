"""Standard (backend-free) unit tests for langchain_timbr.kbclient.KBClient.

Every test mocks the two external touch points:
  * ``langchain_timbr.kbclient.requests`` — the HTTP transport.
  * ``langchain_timbr.kbclient.run_query`` — the SQL executor used for cache
    invalidation and KB resolution.

No live Timbr backend is contacted.
"""
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
import requests

from langchain_timbr.kbclient import (
    KBAuthError,
    KBBadRequestError,
    KBClient,
    KBClientError,
    KBMatch,
    KBOverloadError,
    KBSearchResult,
    KBServerError,
    KBTimeoutError,
)

BASE_URL = "https://timbr.example.com"
TOKEN = "tk_test_token"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_response(status_code, body):
    """Build a fake requests.Response-like object."""
    resp = MagicMock()
    resp.status_code = status_code
    if isinstance(body, (dict, list)):
        resp.json.return_value = body
        resp.text = json.dumps(body)
    else:
        resp.json.side_effect = ValueError("no json")
        resp.text = body
    return resp


def _high_match_body():
    return {
        "match_level": "high",
        "matches": [
            {
                "knowledge_base": "finance_kb",
                "example_name": "regional_revenue_last_quarter",
                "question": "What was total revenue by region last quarter?",
                "query": "SELECT region, SUM(revenue) FROM dtimbr.order GROUP BY region",
                "instructions": "Use fiscal quarter.",
                "validate_sql": 1,
                "confidence": 0.964215,
                "changed_on": "2026-06-14T10:22:00Z",
            }
        ],
        "per_kb_max_changed_on": {"finance_kb": "2026-06-14T10:22:00Z"},
        "latency_ms": 42.31,
    }


def _none_match_body():
    return {
        "match_level": "none",
        "matches": [],
        "per_kb_max_changed_on": {},
        "latency_ms": 18.77,
    }


def _client(**kwargs):
    kwargs.setdefault("cache_ttl_seconds", None)  # caching off unless asked
    return KBClient(BASE_URL, TOKEN, **kwargs)


# --------------------------------------------------------------------------- #
# Success parsing / match levels
# --------------------------------------------------------------------------- #
class TestSearchSuccess:
    @patch("langchain_timbr.kbclient.requests")
    def test_high_match_parsed(self, mock_requests):
        session = mock_requests.Session.return_value
        session.post.return_value = _make_response(200, _high_match_body())
        kb = _client()

        result = kb.search("which region had highest revenue?", ["finance_kb"])

        assert isinstance(result, KBSearchResult)
        assert result.match_level == "high"
        assert result.is_high_confidence is True
        assert result.has_matches is True
        top = result.top
        assert isinstance(top, KBMatch)
        assert top.knowledge_base == "finance_kb"
        assert top.has_query is True
        assert top.has_instructions is True
        assert top.validate_sql == 1
        assert abs(top.confidence - 0.964215) < 1e-9
        assert top.changed_on == datetime(2026, 6, 14, 10, 22, tzinfo=timezone.utc)
        assert result.per_kb_max_changed_on["finance_kb"].tzinfo is not None
        assert result.latency_ms == 42.31

    @patch("langchain_timbr.kbclient.requests")
    def test_none_match(self, mock_requests):
        session = mock_requests.Session.return_value
        session.post.return_value = _make_response(200, _none_match_body())
        kb = _client()

        result = kb.search("unrelated question", ["finance_kb"])

        assert result.match_level == "none"
        assert result.has_matches is False
        assert result.top is None
        assert result.per_kb_max_changed_on == {}

    @patch("langchain_timbr.kbclient.requests")
    def test_null_fields_and_medium(self, mock_requests):
        body = {
            "match_level": "medium",
            "matches": [
                {
                    "knowledge_base": "kb1",
                    "example_name": "ex1",
                    "question": "q",
                    "query": None,
                    "instructions": None,
                    "validate_sql": 0,
                    "confidence": 0.55,
                    "changed_on": None,
                }
            ],
            "per_kb_max_changed_on": {},
            "latency_ms": 5.0,
        }
        session = mock_requests.Session.return_value
        session.post.return_value = _make_response(200, body)
        kb = _client()

        result = kb.search("q", ["kb1"])

        assert result.match_level == "medium"
        assert result.is_high_confidence is False
        m = result.top
        assert m.query is None and m.instructions is None
        assert m.has_query is False and m.has_instructions is False
        assert m.changed_on is None

    @patch("langchain_timbr.kbclient.requests")
    def test_post_body_contains_thresholds_and_top_k(self, mock_requests):
        session = mock_requests.Session.return_value
        session.post.return_value = _make_response(200, _none_match_body())
        kb = _client()

        kb.search("q", ["kb1"], top_k=3, high_threshold=0.8, medium_threshold=0.5)

        body = session.post.call_args.kwargs["json"]
        assert body["question"] == "q"
        assert body["knowledge_bases"] == ["kb1"]
        assert body["top_k"] == 3
        assert body["high_threshold"] == 0.8
        assert body["medium_threshold"] == 0.5

    @patch("langchain_timbr.kbclient.requests")
    def test_api_key_header(self, mock_requests):
        session = mock_requests.Session.return_value
        session.post.return_value = _make_response(200, _none_match_body())
        kb = _client()

        kb.search("q", ["kb1"])

        headers = session.post.call_args.kwargs["headers"]
        assert headers["x-api-key"] == TOKEN
        assert "x-jwt-token" not in headers

    @patch("langchain_timbr.kbclient.requests")
    def test_jwt_headers(self, mock_requests):
        session = mock_requests.Session.return_value
        session.post.return_value = _make_response(200, _none_match_body())
        kb = _client(is_jwt=True, jwt_tenant_id="tenant-1")

        kb.search("q", ["kb1"])

        headers = session.post.call_args.kwargs["headers"]
        assert headers["x-jwt-token"] == TOKEN
        assert headers["x-jwt-tenant-id"] == "tenant-1"
        assert "x-api-key" not in headers


# --------------------------------------------------------------------------- #
# Error mapping
# --------------------------------------------------------------------------- #
class TestErrorMapping:
    @pytest.mark.parametrize(
        "status, exc",
        [
            (400, KBBadRequestError),
            (401, KBAuthError),
            (403, KBAuthError),
            (429, KBOverloadError),
            (500, KBServerError),
        ],
    )
    @patch("langchain_timbr.kbclient.requests")
    def test_status_maps_to_exception(self, mock_requests, status, exc):
        body = {"status": "error", "data": f"error {status}"}
        session = mock_requests.Session.return_value
        session.post.return_value = _make_response(status, body)
        kb = _client()

        with pytest.raises(exc) as ei:
            kb.search("q", ["kb1"])
        assert ei.value.status_code == status
        assert ei.value.message == f"error {status}"

    @patch("langchain_timbr.kbclient.requests")
    def test_timeout_maps_to_kbtimeout(self, mock_requests):
        session = mock_requests.Session.return_value
        session.post.side_effect = requests.exceptions.Timeout("read timed out")
        kb = _client()

        with pytest.raises(KBTimeoutError):
            kb.search("q", ["kb1"])

    @patch("langchain_timbr.kbclient.requests")
    def test_connection_error_exhausts_retries_then_timeout(self, mock_requests):
        session = mock_requests.Session.return_value
        session.post.side_effect = requests.exceptions.ConnectionError("refused")
        kb = _client(max_retries=2, retry_backoff_seconds=0)

        with pytest.raises(KBTimeoutError):
            kb.search("q", ["kb1"])
        # initial attempt + 2 retries
        assert session.post.call_count == 3


# --------------------------------------------------------------------------- #
# Retry behavior
# --------------------------------------------------------------------------- #
class TestRetry:
    @patch("langchain_timbr.kbclient.requests")
    def test_retries_on_503_then_succeeds(self, mock_requests):
        session = mock_requests.Session.return_value
        session.post.side_effect = [
            _make_response(503, {"status": "error", "data": "unavailable"}),
            _make_response(200, _none_match_body()),
        ]
        kb = _client(max_retries=2, retry_backoff_seconds=0)

        result = kb.search("q", ["kb1"])

        assert result.match_level == "none"
        assert session.post.call_count == 2

    @patch("langchain_timbr.kbclient.requests")
    def test_429_not_retried(self, mock_requests):
        session = mock_requests.Session.return_value
        session.post.return_value = _make_response(429, {"status": "error", "data": "queue timeout"})
        kb = _client(max_retries=3, retry_backoff_seconds=0)

        with pytest.raises(KBOverloadError):
            kb.search("q", ["kb1"])
        assert session.post.call_count == 1

    @patch("langchain_timbr.kbclient.requests")
    def test_400_not_retried(self, mock_requests):
        session = mock_requests.Session.return_value
        session.post.return_value = _make_response(400, {"status": "error", "data": "bad"})
        kb = _client(max_retries=3, retry_backoff_seconds=0)

        with pytest.raises(KBBadRequestError):
            kb.search("q", ["kb1"])
        assert session.post.call_count == 1

    @patch("langchain_timbr.kbclient.requests")
    def test_502_exhausts_retries_and_raises_server(self, mock_requests):
        session = mock_requests.Session.return_value
        session.post.return_value = _make_response(502, {"status": "error", "data": "bad gateway"})
        kb = _client(max_retries=2, retry_backoff_seconds=0)

        with pytest.raises(KBServerError):
            kb.search("q", ["kb1"])
        assert session.post.call_count == 3


# --------------------------------------------------------------------------- #
# Client-side validation (fail fast, no HTTP)
# --------------------------------------------------------------------------- #
class TestClientValidation:
    @patch("langchain_timbr.kbclient.requests")
    def test_empty_question(self, mock_requests):
        kb = _client()
        with pytest.raises(KBBadRequestError):
            kb.search("   ", ["kb1"])
        mock_requests.Session.return_value.post.assert_not_called()

    @patch("langchain_timbr.kbclient.requests")
    def test_empty_knowledge_bases(self, mock_requests):
        kb = _client()
        with pytest.raises(KBBadRequestError):
            kb.search("q", [])
        mock_requests.Session.return_value.post.assert_not_called()

    @patch("langchain_timbr.kbclient.requests")
    def test_top_k_out_of_range(self, mock_requests):
        kb = _client()
        with pytest.raises(KBBadRequestError):
            kb.search("q", ["kb1"], top_k=21)
        mock_requests.Session.return_value.post.assert_not_called()

    @patch("langchain_timbr.kbclient.requests")
    def test_medium_above_high(self, mock_requests):
        kb = _client()
        with pytest.raises(KBBadRequestError):
            kb.search("q", ["kb1"], high_threshold=0.5, medium_threshold=0.6)
        mock_requests.Session.return_value.post.assert_not_called()


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #
class TestCache:
    @patch("langchain_timbr.kbclient.run_query")
    @patch("langchain_timbr.kbclient.requests")
    def test_cache_hit_avoids_second_post(self, mock_requests, mock_run_query):
        session = mock_requests.Session.return_value
        session.post.return_value = _make_response(200, _high_match_body())
        # DB says nothing changed since cached value.
        mock_run_query.return_value = [
            {"knowledge_base": "finance_kb", "max_changed_on": "2026-06-14T10:22:00Z"}
        ]
        kb = _client(cache_ttl_seconds=300)

        r1 = kb.search("q", ["finance_kb"])
        r2 = kb.search("q", ["finance_kb"])

        assert r1 is r2  # same cached object
        assert session.post.call_count == 1  # second call served from cache

    @patch("langchain_timbr.kbclient.run_query")
    @patch("langchain_timbr.kbclient.requests")
    def test_cache_invalidated_when_kb_changed(self, mock_requests, mock_run_query):
        session = mock_requests.Session.return_value
        session.post.return_value = _make_response(200, _high_match_body())
        # DB reports a newer changed_on -> stale -> refetch.
        mock_run_query.return_value = [
            {"knowledge_base": "finance_kb", "max_changed_on": "2026-06-15T00:00:00Z"}
        ]
        kb = _client(cache_ttl_seconds=300)

        kb.search("q", ["finance_kb"])
        kb.search("q", ["finance_kb"])

        assert session.post.call_count == 2  # refetched after invalidation

    @patch("langchain_timbr.kbclient.requests")
    def test_pure_ttl_when_no_conn(self, mock_requests):
        # No conn_params override, but base_url+token exist -> conn_params
        # resolves; force pure-TTL by clearing token via override=None path.
        session = mock_requests.Session.return_value
        session.post.return_value = _make_response(200, _high_match_body())
        kb = _client(cache_ttl_seconds=300)
        # Simulate no DB access so caching falls back to pure TTL.
        kb._resolve_conn_params = lambda: None  # type: ignore

        r1 = kb.search("q", ["finance_kb"])
        r2 = kb.search("q", ["finance_kb"])

        assert r1 is r2
        assert session.post.call_count == 1  # served from TTL cache, no DB check

    @patch("langchain_timbr.kbclient.run_query")
    @patch("langchain_timbr.kbclient.requests")
    def test_invalidate_cache_by_kb(self, mock_requests, mock_run_query):
        session = mock_requests.Session.return_value
        session.post.return_value = _make_response(200, _high_match_body())
        mock_run_query.return_value = [
            {"knowledge_base": "finance_kb", "max_changed_on": "2026-06-14T10:22:00Z"}
        ]
        kb = _client(cache_ttl_seconds=300)

        kb.search("q", ["finance_kb"])
        kb.invalidate_cache(["finance_kb"])
        kb.search("q", ["finance_kb"])

        assert session.post.call_count == 2


# --------------------------------------------------------------------------- #
# KB resolution (fail-safe)
# --------------------------------------------------------------------------- #
class TestResolveFromAgent:
    @patch("langchain_timbr.kbclient.run_query")
    def test_agent_csv_parsed(self, mock_run_query):
        mock_run_query.return_value = [{"knowledge_bases": "finance_kb, shared_kpis_kb"}]
        kb = _client()

        assert kb.resolve_knowledge_bases(agent="agentA") == ["finance_kb", "shared_kpis_kb"]

    @patch("langchain_timbr.kbclient.run_query")
    def test_agent_null_returns_empty(self, mock_run_query):
        mock_run_query.return_value = [{"knowledge_bases": None}]
        kb = _client()

        assert kb.resolve_knowledge_bases(agent="agentA") == []

    @patch("langchain_timbr.kbclient.run_query")
    def test_agent_missing_column_fails_safely(self, mock_run_query):
        mock_run_query.side_effect = Exception("no such column: knowledge_bases")
        kb = _client()

        # Warns and continues — no exception.
        assert kb.resolve_knowledge_bases(agent="agentA") == []


class TestResolveFromOntology:
    @patch("langchain_timbr.kbclient.run_query")
    def test_ontology_filter(self, mock_run_query):
        mock_run_query.return_value = [
            {"knowledge_base": "finance_kb", "ontologies": "sales,finance", "changed_on": "2026-01-01T00:00:00Z"},
            {"knowledge_base": "hr_kb", "ontologies": "hr", "changed_on": "2026-01-01T00:00:00Z"},
            {"knowledge_base": "empty_kb", "ontologies": None, "changed_on": "2026-01-01T00:00:00Z"},
        ]
        kb = _client()

        assert kb.resolve_knowledge_bases(ontology="finance") == ["finance_kb"]

    @patch("langchain_timbr.kbclient.run_query")
    def test_ontology_table_missing_fails_safely(self, mock_run_query):
        mock_run_query.side_effect = Exception("Table timbr.sys_knowledgebase not found")
        kb = _client()

        assert kb.resolve_knowledge_bases(ontology="finance") == []

    @patch("langchain_timbr.kbclient.run_query")
    def test_ontology_map_cached_by_changed_on(self, mock_run_query):
        mock_run_query.return_value = [
            {"knowledge_base": "finance_kb", "ontologies": "finance", "changed_on": "2026-01-01T00:00:00Z"},
        ]
        kb = _client()

        kb.resolve_knowledge_bases(ontology="finance")
        kb.resolve_knowledge_bases(ontology="finance")

        # Second call re-queries the table (to read MAX changed_on) but reuses
        # the cached map; still exactly two queries (one per call).
        assert mock_run_query.call_count == 2


# --------------------------------------------------------------------------- #
# Auto-resolution in search()
# --------------------------------------------------------------------------- #
class TestSearchAutoResolve:
    @patch("langchain_timbr.kbclient.run_query")
    @patch("langchain_timbr.kbclient.requests")
    def test_search_auto_resolves_agent(self, mock_requests, mock_run_query):
        session = mock_requests.Session.return_value
        session.post.return_value = _make_response(200, _none_match_body())
        mock_run_query.return_value = [{"knowledge_bases": "finance_kb"}]
        kb = _client()

        kb.search("q", agent="agentA")

        body = session.post.call_args.kwargs["json"]
        assert body["knowledge_bases"] == ["finance_kb"]

    @patch("langchain_timbr.kbclient.run_query")
    @patch("langchain_timbr.kbclient.requests")
    def test_search_raises_when_no_kbs_resolved(self, mock_requests, mock_run_query):
        mock_run_query.return_value = [{"knowledge_bases": None}]
        kb = _client()

        with pytest.raises(KBBadRequestError):
            kb.search("q", agent="agentA")
        mock_requests.Session.return_value.post.assert_not_called()

    @patch("langchain_timbr.kbclient.run_query")
    @patch("langchain_timbr.kbclient.requests")
    def test_kb_in_db_but_api_fails_surfaces_error(self, mock_requests, mock_run_query):
        # KB resolves fine, but the search API is down -> caller can catch.
        mock_run_query.return_value = [{"knowledge_bases": "finance_kb"}]
        session = mock_requests.Session.return_value
        session.post.return_value = _make_response(500, {"status": "error", "data": "unavailable"})
        kb = _client()

        with pytest.raises(KBClientError):
            kb.search("q", agent="agentA")


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #
class TestLifecycle:
    @patch("langchain_timbr.kbclient.requests")
    def test_context_manager_closes_session(self, mock_requests):
        session = mock_requests.Session.return_value
        session.post.return_value = _make_response(200, _none_match_body())

        with _client() as kb:
            kb.search("q", ["kb1"])

        session.close.assert_called_once()
