"""Unit tests for the conversation memory subsystem (utils/memory.py)."""
import json
import pytest
from unittest.mock import Mock, patch, MagicMock

from langchain_timbr.utils.memory import (
    MEMORY_DISABLED,
    MemoryContext,
    MemoryDisabledSentinel,
    resolve_memory,
    fetch_conversation_history,
    _walk_parent_chain,
    _validate_classifier_output,
    build_sql_context,
    build_qa_context,
    format_memory_note_for_sql,
    format_memory_note_for_answer,
    _build_auth_headers,
    _SOFT_SQL_CONTEXT_LIMIT,
    _SOFT_QA_CONTEXT_LIMIT,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def conn_params():
    return {
        "url": "https://test.timbr.ai",
        "token": "test-token",
        "is_jwt": False,
    }


@pytest.fixture
def sample_messages():
    """Three-message history: msg-1 is root, msg-2 follows msg-1, msg-3 is independent."""
    return [
        {
            "message_id": "msg-1",
            "question": "What are total sales?",
            "answer": "Total sales are $1M.",
            "sql": "SELECT SUM(sales) FROM dtimbr.orders",
            "is_follow_up": False,
            "parent_query_id": None,
        },
        {
            "message_id": "msg-2",
            "question": "Break it down by region",
            "answer": "North: $600K, South: $400K.",
            "sql": "SELECT region, SUM(sales) FROM dtimbr.orders GROUP BY region",
            "is_follow_up": True,
            "parent_query_id": "msg-1",
        },
        {
            "message_id": "msg-3",
            "question": "List all customers",
            "answer": "Here are the customers...",
            "sql": "SELECT * FROM dtimbr.customer",
            "is_follow_up": False,
            "parent_query_id": None,
        },
    ]


# ---------------------------------------------------------------------------
# MemoryDisabledSentinel
# ---------------------------------------------------------------------------
class TestMemoryDisabledSentinel:
    def test_singleton(self):
        a = MemoryDisabledSentinel()
        b = MemoryDisabledSentinel()
        assert a is b
        assert a is MEMORY_DISABLED

    def test_falsy(self):
        assert not MEMORY_DISABLED
        assert bool(MEMORY_DISABLED) is False

    def test_repr(self):
        assert repr(MEMORY_DISABLED) == "MemoryDisabledSentinel()"


# ---------------------------------------------------------------------------
# resolve_memory — activation gate
# ---------------------------------------------------------------------------
class TestResolveMemoryGate:
    """Test that resolve_memory returns MEMORY_DISABLED when preconditions fail."""

    def test_disabled_when_enable_memory_false(self, conn_params):
        result = resolve_memory(
            llm=Mock(), conn_params=conn_params, conversation_id="conv1",
            prompt="hello", enable_memory=False, memory_window_size=3,
        )
        assert result is MEMORY_DISABLED

    def test_disabled_when_conversation_id_empty(self, conn_params):
        result = resolve_memory(
            llm=Mock(), conn_params=conn_params, conversation_id="",
            prompt="hello", enable_memory=True, memory_window_size=3,
        )
        assert result is MEMORY_DISABLED

    def test_disabled_when_conversation_id_none(self, conn_params):
        result = resolve_memory(
            llm=Mock(), conn_params=conn_params, conversation_id=None,
            prompt="hello", enable_memory=True, memory_window_size=3,
        )
        assert result is MEMORY_DISABLED

    def test_disabled_when_prompt_empty(self, conn_params):
        result = resolve_memory(
            llm=Mock(), conn_params=conn_params, conversation_id="conv1",
            prompt="", enable_memory=True, memory_window_size=3,
        )
        assert result is MEMORY_DISABLED

    def test_disabled_when_prompt_whitespace(self, conn_params):
        result = resolve_memory(
            llm=Mock(), conn_params=conn_params, conversation_id="conv1",
            prompt="   ", enable_memory=True, memory_window_size=3,
        )
        assert result is MEMORY_DISABLED

    @patch("langchain_timbr.utils.memory.fetch_conversation_history", return_value=None)
    def test_disabled_when_no_history(self, mock_fetch, conn_params):
        result = resolve_memory(
            llm=Mock(), conn_params=conn_params, conversation_id="conv1",
            prompt="hello", enable_memory=True, memory_window_size=3,
        )
        assert result is MEMORY_DISABLED

    @patch("langchain_timbr.utils.memory.fetch_conversation_history", side_effect=RuntimeError("boom"))
    def test_disabled_on_exception(self, mock_fetch, conn_params):
        """Any exception in the pipeline returns MEMORY_DISABLED (silent fail)."""
        result = resolve_memory(
            llm=Mock(), conn_params=conn_params, conversation_id="conv1",
            prompt="hello", enable_memory=True, memory_window_size=3,
        )
        assert result is MEMORY_DISABLED


# ---------------------------------------------------------------------------
# fetch_conversation_history
# ---------------------------------------------------------------------------
class TestFetchConversationHistory:
    @patch("langchain_timbr.utils.memory.requests.get")
    def test_success_list_response(self, mock_get, conn_params):
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = [{"message_id": "m1"}]
        mock_get.return_value = mock_response

        result = fetch_conversation_history(conn_params, "conv1", top=3)
        assert result == [{"message_id": "m1"}]

    @patch("langchain_timbr.utils.memory.requests.get")
    def test_success_dict_data_response(self, mock_get, conn_params):
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {"data": [{"message_id": "m1"}]}
        mock_get.return_value = mock_response

        result = fetch_conversation_history(conn_params, "conv1", top=3)
        assert result == [{"message_id": "m1"}]

    @patch("langchain_timbr.utils.memory.requests.get")
    def test_returns_none_on_http_error(self, mock_get, conn_params):
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        result = fetch_conversation_history(conn_params, "conv1", top=3)
        assert result is None

    @patch("langchain_timbr.utils.memory.requests.get", side_effect=Exception("timeout"))
    def test_returns_none_on_exception(self, mock_get, conn_params):
        result = fetch_conversation_history(conn_params, "conv1", top=3)
        assert result is None

    def test_returns_none_when_no_url(self):
        result = fetch_conversation_history({"url": ""}, "conv1", top=3)
        assert result is None

    @patch("langchain_timbr.utils.memory.requests.get")
    def test_truncated_flag_logged_but_data_returned(self, mock_get, conn_params):
        """API response with truncated=true still returns messages."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "data": [{"message_id": "m1"}],
            "truncated": True,
        }
        mock_get.return_value = mock_response

        result = fetch_conversation_history(conn_params, "conv1", top=3)
        assert result == [{"message_id": "m1"}]

    @patch("langchain_timbr.utils.memory.requests.get")
    def test_no_truncated_field_still_works(self, mock_get, conn_params):
        """Older API without truncated field still works (backward compat)."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {"data": [{"message_id": "m1"}]}
        mock_get.return_value = mock_response

        result = fetch_conversation_history(conn_params, "conv1", top=3)
        assert result == [{"message_id": "m1"}]


# ---------------------------------------------------------------------------
# _walk_parent_chain
# ---------------------------------------------------------------------------
class TestWalkParentChain:
    def test_happy_path(self):
        """Walks root -> leaf correctly, returns ancestors excluding self."""
        id_map = {
            "m-1": {"message_id": "m-1", "parent_query_id": None},
            "m-2": {"message_id": "m-2", "parent_query_id": "m-1"},
            "m-3": {"message_id": "m-3", "parent_query_id": "m-2"},
        }
        ancestors = _walk_parent_chain("m-3", id_map)
        assert [a["message_id"] for a in ancestors] == ["m-1", "m-2"]

    def test_cycle_protection(self):
        """Cycles terminate without infinite loop."""
        id_map = {
            "a": {"message_id": "a", "parent_query_id": "b"},
            "b": {"message_id": "b", "parent_query_id": "a"},
        }
        ancestors = _walk_parent_chain("a", id_map)
        # Should terminate — b's parent is a which is in seen
        assert all(a["message_id"] != "a" for a in ancestors)
        assert len(ancestors) <= 1

    def test_missing_parent_in_id_map(self):
        """Missing parent returns partial (empty) chain."""
        id_map = {
            "x": {"message_id": "x", "parent_query_id": "unknown"},
        }
        ancestors = _walk_parent_chain("x", id_map)
        assert ancestors == []

    def test_no_parent(self):
        """Root message has no ancestors."""
        id_map = {
            "root": {"message_id": "root", "parent_query_id": None},
        }
        ancestors = _walk_parent_chain("root", id_map)
        assert ancestors == []


# ---------------------------------------------------------------------------
# _validate_classifier_output
# ---------------------------------------------------------------------------
class TestValidateClassifierOutput:
    def test_valid_follow_up(self):
        raw = json.dumps({
            "is_follow_up": True,
            "summary": "Refining previous query",
            "parent_message_id": "m-1",
            "relevant_message_ids": ["m-1"],
            "requires_extended_context": False,
        })
        history_ids = {"m-1", "m-2"}
        result = _validate_classifier_output(raw, history_ids)
        assert result is not None
        assert result["is_follow_up"] is True
        assert result["parent_message_id"] == "m-1"
        assert result["relevant_message_ids"] == ["m-1"]

    def test_valid_not_follow_up(self):
        raw = json.dumps({"is_follow_up": False})
        result = _validate_classifier_output(raw, {"m-1"})
        assert result is not None
        assert result["is_follow_up"] is False
        assert result["relevant_message_ids"] == []

    def test_invalid_json(self):
        assert _validate_classifier_output("not json", set()) is None

    def test_non_dict_json(self):
        assert _validate_classifier_output("[1, 2]", set()) is None

    def test_follow_up_without_relevant_ids_forced_false(self):
        raw = json.dumps({"is_follow_up": True, "relevant_message_ids": []})
        result = _validate_classifier_output(raw, set())
        assert result is not None
        assert result["is_follow_up"] is False

    def test_parent_id_not_in_relevant_ids_rejected(self):
        raw = json.dumps({
            "is_follow_up": True,
            "parent_message_id": "m-1",
            "relevant_message_ids": ["m-2"],
        })
        result = _validate_classifier_output(raw, {"m-1", "m-2"})
        assert result is None

    def test_parent_id_not_in_history_rejected(self):
        raw = json.dumps({
            "is_follow_up": True,
            "parent_message_id": "m-99",
            "relevant_message_ids": ["m-99"],
        })
        result = _validate_classifier_output(raw, {"m-1"})
        assert result is None

    def test_relevant_id_not_in_history_rejected(self):
        raw = json.dumps({
            "is_follow_up": True,
            "parent_message_id": "m-1",
            "relevant_message_ids": ["m-1", "m-99"],
        })
        result = _validate_classifier_output(raw, {"m-1"})
        assert result is None

    def test_strips_markdown_fences(self):
        payload = json.dumps({"is_follow_up": False})
        raw = f"```json\n{payload}\n```"
        result = _validate_classifier_output(raw, set())
        assert result is not None
        assert result["is_follow_up"] is False

    def test_requires_extended_context(self):
        raw = json.dumps({
            "is_follow_up": True,
            "parent_message_id": "m-1",
            "relevant_message_ids": ["m-1"],
            "requires_extended_context": True,
            "summary": "complex",
        })
        result = _validate_classifier_output(raw, {"m-1"})
        assert result["requires_extended_context"] is True


# ---------------------------------------------------------------------------
# build_sql_context
# ---------------------------------------------------------------------------
class TestBuildSqlContext:
    def _make_messages_and_id_map(self):
        m1 = {
            "message_id": "m-1", "question": "Q1", "sql": "SQL1",
            "is_follow_up": False, "parent_query_id": None,
        }
        m2 = {
            "message_id": "m-2", "question": "Q2", "sql": "SQL2",
            "is_follow_up": True, "parent_query_id": "m-1",
        }
        m3 = {
            "message_id": "m-3", "question": "Q3", "sql": "SQL3",
            "is_follow_up": False, "parent_query_id": None,
        }
        msgs = [m1, m2, m3]
        id_map = {m["message_id"]: m for m in msgs}
        return msgs, id_map

    def test_primary_chain_plus_siblings(self):
        msgs, id_map = self._make_messages_and_id_map()
        classifier_output = {
            "parent_message_id": "m-2",
            "relevant_message_ids": ["m-2", "m-3"],
            "requires_extended_context": False,
        }
        ctx = build_sql_context(id_map, classifier_output)
        ids = [e["message_id"] for e in ctx]
        # Primary chain: m-1 (ancestor of m-2), m-2 itself. Sibling: m-3.
        assert "m-1" in ids
        assert "m-2" in ids
        assert "m-3" in ids

    def test_soft_limit_applied(self):
        # Create more messages than _SOFT_SQL_CONTEXT_LIMIT
        count = _SOFT_SQL_CONTEXT_LIMIT + 5
        msgs = [
            {"message_id": f"m-{i}", "question": f"Q{i}", "sql": f"SQL{i}", "parent_query_id": None}
            for i in range(count)
        ]
        id_map = {m["message_id"]: m for m in msgs}
        classifier_output = {
            "parent_message_id": None,
            "relevant_message_ids": [f"m-{i}" for i in range(count)],
            "requires_extended_context": False,
        }
        ctx = build_sql_context(id_map, classifier_output)
        assert len(ctx) <= _SOFT_SQL_CONTEXT_LIMIT

    def test_extended_context_bypasses_limit(self):
        count = _SOFT_SQL_CONTEXT_LIMIT + 5
        msgs = [
            {"message_id": f"m-{i}", "question": f"Q{i}", "sql": f"SQL{i}", "parent_query_id": None}
            for i in range(count)
        ]
        id_map = {m["message_id"]: m for m in msgs}
        classifier_output = {
            "parent_message_id": None,
            "relevant_message_ids": [f"m-{i}" for i in range(count)],
            "requires_extended_context": True,
        }
        ctx = build_sql_context(id_map, classifier_output)
        assert len(ctx) == count


# ---------------------------------------------------------------------------
# build_qa_context
# ---------------------------------------------------------------------------
class TestBuildQaContext:
    def test_ordered_by_classifier_ranking(self):
        msgs = [
            {"message_id": "m-1", "question": "Q1", "answer": "A1"},
            {"message_id": "m-2", "question": "Q2", "answer": "A2"},
        ]
        id_map = {m["message_id"]: m for m in msgs}
        classifier_output = {
            "relevant_message_ids": ["m-2", "m-1"],  # m-2 ranked higher
        }
        ctx = build_qa_context(id_map, classifier_output)
        assert ctx[0]["message_id"] == "m-2"
        assert ctx[1]["message_id"] == "m-1"

    def test_soft_limit_applied(self):
        count = _SOFT_QA_CONTEXT_LIMIT + 5
        msgs = [
            {"message_id": f"m-{i}", "question": f"Q{i}", "answer": f"A{i}"}
            for i in range(count)
        ]
        id_map = {m["message_id"]: m for m in msgs}
        classifier_output = {
            "relevant_message_ids": [f"m-{i}" for i in range(count)],
            "requires_extended_context": False,
        }
        ctx = build_qa_context(id_map, classifier_output)
        assert len(ctx) <= _SOFT_QA_CONTEXT_LIMIT

    def test_extended_context_bypasses_limit(self):
        count = _SOFT_QA_CONTEXT_LIMIT + 5
        msgs = [
            {"message_id": f"m-{i}", "question": f"Q{i}", "answer": f"A{i}"}
            for i in range(count)
        ]
        id_map = {m["message_id"]: m for m in msgs}
        classifier_output = {
            "relevant_message_ids": [f"m-{i}" for i in range(count)],
            "requires_extended_context": True,
        }
        ctx = build_qa_context(id_map, classifier_output)
        assert len(ctx) == count


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------
class TestFormatMemoryNoteForSql:
    def test_empty_when_not_follow_up(self):
        ctx = MemoryContext(is_follow_up=False)
        assert format_memory_note_for_sql(ctx) == ""

    def test_empty_when_none(self):
        assert format_memory_note_for_sql(None) == ""

    def test_contains_follow_up_header(self):
        ctx = MemoryContext(
            is_follow_up=True,
            summary="User wants regional breakdown",
            sql_context=[
                {"message_id": "m-1", "question": "Total sales?", "sql": "SELECT SUM(sales) FROM orders"},
            ],
        )
        result = format_memory_note_for_sql(ctx)
        assert "[CONVERSATION MEMORY]" in result
        assert "follow-up" in result.lower()
        assert "regional breakdown" in result
        assert "SELECT SUM(sales)" in result

    def test_no_sql_context_still_works(self):
        ctx = MemoryContext(is_follow_up=True, summary="Context only")
        result = format_memory_note_for_sql(ctx)
        assert "[CONVERSATION MEMORY]" in result
        assert "Prior SQL" not in result


class TestFormatMemoryNoteForAnswer:
    def test_empty_when_not_follow_up(self):
        ctx = MemoryContext(is_follow_up=False)
        assert format_memory_note_for_answer(ctx) == ""

    def test_empty_when_none(self):
        assert format_memory_note_for_answer(None) == ""

    def test_contains_qa_pairs(self):
        ctx = MemoryContext(
            is_follow_up=True,
            summary="Refining question",
            qa_context=[
                {"question": "Total sales?", "answer": "$1M"},
                {"question": "By region?", "answer": "North $600K"},
            ],
        )
        result = format_memory_note_for_answer(ctx)
        assert "[CONVERSATION MEMORY]" in result
        assert "Total sales?" in result
        assert "$1M" in result
        assert "By region?" in result


# ---------------------------------------------------------------------------
# _build_auth_headers
# ---------------------------------------------------------------------------
class TestBuildAuthHeaders:
    def test_api_key_header(self):
        headers = _build_auth_headers({"token": "my-token", "is_jwt": False})
        assert headers["x-api-key"] == "my-token"
        assert "x-jwt-token" not in headers

    def test_jwt_headers(self):
        headers = _build_auth_headers({
            "token": "jwt-tok", "is_jwt": True, "jwt_tenant_id": "tenant1",
        })
        assert headers["x-jwt-token"] == "jwt-tok"
        assert headers["x-jwt-tenant-id"] == "tenant1"
        assert "x-api-key" not in headers

    def test_jwt_without_tenant(self):
        headers = _build_auth_headers({"token": "jwt-tok", "is_jwt": True})
        assert headers["x-jwt-token"] == "jwt-tok"
        assert "x-jwt-tenant-id" not in headers


# ---------------------------------------------------------------------------
# MemoryContext dataclass
# ---------------------------------------------------------------------------
class TestMemoryContext:
    def test_defaults(self):
        ctx = MemoryContext(is_follow_up=False)
        assert ctx.summary == ""
        assert ctx.parent_message_id is None
        assert ctx.relevant_message_ids == []
        assert ctx.sql_context == []
        assert ctx.qa_context == []
        assert ctx.requires_extended_context is False

    def test_truthy_when_follow_up(self):
        ctx = MemoryContext(is_follow_up=True)
        assert ctx  # dataclass is truthy by default
