"""Unit tests for `_parse_sql_and_reason_from_llm_response`.

Covers the three-field schema rollout: parser must extract `decisions`
when present (new API), tolerate its absence (legacy API), and never
raise on a malformed shape.
"""
import json

from langchain_core.messages import AIMessage

from langchain_timbr.utils.timbr_llm_utils import (
    _parse_sql_and_reason_from_llm_response,
)


def _ai(content: str) -> AIMessage:
    return AIMessage(content=content)


def test_legacy_two_field_json_returns_none_decisions():
    payload = json.dumps({
        "result": "SELECT * FROM dtimbr.`order`",
        "reason": "Order is the primary concept",
    })
    parsed = _parse_sql_and_reason_from_llm_response(_ai(payload))

    assert parsed["sql"] == "SELECT * FROM dtimbr.`order`"
    assert parsed["reason"] == "Order is the primary concept"
    assert parsed["decisions"] is None


def test_new_three_field_json_extracts_decisions():
    decisions = [
        {"choice": "use `order` concept", "source": "schema_metadata"},
        {"choice": "limit to 2024", "source": "conversation_note"},
    ]
    payload = json.dumps({
        "reason": "Plan: aggregate orders by year, filter 2024",
        "decisions": decisions,
        "result": "SELECT YEAR(`order_date`), COUNT(*) FROM dtimbr.`order`",
    })
    parsed = _parse_sql_and_reason_from_llm_response(_ai(payload))

    assert parsed["sql"].startswith("SELECT YEAR")
    assert parsed["reason"].startswith("Plan:")
    assert parsed["decisions"] == decisions


def test_malformed_decisions_string_is_dropped_not_raised():
    payload = json.dumps({
        "reason": "ok",
        "decisions": "not a list",
        "result": "SELECT 1",
    })
    parsed = _parse_sql_and_reason_from_llm_response(_ai(payload))

    assert parsed["sql"] == "SELECT 1"
    assert parsed["reason"] == "ok"
    assert parsed["decisions"] is None


def test_malformed_decisions_dict_is_dropped_not_raised():
    payload = json.dumps({
        "reason": "ok",
        "decisions": {"not": "a list"},
        "result": "SELECT 1",
    })
    parsed = _parse_sql_and_reason_from_llm_response(_ai(payload))

    assert parsed["decisions"] is None
    assert parsed["sql"] == "SELECT 1"


def test_plain_text_sql_fallback_returns_none_for_reason_and_decisions():
    parsed = _parse_sql_and_reason_from_llm_response(_ai("SELECT * FROM t;"))

    assert parsed["sql"] == "SELECT * FROM t"
    assert parsed["reason"] is None
    assert parsed["decisions"] is None


def test_json_in_markdown_code_block_still_parses_decisions():
    inner = json.dumps({
        "reason": "r",
        "decisions": [{"choice": "x"}],
        "result": "SELECT 1",
    })
    payload = f"```json\n{inner}\n```"
    parsed = _parse_sql_and_reason_from_llm_response(_ai(payload))

    assert parsed["sql"] == "SELECT 1"
    assert parsed["decisions"] == [{"choice": "x"}]


def test_empty_decisions_list_passes_through():
    """An empty list is still a valid list — preserve it (distinct from None)."""
    payload = json.dumps({
        "reason": "r",
        "decisions": [],
        "result": "SELECT 1",
    })
    parsed = _parse_sql_and_reason_from_llm_response(_ai(payload))

    assert parsed["decisions"] == []
