"""Unit tests for the KB-example integration in the memory subsystem.

These tests exercise the KB-specific behaviours added on top of conversation
memory (utils/memory.py): KB example selection, the `[Approved reference
examples]` block, EXPANDED-INTENT question expansion, the classifier-output
parsing of `should_apply_examples`/`relevant_example_names`, and the
KB-availability gate.  They are fully mock-based and never touch a backend.
"""
from unittest.mock import Mock, patch

import pytest

from langchain_timbr.kbclient import KBMatch
from langchain_timbr.utils.memory import (
    MEMORY_DISABLED,
    MemoryContext,
    apply_memory_question_expansion,
    format_memory_note_for_sql,
    format_memory_note_for_answer,
    resolve_memory,
    _select_kb_examples,
    _validate_classifier_output,
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
def kb_matches():
    return [
        KBMatch(
            knowledge_base="supply_metrics_kb",
            example_name="inventory_stockout_risk",
            question="Which products are at risk of stockout?",
            query="SELECT product FROM dtimbr.inventory WHERE stock < reorder_point",
            instructions="Compare stock against reorder point.",
            validate_sql=0,
            confidence=0.95,
            changed_on=None,
        ),
        KBMatch(
            knowledge_base="supply_metrics_kb",
            example_name="late_delivery_by_mode",
            question="Late deliveries grouped by shipping mode?",
            query="SELECT mode, COUNT(*) FROM dtimbr.shipments WHERE late = 1 GROUP BY mode",
            instructions="Filter on the late flag then group by mode.",
            validate_sql=0,
            confidence=0.88,
            changed_on=None,
        ),
    ]


# ---------------------------------------------------------------------------
# _select_kb_examples
# ---------------------------------------------------------------------------
def test_select_kb_examples_case_insensitive(kb_matches):
    selected = _select_kb_examples(kb_matches, ["Inventory_Stockout_Risk"])
    assert len(selected) == 1
    assert selected[0]["example_name"] == "inventory_stockout_risk"
    assert selected[0]["knowledge_base"] == "supply_metrics_kb"
    assert selected[0]["query"].startswith("SELECT product")


def test_select_kb_examples_empty_when_no_names(kb_matches):
    assert _select_kb_examples(kb_matches, None) == []
    assert _select_kb_examples(kb_matches, []) == []


def test_select_kb_examples_ignores_unknown_names(kb_matches):
    assert _select_kb_examples(kb_matches, ["does_not_exist"]) == []


# ---------------------------------------------------------------------------
# format_memory_note_for_sql — [Approved reference examples] block
# ---------------------------------------------------------------------------
def test_sql_note_includes_kb_examples_block():
    ctx = MemoryContext(
        is_follow_up=False,
        kb_examples=[
            {
                "example_name": "inventory_stockout_risk",
                "knowledge_base": "supply_metrics_kb",
                "instructions": "Compare stock against reorder point.",
                "query": "SELECT product FROM dtimbr.inventory",
            }
        ],
    )
    note = format_memory_note_for_sql(ctx)
    assert "[Approved reference examples]" in note
    assert "inventory_stockout_risk" in note
    assert "Instructions: Compare stock against reorder point." in note
    assert "SELECT product FROM dtimbr.inventory" in note


def test_answer_note_never_includes_kb_examples():
    ctx = MemoryContext(
        is_follow_up=False,
        kb_examples=[
            {
                "example_name": "inventory_stockout_risk",
                "knowledge_base": "supply_metrics_kb",
                "instructions": "x",
                "query": "SELECT 1",
            }
        ],
    )
    note = format_memory_note_for_answer(ctx)
    assert "[Approved reference examples]" not in note
    assert note == ""


def test_sql_note_combines_followup_and_kb():
    ctx = MemoryContext(
        is_follow_up=True,
        summary="prior context",
        sql_context=[{"question": "totals?", "sql": "SELECT SUM(x) FROM t"}],
        kb_examples=[
            {
                "example_name": "ex1",
                "knowledge_base": "kb",
                "instructions": None,
                "query": "SELECT 1",
            }
        ],
    )
    note = format_memory_note_for_sql(ctx)
    assert "[CONVERSATION MEMORY]" in note
    assert "[Approved reference examples]" in note


# ---------------------------------------------------------------------------
# apply_memory_question_expansion
# ---------------------------------------------------------------------------
def test_expansion_appended_when_summary_present():
    ctx = MemoryContext(is_follow_up=False, summary="only late deliveries in 2023")
    out = apply_memory_question_expansion("show deliveries", ctx)
    assert out.startswith("show deliveries")
    assert "EXPANDED INTENT" in out
    assert "only late deliveries in 2023" in out


def test_expansion_absent_when_no_summary():
    ctx = MemoryContext(is_follow_up=False, summary="")
    assert apply_memory_question_expansion("show deliveries", ctx) == "show deliveries"


def test_expansion_absent_when_no_context():
    assert apply_memory_question_expansion("show deliveries", None) == "show deliveries"


# ---------------------------------------------------------------------------
# _validate_classifier_output — KB fields + summary retention
# ---------------------------------------------------------------------------
def test_classifier_parses_kb_fields_non_followup():
    raw = (
        '{"is_follow_up": false, "summary": "expanded intent text", '
        '"should_apply_examples": true, '
        '"relevant_example_names": ["inventory_stockout_risk"]}'
    )
    result = _validate_classifier_output(
        raw, history_ids=set(), valid_example_names={"inventory_stockout_risk"}
    )
    assert result is not None
    assert result["is_follow_up"] is False
    # summary retained even in the non-follow-up branch
    assert result["summary"] == "expanded intent text"
    assert result["should_apply_examples"] is True
    assert result["relevant_example_names"] == ["inventory_stockout_risk"]


def test_classifier_filters_invalid_example_names():
    raw = (
        '{"is_follow_up": false, "summary": "", '
        '"should_apply_examples": true, '
        '"relevant_example_names": ["ghost_example"]}'
    )
    result = _validate_classifier_output(
        raw, history_ids=set(), valid_example_names={"inventory_stockout_risk"}
    )
    assert result is not None
    # unknown name filtered out → should_apply_examples forced False
    assert result["relevant_example_names"] == []
    assert result["should_apply_examples"] is False


# ---------------------------------------------------------------------------
# resolve_memory gate — KB available while memory disabled
# ---------------------------------------------------------------------------
def test_gate_kb_available_memory_disabled(conn_params):
    """With memory disabled but KBs available, resolution still runs (not DISABLED)."""
    with patch(
        "langchain_timbr.utils.memory._resolve_available_kbs",
        return_value=["supply_metrics_kb"],
    ), patch(
        "langchain_timbr.utils.memory._resolve_memory_impl",
        return_value=MemoryContext(is_follow_up=False),
    ) as mock_impl, patch(
        "langchain_timbr.utils.memory.config"
    ) as mock_config:
        mock_config.enable_knowledge_base = True
        result = resolve_memory(
            llm=Mock(),
            conn_params=conn_params,
            conversation_id=None,
            prompt="which products are at risk of stockout?",
            enable_memory=False,
            memory_window_size=5,
            agent="supply_agent",
            ontology=None,
        )
    assert result is not MEMORY_DISABLED
    assert mock_impl.called


def test_gate_disabled_when_no_memory_and_no_kbs(conn_params):
    """No memory, no KBs → DISABLED sentinel, no impl call."""
    with patch(
        "langchain_timbr.utils.memory._resolve_available_kbs",
        return_value=[],
    ), patch(
        "langchain_timbr.utils.memory._resolve_memory_impl",
    ) as mock_impl, patch(
        "langchain_timbr.utils.memory.config"
    ) as mock_config:
        mock_config.enable_knowledge_base = True
        result = resolve_memory(
            llm=Mock(),
            conn_params=conn_params,
            conversation_id=None,
            prompt="hello",
            enable_memory=False,
            memory_window_size=5,
            agent=None,
            ontology=None,
        )
    assert result is MEMORY_DISABLED
    assert not mock_impl.called
