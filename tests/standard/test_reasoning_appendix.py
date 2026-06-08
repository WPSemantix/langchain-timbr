"""Unit tests for `_append_reasoning_context_blocks`.

The reasoning template is rendered server-side via the API-hosted template;
context blocks (note, generated-SQL reasoning, generated-SQL decision trace)
are appended client-side to the trailing HumanMessage before the LLM call.

Conversation memory is not a separate appendix block — it rides inside
``note`` (merged in by ``generate_sql`` via ``format_memory_note_for_sql``)
so the appendix only needs three slots.
"""
import json

from langchain_core.messages import HumanMessage, SystemMessage

from langchain_timbr.utils.timbr_llm_utils import _append_reasoning_context_blocks


def _make_prompt(human_content: str = "Question: Q\nSQL: SELECT 1"):
    return [
        SystemMessage(content="You are a SQL reviewer."),
        HumanMessage(content=human_content),
    ]


def test_no_blocks_leaves_prompt_untouched():
    prompt = _make_prompt()
    original = prompt[-1].content

    _append_reasoning_context_blocks(prompt)

    assert prompt[-1].content == original


def test_note_only_appends_additional_notes_block():
    prompt = _make_prompt()
    _append_reasoning_context_blocks(prompt, note="be strict about date filters")

    content = prompt[-1].content
    assert "Question: Q" in content
    assert "**Additional Notes:**" in content
    assert "be strict about date filters" in content
    assert "**Generated SQL Reasoning:**" not in content
    assert "**Generated SQL Decision Trace:**" not in content


def test_blank_note_does_not_append():
    prompt = _make_prompt()
    original = prompt[-1].content

    _append_reasoning_context_blocks(prompt, note="   ")

    assert prompt[-1].content == original


def test_note_carrying_memory_block_passes_through_verbatim():
    """Memory rides inside `note` — confirm the appendix preserves it as-is
    rather than re-adding a separate memory block."""
    prompt = _make_prompt()
    note_with_memory = (
        "[CONVERSATION MEMORY]\n"
        "This is a follow-up question.\n"
        "Context summary: prior turn about Q1 orders\n"
        "--- [1] Q: \"orders in Q1?\" ---\n"
        "SELECT * FROM dtimbr.`order` WHERE quarter = 1\n"
        "--- End ---"
    )

    _append_reasoning_context_blocks(prompt, note=note_with_memory)

    content = prompt[-1].content
    # The memory text appears exactly once, inside the Additional Notes block
    assert content.count("[CONVERSATION MEMORY]") == 1
    assert "WHERE quarter = 1" in content
    # Notes header is present; no duplicate memory block
    assert content.count("**Additional Notes:**") == 1


def test_decisions_only_appends_decision_trace_as_json():
    prompt = _make_prompt()
    decisions = [
        {"choice": "filter 2024", "source": "conversation_note"},
        {"choice": "use orders concept", "source": "schema_metadata"},
    ]

    _append_reasoning_context_blocks(prompt, decisions=decisions)

    content = prompt[-1].content
    assert "**Generated SQL Decision Trace:**" in content
    # JSON-formatted, indented
    assert '"choice": "filter 2024"' in content
    assert '"source": "conversation_note"' in content


def test_empty_decisions_list_does_not_append():
    """`if decisions:` should skip empty lists — nothing useful to add."""
    prompt = _make_prompt()
    original = prompt[-1].content

    _append_reasoning_context_blocks(prompt, decisions=[])

    assert prompt[-1].content == original


def test_generate_sql_reason_only_appends_reasoning_block():
    prompt = _make_prompt()
    _append_reasoning_context_blocks(
        prompt,
        generate_sql_reason="Plan: aggregate by year then filter 2024",
    )

    content = prompt[-1].content
    assert "**Generated SQL Reasoning:**" in content
    assert "Plan: aggregate by year then filter 2024" in content


def test_all_three_blocks_present_in_documented_order():
    prompt = _make_prompt()
    decisions = [{"choice": "x", "source": "schema_metadata"}]

    _append_reasoning_context_blocks(
        prompt,
        note="user note",
        generate_sql_reason="my plan",
        decisions=decisions,
    )

    content = prompt[-1].content

    # All three blocks present
    assert "**Additional Notes:**" in content
    assert "**Generated SQL Reasoning:**" in content
    assert "**Generated SQL Decision Trace:**" in content

    # Documented order: note → reasoning → decisions
    i_note = content.index("**Additional Notes:**")
    i_reason = content.index("**Generated SQL Reasoning:**")
    i_dec = content.index("**Generated SQL Decision Trace:**")
    assert i_note < i_reason < i_dec


def test_appendix_finds_trailing_human_when_not_last_message():
    """Defensive: if the prompt ever has a trailing non-human message,
    the appendix should still target the last HumanMessage."""
    prompt = [
        SystemMessage(content="sys"),
        HumanMessage(content="human turn"),
        SystemMessage(content="trailing system note"),
    ]

    _append_reasoning_context_blocks(prompt, note="hi")

    assert "**Additional Notes:**" in prompt[1].content
    assert "**Additional Notes:**" not in prompt[2].content


def test_decision_trace_json_is_valid_json():
    prompt = _make_prompt()
    decisions = [{"choice": "x", "source": "schema_metadata"}]
    _append_reasoning_context_blocks(prompt, decisions=decisions)

    content = prompt[-1].content
    marker = "**Generated SQL Decision Trace:**\n"
    raw_json = content[content.index(marker) + len(marker):]
    parsed = json.loads(raw_json)
    assert parsed == decisions
