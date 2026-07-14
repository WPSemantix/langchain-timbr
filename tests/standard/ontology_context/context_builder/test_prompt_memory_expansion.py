"""The dynamic pre-filter / filter prompt builders fold conversation-memory
expanded intent into the question, mirroring the SQL-gen / concept prompts.

Guards two properties:
  - when a ``MemoryContext`` carries a summary, the EXPANDED INTENT marker is
    present in the built user message;
  - when no memory context is supplied, the output is byte-identical to before
    (regression guard for the default-None path).
"""

from langchain_timbr.utils.memory import MemoryContext
from langchain_timbr.ontology_context.context_builder.prompts.concept_prefilter_prompt import (
    build_prefilter_messages,
)
from langchain_timbr.ontology_context.context_builder.prompts.filter_prompt import (
    build_filter_messages,
    build_retry_messages,
)


_SUMMARY = "only late deliveries in 2023"


def _user(messages):
    return messages[1]["content"]


# ---------------------------------------------------------------------------
# Pre-filter prompt
# ---------------------------------------------------------------------------
def test_prefilter_expands_question_when_summary_present():
    ctx = MemoryContext(is_follow_up=True, summary=_SUMMARY)
    msgs = build_prefilter_messages(
        question="show deliveries", anchor="Delivery",
        candidates_block="Delivery\nCustomer", with_descriptions=False,
        memory_context=ctx,
    )
    user = _user(msgs)
    assert "EXPANDED INTENT" in user
    assert _SUMMARY in user


def test_prefilter_unchanged_without_memory_context():
    kwargs = dict(
        question="show deliveries", anchor="Delivery",
        candidates_block="Delivery\nCustomer", with_descriptions=False,
    )
    assert build_prefilter_messages(**kwargs) == build_prefilter_messages(
        **kwargs, memory_context=None
    )
    assert "EXPANDED INTENT" not in _user(build_prefilter_messages(**kwargs))


# ---------------------------------------------------------------------------
# Filter + retry prompt
# ---------------------------------------------------------------------------
def test_filter_expands_question_when_summary_present():
    ctx = MemoryContext(is_follow_up=True, summary=_SUMMARY)
    msgs = build_filter_messages(
        question="show deliveries", anchor="Delivery",
        compact_ddl="## CONCEPTS\n", memory_context=ctx,
    )
    user = _user(msgs)
    assert "EXPANDED INTENT" in user
    assert _SUMMARY in user


def test_retry_expands_question_when_summary_present():
    ctx = MemoryContext(is_follow_up=True, summary=_SUMMARY)
    msgs = build_retry_messages(
        question="show deliveries", anchor="Delivery",
        compact_ddl="## CONCEPTS\n", error_lines=["Path P1: bad"],
        memory_context=ctx,
    )
    user = _user(msgs)
    assert "EXPANDED INTENT" in user
    assert _SUMMARY in user


def test_filter_unchanged_without_memory_context():
    kwargs = dict(
        question="show deliveries", anchor="Delivery", compact_ddl="## CONCEPTS\n",
    )
    assert build_filter_messages(**kwargs) == build_filter_messages(
        **kwargs, memory_context=None
    )
    assert "EXPANDED INTENT" not in _user(build_filter_messages(**kwargs))
