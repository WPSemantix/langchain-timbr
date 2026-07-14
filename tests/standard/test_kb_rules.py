"""Unit tests for knowledge-base rules injection (timbr.sys_knowledgebase_rules).

Everything runs against in-memory fakes — no DB, no LLM. ``run_query`` and the KB
resolver are monkeypatched, and each pipeline stage's rendering is exercised
through its real helper so the injection matrix + safe-fail + backward-compat
guarantees are verified directly.

Injection matrix under test:
  identify / prefilter        -> concept/view/cube SELECTION_RULE
  filter (in-subgraph rels)   -> relationship SELECTION_RULE
  generate_sql concept        -> INSTRUCTION + VALIDATION (never SELECTION)
  generate_sql prop/meas/rel  -> SELECTION_RULE + INSTRUCTION + VALIDATION
  reasoning                   -> concept VALIDATION only
"""

import pytest
from langchain_core.messages import HumanMessage

from langchain_timbr import kbclient as kb
from langchain_timbr import identify_concept_context as icc
from langchain_timbr.ontology_context.context_builder.build_filtered import (
    _render_relationship_rules_block,
)
from langchain_timbr.ontology_context.context_builder.concept_prefilter import (
    _gather_candidates,
    render_names_only,
    render_with_descriptions,
)
from langchain_timbr.ontology_context.context_builder.prompts.filter_prompt import (
    build_filter_messages,
    build_retry_messages,
)
from langchain_timbr.utils.timbr_llm_utils import (
    _append_reasoning_context_blocks,
    _append_rules_subblock,
    _build_columns_str,
    _build_rel_columns_str,
    _rule_meta_items,
)


CONN = {"url": "u", "token": "t", "ontology": "o"}


def _ruleset(by_target, version=None):
    return kb.RuleSet(by_target=by_target, kb_names=("kb1",), version=version)


def _sample_ruleset():
    return _ruleset({
        ("concept", "orders"): {
            "selection": ["only recent orders"],
            "instruction": ["use gross amount"],
            "validation": ["reject cancelled"],
        },
        ("property", "amount"): {
            "selection": ["prefer net"],
            "instruction": ["round to cents"],
            "validation": ["must be positive"],
        },
        ("measure", "total"): {"instruction": ["sum excludes tax"]},
        ("relationship", "placed_by"): {
            "selection": ["active customers only"],
            "validation": ["one per order"],
        },
    })


# --------------------------------------------------------------------------- #
# RuleSet.rules_for + render_object_rules (empty-key suppression)
# --------------------------------------------------------------------------- #
def test_rules_for_filters_by_type_and_kind():
    rs = _sample_ruleset()
    # concept, selection only
    got = rs.rules_for("orders", ("concept", "view", "cube"), {"selection"})
    assert got == {"selection": ["only recent orders"]}
    # concept, instruction+validation (no selection requested -> excluded)
    got = rs.rules_for("orders", ("concept",), {"instruction", "validation"})
    assert "selection" not in got
    assert got["instruction"] == ["use gross amount"]
    assert got["validation"] == ["reject cancelled"]
    # wrong target type -> nothing
    assert rs.rules_for("orders", ("property",), {"selection"}) == {}
    # unknown name -> nothing
    assert rs.rules_for("nope", ("concept",), {"selection"}) == {}


def test_render_object_rules_empty_key_suppression():
    rs = _sample_ruleset()
    # measure has only instruction -> only that label rendered, no bare labels
    txt = kb.render_object_rules(rs.rules_for("total", ("measure",), {"selection", "instruction", "validation"}))
    assert txt == "instructions: sum excludes tax"
    assert "selection_rules:" not in txt
    assert "validation_rules:" not in txt
    # no match -> empty string (this is the backward-compat guarantee)
    assert kb.render_object_rules(rs.rules_for("nope", ("concept",), {"selection"})) == ""


# --------------------------------------------------------------------------- #
# fetch_rules — cache, TTL, safe-fail
# --------------------------------------------------------------------------- #
def _rules_rows(changed="2024-01-01T00:00:00"):
    return [
        {"knowledge_base": "kb1", "rule_name": "r1", "rule_type": "SELECTION_RULE",
         "target_name": "orders", "target_type": "concept",
         "instructions": "only recent orders", "changed_on": changed},
    ]


@pytest.fixture
def patched_kb(monkeypatch):
    """Resolve a fixed KB set and route run_query by SQL shape (load vs version)."""
    kb.clear_rules_cache()
    monkeypatch.setattr(kb.KBClient, "resolve_knowledge_bases", lambda self, **kw: ["kb1"])
    monkeypatch.setattr(kb.config, "enable_knowledge_base", True)
    monkeypatch.setattr(kb.config, "kb_rules_cache_ttl_seconds", 60)

    state = {"load": 0, "version": 0, "version_value": "2024-01-01T00:00:00", "rows": _rules_rows()}

    def fake_run_query(sql, conn_params):
        upper = sql.upper()
        if "MAX(CHANGED_ON)" in upper and "RULE_TYPE" not in upper:
            state["version"] += 1
            return [{"max_changed_on": state["version_value"]}]
        state["load"] += 1
        return list(state["rows"])

    monkeypatch.setattr(kb, "run_query", fake_run_query)
    return state


def test_fetch_rules_cache_hit_within_ttl(patched_kb, monkeypatch):
    monkeypatch.setattr(kb.time, "time", lambda: 1000.0)
    rs1 = kb.fetch_rules(CONN, ontology="o")
    assert not rs1.is_empty()
    assert patched_kb["load"] == 1
    # second call within TTL -> served from cache, no reload / no version query
    rs2 = kb.fetch_rules(CONN, ontology="o")
    assert patched_kb["load"] == 1
    assert patched_kb["version"] == 0
    assert rs2 is rs1


def test_fetch_rules_ttl_expiry_unchanged_no_reload(patched_kb, monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr(kb.time, "time", lambda: clock["t"])
    kb.fetch_rules(CONN, ontology="o")
    assert patched_kb["load"] == 1
    # advance past TTL; MAX(changed_on) unchanged -> version check only, no reload
    clock["t"] += 120
    kb.fetch_rules(CONN, ontology="o")
    assert patched_kb["version"] == 1
    assert patched_kb["load"] == 1


def test_fetch_rules_ttl_expiry_changed_reloads(patched_kb, monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr(kb.time, "time", lambda: clock["t"])
    kb.fetch_rules(CONN, ontology="o")
    assert patched_kb["load"] == 1
    # advance past TTL; MAX(changed_on) advanced -> full reload
    clock["t"] += 120
    patched_kb["version_value"] = "2024-06-01T00:00:00"
    patched_kb["rows"] = _rules_rows("2024-06-01T00:00:00")
    kb.fetch_rules(CONN, ontology="o")
    assert patched_kb["load"] == 2


def test_fetch_rules_safe_fail_on_query_error(monkeypatch):
    kb.clear_rules_cache()
    monkeypatch.setattr(kb.KBClient, "resolve_knowledge_bases", lambda self, **kw: ["kb1"])

    def boom(sql, conn_params):
        raise RuntimeError("older backend: table missing")

    monkeypatch.setattr(kb, "run_query", boom)
    assert kb.fetch_rules(CONN, ontology="o").is_empty()


def test_fetch_rules_safe_fail_on_empty_result(monkeypatch):
    kb.clear_rules_cache()
    monkeypatch.setattr(kb.KBClient, "resolve_knowledge_bases", lambda self, **kw: ["kb1"])
    monkeypatch.setattr(kb, "run_query", lambda sql, cp: [])
    assert kb.fetch_rules(CONN, ontology="o").is_empty()


def test_fetch_rules_no_kbs_is_empty(monkeypatch):
    kb.clear_rules_cache()
    monkeypatch.setattr(kb.KBClient, "resolve_knowledge_bases", lambda self, **kw: [])
    assert kb.fetch_rules(CONN, ontology="o").is_empty()


def test_fetch_rules_kill_switch(monkeypatch):
    kb.clear_rules_cache()
    monkeypatch.setattr(kb.config, "enable_knowledge_base", False)
    assert kb.fetch_rules(CONN, ontology="o").is_empty()


def test_load_rules_skips_unknown_rule_type_and_blank_text(monkeypatch):
    kb.clear_rules_cache()
    monkeypatch.setattr(kb.config, "enable_knowledge_base", True)
    monkeypatch.setattr(kb.KBClient, "resolve_knowledge_bases", lambda self, **kw: ["kb1"])
    rows = [
        {"knowledge_base": "kb1", "rule_type": "SELECTION_RULE", "target_name": "orders",
         "target_type": "concept", "instructions": "keep", "changed_on": None},
        {"knowledge_base": "kb1", "rule_type": "BOGUS", "target_name": "orders",
         "target_type": "concept", "instructions": "drop-unknown-type", "changed_on": None},
        {"knowledge_base": "kb1", "rule_type": "INSTRUCTION", "target_name": "orders",
         "target_type": "concept", "instructions": "   ", "changed_on": None},
    ]
    monkeypatch.setattr(kb, "run_query", lambda sql, cp: list(rows))
    rs = kb.fetch_rules(CONN, ontology="o")
    assert rs.rules_for("orders", ("concept",), {"selection", "instruction", "validation"}) == {
        "selection": ["keep"]
    }


# --------------------------------------------------------------------------- #
# Stage: concept_prefilter — concept/view/cube SELECTION_RULE, no relationship
# --------------------------------------------------------------------------- #
class _Meta:
    def __init__(self, desc):
        self.description = desc


class _Ont:
    def get_concept_metadata(self, name):
        return _Meta(f"{name} desc")


def test_prefilter_injects_concept_selection_not_relationship():
    rs = _ruleset({
        ("concept", "orders"): {"selection": ["only recent orders"]},
        ("relationship", "placed_by"): {"selection": ["should not appear"]},
    })
    cands = _gather_candidates(["orders"], _Ont(), rules=rs)
    block = render_with_descriptions(cands)
    assert "selection_rules: only recent orders" in block
    assert "should not appear" not in block
    # names-only render path also carries the rule (no cascade trimming of rules)
    assert "only recent orders" in render_names_only(cands)


def test_prefilter_backward_compat_without_rules():
    base = render_with_descriptions(_gather_candidates(["orders"], _Ont()))
    assert base == "- orders: orders desc"
    assert "selection_rules" not in base


# --------------------------------------------------------------------------- #
# Stage: filter — relationship SELECTION_RULE, in-subgraph only, both builders
# --------------------------------------------------------------------------- #
class _Edge:
    def __init__(self, rel):
        self.relationship_name = rel


def test_filter_relationship_rules_in_subgraph_only():
    rs = _ruleset({
        ("relationship", "placed_by"): {"selection": ["active customers only"]},
        ("relationship", "not_in_graph"): {"selection": ["excluded"]},
    })
    block = _render_relationship_rules_block(rs, [_Edge("placed_by")])
    assert "placed_by" in block
    assert "active customers only" in block
    assert "not_in_graph" not in block and "excluded" not in block
    # no edges / no rules -> empty
    assert _render_relationship_rules_block(rs, []) == ""
    assert _render_relationship_rules_block(_ruleset({}), [_Edge("placed_by")]) == ""


def test_filter_messages_carry_rules_block_in_both_builders():
    block = "Relationship selection rules:\n- `placed_by`:\n  selection_rules: active customers only"
    kwargs = dict(question="q", anchor="a", compact_ddl="ddl")
    m_filter = build_filter_messages(rules_block=block, **kwargs)
    m_retry = build_retry_messages(error_lines=["e"], rules_block=block, **kwargs)
    assert "active customers only" in m_filter[1]["content"]
    assert "active customers only" in m_retry[1]["content"]

    # backward-compat: empty rules_block -> byte-identical user message
    assert build_filter_messages(**kwargs)[1]["content"] == build_filter_messages(rules_block="", **kwargs)[1]["content"]
    assert (
        build_retry_messages(error_lines=["e"], **kwargs)[1]["content"]
        == build_retry_messages(error_lines=["e"], rules_block="", **kwargs)[1]["content"]
    )


# --------------------------------------------------------------------------- #
# Stage: generate_sql — property/measure/relationship + concept, NO concept selection
# --------------------------------------------------------------------------- #
def test_generate_sql_property_measure_relationship_rules_inline():
    rs = _sample_ruleset()
    cols = _build_columns_str(
        [{"col_name": "amount", "name": "amount", "data_type": "double"}],
        rules=rs, target_type="property",
    )
    assert "selection_rules: prefer net" in cols
    assert "instructions: round to cents" in cols
    assert "validation_rules: must be positive" in cols

    meas = _build_columns_str(
        [{"col_name": "measure.total", "name": "total"}], rules=rs, target_type="measure",
    )
    assert "instructions: sum excludes tax" in meas

    rel = _build_rel_columns_str(
        {"placed_by": {"description": "d", "columns": [{"col_name": "x", "name": "x"}]}},
        rules=rs,
    )
    assert "Rules for placed_by relationship" in rel
    assert "selection_rules: active customers only" in rel
    assert "validation_rules: one per order" in rel


def test_generate_sql_concept_gets_instruction_validation_not_selection():
    rs = _sample_ruleset()
    items = _rule_meta_items(rs, "orders", ("concept", "view", "cube"), ("instruction", "validation"))
    joined = "; ".join(items)
    assert "use gross amount" in joined       # instruction
    assert "reject cancelled" in joined       # validation
    assert "only recent orders" not in joined  # SELECTION_RULE never re-injected here


def test_generate_sql_backward_compat_without_rules():
    col = [{"col_name": "amount", "name": "amount", "data_type": "double"}]
    base = _build_columns_str(col)
    assert "selection_rules" not in base
    # empty ruleset -> byte-identical to no-rules baseline
    assert _build_columns_str(col, rules=_ruleset({}), target_type="property") == base
    rel_arg = {"placed_by": {"description": "d", "columns": [{"col_name": "x", "name": "x"}]}}
    assert _build_rel_columns_str(rel_arg, rules=_ruleset({})) == _build_rel_columns_str(rel_arg)


# --------------------------------------------------------------------------- #
# Stage: reasoning — concept VALIDATION only
# --------------------------------------------------------------------------- #
def test_reasoning_appends_only_validation():
    rs = _sample_ruleset()
    txt = kb.render_object_rules(rs.rules_for("orders", ("concept", "view", "cube"), {"validation"}))
    assert txt == "validation_rules: reject cancelled"

    msgs = [HumanMessage(content="base")]
    _append_reasoning_context_blocks(msgs, validation_rules=txt)
    assert "Knowledge Base Validation Rules" in msgs[0].content
    assert "reject cancelled" in msgs[0].content


def test_reasoning_backward_compat_without_rules():
    msgs = [HumanMessage(content="base")]
    _append_reasoning_context_blocks(msgs, validation_rules="")
    assert msgs[0].content == "base"


# --------------------------------------------------------------------------- #
# Stage: identify_concept — SELECTION_RULE inline under concept lines
# --------------------------------------------------------------------------- #
def test_append_rules_subblock_indents_and_noops_on_empty():
    assert _append_rules_subblock("`orders`", "") == "`orders`"
    assert _append_rules_subblock("`orders`", "selection_rules: x") == "`orders`\n  selection_rules: x"


def _catalog_rows():
    return {
        "concepts": [
            {"concept": "thing"},
            {"concept": "customer", "inheritance": "thing", "inheritance_level": 1},
            {"concept": "order", "inheritance": "thing", "inheritance_level": 1},
        ],
        "properties": [{"property_name": "name", "is_measure": "false"}],
        "concept_properties": [{"concept": "customer", "property_name": "name"}],
        "relationships": [],
        "views": [],
        "view_properties": [],
    }


def test_identify_catalog_lines_inject_concept_selection(monkeypatch):
    catalog = icc._build_catalog(_catalog_rows())
    monkeypatch.setattr(icc, "_load_catalog", lambda conn_params: catalog)
    rs = _ruleset({("concept", "customer"): {"selection": ["only active customers"]}})
    cv = {"customer": {"concept": "customer", "description": "customer desc", "is_view": "false"}}

    out = "\n".join(icc.build_catalog_lines("list customers", {"ontology": "ont"}, cv, rules=rs))
    assert "selection_rules: only active customers" in out
    # backward-compat: without rules, no rule text appears
    base = "\n".join(icc.build_catalog_lines("list customers", {"ontology": "ont"}, cv))
    assert "selection_rules" not in base
