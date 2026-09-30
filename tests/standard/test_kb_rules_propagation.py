"""A KB rule must reach every stage that needs it, and what reasoning reports
must describe the SQL that is returned.

Reported as: "the rule shows up in the concept-selection / reasoning text but
the generated SQL ignores it". Four independent causes, all covered here:

  * a concept SELECTION_RULE ("guest means provider") steered concept selection
    and was then withheld from SQL generation;
  * a rule on a logic sub-type (``provider``) was shown nowhere when logic
    concepts are not selectable — the query runs on the parent (``party``);
  * a rule on a concept that is only joined to was shown to the reasoning
    evaluator but never to the SQL generator;
  * the reasoning pass lost its own feedback on the validation retry, returned a
    reason that did not belong to the returned SQL, always reported 'correct',
    and re-planned from a cache that ignored the feedback.

Everything runs against in-memory fakes — no DB, no LLM, no prompt service.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_timbr import identify_concept_context as ICC
from langchain_timbr import kbclient as kb
from langchain_timbr.ontology_context.context_builder import build_filtered, concept_prefilter
from langchain_timbr.utils import timbr_llm_utils as TLU

CONN = {"url": "u", "token": "t", "ontology": "contracts"}

GUEST_RULE = "Guest is synonym for provider"
ALIAS_RULE = "Goods Warhousing T.A. is also referred to as Welcome Ship Co"
NAME_RULE = "party_name holds the legal entity name"

PARTY_COLUMNS = [
    {"col_name": "party_name", "name": "party_name", "data_type": "string"},
    {"col_name": "_type_of_client", "name": "_type_of_client", "data_type": "int"},
    {"col_name": "_type_of_provider", "name": "_type_of_provider", "data_type": "int"},
]
TERM_COLUMNS = [{"col_name": "contract_term_name", "name": "contract_term_name", "data_type": "string"}]
PARTY_REL = "has_contract_term[contract].party_to_contract[party]"
TERM_RELS = {
    PARTY_REL: {
        "columns": [{"col_name": "party_name", "name": f"{PARTY_REL}.party_name", "data_type": "string"}],
        "measures": [],
        "description": "",
    }
}
# provider/client are logic sub-types of party; the chain lists direct parents only
INHERITANCE = {"provider": ("party",), "client": ("party",), "party": ("legal_entity",)}


def _rules(by_target):
    return kb.RuleSet(by_target=by_target, kb_names=("kb",), version=None)


def _ontology(store=None):
    store = {} if store is None else store
    return SimpleNamespace(
        inheritance_chain_of=lambda name: INHERITANCE.get(name, ()),
        get_filtered_cache=store.get,
        set_filtered_cache=store.__setitem__,
    )


# --------------------------------------------------------------------------- #
# SQL-generation context (the real builder, metadata lookups stubbed)
# --------------------------------------------------------------------------- #
@pytest.fixture
def build_ctx(monkeypatch):
    import langchain_timbr.ontology_context.ontology.shared as shared

    def _make(concept, columns, relationships=None):
        monkeypatch.setattr(TLU, "_get_active_datasource", lambda cp: {"target_type": "databricks"})
        monkeypatch.setattr(TLU, "get_properties_description", lambda conn_params: {})
        monkeypatch.setattr(TLU, "get_relationships_description", lambda conn_params: {})
        monkeypatch.setattr(TLU, "get_tags", lambda conn_params, include_tags=None: {"property_tags": {}})
        monkeypatch.setattr(
            TLU, "get_concept_properties",
            lambda **kw: {"columns": columns, "measures": [], "relationships": relationships or {}},
        )
        monkeypatch.setattr(shared, "get_shared_ontology", lambda cp: _ontology())

        def _run(rules):
            return TLU._build_sql_generation_context(
                question="q", conn_params=CONN, schema="dtimbr", concept=concept,
                concept_metadata={"description": "", "tags": ""}, graph_depth=1,
                include_tags=None, exclude_properties=None, db_is_case_sensitive=False,
                max_limit=100, enable_technical_context=False, metadata_context_mode="static",
                rules=rules,
            )
        return _run
    return _make


def _evaluator_rules(rules, ctx, sql):
    return TLU._collect_reasoning_rules(
        rules, ctx["concept"], sql, ctx["metadata_plan"]["rule_concepts"],
    )


def test_anchor_selection_rule_reaches_sql_generation_and_evaluator(build_ctx):
    rules = _rules({("concept", "party"): {"selection": [GUEST_RULE]}})
    ctx = build_ctx("party", PARTY_COLUMNS)(rules)

    assert f"selection_rules: {GUEST_RULE}" in ctx["concept_description"]
    assert GUEST_RULE in _evaluator_rules(rules, ctx, "SELECT party_name FROM dtimbr.party")


def test_subtype_rule_is_listed_under_the_parent_with_its_type_flag(build_ctx):
    """The rule names `provider`; the query runs on `party`, which reaches
    `provider` only through the `_type_of_provider` flag."""
    rules = _rules({("concept", "provider"): {"selection": [GUEST_RULE]}})
    ctx = build_ctx("party", PARTY_COLUMNS)(rules)

    assert (
        "- sub-type `provider` (rows where `_type_of_provider` = 1) "
        f"selection_rules: {GUEST_RULE}\n"
    ) in ctx["concept_description"]
    # The evaluator sees it even when the SQL dropped the flag entirely.
    assert GUEST_RULE in _evaluator_rules(rules, ctx, "SELECT party_name FROM dtimbr.party")


def test_subtype_rule_reaches_an_ancestor_further_up(build_ctx):
    """A re-anchor can move the query to a grandparent; the chain only lists
    direct parents, so the sub-type has to be found by walking up."""
    rules = _rules({("concept", "provider"): {"selection": [GUEST_RULE]}})
    ctx = build_ctx("legal_entity", PARTY_COLUMNS)(rules)

    assert "sub-type `provider`" in ctx["concept_description"]
    assert ctx["metadata_plan"]["rule_concepts"] == ["legal_entity", "provider"]


@pytest.mark.parametrize("kind", ["selection", "instruction", "validation"])
def test_joined_concept_rule_reaches_sql_generation(build_ctx, kind):
    """The question anchors on `contract_term`; `party` is only joined to."""
    rules = _rules({("concept", "party"): {kind: [ALIAS_RULE]}})
    ctx = build_ctx("contract_term", TERM_COLUMNS, TERM_RELS)(rules)

    line = [l for l in ctx["concept_description"].split("\n") if ALIAS_RULE in l]
    assert len(line) == 1, "shown once, with the anchor's own rules"
    assert line[0].startswith(f"- Rules for `party` (reached through `{PARTY_REL}`): ")


def test_joined_concept_subtype_rule_names_the_flag_through_the_relationship(build_ctx):
    rules = _rules({("concept", "provider"): {"selection": [GUEST_RULE]}})
    ctx = build_ctx("contract_term", TERM_COLUMNS, TERM_RELS)(rules)

    assert (
        f"sub-type `provider` (rows where `{PARTY_REL}._type_of_provider` = 1) "
        f"selection_rules: {GUEST_RULE}"
    ) in ctx["concept_description"]


def test_property_rule_reaches_relationship_columns(build_ctx):
    rules = _rules({("property", "party_name"): {"instruction": [NAME_RULE]}})
    ctx = build_ctx("contract_term", TERM_COLUMNS, TERM_RELS)(rules)

    assert f"instructions: {NAME_RULE}" in ctx["measures_context"]


def test_relationship_rule_is_matched_by_name_under_a_path_keyed_block(build_ctx):
    """Relationship blocks are keyed by their full path (`a[b].rel[c]`); the rule
    is on `rel`. Looking the rule up by the key never matched."""
    rules = _rules({("relationship", "party_to_contract"): {"instruction": ["one party per role"]}})
    ctx = build_ctx("contract_term", TERM_COLUMNS, TERM_RELS)(rules)

    assert f"- Rules for {PARTY_REL} relationship: instructions: one party per role" in ctx["measures_context"]


def test_rules_come_with_the_instruction_to_apply_them(build_ctx):
    """Shown "X is also called Y" and nothing else, the model repeats the rule in
    its reason and still filters on the question's wording."""
    run = build_ctx("contract_term", TERM_COLUMNS, TERM_RELS)

    shown = run(_rules({("concept", "party"): {"instruction": [ALIAS_RULE]}}))
    assert shown["concept_description"].count(TLU.KB_RULES_DIRECTIVE) == 1

    # only when a rule is actually in the prompt
    unrelated = run(_rules({("concept", "invoice"): {"instruction": ["not in play"]}}))
    assert TLU.KB_RULES_DIRECTIVE not in json.dumps(unrelated)


def test_evaluator_is_never_shown_a_concept_rule_the_generator_was_not(build_ctx):
    """The original asymmetry: the evaluator matched rule targets against the SQL
    text, so it saw `party`'s rules through the join while the generator did not."""
    rules = _rules({
        ("concept", "party"): {"selection": [ALIAS_RULE]},
        ("concept", "provider"): {"instruction": [GUEST_RULE]},
        ("concept", "invoice"): {"instruction": ["not in play"]},
    })
    ctx = build_ctx("contract_term", TERM_COLUMNS, TERM_RELS)(rules)
    generator_prompt = json.dumps(ctx)
    sql = f"SELECT 1 FROM dtimbr.contract_term WHERE `{PARTY_REL}.party_name` = 'x'"

    evaluator = _evaluator_rules(rules, ctx, sql)
    for text in (ALIAS_RULE, GUEST_RULE):
        assert text in generator_prompt and text in evaluator
    assert "not in play" not in generator_prompt and "not in play" not in evaluator


def test_no_rules_leaves_the_context_unchanged(build_ctx):
    run = build_ctx("contract_term", TERM_COLUMNS, TERM_RELS)
    base, empty = run(None), run(_rules({}))

    for key in ("concept_description", "columns_str", "measures_context"):
        assert base[key] == empty[key]
    assert "Rules for" not in base["measures_context"]


# --------------------------------------------------------------------------- #
# identify-concept: a non-candidate sub-type's rule is listed under its parent
# --------------------------------------------------------------------------- #
def _catalog():
    nodes = {
        "party": ICC._Node(name="party"),
        "provider": ICC._Node(name="provider", parents=["party"]),
        "contract": ICC._Node(name="contract"),
    }
    return ICC._Catalog(nodes=nodes, children={"party": ["provider"]})


def test_identify_lists_non_candidate_subtype_rules_under_the_parent():
    rules = _rules({("concept", "provider"): {"selection": [GUEST_RULE]}})

    got = ICC.subtype_selection_rules(_catalog(), {"party", "contract"}, rules)
    assert got == {"party": [("provider", f"selection_rules: {GUEST_RULE}")]}


def test_identify_does_not_repeat_a_candidate_subtypes_rules():
    """With logic concepts selectable, `provider` has its own line and rules."""
    rules = _rules({("concept", "provider"): {"selection": [GUEST_RULE]}})

    assert ICC.subtype_selection_rules(_catalog(), {"party", "provider"}, rules) == {}


def test_identify_catalog_renders_subtype_rules(monkeypatch):
    monkeypatch.setattr(ICC, "_load_catalog", lambda cp: _catalog())
    rules = _rules({("concept", "provider"): {"selection": [GUEST_RULE]}})
    candidates = {"party": {"concept": "party"}, "contract": {"concept": "contract"}}

    out = "\n".join(ICC.build_catalog_lines("show all guests", CONN, candidates, rules=rules))
    assert f"sub-type `provider` selection_rules: {GUEST_RULE}" in out
    assert "sub-type" not in "\n".join(ICC.build_catalog_lines("show all guests", CONN, candidates))


def test_prefilter_lists_non_candidate_subtype_rules_under_the_parent():
    ontology = SimpleNamespace(
        get_concept_metadata=lambda name: SimpleNamespace(description=""),
        inheritance_chain_of=lambda name: INHERITANCE.get(name, ()),
    )
    rules = _rules({("concept", "provider"): {"selection": [GUEST_RULE]}})

    by_name = {c.name: c for c in concept_prefilter._gather_candidates(["party", "contract"], ontology, rules)}
    assert by_name["party"].rules_text == f"sub-type `provider` selection_rules: {GUEST_RULE}"
    assert by_name["contract"].rules_text == ""

    # nearest candidate ancestor, found by walking up past a non-candidate parent
    by_name = {c.name: c for c in concept_prefilter._gather_candidates(["legal_entity"], ontology, rules)}
    assert by_name["legal_entity"].rules_text == f"sub-type `provider` selection_rules: {GUEST_RULE}"


# --------------------------------------------------------------------------- #
# relationship planner: a rule can be the only reason to keep a path
# --------------------------------------------------------------------------- #
def test_planner_is_shown_the_rules_of_the_concepts_it_chooses_among():
    """"discount for guests" anchors on `contract_term`. Nothing in the schema
    ties "guests" to `party`; the rule does. A planner that cannot see it drops
    the path, and the SQL generator then gets neither the columns nor the rule."""
    rules = _rules({
        ("concept", "party"): {"instruction": [ALIAS_RULE], "validation": ["names are unique"]},
        ("concept", "provider"): {"selection": [GUEST_RULE]},
        ("concept", "invoice"): {"selection": ["not in the neighbourhood"]},
    })

    block = build_filtered._render_concept_rules_block(
        rules, ["contract_term", "contract", "party"], _ontology(),
    )
    assert block == (
        "Concept rules:\n"
        "- `party`:\n"
        f"  instructions: {ALIAS_RULE}\n"
        f"  sub-type `provider` selection_rules: {GUEST_RULE}"
    )


def test_planner_rules_block_joins_relationship_and_concept_rules():
    edge = SimpleNamespace(relationship_name="party_to_contract")
    rules = _rules({
        ("relationship", "party_to_contract"): {"selection": ["prefer the signing party"]},
        ("concept", "party"): {"selection": [GUEST_RULE]},
    })

    block = build_filtered._render_planner_rules_block(rules, [edge], ["party"], _ontology())
    assert block.startswith("Relationship selection rules:\n- `party_to_contract`:")
    assert "\n\nConcept rules:\n- `party`:\n  selection_rules: " + GUEST_RULE in block

    # nothing applicable -> "" (the planner prompt stays byte-identical)
    assert build_filtered._render_planner_rules_block(rules, [], ["contract"], _ontology()) == ""
    assert build_filtered._render_planner_rules_block(None, [edge], ["party"], _ontology()) == ""


def test_planner_is_not_shown_the_anchors_own_rules(monkeypatch):
    """The anchor's rules already decided concept selection. Shown to the planner,
    a synonym rule on the anchor made it re-anchor away ("show all guests" on
    `party` moved to another concept). Only the neighbours' rules are passed —
    and the anchor's must not come back as a "sub-type" rule under its own
    parent (`legal_entity`) when that parent is in the neighbourhood."""
    seen = []
    monkeypatch.setattr(build_filtered, "EdgeIndex", lambda ontology: None)
    monkeypatch.setattr(
        build_filtered, "_build_subgraph_and_ddl",
        lambda **kw: (
            ["party", "contract", "legal_entity"], {}, [], "ddl",
            [SimpleNamespace(concept="contract_term")],
        ),
    )

    def _step1(**kw):
        seen.append(kw["rules_block"])
        raise RuntimeError("stop after the first planner call")
    monkeypatch.setattr(build_filtered, "_step1_with_validation_retries", _step1)

    rules = _rules({
        ("concept", "party"): {"selection": [GUEST_RULE]},
        ("concept", "provider"): {"selection": [GUEST_RULE]},   # sub-type of the anchor
        ("concept", "contract_term"): {"instruction": ["terms are per contract"]},
    })
    build_filtered.build_filtered_metadata(
        question="show all guests", anchor="party", ontology=_ontology(), llm=None,
        config=SimpleNamespace(max_graph_depth=3), graph_depth=1, rules=rules,
    )

    assert seen == ["Concept rules:\n- `contract_term`:\n  instructions: terms are per contract"]


# --------------------------------------------------------------------------- #
# generate_sql: reasoning + validation flow (fake LLM, stubbed context builder)
# --------------------------------------------------------------------------- #
SQL_FIRST = "SELECT party_name FROM dtimbr.party WHERE party_name = 'Welcome Ship Co'"
SQL_REASONED = "SELECT party_name FROM dtimbr.party WHERE party_nme = 'Goods Warhousing T.A.'"
SQL_VALIDATED = "SELECT party_name FROM dtimbr.party WHERE party_name = 'Goods Warhousing T.A.'"
CRITIQUE = "Per the KB, Welcome Ship Co is Goods Warhousing T.A.; filter on that name."
FIRST_REASON = "first pass reason"
REGEN_REASON = "Applied the KB synonym: filtered on Goods Warhousing T.A."
FIRST_PLAN = {"anchor": "party", "relationships": ["has_party[contract]"], "rule_concepts": ["party"]}


class _Template:
    def __init__(self, name, renders):
        self.name, self.renders = name, renders

    def format_messages(self, **kw):
        self.renders.append({"template": self.name, **kw})
        return [SystemMessage(content=f"sys:{self.name}"), HumanMessage(content=json.dumps(kw, default=str))]


class _LLM:
    _llm_type = "fake"

    def __init__(self, assessments):
        self._assessments = list(assessments)

    def invoke(self, prompt):
        text = TLU._prompt_to_string(prompt)
        if "sys:reasoning_eval" in text:
            verdict = self._assessments.pop(0)
            if isinstance(verdict, Exception):
                raise verdict
            return AIMessage(content=json.dumps({"assessment": verdict, "reasoning": CRITIQUE}))
        if "sys:after_validate" in text:
            return AIMessage(content=json.dumps({"result": SQL_VALIDATED}))
        if "was assessed as" in text:  # reasoning regeneration: the note carries the critique
            return AIMessage(content=json.dumps({"result": SQL_REASONED, "reason": REGEN_REASON}))
        return AIMessage(content=json.dumps({"result": SQL_FIRST, "reason": FIRST_REASON}))


@pytest.fixture
def flow(monkeypatch):
    renders, builds = [], []
    state = {"validate": [], "degrade_on_build": None}

    def _context(**kw):
        builds.append(kw)
        if state["degrade_on_build"] == len(builds) and kw.get("status_sink") is not None:
            kw["status_sink"]["metadata_context_degraded"] = True
        return {
            "cur_date": "d", "datasource_type": "x", "schema": "dtimbr", "concept": "party",
            "concept_description": "", "concept_tags": "", "columns_str": "`party_name`",
            "measures_context": "", "transitive_context": "", "sensitivity_txt": "", "max_limit": 100,
            "metadata_plan": dict(FIRST_PLAN),
        }

    def _template(conn_params=None, reasoning=False, after_validate=False):
        name = "after_validate" if after_validate else ("reasoning" if reasoning else "generate_sql")
        return _Template(name, renders)

    def _validate(sql, conn_params):
        ok, err = state["validate"].pop(0) if state["validate"] else (True, None)
        return ok, err, sql

    monkeypatch.setattr(TLU, "_build_sql_generation_context", _context)
    monkeypatch.setattr(TLU, "get_generate_sql_prompt_template", _template)
    monkeypatch.setattr(
        TLU, "get_generate_sql_reasoning_prompt_template", lambda cp: _Template("reasoning_eval", renders)
    )
    monkeypatch.setattr(TLU, "validate_sql", _validate)
    monkeypatch.setattr(TLU, "determine_concept", lambda **kw: {
        "concept": "party", "schema": "dtimbr", "concept_metadata": {}, "identify_concept_reason": "r",
        "usage_metadata": {}, "duration_ms": 0, "conn_params": CONN,
    })

    def run(assessments, **over):
        kw = dict(
            question="show details about Welcome Ship Co", llm=_LLM(assessments), conn_params=CONN,
            concept="party", schema="dtimbr", should_validate_sql=True, retries=2,
            enable_reasoning=True, reasoning_steps=1, max_graph_depth=3,
            metadata_context_mode="static", note="",
        )
        kw.update(over)
        return TLU.generate_sql(**kw)

    return SimpleNamespace(run=run, renders=renders, builds=builds, state=state)


def test_validation_retry_keeps_the_reasoning_feedback(flow):
    """The rewrite that applied the fix fails validation. The retry must still
    know why the SQL was rewritten, or it 'fixes' the error by undoing the fix."""
    flow.state["validate"] = [(False, "column party_nme not found"), (True, None)]
    res = flow.run(["partial"])

    retry = [r for r in flow.renders if r["template"] == "after_validate"][0]
    assert CRITIQUE in retry["note"]
    assert "column party_nme not found" in retry["note"]
    assert res["sql"] == SQL_VALIDATED
    assert res["generate_sql_reason"] == (
        f"{REGEN_REASON}\n\nAdjusted after validation error: column party_nme not found"
    )


def test_reasoning_status_is_the_evaluators_verdict(flow):
    assert flow.run(["correct"])["reasoning_status"] == "correct"


def test_a_rewrite_made_from_the_feedback_is_reported_as_correct(flow):
    """No step is left to re-evaluate it, and 'partial' was the verdict on the
    SQL it replaced. The reason is the rewrite's own, not the evaluator's."""
    res = flow.run(["partial"])

    assert res["sql"] == SQL_REASONED
    assert res["reasoning_status"] == "correct"
    assert res["generate_sql_reason"] == REGEN_REASON


def test_a_rewrite_confirmed_by_a_later_step_is_correct(flow):
    res = flow.run(["partial", "correct"], reasoning_steps=2)

    assert res["sql"] == SQL_REASONED
    assert res["reasoning_status"] == "correct"
    assert res["generate_sql_reason"] == CRITIQUE  # the evaluator's text, about this same SQL


def test_degraded_regeneration_keeps_the_first_sql_and_its_own_reason(flow):
    """The loop stops without rewriting. Reporting the evaluator's critique as
    the reason described a fix the returned SQL does not contain."""
    flow.state["degrade_on_build"] = 2  # 1 = first pass, 2 = reasoning regeneration
    res = flow.run(["partial"])

    assert res["sql"] == SQL_FIRST
    assert res["generate_sql_reason"] == FIRST_REASON
    assert res["reasoning_status"] == "partial"


def test_failed_evaluation_keeps_the_first_reason_and_stays_correct(flow):
    res = flow.run([RuntimeError("evaluator down")])

    assert res["sql"] == SQL_FIRST
    assert res["generate_sql_reason"] == FIRST_REASON
    assert res["reasoning_status"] == "correct"


def test_without_reasoning_the_status_stays_correct(flow):
    assert flow.run([], enable_reasoning=False)["reasoning_status"] == "correct"


def test_regeneration_hands_the_previous_plan_to_the_context_builder(flow):
    flow.run(["partial"])

    assert flow.builds[0].get("previous_plan") is None
    assert flow.builds[1]["previous_plan"] == FIRST_PLAN


def test_evaluator_receives_the_generators_rule_concepts(flow, monkeypatch):
    seen = []
    real = TLU._collect_reasoning_rules
    monkeypatch.setattr(
        TLU, "_collect_reasoning_rules",
        lambda rules, concept, sql, context_concepts=None: seen.append(context_concepts)
        or real(rules, concept, sql, context_concepts),
    )
    flow.run(["correct"])

    assert seen == [FIRST_PLAN["rule_concepts"]]


# --------------------------------------------------------------------------- #
# Dynamic metadata-context: cache key and re-plan
# --------------------------------------------------------------------------- #
@pytest.fixture
def dynamic(monkeypatch):
    import langchain_timbr.ontology_context as OC

    planner_notes: list = []
    monkeypatch.setattr(OC, "get_shared_ontology", lambda cp, _o=_ontology(): _o)

    def _planner(**kw):
        planner_notes.append(kw["note"])
        return SimpleNamespace(
            accepted_overrides=None, stats={}, error=None, effective_anchor=None,
            filtered_concepts={"party"}, validated_paths=[], compact_ddl="",
        )
    monkeypatch.setattr(OC, "build_filtered_metadata", _planner)

    def call(**over):
        kw = dict(
            mode="dynamic", question="q", anchor="party", conn_params=CONN, graph_depth=1,
            columns=PARTY_COLUMNS, measures=[], tags={}, exclude_properties=None,
            static_columns_str="c", static_measures_str="", static_rel_prop_str="",
            llm=None, config_overrides={}, note="",
        )
        kw.update(over)
        return TLU._apply_dynamic_metadata_context(**kw)

    return SimpleNamespace(call=call, planner_notes=planner_notes)


def test_identical_planner_inputs_are_served_from_cache(dynamic):
    sink: dict = {}
    dynamic.call()
    dynamic.call(plan_sink=sink)

    assert len(dynamic.planner_notes) == 1
    assert sink["anchor"] == "party" and sink["relationships"] == [] and sink["joined"] == {}


def test_evaluator_feedback_in_the_note_forces_a_replan(dynamic):
    dynamic.call()
    dynamic.call(note=CRITIQUE)

    assert dynamic.planner_notes == ["", CRITIQUE]


def test_a_rule_change_forces_a_replan(dynamic):
    dynamic.call(rules=_rules({("concept", "party"): {"selection": [GUEST_RULE]}}))
    dynamic.call(rules=_rules({("concept", "party"): {"selection": [ALIAS_RULE]}}))
    dynamic.call(rules=_rules({}))

    assert len(dynamic.planner_notes) == 3


def test_replan_is_shown_the_previous_selection_to_revalidate(dynamic):
    dynamic.call(note=CRITIQUE, previous_plan=FIRST_PLAN)

    note = dynamic.planner_notes[0]
    assert note.startswith(CRITIQUE)
    assert "Previous schema selection (re-validate)" in note
    assert "anchor `party`" in note and "`has_party[contract]`" in note
