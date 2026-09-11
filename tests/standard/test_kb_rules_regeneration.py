"""Knowledge-base context must survive every SQL regeneration path.

``generate_sql`` threaded ``rules`` into the first ``_build_sql_generation_context``
call and nowhere else, so both regeneration paths — the reasoning retry and the
validation retry — built a rule-free prompt. ``rules=None`` is the documented
"inject nothing" value, so the loss was completely silent: a concept INSTRUCTION
that the first pass honored simply stopped existing on the retry.

Everything here runs against in-memory fakes — no DB, no LLM, no prompt service.
The context builder is replaced by a stub that renders through the REAL rule
helpers (``_rule_meta_items`` / ``_build_columns_str``), so these tests exercise
the wiring — does ``rules`` arrive? — on top of rendering that is already covered
by ``test_kb_rules.py``.

Covered:
  reasoning retry            -> receives the same RuleSet as the first pass
  validation retry           -> reuses the first pass's context verbatim
  concept INSTRUCTION        -> present in EVERY generation prompt
  property SELECTION         -> present in EVERY generation prompt
  reasoning evaluator        -> concept instruction + rules for objects named in the SQL
  KB golden examples         -> present in every generation prompt and the evaluator
  ValidateTimbrSqlChain      -> regenerates with rules + memory
  no rules                   -> byte-identical prompts (backward compat)
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_timbr import kbclient as kb
from langchain_timbr.utils import timbr_llm_utils as TLU
from langchain_timbr.utils.memory import MemoryContext


CONN = {"url": "u", "token": "t", "ontology": "ont"}
CONCEPT = "shipment"

# Sanitized stand-ins for the reported rules. No customer names or columns.
DATE_RULE = "use ship_date for date filtering unless another date is explicitly specified"
SYNONYM_RULE = "'Acme Haulage' is also referred to as 'Northwind Freight'"
LOAD_RULE = "load is the unique truck identifier"

COLUMNS = [
    {"col_name": "ship_date", "name": "ship_date", "data_type": "date"},
    {"col_name": "depart_ts", "name": "depart_ts", "data_type": "timestamp"},
    {"col_name": "carrier_name", "name": "carrier_name", "data_type": "string"},
]

SQL_FIRST = "SELECT COUNT(*) FROM dtimbr.`shipment` WHERE ship_date = '2026-09-11'"
SQL_RETRY = "SELECT COUNT(*) FROM dtimbr.`shipment` WHERE ship_date >= '2026-09-11'"


def _ruleset():
    return kb.RuleSet(
        by_target={
            ("concept", CONCEPT): {"instruction": [DATE_RULE]},
            ("property", "carrier_name"): {"selection": [SYNONYM_RULE]},
            ("property", "load"): {"instruction": [LOAD_RULE]},
        },
        kb_names=("kb1",),
        version=None,
    )


def _empty_ruleset():
    return kb.RuleSet(by_target={}, kb_names=("kb1",), version=None)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class _FakeTemplate:
    """Renders the SQL-gen template kwargs as a JSON HumanMessage.

    The system message carries the template's identity so the fake LLM can tell
    which call it is answering, and so assertions can slice renders by template.
    """

    def __init__(self, name: str, renders: list):
        self.name = name
        self.renders = renders

    def format_messages(self, **kwargs):
        self.renders.append({"template": self.name, **kwargs})
        return [
            SystemMessage(content=f"sys:{self.name}"),
            HumanMessage(content=json.dumps(kwargs, default=str)),
        ]


class _FakeReasoningTemplate:
    """Evaluator template. Captures the message LIST, not a copy of the kwargs —
    ``_append_reasoning_context_blocks`` mutates it in place, so the appended
    rule blocks are only visible through the list itself."""

    def __init__(self, renders: list):
        self.renders = renders

    def format_messages(self, **kwargs):
        msgs = [
            SystemMessage(content="sys:reasoning_eval"),
            HumanMessage(content=json.dumps(kwargs, default=str)),
        ]
        self.renders.append(msgs)
        return msgs


class _FakeLLM:
    _llm_type = "fake"

    def __init__(self, assessments=None):
        # One entry per reasoning step; exhausted -> "correct" (stop reasoning).
        self._assessments = list(assessments or [])
        self.prompts: list[str] = []

    def invoke(self, prompt):
        text = TLU._prompt_to_string(prompt)
        self.prompts.append(text)
        if "sys:reasoning_eval" in text:
            assessment = self._assessments.pop(0) if self._assessments else "correct"
            return AIMessage(
                content=json.dumps({"assessment": assessment, "reasoning": "evaluator says so"})
            )
        if "sys:after_validate" in text:
            return AIMessage(content=json.dumps({"result": SQL_RETRY}))
        return AIMessage(
            content=json.dumps({"result": SQL_FIRST, "reason": "because", "decisions": []})
        )


@pytest.fixture
def harness(monkeypatch):
    """Stub every boundary ``generate_sql`` crosses and record what it built.

    Returns a dict with:
      ``contexts``  — one entry per ``_build_sql_generation_context`` call (its kwargs)
      ``renders``   — one entry per SQL-gen template render (its kwargs + template name)
      ``evals``     — one message list per reasoning-evaluator render
      ``validate``  — mutable list of ``validate_sql`` outcomes to hand out, in order
    """
    contexts: list[dict] = []
    renders: list[dict] = []
    evals: list[list] = []
    state = {"validate": []}

    def _build_context(**kwargs):
        contexts.append(kwargs)
        rules = kwargs.get("rules")
        concept = kwargs["concept"]
        # Render through the real helpers so the assertions below see exactly
        # what the production prompt would carry.
        description = ""
        for item in TLU._rule_meta_items(
            rules, concept, TLU._CVC_TYPES, ("instruction", "validation")
        ):
            description += f"- {item}\n"
        return {
            "cur_date": "2026-09-11",
            "datasource_type": "postgres",
            "schema": kwargs["schema"],
            "concept": concept,
            "concept_description": description,
            "concept_tags": "",
            "columns_str": TLU._build_columns_str(
                COLUMNS, rules=rules, target_type="property"
            ),
            "measures_context": "",
            "transitive_context": "",
            "sensitivity_txt": "",
            "max_limit": kwargs["max_limit"],
        }

    def _gen_template(conn_params=None, reasoning=False, after_validate=False):
        name = (
            "after_validate" if after_validate else ("reasoning" if reasoning else "generate_sql")
        )
        return _FakeTemplate(name, renders)

    def _validate_sql(sql, conn_params):
        if state["validate"]:
            ok, err = state["validate"].pop(0)
            return ok, err, sql
        return True, None, sql

    monkeypatch.setattr(TLU, "_build_sql_generation_context", _build_context)
    monkeypatch.setattr(TLU, "get_generate_sql_prompt_template", _gen_template)
    monkeypatch.setattr(
        TLU, "get_generate_sql_reasoning_prompt_template", lambda conn_params: _FakeReasoningTemplate(evals)
    )
    monkeypatch.setattr(TLU, "validate_sql", _validate_sql)
    monkeypatch.setattr(
        TLU,
        "determine_concept",
        lambda **kw: {
            "concept": CONCEPT,
            "schema": "dtimbr",
            "concept_metadata": {"description": "a shipment", "tags": ""},
            "identify_concept_reason": "because",
            "usage_metadata": {},
            "duration_ms": 0,
            "conn_params": CONN,
        },
    )

    return {"contexts": contexts, "renders": renders, "evals": evals, "state": state}


def _run(harness, llm, *, rules=None, memory_context=None, validate=(), **overrides):
    """Drive ``generate_sql`` with the harness in place."""
    harness["state"]["validate"] = list(validate)
    kwargs = dict(
        question="how many shipments left today?",
        llm=llm,
        conn_params=CONN,
        concept=CONCEPT,
        schema="dtimbr",
        should_validate_sql=True,
        retries=2,
        enable_reasoning=False,
        reasoning_steps=1,
        # Every chain resolves this from config before calling; passing it here
        # keeps the harness faithful.
        max_graph_depth=5,
        rules=rules,
        memory_context=memory_context,
        metadata_context_mode="static",
    )
    kwargs.update(overrides)
    return TLU.generate_sql(**kwargs)


def _gen_renders(harness):
    """Every SQL-generation render (first pass + every regeneration)."""
    return harness["renders"]


# --------------------------------------------------------------------------- #
# Root cause 2.1 — rules never reached the regeneration context builders
# --------------------------------------------------------------------------- #
def test_reasoning_regen_receives_rules(harness):
    """The reasoning retry must build its context from the SAME RuleSet as the
    first pass. Forced ``partial`` verdict so a regeneration actually happens."""
    rules = _ruleset()
    llm = _FakeLLM(assessments=["partial"])

    _run(harness, llm, rules=rules, enable_reasoning=True, reasoning_steps=1)

    builds = harness["contexts"]
    assert len(builds) >= 2, "expected a first pass plus a reasoning regeneration"
    assert builds[0]["rules"] is rules
    assert builds[1]["rules"] is rules, "reasoning regeneration built a rule-free context"


def test_concept_instruction_survives_every_generation(harness):
    """Report A, sanitized: a concept INSTRUCTION honored by the first pass must
    still be in the prompt of every regeneration. One reasoning retry plus one
    validation retry — the rule text must appear in all three prompts."""
    rules = _ruleset()
    llm = _FakeLLM(assessments=["partial"])

    _run(
        harness,
        llm,
        rules=rules,
        enable_reasoning=True,
        reasoning_steps=1,
        validate=[(False, "syntax error near DATE")],
    )

    renders = _gen_renders(harness)
    assert len(renders) >= 3, f"expected first pass + reasoning retry + validation retry, got {len(renders)}"
    for idx, render in enumerate(renders):
        assert DATE_RULE in render["description"], (
            f"generation call {idx} ({render['template']}) lost the concept instruction"
        )


def test_property_rule_survives_every_generation(harness):
    """Report B, sanitized: a property-level synonym rule must reach every
    generation prompt, not only the first."""
    rules = _ruleset()
    llm = _FakeLLM(assessments=["partial"])

    _run(
        harness,
        llm,
        rules=rules,
        enable_reasoning=True,
        reasoning_steps=1,
        validate=[(False, "syntax error")],
    )

    renders = _gen_renders(harness)
    assert len(renders) >= 3
    for idx, render in enumerate(renders):
        assert SYNONYM_RULE in render["columns"], (
            f"generation call {idx} ({render['template']}) lost the property rule"
        )


# --------------------------------------------------------------------------- #
# Validation retry — reuses the first pass's context instead of rebuilding
# --------------------------------------------------------------------------- #
def test_validation_retry_reuses_context_without_rebuilding(harness):
    """The SQL was syntactically invalid; the context was not the problem.

    Rebuilding it was both waste and a correctness hazard: the rebuild's static
    strings are hashed into the dynamic-rebuild memo key, so a retry that
    rendered them without rules missed the memo and re-ran the planner LLM.
    Reusing the first pass's context removes both problems at once.
    """
    rules = _ruleset()
    llm = _FakeLLM()

    _run(harness, llm, rules=rules, validate=[(False, "syntax error")])

    assert len(harness["contexts"]) == 1, (
        f"validation retry rebuilt the context ({len(harness['contexts'])} builds); "
        "it must reuse the first pass's"
    )
    renders = _gen_renders(harness)
    assert len(renders) == 2
    assert renders[1]["description"] == renders[0]["description"]
    assert renders[1]["columns"] == renders[0]["columns"]


def test_validation_retry_after_reasoning_reuses_the_reasoning_context(harness):
    """When reasoning regenerated, the validation retry must reuse the LATEST
    context (the reasoning pass's, which may carry a deeper graph), not the
    first pass's."""
    llm = _FakeLLM(assessments=["partial"])

    _run(
        harness,
        llm,
        rules=_ruleset(),
        enable_reasoning=True,
        reasoning_steps=1,
        validate=[(False, "syntax error")],
    )

    # Two builds only: first pass + reasoning regen. The validation retry adds none.
    assert len(harness["contexts"]) == 2
    renders = _gen_renders(harness)
    assert len(renders) == 3
    assert renders[2]["columns"] == renders[1]["columns"]


def test_validation_retry_uses_the_after_validate_template(harness):
    """The retry must request the server's ``generate_sql_after_validate``
    variant (``?reason=true``) so the LLM is not asked to re-derive a full
    reason it already produced."""
    llm = _FakeLLM()

    _run(harness, llm, rules=_ruleset(), validate=[(False, "syntax error")])

    renders = _gen_renders(harness)
    assert renders[0]["template"] == "generate_sql"
    assert renders[1]["template"] == "after_validate"


def test_validation_retry_carries_the_error_but_no_stale_reason(harness):
    """The retry note must name the validation error, and the run's reported
    reason stays the first pass's (the retry template no longer emits one)."""
    llm = _FakeLLM()

    result = _run(harness, llm, rules=_ruleset(), validate=[(False, "syntax error near DATE")])

    assert "syntax error near DATE" in _gen_renders(harness)[1]["note"]
    assert result["generate_sql_reason"] == "because"
    steps = [entry["step"] for entry in result["generate_sql_reasons"]]
    assert steps == ["generate_sql"], f"retry should not append a re-derived reason, got {steps}"


# --------------------------------------------------------------------------- #
# Root cause 2.3 — the evaluator only ever saw concept VALIDATION rules
# --------------------------------------------------------------------------- #
def test_evaluator_sees_concept_instruction_rules(harness):
    """An evaluator blind to the concept's INSTRUCTION rules cannot know that
    ``ship_date`` was mandated, so it marks a compliant query ``partial`` — which
    is what creates the rule-losing retry in the first place."""
    llm = _FakeLLM(assessments=["correct"])

    _run(harness, llm, rules=_ruleset(), enable_reasoning=True, reasoning_steps=1)

    assert harness["evals"], "evaluator was never rendered"
    content = harness["evals"][0][-1].content
    assert DATE_RULE in content


def test_collect_reasoning_rules_includes_objects_named_in_sql():
    rules = _ruleset()
    text = TLU._collect_reasoning_rules(rules, CONCEPT, SQL_FIRST + " AND carrier_name = 'x'")

    assert DATE_RULE in text           # anchor concept, unconditional
    assert SYNONYM_RULE in text        # property named in the SQL
    assert LOAD_RULE not in text       # property absent from the SQL


def test_collect_reasoning_rules_match_is_word_bounded():
    """A rule on ``load`` must not fire for SQL that only mentions
    ``download_count`` — a plain substring test would."""
    rules = _ruleset()
    text = TLU._collect_reasoning_rules(
        rules, CONCEPT, "SELECT download_count FROM dtimbr.`shipment`"
    )

    assert LOAD_RULE not in text
    assert DATE_RULE in text  # the anchor is never gated on SQL presence


def test_collect_reasoning_rules_empty_without_rules():
    assert TLU._collect_reasoning_rules(None, CONCEPT, SQL_FIRST) == ""
    assert TLU._collect_reasoning_rules(_empty_ruleset(), CONCEPT, SQL_FIRST) == ""


def test_reasoning_appendix_suppresses_empty_rules_block():
    msgs = [HumanMessage(content="base")]
    TLU._append_reasoning_context_blocks(msgs, kb_rules="")
    assert msgs[0].content == "base"


# --------------------------------------------------------------------------- #
# KB golden examples — already carried by ``note``; pinned so it stays that way
# --------------------------------------------------------------------------- #
def test_kb_examples_present_in_every_generation_and_the_evaluator(harness):
    memory = MemoryContext(
        is_follow_up=False,
        kb_examples=[
            {
                "knowledge_base": "kb1",
                "example_name": "daily_shipment_count",
                "instructions": "count distinct load per site",
                "query": "SELECT COUNT(DISTINCT load) FROM dtimbr.`shipment`",
            }
        ],
    )
    llm = _FakeLLM(assessments=["partial"])

    _run(
        harness,
        llm,
        rules=_ruleset(),
        memory_context=memory,
        enable_reasoning=True,
        reasoning_steps=1,
        validate=[(False, "syntax error")],
    )

    renders = _gen_renders(harness)
    assert len(renders) >= 3
    for idx, render in enumerate(renders):
        assert "[Approved reference examples]" in render["note"], (
            f"generation call {idx} ({render['template']}) lost the KB examples"
        )
    assert "[Approved reference examples]" in harness["evals"][0][-1].content


# --------------------------------------------------------------------------- #
# Root cause 2.4 — ValidateTimbrSqlChain regenerated with no KB context at all
# --------------------------------------------------------------------------- #
def test_validate_chain_regenerates_with_rules_and_memory(monkeypatch):
    """``ValidateTimbrSqlChain`` accepted ``enable_memory`` but never resolved
    it, and never fetched rules — so its regeneration lost KB rules AND KB
    golden examples, unlike its sibling chains."""
    from langchain_timbr.langchain import validate_timbr_sql_chain as VC

    rules = _ruleset()
    memory = MemoryContext(is_follow_up=False, kb_examples=[{"example_name": "e", "query": "SELECT 1"}])
    captured: dict = {}

    monkeypatch.setattr(VC, "validate_sql", lambda sql, conn: (False, "syntax error", sql))
    monkeypatch.setattr(VC, "fetch_rules", lambda *a, **k: rules, raising=False)
    monkeypatch.setattr("langchain_timbr.kbclient.fetch_rules", lambda *a, **k: rules)
    monkeypatch.setattr("langchain_timbr.utils.memory.resolve_memory", lambda **k: memory)

    def _fake_generate_sql(**kwargs):
        captured.update(kwargs)
        return {"sql": SQL_RETRY, "schema": "dtimbr", "concept": CONCEPT, "is_sql_valid": True,
                "error": None, "usage_metadata": {}, "reasoning_status": "correct",
                "identify_concept_reason": None, "generate_sql_reason": "r"}

    monkeypatch.setattr(VC, "generate_sql", _fake_generate_sql)

    chain = VC.ValidateTimbrSqlChain(
        llm=_FakeLLM(), url="u", token="t", ontology="ont", enable_trace=False,
    )
    chain.invoke({"prompt": "how many shipments?", "sql": "SELECT bad FROM x"})

    assert captured.get("rules") is rules, "validate chain regenerated without KB rules"
    assert captured.get("memory_context") is memory, "validate chain regenerated without KB examples"


# --------------------------------------------------------------------------- #
# Backward compatibility — no rules means byte-identical prompts
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rules", [None, "empty"])
def test_no_rules_leaves_prompts_rule_free(harness, rules):
    resolved = _empty_ruleset() if rules == "empty" else None
    llm = _FakeLLM(assessments=["partial"])

    _run(
        harness,
        llm,
        rules=resolved,
        enable_reasoning=True,
        reasoning_steps=1,
        validate=[(False, "syntax error")],
    )

    for render in _gen_renders(harness):
        assert render["description"] == ""
        assert "selection_rules" not in render["columns"]
        assert "instructions:" not in render["columns"]
    assert "Knowledge Base Rules" not in harness["evals"][0][-1].content
