"""Plan 05 — technical context is built AFTER the planner.

Two halves:

1. ``_build_sql_generation_context`` skips the upfront technical-context pass
   when the dynamic metadata-context pipeline will run (and only then), handing
   the pipeline an EMPTY ``tc_seen_names`` instead. Static mode, non-dtimbr
   schemas, and the kill switch keep the upfront pass. When the dynamic pipeline
   raises, the fallback runs the pass so the prompt still carries statistics.

2. ``_apply_dynamic_metadata_context`` with an empty ``seen`` tops up BOTH the
   rebuilt relationship columns and the flat anchor columns — the top-up is the
   only technical-context build on that path — and does not fire when the
   upfront pass already covered the columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import langchain_timbr.ontology_context as oc
import langchain_timbr.technical_context as tc_pkg
from langchain_timbr import config
from langchain_timbr.ontology_context import DynamicMetadataResult
from langchain_timbr.ontology_context.context_builder.metadata_types import (
    PathSegment,
    SelectedPath,
)
from langchain_timbr.utils import timbr_llm_utils as TLU
from langchain_timbr.utils.timbr_llm_utils import (
    _apply_dynamic_metadata_context,
    _build_sql_generation_context,
)


# ---------------------------------------------------------------------------
# Half 1 — the deferral decision in _build_sql_generation_context
# ---------------------------------------------------------------------------


class _TCResult:
    def __init__(self, annotations):
        self.column_annotations = annotations


def _fake_build_technical_context(calls):
    """Annotate every column as ``TC(<name>)`` and record the call."""

    def _build(**kwargs):
        names = [c["name"] for c in kwargs["columns"] if c.get("name")]
        calls.append(kwargs)
        return _TCResult({n: f"TC({n})" for n in names})

    return _build


def _patch_metadata(monkeypatch):
    """Stub out every Timbr round-trip _build_sql_generation_context makes."""
    monkeypatch.setattr(TLU, "get_datasources", lambda *a, **k: [{"target_type": "postgres"}])
    monkeypatch.setattr(TLU, "get_properties_description", lambda **k: {})
    monkeypatch.setattr(TLU, "get_relationships_description", lambda **k: {})
    monkeypatch.setattr(TLU, "get_tags", lambda **k: {"property_tags": {}})
    monkeypatch.setattr(
        TLU, "get_concept_properties",
        lambda **k: {
            "columns": [{"col_name": "sale_amount", "data_type": "double"}],
            "measures": [],
            # Empty: a non-empty dict would send the static partition pass to the
            # shared ontology, which this test has no server for.
            "relationships": {},
        },
    )


def _context(**overrides):
    kwargs = dict(
        question="how much did we sell in indonesia",
        conn_params={},
        schema="dtimbr",
        concept="sales",
        concept_metadata={},
        graph_depth=1,
        include_tags=None,
        exclude_properties=[],
        db_is_case_sensitive=False,
        max_limit=100,
        metadata_context_mode="dynamic",
    )
    kwargs.update(overrides)
    return _build_sql_generation_context(**kwargs)


class TestUpfrontPassDeferral:
    def test_dynamic_mode_defers_the_pass_and_empties_seen(self, monkeypatch):
        """The planner never reads annotations, so nothing is built before it —
        the pipeline gets an empty seen-set and a live top-up closure instead."""
        _patch_metadata(monkeypatch)
        calls = []
        monkeypatch.setattr(tc_pkg, "build_technical_context", _fake_build_technical_context(calls))
        seen = {}

        def _fake_dynamic(**kwargs):
            seen.update(kwargs)
            return "dyn_cols", "dyn_meas", "dyn_rels", None

        monkeypatch.setattr(TLU, "_apply_dynamic_metadata_context", _fake_dynamic)

        ctx = _context()

        assert calls == []                       # nothing built upfront
        assert seen["tc_seen_names"] == set()    # the key line
        assert seen["tc_annotations"] == {}
        assert callable(seen["tc_topup"])
        assert ctx["columns_str"] == "dyn_cols"

    def test_static_mode_builds_upfront(self, monkeypatch):
        """No planner to defer to — behaviour is unchanged."""
        _patch_metadata(monkeypatch)
        calls = []
        monkeypatch.setattr(tc_pkg, "build_technical_context", _fake_build_technical_context(calls))

        ctx = _context(metadata_context_mode="static")

        assert len(calls) == 1
        assert calls[0]["concept"] == "sales"
        assert "statistics: TC(sale_amount)" in ctx["columns_str"]

    def test_non_dtimbr_schema_builds_upfront(self, monkeypatch):
        """vtimbr skips the dynamic pipeline, so it must keep the upfront pass."""
        _patch_metadata(monkeypatch)
        calls = []
        monkeypatch.setattr(tc_pkg, "build_technical_context", _fake_build_technical_context(calls))

        ctx = _context(schema="vtimbr")

        assert len(calls) == 1
        assert "statistics: TC(sale_amount)" in ctx["columns_str"]

    def test_kill_switch_restores_the_upfront_pass(self, monkeypatch):
        _patch_metadata(monkeypatch)
        calls = []
        monkeypatch.setattr(tc_pkg, "build_technical_context", _fake_build_technical_context(calls))
        monkeypatch.setattr(config, "defer_technical_context", False)
        seen = {}

        def _fake_dynamic(**kwargs):
            seen.update(kwargs)
            return "dyn_cols", "dyn_meas", "dyn_rels", None

        monkeypatch.setattr(TLU, "_apply_dynamic_metadata_context", _fake_dynamic)

        _context()

        assert len(calls) == 1
        assert seen["tc_seen_names"] == {"sale_amount"}
        assert seen["tc_annotations"] == {"sale_amount": "TC(sale_amount)"}

    def test_dynamic_failure_fallback_still_annotates(self, monkeypatch):
        """The call-site crash guard reverts to the static strings, which were
        rendered without statistics — the pass has to run there instead."""
        _patch_metadata(monkeypatch)
        calls = []
        monkeypatch.setattr(tc_pkg, "build_technical_context", _fake_build_technical_context(calls))

        def _boom(**kwargs):
            raise RuntimeError("planner exploded")

        monkeypatch.setattr(TLU, "_apply_dynamic_metadata_context", _boom)

        ctx = _context()

        assert len(calls) == 1
        assert "statistics: TC(sale_amount)" in ctx["columns_str"]


# ---------------------------------------------------------------------------
# Half 2 — the top-up covers the whole surviving set
# ---------------------------------------------------------------------------


@dataclass
class _FakeProp:
    name: str
    data_type: str = "string"
    description: str | None = None


@dataclass
class _FakeMeasure:
    name: str
    data_type: str = "int"
    description: str | None = None
    scoped_to_relationship: str | None = None


@dataclass
class _FakeRel:
    name: str
    target_concept: str
    transitivity: int = 1
    description: str | None = None


@dataclass
class _FakeConcept:
    name: str
    description: str | None = None
    properties: dict = field(default_factory=dict)
    measures: dict = field(default_factory=dict)
    relationships: dict = field(default_factory=dict)


class _FakeOntology:
    def __init__(self, concepts):
        self._concepts = concepts

    def get_concept_metadata(self, name):
        if name not in self._concepts:
            raise KeyError(name)
        return self._concepts[name]

    def cardinality_of(self, from_concept, rel_name):
        return "N:1"

    def get_filtered_cache(self, key):
        return None

    def set_filtered_cache(self, key, entry):
        return None


def _build_ontology():
    sales = _FakeConcept(
        name="sales",
        properties={"sale_amount": _FakeProp("sale_amount")},
        measures={"total_sales": _FakeMeasure("total_sales")},
        relationships={"of_customer": _FakeRel("of_customer", "customer")},
    )
    customer = _FakeConcept(
        name="customer",
        properties={"customer_name": _FakeProp("customer_name")},
        measures={},
        relationships={},
    )
    return _FakeOntology({"sales": sales, "customer": customer})


def _run_pipeline(monkeypatch, *, tc_seen_names, tc_topup, result=None):
    """Rebuild sales -> of_customer[customer] with the given seen-set."""
    if result is None:
        result = DynamicMetadataResult(
            filtered_concepts={"sales", "customer"},
            path_rel_keys=set(),
            validated_paths=[SelectedPath(path_id="P1", segments=[
                PathSegment(**{"from": "sales", "rel": "of_customer", "to": "customer"}),
            ])],
            compact_ddl="## CONCEPTS\n### sales [anchor]\n",
            stats={"resolved_by": "llm_paths"},
            effective_anchor=None,
        )
    monkeypatch.setattr(oc, "get_shared_ontology", lambda conn_params: _build_ontology())
    monkeypatch.setattr(oc, "build_filtered_metadata", lambda **kwargs: result)
    return _apply_dynamic_metadata_context(
        mode="dynamic",
        question="how much did we sell in indonesia",
        anchor="sales",
        conn_params={},
        graph_depth=1,
        # Statically-sourced flat dicts: `col_name` only, no `name`.
        columns=[{"col_name": "sale_amount", "data_type": "double"}],
        measures=[{"col_name": "measure.total_sales", "data_type": "double"}],
        tags={},
        exclude_properties=[],
        static_columns_str="x",
        static_measures_str="x",
        static_rel_prop_str="x",
        llm=None,
        config_overrides=dict(metadata_context_max_tokens=12000),
        tc_annotations={},
        tc_topup=tc_topup,
        tc_seen_names=tc_seen_names,
    )


class TestTopupCoversSurvivingSet:
    def test_empty_seen_tops_up_flat_and_relationship_columns(self, monkeypatch):
        batches = []

        def tc_topup(columns, bound_concept=None):
            batches.append((bound_concept, [c["name"] for c in columns]))
            return {c["name"]: f"TC({c['name']})" for c in columns}

        columns_str, measures_str, rel_prop_str, _eff = _run_pipeline(
            monkeypatch, tc_seen_names=set(), tc_topup=tc_topup,
        )

        assert len(batches) == 1
        bound_concept, names = batches[0]
        # Bare direct columns resolve their stats against the SQL FROM root.
        assert bound_concept == "sales"
        assert {"sale_amount", "measure.total_sales",
                "of_customer[customer].customer_name"} <= set(names)

        # ... and every one of them reaches the prompt annotated.
        assert "statistics: TC(sale_amount)" in columns_str
        assert "statistics: TC(measure.total_sales)" in measures_str
        assert "statistics: TC(of_customer[customer].customer_name)" in rel_prop_str

    def test_handled_planner_failure_still_annotates_the_anchor_block(self, monkeypatch):
        """A handled pipeline error emits anchor-only context (never static).
        With TC deferred those flat columns are all the prompt has left, so the
        top-up still has to reach them."""
        columns_str, measures_str, rel_prop_str, _eff = _run_pipeline(
            monkeypatch,
            tc_seen_names=set(),
            tc_topup=lambda columns, bound_concept=None: {
                c["name"]: f"TC({c['name']})" for c in columns
            },
            result=DynamicMetadataResult(
                filtered_concepts=set(),
                path_rel_keys=set(),
                validated_paths=[],
                compact_ddl="",
                stats={"resolved_by": "error"},
                error="planner returned SQL instead of JSON",
                effective_anchor=None,
            ),
        )

        assert rel_prop_str == ""                       # anchor-only
        assert "statistics: TC(sale_amount)" in columns_str
        assert "statistics: TC(measure.total_sales)" in measures_str

    def test_columns_the_upfront_pass_saw_are_not_topped_up(self, monkeypatch):
        """Gathering flat columns unconditionally must not cost the non-deferred
        path a second (LLM-backed) technical-context pass."""
        seen_names = []

        def _record(columns, bound_concept=None):
            seen_names.extend(c["name"] for c in columns)
            return {}

        _run_pipeline(monkeypatch, tc_seen_names=set(), tc_topup=_record)

        calls = []

        def _tc_topup(columns, bound_concept=None):
            calls.append(columns)
            return {}

        _run_pipeline(monkeypatch, tc_seen_names=set(seen_names), tc_topup=_tc_topup)

        assert calls == []
