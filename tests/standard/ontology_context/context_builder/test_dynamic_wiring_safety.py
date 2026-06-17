"""Backward-compatibility regression for Plan 2 wiring.

Ensures that ``_apply_dynamic_metadata_context`` returns the static
strings unchanged in the only safe no-op scenario:
  - schema gate (non-dtimbr) — covered upstream by _build_sql_generation_context
  - mode='static' — returns static strings, never touches Ontology / LLM

And locks the new dynamic-never-static contract: in ``dynamic`` mode the
wiring layer no longer reverts to the full static strings on
error / empty / empty-rebuild outcomes (those now emit a lean anchor-only
slice). 'auto' was retired and is coerced to 'dynamic' upstream.
"""

from __future__ import annotations

import pytest

from langchain_timbr.utils.timbr_llm_utils import (
    _count_metadata_tokens,
    _apply_dynamic_metadata_context,
    _inject_tc_annotations_into_rebuild,
    _strip_transitivity_marker,
)


STATIC_COLUMNS = "- col1: varchar\n- col2: bigint"
STATIC_MEASURES = "- measure1: count"
STATIC_RELS = "- rel1: details"


class TestStaticModeIsNoOp:
    def test_mode_static_returns_static_unchanged(self):
        a, b, c, anchor = _apply_dynamic_metadata_context(
            mode="static",
            question="any",
            anchor="x",
            conn_params={},   # Never touched in static mode
            graph_depth=1,
            columns=[],
            measures=[],
            tags=None,
            exclude_properties=None,
            static_columns_str=STATIC_COLUMNS,
            static_measures_str=STATIC_MEASURES,
            static_rel_prop_str=STATIC_RELS,
            llm=None,         # Never used
            config_overrides={},
        )
        assert a == STATIC_COLUMNS
        assert b == STATIC_MEASURES
        assert c == STATIC_RELS


class TestStripTransitivityMarker:
    def test_strips_marker_before_bracket(self):
        assert _strip_transitivity_marker("rel[company*3].name") == "rel[company].name"

    def test_strips_marker_in_nested_chain(self):
        assert (
            _strip_transitivity_marker("a[x*2].b[company*3].name")
            == "a[x].b[company].name"
        )

    def test_leaves_unmarked_names_untouched(self):
        assert _strip_transitivity_marker("rel[company].name") == "rel[company].name"
        assert _strip_transitivity_marker("plain_col") == "plain_col"

    def test_handles_none_and_empty(self):
        assert _strip_transitivity_marker(None) is None
        assert _strip_transitivity_marker("") == ""


class TestInjectTcAnnotationsMarkerInsensitive:
    """Fix #1: the static TC keys carry no ``*N`` marker, but the dynamic
    rebuild bakes one in. Injection must match across that difference."""

    def test_exact_match_still_wins(self):
        rels = {"rel": {"columns": [{"name": "rel[company].name"}], "measures": []}}
        _inject_tc_annotations_into_rebuild(rels, {"rel[company].name": "known values: [...]"})
        assert rels["rel"]["columns"][0]["technical_context"] == "known values: [...]"

    def test_marker_on_rebuilt_name_matches_unmarked_key(self):
        # Rebuilt column has *3; the TC annotation key does not.
        rels = {"rel": {"columns": [{"name": "rel[company*3].name"}], "measures": []}}
        _inject_tc_annotations_into_rebuild(rels, {"rel[company].name": "stats here"})
        assert rels["rel"]["columns"][0]["technical_context"] == "stats here"

    def test_measures_are_injected_too(self):
        rels = {"rel": {"columns": [], "measures": [{"name": "measure.rel[company*2].cnt"}]}}
        _inject_tc_annotations_into_rebuild(rels, {"measure.rel[company].cnt": "(5 distinct)"})
        assert rels["rel"]["measures"][0]["technical_context"] == "(5 distinct)"

    def test_unrelated_column_gets_no_annotation(self):
        rels = {"rel": {"columns": [{"name": "rel[order*3].total"}], "measures": []}}
        _inject_tc_annotations_into_rebuild(rels, {"rel[company].name": "stats"})
        assert "technical_context" not in rels["rel"]["columns"][0]


class TestCountMetadataTokens:
    def test_counts_non_empty(self):
        n = _count_metadata_tokens("hello world", "more text")
        assert n > 0

    def test_handles_none_parts(self):
        # Defensive: None segments must not raise
        n = _count_metadata_tokens(None, "x", None)
        assert n >= 1


class TestNoStaticRevertInDynamicMode:
    """Locks the dynamic-never-static contract. In ``dynamic`` mode the wiring
    layer no longer reverts to the full static strings on error / empty /
    empty-rebuild outcomes — those emit a lean anchor-only slice instead."""

    def test_removed_static_fallback_predicates_gone_from_source(self):
        # The two removed static-fallback sites had these signature strings.
        # Their absence locks against a silent regression.
        import inspect
        src = inspect.getsource(_apply_dynamic_metadata_context)
        assert 'resolved_by == "empty"' not in src
        assert "produced empty rebuild" not in src
        assert "falling back to STATIC" not in src


# --- Behavioral tests for the no-paths / token-gate routing -----------------

_ANCHOR_COLUMNS = [{"name": "acol", "col_name": "acol", "data_type": "varchar"}]
_ANCHOR_MEASURES = [{"name": "ameas", "col_name": "ameas", "data_type": "bigint"}]
_STATIC_SENTINEL = "STATIC_SENTINEL_STRING"


class _FakeOntology:
    def __init__(self):
        self._cache = {}

    def get_filtered_cache(self, key):
        return self._cache.get(key)

    def set_filtered_cache(self, key, val):
        self._cache[key] = val


def _run_dynamic(result, *, config_overrides=None, rels_stub=None):
    import contextlib
    from unittest.mock import patch

    onto = _FakeOntology()
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch(
            "langchain_timbr.ontology_context.get_shared_ontology",
            return_value=onto,
        ))
        stack.enter_context(patch(
            "langchain_timbr.ontology_context.build_filtered_metadata",
            return_value=result,
        ))
        if rels_stub is not None:
            stack.enter_context(patch(
                "langchain_timbr.ontology_context.context_builder.rebuild."
                "build_relationships_from_paths",
                return_value=rels_stub,
            ))
        return _apply_dynamic_metadata_context(
            mode="dynamic",
            question="q",
            anchor="a",
            conn_params={"x": 1},
            graph_depth=1,
            columns=list(_ANCHOR_COLUMNS),
            measures=list(_ANCHOR_MEASURES),
            tags={},
            exclude_properties=None,
            static_columns_str=_STATIC_SENTINEL,
            static_measures_str=_STATIC_SENTINEL,
            static_rel_prop_str=_STATIC_SENTINEL,
            llm=None,
            config_overrides=config_overrides or {},
            tc_annotations=None,
            tc_topup=None,
            tc_seen_names=None,
            properties_desc=None,
        )


def _result(**kw):
    from langchain_timbr.ontology_context import DynamicMetadataResult
    base = dict(
        filtered_concepts={"a"},
        path_rel_keys=set(),
        validated_paths=[],
        compact_ddl="schema with full visibility",
    )
    base.update(kw)
    return DynamicMetadataResult(**base)


_DEGRADED_DDL = "schema ... props: [hidden by cascade — assume present] ..."
_DEPTH1_RELS = {
    "rel1": {
        "columns": [{"name": "rel1[c].x", "col_name": "x", "data_type": "varchar"}],
        "measures": [],
        "description": "",
    }
}


class TestNoPathsRouting:
    def test_hard_error_emits_anchor_columns_not_static(self):
        cols, meas, rels, anchor = _run_dynamic(
            _result(filtered_concepts=set(), compact_ddl="", error="boom")
        )
        assert "acol" in cols          # anchor columns rebuilt
        assert _STATIC_SENTINEL not in cols
        assert rels == ""              # no relationships, NOT the static rels

    def test_anchor_only_not_degraded_emits_no_relationships(self):
        cols, meas, rels, _ = _run_dynamic(
            _result(stats={"resolved_by": "llm_paths_anchor_only"})
        )
        assert "acol" in cols
        assert _STATIC_SENTINEL not in cols
        assert rels == ""

    def test_empty_resolved_by_emits_anchor_only_not_static(self):
        cols, meas, rels, _ = _run_dynamic(
            _result(stats={"resolved_by": "empty"})
        )
        assert "acol" in cols
        assert _STATIC_SENTINEL not in cols
        assert rels == ""


class TestDegradedDepth1TokenGate:
    def test_under_budget_includes_relationships(self):
        cols, meas, rels, _ = _run_dynamic(
            _result(compact_ddl=_DEGRADED_DDL,
                    stats={"resolved_by": "depth_capped_static"}),
            rels_stub=_DEPTH1_RELS,
        )
        assert "rel1" in rels          # 1-hop safety net included
        assert _STATIC_SENTINEL not in rels

    def test_over_budget_drops_relationships(self):
        cols, meas, rels, _ = _run_dynamic(
            _result(compact_ddl=_DEGRADED_DDL,
                    stats={"resolved_by": "depth_capped_static"}),
            rels_stub=_DEPTH1_RELS,
            config_overrides={"metadata_context_max_tokens": 1},
        )
        assert rels == ""              # dropped — NOT reverted to static
        assert _STATIC_SENTINEL not in rels
        assert "acol" in cols          # anchor columns still present

    def test_token_gate_boundary_is_strict_greater_than(self):
        # Render once under a generous budget to learn the exact token count.
        cols, meas, rels, _ = _run_dynamic(
            _result(compact_ddl=_DEGRADED_DDL,
                    stats={"resolved_by": "depth_capped_static"}),
            rels_stub=_DEPTH1_RELS,
        )
        n = _count_metadata_tokens(cols, meas, rels)

        # budget == n → kept (gate is strict ``>``)
        _, _, rels_eq, _ = _run_dynamic(
            _result(compact_ddl=_DEGRADED_DDL,
                    stats={"resolved_by": "depth_capped_static"}),
            rels_stub=_DEPTH1_RELS,
            config_overrides={"metadata_context_max_tokens": n},
        )
        assert "rel1" in rels_eq

        # budget == n-1 → dropped
        _, _, rels_lt, _ = _run_dynamic(
            _result(compact_ddl=_DEGRADED_DDL,
                    stats={"resolved_by": "depth_capped_static"}),
            rels_stub=_DEPTH1_RELS,
            config_overrides={"metadata_context_max_tokens": max(1, n - 1)},
        )
        assert rels_lt == ""
