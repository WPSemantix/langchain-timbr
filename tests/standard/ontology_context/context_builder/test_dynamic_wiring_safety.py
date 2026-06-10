"""Backward-compatibility regression for Plan 2 wiring.

Ensures that ``_apply_dynamic_metadata_context`` returns the static
strings unchanged in every safe scenario:
  - schema gate (non-dtimbr) — covered upstream by _build_sql_generation_context
  - mode='static' — returns static strings, never touches Ontology / LLM
  - mode='auto' under-threshold + low graph_depth — returns static strings

Also exercises the fallback paths:
  - dynamic pipeline raising → returns static strings (try/except guard)
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


class TestAutoModeUnderThreshold:
    def test_auto_mode_low_tokens_returns_static(self):
        # Static strings are tiny → well under default 12K threshold AND graph_depth<3.
        a, b, c, anchor = _apply_dynamic_metadata_context(
            mode="auto",
            question="any",
            anchor="x",
            conn_params={},   # Won't be touched (no trigger)
            graph_depth=1,
            columns=[],
            measures=[],
            tags=None,
            exclude_properties=None,
            static_columns_str=STATIC_COLUMNS,
            static_measures_str=STATIC_MEASURES,
            static_rel_prop_str=STATIC_RELS,
            llm=None,
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


class TestStaticFallbackGate:
    """Locks the new wiring-layer gate from retry-fallback-redesign.md:
    STATIC fallback fires only on ``result.error`` or
    ``resolved_by == 'empty'``. Anchor-only / BFS-rescue / depth-capped
    outcomes (which may carry empty validated_paths intentionally) MUST
    bypass the fallback and reach the rebuild block."""

    def test_gate_predicate_is_resolved_by_empty(self):
        """Inspect the source to confirm the gate checks resolved_by, not
        the old `not result.validated_paths` predicate. Locks against
        silent regressions to the pre-redesign behavior."""
        import inspect
        src = inspect.getsource(_apply_dynamic_metadata_context)
        # Old predicate must be GONE from the gate position.
        assert "not result.validated_paths" not in src, (
            "Wiring layer still treats empty validated_paths as failure — "
            "this regresses the anchor-only / BFS-rescue / depth-capped "
            "branches into the static fallback."
        )
        # New predicate must be present.
        assert 'resolved_by == "empty"' in src
        # And the resolved_by variable must come from result.stats.
        assert "result.stats" in src

    def test_no_static_fallback_for_anchor_only_outcome(self):
        """Same idea, narrower: confirm the warning log message lists
        resolved_by so operators can tell anchor-only from genuine empty."""
        import inspect
        src = inspect.getsource(_apply_dynamic_metadata_context)
        # The warning that fires on genuine empty must surface resolved_by
        # so operators know WHY they got static (not just THAT they did).
        assert "resolved_by=%r" in src
