"""Unit tests for extraction, assembly, prompt_format, and modes."""

import pytest
from unittest.mock import MagicMock

from langchain_timbr.technical_context.extraction.ngram import extract_prompt_tokens
from langchain_timbr.technical_context.assembly.per_column import assemble_annotation, assemble_column_payload, format_annotation
from langchain_timbr.technical_context.assembly.trimming import trim_to_budget
from langchain_timbr.technical_context.assembly.multi_match import run_all_matchers
from langchain_timbr.technical_context.modes import select_columns_for_annotation
from langchain_timbr.technical_context.config import TechnicalContextConfig
from langchain_timbr.technical_context.types import ColumnPayload, ColumnRef, MatchResult, SemanticType


def _make_stats(distinct_count=10, top_k=None, min_value=None, max_value=None, non_null_count=100):
    stats = MagicMock()
    stats.distinct_count = distinct_count
    stats.non_null_count = non_null_count
    stats.top_k = top_k or []
    stats.min_value = min_value
    stats.max_value = max_value
    return stats


def _make_top_k_entry(value):
    entry = MagicMock()
    entry.value = value
    return entry


class TestExtractPromptTokens:
    """Tests for extract_prompt_tokens()."""

    def test_basic_words(self):
        tokens = extract_prompt_tokens("Show me orders from USA")
        assert "Show" in tokens
        assert "USA" in tokens

    def test_quoted_strings(self):
        tokens = extract_prompt_tokens('Find customers named "John Smith"')
        assert "John Smith" in tokens

    def test_single_quoted(self):
        tokens = extract_prompt_tokens("Find 'Acme Corp' orders")
        assert "Acme Corp" in tokens

    def test_ngrams_generated(self):
        tokens = extract_prompt_tokens("New York City")
        assert "New York" in tokens
        assert "York City" in tokens
        assert "New York City" in tokens

    def test_empty(self):
        assert extract_prompt_tokens("") == []

    def test_short_tokens_filtered(self):
        tokens = extract_prompt_tokens("I am a")
        assert "I" not in tokens
        assert "a" not in tokens
        assert "am" in tokens

    def test_deduplication(self):
        tokens = extract_prompt_tokens("test test test")
        assert tokens.count("test") == 1


class TestAssembleAnnotation:
    """Tests for assemble_annotation()."""

    def test_categorical_with_known_values(self):
        top_k = [_make_top_k_entry("Active"), _make_top_k_entry("Inactive")]
        stats = _make_stats(distinct_count=2, top_k=top_k)
        col_ref = ColumnRef(name="status", sql_type="varchar", ontology_distance=0, priority_band=2, semantic_type=SemanticType.CATEGORICAL_TEXT)
        config = TechnicalContextConfig()
        result = assemble_annotation(col_ref, stats, [], config)
        assert result is not None
        assert "known values:" in result
        assert "'Active'" in result
        assert "'Inactive'" in result

    def test_numeric_range(self):
        stats = _make_stats(min_value=0, max_value=10000)
        col_ref = ColumnRef(name="total", sql_type="decimal", ontology_distance=0, priority_band=2, semantic_type=SemanticType.NUMERIC)
        config = TechnicalContextConfig()
        result = assemble_annotation(col_ref, stats, [], config)
        assert result == "value range: 0 to 10000"

    def test_date_range(self):
        stats = _make_stats(min_value="2023-01-01", max_value="2023-12-31")
        col_ref = ColumnRef(name="created", sql_type="date", ontology_distance=0, priority_band=2, semantic_type=SemanticType.DATE)
        config = TechnicalContextConfig()
        result = assemble_annotation(col_ref, stats, [], config)
        assert result == "date range: 2023-01-01 to 2023-12-31"

    def test_id_no_annotation_without_match(self):
        stats = _make_stats(distinct_count=9500, non_null_count=10000)
        col_ref = ColumnRef(name="id", sql_type="bigint", ontology_distance=0, priority_band=2, semantic_type=SemanticType.ID)
        config = TechnicalContextConfig()
        result = assemble_annotation(col_ref, stats, [], config)
        assert result is None

    def test_with_matched_values(self):
        top_k = [_make_top_k_entry("USA"), _make_top_k_entry("France")]
        stats = _make_stats(distinct_count=5, top_k=top_k)
        col_ref = ColumnRef(name="country", sql_type="varchar", ontology_distance=0, priority_band=1, semantic_type=SemanticType.CATEGORICAL_TEXT)
        matches = [MatchResult(column_name="country", matched_value="USA", score=100, match_type="exact", candidate="usa")]
        config = TechnicalContextConfig()
        result = assemble_annotation(col_ref, stats, matches, config)
        # Matched values sort first in the known values list
        assert "known values:" in result
        assert "'USA'" in result
        # USA should appear before France (matched sort first)
        assert result.index("USA") < result.index("France")

    def test_boolean(self):
        top_k = [_make_top_k_entry("true"), _make_top_k_entry("false")]
        stats = _make_stats(distinct_count=2, top_k=top_k)
        col_ref = ColumnRef(name="active", sql_type="boolean", ontology_distance=0, priority_band=2, semantic_type=SemanticType.BOOLEAN)
        config = TechnicalContextConfig()
        result = assemble_annotation(col_ref, stats, [], config)
        assert "values:" in result


class TestTrimToBudget:
    """Tests for trim_to_budget()."""

    def test_within_budget_no_change(self):
        payloads = {
            "col1": ColumnPayload(format_hint="top_k", values=["a", "b"], distinct_count=10),
            "col2": ColumnPayload(format_hint="top_k", values=["x", "y"], distinct_count=5),
        }
        col_refs = {
            "col1": ColumnRef(name="col1", sql_type="varchar", ontology_distance=0, priority_band=2, semantic_type=SemanticType.CATEGORICAL_TEXT),
            "col2": ColumnRef(name="col2", sql_type="varchar", ontology_distance=0, priority_band=2, semantic_type=SemanticType.CATEGORICAL_TEXT),
        }
        config = TechnicalContextConfig(max_tokens=3000)
        result = trim_to_budget(payloads, col_refs, set(), config)
        assert "col1" in result
        assert "col2" in result
        assert len(result["col1"].values) == 2

    def test_over_budget_reduces_k(self):
        # Create payloads with many values that exceed budget
        big_values = [f"value_{i}" for i in range(200)]
        payloads = {
            "big_col": ColumnPayload(format_hint="top_k", values=list(big_values), distinct_count=5000),
        }
        col_refs = {
            "big_col": ColumnRef(name="big_col", sql_type="varchar", ontology_distance=2, priority_band=4, semantic_type=SemanticType.CATEGORICAL_TEXT),
        }
        config = TechnicalContextConfig(max_tokens=50)  # Very low budget
        result = trim_to_budget(payloads, col_refs, set(), config)
        # Should have reduced values, not deleted column
        assert "big_col" in result
        assert len(result["big_col"].values) < 200

    def test_protected_matched_columns_not_trimmed(self):
        payloads = {
            "matched_col": ColumnPayload(format_hint="top_k", values=[f"v{i}" for i in range(100)], distinct_count=1000),
            "unmatched_col": ColumnPayload(format_hint="top_k", values=[f"u{i}" for i in range(100)], distinct_count=500),
        }
        col_refs = {
            "matched_col": ColumnRef(name="matched_col", sql_type="varchar", ontology_distance=0, priority_band=1, semantic_type=SemanticType.CATEGORICAL_TEXT),
            "unmatched_col": ColumnRef(name="unmatched_col", sql_type="varchar", ontology_distance=2, priority_band=4, semantic_type=SemanticType.CATEGORICAL_TEXT),
        }
        config = TechnicalContextConfig(max_tokens=50)
        result = trim_to_budget(payloads, col_refs, {"matched_col"}, config)
        # Matched column is protected — its values should not be reduced
        assert len(result["matched_col"].values) == 100
        # Unmatched column should be reduced
        assert len(result.get("unmatched_col", ColumnPayload(format_hint="top_k")).values) < 100

    def test_protected_min_max_not_trimmed(self):
        payloads = {
            "amount": ColumnPayload(format_hint="min_max", min_value=0, max_value=9999, distinct_count=500),
            "big_col": ColumnPayload(format_hint="top_k", values=[f"v{i}" for i in range(200)], distinct_count=5000),
        }
        col_refs = {
            "amount": ColumnRef(name="amount", sql_type="decimal", ontology_distance=0, priority_band=2, semantic_type=SemanticType.NUMERIC),
            "big_col": ColumnRef(name="big_col", sql_type="varchar", ontology_distance=2, priority_band=4, semantic_type=SemanticType.CATEGORICAL_TEXT),
        }
        config = TechnicalContextConfig(max_tokens=50)
        result = trim_to_budget(payloads, col_refs, set(), config)
        # min_max is protected
        assert "amount" in result
        assert result["amount"].min_value == 0

    def test_empty_payloads(self):
        result = trim_to_budget({}, {}, set(), TechnicalContextConfig())
        assert result == {}

    def test_phase3_drops_columns_over_safety_ceiling(self):
        # Create so many values that even after full trim sequence, it's over safety_ceiling
        payloads = {
            f"col{i}": ColumnPayload(
                format_hint="top_k",
                values=[f"v{j}" for j in range(5)],  # Already at minimum K
                distinct_count=5000,
            )
            for i in range(500)  # Many columns
        }
        col_refs = {
            f"col{i}": ColumnRef(
                name=f"col{i}", sql_type="varchar", ontology_distance=2,
                priority_band=4, semantic_type=SemanticType.CATEGORICAL_TEXT,
            )
            for i in range(500)
        }
        config = TechnicalContextConfig(max_tokens=10, safety_ceiling=100)
        result = trim_to_budget(payloads, col_refs, set(), config)
        # Some columns should have been replaced with name_only (no values)
        name_only_count = sum(1 for p in result.values() if p.format_hint == "name_only")
        assert name_only_count > 0

    def test_higher_band_trimmed_first(self):
        """Within trim, higher band (lower priority) columns are trimmed first."""
        payloads = {
            "high_priority": ColumnPayload(format_hint="top_k", values=[f"v{i}" for i in range(50)], distinct_count=200),
            "low_priority": ColumnPayload(format_hint="top_k", values=[f"v{i}" for i in range(50)], distinct_count=200),
        }
        col_refs = {
            "high_priority": ColumnRef(name="high_priority", sql_type="varchar", ontology_distance=0, priority_band=1, semantic_type=SemanticType.CATEGORICAL_TEXT),
            "low_priority": ColumnRef(name="low_priority", sql_type="varchar", ontology_distance=2, priority_band=5, semantic_type=SemanticType.CATEGORICAL_TEXT),
        }
        # Budget that forces some trimming but not full reduction
        config = TechnicalContextConfig(max_tokens=200)
        result = trim_to_budget(payloads, col_refs, set(), config)
        # Low priority should be trimmed more aggressively
        if "high_priority" in result and "low_priority" in result:
            assert len(result["low_priority"].values) <= len(result["high_priority"].values)


class TestCategoricalEnumAssembly:
    """Reported bug: dimension-leaf label columns reached via a relationship
    were classified ID (ratio ≈ 1.0) and silently dropped because the matcher
    loop early-skipped ID/FREE_TEXT and assemble_column_payload returned None
    for ID without matches.

    The fix:
      1. Small absolute distinct_count → CATEGORICAL_ENUM (separate from ID).
      2. CATEGORICAL_ENUM assembly emits the full value domain, matches or not.
      3. The matcher loop runs unconditionally; misclassified ID/FREE_TEXT
         degrade to name_only rather than producing no annotation."""

    def test_relationship_leaf_dimension_label_emits_full_domain(self):
        """The reported case: 3 distinct status values reached via a join.
        Without the fix this was classified ID → no annotation. With the fix
        it's CATEGORICAL_ENUM → all 3 values emitted regardless of matches."""
        top_k = [_make_top_k_entry("Active"), _make_top_k_entry("Cancelled"), _make_top_k_entry("Pending")]
        stats = _make_stats(distinct_count=3, non_null_count=3, top_k=top_k)
        col_ref = ColumnRef(
            name="orders[order].status[status_dim].name",
            sql_type="varchar",
            ontology_distance=2,
            priority_band=5,
            semantic_type=SemanticType.CATEGORICAL_ENUM,
        )
        config = TechnicalContextConfig()
        payload = assemble_column_payload(col_ref, stats, [], config)
        assert payload is not None, "categorical enum must NOT return None"
        assert payload.format_hint == "all"
        assert set(payload.values) == {"Active", "Cancelled", "Pending"}
        rendered = format_annotation(payload, config)
        assert rendered is not None
        for v in ("Active", "Cancelled", "Pending"):
            assert f"'{v}'" in rendered

    def test_categorical_enum_with_match_sorts_match_first_keeps_all(self):
        top_k = [_make_top_k_entry("Active"), _make_top_k_entry("Cancelled"), _make_top_k_entry("Pending")]
        stats = _make_stats(distinct_count=3, non_null_count=3, top_k=top_k)
        col_ref = ColumnRef(
            name="status", sql_type="varchar", ontology_distance=0,
            priority_band=1, semantic_type=SemanticType.CATEGORICAL_ENUM,
        )
        matches = [MatchResult(
            column_name="status", matched_value="Cancelled", score=100,
            match_type="exact", candidate="cancelled",
        )]
        config = TechnicalContextConfig()
        payload = assemble_column_payload(col_ref, stats, matches, config)
        assert payload is not None
        # Match sorts to the front, every domain value still present.
        assert payload.values[0] == "Cancelled"
        assert set(payload.values) == {"Active", "Cancelled", "Pending"}
        assert "Cancelled" in payload.matched_values

    def test_high_cardinality_id_with_matches_degrades_to_name_only(self):
        """When a column genuinely IS an ID (above the enum threshold) AND
        the matcher loop runs (no early skip), a prompt match degrades to
        name_only with the matched value — a useful annotation rather than
        a silent drop."""
        stats = _make_stats(distinct_count=5000, non_null_count=5000)
        col_ref = ColumnRef(
            name="user_uuid", sql_type="varchar", ontology_distance=0,
            priority_band=1, semantic_type=SemanticType.ID,
        )
        matches = [MatchResult(
            column_name="user_uuid",
            matched_value="abc-123-def",
            score=100, match_type="exact",
            candidate="abc-123-def",
        )]
        config = TechnicalContextConfig()
        payload = assemble_column_payload(col_ref, stats, matches, config)
        assert payload is not None
        assert payload.format_hint == "name_only"
        assert "abc-123-def" in payload.values

    def test_true_id_high_cardinality_no_matches_returns_none(self):
        # Same shape minus the match — annotation legitimately omitted.
        stats = _make_stats(distinct_count=5000, non_null_count=5000)
        col_ref = ColumnRef(
            name="user_uuid", sql_type="varchar", ontology_distance=0,
            priority_band=2, semantic_type=SemanticType.ID,
        )
        config = TechnicalContextConfig()
        assert assemble_column_payload(col_ref, stats, [], config) is None

    def test_free_text_emits_count_only(self):
        # Very-high-cardinality free text still gets the count annotation.
        stats = _make_stats(distinct_count=50000, non_null_count=100000)
        col_ref = ColumnRef(
            name="description", sql_type="text", ontology_distance=0,
            priority_band=2, semantic_type=SemanticType.FREE_TEXT,
        )
        config = TechnicalContextConfig()
        payload = assemble_column_payload(col_ref, stats, [], config)
        assert payload is not None
        assert payload.format_hint == "count_only"
        assert payload.distinct_count == 50000

    def test_categorical_enum_without_top_k_with_match_returns_name_only(self):
        # Defensive: enum classification but stats happen to lack top_k —
        # still emit matched values if any rather than silently dropping.
        stats = _make_stats(distinct_count=5, non_null_count=5, top_k=[])
        col_ref = ColumnRef(
            name="status", sql_type="varchar", ontology_distance=0,
            priority_band=1, semantic_type=SemanticType.CATEGORICAL_ENUM,
        )
        matches = [MatchResult(
            column_name="status", matched_value="Active", score=100,
            match_type="exact", candidate="active",
        )]
        config = TechnicalContextConfig()
        payload = assemble_column_payload(col_ref, stats, matches, config)
        assert payload is not None
        assert payload.format_hint == "name_only"
        assert "Active" in payload.values


class TestCategoricalEnumTieringFixes:
    """Follow-ups: the tiering decision (``all`` vs ``top_k``) must consider
    shape and starting K, not just classification. ENUMs above
    show_all_under, or with messy value shape, must be trimmable. Matched
    values must not duplicate."""

    def test_small_clean_enum_lands_in_protected_all_tier(self):
        # distinct (3) ≤ show_all_under (50) AND values are short, uniform.
        top_k = [_make_top_k_entry(v) for v in ("Cotton", "Leather", "Metal")]
        stats = _make_stats(distinct_count=3, non_null_count=13, top_k=top_k)
        col_ref = ColumnRef(
            name="material_name", sql_type="varchar", ontology_distance=2,
            priority_band=5, semantic_type=SemanticType.CATEGORICAL_ENUM,
        )
        config = TechnicalContextConfig()
        payload = assemble_column_payload(col_ref, stats, [], config)
        assert payload is not None
        assert payload.format_hint == "all", (
            "small enum with clean shape should land in protected 'all' tier"
        )
        assert set(payload.values) == {"Cotton", "Leather", "Metal"}

    def test_large_enum_above_show_all_under_is_trimmable_top_k(self):
        # distinct (200) above show_all_under (50) but below
        # categorical_enum_max_distinct (500) → classified ENUM yet must be
        # emitted as trimmable 'top_k' so the trimmer can shed values
        # under budget pressure.
        top_k = [_make_top_k_entry(f"VAL{i:03d}") for i in range(200)]
        stats = _make_stats(distinct_count=200, non_null_count=10000, top_k=top_k)
        col_ref = ColumnRef(
            name="region_code", sql_type="varchar", ontology_distance=0,
            priority_band=2, semantic_type=SemanticType.CATEGORICAL_ENUM,
        )
        config = TechnicalContextConfig()
        payload = assemble_column_payload(col_ref, stats, [], config)
        assert payload is not None
        assert payload.format_hint == "top_k", (
            "enum above show_all_under must be trimmable, not protected 'all'"
        )
        # Starting K is the full domain — trimmer takes it from here.
        assert len(payload.values) == 200

    def test_messy_shape_enum_demoted_to_top_k_even_when_small(self):
        # distinct=10 (under show_all_under=50), BUT values are long paragraphs.
        # Shape gate fails → trimmable 'top_k' instead of protected 'all'.
        long_value = "This is a very long descriptive sentence that goes on " * 4
        top_k = [_make_top_k_entry(f"{long_value} variant {i}") for i in range(10)]
        stats = _make_stats(distinct_count=10, non_null_count=10000, top_k=top_k)
        col_ref = ColumnRef(
            name="long_description", sql_type="varchar",
            ontology_distance=0, priority_band=2,
            semantic_type=SemanticType.CATEGORICAL_ENUM,
        )
        config = TechnicalContextConfig()
        payload = assemble_column_payload(col_ref, stats, [], config)
        assert payload is not None
        assert payload.format_hint == "top_k", (
            "small-distinct column with long messy values must NOT be locked "
            "into the protected 'all' tier"
        )

    def test_messy_shape_high_length_variance_demoted(self):
        # distinct (5) ≤ show_all_under, but one value is way longer than the
        # rest — fails the variance check → top_k.
        top_k = [
            _make_top_k_entry("A"),
            _make_top_k_entry("B"),
            _make_top_k_entry("C"),
            _make_top_k_entry("D"),
            _make_top_k_entry("X" * 60),  # 60 chars vs 1 — ratio 60x
        ]
        stats = _make_stats(distinct_count=5, non_null_count=100, top_k=top_k)
        col_ref = ColumnRef(
            name="mixed_label", sql_type="varchar",
            ontology_distance=0, priority_band=2,
            semantic_type=SemanticType.CATEGORICAL_ENUM,
        )
        config = TechnicalContextConfig()
        payload = assemble_column_payload(col_ref, stats, [], config)
        assert payload is not None
        assert payload.format_hint == "top_k"

    def test_matched_values_deduped_when_multiple_matches_same_value(self):
        # Two tokens ("cancelled" and "canceled") both produce a match against
        # the same value 'Cancelled'. Without dedup, 'Cancelled' would appear
        # twice in payload.values.
        top_k = [_make_top_k_entry("Active"), _make_top_k_entry("Cancelled"), _make_top_k_entry("Pending")]
        stats = _make_stats(distinct_count=3, non_null_count=3, top_k=top_k)
        col_ref = ColumnRef(
            name="status", sql_type="varchar", ontology_distance=0,
            priority_band=1, semantic_type=SemanticType.CATEGORICAL_ENUM,
        )
        matches = [
            MatchResult(column_name="status", matched_value="Cancelled", score=100,
                        match_type="exact", candidate="cancelled"),
            MatchResult(column_name="status", matched_value="Cancelled", score=92,
                        match_type="fuzzy", candidate="canceled"),
        ]
        config = TechnicalContextConfig()
        payload = assemble_column_payload(col_ref, stats, matches, config)
        assert payload is not None
        assert payload.values.count("Cancelled") == 1, (
            f"duplicate matched value not deduped; values={payload.values}"
        )

    def test_dedup_across_strong_and_weak_buckets(self):
        # Same value matched at both strong and weak score — must appear once
        # in the strong position only, not duplicated in the weak bucket.
        top_k = [_make_top_k_entry("USA"), _make_top_k_entry("UK")]
        stats = _make_stats(distinct_count=2, non_null_count=2, top_k=top_k)
        col_ref = ColumnRef(
            name="country", sql_type="varchar", ontology_distance=0,
            priority_band=1, semantic_type=SemanticType.CATEGORICAL_ENUM,
        )
        config = TechnicalContextConfig()
        matches = [
            MatchResult(column_name="country", matched_value="USA", score=100,
                        match_type="exact", candidate="usa"),
            MatchResult(column_name="country", matched_value="USA",
                        score=config.fuzzy_threshold_default - config.fuzzy_sort_gap + 1,
                        match_type="fuzzy", candidate="united states"),
        ]
        payload = assemble_column_payload(col_ref, stats, matches, config)
        assert payload is not None
        assert payload.values.count("USA") == 1

    def test_no_top_k_enum_emits_warning(self, caplog):
        # Classifier/stats disagreement: ENUM classification but no top_k.
        # Behavior preserved (name_only with matches, None without) but the
        # warning log makes the misalignment visible.
        import logging
        stats = _make_stats(distinct_count=5, non_null_count=5, top_k=[])
        col_ref = ColumnRef(
            name="status", sql_type="varchar", ontology_distance=0,
            priority_band=1, semantic_type=SemanticType.CATEGORICAL_ENUM,
        )
        config = TechnicalContextConfig()
        with caplog.at_level(logging.WARNING,
                             logger="langchain_timbr.technical_context.assembly.per_column"):
            assemble_column_payload(col_ref, stats, [], config)
        assert any(
            "CATEGORICAL_ENUM" in rec.message and "no top_k" in rec.message
            for rec in caplog.records
        ), f"expected warning about classifier/stats disagreement; got {[r.message for r in caplog.records]}"


class TestRunAllMatchers:
    """Tests for run_all_matchers()."""

    def test_exact_takes_priority(self):
        config = TechnicalContextConfig()
        results = run_all_matchers(
            prompt_text="Show orders from USA",
            prompt_tokens=["USA"],
            column_name="country",
            known_values=["USA", "France"],
            config=config,
        )
        # USA should be matched exactly
        usa_match = [r for r in results if r.matched_value == "USA"]
        assert len(usa_match) == 1
        assert usa_match[0].match_type == "exact"

    def test_deduplication_across_matchers(self):
        config = TechnicalContextConfig()
        results = run_all_matchers(
            prompt_text="United States of America",
            prompt_tokens=["United States"],
            column_name="country",
            known_values=["United States"],
            config=config,
        )
        # Should not have duplicate entries for same value
        matched_values = [r.matched_value for r in results]
        assert matched_values.count("United States") == 1

    def test_empty_values(self):
        config = TechnicalContextConfig()
        results = run_all_matchers("prompt", ["token"], "col", [], config)
        assert results == []


class TestTcAnnotationsInjectedIntoDynamicRebuild:
    """REGRESSION: the dynamic rebuild step builds NEW relationship column
    dicts via build_relationships_from_paths, discarding the technical_context
    field that the TC pipeline put on the upstream static dicts. Without the
    re-injection pass, a CATEGORICAL_ENUM annotation computed for a column
    reached via a relationship (e.g. material[material].material_name with
    distinct=13) never makes it into the SQL-gen prompt — the column shows
    up with no statistics block.

    The helper _inject_tc_annotations_into_rebuild copies annotations back
    onto the rebuilt dicts by exact column name; both sides use the canonical
    `rel[target].prop` chain shape so name lookup is exact."""

    def test_inject_copies_annotation_onto_rebuilt_column(self):
        from langchain_timbr.utils.timbr_llm_utils import (
            _inject_tc_annotations_into_rebuild,
            _build_rel_columns_str,
        )
        filtered_relationships = {
            "includes_product": {
                "description": "",
                "columns": [
                    {
                        "name": "includes_product[product].contains[material].material_name",
                        "col_name": "material_name",
                        "data_type": "varchar",
                    },
                ],
                "measures": [],
            },
        }
        tc_annotations = {
            "includes_product[product].contains[material].material_name":
                "known values: ['Cotton', 'Leather', 'Metal']",
        }
        _inject_tc_annotations_into_rebuild(filtered_relationships, tc_annotations)
        col = filtered_relationships["includes_product"]["columns"][0]
        assert col["technical_context"] == "known values: ['Cotton', 'Leather', 'Metal']"
        # The string builder surfaces the injected annotation downstream so
        # the SQL-gen prompt actually sees the value domain.
        rel_str = _build_rel_columns_str(filtered_relationships)
        assert "statistics: known values:" in rel_str
        assert "'Cotton'" in rel_str

    def test_inject_handles_measures_too(self):
        from langchain_timbr.utils.timbr_llm_utils import (
            _inject_tc_annotations_into_rebuild,
        )
        filtered_relationships = {
            "made_order": {
                "description": "",
                "columns": [],
                "measures": [
                    {
                        "name": "measure.made_order[order].count_of_order",
                        "col_name": "count_of_order",
                        "data_type": "int",
                    },
                ],
            },
        }
        tc_annotations = {
            "measure.made_order[order].count_of_order": "value range: 1 to 1000",
        }
        _inject_tc_annotations_into_rebuild(filtered_relationships, tc_annotations)
        m = filtered_relationships["made_order"]["measures"][0]
        assert m["technical_context"] == "value range: 1 to 1000"

    def test_inject_skips_columns_without_a_matching_annotation(self):
        from langchain_timbr.utils.timbr_llm_utils import (
            _inject_tc_annotations_into_rebuild,
        )
        filtered_relationships = {
            "made_order": {
                "description": "",
                "columns": [
                    {"name": "made_order[order].order_id", "col_name": "order_id"},
                    {"name": "made_order[order].status", "col_name": "status"},
                ],
                "measures": [],
            },
        }
        # Only one column has an annotation upstream.
        tc_annotations = {"made_order[order].status": "known values: ['Active']"}
        _inject_tc_annotations_into_rebuild(filtered_relationships, tc_annotations)
        cols = filtered_relationships["made_order"]["columns"]
        # order_id: untouched (no annotation existed)
        assert "technical_context" not in cols[0]
        # status: injected
        assert cols[1]["technical_context"] == "known values: ['Active']"

    def test_inject_with_empty_annotations_is_noop(self):
        from langchain_timbr.utils.timbr_llm_utils import (
            _inject_tc_annotations_into_rebuild,
        )
        filtered_relationships = {
            "made_order": {
                "description": "",
                "columns": [{"name": "x", "col_name": "x"}],
                "measures": [],
            },
        }
        _inject_tc_annotations_into_rebuild(filtered_relationships, {})
        assert "technical_context" not in filtered_relationships["made_order"]["columns"][0]


class TestBuildColumnsStrTechnicalContext:
    """Tests for _build_columns_str() reading the 'technical_context' field from column dicts."""

    def test_technical_context_included_in_output(self):
        from langchain_timbr.utils.timbr_llm_utils import _build_columns_str
        columns = [{"name": "status", "col_name": "status", "data_type": "varchar",
                     "comment": "Order status", "technical_context": "known values: ['Active', 'Inactive']"}]
        result = _build_columns_str(columns)
        assert "statistics: known values:" in result
        assert "'Active'" in result

    def test_no_technical_context_key(self):
        from langchain_timbr.utils.timbr_llm_utils import _build_columns_str
        columns = [{"name": "status", "col_name": "status", "data_type": "varchar", "comment": "Order status"}]
        result = _build_columns_str(columns)
        assert "statistics:" not in result

    def test_empty_technical_context_ignored(self):
        from langchain_timbr.utils.timbr_llm_utils import _build_columns_str
        columns = [{"name": "status", "col_name": "status", "data_type": "varchar",
                     "comment": "", "technical_context": ""}]
        result = _build_columns_str(columns)
        assert "statistics:" not in result

    def test_technical_context_with_other_meta(self):
        from langchain_timbr.utils.timbr_llm_utils import _build_columns_str
        columns = [{"name": "amount", "col_name": "amount", "data_type": "decimal",
                     "comment": "Total amount", "technical_context": "value range: 0.5 to 9999.99"}]
        result = _build_columns_str(columns)
        assert "type: decimal" in result
        assert "description: Total amount" in result
        assert "statistics: value range: 0.5 to 9999.99" in result


class TestSelectColumnsForAnnotation:
    """Tests for select_columns_for_annotation()."""

    def test_include_all_mode(self):
        cols = [
            ColumnRef(name="a", sql_type="int", ontology_distance=0, priority_band=2, semantic_type=SemanticType.NUMERIC),
            ColumnRef(name="b", sql_type="text", ontology_distance=2, priority_band=5, semantic_type=SemanticType.FREE_TEXT),
        ]
        config = TechnicalContextConfig(mode="include_all")
        result = select_columns_for_annotation(cols, {}, config)
        assert len(result) == 2

    def test_filter_matched_mode(self):
        """select_columns_for_annotation returns all columns (modes don't filter)."""
        cols = [
            ColumnRef(name="a", sql_type="int", ontology_distance=0, priority_band=1, semantic_type=SemanticType.NUMERIC),
            ColumnRef(name="b", sql_type="varchar", ontology_distance=0, priority_band=2, semantic_type=SemanticType.CATEGORICAL_TEXT),
        ]
        matches = {"a": [MatchResult(column_name="a", matched_value="1", score=100, match_type="exact", candidate="1")]}
        config = TechnicalContextConfig(mode="filter_matched")
        result = select_columns_for_annotation(cols, matches, config)
        # All columns are returned — modes control starting K, not column selection
        assert len(result) == 2

    def test_auto_mode_includes_direct_categorical(self):
        """select_columns_for_annotation returns all columns regardless of mode."""
        cols = [
            ColumnRef(name="status", sql_type="varchar", ontology_distance=0, priority_band=2, semantic_type=SemanticType.CATEGORICAL_TEXT),
            ColumnRef(name="desc", sql_type="text", ontology_distance=0, priority_band=2, semantic_type=SemanticType.FREE_TEXT),
            ColumnRef(name="far_col", sql_type="varchar", ontology_distance=2, priority_band=5, semantic_type=SemanticType.CATEGORICAL_TEXT),
        ]
        config = TechnicalContextConfig(mode="auto")
        result = select_columns_for_annotation(cols, {}, config)
        names = [c.name for c in result]
        # All columns are returned — no filtering based on mode
        assert "status" in names
        assert "desc" in names
        assert "far_col" in names
        assert len(result) == 3
