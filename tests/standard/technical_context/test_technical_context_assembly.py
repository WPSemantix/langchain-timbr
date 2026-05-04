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
