"""Unit tests for semantic type classification."""

import pytest
from unittest.mock import MagicMock
from langchain_timbr.technical_context.semantic_type import (
    classify_semantic_type,
    compute_ontology_distance,
    compute_priority_band,
)
from langchain_timbr.technical_context.types import SemanticType


def _make_stats(distinct_count=100, non_null_count=1000, top_k=None, min_value=None, max_value=None):
    """Create a mock ColumnStatistics."""
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


class TestClassifySemanticType:
    """Tests for classify_semantic_type()."""

    def test_boolean_by_type(self):
        assert classify_semantic_type("flag", "boolean", None) == SemanticType.BOOLEAN

    def test_boolean_by_bit(self):
        assert classify_semantic_type("active", "bit", None) == SemanticType.BOOLEAN

    def test_boolean_by_stats(self):
        top_k = [_make_top_k_entry("true"), _make_top_k_entry("false")]
        stats = _make_stats(distinct_count=2, top_k=top_k)
        assert classify_semantic_type("is_active", "varchar(10)", stats) == SemanticType.BOOLEAN

    def test_numeric(self):
        stats = _make_stats(distinct_count=500, non_null_count=1000)
        assert classify_semantic_type("total", "decimal(18,2)", stats) == SemanticType.NUMERIC

    def test_numeric_id_detection(self):
        stats = _make_stats(distinct_count=950, non_null_count=1000)
        assert classify_semantic_type("id", "bigint", stats) == SemanticType.ID

    def test_date(self):
        assert classify_semantic_type("created_at", "timestamp", None) == SemanticType.DATE

    def test_date_type_keyword(self):
        assert classify_semantic_type("birth_date", "date", None) == SemanticType.DATE

    def test_string_id(self):
        stats = _make_stats(distinct_count=9500, non_null_count=10000)
        assert classify_semantic_type("uuid", "varchar(36)", stats) == SemanticType.ID

    def test_free_text(self):
        stats = _make_stats(distinct_count=50000, non_null_count=100000)
        assert classify_semantic_type("description", "text", stats) == SemanticType.FREE_TEXT

    def test_code_like(self):
        # distinct above enum threshold so CODE_LIKE classification kicks in
        # (small-distinct string columns are CATEGORICAL_ENUM by design).
        top_k = [_make_top_k_entry("US"), _make_top_k_entry("EU"), _make_top_k_entry("APAC")]
        stats = _make_stats(distinct_count=600, top_k=top_k)
        assert classify_semantic_type("region_code", "varchar(10)", stats) == SemanticType.CODE_LIKE

    def test_business_key(self):
        # Business keys are longer and structured, won't match CODE_LIKE (2-10 chars).
        # distinct above enum threshold so this lands in BUSINESS_KEY_LIKE rather
        # than CATEGORICAL_ENUM.
        top_k = [_make_top_k_entry("ORDER-20230001"), _make_top_k_entry("ORDER-20230002"), _make_top_k_entry("ORDER-20230003")]
        stats = _make_stats(distinct_count=600, top_k=top_k)
        assert classify_semantic_type("order_num", "varchar(20)", stats) == SemanticType.BUSINESS_KEY_LIKE

    def test_categorical_text_above_enum_threshold(self):
        # Plain categorical text with distinct count above the enum threshold —
        # not code-like, not business-key-like → CATEGORICAL_TEXT fallback.
        top_k = [_make_top_k_entry("Some name"), _make_top_k_entry("Another name")]
        stats = _make_stats(distinct_count=600, top_k=top_k)
        assert classify_semantic_type("display_name", "varchar(50)", stats) == SemanticType.CATEGORICAL_TEXT

    def test_no_stats(self):
        assert classify_semantic_type("col", "varchar(255)", None) == SemanticType.CATEGORICAL_TEXT

    def test_stats_with_negative_distinct(self):
        stats = _make_stats(distinct_count=-1)
        assert classify_semantic_type("col", "varchar(255)", stats) == SemanticType.CATEGORICAL_TEXT


class TestComputeOntologyDistance:
    """Tests for compute_ontology_distance()."""

    def test_no_dots(self):
        assert compute_ontology_distance("status") == 0

    def test_one_dot(self):
        assert compute_ontology_distance("orders[order].total") == 1

    def test_two_dots(self):
        assert compute_ontology_distance("a.b.c") == 2

    def test_empty(self):
        assert compute_ontology_distance("") == 0


class TestComputePriorityBand:
    """Tests for compute_priority_band()."""

    def test_direct_matched(self):
        assert compute_priority_band(0, True) == 1

    def test_direct_unmatched(self):
        assert compute_priority_band(0, False) == 2

    def test_one_hop_matched(self):
        assert compute_priority_band(1, True) == 3

    def test_one_hop_unmatched(self):
        assert compute_priority_band(1, False) == 4

    def test_two_plus_hops(self):
        assert compute_priority_band(2, True) == 5
        assert compute_priority_band(2, False) == 5
        assert compute_priority_band(3, True) == 5


class TestCategoricalEnumClassification:
    """Reported bug: dimension-leaf label columns (e.g. status.name reached via
    a relationship) have distinct_count ≈ non_null_count by construction, so
    the unique-ratio check misclassified them as ID. The fix gates classification
    on absolute distinct_count, not the ratio."""

    def test_low_cardinality_dimension_leaf_classified_as_enum_not_id(self):
        # 3 distinct values, 3 non-null rows in the dimension target → ratio 1.0.
        # Under the old logic this returned ID and got silently dropped.
        top_k = [_make_top_k_entry("Active"), _make_top_k_entry("Cancelled"), _make_top_k_entry("Pending")]
        stats = _make_stats(distinct_count=3, non_null_count=3, top_k=top_k)
        sem = classify_semantic_type(
            "orders[order].status[status_dim].name",
            "varchar(50)",
            stats,
        )
        assert sem == SemanticType.CATEGORICAL_ENUM, (
            f"expected CATEGORICAL_ENUM (small distinct wins over ratio-based ID); "
            f"got {sem}"
        )

    def test_high_cardinality_unique_string_still_classified_as_id(self):
        # 5000 distinct, ratio = 1.0, distinct above enum threshold → ID stands.
        stats = _make_stats(distinct_count=5000, non_null_count=5000)
        assert classify_semantic_type("user_uuid", "varchar(36)", stats) == SemanticType.ID

    def test_very_high_cardinality_still_free_text(self):
        # 50000 distinct → above free_text_threshold AND above enum threshold → FREE_TEXT.
        stats = _make_stats(distinct_count=50000, non_null_count=100000)
        assert classify_semantic_type("description", "text", stats) == SemanticType.FREE_TEXT

    def test_enum_threshold_is_configurable(self):
        # Lowering categorical_enum_max_distinct shrinks the gate.
        stats = _make_stats(distinct_count=100, non_null_count=100)
        sem_default = classify_semantic_type(
            "label", "varchar(50)", stats,
        )
        assert sem_default == SemanticType.CATEGORICAL_ENUM
        sem_lowered = classify_semantic_type(
            "label", "varchar(50)", stats,
            categorical_enum_max_distinct=10,
            id_unique_ratio=0.95,
        )
        # With a low gate, distinct=100 > 10, ratio=1.0 ≥ 0.95 → falls through to ID.
        assert sem_lowered == SemanticType.ID

    def test_numeric_unchanged_by_enum_gate(self):
        # The enum gate is string-only; numeric IDs remain ID and other numerics
        # remain NUMERIC regardless of distinct count.
        stats = _make_stats(distinct_count=5, non_null_count=100)
        assert classify_semantic_type("priority", "int", stats) == SemanticType.NUMERIC
        stats_id = _make_stats(distinct_count=950, non_null_count=1000)
        assert classify_semantic_type("id", "bigint", stats_id) == SemanticType.ID

    def test_date_unchanged_by_enum_gate(self):
        # Date columns route to DATE before the string branch — small-distinct
        # dates still get min/max range treatment, not enum.
        stats = _make_stats(distinct_count=5, non_null_count=100)
        assert classify_semantic_type("created_on", "date", stats) == SemanticType.DATE
