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
        top_k = [_make_top_k_entry("US"), _make_top_k_entry("EU"), _make_top_k_entry("APAC")]
        stats = _make_stats(distinct_count=10, top_k=top_k)
        assert classify_semantic_type("region_code", "varchar(10)", stats) == SemanticType.CODE_LIKE

    def test_business_key(self):
        # Business keys are longer and structured, won't match CODE_LIKE (2-10 chars)
        top_k = [_make_top_k_entry("ORDER-20230001"), _make_top_k_entry("ORDER-20230002"), _make_top_k_entry("ORDER-20230003")]
        stats = _make_stats(distinct_count=500, top_k=top_k)
        assert classify_semantic_type("order_num", "varchar(20)", stats) == SemanticType.BUSINESS_KEY_LIKE

    def test_categorical_fallback(self):
        top_k = [_make_top_k_entry("Active"), _make_top_k_entry("Inactive"), _make_top_k_entry("Pending")]
        stats = _make_stats(distinct_count=3, top_k=top_k)
        assert classify_semantic_type("status", "varchar(50)", stats) == SemanticType.CATEGORICAL_TEXT

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
