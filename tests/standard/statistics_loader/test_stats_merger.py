"""Tests for stats_merger module."""

import pytest
from datetime import datetime
from decimal import Decimal

from langchain_timbr.technical_context.statistics_loader.stats_merger import merge_rows
from langchain_timbr.technical_context.statistics_loader.types import (
    RawStatsRow,
    TopKEntry,
    ConceptMappingSet,
    MappingRef,
    ColumnStatistics,
)


@pytest.fixture
def mapping_set():
    """Sample mapping set for tests."""
    return ConceptMappingSet(
        concept="customer",
        mappings=[
            MappingRef(mapping_name="map_a", source_concept="customer", number_of_rows=1000, via="direct"),
            MappingRef(mapping_name="map_b", source_concept="customer", number_of_rows=500, via="derived"),
        ],
        total_rows=1500,
    )


class TestMergeRows:
    """Test multi-mapping merge logic."""

    def test_empty_rows(self, mapping_set):
        """Empty rows returns sentinel."""
        result = merge_rows([], mapping_set)
        assert result.distinct_count == -1
        assert result.non_null_count == -1
        assert result.top_k is None
        assert result.min_value is None
        assert result.max_value is None
        assert result.approx_union is False
        assert result.total_source_rows == 1500
        assert result.contributing_mappings == []

    def test_single_row_topk(self, mapping_set):
        """Single row with top_k — no merge needed."""
        row = RawStatsRow(
            property_name="status",
            target_name="map_a",
            target_type="mapping",
            distinct_count=5,
            non_null_count=1000,
            top_k=[TopKEntry("ACTIVE", 800), TopKEntry("INACTIVE", 200)],
            min_value=None,
            max_value=None,
            raw_stats=None,
            updated_at=datetime(2024, 1, 1),
        )
        result = merge_rows([row], mapping_set)
        assert result.distinct_count == 5
        assert result.non_null_count == 1000
        assert result.top_k is not None
        assert len(result.top_k) == 2
        assert result.approx_union is False
        assert result.contributing_mappings == ["map_a"]

    def test_multi_row_topk_union(self, mapping_set):
        """Multiple rows with top_k — union by value, sum counts."""
        rows = [
            RawStatsRow(
                property_name="status",
                target_name="map_a",
                target_type="mapping",
                distinct_count=3,
                non_null_count=1000,
                top_k=[TopKEntry("ACTIVE", 800), TopKEntry("PENDING", 150)],
                min_value=None,
                max_value=None,
                raw_stats=None,
                updated_at=datetime(2024, 1, 1),
            ),
            RawStatsRow(
                property_name="status",
                target_name="map_b",
                target_type="mapping",
                distinct_count=4,
                non_null_count=500,
                top_k=[TopKEntry("ACTIVE", 300), TopKEntry("CLOSED", 200)],
                min_value=None,
                max_value=None,
                raw_stats=None,
                updated_at=datetime(2024, 2, 1),
            ),
        ]
        result = merge_rows(rows, mapping_set)

        assert result.approx_union is True
        assert result.distinct_count == 7  # 3 + 4 (upper bound)
        assert result.non_null_count == 1500  # 1000 + 500

        # top_k: ACTIVE=1100, CLOSED=200, PENDING=150
        assert result.top_k is not None
        topk_dict = {e.value: e.count for e in result.top_k}
        assert topk_dict["ACTIVE"] == 1100
        assert topk_dict["CLOSED"] == 200
        assert topk_dict["PENDING"] == 150
        # Sorted by count descending
        assert result.top_k[0].value == "ACTIVE"

        assert result.updated_at == datetime(2024, 2, 1)
        assert set(result.contributing_mappings) == {"map_a", "map_b"}

    def test_multi_row_minmax(self, mapping_set):
        """Multiple rows with min/max — aggregate across."""
        rows = [
            RawStatsRow(
                property_name="amount",
                target_name="map_a",
                target_type="mapping",
                distinct_count=100,
                non_null_count=900,
                top_k=None,
                min_value=Decimal("-10.5"),
                max_value=Decimal("100.0"),
                raw_stats=None,
                updated_at=datetime(2024, 3, 1),
            ),
            RawStatsRow(
                property_name="amount",
                target_name="map_b",
                target_type="mapping",
                distinct_count=50,
                non_null_count=400,
                top_k=None,
                min_value=Decimal("-2.0"),
                max_value=Decimal("500.0"),
                raw_stats=None,
                updated_at=datetime(2024, 1, 1),
            ),
        ]
        result = merge_rows(rows, mapping_set)

        assert result.min_value == Decimal("-10.5")  # min of mins
        assert result.max_value == Decimal("500.0")  # max of maxes
        assert result.distinct_count == 150
        assert result.non_null_count == 1300
        assert result.approx_union is True
        assert result.updated_at == datetime(2024, 3, 1)

    def test_single_row_minmax(self, mapping_set):
        """Single row with min/max — no merge."""
        row = RawStatsRow(
            property_name="price",
            target_name="map_a",
            target_type="mapping",
            distinct_count=200,
            non_null_count=1000,
            top_k=None,
            min_value=Decimal("0.01"),
            max_value=Decimal("9999.99"),
            raw_stats=None,
            updated_at=datetime(2024, 5, 1),
        )
        result = merge_rows([row], mapping_set)
        assert result.min_value == Decimal("0.01")
        assert result.max_value == Decimal("9999.99")
        assert result.approx_union is False

    def test_mixed_topk_minmax_prefers_topk(self, mapping_set):
        """Mixed top_k and min/max rows — prefers top_k with warning."""
        rows = [
            RawStatsRow(
                property_name="col",
                target_name="map_a",
                target_type="mapping",
                distinct_count=5,
                non_null_count=100,
                top_k=[TopKEntry("X", 50)],
                min_value=None,
                max_value=None,
                raw_stats=None,
                updated_at=datetime(2024, 1, 1),
            ),
            RawStatsRow(
                property_name="col",
                target_name="map_b",
                target_type="mapping",
                distinct_count=10,
                non_null_count=200,
                top_k=None,
                min_value=Decimal("1"),
                max_value=Decimal("100"),
                raw_stats=None,
                updated_at=datetime(2024, 2, 1),
            ),
        ]
        result = merge_rows(rows, mapping_set)
        # top_k wins
        assert result.top_k is not None
        assert result.min_value is None
        assert result.max_value is None

    def test_sentinel_counts(self, mapping_set):
        """Rows with -1 counts don't contribute to sum."""
        rows = [
            RawStatsRow(
                property_name="col",
                target_name="map_a",
                target_type="mapping",
                distinct_count=-1,
                non_null_count=-1,
                top_k=[TopKEntry("A", 10)],
                min_value=None,
                max_value=None,
                raw_stats=None,
                updated_at=datetime(2024, 1, 1),
            ),
        ]
        result = merge_rows(rows, mapping_set)
        assert result.distinct_count == -1
        assert result.non_null_count == -1

    def test_total_source_rows_from_mapping_set(self, mapping_set):
        """total_source_rows comes from mapping_set.total_rows."""
        result = merge_rows([], mapping_set)
        assert result.total_source_rows == 1500


class TestMergePreservesNormalizedForms:
    """The union across mappings must carry the normalized forms, not drop them.

    Dropping them would silently push the work back into the matcher for every
    merged column — the exact cost this design removes.
    """

    def test_union_carries_norms_from_source_entries(self, mapping_set):
        rows = [
            RawStatsRow(
                property_name="p", target_name="map_a", target_type="mapping",
                distinct_count=2, non_null_count=2,
                top_k=[TopKEntry(value="Test1", count=4,
                                 norm="test1", norm_space="test1")],
                min_value=None, max_value=None, raw_stats=None, updated_at=None,
            ),
            RawStatsRow(
                property_name="p", target_name="map_b", target_type="mapping",
                distinct_count=2, non_null_count=2,
                top_k=[TopKEntry(value="Test1", count=2,
                                 norm="test1", norm_space="test1")],
                min_value=None, max_value=None, raw_stats=None, updated_at=None,
            ),
        ]

        merged = merge_rows(rows, mapping_set)

        assert len(merged.top_k) == 1
        assert merged.top_k[0].value == "Test1"
        assert merged.top_k[0].count == 6           # counts summed
        assert merged.top_k[0].norm == "test1"   # and the derived form kept
        assert merged.top_k[0].norm_space == "test1"

    def test_single_row_fast_path_keeps_entries_intact(self, mapping_set):
        entry = TopKEntry(value="Good Place", count=9,
                          norm="goodplace", norm_space="good place")
        rows = [RawStatsRow(
            property_name="p", target_name="map_a", target_type="mapping",
            distinct_count=1, non_null_count=1, top_k=[entry],
            min_value=None, max_value=None, raw_stats=None, updated_at=None,
        )]

        merged = merge_rows(rows, mapping_set)

        assert merged.top_k[0].norm == "goodplace"
        assert merged.top_k[0].norm_space == "good place"


class TestMergeCarriesValueKind:

    def _row(self, target, kind, top_k, years=None):
        return RawStatsRow(
            property_name="p", target_name=target, target_type="mapping",
            distinct_count=1, non_null_count=1, top_k=top_k,
            min_value=None, max_value=None, raw_stats=None, updated_at=None,
            value_kind=kind, date_years=years,
        )

    def test_agreeing_rows_keep_the_kind(self, mapping_set):
        rows = [self._row("a", "numeric", [TopKEntry(value="1", count=1)]),
                self._row("b", "numeric", [TopKEntry(value="2", count=1)])]
        assert merge_rows(rows, mapping_set).value_kind == "numeric"

    def test_one_text_mapping_makes_the_column_text(self, mapping_set):
        """Safe direction: the column just keeps the full matcher cascade."""
        rows = [self._row("a", "numeric", [TopKEntry(value="1", count=1)]),
                self._row("b", "text", [TopKEntry(value="one", count=1)])]
        assert merge_rows(rows, mapping_set).value_kind == "text"

    def test_year_index_unions_months_and_sums_counts(self, mapping_set):
        rows = [self._row("a", "date", [TopKEntry(value="2024-01-05", count=1)],
                          {"2024": (frozenset({"01"}), 3)}),
                self._row("b", "date", [TopKEntry(value="2024-06-05", count=1)],
                          {"2024": (frozenset({"06"}), 2)})]
        merged = merge_rows(rows, mapping_set)
        assert merged.date_years["2024"] == (frozenset({"01", "06"}), 5)
