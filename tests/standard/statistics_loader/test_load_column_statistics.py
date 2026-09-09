"""Tests for load_column_statistics orchestration (loader.py)."""

import logging
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
from decimal import Decimal

from langchain_timbr.technical_context.statistics_loader import (
    load_column_statistics,
    StatisticsLoaderConfig,
    ColumnStatistics,
)
from langchain_timbr.technical_context.statistics_loader.types import (
    OntologyConceptRow,
    ConceptMappingRow,
)


class TestLoadColumnStatisticsVtimbr:
    """Test vtimbr path of load_column_statistics."""

    @patch("langchain_timbr.technical_context.statistics_loader.loader.fetch_stats_for_view")
    @patch("langchain_timbr.technical_context.statistics_loader.loader.load_view_row_counts")
    def test_basic_vtimbr(self, mock_view_counts, mock_fetch_view, conn_params):
        """Basic vtimbr path — view stats loaded and matched."""
        from langchain_timbr.technical_context.statistics_loader.types import RawStatsRow, TopKEntry

        mock_view_counts.return_value = {"my_view": 5000}
        mock_fetch_view.return_value = [
            RawStatsRow(
                property_name="status",
                target_name="my_view",
                target_type="view",
                distinct_count=5,
                non_null_count=5000,
                top_k=[TopKEntry("ACTIVE", 3000), TopKEntry("CLOSED", 2000)],
                min_value=None,
                max_value=None,
                raw_stats=None,
                updated_at=datetime(2024, 6, 1),
            ),
        ]

        result = load_column_statistics(
            schema="vtimbr",
            table_name="my_view",
            columns=[
                {"name": "status", "type": "varchar(20)"},
                {"name": "missing_col", "type": "int"},
            ],
            conn_params=conn_params,
        )

        assert "status" in result
        assert result["status"].distinct_count == 5
        assert result["status"].top_k is not None
        assert result["status"].total_source_rows == 5000
        assert result["status"].approx_union is False

        # Missing column gets sentinel
        assert "missing_col" in result
        assert result["missing_col"].distinct_count == -1

    @patch("langchain_timbr.technical_context.statistics_loader.loader.fetch_stats_for_view")
    @patch("langchain_timbr.technical_context.statistics_loader.loader.load_view_row_counts")
    def test_view_not_in_sys_views(self, mock_view_counts, mock_fetch_view, conn_params):
        """View not in sys_views still works with -1 total_source_rows."""
        mock_view_counts.return_value = {}  # view not found
        mock_fetch_view.return_value = []

        result = load_column_statistics(
            schema="vtimbr",
            table_name="unknown_view",
            columns=[{"name": "col_a", "type": "int"}],
            conn_params=conn_params,
        )

        assert result["col_a"].total_source_rows == -1


class TestLoadColumnStatisticsDtimbr:
    """Test dtimbr path of load_column_statistics."""

    @patch("langchain_timbr.technical_context.statistics_loader.loader.fetch_stats_for_mappings")
    @patch("langchain_timbr.technical_context.statistics_loader.loader.load_view_row_counts")
    @patch("langchain_timbr.technical_context.statistics_loader.loader.load_concept_mappings")
    @patch("langchain_timbr.technical_context.statistics_loader.loader.load_ontology_concepts")
    def test_basic_dtimbr_direct_column(
        self, mock_ontology, mock_mappings, mock_views, mock_fetch_stats, conn_params
    ):
        """dtimbr path — direct column with stats from single mapping."""
        from langchain_timbr.technical_context.statistics_loader.types import RawStatsRow, TopKEntry

        mock_ontology.return_value = {
            "customer": OntologyConceptRow(concept="customer", inheritance="", query=None),
        }
        mock_mappings.return_value = {
            "customer": [ConceptMappingRow(concept="customer", mapping_name="map_cust", number_of_rows=1000)],
        }
        mock_views.return_value = {}
        mock_fetch_stats.return_value = [
            RawStatsRow(
                property_name="customer_id",
                target_name="map_cust",
                target_type="mapping",
                distinct_count=1000,
                non_null_count=1000,
                top_k=None,
                min_value=1,
                max_value=99999,
                raw_stats=None,
                updated_at=datetime(2024, 3, 1),
            ),
        ]

        result = load_column_statistics(
            schema="dtimbr",
            table_name="customer",
            columns=[{"name": "customer_id", "type": "bigint"}],
            conn_params=conn_params,
        )

        assert "customer_id" in result
        assert result["customer_id"].distinct_count == 1000
        assert result["customer_id"].min_value == 1
        assert result["customer_id"].max_value == 99999
        assert result["customer_id"].total_source_rows == 1000

    @patch("langchain_timbr.technical_context.statistics_loader.loader.fetch_stats_for_mappings")
    @patch("langchain_timbr.technical_context.statistics_loader.loader.load_view_row_counts")
    @patch("langchain_timbr.technical_context.statistics_loader.loader.load_concept_mappings")
    @patch("langchain_timbr.technical_context.statistics_loader.loader.load_ontology_concepts")
    def test_dtimbr_relationship_column(
        self, mock_ontology, mock_mappings, mock_views, mock_fetch_stats, conn_params
    ):
        """dtimbr path — relationship column resolved to owning concept's mappings."""
        from langchain_timbr.technical_context.statistics_loader.types import RawStatsRow, TopKEntry

        mock_ontology.return_value = {
            "customer": OntologyConceptRow(concept="customer", inheritance="", query=None),
            "order": OntologyConceptRow(concept="order", inheritance="", query=None),
        }
        mock_mappings.return_value = {
            "customer": [ConceptMappingRow(concept="customer", mapping_name="map_cust", number_of_rows=1000)],
            "order": [ConceptMappingRow(concept="order", mapping_name="map_order", number_of_rows=5000)],
        }
        mock_views.return_value = {}
        mock_fetch_stats.return_value = [
            RawStatsRow(
                property_name="total",
                target_name="map_order",
                target_type="mapping",
                distinct_count=500,
                non_null_count=5000,
                top_k=None,
                min_value=Decimal("0.01"),
                max_value=Decimal("9999.99"),
                raw_stats=None,
                updated_at=datetime(2024, 4, 1),
            ),
        ]

        result = load_column_statistics(
            schema="dtimbr",
            table_name="customer",
            columns=[{"name": "orders[order].total", "type": "decimal(18,2)"}],
            conn_params=conn_params,
        )

        assert "orders[order].total" in result
        assert result["orders[order].total"].min_value == Decimal("0.01")
        assert result["orders[order].total"].total_source_rows == 5000

    @patch("langchain_timbr.technical_context.statistics_loader.loader.fetch_stats_for_mappings")
    @patch("langchain_timbr.technical_context.statistics_loader.loader.load_view_row_counts")
    @patch("langchain_timbr.technical_context.statistics_loader.loader.load_concept_mappings")
    @patch("langchain_timbr.technical_context.statistics_loader.loader.load_ontology_concepts")
    def test_dtimbr_missing_concept(
        self, mock_ontology, mock_mappings, mock_views, mock_fetch_stats, conn_params
    ):
        """dtimbr path — concept not in ontology gets sentinel stats."""
        mock_ontology.return_value = {}
        mock_mappings.return_value = {}
        mock_views.return_value = {}
        mock_fetch_stats.return_value = []

        result = load_column_statistics(
            schema="dtimbr",
            table_name="nonexistent",
            columns=[{"name": "col_a", "type": "int"}],
            conn_params=conn_params,
        )

        assert "col_a" in result
        assert result["col_a"].distinct_count == -1


class TestSelfReferencingRelationshipColumns:
    """Columns on self-referencing relationships arrive from timbr carrying a
    ``*N`` depth marker (``has_parent[work_item*1].label``). The marker must not
    reach the ontology lookup — otherwise the concept misses, no mappings are
    resolved, and the column silently loses its statistics."""

    @patch("langchain_timbr.technical_context.statistics_loader.loader.fetch_stats_for_mappings")
    @patch("langchain_timbr.technical_context.statistics_loader.loader.load_view_row_counts")
    @patch("langchain_timbr.technical_context.statistics_loader.loader.load_concept_mappings")
    @patch("langchain_timbr.technical_context.statistics_loader.loader.load_ontology_concepts")
    def test_transitive_column_resolves_stats(
        self, mock_ontology, mock_mappings, mock_views, mock_fetch_stats, conn_params, caplog
    ):
        """`has_parent[work_item*1].label` resolves against `work_item`."""
        from langchain_timbr.technical_context.statistics_loader.types import RawStatsRow, TopKEntry

        mock_ontology.return_value = {
            "work_item": OntologyConceptRow(concept="work_item", inheritance="", query=None),
        }
        mock_mappings.return_value = {
            "work_item": [
                ConceptMappingRow(concept="work_item", mapping_name="map_work_item", number_of_rows=4200),
            ],
        }
        mock_views.return_value = {}
        mock_fetch_stats.return_value = [
            RawStatsRow(
                property_name="label",
                target_name="map_work_item",
                target_type="mapping",
                distinct_count=37,
                non_null_count=4200,
                top_k=[TopKEntry("Epic", 20), TopKEntry("Story", 17)],
                min_value=None,
                max_value=None,
                raw_stats=None,
                updated_at=datetime(2024, 5, 1),
            ),
        ]

        col = "has_parent[work_item*1].label"
        with caplog.at_level(logging.WARNING):
            result = load_column_statistics(
                schema="dtimbr",
                table_name="work_item",
                columns=[{"name": col, "type": "varchar(255)"}],
                conn_params=conn_params,
            )

        # Keyed by the original name — the marker stays in the annotation key.
        assert col in result
        assert result[col].distinct_count == 37
        assert result[col].total_source_rows == 4200
        assert result[col].top_k is not None
        assert "work_item*1" not in caplog.text

    @patch("langchain_timbr.technical_context.statistics_loader.loader.fetch_stats_for_mappings")
    @patch("langchain_timbr.technical_context.statistics_loader.loader.load_view_row_counts")
    @patch("langchain_timbr.technical_context.statistics_loader.loader.load_concept_mappings")
    @patch("langchain_timbr.technical_context.statistics_loader.loader.load_ontology_concepts")
    def test_depth_does_not_change_the_stats(
        self, mock_ontology, mock_mappings, mock_views, mock_fetch_stats, conn_params
    ):
        """The marker is positional — every depth resolves to the same concept,
        so all of them get the concept's stats. A leaked `work_item*N` would
        leave the marked columns on sentinel values."""
        from langchain_timbr.technical_context.statistics_loader.types import RawStatsRow

        mock_ontology.return_value = {
            "work_item": OntologyConceptRow(concept="work_item", inheritance="", query=None),
        }
        mock_mappings.return_value = {
            "work_item": [
                ConceptMappingRow(concept="work_item", mapping_name="map_work_item", number_of_rows=4200),
            ],
        }
        mock_views.return_value = {}
        mock_fetch_stats.return_value = [
            RawStatsRow(
                property_name="label",
                target_name="map_work_item",
                target_type="mapping",
                distinct_count=37,
                non_null_count=4200,
                top_k=None,
                min_value=None,
                max_value=None,
                raw_stats=None,
                updated_at=datetime(2024, 5, 1),
            ),
        ]

        result = load_column_statistics(
            schema="dtimbr",
            table_name="user",  # root is NOT work_item — the hop is the only way in
            columns=[
                {"name": "assigned_work[work_item*1].label", "type": "varchar(255)"},
                {"name": "assigned_work[work_item*3].label", "type": "varchar(255)"},
            ],
            conn_params=conn_params,
        )

        for col in ("assigned_work[work_item*1].label", "assigned_work[work_item*3].label"):
            assert result[col].distinct_count == 37, col
            assert result[col].total_source_rows == 4200, col
