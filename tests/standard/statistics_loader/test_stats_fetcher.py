"""Tests for stats_fetcher module."""

import pytest
from unittest.mock import patch, call
from datetime import datetime

from langchain_timbr.technical_context.statistics_loader.stats_fetcher import (
    fetch_stats_for_mappings,
    fetch_stats_for_view,
)
from langchain_timbr.technical_context.statistics_loader.config import StatisticsLoaderConfig


class TestFetchStatsForMappings:
    """Test batched mapping stats fetching."""

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_single_chunk(self, mock_run_query, conn_params, default_config):
        """All mappings fit in one chunk."""
        mock_run_query.return_value = [
            {
                "property_name": "customer_id",
                "target_name": "map_a",
                "target_type": "mapping",
                "distinct_count": 1000,
                "non_null_count": 1000,
                "stats": '{"top_k": [{"value": "C001", "count": 1}]}',
                "updated_at": "2024-01-15T10:00:00",
            }
        ]

        result = fetch_stats_for_mappings(
            mapping_names={"map_a", "map_b"},
            conn_params=conn_params,
            columns_type_map={"customer_id": "varchar(50)"},
            config=default_config,
        )

        assert len(result) == 1
        assert result[0].property_name == "customer_id"
        assert result[0].distinct_count == 1000
        assert mock_run_query.call_count == 1

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_multiple_chunks(self, mock_run_query, conn_params, small_chunk_config):
        """Mappings split across multiple chunks."""
        mock_run_query.return_value = []

        fetch_stats_for_mappings(
            mapping_names={"map_1", "map_2", "map_3", "map_4", "map_5"},
            conn_params=conn_params,
            columns_type_map={},
            config=small_chunk_config,  # chunk_size=3
        )

        # 5 mappings with chunk_size=3 → 2 queries
        assert mock_run_query.call_count == 2

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_empty_input(self, mock_run_query, conn_params, default_config):
        """Empty mapping_names returns empty list without query."""
        result = fetch_stats_for_mappings(
            mapping_names=set(),
            conn_params=conn_params,
            columns_type_map={},
            config=default_config,
        )
        assert result == []
        assert mock_run_query.call_count == 0

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_exact_chunk_boundary(self, mock_run_query, conn_params, small_chunk_config):
        """Exactly chunk_size mappings → 1 query."""
        mock_run_query.return_value = []

        fetch_stats_for_mappings(
            mapping_names={"map_1", "map_2", "map_3"},
            conn_params=conn_params,
            columns_type_map={},
            config=small_chunk_config,  # chunk_size=3
        )

        assert mock_run_query.call_count == 1

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_db_error_propagates(self, mock_run_query, conn_params, default_config):
        """Database errors propagate."""
        mock_run_query.side_effect = Exception("Connection failed")

        with pytest.raises(Exception, match="Connection failed"):
            fetch_stats_for_mappings(
                mapping_names={"map_a"},
                conn_params=conn_params,
                columns_type_map={},
                config=default_config,
            )


class TestFetchStatsForView:
    """Test view stats fetching."""

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_basic_fetch(self, mock_run_query, conn_params):
        """Basic view stats fetch."""
        mock_run_query.return_value = [
            {
                "property_name": "col_a",
                "target_name": "my_view",
                "target_type": "view",
                "distinct_count": 50,
                "non_null_count": 100,
                "stats": '{"min_value": "1", "max_value": "100"}',
                "updated_at": "2024-06-01T12:00:00",
            }
        ]

        result = fetch_stats_for_view(
            view_name="my_view",
            conn_params=conn_params,
            columns_type_map={"col_a": "int"},
        )

        assert len(result) == 1
        assert result[0].property_name == "col_a"
        assert result[0].min_value == 1
        assert result[0].max_value == 100

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_empty_result(self, mock_run_query, conn_params):
        """No stats for view returns empty list."""
        mock_run_query.return_value = []

        result = fetch_stats_for_view(
            view_name="no_stats_view",
            conn_params=conn_params,
            columns_type_map={},
        )
        assert result == []
