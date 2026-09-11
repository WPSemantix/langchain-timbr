"""Tests for stats_fetcher module."""

import threading
import time

import pytest
from unittest.mock import patch, call
from datetime import datetime

from langchain_timbr.technical_context.statistics_loader.stats_fetcher import (
    fetch_stats_for_mappings,
    fetch_stats_for_view,
)
from langchain_timbr.technical_context.statistics_loader.config import StatisticsLoaderConfig
from langchain_timbr.technical_context.statistics_loader.stats_cache import StatsCache


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


class TestSingleFlight:
    """Concurrent cold callers must not each download the same statistics."""

    @staticmethod
    def _row(prop, mapping="map_a"):
        return {
            "property_name": prop,
            "target_name": mapping,
            "target_type": "mapping",
            "distinct_count": 10,
            "non_null_count": 10,
            "stats": '{"top_k": [{"value": "x", "count": 1}]}',
            "updated_at": "2024-01-15 10:00:00",
        }

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_concurrent_cold_callers_produce_one_query(self, mock_run_query, conn_params):
        """N threads, one cold cache, one fetch.

        Without single-flight every thread misses the cache at the same instant
        and each downloads the whole payload. The metadata path measured 150
        queries for 50 threads before the same fix.
        """
        config = StatisticsLoaderConfig(
            cache_validation_interval_seconds=10 ** 9,
            cache_cold_validation_interval_seconds=10 ** 9,
        )
        cache = StatsCache(config, conn_params)

        def slow_fetch(query, params=None, *args, **kwargs):
            time.sleep(0.05)          # wide enough for every thread to pile up
            return [self._row("col_a")]

        mock_run_query.side_effect = slow_fetch

        threads = 20
        results = [None] * threads
        barrier = threading.Barrier(threads)

        def worker(i):
            barrier.wait()            # release them all together
            results[i] = fetch_stats_for_mappings(
                {"map_a"}, conn_params, {"col_a": "varchar"}, config, cache,
            )

        workers = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
        for w in workers:
            w.start()
        for w in workers:
            w.join(timeout=30)

        assert all(not w.is_alive() for w in workers), "a waiter never woke up"
        assert mock_run_query.call_count == 1
        assert all(len(r) == 1 and r[0].property_name == "col_a" for r in results)

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_gate_released_when_the_fetch_raises(self, mock_run_query, conn_params):
        """A failed fetch must not leave every later caller waiting forever."""
        config = StatisticsLoaderConfig(
            cache_validation_interval_seconds=10 ** 9,
            cache_cold_validation_interval_seconds=10 ** 9,
        )
        cache = StatsCache(config, conn_params)

        mock_run_query.side_effect = Exception("boom")
        with pytest.raises(Exception):
            fetch_stats_for_mappings(
                {"map_a"}, conn_params, {"col_a": "varchar"}, config, cache,
            )

        assert cache._inflight == {}

        # The next caller gets through rather than blocking on a dead gate.
        mock_run_query.side_effect = None
        mock_run_query.return_value = [self._row("col_a")]
        rows = fetch_stats_for_mappings(
            {"map_a"}, conn_params, {"col_a": "varchar"}, config, cache,
        )
        assert len(rows) == 1

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_disabled_singleflight_still_correct(self, mock_run_query, conn_params):
        """The kill switch changes query count, never results."""
        config = StatisticsLoaderConfig(
            cache_validation_interval_seconds=10 ** 9,
            cache_cold_validation_interval_seconds=10 ** 9,
            singleflight_enabled=False,
        )
        cache = StatsCache(config, conn_params)
        mock_run_query.return_value = [self._row("col_a")]

        rows = fetch_stats_for_mappings(
            {"map_a"}, conn_params, {"col_a": "varchar"}, config, cache,
        )

        assert len(rows) == 1
        assert cache._inflight == {}
