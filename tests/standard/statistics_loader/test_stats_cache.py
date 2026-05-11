"""Tests for StatsCache: property-level caching, partial hits, eviction."""

import time
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest

from langchain_timbr.technical_context.statistics_loader.stats_cache import (
    StatsCache,
    _estimate_row_size_bytes,
)
from langchain_timbr.technical_context.statistics_loader.config import StatisticsLoaderConfig
from langchain_timbr.technical_context.statistics_loader.types import RawStatsRow, TopKEntry


# ─── Helpers ────────────────────────────────────────────────────────────────


def _make_row(
    prop: str,
    target_name: str = "map_a",
    target_type: str = "mapping",
    distinct_count: int = 100,
    updated_at: datetime | None = None,
    top_k: list[TopKEntry] | None = None,
) -> RawStatsRow:
    return RawStatsRow(
        property_name=prop,
        target_name=target_name,
        target_type=target_type,
        distinct_count=distinct_count,
        non_null_count=100,
        top_k=top_k,
        min_value=None,
        max_value=None,
        raw_stats=None,
        updated_at=updated_at or datetime(2024, 1, 15, 10, 0, 0),
    )


def _make_config(**overrides) -> StatisticsLoaderConfig:
    defaults = {
        "cache_enabled": True,
        "cache_validation_interval_seconds": 600,
        "cache_idle_eviction_seconds": 3600,
        "cache_max_total_mb": 500,
    }
    defaults.update(overrides)
    return StatisticsLoaderConfig(**defaults)


def _conn_params(ontology: str = "test_ontology") -> dict:
    return {"url": "http://localhost:11000", "token": "t", "ontology": ontology}


# ─── Full DB Fetch (no cache) ──────────────────────────────────────────────


class TestFullFetchFromDB:
    """When cache is empty, all targets should be returned as missing."""

    def test_empty_cache_returns_all_missing(self):
        cache = StatsCache(_make_config(), _conn_params())
        target_keys = [("mapping", "map_a"), ("mapping", "map_b")]
        props = {"col_a", "col_b"}

        cached, missing = cache.get_many("test_ontology", target_keys, props)

        assert cached == []
        assert len(missing) == 2
        for _, _, miss_props in missing:
            assert miss_props == props

    def test_requested_properties_none_returns_cached_and_marks_missing(self):
        """requested_properties=None returns any cached rows but also marks target missing."""
        cache = StatsCache(_make_config(), _conn_params())
        cache._last_validated["test_ontology"] = time.monotonic()
        # Pre-populate cache
        cache.put_many("test_ontology", [_make_row("col_a", "map_a")])

        cached, missing = cache.get_many(
            "test_ontology", [("mapping", "map_a")], requested_properties=None,
        )
        # Cached rows are returned, but target is still marked missing (can't determine completeness)
        assert len(cached) == 1
        assert cached[0].property_name == "col_a"
        assert len(missing) == 1
        assert missing[0] == ("mapping", "map_a", None)

    def test_cache_disabled_returns_all_missing(self):
        config = _make_config(cache_enabled=False)
        cache = StatsCache(config, _conn_params())
        cache.put_many("test_ontology", [_make_row("col_a")])

        cached, missing = cache.get_many(
            "test_ontology", [("mapping", "map_a")], {"col_a"},
        )
        assert cached == []
        assert missing == [("mapping", "map_a", None)]


# ─── Full Cache Hit ─────────────────────────────────────────────────────────


class TestFullCacheHit:
    """When all requested properties are cached, no missing should be returned."""

    def test_all_properties_cached(self):
        cache = StatsCache(_make_config(), _conn_params())
        rows = [
            _make_row("col_a", "map_a"),
            _make_row("col_b", "map_a"),
        ]
        cache.put_many("test_ontology", rows)
        cache._last_validated["test_ontology"] = time.monotonic()

        cached, missing = cache.get_many(
            "test_ontology", [("mapping", "map_a")], {"col_a", "col_b"},
        )

        assert len(cached) == 2
        assert missing == []
        prop_names = {r.property_name for r in cached}
        assert prop_names == {"col_a", "col_b"}

    def test_multiple_targets_all_cached(self):
        cache = StatsCache(_make_config(), _conn_params())
        rows = [
            _make_row("col_a", "map_a"),
            _make_row("col_a", "map_b"),
        ]
        cache.put_many("test_ontology", rows)
        cache._last_validated["test_ontology"] = time.monotonic()

        cached, missing = cache.get_many(
            "test_ontology",
            [("mapping", "map_a"), ("mapping", "map_b")],
            {"col_a"},
        )

        assert len(cached) == 2
        assert missing == []

    def test_cache_hit_does_not_query_db(self):
        """Full cache hit should never trigger a DB call from the fetcher."""
        cache = StatsCache(_make_config(), _conn_params())
        rows = [_make_row("col_a", "map_a"), _make_row("col_b", "map_a")]
        cache.put_many("test_ontology", rows)
        cache._last_validated["test_ontology"] = time.monotonic()

        with patch("langchain_timbr.utils.timbr_utils.run_query") as mock_rq:
            from langchain_timbr.technical_context.statistics_loader.stats_fetcher import (
                fetch_stats_for_mappings,
            )
            result = fetch_stats_for_mappings(
                mapping_names={"map_a"},
                conn_params=_conn_params(),
                columns_type_map={"col_a": "varchar", "col_b": "int"},
                config=_make_config(),
                cache=cache,
                include_properties=["col_a", "col_b"],
            )

        assert len(result) == 2
        mock_rq.assert_not_called()

    def test_view_cache_hit_no_db(self):
        """Full cache hit for view path should not query DB."""
        cache = StatsCache(_make_config(), _conn_params())
        cache.put_many("test_ontology", [
            _make_row("col_x", "my_view", "view"),
        ])
        cache._last_validated["test_ontology"] = time.monotonic()

        with patch("langchain_timbr.utils.timbr_utils.run_query") as mock_rq:
            from langchain_timbr.technical_context.statistics_loader.stats_fetcher import (
                fetch_stats_for_view,
            )
            result = fetch_stats_for_view(
                view_name="my_view",
                conn_params=_conn_params(),
                columns_type_map={"col_x": "int"},
                cache=cache,
                include_properties=["col_x"],
            )

        assert len(result) == 1
        assert result[0].property_name == "col_x"
        mock_rq.assert_not_called()


# ─── Partial Cache Hit ──────────────────────────────────────────────────────


class TestPartialCacheHit:
    """Some properties cached, rest should be fetched from DB."""

    def test_partial_properties_returns_cached_and_missing(self):
        cache = StatsCache(_make_config(), _conn_params())
        cache.put_many("test_ontology", [_make_row("col_a", "map_a")])
        cache._last_validated["test_ontology"] = time.monotonic()

        cached, missing = cache.get_many(
            "test_ontology", [("mapping", "map_a")], {"col_a", "col_b", "col_c"},
        )

        assert len(cached) == 1
        assert cached[0].property_name == "col_a"
        assert len(missing) == 1
        _, _, miss_props = missing[0]
        assert miss_props == {"col_b", "col_c"}

    def test_partial_targets_mixed_hit_miss(self):
        """One target fully cached, another completely missing."""
        cache = StatsCache(_make_config(), _conn_params())
        cache.put_many("test_ontology", [
            _make_row("col_a", "map_a"),
            _make_row("col_b", "map_a"),
        ])
        cache._last_validated["test_ontology"] = time.monotonic()

        cached, missing = cache.get_many(
            "test_ontology",
            [("mapping", "map_a"), ("mapping", "map_b")],
            {"col_a", "col_b"},
        )

        assert len(cached) == 2  # both from map_a
        assert len(missing) == 1
        assert missing[0][1] == "map_b"
        assert missing[0][2] == {"col_a", "col_b"}

    @patch("langchain_timbr.technical_context.statistics_loader.stats_fetcher.load_mapping_properties_index")
    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_fetcher_only_queries_missing_properties(self, mock_run_query, mock_props_index):
        """Fetcher should query DB only for properties not in cache."""
        from langchain_timbr.technical_context.statistics_loader.stats_fetcher import (
            fetch_stats_for_mappings,
        )

        cache = StatsCache(_make_config(), _conn_params())
        cache.put_many("test_ontology", [_make_row("col_a", "map_a")])
        cache._last_validated["test_ontology"] = time.monotonic()

        # Property index: map_a has both col_a and col_b in the stats table
        mock_props_index.return_value = {"map_a": {"col_a", "col_b"}}

        mock_run_query.return_value = [
            {
                "property_name": "col_b",
                "target_name": "map_a",
                "target_type": "mapping",
                "distinct_count": 50,
                "non_null_count": 50,
                "stats": None,
                "updated_at": "2024-01-15T10:00:00",
            }
        ]

        result = fetch_stats_for_mappings(
            mapping_names={"map_a"},
            conn_params=_conn_params(),
            columns_type_map={"col_a": "varchar", "col_b": "int"},
            config=_make_config(),
            cache=cache,
            include_properties=["col_a", "col_b"],
        )

        # Should return both: col_a from cache, col_b from DB
        assert len(result) == 2
        prop_names = {r.property_name for r in result}
        assert prop_names == {"col_a", "col_b"}

        # DB query should only request col_b (col_a is cached)
        assert mock_run_query.call_count == 1
        query_sql = mock_run_query.call_args[0][0]
        assert "col_b" in query_sql
        # col_a should NOT be in the DB query
        assert "'col_a'" not in query_sql

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_fetched_rows_are_cached_for_next_call(self, mock_run_query):
        """After partial fetch, subsequent call should be full cache hit."""
        from langchain_timbr.technical_context.statistics_loader.stats_fetcher import (
            fetch_stats_for_mappings,
        )

        cache = StatsCache(_make_config(), _conn_params())
        cache._last_validated["test_ontology"] = time.monotonic()

        # First call — everything from DB
        mock_run_query.return_value = [
            {
                "property_name": "col_a",
                "target_name": "map_a",
                "target_type": "mapping",
                "distinct_count": 100,
                "non_null_count": 100,
                "stats": None,
                "updated_at": "2024-01-15T10:00:00",
            },
            {
                "property_name": "col_b",
                "target_name": "map_a",
                "target_type": "mapping",
                "distinct_count": 50,
                "non_null_count": 50,
                "stats": None,
                "updated_at": "2024-01-15T10:00:00",
            },
        ]

        result1 = fetch_stats_for_mappings(
            mapping_names={"map_a"},
            conn_params=_conn_params(),
            columns_type_map={"col_a": "varchar", "col_b": "int"},
            config=_make_config(),
            cache=cache,
            include_properties=["col_a", "col_b"],
        )
        assert len(result1) == 2
        assert mock_run_query.call_count == 1

        # Second call — should be full cache hit, no DB
        mock_run_query.reset_mock()
        result2 = fetch_stats_for_mappings(
            mapping_names={"map_a"},
            conn_params=_conn_params(),
            columns_type_map={"col_a": "varchar", "col_b": "int"},
            config=_make_config(),
            cache=cache,
            include_properties=["col_a", "col_b"],
        )
        assert len(result2) == 2
        mock_run_query.assert_not_called()


# ─── Include / Exclude Logic ───────────────────────────────────────────────


class TestIncludeExcludeLogic:
    """Verify include/exclude filtering interacts correctly with cache."""

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_include_only_fetches_whitelisted(self, mock_run_query):
        from langchain_timbr.technical_context.statistics_loader.stats_fetcher import (
            fetch_stats_for_mappings,
        )
        mock_run_query.return_value = [
            {
                "property_name": "col_a",
                "target_name": "map_a",
                "target_type": "mapping",
                "distinct_count": 10,
                "non_null_count": 10,
                "stats": None,
                "updated_at": "2024-01-15T10:00:00",
            },
        ]

        cache = StatsCache(_make_config(), _conn_params())
        cache._last_validated["test_ontology"] = time.monotonic()

        result = fetch_stats_for_mappings(
            mapping_names={"map_a"},
            conn_params=_conn_params(),
            columns_type_map={"col_a": "varchar", "col_b": "int", "col_c": "int"},
            config=_make_config(),
            cache=cache,
            include_properties=["col_a"],
        )

        # SQL should include only col_a
        query_sql = mock_run_query.call_args[0][0]
        assert "'col_a'" in query_sql
        assert "'col_b'" not in query_sql
        assert "'col_c'" not in query_sql

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_exclude_removes_from_request(self, mock_run_query):
        from langchain_timbr.technical_context.statistics_loader.stats_fetcher import (
            fetch_stats_for_mappings,
        )
        mock_run_query.return_value = []

        cache = StatsCache(_make_config(), _conn_params())
        cache._last_validated["test_ontology"] = time.monotonic()

        fetch_stats_for_mappings(
            mapping_names={"map_a"},
            conn_params=_conn_params(),
            columns_type_map={"col_a": "varchar", "col_b": "int"},
            config=_make_config(),
            cache=cache,
            include_properties=["col_a", "col_b"],
            exclude_properties=["col_b"],
        )

        # SQL should only request col_a (col_b excluded)
        query_sql = mock_run_query.call_args[0][0]
        assert "'col_a'" in query_sql
        assert "'col_b'" not in query_sql

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_include_exclude_cache_stores_only_fetched(self, mock_run_query):
        """Cache should only store what was fetched, not excluded properties."""
        from langchain_timbr.technical_context.statistics_loader.stats_fetcher import (
            fetch_stats_for_mappings,
        )
        cache = StatsCache(_make_config(), _conn_params())
        cache._last_validated["test_ontology"] = time.monotonic()

        mock_run_query.return_value = [
            {
                "property_name": "col_a",
                "target_name": "map_a",
                "target_type": "mapping",
                "distinct_count": 10,
                "non_null_count": 10,
                "stats": None,
                "updated_at": "2024-01-15T10:00:00",
            },
        ]

        fetch_stats_for_mappings(
            mapping_names={"map_a"},
            conn_params=_conn_params(),
            columns_type_map={"col_a": "varchar", "col_b": "int"},
            config=_make_config(),
            cache=cache,
            include_properties=["col_a"],
        )

        # Cache should have col_a but not col_b
        assert cache.stats()["entries"] == 1
        cached, missing = cache.get_many(
            "test_ontology", [("mapping", "map_a")], {"col_a"},
        )
        assert len(cached) == 1
        assert cached[0].property_name == "col_a"

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_no_include_no_exclude_uses_columns_type_map_as_requested(self, mock_run_query):
        """Without include/exclude, fetcher uses columns_type_map keys as requested_properties.
        Cache returns hits; DB is only queried for missing properties."""
        from langchain_timbr.technical_context.statistics_loader.stats_fetcher import (
            fetch_stats_for_mappings,
        )
        mock_run_query.return_value = []

        cache = StatsCache(_make_config(), _conn_params())
        cache._last_validated["test_ontology"] = time.monotonic()

        fetch_stats_for_mappings(
            mapping_names={"map_a"},
            conn_params=_conn_params(),
            columns_type_map={"col_a": "varchar", "col_b": "int"},
            config=_make_config(),
            cache=cache,
            include_properties=None,
            exclude_properties=None,
        )

        # SQL should not contain NOT IN filter (no exclude_properties)
        query_sql = mock_run_query.call_args[0][0]
        assert "property_name NOT IN" not in query_sql


# ─── LRU Size Eviction ──────────────────────────────────────────────────────


class TestLRUSizeEviction:
    """Test that entries are evicted when cache exceeds max size."""

    def test_evicts_lru_when_over_budget(self):
        """With a tiny budget, inserting new entries evicts oldest."""
        config = _make_config()
        cache = StatsCache(config, _conn_params())
        cache._last_validated["test_ontology"] = time.monotonic()
        # Hack: directly set the max to a tiny amount
        max_one_entry = 250  # fits 1 entry (200 bytes) but not 2 (400 bytes)
        cache._config.cache_max_total_mb = max_one_entry / (1024 * 1024)

        row_a = _make_row("col_a", "map_a")
        row_b = _make_row("col_b", "map_a")

        cache.put_many("test_ontology", [row_a])
        assert cache.stats()["entries"] == 1

        cache.put_many("test_ontology", [row_b])
        # col_a should be evicted (LRU), col_b should remain
        assert cache.stats()["entries"] == 1
        cached, _ = cache.get_many("test_ontology", [("mapping", "map_a")], {"col_b"})
        assert len(cached) == 1
        assert cached[0].property_name == "col_b"

        # col_a should be gone
        cached, missing = cache.get_many("test_ontology", [("mapping", "map_a")], {"col_a"})
        assert cached == []
        assert len(missing) == 1

    def test_mru_entry_survives_eviction(self):
        """Most recently used entry should not be evicted."""
        config = _make_config()
        cache = StatsCache(config, _conn_params())
        cache._last_validated["test_ontology"] = time.monotonic()
        # Budget fits 2 entries but not 3
        entry_size = _estimate_row_size_bytes(_make_row("x"))
        cache._config.cache_max_total_mb = (entry_size * 2 + 50) / (1024 * 1024)

        cache.put_many("test_ontology", [
            _make_row("col_a", "map_a"),
            _make_row("col_b", "map_a"),
        ])
        assert cache.stats()["entries"] == 2

        # Access col_a to make it MRU
        cache.get_many("test_ontology", [("mapping", "map_a")], {"col_a"})

        # Insert col_c — should evict col_b (LRU), keep col_a (MRU)
        cache.put_many("test_ontology", [_make_row("col_c", "map_a")])
        assert cache.stats()["entries"] == 2

        cached, _ = cache.get_many(
            "test_ontology", [("mapping", "map_a")], {"col_a", "col_b", "col_c"},
        )
        prop_names = {r.property_name for r in cached}
        assert "col_a" in prop_names
        assert "col_c" in prop_names
        assert "col_b" not in prop_names

    def test_large_top_k_increases_entry_size(self):
        """Entries with top_k should consume more size budget."""
        small_row = _make_row("col_small")
        large_row = _make_row(
            "col_large", top_k=[TopKEntry(value="x" * 100, count=1) for _ in range(10)],
        )

        small_size = _estimate_row_size_bytes(small_row)
        large_size = _estimate_row_size_bytes(large_row)
        assert large_size > small_size * 2


# ─── Idle TTL Eviction ──────────────────────────────────────────────────────


class TestIdleEviction:
    """Test that entries unused for > cache_idle_eviction_seconds are swept."""

    def test_idle_entries_evicted(self):
        """Entries older than idle threshold are removed on next get_many."""
        config = _make_config(cache_idle_eviction_seconds=10)
        cache = StatsCache(config, _conn_params())
        cache.put_many("test_ontology", [_make_row("col_a", "map_a")])
        assert cache.stats()["entries"] == 1

        # Capture future time before patching (patch affects shared time module)
        future_time = time.monotonic() + 15

        with patch("langchain_timbr.technical_context.statistics_loader.stats_cache.time.monotonic") as mock_time:
            mock_time.return_value = future_time

            cached, missing = cache.get_many(
                "test_ontology", [("mapping", "map_a")], {"col_a"},
            )

        # Entry should have been swept
        assert cached == []
        assert len(missing) == 1
        assert cache.stats()["entries"] == 0

    def test_accessed_entries_survive_idle_sweep(self):
        """Recently accessed entries should not be evicted."""
        config = _make_config(cache_idle_eviction_seconds=10)
        cache = StatsCache(config, _conn_params())
        cache.put_many("test_ontology", [
            _make_row("col_a", "map_a"),
            _make_row("col_b", "map_a"),
        ])

        base_time = time.monotonic()
        access_time = base_time + 5
        check_time = base_time + 12

        # Access col_a at t+5 (within idle window)
        with patch("langchain_timbr.technical_context.statistics_loader.stats_cache.time.monotonic") as mock_time:
            mock_time.return_value = access_time
            cache.get_many("test_ontology", [("mapping", "map_a")], {"col_a"})

        # At t+12: col_b (last_accessed at base) should be evicted, col_a (at base+5) should survive
        with patch("langchain_timbr.technical_context.statistics_loader.stats_cache.time.monotonic") as mock_time:
            mock_time.return_value = check_time
            cached, missing = cache.get_many(
                "test_ontology", [("mapping", "map_a")], {"col_a", "col_b"},
            )

        assert len(cached) == 1
        assert cached[0].property_name == "col_a"
        assert len(missing) == 1
        _, _, miss_props = missing[0]
        assert "col_b" in miss_props

    def test_very_short_idle_threshold_sweeps_immediately(self):
        """With idle=0, any entry is immediately stale on next request."""
        config = _make_config(cache_idle_eviction_seconds=0)
        cache = StatsCache(config, _conn_params())
        cache.put_many("test_ontology", [_make_row("col_a", "map_a")])

        # Capture future time before patching
        future_time = time.monotonic() + 0.001

        # Even a tiny time delta should sweep
        with patch("langchain_timbr.technical_context.statistics_loader.stats_cache.time.monotonic") as mock_time:
            mock_time.return_value = future_time
            cached, missing = cache.get_many(
                "test_ontology", [("mapping", "map_a")], {"col_a"},
            )

        assert cached == []
        assert cache.stats()["entries"] == 0


# ─── Batch Validation Eviction ──────────────────────────────────────────────


class TestBatchValidation:
    """Test per-property staleness check via batch validation queries."""

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_stale_entry_invalidated(self, mock_run_query):
        """Entry with older updated_at than DB should be evicted."""
        config = _make_config(cache_validation_interval_seconds=0)  # always validate
        cache = StatsCache(config, _conn_params())

        # Cache a row with updated_at = 2024-01-15
        cache.put_many("test_ontology", [
            _make_row("col_a", "map_a", updated_at=datetime(2024, 1, 15)),
        ])

        # DB returns newer updated_at
        mock_run_query.return_value = [
            {"target_name": "map_a", "property_name": "col_a",
             "updated_at": datetime(2024, 2, 1)},
        ]

        cached, missing = cache.get_many(
            "test_ontology", [("mapping", "map_a")], {"col_a"},
        )

        # Entry should be invalidated (stale)
        assert cached == []
        assert len(missing) == 1

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_fresh_entry_kept(self, mock_run_query):
        """Entry with same updated_at as DB should survive validation."""
        config = _make_config(cache_validation_interval_seconds=0)
        cache = StatsCache(config, _conn_params())

        ts = datetime(2024, 1, 15, 10, 0, 0)
        cache.put_many("test_ontology", [
            _make_row("col_a", "map_a", updated_at=ts),
        ])

        # DB returns same timestamp
        mock_run_query.return_value = [
            {"target_name": "map_a", "property_name": "col_a", "updated_at": ts},
        ]

        cached, missing = cache.get_many(
            "test_ontology", [("mapping", "map_a")], {"col_a"},
        )

        assert len(cached) == 1
        assert cached[0].property_name == "col_a"
        assert missing == []

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_deleted_property_evicted(self, mock_run_query):
        """If property no longer exists in DB, it's evicted from cache."""
        config = _make_config(cache_validation_interval_seconds=0)
        cache = StatsCache(config, _conn_params())
        cache.put_many("test_ontology", [
            _make_row("col_a", "map_a"),
            _make_row("col_b", "map_a"),
        ])

        # DB only has col_a, col_b was deleted
        mock_run_query.return_value = [
            {"target_name": "map_a", "property_name": "col_a",
             "updated_at": datetime(2024, 1, 15, 10, 0, 0)},
        ]

        cached, missing = cache.get_many(
            "test_ontology", [("mapping", "map_a")], {"col_a", "col_b"},
        )

        assert len(cached) == 1
        assert cached[0].property_name == "col_a"
        assert len(missing) == 1
        _, _, miss_props = missing[0]
        assert "col_b" in miss_props

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_deleted_target_evicts_all_properties(self, mock_run_query):
        """If target has no rows in DB at all, all its cached entries are evicted."""
        config = _make_config(cache_validation_interval_seconds=0)
        cache = StatsCache(config, _conn_params())
        cache.put_many("test_ontology", [
            _make_row("col_a", "map_deleted"),
            _make_row("col_b", "map_deleted"),
        ])

        # DB returns empty — target not found
        mock_run_query.return_value = []

        cached, missing = cache.get_many(
            "test_ontology", [("mapping", "map_deleted")], {"col_a", "col_b"},
        )

        assert cached == []
        assert cache.stats()["entries"] == 0

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_validation_interval_prevents_repeated_queries(self, mock_run_query):
        """Validation only runs once per interval, not on every get_many."""
        config = _make_config(cache_validation_interval_seconds=600)
        cache = StatsCache(config, _conn_params())
        cache.put_many("test_ontology", [_make_row("col_a", "map_a")])

        mock_run_query.return_value = [
            {"target_name": "map_a", "property_name": "col_a",
             "updated_at": datetime(2024, 1, 15, 10, 0, 0)},
        ]

        # First call triggers validation
        cache.get_many("test_ontology", [("mapping", "map_a")], {"col_a"})
        assert mock_run_query.call_count == 1

        # Second call within interval — no validation query
        mock_run_query.reset_mock()
        cache.get_many("test_ontology", [("mapping", "map_a")], {"col_a"})
        mock_run_query.assert_not_called()

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_validation_query_failure_keeps_cache(self, mock_run_query):
        """If validation query fails, cached entries remain."""
        config = _make_config(cache_validation_interval_seconds=0)
        cache = StatsCache(config, _conn_params())
        cache.put_many("test_ontology", [_make_row("col_a", "map_a")])

        mock_run_query.side_effect = Exception("DB connection failed")

        cached, missing = cache.get_many(
            "test_ontology", [("mapping", "map_a")], {"col_a"},
        )

        # Entry should remain (fail-open)
        assert len(cached) == 1
        assert missing == []


# ─── Invalidate & Clear ─────────────────────────────────────────────────────


class TestInvalidateAndClear:
    """Test explicit invalidation methods."""

    def test_invalidate_ontology_removes_only_target_ontology(self):
        cache = StatsCache(_make_config(), _conn_params())
        cache._last_validated["ontology_a"] = time.monotonic()
        cache._last_validated["ontology_b"] = time.monotonic()
        cache.put_many("ontology_a", [_make_row("col_a", "map_a")])
        cache.put_many("ontology_b", [_make_row("col_b", "map_b")])
        assert cache.stats()["entries"] == 2

        cache.invalidate_ontology("ontology_a")

        assert cache.stats()["entries"] == 1
        cached, _ = cache.get_many("ontology_b", [("mapping", "map_b")], {"col_b"})
        assert len(cached) == 1

    def test_clear_removes_all(self):
        cache = StatsCache(_make_config(), _conn_params())
        cache.put_many("test_ontology", [
            _make_row("col_a", "map_a"),
            _make_row("col_b", "map_b"),
        ])

        cache.clear()

        assert cache.stats() == {"entries": 0, "total_mb": 0.0, "ontologies_validated": 0}


# ─── put_many / get_many Correctness ───────────────────────────────────────


class TestPutGetCorrectness:
    """Verify data integrity through put/get cycles."""

    def test_put_overwrites_existing_entry(self):
        cache = StatsCache(_make_config(), _conn_params())
        cache._last_validated["test_ontology"] = time.monotonic()
        row_v1 = _make_row("col_a", "map_a", distinct_count=100)
        row_v2 = _make_row("col_a", "map_a", distinct_count=999)

        cache.put_many("test_ontology", [row_v1])
        cache.put_many("test_ontology", [row_v2])

        cached, _ = cache.get_many("test_ontology", [("mapping", "map_a")], {"col_a"})
        assert len(cached) == 1
        assert cached[0].distinct_count == 999
        assert cache.stats()["entries"] == 1

    def test_different_targets_same_property_stored_separately(self):
        cache = StatsCache(_make_config(), _conn_params())
        cache._last_validated["test_ontology"] = time.monotonic()
        cache.put_many("test_ontology", [
            _make_row("col_a", "map_a", distinct_count=10),
            _make_row("col_a", "map_b", distinct_count=20),
        ])

        assert cache.stats()["entries"] == 2

        cached_a, _ = cache.get_many("test_ontology", [("mapping", "map_a")], {"col_a"})
        cached_b, _ = cache.get_many("test_ontology", [("mapping", "map_b")], {"col_a"})
        assert cached_a[0].distinct_count == 10
        assert cached_b[0].distinct_count == 20

    def test_view_and_mapping_same_name_stored_separately(self):
        cache = StatsCache(_make_config(), _conn_params())
        cache._last_validated["test_ontology"] = time.monotonic()
        cache.put_many("test_ontology", [
            _make_row("col_a", "shared_name", "mapping", distinct_count=1),
            _make_row("col_a", "shared_name", "view", distinct_count=2),
        ])

        assert cache.stats()["entries"] == 2

        cached_m, _ = cache.get_many("test_ontology", [("mapping", "shared_name")], {"col_a"})
        cached_v, _ = cache.get_many("test_ontology", [("view", "shared_name")], {"col_a"})
        assert cached_m[0].distinct_count == 1
        assert cached_v[0].distinct_count == 2

    def test_total_bytes_tracking_accurate(self):
        cache = StatsCache(_make_config(), _conn_params())
        row = _make_row("col_a")
        expected_size = _estimate_row_size_bytes(row)

        cache.put_many("test_ontology", [row])
        assert cache._total_bytes == expected_size

        cache.put_many("test_ontology", [_make_row("col_b")])
        assert cache._total_bytes == expected_size * 2

        cache.clear()
        assert cache._total_bytes == 0
