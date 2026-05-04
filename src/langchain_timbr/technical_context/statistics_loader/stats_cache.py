"""Stats cache with request-driven sweep and three composing eviction layers.

Layers:
1. Idle eviction (request-driven): entries unused > cache_idle_eviction_seconds
2. Stale eviction (batch validation): per-ontology TTL gate triggers MAX(updated_at) check
3. Size eviction (LRU on insert): when over cache_max_total_mb, evict oldest

No threading, no background workers — all maintenance runs synchronously at request time.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Any

from ...utils import timbr_utils as _timbr_utils
from .config import StatisticsLoaderConfig
from .types import RawStatsRow

logger = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    rows: list[RawStatsRow]
    cached_updated_at: datetime | None
    last_accessed_at: float
    size_bytes: int


class StatsCache:
    """Cache for RawStatsRow with three eviction layers, all request-driven."""

    def __init__(self, config: StatisticsLoaderConfig, conn_params: dict):
        self._config = config
        self._conn_params = conn_params
        self._cache: OrderedDict[tuple, _CacheEntry] = OrderedDict()
        self._last_validated: dict[str, float] = {}  # ontology -> monotonic time
        self._total_bytes = 0
        self._lock = Lock()

    def get_many(
        self,
        ontology: str,
        target_keys: list[tuple[str, str]],  # [(target_type, target_name), ...]
    ) -> tuple[dict[tuple[str, str], list[RawStatsRow]], list[tuple[str, str]]]:
        """Lookup cached entries for target_keys.

        Triggers (in order):
        1. Idle sweep — drops entries unused > cache_idle_eviction_seconds
        2. Batch validation — if interval elapsed for this ontology, queries DB
           for MAX(updated_at) per target and invalidates stale entries
        3. Lookup — returns hits, marks them as MRU, lists misses

        Returns:
            (cached_rows_by_key, missing_keys) where cached_rows_by_key maps
            (target_type, target_name) -> list[RawStatsRow] for cache hits,
            and missing_keys is the list of (target_type, target_name) tuples
            that need to be fetched.
        """
        if not self._config.cache_enabled:
            return {}, target_keys

        # Layer 1: idle sweep (cheap — microseconds for typical N)
        self._sweep_idle()

        now = time.monotonic()

        # Layer 2: batch validation if interval elapsed
        last_val = self._last_validated.get(ontology, 0)
        if now - last_val > self._config.cache_validation_interval_seconds:
            self._batch_validate(ontology, target_keys)
            self._last_validated[ontology] = now

        # Lookup with LRU update
        hits: dict[tuple[str, str], list[RawStatsRow]] = {}
        misses: list[tuple[str, str]] = []
        with self._lock:
            for target_type, target_name in target_keys:
                key = (ontology, target_type, target_name)
                entry = self._cache.get(key)
                if entry is not None:
                    entry.last_accessed_at = now
                    self._cache.move_to_end(key)  # mark as MRU
                    hits[(target_type, target_name)] = entry.rows
                else:
                    misses.append((target_type, target_name))

        return hits, misses

    def put_many(
        self,
        ontology: str,
        rows_by_target: dict[tuple[str, str], list[RawStatsRow]],
    ) -> None:
        """Store fetched rows. Applies LRU eviction if over memory budget."""
        if not self._config.cache_enabled:
            return

        now = time.monotonic()
        max_bytes = self._config.cache_max_total_mb * 1024 * 1024

        with self._lock:
            for (target_type, target_name), rows in rows_by_target.items():
                key = (ontology, target_type, target_name)
                size = _estimate_size_bytes(rows)
                max_updated = max(
                    (r.updated_at for r in rows if r.updated_at is not None),
                    default=None,
                )

                # Remove existing entry to recompute size cleanly
                if key in self._cache:
                    self._total_bytes -= self._cache[key].size_bytes
                    del self._cache[key]

                # Layer 3: LRU eviction until new entry fits
                while self._total_bytes + size > max_bytes and self._cache:
                    evict_key, evict_entry = next(iter(self._cache.items()))
                    self._total_bytes -= evict_entry.size_bytes
                    del self._cache[evict_key]
                    logger.debug("LRU evicted %s (size=%d bytes)", evict_key, evict_entry.size_bytes)

                # Insert at end = MRU position
                self._cache[key] = _CacheEntry(
                    rows=rows,
                    cached_updated_at=max_updated,
                    last_accessed_at=now,
                    size_bytes=size,
                )
                self._total_bytes += size

    def invalidate_ontology(self, ontology: str) -> None:
        """Drop all entries for an ontology (e.g., after explicit stats refresh)."""
        with self._lock:
            keys_to_drop = [k for k in self._cache if k[0] == ontology]
            for key in keys_to_drop:
                self._total_bytes -= self._cache[key].size_bytes
                del self._cache[key]
            self._last_validated.pop(ontology, None)
        if keys_to_drop:
            logger.info("Invalidated %d cache entries for ontology %s", len(keys_to_drop), ontology)

    def clear(self) -> None:
        """Drop all entries."""
        with self._lock:
            self._cache.clear()
            self._last_validated.clear()
            self._total_bytes = 0

    def stats(self) -> dict[str, Any]:
        """Return cache health stats for observability."""
        with self._lock:
            return {
                "entries": len(self._cache),
                "total_mb": round(self._total_bytes / 1024 / 1024, 2),
                "ontologies_validated": len(self._last_validated),
            }

    def _sweep_idle(self) -> None:
        """Drop entries unused for cache_idle_eviction_seconds. Called on every request."""
        now = time.monotonic()
        threshold = self._config.cache_idle_eviction_seconds
        with self._lock:
            keys_to_drop = [
                key for key, entry in self._cache.items()
                if now - entry.last_accessed_at > threshold
            ]
            for key in keys_to_drop:
                self._total_bytes -= self._cache[key].size_bytes
                del self._cache[key]
        if keys_to_drop:
            logger.debug("Idle-swept %d cache entries", len(keys_to_drop))

    def _batch_validate(
        self,
        ontology: str,
        target_keys: list[tuple[str, str]],
    ) -> None:
        """Query DB for MAX(updated_at) per target, invalidate stale entries.

        Runs at most once per cache_validation_interval_seconds per ontology.
        On query failure, logs warning and skips (entries remain cached).
        """
        if not target_keys:
            return

        mappings = [name for (t, name) in target_keys if t == "mapping"]
        views = [name for (t, name) in target_keys if t == "view"]

        current_updated: dict[tuple[str, str], datetime | None] = {}

        for target_type, names in (("mapping", mappings), ("view", views)):
            if not names:
                continue
            in_clause = ", ".join(f"'{n}'" for n in names)
            query = (
                f"SELECT target_name, MAX(updated_at) as max_updated "
                f"FROM timbr.sys_properties_statistics "
                f"WHERE target_type = '{target_type}' AND target_name IN ({in_clause}) "
                f"GROUP BY target_name"
            )
            try:
                rows = _timbr_utils.run_query(query, self._conn_params)
                for row in rows:
                    current_updated[(target_type, row["target_name"])] = _parse_datetime(row.get("max_updated"))
            except Exception:
                logger.warning(
                    "Batch validation query failed for ontology=%s target_type=%s; skipping validation",
                    ontology, target_type,
                )
                return  # don't partially invalidate on failure

        # Compare and invalidate stale or missing entries
        with self._lock:
            for (target_type, target_name) in target_keys:
                key = (ontology, target_type, target_name)
                entry = self._cache.get(key)
                if entry is None:
                    continue

                db_updated = current_updated.get((target_type, target_name))
                if db_updated is None:
                    # Target no longer exists in DB
                    self._total_bytes -= entry.size_bytes
                    del self._cache[key]
                    continue

                if entry.cached_updated_at is None or db_updated > entry.cached_updated_at:
                    self._total_bytes -= entry.size_bytes
                    del self._cache[key]


def _estimate_size_bytes(rows: list[RawStatsRow]) -> int:
    """Rough memory estimate for a list of RawStatsRow."""
    total = 0
    for row in rows:
        total += 200  # base overhead per row
        if row.top_k:
            total += sum(len(e.value) * 2 + 16 for e in row.top_k)
        if row.raw_stats:
            total += 500  # rough JSON dict overhead
    return total


def _parse_datetime(value: Any) -> datetime | None:
    """Parse a datetime value from various formats."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None
