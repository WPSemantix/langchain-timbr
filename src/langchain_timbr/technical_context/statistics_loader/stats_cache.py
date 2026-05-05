"""Stats cache with request-driven sweep and three composing eviction layers.

Layers:
1. Idle eviction (request-driven): entries unused > cache_idle_eviction_seconds
2. Stale eviction (batch validation): per-ontology TTL gate triggers updated_at check
3. Size eviction (LRU on insert): when over cache_max_total_mb, evict oldest

Cache key: (ontology, target_type, target_name, property_name) — one RawStatsRow per entry.
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

# Cache key: (ontology, target_type, target_name, property_name)
_CacheKey = tuple[str, str, str, str]


@dataclass
class _CacheEntry:
    row: RawStatsRow
    last_accessed_at: float
    size_bytes: int


class StatsCache:
    """Property-level cache for RawStatsRow with three eviction layers."""

    def __init__(self, config: StatisticsLoaderConfig, conn_params: dict):
        self._config = config
        self._conn_params = conn_params
        self._cache: OrderedDict[_CacheKey, _CacheEntry] = OrderedDict()
        self._last_validated: dict[str, float] = {}  # ontology -> monotonic time
        self._total_bytes = 0
        self._lock = Lock()

    def get_many(
        self,
        ontology: str,
        target_keys: list[tuple[str, str]],  # [(target_type, target_name), ...]
        requested_properties: set[str] | None = None,
    ) -> tuple[list[RawStatsRow], list[tuple[str, str, set[str] | None]]]:
        """Lookup cached entries for target_keys filtered by requested_properties.

        Triggers (in order):
        1. Idle sweep — drops entries unused > cache_idle_eviction_seconds
        2. Batch validation — if interval elapsed for this ontology, queries DB
           for updated_at per property and invalidates stale entries
        3. Lookup — returns cached rows, lists targets with missing properties

        Args:
            ontology: The ontology namespace.
            target_keys: List of (target_type, target_name) tuples.
            requested_properties: Set of property names to look up per target.
                None means "all properties" — returns all cached rows but always
                includes each target in missing with missing_props=None.

        Returns:
            (cached_rows, missing) where:
            - cached_rows: flat list of RawStatsRow found in cache
            - missing: list of (target_type, target_name, missing_props) tuples.
              missing_props is a set of property names not found in cache, or
              None meaning "fetch all properties for this target"
        """
        if not self._config.cache_enabled:
            return [], [(t, n, None) for t, n in target_keys]

        # Layer 1: idle sweep
        self._sweep_idle()

        now = time.monotonic()

        # Layer 2: batch validation if interval elapsed
        last_val = self._last_validated.get(ontology, 0)
        if now - last_val > self._config.cache_validation_interval_seconds:
            self._batch_validate(ontology, target_keys)
            self._last_validated[ontology] = now

        # Lookup
        cached_rows: list[RawStatsRow] = []
        missing: list[tuple[str, str, set[str] | None]] = []

        with self._lock:
            for target_type, target_name in target_keys:
                if requested_properties is None:
                    # True fallback: no reference list available at all.
                    # Return any cached rows for the target but also mark it
                    # missing so the caller can decide whether to fetch from DB.
                    now_inner = time.monotonic()
                    all_rows = self._get_all_for_target(ontology, target_type, target_name, now_inner)
                    cached_rows.extend(all_rows)
                    missing.append((target_type, target_name, None))
                else:
                    # Look up each requested property individually
                    found: list[RawStatsRow] = []
                    not_found: set[str] = set()
                    for prop in requested_properties:
                        key: _CacheKey = (ontology, target_type, target_name, prop)
                        entry = self._cache.get(key)
                        if entry is not None:
                            entry.last_accessed_at = now
                            self._cache.move_to_end(key)  # MRU
                            found.append(entry.row)
                        else:
                            not_found.add(prop)
                    cached_rows.extend(found)
                    if not_found:
                        missing.append((target_type, target_name, not_found))

        return cached_rows, missing

    def _get_all_for_target(
        self, ontology: str, target_type: str, target_name: str, now: float,
    ) -> list[RawStatsRow]:
        """Return all cached rows for a target (lock must be held)."""
        prefix = (ontology, target_type, target_name)
        matching_keys = [key for key in self._cache if key[:3] == prefix]
        rows: list[RawStatsRow] = []
        for key in matching_keys:
            entry = self._cache[key]
            entry.last_accessed_at = now
            self._cache.move_to_end(key)
            rows.append(entry.row)
        return rows

    def put_many(self, ontology: str, rows: list[RawStatsRow]) -> None:
        """Store fetched rows as individual property-level cache entries.

        Applies LRU eviction if over memory budget.
        """
        if not self._config.cache_enabled or not rows:
            return

        now = time.monotonic()
        max_bytes = self._config.cache_max_total_mb * 1024 * 1024

        with self._lock:
            for row in rows:
                key: _CacheKey = (
                    ontology, row.target_type, row.target_name, row.property_name,
                )
                size = _estimate_row_size_bytes(row)

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
                    row=row,
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
        """Drop entries unused for cache_idle_eviction_seconds."""
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
        """Query DB for updated_at per property, invalidate stale entries.

        Runs at most once per cache_validation_interval_seconds per ontology.
        On query failure, logs warning and skips (entries remain cached).
        """
        if not target_keys:
            return

        mappings = [name for (t, name) in target_keys if t == "mapping"]
        views = [name for (t, name) in target_keys if t == "view"]

        # Collect current updated_at per (target_type, target_name, property_name) from DB
        current_updated: dict[tuple[str, str, str], datetime | None] = {}
        # Track which targets exist in DB (any property present means target exists)
        targets_in_db: set[tuple[str, str]] = set()

        for target_type, names in (("mapping", mappings), ("view", views)):
            if not names:
                continue
            in_clause = ", ".join(f"'{n}'" for n in names)
            query = (
                f"SELECT target_name, property_name, updated_at "
                f"FROM timbr.sys_properties_statistics "
                f"WHERE target_type = '{target_type}' AND target_name IN ({in_clause})"
            )
            try:
                rows = _timbr_utils.run_query(query, self._conn_params)
                for row in rows:
                    tname = row["target_name"]
                    pname = row["property_name"]
                    current_updated[(target_type, tname, pname)] = _parse_datetime(
                        row.get("updated_at"),
                    )
                    targets_in_db.add((target_type, tname))
            except Exception:
                logger.warning(
                    "Batch validation query failed for ontology=%s target_type=%s; "
                    "skipping validation",
                    ontology, target_type,
                )
                return  # don't partially invalidate on failure

        # Compare cached entries against DB values
        with self._lock:
            for target_type, target_name in target_keys:
                prefix = (ontology, target_type, target_name)
                cached_keys = [k for k in self._cache if k[:3] == prefix]

                if not cached_keys:
                    continue

                # If target has no rows at all in DB, drop all cached entries
                if (target_type, target_name) not in targets_in_db:
                    for key in cached_keys:
                        self._total_bytes -= self._cache[key].size_bytes
                        del self._cache[key]
                    continue

                # Per-property staleness check
                for key in cached_keys:
                    _, _, _, prop_name = key
                    entry = self._cache[key]
                    db_updated = current_updated.get(
                        (target_type, target_name, prop_name),
                    )
                    if db_updated is None:
                        # Property no longer in DB
                        self._total_bytes -= entry.size_bytes
                        del self._cache[key]
                    elif (
                        entry.row.updated_at is None
                        or db_updated > entry.row.updated_at
                    ):
                        # Stale entry
                        self._total_bytes -= entry.size_bytes
                        del self._cache[key]


def _estimate_row_size_bytes(row: RawStatsRow) -> int:
    """Rough memory estimate for a single RawStatsRow."""
    total = 200  # base overhead
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
