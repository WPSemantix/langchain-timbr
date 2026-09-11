"""Stats cache with request-driven sweep and three composing eviction layers.

Layers:
1. Idle eviction (request-driven): entries unused > cache_idle_eviction_seconds
2. Freshness check (per-target watermark): a mapping/view whose MAX(updated_at)
   moved has only its changed properties dropped, so the next fetch replaces them
3. Size eviction (LRU on insert): when over cache_max_total_mb, evict oldest

Cache key: (ontology, target_type, target_name, property_name) — one RawStatsRow per entry.
No threading, no background workers — all maintenance runs synchronously at request time.

Freshness is decided by the statistics table's own ``updated_at``, tracked per
mapping/view, never by the ontology's DDL version: statistics are recomputed on a
schedule of their own, so a version bump is neither necessary nor sufficient.

The rule that governs every eviction here: **absence is not staleness.** A target
that does not come back from the freshness probe is either invisible to this
caller or has no statistics at all, and neither is a reason to throw away data.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from threading import Event, Lock
from typing import Any

from ...utils import timbr_utils as _timbr_utils
from .config import StatisticsLoaderConfig
from .types import RawStatsRow

logger = logging.getLogger(__name__)

# Cache key: (ontology, target_type, target_name, property_name)
_CacheKey = tuple[str, str, str, str]
# Target key: (ontology, target_type, target_name)
_TargetKey = tuple[str, str, str]


@dataclass
class _CacheEntry:
    row: RawStatsRow
    last_accessed_at: float
    size_bytes: int


class StatsCache:
    """Property-level cache for RawStatsRow with three eviction layers."""

    def __init__(self, config: StatisticsLoaderConfig, conn_params: dict):
        self._config = config
        # Fallback only. Every query this cache issues should run with the
        # *calling* request's credentials, passed into get_many — a process-wide
        # singleton that queried with whoever booted the worker would read the
        # wrong ontology, and would validate one user's entries against another
        # user's permissions.
        self._conn_params = conn_params
        self._cache: OrderedDict[_CacheKey, _CacheEntry] = OrderedDict()
        self._target_watermark: dict[_TargetKey, datetime] = {}
        self._last_probe: dict[str, float] = {}   # ontology -> last probe time
        self._total_bytes = 0
        self._lock = Lock()
        self._inflight: dict[str, Event] = {}

    def get_many(
        self,
        ontology: str,
        target_keys: list[tuple[str, str]],  # [(target_type, target_name), ...]
        requested_properties: set[str] | None = None,
        conn_params: dict | None = None,
    ) -> tuple[list[RawStatsRow], list[tuple[str, str, set[str] | None]]]:
        """Lookup cached entries for target_keys filtered by requested_properties.

        Triggers (in order):
        1. Idle sweep — drops entries unused > cache_idle_eviction_seconds
        2. Freshness check — per target, at most once per validation interval,
           compares the target's MAX(updated_at) against the recorded watermark
           and drops only the properties that actually changed
        3. Lookup — returns cached rows, lists targets with missing properties

        Args:
            ontology: The ontology namespace.
            target_keys: List of (target_type, target_name) tuples.
            requested_properties: Set of property names to look up per target.
                None means "all properties" — returns all cached rows but always
                includes each target in missing with missing_props=None.
            conn_params: The calling request's connection parameters, used for
                the freshness probe. Falls back to the ones this cache was built
                with only when a caller passes none.

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

        # Layer 2: per-target freshness
        self._check_freshness(ontology, target_keys, conn_params)

        now = time.monotonic()

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

        Also seeds each target's watermark from the rows just stored, so a fetch
        doubles as a probe and a freshly-fetched target needs no query to
        establish its baseline. The watermark only ever moves forward: a fetch of
        a subset of properties must not lower it.

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

                if row.updated_at is not None:
                    target: _TargetKey = (ontology, row.target_type, row.target_name)
                    known = self._target_watermark.get(target)
                    if known is None or row.updated_at > known:
                        self._target_watermark[target] = row.updated_at

    def claim_or_wait(self, ontology: str) -> bool:
        """Single-flight gate for the statistics fetch, keyed on the ontology.

        Returns True if this thread owns the fetch and should do the work, False
        if it waited for the owner to finish — in which case the caller re-reads
        the cache and fetches only whatever the owner did not cover.

        The wait is deliberately uncapped: the bound belongs on the socket
        (``TIMBR_HTTP_READ_TIMEOUT``), not here. If it were capped, a slow but
        healthy fetch would produce exactly the stampede this prevents.
        """
        if not self._config.cache_enabled or not self._config.singleflight_enabled:
            return True
        with self._lock:
            waiter = self._inflight.get(ontology)
            if waiter is None:
                self._inflight[ontology] = Event()
                return True
        waiter.wait()
        return False

    def end_fetch(self, ontology: str) -> None:
        """Release the single-flight gate and wake every waiter."""
        with self._lock:
            waiter = self._inflight.pop(ontology, None)
        if waiter is not None:
            waiter.set()

    def invalidate_ontology(self, ontology: str) -> None:
        """Drop all entries for an ontology (e.g., after explicit stats refresh)."""
        with self._lock:
            keys_to_drop = [k for k in self._cache if k[0] == ontology]
            for key in keys_to_drop:
                self._total_bytes -= self._cache[key].size_bytes
                del self._cache[key]
            for target in [t for t in self._target_watermark if t[0] == ontology]:
                del self._target_watermark[target]
            self._last_probe.pop(ontology, None)
        if keys_to_drop:
            logger.info("Invalidated %d cache entries for ontology %s", len(keys_to_drop), ontology)

    def clear(self) -> None:
        """Drop all entries."""
        with self._lock:
            self._cache.clear()
            self._target_watermark.clear()
            self._last_probe.clear()
            self._total_bytes = 0

    def stats(self) -> dict[str, Any]:
        """Return cache health stats for observability."""
        with self._lock:
            return {
                "entries": len(self._cache),
                "total_mb": round(self._total_bytes / 1024 / 1024, 2),
                "targets_watermarked": len(self._target_watermark),
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

    def _check_freshness(
        self,
        ontology: str,
        target_keys: list[tuple[str, str]],
        conn_params: dict | None = None,
    ) -> None:
        """Per-target watermark check: drop only the properties that changed.

        Runs at most once per interval per ontology, and covers every target the
        request needs in a single query — not one query per target. Only targets
        that actually hold cached entries are included: there is nothing to
        invalidate otherwise.
        A target absent from the probe result is left completely alone: it is
        either invisible to this caller or has no statistics, and neither means
        the data we hold is wrong.

        On query failure nothing is evicted; the check retries next interval.
        """
        if not target_keys:
            return

        params = conn_params or self._conn_params
        if not params:
            return

        now = time.monotonic()

        # One clock per ontology, so the whole ontology is checked at most once
        # per interval no matter how many requests arrive or which mappings they
        # touch. Cold — no watermark yet for anything this request wants — checks
        # sooner, so a newly warmed ontology acquires its baselines quickly.
        has_baseline = any(
            (ontology, t, n) in self._target_watermark for t, n in target_keys
        )
        interval = (
            self._config.cache_validation_interval_seconds if has_baseline
            else self._config.cache_cold_validation_interval_seconds
        )
        if now - self._last_probe.get(ontology, 0.0) <= interval:
            return
        # Claim the interval before doing the work, so the scan below also runs
        # at most once per interval rather than on every request.
        self._last_probe[ontology] = now

        # Only targets that actually hold entries are worth checking: there is
        # nothing to invalidate otherwise, and the fetch will seed the watermark.
        wanted = {(ontology, t, n) for t, n in target_keys}
        due: dict[str, list[str]] = {}
        with self._lock:
            for target in {key[:3] for key in self._cache if key[:3] in wanted}:
                due.setdefault(target[1], []).append(target[2])

        if not due:
            return

        # (target_type, target_name, previous watermark, new watermark)
        changed: list[tuple[str, str, datetime, datetime]] = []
        for target_type, names in due.items():
            db_max_by_name = self._probe(target_type, names, params)
            if db_max_by_name is None:
                continue  # query failed — keep everything, retry next interval

            with self._lock:
                for name, db_max in db_max_by_name.items():
                    target = (ontology, target_type, name)
                    previous = self._target_watermark.get(target)
                    if previous is None:
                        # No baseline to compare against — adopt it, evict nothing.
                        self._target_watermark[target] = db_max
                    elif db_max > previous:
                        changed.append((target_type, name, previous, db_max))

        if changed:
            self._evict_changed(ontology, changed, params)

    def _probe(
        self, target_type: str, names: list[str], conn_params: dict,
    ) -> dict[str, datetime] | None:
        """MAX(updated_at) per target. None means the query failed."""
        in_clause = ", ".join(f"'{n}'" for n in names)
        query = (
            f"SELECT target_name, MAX(updated_at) AS mx "
            f"FROM timbr.sys_properties_statistics "
            f"WHERE target_type = '{target_type}' AND target_name IN ({in_clause}) "
            f"GROUP BY target_name"
        )
        try:
            rows = _timbr_utils.run_query(query, conn_params)
        except Exception:
            logger.warning(
                "Freshness probe failed for target_type=%s (%d targets); "
                "keeping cached entries",
                target_type, len(names),
            )
            return None

        result: dict[str, datetime] = {}
        for row in rows:
            name = row.get("target_name")
            db_max = _parse_datetime(row.get("mx"))
            # An unreadable timestamp is unknown, not stale.
            if name and db_max is not None:
                result[name] = db_max
        return result

    def _evict_changed(
        self,
        ontology: str,
        changed: list[tuple[str, str, datetime, datetime]],
        conn_params: dict,
    ) -> None:
        """Drop exactly the properties whose updated_at moved past the watermark.

        Eviction rather than in-place replacement on purpose: the caller re-fetches
        them in the same request, and only the caller has the column type map that
        makes min/max values comparable. The properties are gone for the remainder
        of this lookup and back before the request ends.
        """
        clauses = [
            f"(target_type = '{t}' AND target_name = '{n}' "
            f"AND updated_at >= '{previous.isoformat(sep=' ')}')"
            for t, n, previous, _ in changed
        ]
        query = (
            f"SELECT target_type, target_name, property_name "
            f"FROM timbr.sys_properties_statistics "
            f"WHERE {' OR '.join(clauses)}"
        )
        try:
            rows = _timbr_utils.run_query(query, conn_params)
        except Exception:
            logger.warning(
                "Changed-property query failed for ontology=%s; keeping cached entries",
                ontology,
            )
            return  # watermarks stay put, so the next interval retries

        dropped = 0
        with self._lock:
            for row in rows:
                key: _CacheKey = (
                    ontology,
                    row.get("target_type") or "mapping",
                    row.get("target_name") or "",
                    row.get("property_name") or "",
                )
                entry = self._cache.pop(key, None)
                if entry is not None:
                    self._total_bytes -= entry.size_bytes
                    dropped += 1
            # Only now that the changed set is known do the watermarks advance.
            for target_type, name, _, db_max in changed:
                self._target_watermark[(ontology, target_type, name)] = db_max

        if dropped:
            logger.debug(
                "Freshness check dropped %d changed properties across %d targets",
                dropped, len(changed),
            )


def _estimate_row_size_bytes(row: RawStatsRow) -> int:
    """Rough memory estimate for a single RawStatsRow."""
    total = 200  # base overhead
    if row.top_k:
        # value + the two derived normalized forms it now carries
        total += sum(
            (len(e.value) + len(e.norm or "") + len(e.norm_space or "")) * 2 + 16
            for e in row.top_k
        )
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
