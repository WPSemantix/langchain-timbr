"""Batched statistics fetcher from timbr.sys_properties_statistics.

Fetches raw statistics rows for either:
  - A set of mapping names (dtimbr path) — chunked by IN-clause size
  - A single view name (vtimbr path) — single query
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, TYPE_CHECKING

from ...utils import timbr_utils as _timbr_utils
from .types import RawStatsRow, TopKEntry
from .stats_parser import parse_stats_json
from .config import StatisticsLoaderConfig

if TYPE_CHECKING:
    from .stats_cache import StatsCache

logger = logging.getLogger(__name__)

_STATS_COLUMNS = (
    "property_name, target_name, target_type, "
    "distinct_count, non_null_count, stats, updated_at"
)

# Allowed characters in property names for SQL interpolation (letters, digits, underscore, dot, brackets)
_SAFE_PROPERTY_NAME_RE = re.compile(r"^[\w.\[\]]+$", re.UNICODE)


def _validate_property_names(names: list[str]) -> list[str]:
    """Validate and return only safe property names for SQL interpolation."""
    safe = []
    for name in names:
        if name and _SAFE_PROPERTY_NAME_RE.match(name):
            safe.append(name)
        else:
            logger.warning("Skipping unsafe property name for SQL filter: %r", name)
    return safe


def _build_property_filter_clause(
    include_properties: list[str] | None = None,
    exclude_properties: list[str] | None = None,
) -> str:
    """Build SQL WHERE clause fragment for property name filtering.

    Returns empty string if no filtering is needed.
    """
    parts: list[str] = []
    if include_properties:
        safe_names = _validate_property_names(include_properties)
        if safe_names:
            in_list = ", ".join(f"'{n}'" for n in safe_names)
            parts.append(f" AND property_name IN ({in_list})")
    if exclude_properties:
        safe_names = _validate_property_names(exclude_properties)
        if safe_names:
            in_list = ", ".join(f"'{n}'" for n in safe_names)
            parts.append(f" AND property_name NOT IN ({in_list})")
    return "".join(parts)


def fetch_stats_for_mappings(
    mapping_names: set[str],
    conn_params: dict,
    columns_type_map: dict[str, str],
    config: StatisticsLoaderConfig,
    cache: StatsCache | None = None,
    include_properties: list[str] | None = None,
    exclude_properties: list[str] | None = None,
) -> list[RawStatsRow]:
    """Fetch statistics rows for a set of mapping names (dtimbr path).

    Cache (if provided) is consulted first; only missing targets are fetched from DB.
    Splits mapping_names into chunks of config.in_clause_chunk_size to avoid
    overly large IN-clauses.

    Args:
        mapping_names: Set of mapping names to fetch stats for.
        conn_params: Timbr connection parameters.
        columns_type_map: Dict of property_name -> sql_type for type-aware parsing.
        config: Statistics loader configuration.
        cache: Optional StatsCache instance for caching fetched rows.
        include_properties: Whitelist of property names to fetch. Empty/None = all.
        exclude_properties: Blacklist of property names to exclude. Empty/None = none.

    Returns:
        Flat list of RawStatsRow across all chunks.
    """
    if not mapping_names:
        return []

    ontology = conn_params.get("ontology", "")
    target_keys = [("mapping", name) for name in sorted(mapping_names)]

    # Check cache (also triggers idle sweep + batch validation)
    cached_rows: list[RawStatsRow] = []
    missing_keys: list[tuple[str, str]] = target_keys
    if cache is not None:
        cached_by_key, missing_keys = cache.get_many(ontology, target_keys)
        for rows in cached_by_key.values():
            cached_rows.extend(rows)

    if not missing_keys:
        return cached_rows

    # Fetch missing mappings from DB (chunked)
    missing_names = [name for (_, name) in missing_keys]
    fetched: list[RawStatsRow] = []
    chunk_size = config.in_clause_chunk_size

    for i in range(0, len(missing_names), chunk_size):
        chunk = missing_names[i : i + chunk_size]
        in_clause = ", ".join(f"'{name}'" for name in chunk)
        prop_filter = _build_property_filter_clause(include_properties, exclude_properties)
        query = (
            f"SELECT {_STATS_COLUMNS} "
            f"FROM timbr.sys_properties_statistics "
            f"WHERE target_type = 'mapping' AND target_name IN ({in_clause})"
            f"{prop_filter}"
        )

        try:
            rows = _timbr_utils.run_query(query, conn_params)
        except Exception:
            raise  # db_executor errors propagate

        for row in rows:
            parsed = _parse_row(row, columns_type_map)
            if parsed:
                fetched.append(parsed)

    # Group fetched rows by (target_type, target_name) and cache
    if cache is not None and fetched:
        by_target: dict[tuple[str, str], list[RawStatsRow]] = {}
        for row in fetched:
            by_target.setdefault((row.target_type, row.target_name), []).append(row)
        cache.put_many(ontology, by_target)

    return cached_rows + fetched


def fetch_stats_for_view(
    view_name: str,
    conn_params: dict,
    columns_type_map: dict[str, str],
    cache: StatsCache | None = None,
    include_properties: list[str] | None = None,
    exclude_properties: list[str] | None = None,
) -> list[RawStatsRow]:
    """Fetch statistics rows for a single view (vtimbr path).

    Args:
        view_name: The view name to fetch stats for.
        conn_params: Timbr connection parameters.
        columns_type_map: Dict of property_name -> sql_type for type-aware parsing.
        cache: Optional StatsCache instance for caching fetched rows.
        include_properties: Whitelist of property names to fetch. Empty/None = all.
        exclude_properties: Blacklist of property names to exclude. Empty/None = none.

    Returns:
        List of RawStatsRow for the view.
    """
    ontology = conn_params.get("ontology", "")
    target_keys = [("view", view_name)]

    if cache is not None:
        cached_by_key, missing_keys = cache.get_many(ontology, target_keys)
        if not missing_keys:
            return cached_by_key.get(("view", view_name), [])

    prop_filter = _build_property_filter_clause(include_properties, exclude_properties)
    query = (
        f"SELECT {_STATS_COLUMNS} "
        f"FROM timbr.sys_properties_statistics "
        f"WHERE target_name = '{view_name}' AND target_type = 'view'"
        f"{prop_filter}"
    )

    try:
        rows = _timbr_utils.run_query(query, conn_params)
    except Exception:
        raise  # db_executor errors propagate

    result: list[RawStatsRow] = []
    for row in rows:
        parsed = _parse_row(row, columns_type_map)
        if parsed:
            result.append(parsed)

    if cache is not None and result:
        cache.put_many(ontology, {("view", view_name): result})

    return result


def _parse_row(row: dict, columns_type_map: dict[str, str]) -> RawStatsRow | None:
    """Parse a single result row into a RawStatsRow."""
    property_name = row.get("property_name")
    if not property_name:
        return None

    target_name = row.get("target_name", "")
    target_type = row.get("target_type", "mapping")

    # Parse counts with -1 sentinel for missing
    distinct_count = _safe_int(row.get("distinct_count"), default=-1)
    non_null_count = _safe_int(row.get("non_null_count"), default=-1)

    # Parse stats JSON
    stats_str = row.get("stats")
    sql_type = columns_type_map.get(property_name)
    top_k, min_value, max_value = parse_stats_json(stats_str, sql_type)

    # Parse updated_at
    updated_at = _parse_datetime(row.get("updated_at"))

    # Retain raw stats dict for debugging
    raw_stats = None
    if stats_str:
        try:
            import json
            raw_stats = json.loads(stats_str)
        except Exception:
            pass

    return RawStatsRow(
        property_name=property_name,
        target_name=target_name,
        target_type=target_type,
        distinct_count=distinct_count,
        non_null_count=non_null_count,
        top_k=top_k,
        min_value=min_value,
        max_value=max_value,
        raw_stats=raw_stats,
        updated_at=updated_at,
    )


def _safe_int(value: Any, default: int = -1) -> int:
    """Safely convert a value to int, returning default on failure."""
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


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
