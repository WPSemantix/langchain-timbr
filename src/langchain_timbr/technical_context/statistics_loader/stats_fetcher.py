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
from .ontology_cache import load_mapping_properties_index

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


def _build_compound_mapping_query(
    chunk: list[str],
    per_mapping_missing: dict[str, set[str] | None],
    props_index: dict[str, set[str]],
    include_properties: list[str] | None = None,
    exclude_properties: list[str] | None = None,
) -> str:
    """Build a compound OR query fetching per-mapping missing properties.

    Produces:
        SELECT ... FROM ... WHERE target_type = 'mapping' AND (
            (target_name = 'X' AND property_name IN ('a','b'))
            OR (target_name = 'Y' AND property_name IN ('c','d'))
            ...
        )

    For each mapping, the properties to fetch are:
        intersection of (missing_from_cache, exists_in_index, include_properties)
        minus exclude_properties.

    If a mapping is not in the index or missing_props is None (fetch all),
    it uses include_properties as the filter (or no property filter for that mapping).
    """
    include_set = set(include_properties) if include_properties else None
    exclude_set = set(exclude_properties) if exclude_properties else None

    or_parts: list[str] = []
    for mapping_name in chunk:
        missing_props = per_mapping_missing.get(mapping_name)
        available_props = props_index.get(mapping_name)

        # Determine which properties to fetch for this mapping
        if missing_props is None:
            # Cache said "fetch all" for this target
            if include_set:
                needed = set(include_set)
            elif available_props:
                needed = set(available_props)
            else:
                needed = None  # no filter — fetch everything for this mapping
        else:
            # Intersect missing with what exists in the index
            if available_props:
                needed = missing_props & available_props
            else:
                # Not in index (stale index?) — use missing as-is
                needed = set(missing_props)

            # Intersect with include_properties if specified
            if include_set:
                needed = needed & include_set

        # Apply exclude
        if needed is not None and exclude_set:
            needed -= exclude_set

        # Build the clause for this mapping
        safe_name = mapping_name.replace("'", "''")
        if needed is not None and needed:
            safe_props = _validate_property_names(sorted(needed))
            if safe_props:
                prop_in = ", ".join(f"'{p}'" for p in safe_props)
                or_parts.append(
                    f"(target_name = '{safe_name}' AND property_name IN ({prop_in}))"
                )
            else:
                # All prop names failed validation — fetch all for this mapping
                or_parts.append(f"(target_name = '{safe_name}')")
        elif needed is None:
            # No property filter for this mapping
            or_parts.append(f"(target_name = '{safe_name}')")
        # else: needed is empty set — nothing to fetch for this mapping, skip

    if not or_parts:
        # Nothing to fetch — return a query that returns no rows
        return (
            f"SELECT {_STATS_COLUMNS} "
            f"FROM timbr.sys_properties_statistics "
            f"WHERE 1 = 0"
        )

    or_clause = " OR ".join(or_parts)
    return (
        f"SELECT {_STATS_COLUMNS} "
        f"FROM timbr.sys_properties_statistics "
        f"WHERE target_type = 'mapping' AND ({or_clause})"
    )


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

    Cache (if provided) is consulted first at property-level granularity;
    only missing targets/properties are fetched from DB.
    Splits names into chunks of config.in_clause_chunk_size to avoid
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

    # Compute requested_properties for cache lookup.
    # When no include_properties given, use all columns from the type map (already
    # exclude-filtered by the caller) so the cache can do property-level lookup
    # instead of always falling back to DB.
    if include_properties:
        requested_properties: set[str] = set(include_properties)
        if exclude_properties:
            requested_properties -= set(exclude_properties)
    else:
        requested_properties = set()
        for col in columns_type_map.keys():
            # Normalize: extract short property name
            # - If col contains "]_", take everything after the last "]_"
            # - Else if col contains ".", take everything after the last "."
            if "]_" in col:
                prop = col.rsplit("]_", 1)[-1]
            elif "." in col:
                prop = col.rsplit(".", 1)[-1]
            else:
                prop = col

            if prop.startswith("_type_of") or prop.startswith("measure."):
                continue

            requested_properties.add(prop)
        if exclude_properties:
            requested_properties -= set(exclude_properties)

    # Check cache
    cached_rows: list[RawStatsRow] = []
    names_to_fetch: list[str] = sorted(mapping_names)
    # Per-mapping missing properties: {mapping_name: set of props to fetch}
    per_mapping_missing: dict[str, set[str] | None] = {}
    use_compound_filter = False

    if cache is not None:
        cached_rows, missing = cache.get_many(ontology, target_keys, requested_properties)
        if not missing:
            return cached_rows

        names_to_fetch = [name for (_, name, _) in missing]

        # Build per-mapping missing props dict
        for _, name, props in missing:
            per_mapping_missing[name] = props

        # Determine if we need per-mapping granularity (partial hit) or simple filter
        has_partial_hit = bool(cached_rows)
        if has_partial_hit:
            use_compound_filter = True

    # Load property index to know what actually exists per mapping in the DB.
    # This is cached until ontology version changes (typically 1 round-trip).
    props_index: dict[str, set[str]] | None = None
    if use_compound_filter:
        props_index = load_mapping_properties_index(conn_params)

    # Fetch missing from DB (chunked)
    fetched: list[RawStatsRow] = []
    chunk_size = config.in_clause_chunk_size

    for i in range(0, len(names_to_fetch), chunk_size):
        chunk = names_to_fetch[i : i + chunk_size]

        if use_compound_filter and props_index is not None:
            # Build compound OR filter: per-mapping property IN-clauses
            query = _build_compound_mapping_query(
                chunk, per_mapping_missing, props_index,
                include_properties, exclude_properties,
            )
        else:
            # Simple query: all mappings with a single property filter
            in_clause = ", ".join(f"'{name}'" for name in chunk)
            if include_properties:
                # Subtract exclude from include — single IN-clause is sufficient
                effective = sorted(set(include_properties) - set(exclude_properties or []))
                safe_names = _validate_property_names(effective)
                if safe_names:
                    prop_in = ", ".join(f"'{n}'" for n in safe_names)
                    prop_filter = f" AND property_name IN ({prop_in})"
                else:
                    prop_filter = ""
            else:
                # No include — only apply exclude if present
                prop_filter = _build_property_filter_clause(None, exclude_properties)
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

    # Cache fetched rows
    if cache is not None and fetched:
        cache.put_many(ontology, fetched)

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

    Cache (if provided) is consulted first at property-level granularity;
    only missing properties are fetched from DB.

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

    # Compute requested_properties for cache lookup.
    # When no include_properties given, use all columns from the type map (already
    # exclude-filtered by the caller) so the cache can do property-level lookup
    # instead of always falling back to DB.
    if include_properties:
        requested_properties: set[str] = set(include_properties)
        if exclude_properties:
            requested_properties -= set(exclude_properties)
    else:
        requested_properties = set(columns_type_map.keys())
        if exclude_properties:
            requested_properties -= set(exclude_properties)

    # Check cache
    cached_rows: list[RawStatsRow] = []
    missing_props: set[str] | None = None  # None = fetch all

    if cache is not None:
        cached_rows, missing = cache.get_many(ontology, target_keys, requested_properties)
        if not missing:
            return cached_rows
        # Extract missing_props from the single target
        _, _, missing_props = missing[0]

    # Build property filter for DB query
    # Only narrow the filter when we have an explicit whitelist OR a
    # partial cache hit (not everything is missing).  On a full miss
    # without include_properties we fetch all columns — no IN-clause.
    if missing_props is not None and (
        include_properties or missing_props != requested_properties
    ):
        safe_names = _validate_property_names(list(missing_props))
        if safe_names:
            prop_in = ", ".join(f"'{n}'" for n in safe_names)
            fetch_prop_filter = f" AND property_name IN ({prop_in})"
        else:
            fetch_prop_filter = ""
    else:
        fetch_prop_filter = _build_property_filter_clause(include_properties, exclude_properties)

    query = (
        f"SELECT {_STATS_COLUMNS} "
        f"FROM timbr.sys_properties_statistics "
        f"WHERE target_name = '{view_name}' AND target_type = 'view'"
        f"{fetch_prop_filter}"
    )

    try:
        rows = _timbr_utils.run_query(query, conn_params)
    except Exception:
        raise  # db_executor errors propagate

    fetched: list[RawStatsRow] = []
    for row in rows:
        parsed = _parse_row(row, columns_type_map)
        if parsed:
            fetched.append(parsed)

    if cache is not None and fetched:
        cache.put_many(ontology, fetched)

    return cached_rows + fetched


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
