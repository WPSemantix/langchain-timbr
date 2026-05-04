"""Main orchestration: load_column_statistics entry point.

Dispatches to vtimbr or dtimbr path based on schema parameter.
"""

from __future__ import annotations

import logging
from typing import Optional

from .config import StatisticsLoaderConfig
from .types import (
    ColumnPath,
    ColumnStatistics,
    ConceptMappingSet,
    RawStatsRow,
)
from .path_parser import parse_column_path, ColumnPathParseError
from .ontology_cache import load_ontology_concepts, load_concept_mappings, load_view_row_counts
from .inheritance import build_descendants_map
from .mapping_resolver import resolve_concept_mappings
from .stats_fetcher import fetch_stats_for_mappings, fetch_stats_for_view
from .stats_cache import StatsCache
from .stats_merger import merge_rows

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = StatisticsLoaderConfig()


def load_column_statistics(
    schema: str,
    table_name: str,
    columns: list[dict],
    conn_params: dict,
    config: Optional[StatisticsLoaderConfig] = None,
    cache: Optional[StatsCache] = None,
) -> dict[str, ColumnStatistics]:
    """Load per-column statistics from timbr.sys_properties_statistics.

    Dispatches to vtimbr (view) or dtimbr (concept) path based on schema.
    Resolves multi-hop relationship column paths, merges across mappings,
    and returns a dict keyed by column name (verbatim from input).

    Args:
        schema: Either "vtimbr" or "dtimbr".
        table_name: Concept name (dtimbr) or view name (vtimbr).
        columns: List of column dicts with at minimum "name" and "type" keys.
                 Example: [{"name": "customer_id", "type": "bigint"}, ...]
        conn_params: Timbr connection parameters dict.
        config: Optional configuration. Uses defaults if not provided.

    Returns:
        Dict mapping column name -> ColumnStatistics.

    Raises:
        Exception: If db_executor (run_query) fails — propagated.

    Example:
        >>> from langchain_timbr.technical_context.statistics_loader import (
        ...     load_column_statistics, StatisticsLoaderConfig,
        ... )
        >>> stats = load_column_statistics(
        ...     schema="dtimbr",
        ...     table_name="customer",
        ...     columns=[
        ...         {"name": "customer_id", "type": "bigint"},
        ...         {"name": "orders[order].total", "type": "decimal(18,2)"},
        ...     ],
        ...     conn_params=conn_params,
        ... )
    """
    if config is None:
        config = _DEFAULT_CONFIG

    if schema == "vtimbr":
        return _load_vtimbr(table_name, columns, conn_params, config, cache)
    else:
        return _load_dtimbr(schema, table_name, columns, conn_params, config, cache)


def _load_vtimbr(
    view_name: str,
    columns: list[dict],
    conn_params: dict,
    config: StatisticsLoaderConfig,
    cache: Optional[StatsCache] = None,
) -> dict[str, ColumnStatistics]:
    """vtimbr path: single view, no merge needed."""
    # 1. Get view row count (cached)
    view_row_counts = load_view_row_counts(conn_params=conn_params)
    view_rows = view_row_counts.get(view_name, -1)
    if view_rows == -1:
        logger.info("View '%s' not found in sys_views, total_source_rows will be -1", view_name)

    # 2. Build type map for parsing
    columns_type_map = _build_type_map(columns)

    # 3. Fetch view stats (single query)
    raw_rows = fetch_stats_for_view(
        view_name, conn_params, columns_type_map, cache=cache,
        include_properties=config.include_properties or None,
        exclude_properties=config.exclude_properties or None,
    )

    # 4. Index by property_name
    stats_by_prop: dict[str, RawStatsRow] = {}
    for row in raw_rows:
        stats_by_prop[row.property_name] = row

    # 5. Build result per column
    result: dict[str, ColumnStatistics] = {}
    dummy_mapping_set = ConceptMappingSet(concept=view_name, mappings=[], total_rows=view_rows)

    for col in columns:
        col_name = col.get("name", "")
        raw_row = stats_by_prop.get(col_name)

        if raw_row:
            result[col_name] = ColumnStatistics(
                distinct_count=raw_row.distinct_count,
                non_null_count=raw_row.non_null_count,
                top_k=raw_row.top_k,
                min_value=raw_row.min_value,
                max_value=raw_row.max_value,
                updated_at=raw_row.updated_at,
                approx_union=False,
                total_source_rows=view_rows,
                contributing_mappings=[raw_row.target_name],
            )
        else:
            result[col_name] = _sentinel(view_rows, config)

    return result


def _load_dtimbr(
    schema: str,
    table_name: str,
    columns: list[dict],
    conn_params: dict,
    config: StatisticsLoaderConfig,
    cache: Optional[StatsCache] = None,
) -> dict[str, ColumnStatistics]:
    """dtimbr path: parse paths, resolve mappings, fetch and merge."""
    # A.1 — Parse columns into paths
    paths: list[ColumnPath] = []
    path_errors: dict[str, str] = {}

    for col in columns:
        col_name = col.get("name", "")
        try:
            path = parse_column_path(col_name, table_name)
            paths.append(path)
        except ColumnPathParseError as e:
            logger.warning("Cannot parse column path '%s': %s", col_name, e)
            path_errors[col_name] = str(e)
            if config.on_missing_stats == "raise":
                raise

    # A.2 — Collect concept set
    concepts_needed: set[str] = {table_name}
    for path in paths:
        for _, concept in path.hops:
            concepts_needed.add(concept)
        concepts_needed.add(path.owning_concept)

    # A.3 — Load ontology data (cached)
    ontology = load_ontology_concepts(conn_params=conn_params)
    mappings_by_concept = load_concept_mappings(conn_params=conn_params)

    # A.4 — Build descendants map
    descendants = build_descendants_map(ontology, max_depth=config.inheritance_max_depth)

    # A.5 — Resolve mappings per concept
    concept_mapping_sets: dict[str, ConceptMappingSet] = {}
    all_mapping_names: set[str] = set()

    for concept in concepts_needed:
        if concept not in ontology:
            logger.warning("Concept '%s' not found in ontology", concept)
            concept_mapping_sets[concept] = ConceptMappingSet(
                concept=concept, mappings=[], total_rows=0
            )
            continue

        mapping_set = resolve_concept_mappings(
            concept=concept,
            ontology=ontology,
            mappings_by_concept=mappings_by_concept,
            descendants=descendants,
            config=config,
        )
        concept_mapping_sets[concept] = mapping_set
        all_mapping_names.update(m.mapping_name for m in mapping_set.mappings)

    # A.6 — Build type map and fetch stats (batched)
    columns_type_map = _build_type_map(columns)
    # Also add final_property -> type for relationship columns
    for col in columns:
        col_name = col.get("name", "")
        col_type = col.get("type", "")
        if "[" in col_name and "." in col_name:
            # Extract final property name
            final_prop = col_name.rsplit(".", 1)[-1] if "." in col_name else col_name
            if final_prop and final_prop not in columns_type_map:
                columns_type_map[final_prop] = col_type

    raw_rows = fetch_stats_for_mappings(
        mapping_names=all_mapping_names,
        conn_params=conn_params,
        columns_type_map=columns_type_map,
        config=config,
        cache=cache,
        include_properties=config.include_properties or None,
        exclude_properties=config.exclude_properties or None,
    )

    # A.7 — Index raw rows by (target_name, property_name) for fast lookup
    rows_by_mapping_prop: dict[tuple[str, str], list[RawStatsRow]] = {}
    for row in raw_rows:
        key = (row.target_name, row.property_name)
        if key not in rows_by_mapping_prop:
            rows_by_mapping_prop[key] = []
        rows_by_mapping_prop[key].append(row)

    # A.8 — Match & merge per column
    result: dict[str, ColumnStatistics] = {}

    for path in paths:
        mapping_set = concept_mapping_sets.get(path.owning_concept)
        if not mapping_set:
            result[path.raw] = _sentinel(-1, config)
            continue

        # Collect all stats rows matching this column's final_property
        # across the owning concept's mappings
        matching_rows: list[RawStatsRow] = []
        mapping_name_set = {m.mapping_name for m in mapping_set.mappings}

        for mapping_name in mapping_name_set:
            key = (mapping_name, path.final_property)
            matching_rows.extend(rows_by_mapping_prop.get(key, []))

        result[path.raw] = merge_rows(matching_rows, mapping_set)

    # Add sentinel for unparseable columns
    for col_name in path_errors:
        result[col_name] = _sentinel(-1, config)

    return result


def _build_type_map(columns: list[dict]) -> dict[str, str]:
    """Build a property_name -> sql_type mapping from input columns."""
    type_map: dict[str, str] = {}
    for col in columns:
        name = col.get("name", "")
        col_type = col.get("type", "")
        if name and col_type:
            type_map[name] = col_type
    return type_map


def _sentinel(total_source_rows: int, config: StatisticsLoaderConfig) -> ColumnStatistics:
    """Create a sentinel ColumnStatistics for missing data."""
    return ColumnStatistics(
        distinct_count=-1,
        non_null_count=-1,
        top_k=None,
        min_value=None,
        max_value=None,
        updated_at=None,
        approx_union=False,
        total_source_rows=total_source_rows,
        contributing_mappings=[],
    )
