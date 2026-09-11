"""Multi-mapping merge of raw statistics rows into final ColumnStatistics.

Handles:
  - Single-row pass-through (no merge needed)
  - Multi-row top_k union (sum counts by value)
  - Multi-row min/max aggregation
  - Mixed top_k/min-max defensive path
  - Sentinel generation for missing data
"""

from __future__ import annotations

import logging
from collections import defaultdict

from .types import RawStatsRow, TopKEntry, ColumnStatistics, ConceptMappingSet

logger = logging.getLogger(__name__)


def _is_canonical_top_k(entries: list[TopKEntry]) -> bool:
    """True when the union+sort below would return ``entries`` unchanged.

    That holds when values are unique and counts are non-increasing — which is
    what a single mapping's stats row already looks like. Checking costs one
    pass; rebuilding costs an allocation per value plus a sort.
    """
    seen: set[str] = set()
    prev: int | None = None
    for e in entries:
        if prev is not None and e.count > prev:
            return False
        if e.value in seen:
            return False
        seen.add(e.value)
        prev = e.count
    return True


def merge_rows(
    rows: list[RawStatsRow],
    mapping_set: ConceptMappingSet,
) -> ColumnStatistics:
    """Merge multiple RawStatsRow into a single ColumnStatistics.

    Args:
        rows: List of stats rows matching a single property across mappings.
        mapping_set: The resolved mapping set (for total_rows and context).

    Returns:
        Merged ColumnStatistics. Sentinel values used when no data available.
    """
    if not rows:
        return ColumnStatistics(
            distinct_count=-1,
            non_null_count=-1,
            top_k=None,
            min_value=None,
            max_value=None,
            updated_at=None,
            approx_union=False,
            total_source_rows=mapping_set.total_rows,
            contributing_mappings=[],
        )

    has_topk = any(r.top_k for r in rows)
    has_minmax = any(r.min_value is not None or r.max_value is not None for r in rows)

    if has_topk and has_minmax:
        # Mixed — should not happen. Prefer top_k.
        logger.warning(
            "Mixed top_k and min/max for property '%s' across mappings %s",
            rows[0].property_name,
            [r.target_name for r in rows],
        )

    top_k: list[TopKEntry] | None = None
    min_value = None
    max_value = None

    if has_topk:
        single = rows[0].top_k if len(rows) == 1 else None
        if single is not None and _is_canonical_top_k(single):
            # Nothing to merge — hand back the row's own list (copied, so callers
            # can't mutate the cached stats row).
            top_k = list(single)
        else:
            # Union by value, sum counts. The normalized forms are carried
            # from the source entries rather than recomputed — same string, same
            # answer, and recomputing here would undo the point of deriving them
            # once at parse time.
            merged: dict[str, int] = defaultdict(int)
            norms: dict[str, tuple[str | None, str | None]] = {}
            for r in rows:
                for entry in (r.top_k or []):
                    merged[entry.value] += entry.count
                    if entry.value not in norms:
                        norms[entry.value] = (entry.norm, entry.norm_space)
            top_k = sorted(
                [
                    TopKEntry(
                        value=v, count=c,
                        norm=norms.get(v, (None, None))[0],
                        norm_space=norms.get(v, (None, None))[1],
                    )
                    for v, c in merged.items()
                ],
                key=lambda e: -e.count,
            )
    elif has_minmax:
        mins = [r.min_value for r in rows if r.min_value is not None]
        maxs = [r.max_value for r in rows if r.max_value is not None]
        # Values already parsed to comparable types by stats_parser
        min_value = min(mins) if mins else None
        max_value = max(maxs) if maxs else None

    # Aggregate counts — sum across rows (upper bound for distinct)
    distinct_counts = [r.distinct_count for r in rows if r.distinct_count >= 0]
    non_null_counts = [r.non_null_count for r in rows if r.non_null_count >= 0]

    distinct_count = sum(distinct_counts) if distinct_counts else -1
    non_null_count = sum(non_null_counts) if non_null_counts else -1

    # Latest updated_at
    updated_ats = [r.updated_at for r in rows if r.updated_at is not None]
    updated_at = max(updated_ats) if updated_ats else None

    # Value kind: only claim numeric/date when every contributing mapping agrees.
    # One text mapping makes the merged column text, which is the safe direction —
    # it just means the column keeps the full matcher cascade.
    kinds = {r.value_kind for r in rows}
    value_kind = kinds.pop() if len(kinds) == 1 else "text"

    # Year index: union the months, sum the counts.
    date_years = None
    if value_kind == "date":
        acc: dict[str, tuple[set, int]] = {}
        for r in rows:
            for year, (months, count) in (r.date_years or {}).items():
                if year in acc:
                    acc[year][0].update(months)
                    acc[year] = (acc[year][0], acc[year][1] + count)
                else:
                    acc[year] = (set(months), count)
        date_years = {y: (frozenset(m), n) for y, (m, n) in acc.items()} or None

    return ColumnStatistics(
        distinct_count=distinct_count,
        non_null_count=non_null_count,
        top_k=top_k,
        min_value=min_value,
        max_value=max_value,
        updated_at=updated_at,
        value_kind=value_kind,
        date_years=date_years,
        approx_union=(len(rows) > 1),
        total_source_rows=mapping_set.total_rows,
        contributing_mappings=[r.target_name for r in rows],
    )
