"""Per-column annotation assembly.

Builds structured payloads for each column, then formats them into strings.

Two-stage pipeline:
1. assemble_column_payload() → ColumnPayload (structured, trimmable)
2. format_annotation() → str | None (final rendered text)

The trimmer operates between stages 1 and 2 on the structured payloads.

Two-threshold ranking (applies uniformly across all modes):
- Strong match (score >= surface_threshold): sorts to front, protected from trim
- Weak match (sort_threshold <= score < surface_threshold): sorts after strong,
  not protected from trim (nice to have)
- Unmatched: frequency order, follows weak matches

Mode differences (affect starting K in payload assembly):
- include_all: starting K = full top_k, two-tier ranking applied
- filter_matched: starting K = max_values_per_column, strong matches pulled in
  from full top_k even if outside default K
"""

from __future__ import annotations

from typing import Literal

from ..config import TechnicalContextConfig
from ..types import ColumnPayload, ColumnRef, MatchResult, SemanticType
from ..statistics_loader.types import ColumnStatistics

# Number of backup values for unmatched columns in filter_matched mode
_FILTER_MATCHED_BACKUP_K = 5


def assemble_column_payload(
    col_ref: ColumnRef,
    stats: ColumnStatistics | None,
    matches: list[MatchResult],
    config: TechnicalContextConfig,
    effective_mode: Literal["include_all", "filter_matched"] = "include_all",
) -> ColumnPayload | None:
    """Build structured payload for a single column.

    Returns a ColumnPayload that the trimmer can reduce (for top_k hint)
    or that is protected (for all other hints). Returns None only when
    no meaningful annotation can be produced (ID with no matches, no stats).

    The values list in the payload is already ordered: matched values first,
    then by frequency. Simple [:k] slicing preserves matches automatically.

    Args:
        col_ref: Column reference with semantic type info.
        stats: Column statistics (may be None).
        matches: Match results for this column.
        config: Configuration for limits and truncation.
        effective_mode: The effective mode controlling value breadth.

    Returns:
        ColumnPayload or None.
    """
    sem_type = col_ref.semantic_type
    has_matches = bool(matches)
    matched_value_set = {m.matched_value for m in matches} if matches else set()

    # ID columns: name_only (no annotation unless has matches)
    if sem_type == SemanticType.ID:
        if has_matches:
            matched_vals = [m.matched_value for m in matches]
            return ColumnPayload(
                format_hint="name_only",
                values=matched_vals,
                matched_values=matched_value_set,
                distinct_count=stats.distinct_count if stats else -1,
            )
        return None

    # FREE_TEXT columns: count_only
    if sem_type == SemanticType.FREE_TEXT:
        values = [m.matched_value for m in matches] if has_matches else []
        return ColumnPayload(
            format_hint="count_only",
            values=values,
            matched_values=matched_value_set,
            distinct_count=stats.distinct_count if stats and stats.distinct_count else -1,
        )

    # NUMERIC columns: min_max range
    if sem_type == SemanticType.NUMERIC:
        if not stats or (stats.min_value is None and stats.max_value is None):
            if has_matches:
                return ColumnPayload(
                    format_hint="name_only",
                    values=[m.matched_value for m in matches],
                    matched_values=matched_value_set,
                )
            return None
        return ColumnPayload(
            format_hint="min_max",
            values=[m.matched_value for m in matches] if has_matches else [],
            matched_values=matched_value_set,
            distinct_count=stats.distinct_count if stats else -1,
            min_value=stats.min_value,
            max_value=stats.max_value,
            range_label="value range",
        )

    # DATE columns: min_max range
    if sem_type == SemanticType.DATE:
        if not stats or (stats.min_value is None and stats.max_value is None):
            if has_matches:
                return ColumnPayload(
                    format_hint="name_only",
                    values=[m.matched_value for m in matches],
                    matched_values=matched_value_set,
                )
            return None
        return ColumnPayload(
            format_hint="min_max",
            values=[m.matched_value for m in matches] if has_matches else [],
            matched_values=matched_value_set,
            distinct_count=stats.distinct_count if stats else -1,
            min_value=stats.min_value,
            max_value=stats.max_value,
            range_label="date range",
        )

    # BOOLEAN columns
    if sem_type == SemanticType.BOOLEAN:
        if not stats or not stats.top_k:
            return None
        values = [str(e.value) for e in stats.top_k[:2]]
        return ColumnPayload(
            format_hint="boolean",
            values=values,
            matched_values=matched_value_set,
            distinct_count=stats.distinct_count if stats else -1,
        )

    # CATEGORICAL_TEXT, CODE_LIKE, BUSINESS_KEY_LIKE: top_k or all
    if not stats or not stats.top_k:
        if has_matches:
            return ColumnPayload(
                format_hint="top_k",
                values=[m.matched_value for m in matches],
                matched_values=matched_value_set,
                distinct_count=stats.distinct_count if stats else -1,
            )
        return None

    # Determine format_hint: "all" if show_all_under tier, else "top_k"
    is_show_all = stats.distinct_count and stats.distinct_count <= config.show_all_under

    # Determine starting K
    if is_show_all:
        k = len(stats.top_k)
    elif sem_type == SemanticType.BUSINESS_KEY_LIKE:
        k = 3
    elif effective_mode == "include_all":
        k = len(stats.top_k)
    elif effective_mode == "filter_matched":
        k = config.max_values_per_column if has_matches else _FILTER_MATCHED_BACKUP_K
    else:
        k = config.max_values_per_column

    # Two-threshold ranking: strong matches first, weak second, frequency rest
    all_top_k_values = [str(e.value) for e in stats.top_k]
    values, strong_count = _order_values_with_matches(
        all_top_k_values, matches, config, sem_type, k, effective_mode,
    )

    if not values and not has_matches:
        return None

    format_hint: Literal["top_k", "all"] = "all" if is_show_all else "top_k"

    # strong_matched_values: only strong matches (protected from trim)
    surface_threshold = (
        config.fuzzy_threshold_strict
        if sem_type in (SemanticType.CODE_LIKE, SemanticType.BUSINESS_KEY_LIKE)
        else config.fuzzy_threshold_default
    )
    strong_matched_values = {m.matched_value for m in matches if m.score >= surface_threshold}

    return ColumnPayload(
        format_hint=format_hint,
        values=values,
        matched_values=strong_matched_values,
        distinct_count=stats.distinct_count if stats.distinct_count else -1,
    )


def _order_values_with_matches(
    all_top_k_values: list[str],
    matches: list[MatchResult],
    config: TechnicalContextConfig,
    semantic_type: SemanticType | None,
    k: int,
    effective_mode: str,
) -> tuple[list[str], int]:
    """Order values: strong matches first, weak matches second, frequency rest.

    Two thresholds based on semantic type:
    - surface_threshold: strong match (protected from trim, sort to very front)
    - sort_threshold: weak match (sort after strong, NOT protected)

    In filter_matched mode, strong matches from FULL top_k are pulled into the
    displayed values even if they're outside the default K window.

    Args:
        all_top_k_values: All values from stats.top_k (frequency order).
        matches: Match results with scores.
        config: Configuration with thresholds.
        semantic_type: Column's semantic type.
        k: Starting K (how many values to show).
        effective_mode: "include_all" or "filter_matched".

    Returns:
        (ordered_values, strong_match_count)
    """
    if not matches:
        return all_top_k_values[:k], 0

    # Select thresholds based on semantic type
    if semantic_type in (SemanticType.CODE_LIKE, SemanticType.BUSINESS_KEY_LIKE):
        surface_threshold = config.fuzzy_threshold_strict
    else:
        surface_threshold = config.fuzzy_threshold_default
    sort_threshold = surface_threshold - config.fuzzy_sort_gap

    # Bucket matches by strength
    strong = [m for m in matches if m.score >= surface_threshold]
    weak = [m for m in matches if sort_threshold <= m.score < surface_threshold]

    # Sort each bucket by score DESC
    strong.sort(key=lambda m: -m.score)
    weak.sort(key=lambda m: -m.score)

    matched_values_strong = [m.matched_value for m in strong]
    matched_values_weak = [m.matched_value for m in weak]
    matched_set = set(matched_values_strong) | set(matched_values_weak)

    # In filter_matched mode, pull strong matches from full top_k even if outside K window
    if effective_mode == "filter_matched":
        # Strong matches get included regardless of their position in top_k
        window_values = set(all_top_k_values[:k])
        for sv in matched_values_strong:
            if sv not in window_values:
                window_values.add(sv)
        # Frequency-ordered remainder from the K window (excluding matched)
        remainder = [v for v in all_top_k_values[:k] if v not in matched_set]
    else:
        # include_all: use full available K
        remainder = [v for v in all_top_k_values[:k] if v not in matched_set]

    # Final order: strong → weak → frequency remainder
    ordered = matched_values_strong + matched_values_weak + remainder

    # For BUSINESS_KEY_LIKE with matches, limit unmatched trailing values
    if semantic_type == SemanticType.BUSINESS_KEY_LIKE and (strong or weak):
        n_matched = len(matched_values_strong) + len(matched_values_weak)
        ordered = ordered[:n_matched + 3]

    return ordered, len(strong)


def format_annotation(
    payload: ColumnPayload,
    config: TechnicalContextConfig,
) -> str | None:
    """Render a ColumnPayload into a final annotation string.

    This runs AFTER trimming — the payload's values list has already been
    reduced to the final K by the trimmer.

    Args:
        payload: Structured column payload.
        config: Configuration for value truncation.

    Returns:
        Annotation string or None (for name_only with no values).
    """
    hint = payload.format_hint

    if hint == "name_only":
        # ID columns with matches: show matched values
        if payload.values:
            formatted = [f"'{v}'" for v in payload.values]
            return f"matched values from prompt: [{', '.join(formatted)}]"
        return None

    if hint == "count_only":
        # FREE_TEXT: distinct count + optional matched values
        parts: list[str] = []
        if payload.distinct_count > 0:
            parts.append(f"({payload.distinct_count} distinct values)")
        if payload.values and payload.matched_values:
            formatted = [f"'{v}'" for v in payload.values]
            parts.append(f"matched values from prompt: [{', '.join(formatted)}]")
        return ", ".join(parts) if parts else None

    if hint == "min_max":
        # Numeric/date range
        parts = []
        if payload.min_value is not None and payload.max_value is not None:
            parts.append(f"{payload.range_label}: {payload.min_value} to {payload.max_value}")
        if payload.values and payload.matched_values:
            formatted = [f"'{v}'" for v in payload.values]
            parts.append(f"matched values from prompt: [{', '.join(formatted)}]")
        return ", ".join(parts) if parts else None

    if hint == "boolean":
        if payload.values:
            return f"values: [{', '.join(payload.values)}]"
        return None

    # "top_k" or "all": known values list
    if not payload.values:
        return None

    formatted = [f"'{v}'" for v in payload.values]
    result = f"known values: [{', '.join(formatted)}]"

    # Add distinct count hint if there are more values than shown
    if payload.distinct_count > 0 and payload.distinct_count > len(payload.values):
        result += f" ({payload.distinct_count} distinct total)"

    return result


def assemble_annotation(
    col_ref: ColumnRef,
    stats: ColumnStatistics | None,
    matches: list[MatchResult],
    config: TechnicalContextConfig,
    effective_mode: Literal["include_all", "filter_matched"] = "include_all",
) -> str | None:
    """Convenience: assemble payload then format in one step.

    For direct use when trimming is not needed (e.g., tests, single-column queries).
    The orchestrator uses the two-stage API (assemble_column_payload + format_annotation)
    with trim_to_budget in between.
    """
    payload = assemble_column_payload(col_ref, stats, matches, config, effective_mode)
    if payload is None:
        return None
    return format_annotation(payload, config)


def _truncate(value: str, max_chars: int) -> str:
    """Truncate a value string if too long."""
    if len(value) <= max_chars:
        return value
    return value[:max_chars - 3] + "..."
