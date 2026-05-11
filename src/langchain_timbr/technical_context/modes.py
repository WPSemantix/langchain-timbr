"""Mode selection logic for the Technical Context Builder.

Modes control two things:
1. How candidates are extracted from the user's prompt (heuristic vs LLM)
2. How many values per column to include (broad vs focused starting K)

Critical: ALL in-scope columns with stats get annotated in every mode.
No column is skipped based on whether it had a match.
"""

from __future__ import annotations

import logging
from typing import Any

from .config import TechnicalContextConfig
from .types import ColumnRef, MatchResult, SemanticType

logger = logging.getLogger(__name__)


def select_columns_for_annotation(
    columns: list[ColumnRef],
    matches_by_column: dict[str, list[MatchResult]],
    config: TechnicalContextConfig,
) -> list[ColumnRef]:
    """Select which columns should receive annotations.

    All in-scope columns with stats get annotated in every mode.
    This function returns all columns unconditionally — mode differences
    are handled in the assembly step (starting K and backup behavior).

    Args:
        columns: All available columns with semantic types.
        matches_by_column: Column name -> list of MatchResults (unused for filtering).
        config: Configuration (unused for filtering).

    Returns:
        All columns (no filtering).
    """
    return columns


def estimate_include_all_cost(
    columns: list[ColumnRef],
    stats_map: dict[str, Any],
    config: "TechnicalContextConfig",
) -> int:
    """Estimate the token cost of the technical context annotations in include_all mode.

    Uses tiktoken (cl100k_base) for accurate token counting.
    Only counts the annotation text itself (not column name/type headers).

    Args:
        columns: All column refs.
        stats_map: Column name -> ColumnStatistics.
        config: Configuration with show_all_under.

    Returns:
        Estimated token count for the TC annotations.
    """
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        # Fallback: rough estimate chars / 4
        return _estimate_cost_fallback(columns, stats_map, config)

    total_tokens = 0
    for col in columns:
        stats = stats_map.get(col.name)
        if not stats:
            continue

        annotation_text = _project_annotation_text(col, stats, config)
        if annotation_text:
            total_tokens += len(encoding.encode(annotation_text))

    return total_tokens


def _estimate_cost_fallback(
    columns: list[ColumnRef],
    stats_map: dict[str, Any],
    config: "TechnicalContextConfig",
) -> int:
    """Fallback cost estimation when tiktoken is unavailable (chars / 4)."""
    total_chars = 0
    for col in columns:
        stats = stats_map.get(col.name)
        if not stats:
            continue
        annotation_text = _project_annotation_text(col, stats, config)
        if annotation_text:
            total_chars += len(annotation_text)
    return total_chars // 4


def _project_annotation_text(
    col: ColumnRef,
    stats: Any,
    config: "TechnicalContextConfig",
) -> str:
    """Project what annotation text a column would produce in include_all mode.

    Uses all available top_k values (broad coverage).
    """
    sem_type = col.semantic_type

    if sem_type == SemanticType.ID:
        # surrogate_id_like: name + type only (no values)
        return ""

    if sem_type == SemanticType.FREE_TEXT:
        # free_text: count note only
        if stats.distinct_count:
            return f"({stats.distinct_count} distinct values)"
        return ""

    if sem_type == SemanticType.NUMERIC:
        if stats.min_value is not None and stats.max_value is not None:
            return f"value range: {stats.min_value} to {stats.max_value}"
        return ""

    if sem_type == SemanticType.DATE:
        if stats.min_value is not None and stats.max_value is not None:
            return f"date range: {stats.min_value} to {stats.max_value}"
        return ""

    # Categorical types: project value list at full starting K
    if not stats.top_k:
        return ""

    values = [str(e.value) for e in stats.top_k]
    formatted = [f"'{v}'" for v in values]
    return f"known values: [{', '.join(formatted)}]"
