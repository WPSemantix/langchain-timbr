"""Semantic type classification and ontology distance computation."""

from __future__ import annotations

import re
from typing import Any

from .types import SemanticType
from .statistics_loader.types import ColumnStatistics


# Patterns for code-like detection (short, alphanumeric, consistent format)
_CODE_LIKE_RE = re.compile(r"^[A-Z0-9_\-]{2,10}$")
# Patterns for business-key-like (structured with separators)
_BUSINESS_KEY_RE = re.compile(r"^[A-Z0-9]+[-_][A-Z0-9]+", re.IGNORECASE)


def classify_semantic_type(
    col_name: str,
    sql_type: str,
    stats: ColumnStatistics | None,
    *,
    free_text_threshold: int = 10000,
    id_unique_ratio: float = 0.95,
) -> SemanticType:
    """Classify a column's semantic type based on its SQL type and statistics.

    Args:
        col_name: Column name (used for heuristics).
        sql_type: SQL data type string (e.g., "int", "varchar(255)", "decimal(18,2)").
        stats: Column statistics (may be None or sentinel with -1 counts).
        free_text_threshold: Distinct count above which → FREE_TEXT.
        id_unique_ratio: Ratio above which → ID.

    Returns:
        SemanticType classification.
    """
    sql_lower = sql_type.lower() if sql_type else ""

    # Boolean detection
    if _is_boolean_type(sql_lower, stats):
        return SemanticType.BOOLEAN

    # Numeric types
    if _is_numeric_type(sql_lower):
        # Check if it's actually an ID
        if stats and _is_id_like(stats, id_unique_ratio):
            return SemanticType.ID
        return SemanticType.NUMERIC

    # Date/timestamp types
    if _is_date_type(sql_lower):
        return SemanticType.DATE

    # String types — need stats to discriminate
    if stats is None or stats.distinct_count == -1:
        return SemanticType.CATEGORICAL_TEXT

    # ID detection for strings
    if _is_id_like(stats, id_unique_ratio):
        return SemanticType.ID

    # Free text detection
    if stats.distinct_count > free_text_threshold:
        return SemanticType.FREE_TEXT

    # Code-like detection: check top_k values for pattern
    if stats.top_k and _looks_code_like(stats.top_k):
        return SemanticType.CODE_LIKE

    # Business-key-like detection
    if stats.top_k and _looks_business_key(stats.top_k):
        return SemanticType.BUSINESS_KEY_LIKE

    return SemanticType.CATEGORICAL_TEXT


def compute_ontology_distance(col_name: str) -> int:
    """Compute ontology distance from dot count in column name.

    Direct columns (no dots) → 0, one dot (e.g., "orders[order].total") → 1,
    two dots → 2, etc.

    Args:
        col_name: Full column name.

    Returns:
        Integer distance (0 = direct).
    """
    if not col_name:
        return 0
    return col_name.count(".")


def compute_priority_band(ontology_distance: int, is_matched: bool) -> int:
    """Compute priority band (1-5) from distance and match status.

    Band 1: direct (0 hops) + matched in prompt
    Band 2: direct (0 hops) + not matched
    Band 3: 1-hop + matched
    Band 4: 1-hop + not matched
    Band 5: 2+ hops

    Args:
        ontology_distance: From compute_ontology_distance().
        is_matched: Whether a prompt value matched this column.

    Returns:
        Priority band 1-5 (1 = highest priority).
    """
    if ontology_distance == 0:
        return 1 if is_matched else 2
    elif ontology_distance == 1:
        return 3 if is_matched else 4
    else:
        return 5


def _is_boolean_type(sql_lower: str, stats: ColumnStatistics | None) -> bool:
    """Check if column is boolean-like."""
    if "bool" in sql_lower or "bit" in sql_lower:
        return True
    if stats and stats.distinct_count == 2:
        # Check if values are boolean-like
        if stats.top_k and len(stats.top_k) == 2:
            vals = {str(e.value).lower() for e in stats.top_k}
            boolean_pairs = [
                {"true", "false"}, {"yes", "no"}, {"y", "n"},
                {"0", "1"}, {"t", "f"}, {"active", "inactive"},
            ]
            if vals in boolean_pairs:
                return True
    return False


def _is_numeric_type(sql_lower: str) -> bool:
    """Check if SQL type is numeric."""
    numeric_keywords = (
        "int", "bigint", "smallint", "tinyint", "float", "double",
        "decimal", "numeric", "real", "number",
    )
    return any(kw in sql_lower for kw in numeric_keywords)


def _is_date_type(sql_lower: str) -> bool:
    """Check if SQL type is date/timestamp."""
    return any(kw in sql_lower for kw in ("date", "time", "timestamp"))


def _is_id_like(stats: ColumnStatistics, threshold: float) -> bool:
    """Check if column looks like an ID (high uniqueness ratio)."""
    if stats.distinct_count <= 0 or stats.non_null_count <= 0:
        return False
    ratio = stats.distinct_count / stats.non_null_count
    return ratio >= threshold


def _looks_code_like(top_k: list) -> bool:
    """Check if top_k values look like codes (short, uppercase, alphanumeric)."""
    if not top_k:
        return False
    sample = top_k[:min(5, len(top_k))]
    code_count = sum(1 for e in sample if _CODE_LIKE_RE.match(str(e.value)))
    return code_count >= len(sample) * 0.6


def _looks_business_key(top_k: list) -> bool:
    """Check if top_k values look like business keys (structured with separators)."""
    if not top_k:
        return False
    sample = top_k[:min(5, len(top_k))]
    key_count = sum(1 for e in sample if _BUSINESS_KEY_RE.match(str(e.value)))
    return key_count >= len(sample) * 0.6
