"""Data types for the Technical Context Builder."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class SemanticType(Enum):
    """Classification of a column's semantic meaning based on its statistics."""

    BOOLEAN = "boolean"
    ID = "id"
    NUMERIC = "numeric"
    DATE = "date"
    FREE_TEXT = "free_text"
    CODE_LIKE = "code_like"
    BUSINESS_KEY_LIKE = "business_key_like"
    CATEGORICAL_TEXT = "categorical_text"


@dataclass
class ColumnRef:
    """Column reference with metadata for technical context processing."""

    name: str  # full column name as used in SQL (e.g., "orders[order].total")
    sql_type: str  # SQL data type (e.g., "decimal(18,2)")
    ontology_distance: int  # computed from dot count in name
    priority_band: int  # 1-5, derived from distance + match status
    semantic_type: SemanticType | None = None


@dataclass
class MatchResult:
    """Result of matching a prompt token against a column's known values."""

    column_name: str
    matched_value: str  # the value from statistics that matched
    score: int  # 0-100
    match_type: Literal["exact", "substring", "fuzzy"]
    candidate: str  # what from the prompt matched


@dataclass
class ColumnPayload:
    """Structured per-column annotation payload (pre-formatting, pre-trimming).

    format_hint controls both rendering and trim eligibility:
    - "top_k": trimmable — known values list with variable K
    - "all": show_all_under tier — protected, all values shown
    - "min_max": numeric/date range — protected
    - "name_only": ID with no meaningful annotation — protected (minimal)
    - "count_only": free_text — just distinct count — protected (minimal)
    - "boolean": boolean values — protected (minimal)
    """

    format_hint: Literal["top_k", "all", "min_max", "name_only", "count_only", "boolean"]
    values: list[str] = field(default_factory=list)  # matched first, then by frequency
    matched_values: set[str] = field(default_factory=set)  # subset of values from matching
    distinct_count: int = -1
    min_value: Any = None
    max_value: Any = None
    range_label: str = "value range"  # "value range" or "date range"


@dataclass
class TechnicalContextResult:
    """Output of build_technical_context — ready for injection into prompt."""

    column_annotations: dict[str, str]  # col_name -> annotation string
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.column_annotations
