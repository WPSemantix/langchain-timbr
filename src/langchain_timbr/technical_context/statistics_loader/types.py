"""Data types for the statistics loader pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass
class ColumnPath:
    """Parsed representation of an input column name."""

    raw: str
    hops: list[tuple[str, str]]  # [(relationship_name, target_concept), ...]
    final_property: str
    owning_concept: str  # last concept in hops, or selected_table if no hops


@dataclass
class OntologyConceptRow:
    """One row from timbr.sys_ontology."""

    concept: str
    inheritance: str  # comma-separated parent concepts
    query: str | None  # logic-concept query string, or None


@dataclass
class ConceptMappingRow:
    """One row from timbr.sys_concept_mappings."""

    concept: str
    mapping_name: str
    number_of_rows: int  # -1 if unknown/NULL


@dataclass
class MappingRef:
    """One mapping that contributes stats for a concept."""

    mapping_name: str
    source_concept: str
    number_of_rows: int  # -1 if unknown
    via: Literal["direct", "derived", "logic"]


@dataclass
class ConceptMappingSet:
    """Resolved set of mappings for a single concept."""

    concept: str
    mappings: list[MappingRef]
    total_rows: int  # sum of number_of_rows (NULL/-1 treated as 0)


# What a column's values look like. Derived once when the row is parsed, from
# the values themselves — the declared SQL type finds only 8 of the 148 columns
# that are numeric in practice, because zero-padded identifiers are varchar.
ValueKind = Literal["numeric", "date", "text"]


@dataclass
class TopKEntry:
    """One entry in a top-K frequency list."""

    value: str
    count: int
    # Normalized forms of ``value``, derived once when the row is parsed.
    # Derived data, not statistics: they exist so the matcher never re-derives
    # them per request, and so they live and die with the value they belong to
    # rather than in a second cache with its own cap. If the normalization rules
    # ever change, these are stale until the row is re-fetched — a rule change
    # ships with a process restart, which is what makes that safe.
    norm: str | None = None
    norm_space: str | None = None
    # Leading zeros removed, for numeric columns only. Zero-padded identifiers
    # (`000010`) are the norm in this data — 82 of 148 numeric columns — and
    # nobody asks for them that way, so `10` has to be able to find `000010`.
    norm_stripped: str | None = None


@dataclass
class RawStatsRow:
    """One row from sys_properties_statistics, with stats JSON parsed."""

    property_name: str
    target_name: str  # mapping_name or view_name
    target_type: Literal["mapping", "view"]
    distinct_count: int  # -1 if not calculated
    non_null_count: int  # -1 if not calculated
    top_k: list[TopKEntry] | None
    min_value: Any | None
    max_value: Any | None
    raw_stats: dict | None  # original parsed JSON
    updated_at: datetime | None
    value_kind: ValueKind = "text"
    # For date columns: year -> (months present, how many dates). Replaces
    # listing the dates themselves, which for one year of one column is 184
    # values and across 98 date columns would be ~16x the whole token budget.
    date_years: dict[str, tuple[frozenset[str], int]] | None = None


@dataclass
class ColumnStatistics:
    """Final merged statistics for one column — consumed by Stage B."""

    distinct_count: int = -1  # -1 = not calculated
    non_null_count: int = -1  # -1 = not calculated
    top_k: list[TopKEntry] | None = None
    min_value: Any | None = None
    max_value: Any | None = None
    updated_at: datetime | None = None
    value_kind: ValueKind = "text"
    date_years: dict[str, tuple[frozenset[str], int]] | None = None
    approx_union: bool = False  # True when merged across >1 row
    total_source_rows: int = -1  # view rows (vtimbr) or sum of mapping rows (dtimbr)
    contributing_mappings: list[str] = field(default_factory=list)
