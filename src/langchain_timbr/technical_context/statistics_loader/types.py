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


@dataclass
class TopKEntry:
    """One entry in a top-K frequency list."""

    value: str
    count: int


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


@dataclass
class ColumnStatistics:
    """Final merged statistics for one column — consumed by Stage B."""

    distinct_count: int = -1  # -1 = not calculated
    non_null_count: int = -1  # -1 = not calculated
    top_k: list[TopKEntry] | None = None
    min_value: Any | None = None
    max_value: Any | None = None
    updated_at: datetime | None = None
    approx_union: bool = False  # True when merged across >1 row
    total_source_rows: int = -1  # view rows (vtimbr) or sum of mapping rows (dtimbr)
    contributing_mappings: list[str] = field(default_factory=list)
