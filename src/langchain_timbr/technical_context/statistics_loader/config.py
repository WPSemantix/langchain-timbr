"""Configuration for the statistics loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class StatisticsLoaderConfig:
    """Configuration for load_column_statistics.

    Example:
        >>> config = StatisticsLoaderConfig(in_clause_chunk_size=200)
        >>> stats = load_column_statistics(..., config=config)
    """

    in_clause_chunk_size: int = 1000
    """Maximum number of mapping names per IN-clause chunk."""

    logic_query_max_depth: int = 10
    """Safety cap for logic→logic recursion in mapping resolution."""

    inheritance_max_depth: int = 20
    """Safety cap for inheritance chain traversal."""

    on_missing_stats: Literal["sentinel", "raise"] = "sentinel"
    """Behavior when statistics are missing for a column."""

    on_logic_parse_failure: Literal["skip", "raise"] = "skip"
    """Behavior when a logic-concept query string cannot be parsed."""

    log_level: str = "INFO"
    """Logging level for statistics loader operations."""

    # --- Caching ---
    cache_enabled: bool = True
    """Enable in-memory caching of fetched RawStatsRow lists."""

    cache_validation_interval_seconds: int = 600
    """Per-ontology TTL gate: batch validation query at most every N seconds."""

    cache_idle_eviction_seconds: int = 3600
    """Evict entries unused for longer than this (seconds). Checked on every request."""

    cache_max_total_mb: int = 500
    """Maximum total cache size in MB. LRU eviction when exceeded."""

    # --- Property filtering (SQL-level WHERE clauses) ---
    include_properties: list = field(default_factory=list)
    """Whitelist: only fetch stats for these property names. Empty = fetch all."""

    exclude_properties: list = field(default_factory=list)
    """Blacklist: exclude these property names from stats queries."""

    def __post_init__(self) -> None:
        if self.in_clause_chunk_size <= 0:
            raise ValueError(f"in_clause_chunk_size must be > 0, got {self.in_clause_chunk_size}")
        if self.logic_query_max_depth <= 0:
            raise ValueError(f"logic_query_max_depth must be > 0, got {self.logic_query_max_depth}")
        if self.inheritance_max_depth <= 0:
            raise ValueError(f"inheritance_max_depth must be > 0, got {self.inheritance_max_depth}")
        if self.on_missing_stats not in ("sentinel", "raise"):
            raise ValueError(f"on_missing_stats must be 'sentinel' or 'raise', got '{self.on_missing_stats}'")
        if self.on_logic_parse_failure not in ("skip", "raise"):
            raise ValueError(f"on_logic_parse_failure must be 'skip' or 'raise', got '{self.on_logic_parse_failure}'")
