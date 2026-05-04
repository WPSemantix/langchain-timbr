"""TechnicalContextConfig — configuration for the Technical Context Builder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TechnicalContextConfig:
    """Configuration for the Technical Context Builder.

    Controls mode selection, token budgets, matching thresholds, and trimming behavior.
    """

    enable_technical_context: bool = True
    mode: Literal["include_all", "filter_matched", "auto"] = "auto"
    max_tokens: int = 3000
    safety_ceiling: int = 10000
    max_values_per_column: int = 20
    show_all_under: int = 50

    # Tier classification thresholds
    free_text_distinct_threshold: int = 10000
    id_unique_ratio_threshold: float = 0.95

    # Matching thresholds (two-tier: surface = strong match, sort = weak match)
    fuzzy_threshold_default: int = 88       # surface threshold for categorical_text
    fuzzy_threshold_strict: int = 95        # surface threshold for code_like / business_key_like
    fuzzy_sort_gap: int = 18                # sort threshold = surface - gap (implicit weak match bar)

    # Property filtering (SQL-level: restricts which properties get stats fetched)
    technical_context_properties: list = field(default_factory=list)
    """Whitelist of property names to fetch stats for. Empty = fetch all."""
    exclude_properties: list = field(default_factory=list)
    """Blacklist of property names to exclude from stats fetching."""

    # Trim sequence (per-band, highest-cardinality first within band)
    trim_sequence: tuple = (200, 100, 50, 20, 10, 5)

    def __post_init__(self):
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if self.safety_ceiling <= 0:
            raise ValueError("safety_ceiling must be > 0")
        if self.max_tokens >= self.safety_ceiling:
            raise ValueError("max_tokens must be < safety_ceiling")
        if self.max_values_per_column <= 0:
            raise ValueError("max_values_per_column must be > 0")
        if self.show_all_under < 0:
            raise ValueError("show_all_under must be >= 0")
        if not (0 < self.id_unique_ratio_threshold <= 1.0):
            raise ValueError("id_unique_ratio_threshold must be in (0, 1.0]")
        if not (0 < self.fuzzy_threshold_default <= 100):
            raise ValueError("fuzzy_threshold_default must be in (0, 100]")
        if not (0 < self.fuzzy_threshold_strict <= 100):
            raise ValueError("fuzzy_threshold_strict must be in (0, 100]")
        if self.fuzzy_sort_gap < 0 or self.fuzzy_sort_gap >= self.fuzzy_threshold_default:
            raise ValueError("fuzzy_sort_gap must be in [0, fuzzy_threshold_default)")
        if self.free_text_distinct_threshold <= 0:
            raise ValueError("free_text_distinct_threshold must be > 0")
        if self.mode not in ("include_all", "filter_matched", "auto"):
            raise ValueError(f"mode must be include_all, filter_matched, or auto; got {self.mode}")
        if not self.trim_sequence:
            raise ValueError("trim_sequence must not be empty")
