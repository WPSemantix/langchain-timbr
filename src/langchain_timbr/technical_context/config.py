"""TechnicalContextConfig — configuration for the Technical Context Builder."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class TechnicalContextConfig:
    """Configuration for the Technical Context Builder.

    Controls mode selection, token budgets, matching thresholds, and trimming behavior.
    """
    mode: Literal["include_all", "filter_matched", "auto"] = "auto"
    max_tokens: int = 3000
    """Soft token budget for the annotations. Values above ``safety_ceiling`` are
    clamped to it (with a warning) rather than rejected — an oversized budget must
    never turn the feature off."""
    safety_ceiling: int = 20000
    """Hard cap on the annotation token total. Also the largest accepted ``max_tokens``."""
    max_values_per_column: int = 20
    show_all_under: int = 50

    # Tier classification thresholds
    free_text_distinct_threshold: int = 10000
    id_unique_ratio_threshold: float = 0.95
    # Absolute distinct_count at or below which a string column is classified
    # CATEGORICAL_ENUM and gets its full value domain emitted. Wins over the
    # ID / FREE_TEXT / CODE_LIKE classifications — the unique-ratio signal is
    # grain-dependent and unreliable for relationship-joined dimension label
    # columns (distinct ≈ non_null ≈ 1.0 by construction).
    categorical_enum_max_distinct: int = 500

    # Matching thresholds (two-tier: surface = strong match, sort = weak match)
    fuzzy_threshold_default: int = 88       # surface threshold for categorical_text
    fuzzy_threshold_strict: int = 95        # surface threshold for code_like / business_key_like
    fuzzy_sort_gap: int = 18                # sort threshold = surface - gap (implicit weak match bar)

    # Property filtering (SQL-level: restricts which properties get stats fetched)
    technical_context_properties: list = field(default_factory=list)
    """Whitelist of property names to fetch stats for. Empty = fetch all."""
    exclude_properties: list = field(default_factory=list)
    """Blacklist of property names to exclude from stats fetching."""

    def __post_init__(self):
        if self.max_tokens <= 0:
            # Callers disable the feature by not asking for it (a non-positive budget is
            # gated as "off" before this constructor is reached), so reaching here is a
            # programming error rather than a misconfiguration.
            raise ValueError("max_tokens must be > 0")
        if self.safety_ceiling <= 0:
            raise ValueError("safety_ceiling must be > 0")
        if self.max_tokens > self.safety_ceiling:
            # A budget past the cap is a misconfiguration, not a reason to drop the whole
            # technical context: clamp it and let the trimmer fit the content to the cap.
            logger.warning(
                "technical context max_tokens=%d exceeds the hard cap of %d; clamping to the "
                "cap. The context is trimmed to fit it instead of being dropped.",
                self.max_tokens, self.safety_ceiling,
            )
            self.max_tokens = self.safety_ceiling
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
        if self.categorical_enum_max_distinct <= 0:
            raise ValueError("categorical_enum_max_distinct must be > 0")
        if self.categorical_enum_max_distinct >= self.free_text_distinct_threshold:
            raise ValueError(
                "categorical_enum_max_distinct must be < free_text_distinct_threshold"
            )
        if self.mode not in ("include_all", "filter_matched", "auto"):
            raise ValueError(f"mode must be include_all, filter_matched, or auto; got {self.mode}")
