"""MetadataContextConfig — runtime configuration for the dynamic metadata pipeline.

Mirrors the technical_context/config.py pattern. Constructed from `..config`
module defaults with optional per-call overrides.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

from ... import config as _config


def normalize_mode(mode: str | None) -> str:
    """Normalize a metadata-context mode string.

    ``auto`` was retired; it is coerced to ``dynamic`` with a
    ``DeprecationWarning`` for backward compatibility with existing configs.
    Other values pass through unchanged (validation happens in
    ``MetadataContextConfig.__post_init__``).
    """
    resolved = (mode or "static").lower()
    if resolved == "auto":
        warnings.warn(
            "METADATA_CONTEXT_MODE='auto' is deprecated and now treated as "
            "'dynamic'. Set the mode to 'static' or 'dynamic' explicitly.",
            DeprecationWarning,
            stacklevel=2,
        )
        return "dynamic"
    return resolved


@dataclass(frozen=True)
class MetadataContextConfig:
    """Runtime configuration for the dynamic metadata-context pipeline.

    All fields default to the env-backed values from `langchain_timbr.config`.
    Callers may override per-invocation; the orchestrator does not mutate this.
    """
    mode: Literal["static", "dynamic"] = "static"

    # SQL-gen metadata budget — SOFT cap only. There is NO hard ceiling: per
    # "dynamic-over-budget is preferred over static-but-much-larger",
    # oversizing past this cap is logged but emits rebuilt strings as-is.
    metadata_context_max_tokens: int = 12_000

    # DDL prompt budget (filter LLM input). Hard ceiling is log-only —
    # serialize_compact_ddl's cascade emits stage-4 output without failing
    # when the rendered DDL exceeds it.
    metadata_context_filter_max_tokens: int = 6_000
    metadata_context_filter_max_tokens_hard_ceiling: int = 12_000

    # Subgraph traversal — uncapped BFS. DDL-size control lives downstream
    # in the concept pre-filter (see ``max_concept_prefilter_token``).
    # Pre-filter prompt budget (distinct from ``metadata_context_filter_max_tokens``
    # which is the budget for the final SQL-gen-bound DDL).
    max_concept_prefilter_token: int = 2_000
    # Detail-band cap — concept_prefilter also fires when the candidate
    # concept count would meet/exceed this, independent of token size.
    # Overflow concepts are demoted to the menu band, not dropped.
    max_detail_concepts: int = 20
    # Menu-band outer bound — see langchain_timbr.config.max_graph_depth.
    # Callers must satisfy ``graph_depth < max_graph_depth`` (validated at
    # call time by ``build_filtered_metadata`` — clamped, not raised).
    max_graph_depth: int = 5
    # Max number of Step 1 LLM retry attempts when validation finds errors.
    # 0 disables retry. Default 2 — gives the planner two chances to fix
    # mistakes with injected reason codes. Total Step 1 LLM calls per
    # chain.invoke() is at most 1 + metadata_context_dynamic_retry. When
    # exhausted, the dynamic pipeline returns empty and the wiring layer
    # falls back to static metadata strings (no internal BFS / shortest-
    # path / pre-filter rescue — see build_filtered.py).
    metadata_context_dynamic_retry: int = 2
    static_attempt_edge_threshold: int = 100

    # Feature toggles
    include_logic_concepts: bool = False
    enable_fanout_hints: bool = True

    def __post_init__(self):
        if self.mode not in ("static", "dynamic"):
            raise ValueError(
                f"mode must be 'static' or 'dynamic'; got {self.mode!r}"
            )
        if self.metadata_context_max_tokens <= 0:
            raise ValueError("metadata_context_max_tokens must be > 0")
        if self.metadata_context_filter_max_tokens <= 0:
            raise ValueError("metadata_context_filter_max_tokens must be > 0")
        if (
            self.metadata_context_filter_max_tokens_hard_ceiling
            < self.metadata_context_filter_max_tokens
        ):
            raise ValueError(
                "metadata_context_filter_max_tokens_hard_ceiling must be "
                ">= metadata_context_filter_max_tokens"
            )
        if self.max_concept_prefilter_token <= 0:
            raise ValueError("max_concept_prefilter_token must be > 0")
        if self.max_detail_concepts <= 0:
            raise ValueError("max_detail_concepts must be > 0")
        if self.max_graph_depth <= 0:
            raise ValueError("max_graph_depth must be > 0")
        if self.static_attempt_edge_threshold < 0:
            raise ValueError("static_attempt_edge_threshold must be >= 0")
        if self.metadata_context_dynamic_retry < 0:
            raise ValueError("metadata_context_dynamic_retry must be >= 0")


def config_from_module(**overrides) -> MetadataContextConfig:
    """Build a MetadataContextConfig from the module-level config, with optional overrides.

    Used by the SQL-gen wiring so per-chain kwargs flow into a single config object.
    Unknown override keys are ignored (defensive — kwargs may include unrelated fields).
    """
    base = dict(
        mode=normalize_mode(getattr(_config, "metadata_context_mode", "static")),
        metadata_context_max_tokens=getattr(_config, "metadata_context_max_tokens", 12_000),
        metadata_context_filter_max_tokens=getattr(_config, "metadata_context_filter_max_tokens", 6_000),
        metadata_context_filter_max_tokens_hard_ceiling=getattr(
            _config, "metadata_context_filter_max_tokens_hard_ceiling", 12_000,
        ),
        metadata_context_dynamic_retry=getattr(_config, "metadata_context_dynamic_retry", 2),
        static_attempt_edge_threshold=getattr(_config, "static_attempt_edge_threshold", 100),
        include_logic_concepts=getattr(_config, "include_logic_concepts", False),
        max_concept_prefilter_token=getattr(_config, "max_concept_prefilter_token", 2_000),
        max_detail_concepts=getattr(_config, "max_detail_concepts", 20),
        max_graph_depth=getattr(_config, "max_graph_depth", 5),
    )
    valid_keys = set(base.keys()) | {"enable_fanout_hints"}
    for k, v in overrides.items():
        if k in valid_keys and v is not None:
            base[k] = v
    base["mode"] = normalize_mode(base["mode"])
    return MetadataContextConfig(**base)
