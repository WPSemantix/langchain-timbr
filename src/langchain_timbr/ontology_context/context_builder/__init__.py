"""context_builder — Plan 2: dynamic SQL-gen metadata-context pipeline.

Provides:
  - MetadataContextConfig + config_from_module
  - Types: EdgeMeta, SelectedPath, PathSegment, Step1Output, TransitivityOverride, ValidationError
  - EdgeIndex (lazy wrapper over Ontology)
  - retrieve_subgraph + serialize_compact_ddl + edge-count heuristics
  - validate_paths + validate_overrides
  - generate_fallback_paths (deterministic shortest-path)
  - rebuild helpers: collect_path_concepts, collect_path_relationships,
    filter_columns_for_concepts, build_relationships_from_paths,
    apply_transitivity_overrides
  - build_filtered_metadata (orchestrator) + DynamicMetadataResult
  - prompts/ — Step 1 filter prompt
"""

from __future__ import annotations

from .build_filtered import DynamicMetadataResult, build_filtered_metadata
from .concept_prefilter import (
    PrefilterResult,
    estimate_full_ddl_tokens,
    run_concept_prefilter,
)
from .edge_index import EdgeIndex
from .fallback import generate_fallback_paths
from .metadata_config import MetadataContextConfig, config_from_module, normalize_mode
from .metadata_types import (
    EdgeMeta,
    PathSegment,
    SelectedPath,
    Step1Output,
    TransitivityOverride,
    ValidationError,
)
from .rebuild import (
    apply_transitivity_overrides,
    build_relationships_from_paths,
    collect_path_concepts,
    collect_path_relationships,
    filter_columns_for_concepts,
)
from .subgraph import (
    estimate_subgraph_edge_count,
    retrieve_subgraph,
    serialize_compact_ddl,
    should_skip_static_build,
)
from .validator import validate_overrides, validate_paths

__all__ = [
    "MetadataContextConfig",
    "config_from_module",
    "normalize_mode",
    "EdgeMeta",
    "PathSegment",
    "SelectedPath",
    "Step1Output",
    "TransitivityOverride",
    "ValidationError",
    "EdgeIndex",
    "estimate_subgraph_edge_count",
    "retrieve_subgraph",
    "serialize_compact_ddl",
    "should_skip_static_build",
    "validate_overrides",
    "validate_paths",
    "generate_fallback_paths",
    "apply_transitivity_overrides",
    "build_relationships_from_paths",
    "collect_path_concepts",
    "collect_path_relationships",
    "filter_columns_for_concepts",
    "build_filtered_metadata",
    "DynamicMetadataResult",
    "PrefilterResult",
    "estimate_full_ddl_tokens",
    "run_concept_prefilter",
]
