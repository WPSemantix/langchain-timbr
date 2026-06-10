"""Timbr Ontology Context — split into two sub-packages:

- ``ontology/`` (Plan 1) — fetches and parses concept metadata via
  ``DESCRIBE CONCEPT`` + ``sys_concept_relationships``. Provides Ontology,
  ConceptMetadata, RelationshipMeta, cardinality derivation, etc.

- ``context_builder/`` (Plan 2) — dynamic SQL-gen metadata-context pipeline.
  BFS subgraph retrieval, Compact DDL serialization, Step 1 LLM filter,
  validator, fallback, rebuild rewriter, transitivity overrides.

This top-level ``__init__`` re-exports both sub-packages' public APIs so
external imports of the form ``from langchain_timbr.ontology_context import X``
continue to work unchanged across the refactor.
"""

from __future__ import annotations

# Plan 1 — ontology sub-package
from .ontology import (
    ConceptMetadata,
    MeasureMeta,
    Ontology,
    PropertyMeta,
    RelationshipAdditionalProperty,
    RelationshipLookupEntry,
    RelationshipMeta,
    TimbrOntologyClient,
    classify,
    derive_cardinality,
    format_relationship_path,
    get_shared_ontology,
    list_relationship_paths,
    parse_describe_output,
    reset_shared_ontologies,
    should_include_in_ddl,
)

# Plan 2 — context_builder sub-package
from .context_builder import (
    DynamicMetadataResult,
    EdgeIndex,
    EdgeMeta,
    MetadataContextConfig,
    PathSegment,
    SelectedPath,
    Step1Output,
    TransitivityOverride,
    ValidationError,
    apply_transitivity_overrides,
    build_filtered_metadata,
    build_relationships_from_paths,
    collect_path_concepts,
    collect_path_relationships,
    config_from_module,
    estimate_subgraph_edge_count,
    filter_columns_for_concepts,
    generate_fallback_paths,
    retrieve_subgraph,
    serialize_compact_ddl,
    should_skip_static_build,
    validate_overrides,
    validate_paths,
)

__all__ = [
    # Plan 1
    "Ontology",
    "TimbrOntologyClient",
    "ConceptMetadata",
    "MeasureMeta",
    "PropertyMeta",
    "RelationshipAdditionalProperty",
    "RelationshipLookupEntry",
    "RelationshipMeta",
    "classify",
    "parse_describe_output",
    "derive_cardinality",
    "should_include_in_ddl",
    "format_relationship_path",
    "list_relationship_paths",
    "get_shared_ontology",
    "reset_shared_ontologies",
    # Plan 2
    "MetadataContextConfig",
    "config_from_module",
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
]
