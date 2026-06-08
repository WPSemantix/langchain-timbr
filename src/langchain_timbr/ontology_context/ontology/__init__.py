"""ontology — Plan 1: structured wrapper over DESCRIBE CONCEPT + sys_concept_relationships.

Provides:
  - Ontology class with version-keyed lazy cache
  - TimbrOntologyClient (run_query wrapper)
  - Frozen dataclasses (PropertyMeta, MeasureMeta, RelationshipMeta, ConceptMetadata, ...)
  - Parser for DESCRIBE CONCEPT column-name encoding (incl. ~ inverse marker)
  - derive_cardinality (PK-match logic)
  - should_include_in_ddl (inverse-bounce-back filter)
  - format_relationship_path / list_relationship_paths
  - get_shared_ontology (process-wide Ontology factory)
"""

from __future__ import annotations

from .cardinality import derive_cardinality
from .client import TimbrOntologyClient
from .graph import Ontology
from .inverse import should_include_in_ddl
from .models import (
    ConceptMetadata,
    MeasureMeta,
    PropertyMeta,
    RelationshipAdditionalProperty,
    RelationshipLookupEntry,
    RelationshipMeta,
)
from .parser import classify, parse_describe_output
from .paths import format_relationship_path, list_relationship_paths
from .shared import get_shared_ontology, reset_shared_ontologies

__all__ = [
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
]
