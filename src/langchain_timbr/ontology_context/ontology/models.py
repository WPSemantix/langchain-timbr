"""Frozen dataclasses representing parsed Timbr ontology metadata.

These are value objects produced by the parser and cached by the Ontology graph.
All sequence fields use tuple (not list) so dataclasses stay immutable end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PropertyMeta:
    name: str
    data_type: str
    description: str | None
    is_inherited: bool
    is_pk: bool
    is_fk: bool


@dataclass(frozen=True)
class MeasureMeta:
    name: str
    data_type: str
    description: str | None
    is_inherited: bool
    scoped_to_relationship: str | None


@dataclass(frozen=True)
class RelationshipAdditionalProperty:
    name: str
    data_type: str


@dataclass(frozen=True)
class RelationshipMeta:
    name: str
    target_concept: str
    transitivity: int
    is_mtm: bool
    is_inverse: bool
    description: str | None
    source_join_keys: tuple[str, ...]
    target_join_keys: tuple[str, ...]
    additional_properties: tuple[RelationshipAdditionalProperty, ...]
    target_properties: tuple[str, ...]


@dataclass(frozen=True)
class RelationshipLookupEntry:
    is_mtm: bool
    is_inverse: bool
    description: str | None
    source_join_keys: tuple[str, ...]
    target_join_keys: tuple[str, ...]


@dataclass(frozen=True)
class ConceptMetadata:
    name: str
    description: str | None
    properties: dict[str, PropertyMeta]
    measures: dict[str, MeasureMeta]
    relationships: dict[str, RelationshipMeta]
    # Parent chain from this concept up to `thing`, e.g. ('organization', 'thing')
    # for `company`. Populated from `sys_ontology.inheritance` by the graph
    # layer. Empty tuple when no inheritance info is available.
    inheritance_chain: tuple[str, ...] = ()
