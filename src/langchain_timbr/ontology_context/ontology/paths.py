"""Relationship path string formatting + enumeration."""

from __future__ import annotations

from typing import Callable

from .models import ConceptMetadata, RelationshipMeta


def format_relationship_path(
    rel: RelationshipMeta,
    *,
    target_property: str | None = None,
    additional_property: str | None = None,
) -> str:
    """Format a single relationship path string.

    Returns one of:
      - `made_order[order]`
      - `made_order[order].customer_name`            (target_property)
      - `has_employee[person]_title`                 (additional_property)
      - `has_acquired[company*2].twitter_username`   (transitive + target_property)
      - `has_acquired[company*2]_acquisition_id`     (transitive + additional_property)
    """
    if target_property and additional_property:
        raise ValueError(
            "format_relationship_path: target_property and additional_property "
            "are mutually exclusive"
        )

    target_token = rel.target_concept
    if rel.transitivity > 1:
        target_token = f"{rel.target_concept}*{rel.transitivity}"
    base = f"{rel.name}[{target_token}]"

    if target_property:
        return f"{base}.{target_property}"
    if additional_property:
        return f"{base}_{additional_property}"
    return base


def list_relationship_paths(
    concept_meta: ConceptMetadata,
    *,
    include_target_properties: bool = True,
    include_additional_properties: bool = True,
    filter_fn: Callable[[RelationshipMeta], bool] | None = None,
) -> list[str]:
    """Enumerate path strings for all relationships of a concept (optionally filtered)."""
    out: list[str] = []
    for rel in concept_meta.relationships.values():
        if filter_fn and not filter_fn(rel):
            continue
        out.append(format_relationship_path(rel))
        if include_target_properties:
            for prop in rel.target_properties:
                out.append(format_relationship_path(rel, target_property=prop))
        if include_additional_properties:
            for ap in rel.additional_properties:
                out.append(format_relationship_path(rel, additional_property=ap.name))
    return out
