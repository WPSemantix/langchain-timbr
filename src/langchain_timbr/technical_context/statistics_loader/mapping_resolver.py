"""Resolve the full set of mappings backing a concept.

Combines:
  1. Direct mappings from sys_concept_mappings
  2. Derived mappings from descendant concepts (via inheritance)
  3. Logic-recursive mappings (by parsing logic-concept query strings)

Deduplicates by mapping_name with priority: direct > derived > logic.
"""

from __future__ import annotations

import logging

from .types import (
    OntologyConceptRow,
    ConceptMappingRow,
    MappingRef,
    ConceptMappingSet,
)
from .logic_query import parse_logic_query
from .config import StatisticsLoaderConfig

logger = logging.getLogger(__name__)


def resolve_concept_mappings(
    concept: str,
    ontology: dict[str, OntologyConceptRow],
    mappings_by_concept: dict[str, list[ConceptMappingRow]],
    descendants: dict[str, set[str]],
    config: StatisticsLoaderConfig,
    _visited: set[str] | None = None,
    _depth: int = 0,
) -> ConceptMappingSet:
    """Resolve all mappings backing a concept (direct + derived + logic-recursive).

    Args:
        concept: The concept to resolve mappings for.
        ontology: Full ontology dict from load_ontology_concepts.
        mappings_by_concept: Full mappings dict from load_concept_mappings.
        descendants: Descendants map from build_descendants_map.
        config: Statistics loader configuration.
        _visited: Internal cycle-detection set (do not pass externally).
        _depth: Internal recursion depth (do not pass externally).

    Returns:
        ConceptMappingSet with deduplicated mappings and total_rows.
    """
    if _visited is None:
        _visited = set()

    # Cycle protection
    if concept in _visited:
        logger.warning("Cycle detected in logic-query resolution at concept='%s'", concept)
        return ConceptMappingSet(concept=concept, mappings=[], total_rows=0)

    # Depth cap
    if _depth > config.logic_query_max_depth:
        logger.warning(
            "Logic-query recursion depth cap (%d) exceeded at concept='%s'",
            config.logic_query_max_depth,
            concept,
        )
        return ConceptMappingSet(concept=concept, mappings=[], total_rows=0)

    seen_mapping_names: dict[str, MappingRef] = {}

    # 1. Direct mappings
    direct_rows = mappings_by_concept.get(concept, [])
    for row in direct_rows:
        if row.mapping_name not in seen_mapping_names:
            seen_mapping_names[row.mapping_name] = MappingRef(
                mapping_name=row.mapping_name,
                source_concept=concept,
                number_of_rows=row.number_of_rows,
                via="direct",
            )

    # 2. Derived mappings (from descendants)
    for desc_concept in descendants.get(concept, set()):
        desc_rows = mappings_by_concept.get(desc_concept, [])
        for row in desc_rows:
            if row.mapping_name not in seen_mapping_names:
                seen_mapping_names[row.mapping_name] = MappingRef(
                    mapping_name=row.mapping_name,
                    source_concept=desc_concept,
                    number_of_rows=row.number_of_rows,
                    via="derived",
                )

    # 3. Logic-recursive mappings
    ontology_row = ontology.get(concept)
    if ontology_row and ontology_row.query:
        referenced_concept = parse_logic_query(ontology_row.query)
        if referenced_concept:
            _visited.add(concept)
            logic_set = resolve_concept_mappings(
                concept=referenced_concept,
                ontology=ontology,
                mappings_by_concept=mappings_by_concept,
                descendants=descendants,
                config=config,
                _visited=_visited,
                _depth=_depth + 1,
            )
            for ref in logic_set.mappings:
                if ref.mapping_name not in seen_mapping_names:
                    seen_mapping_names[ref.mapping_name] = MappingRef(
                        mapping_name=ref.mapping_name,
                        source_concept=ref.source_concept,
                        number_of_rows=ref.number_of_rows,
                        via="logic",
                    )
        elif config.on_logic_parse_failure == "raise":
            from .logic_query import LogicQueryParseError

            raise LogicQueryParseError(
                f"Failed to parse logic query for concept '{concept}': {ontology_row.query}"
            )
        else:
            logger.warning(
                "Could not parse logic query for concept '%s': %s",
                concept,
                ontology_row.query,
            )

    # Build final result
    mappings = list(seen_mapping_names.values())
    total_rows = sum(
        m.number_of_rows for m in mappings if m.number_of_rows > 0
    )

    return ConceptMappingSet(
        concept=concept,
        mappings=mappings,
        total_rows=total_rows,
    )
