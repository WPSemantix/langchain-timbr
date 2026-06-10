"""Edge-index wrapper over Plan 1's Ontology.

Builds EdgeMeta objects on demand by:
  1. Calling Ontology.get_concept_metadata(concept) (cached lazily by Plan 1).
  2. Calling Ontology.cardinality_of(concept, rel_name) for each relationship.

Caches the per-concept outbound-edge list and a global (from, rel, to) lookup
incrementally as BFS visits concepts. Never issues SQL of its own.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from ..ontology.graph import Ontology
from .metadata_types import EdgeMeta


class EdgeIndex:
    """Lazy edge view over an Ontology instance.

    Maintains two memoized indexes that grow as concepts are visited:
      - ``outbound_edges[concept]`` -> list[EdgeMeta]
      - ``edge_map[(from, rel, to)]`` -> EdgeMeta
    """

    def __init__(self, ontology: Ontology):
        self._ontology = ontology
        self._outbound: Dict[str, List[EdgeMeta]] = {}
        self._edge_map: Dict[Tuple[str, str, str], EdgeMeta] = {}

    @property
    def ontology(self) -> Ontology:
        return self._ontology

    def outbound_edges(self, concept: str) -> List[EdgeMeta]:
        cached = self._outbound.get(concept)
        if cached is not None:
            return cached
        edges = self._materialize(concept)
        self._outbound[concept] = edges
        for e in edges:
            self._edge_map[(e.from_concept, e.relationship_name, e.to_concept)] = e
        return edges

    def lookup(self, from_concept: str, relationship_name: str, to_concept: str) -> EdgeMeta | None:
        key = (from_concept, relationship_name, to_concept)
        edge = self._edge_map.get(key)
        if edge is not None:
            return edge
        # Materialize the source concept if we haven't visited it yet — the edge
        # might exist but we just haven't enumerated outbound edges yet.
        self.outbound_edges(from_concept)
        return self._edge_map.get(key)

    def _materialize(self, concept: str) -> List[EdgeMeta]:
        try:
            meta = self._ontology.get_concept_metadata(concept)
        except Exception:
            # If the concept can't be described (e.g. logical concept without
            # describe support), surface as empty outbound rather than raise —
            # BFS will simply not expand from it.
            return []
        edges: List[EdgeMeta] = []
        for rel in meta.relationships.values():
            try:
                cardinality = self._ontology.cardinality_of(concept, rel.name)
            except Exception:
                # Conservative fallback if cardinality resolution fails — treat
                # as 1:N (the validator uses cardinality for ranking only).
                cardinality = "N:M" if rel.is_mtm else "1:N"
            edges.append(EdgeMeta(
                from_concept=concept,
                relationship_name=rel.name,
                to_concept=rel.target_concept,
                transitivity=rel.transitivity,
                is_mtm=rel.is_mtm,
                is_inverse=rel.is_inverse,
                cardinality=cardinality,
                description=rel.description,
                is_self_ref=(rel.target_concept == concept),
            ))
        return edges
