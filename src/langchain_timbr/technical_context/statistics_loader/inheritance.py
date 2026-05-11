"""Build a descendants map from ontology inheritance data.

Supports multi-parent inheritance (comma-separated in the inheritance field).
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque

from .types import OntologyConceptRow

logger = logging.getLogger(__name__)


def build_descendants_map(
    concepts: dict[str, OntologyConceptRow],
    max_depth: int = 16,
) -> dict[str, set[str]]:
    """Build a mapping from each concept to all its transitive descendants.

    For each concept C, returns the set of concepts that inherit from C
    (children, grandchildren, etc.). Supports multi-parent inheritance
    via comma-separated inheritance fields.

    Args:
        concepts: Dict of concept_name -> OntologyConceptRow.
        max_depth: Maximum BFS depth for traversal. Truncates with warning on hit.

    Returns:
        Dict mapping concept_name -> set of descendant concept names.
    """
    # Step 1: Build parent → children map
    children_of: dict[str, set[str]] = defaultdict(set)

    for concept_name, row in concepts.items():
        if not row.inheritance:
            continue
        parents = [p.strip() for p in row.inheritance.split(",") if p.strip()]
        for parent in parents:
            children_of[parent].add(concept_name)

    # Step 2: BFS from each concept to compute transitive descendants
    descendants: dict[str, set[str]] = {}

    for concept_name in concepts:
        if concept_name not in children_of:
            descendants[concept_name] = set()
            continue

        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque()

        # Seed with direct children
        for child in children_of[concept_name]:
            queue.append((child, 1))

        depth_cap_hit = False
        while queue:
            current, depth = queue.popleft()
            if current in visited:
                continue
            if depth > max_depth:
                if not depth_cap_hit:
                    logger.warning(
                        "Inheritance depth cap (%d) hit while computing descendants of '%s'",
                        max_depth,
                        concept_name,
                    )
                    depth_cap_hit = True
                continue

            visited.add(current)

            # Add current's children to the queue
            for grandchild in children_of.get(current, set()):
                if grandchild not in visited:
                    queue.append((grandchild, depth + 1))

        descendants[concept_name] = visited

    return descendants
