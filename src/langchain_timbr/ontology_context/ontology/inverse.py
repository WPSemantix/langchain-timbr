"""DDL-layer inverse-filter rule.

Plan 1 fetches every relationship row including is_inverse=1. Consumers
(e.g. Plan 2's DDL serializer) call should_include_in_ddl to drop bounce-back
inverses while keeping self-ref both-directions.
"""

from __future__ import annotations

from .models import RelationshipMeta


def should_include_in_ddl(
    rel: RelationshipMeta,
    *,
    current_concept: str,
    previous_hop_concept: str | None,
) -> bool:
    """Return True if this relationship should appear in the DDL for current_concept.

    - Anchor (no previous hop): include everything.
    - Self-ref (rel.target_concept == current_concept): include (both directions
      reach different instances of the same concept type).
    - Bounce-back inverse (is_inverse=True AND target == previous_hop_concept AND
      not self-ref): drop.
    - All other cases: include.
    """
    if previous_hop_concept is None:
        return True
    if rel.target_concept == current_concept:
        return True
    if rel.is_inverse and rel.target_concept == previous_hop_concept:
        return False
    return True
