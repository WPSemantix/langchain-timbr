"""Deterministic fallback — Tier 3/4 path enumeration when the LLM filter
produces no usable paths.

Enumerates ALL simple paths from the anchor that terminate at a concept in
``selected_concepts``, bounded by:

  - ``length_cap`` — maximum number of segments per path (typically the chain's
    ``graph_depth``; same per-call hop ceiling Step 0 retrieval uses).
  - No node revisits within a single path — prevents cycles and combinatorial
    blowup on self-referential / hub-heavy concepts.
  - Loose interior: paths may pass through ANY concept in the subgraph en
    route to a selected target. This is intentional — the LLM's
    ``selected_concepts`` are the endpoints it thinks matter; intermediate
    waypoints ("bridges" like a `customer → order → product` order concept)
    are commonly omitted from the selection. Strict-interior mode would
    silently kill recall on exactly the multi-hop questions this fallback
    is supposed to rescue.
  - Subgraph-bounded: edges come from ``EdgeIndex.outbound_edges``, which only
    sees concepts present in the BFS-retrieved subgraph. So "loose" is bounded
    by the subgraph itself, not unbounded.

Output is deduped on full path identity (the ordered ``(from, rel, to)``
segment tuple) — two paths reaching the same target via different relationship
sequences are distinct and both kept.

Safety cap (``_SAFETY_PATH_LIMIT``, default 1000) is a circuit breaker for
pathological dense subgraphs, NOT an output-shaping knob. If it trips, we log
loudly and return what we've collected — the caller still gets a usable result
but emits a stat so audit pipelines can flag the case.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Set, Tuple

from .edge_index import EdgeIndex
from .metadata_types import EdgeMeta, PathSegment, SelectedPath

logger = logging.getLogger(__name__)


# Circuit breaker. Generous enough that a healthy enumeration never trips it;
# tight enough that a runaway DFS can't OOM the process. When tripped, the
# enumeration aborts early and returns the paths already collected — the
# caller observes ``stats["fallback_safety_cap_tripped"] = True`` via the
# orchestrator (see build_filtered.py).
_SAFETY_PATH_LIMIT = 1000


def generate_fallback_paths(
    anchor: str,
    selected_concepts: Iterable[str],
    edge_index: EdgeIndex,
    *,
    length_cap: int,
    safety_cap: int = _SAFETY_PATH_LIMIT,
) -> List[SelectedPath]:
    """Enumerate all simple paths from anchor to each selected concept.

    Args:
        anchor: starting concept (SQL FROM target).
        selected_concepts: LLM-chosen endpoints the paths must terminate at.
            Bridge concepts NOT in this set are allowed on the path interior.
        edge_index: provides outbound edges per concept.
        length_cap: max segments per path. Typically the chain's graph_depth.
        safety_cap: max total paths to collect before aborting enumeration.
            Circuit breaker against pathological dense subgraphs.

    Returns:
        List of ``SelectedPath`` objects, deduped on full segment-identity
        tuple, in the order discovered by DFS. No ranking, no top-K.
    """
    targets: Set[str] = {c for c in selected_concepts if c and c != anchor}
    if not targets or length_cap <= 0:
        return []

    candidates: List[SelectedPath] = []
    seen_keys: Set[Tuple[Tuple[str, str, str], ...]] = set()
    counter = 0

    def _walk(current: str, visited: Set[str], segments: List[EdgeMeta]) -> bool:
        """DFS one step from ``current``. Returns True to keep going, False to
        signal the safety cap was tripped and the caller should abort."""
        nonlocal counter
        if len(segments) >= length_cap:
            return True
        for edge in edge_index.outbound_edges(current):
            nxt = edge.to_concept
            if nxt in visited:
                continue  # no node revisits
            new_segments = segments + [edge]
            # Record as a candidate if this step lands on a selected target.
            if nxt in targets:
                key: Tuple[Tuple[str, str, str], ...] = tuple(
                    (e.from_concept, e.relationship_name, e.to_concept)
                    for e in new_segments
                )
                if key not in seen_keys:
                    seen_keys.add(key)
                    candidates.append(_to_selected_path(new_segments, counter))
                    counter += 1
                    if len(candidates) >= safety_cap:
                        logger.error(
                            "Fallback path enumeration hit safety cap %d "
                            "(anchor=%r, targets=%s) — aborting DFS. This "
                            "usually signals a pathological dense subgraph or "
                            "an over-broad selected_concepts list.",
                            safety_cap, anchor, sorted(targets),
                        )
                        return False
            # Continue DFS through nxt regardless of whether it's a target.
            # Bridge concepts (not in targets) are allowed on the interior.
            if not _walk(nxt, visited | {nxt}, new_segments):
                return False
        return True

    _walk(anchor, {anchor}, [])
    return candidates


def _to_selected_path(edges: List[EdgeMeta], idx: int) -> SelectedPath:
    """Materialize a SelectedPath from a list of EdgeMeta segments."""
    segments = [
        PathSegment(
            **{
                "from": e.from_concept,
                "rel": e.relationship_name,
                "to": e.to_concept,
            }
        )
        for e in edges
    ]
    target = edges[-1].to_concept
    source = edges[0].from_concept
    return SelectedPath(
        path_id=f"F{idx + 1}",
        purpose=f"fallback path from {source} to {target}",
        segments=segments,
        is_recursive=False,
    )


__all__ = ["generate_fallback_paths"]
