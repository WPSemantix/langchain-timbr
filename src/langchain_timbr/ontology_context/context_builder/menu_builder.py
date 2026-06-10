"""Menu builder — depth-band menu computation for the dynamic pipeline.

Two-band split of the BFS-reachable concept set:

  - **detail band** (hop ≤ detail_depth): rendered with full Compact-DDL
    detail. The planner constructs paths from these concepts.
  - **menu band** (detail_depth < hop ≤ max_graph_depth): rendered as names-only
    in the ``## REACHABLE`` band. The planner can request any of these be
    promoted to the detail band via the ``expand_to`` action.

Menu entries carry their ``source`` tag so the orchestrator can route
``expand_to`` requests correctly:

  - ``"depth_band"`` → BFS visited the concept at the max_graph_depth frontier
    but did NOT enumerate outbound edges from it (BFS stopped). Promoting
    requires materializing those outbound edges via
    ``materialize_concept_outbound_edges`` so the serializer can render the
    concept's ``rels:`` block and the validator can verify paths starting
    from it.
  - ``"prefilter_overflow"`` → the concept_prefilter LLM demoted this
    concept out of the detail band on size grounds (token or count
    overflow). Its outbound edges WERE enumerated by the initial BFS
    (it's at hop ≤ detail_depth structurally). Promoting just requires
    pinning it back into the detail band — no edge materialization
    needed.

Recursive / transitive (``*N``) relationships count as ONE hop in BFS —
the depth-N walk happens inside a single edge traversal at SQL execution
time, not at planning time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence, Set, Tuple

from .edge_index import EdgeIndex
from .metadata_types import EdgeMeta


MenuSource = Literal["depth_band", "prefilter_overflow"]


@dataclass(frozen=True)
class MenuEntry:
    """A demoted concept in the ``## REACHABLE`` band.

    The ``source`` tag drives ``expand_to`` routing — the orchestrator
    handles the two sources differently (see module docstring).
    """
    concept: str
    hop: int
    source: MenuSource


def build_hop_map(
    anchor: str,
    edge_index: EdgeIndex,
    max_graph_depth: int,
) -> Dict[str, int]:
    """BFS from ``anchor`` out to ``max_graph_depth``, recording first-seen hop
    per concept.

    Returns: ``{concept_name: hop_distance}`` for all concepts visited
    within ``max_graph_depth`` hops. The anchor itself maps to 0.

    Notes:
      - Recursive / transitive (``*N``) relationships count as ONE hop.
        ``EdgeMeta.transitivity`` is intentionally ignored here — depth is
        walked at execution time, not at BFS time.
      - BFS does NOT call ``outbound_edges`` for concepts discovered at
        the max_graph_depth frontier (their outbound rels remain unmaterialized
        until ``expand_to`` promotes them). This is the correct behavior
        for the menu/detail split — see ``materialize_concept_outbound_edges``.
    """
    if max_graph_depth <= 0:
        return {anchor: 0}
    hop_map: Dict[str, int] = {anchor: 0}
    frontier: List[str] = [anchor]
    for hop in range(1, max_graph_depth + 1):
        next_frontier: List[str] = []
        for concept in frontier:
            for edge in edge_index.outbound_edges(concept):
                target = edge.to_concept
                if target not in hop_map:
                    hop_map[target] = hop
                    next_frontier.append(target)
        if not next_frontier:
            break
        frontier = next_frontier
    return hop_map


def split_bands(
    hop_map: Dict[str, int],
    detail_depth: int,
    max_graph_depth: int,
) -> Tuple[List[str], List[MenuEntry]]:
    """Split the BFS-discovered concepts into detail and depth-band menu.

    Args:
        hop_map: output of ``build_hop_map``.
        detail_depth: max hop included in the detail band (typically the
            chain's ``graph_depth``).
        max_graph_depth: outer reachability bound (config knob).

    Returns:
        ``(detail_concepts, menu_entries)`` where:

          - ``detail_concepts`` lists concepts at hop ≤ ``detail_depth``,
            in BFS-discovery order.
          - ``menu_entries`` lists concepts at ``detail_depth < hop ≤
            max_graph_depth``, tagged ``source="depth_band"``, sorted by
            ``(hop, name)`` so the rendered ``## REACHABLE`` band shows
            shortest-hop first, lex within hop.

    Out of scope (per the plan):
      - Multi-factor menu ranking (BM25, business-importance). Today's
        ordering is shortest-hop + lex.
      - Concepts at hop > ``max_graph_depth`` are simply not represented (the
        planner must ``reanchor`` to reach them, not ``expand_to``).
    """
    detail: List[str] = []
    menu_entries: List[MenuEntry] = []
    for concept, hop in hop_map.items():
        if hop <= detail_depth:
            detail.append(concept)
        elif hop <= max_graph_depth:
            menu_entries.append(
                MenuEntry(concept=concept, hop=hop, source="depth_band")
            )
    menu_entries.sort(key=lambda e: (e.hop, e.concept))
    return detail, menu_entries


def materialize_concept_outbound_edges(
    concept: str,
    edge_index: EdgeIndex,
    edges: List[EdgeMeta],
    edge_seen: Set[Tuple[str, str, str]],
) -> List[EdgeMeta]:
    """DEPRECATED — kept only for backward compatibility with callers that
    may import this symbol.

    Under the current ``expand_to`` semantics (Fix 4 in
    ``proactive-menu-fixes.md``), ``expand_to`` does NOT promote a target
    concept to full detail. Instead, it reveals the connecting
    relationship chain via minimal blocks (handled in
    ``build_filtered._compute_expand_minimal_concepts`` + ``subgraph._render``).
    Outbound-edge materialization is no longer required because the
    minimal block carries only inbound connecting rels, not the target's
    outbound rels.

    Legacy contract (still works for ad-hoc callers): materialize outbound
    edges from a promoted concept by querying ``edge_index``, deduping
    against ``edge_seen``, appending newly-found edges to ``edges``, and
    returning the new edges.
    """
    added: List[EdgeMeta] = []
    for edge in edge_index.outbound_edges(concept):
        key = (edge.from_concept, edge.relationship_name, edge.to_concept)
        if key in edge_seen:
            continue
        edge_seen.add(key)
        edges.append(edge)
        added.append(edge)
    return added


__all__ = [
    "MenuEntry",
    "MenuSource",
    "build_hop_map",
    "materialize_concept_outbound_edges",
    "split_bands",
]
