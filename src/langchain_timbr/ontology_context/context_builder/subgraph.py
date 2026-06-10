"""Step 0 — anchor-rooted subgraph retrieval + Compact DDL serialization.

Two responsibilities split into pure functions:
  - ``estimate_subgraph_edge_count`` / ``should_skip_static_build`` —
    fast-path edge-count heuristic used by the auto-mode trigger.
  - ``retrieve_subgraph`` — uncapped BFS from an anchor, returning the visited
    concepts and ``hop_predecessors`` (used by the DDL inverse filter). DDL-size
    control happens downstream: the concept pre-filter (when estimated DDL
    exceeds the prompt budget) narrows the concept set before serialization.
  - ``serialize_compact_ddl`` — 5-stage compression cascade fitting the DDL
    output to ``metadata_context_filter_max_tokens``.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from ..ontology.graph import Ontology
from ..ontology.inverse import should_include_in_ddl
from .edge_index import EdgeIndex
from .metadata_config import MetadataContextConfig
from .metadata_types import EdgeMeta


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

def estimate_subgraph_edge_count(
    anchor: str,
    edge_index: EdgeIndex,
    max_hop: int,
) -> int:
    """Cheap BFS edge count from anchor up to ``max_hop`` hops. No serialization."""
    if max_hop <= 0:
        return 0
    visited: Set[str] = {anchor}
    edge_count = 0
    frontier: Set[str] = {anchor}
    for _ in range(max_hop):
        next_frontier: Set[str] = set()
        for concept in frontier:
            for edge in edge_index.outbound_edges(concept):
                edge_count += 1
                if edge.to_concept not in visited:
                    next_frontier.add(edge.to_concept)
                    visited.add(edge.to_concept)
        frontier = next_frontier
        if not frontier:
            break
    return edge_count


def should_skip_static_build(
    graph_depth: int,
    anchor: str,
    edge_index: EdgeIndex,
    config: MetadataContextConfig,
) -> bool:
    """Fast-path heuristic: in ``auto`` mode, skip the static build when both
    ``graph_depth >= 3`` AND the anchor's BFS edge count exceeds the threshold."""
    if graph_depth < 3:
        return False
    count = estimate_subgraph_edge_count(anchor, edge_index, graph_depth)
    if count <= config.static_attempt_edge_threshold:
        return False
    return True


# ---------------------------------------------------------------------------
# BFS retrieval
# ---------------------------------------------------------------------------

def retrieve_subgraph(
    anchor: str,
    edge_index: EdgeIndex,
    config: MetadataContextConfig,
    *,
    max_hop: int | None = None,
) -> Tuple[List[str], Dict[str, str | None], List[EdgeMeta]]:
    """Uncapped BFS from ``anchor``; returns (concepts, predecessors, edges).

    No per-node or total-edge caps — every outbound edge from every visited
    concept up to ``max_hop`` is retained. DDL-size control is delegated to
    the downstream concept pre-filter, which trims the concept set when the
    estimated DDL would exceed the prompt budget.

    Returns:
        concepts: list of concept names visited (anchor first, then BFS order).
        predecessors: dict mapping each concept to the concept BFS first reached it from
            (anchor -> None). Self-ref concepts keep their first non-self predecessor.
        edges: list of EdgeMeta objects retained in the subgraph (deduped).
    """
    hop_ceiling = max_hop if max_hop is not None else 3
    if hop_ceiling <= 0:
        return [anchor], {anchor: None}, []

    visited: List[str] = [anchor]
    visited_set: Set[str] = {anchor}
    predecessors: Dict[str, str | None] = {anchor: None}
    retained_edges: List[EdgeMeta] = []
    edge_seen: Set[Tuple[str, str, str]] = set()

    frontier: List[str] = [anchor]

    for _ in range(hop_ceiling):
        next_frontier: List[str] = []
        for concept in frontier:
            for edge in edge_index.outbound_edges(concept):
                key = (edge.from_concept, edge.relationship_name, edge.to_concept)
                if key in edge_seen:
                    continue
                edge_seen.add(key)
                retained_edges.append(edge)
                target = edge.to_concept
                if target not in visited_set:
                    visited_set.add(target)
                    visited.append(target)
                    predecessors[target] = concept if concept != target else None
                    next_frontier.append(target)
        if not next_frontier:
            break
        frontier = next_frontier

    return visited, predecessors, retained_edges


# ---------------------------------------------------------------------------
# Compact DDL serialization + compression cascade
# ---------------------------------------------------------------------------

def serialize_compact_ddl(
    concepts: List[str],
    edges: List[EdgeMeta],
    ontology: Ontology,
    predecessors: Dict[str, str | None],
    config: MetadataContextConfig,
    *,
    menu_concepts: List[str] | None = None,
    expand_minimal_concepts: List[str] | None = None,
) -> Tuple[str, int]:
    """Serialize the subgraph as the concept-centric Compact DDL.

    Output layout:
        ## CONCEPTS

        ### <concept> [anchor]
        description: <concept_description>
        props:
          str: ...
          num: ...
        measures: ...
        rels:
          -[<rel>, <card>]-> <target>  # <rel_description>
          -[<rel>, <card>]-> <self>    # recursive | <rel_description>
        incoming:
          - <source_concept>.<rel_name>

        ### <expanded_concept> [hop=N]   <-- minimal block, used for expand_to
        description: <concept_description>
        rels:
          -[<rel>, <card>]-> <target>   # <rel_description>
          -[<rel>, <card>]-> <other_target>   # may point at a ## REACHABLE name

        ## REACHABLE: name1, name2, name3
            (only emitted when ``menu_concepts`` is non-empty)

    Self-referential relationships are inlined within each concept's ``rels:``
    block, marked with a ``# recursive`` annotation. The ``incoming:`` block
    lists relationships from OTHER concepts in the rendered subgraph that
    point TO this concept; self-refs are intentionally NOT duplicated there.

    The ``## REACHABLE`` band lists names-only of concepts that were demoted
    out of the detail band by the concept pre-filter (token or count
    overflow) or that sit at hops beyond ``detail_depth`` in the depth-band
    menu. The path-selection LLM can request any of these be revealed via
    ``expand_to(targets)``.

    ``expand_to`` semantics — when ``expand_minimal_concepts`` is supplied:
      - Each named concept gets a MINIMAL block in ``## CONCEPTS``:
        heading + ``description:`` line + ``rels:`` block listing the
        concept's COMPLETE outgoing edges (bounded by the BFS-truncated
        edge set, i.e. targets within ``max_graph_depth`` of the anchor). No
        ``props:``, ``measures:``, or ``incoming:`` — only the outgoing
        rels: opens up. Edges may point at concepts still in
        ``## REACHABLE``; those target names stay in REACHABLE (visibility
        does NOT promote them — only the named ``expand_to`` targets and
        the intermediate concepts on the predecessor chain to them earn
        minimal blocks).
      - Each named expand target is removed from ``## REACHABLE`` on render
        — it's earned a minimal block.
      - Detail-band concepts on the predecessor chain keep their full DDL.

    Universal descriptions:
      - Every concept block (full OR minimal) emits a ``description: ...``
        line when ``ontology.get_concept_metadata(c).description`` is
        non-empty.
      - Every ``rels:`` line carries an inline ``# <description>`` annotation
        from ``EdgeMeta.description`` when present (combined with
        ``recursive`` via ``|`` when both apply).

    Fitted to ``metadata_context_filter_max_tokens`` via a 4-stage cascade. Returns
    ``(text, stage)`` where ``stage`` is the cascade level (1=full, 4=most
    aggressive). If even stage 4 exceeds the budget, emits stage-4 output and
    leaves it to the caller to observe via logs.
    """
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")

        def _count(text: str) -> int:
            return len(encoding.encode(text))
    except Exception:
        def _count(text: str) -> int:
            return max(1, len(text) // 4)

    # Build BFS hop distance from predecessors (anchor has predecessor=None).
    # Hop map covers every concept that may render — detail set plus expand-
    # minimal set — so heading labels resolve uniformly.
    expand_minimal_list = list(expand_minimal_concepts or [])
    hop_inputs = list(concepts) + [
        c for c in expand_minimal_list if c not in concepts
    ]
    hop_by_concept = _compute_hop_distances(hop_inputs, predecessors)
    # Apply inverse-bounce-back filter once; reused across cascade stages.
    kept_edges = _filter_edges_for_ddl(edges, predecessors)

    stages = (1, 2, 3, 4)
    last_text = ""
    last_stage = 4
    for stage in stages:
        text = _render(
            stage, concepts, kept_edges, ontology, hop_by_concept, config,
            menu_concepts=menu_concepts or [],
            expand_minimal_concepts=expand_minimal_list,
        )
        last_text = text
        last_stage = stage
        if _count(text) <= config.metadata_context_filter_max_tokens:
            return text, stage
    return last_text, last_stage


def _render(
    stage: int,
    concepts: List[str],
    edges: List[EdgeMeta],
    ontology: Ontology,
    hop_by_concept: Dict[str, int],
    config: MetadataContextConfig,
    *,
    menu_concepts: List[str] | None = None,
    expand_minimal_concepts: List[str] | None = None,
) -> str:
    """Render the concept-centric DDL at the given cascade stage (1..4).

    Cascade (applies to FULL-detail blocks only — minimal blocks always emit
    just description + full outgoing rels regardless of stage):
      1 — full output (props + measures + rels + incoming + type hints)
      2 — drop ``[dec]``/``[int]`` type hints (keep raw ``num``)
      3 — drop ``measures:`` lines entirely (keep props + rels + incoming)
      4 — drop ``props:`` lines entirely (keep only concept headings + rels + incoming)

    Always preserved: concept headings with hop labels and (when known)
    ``description:`` line, ``rels:`` block (mandatory for every concept block
    — sentinel ``rels: (none)`` when there are no outgoing edges in scope),
    self-refs inlined with a ``# recursive`` annotation, ``incoming:`` lines
    for full blocks, and the ``## REACHABLE`` menu band when non-empty.

    Minimal-block rels: render the concept's COMPLETE outgoing rels (within
    the BFS-truncated edge set, which already bounds targets to within
    ``max_graph_depth`` of the anchor). One ``expand_to`` reveals the full chain;
    the LLM can ``build_path`` straight through on the next round without a
    second expand. Edge targets that themselves remain in ``## REACHABLE``
    are shown on the minimal block's rels: line but are NOT auto-promoted to
    minimal blocks — visibility ≠ promotion.
    """
    include_props = stage <= 3
    include_measures = stage <= 2
    include_type_hints = stage <= 1

    detail_set = set(concepts)
    minimal_list = list(expand_minimal_concepts or [])
    minimal_set = {c for c in minimal_list if c not in detail_set}
    all_concepts_in_render = detail_set | minimal_set

    out: List[str] = []

    # ---- CONCEPTS (anchor first, then by hop, lex within hop) -----------
    out.append("## CONCEPTS")
    out.append("")

    # Pre-build outgoing/incoming edge indexes covering both detail and
    # minimal blocks. Outgoing self-refs are included for full blocks;
    # incoming only counts edges from other concepts that are themselves
    # rendered (in either band).
    outgoing_by_concept: Dict[str, List[EdgeMeta]] = {}
    incoming_by_concept: Dict[str, List[EdgeMeta]] = {}
    for e in edges:
        outgoing_by_concept.setdefault(e.from_concept, []).append(e)
        if e.is_self_ref:
            continue
        if e.from_concept not in all_concepts_in_render:
            continue
        incoming_by_concept.setdefault(e.to_concept, []).append(e)

    ordered = sorted(
        all_concepts_in_render,
        key=lambda c: (hop_by_concept.get(c, 9999), c),
    )
    for c in ordered:
        hop = hop_by_concept.get(c)
        label = "[anchor]" if hop == 0 else f"[hop={hop}]" if hop is not None else ""
        if label:
            out.append(f"### {c} {label}")
        else:
            out.append(f"### {c}")

        # Universal description (Part A of Fix 4): emit when present. Skips
        # silently when the source has no description on file.
        desc = _get_concept_description(ontology, c)
        if desc:
            out.append(f"description: {desc}")

        if c in minimal_set:
            # Minimal block — description (already emitted above) + the
            # concept's FULL outgoing rels (no on-path/connecting filter,
            # bounded automatically by the BFS-truncated edge set to within
            # max_graph_depth of the anchor). One expand reveals the full chain;
            # the planner can build_path STRAIGHT THROUGH this concept on
            # the next round without a second expand_to. No props/measures/
            # incoming on minimal blocks — only the rels: surface opens up.
            # rels: is mandatory; sentinel `(none)` when there are no
            # in-scope outgoing edges, so the LLM can distinguish a leaf
            # from missing data.
            rels_block = _build_rels_block(
                outgoing_by_concept.get(c, []),
                with_descriptions=True,
                sentinel_when_empty=True,
            )
            out.extend(rels_block)
            out.append("")
            continue

        # Full-detail block (cascade-aware).
        # props grouped by normalized type (directly-selectable, inheritance-aware).
        # When the cascade drops the block (stage 4), emit an explicit truncation
        # marker so the LLM distinguishes "no props on this concept" from
        # "props hidden — check the source-of-truth before declaring not_needed".
        if include_props:
            prop_block = _build_props_block(
                c, ontology, include_type_hints=include_type_hints,
            )
            out.extend(prop_block)
        elif _concept_has_selectable_props(c, ontology):
            out.append("props: [hidden by cascade — assume present, do not treat as absent]")

        # native + inherited measures only (no relationship-scoped). Same
        # truncation-marker discipline as props above.
        if include_measures:
            measure_line = _build_measures_line(c, ontology)
            if measure_line is not None:
                out.append(measure_line)
        elif _concept_has_native_measures(c, ontology):
            out.append("measures: [hidden by cascade — assume present, do not treat as absent]")

        # outgoing relationships — MANDATORY for detail blocks (Fix 2 sentinel).
        # Self-refs inlined with `# recursive`; descriptions inline when present.
        rels_block = _build_rels_block(
            outgoing_by_concept.get(c, []),
            with_descriptions=True,
            sentinel_when_empty=True,
        )
        out.extend(rels_block)

        # incoming relationships from other rendered concepts.
        incoming_block = _build_incoming_block(
            incoming_by_concept.get(c, []),
        )
        out.extend(incoming_block)

        # sub-concepts (when include_logic_concepts is enabled)
        if config.include_logic_concepts:
            subs = _sub_concepts(ontology, c)
            if subs:
                out.append(f"subconcepts: {', '.join(subs)}")

        out.append("")

    # ---- REACHABLE menu band (names-only, recoverable via expand_to) -----
    # Order is determined by the CALLER (typically menu_builder.split_bands,
    # which sorts by shortest-hop then alphabetical within hop). This
    # serializer only deduplicates while preserving the caller's order —
    # no re-sort. Determinism comes from the caller's stable sort, not
    # from a separate ranking pipeline here (see plan: no multi-factor
    # ranking, no BM25 / business-importance scoring).
    #
    # Concepts that earned a minimal block via ``expand_to`` are removed
    # from this band — they're no longer "behind the curtain."
    if menu_concepts:
        seen: Set[str] = set()
        ordered_unique: List[str] = []
        for c in menu_concepts:
            if c and c not in seen and c not in minimal_set:
                seen.add(c)
                ordered_unique.append(c)
        if ordered_unique:
            out.append("## REACHABLE: " + ", ".join(ordered_unique))
            out.append("")

    return "\n".join(out).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Concept-centric helpers
# ---------------------------------------------------------------------------

_ENTITY_COLS_TO_DROP = frozenset({"entity_id", "entity_label", "entity_type"})


def _compute_hop_distances(
    concepts: List[str],
    predecessors: Dict[str, str | None],
) -> Dict[str, int]:
    """Derive BFS distance from anchor for each visited concept.

    Anchor has predecessor=None (distance 0). Each other concept's distance is
    1 + the predecessor's distance.
    """
    distances: Dict[str, int] = {}
    # Resolve in dependency order — process each concept once its predecessor
    # is already resolved.
    remaining = set(concepts)
    # Find roots first (predecessor=None means anchor).
    for c in concepts:
        if predecessors.get(c) is None:
            distances[c] = 0
            remaining.discard(c)
    # Fixed-point iteration; N is small (typically <30).
    while remaining:
        progress = False
        for c in list(remaining):
            pred = predecessors.get(c)
            if pred is None:
                # Stranded concept (shouldn't happen if BFS was clean) — distance 0.
                distances[c] = 0
                remaining.discard(c)
                progress = True
            elif pred in distances:
                distances[c] = distances[pred] + 1
                remaining.discard(c)
                progress = True
        if not progress:
            # Defensive: anything left has a broken predecessor chain — assign
            # large distance so it lands last.
            for c in remaining:
                distances[c] = 9999
            break
    return distances


def _normalize_sql_type(data_type: str) -> Tuple[str, str | None]:
    """Return ``(group, hint)`` for a SQL type.

    group in {'str', 'num', 'date', 'bool'}; hint is '[int]' or '[dec]' for
    num, else None. Defensive default: 'str' (Plan 2 update spec says so).
    """
    if not data_type:
        return "str", None
    t = data_type.strip().lower()
    # Strip parametric suffix like "decimal(18,2)" → "decimal".
    if "(" in t:
        t = t.split("(", 1)[0]
    if t in ("boolean", "bool"):
        return "bool", None
    if t in ("date", "timestamp", "datetime"):
        return "date", None
    if t in ("integer", "int", "bigint", "smallint", "tinyint"):
        return "num", "[int]"
    if t in ("decimal", "numeric", "double", "float", "real"):
        return "num", "[dec]"
    if t in ("varchar", "char", "text", "string", "nvarchar"):
        return "str", None
    return "str", None


def _build_props_block(
    concept: str,
    ontology: Ontology,
    *,
    include_type_hints: bool,
) -> List[str]:
    """Build the ``props:`` block grouped by normalized type.

    Returns an empty list when the concept has no selectable props (after
    dropping ``entity_*`` columns).
    """
    try:
        meta = ontology.get_concept_metadata(concept)
    except Exception:
        return []
    groups: Dict[str, List[str]] = {"str": [], "num": [], "date": [], "bool": []}
    for prop in meta.properties.values():
        if prop.name in _ENTITY_COLS_TO_DROP:
            continue
        group, hint = _normalize_sql_type(prop.data_type)
        if group == "num" and hint and include_type_hints:
            groups[group].append(f"{prop.name} {hint}")
        else:
            groups[group].append(prop.name)
    # Strip empty groups; sort within each.
    non_empty = {g: sorted(names) for g, names in groups.items() if names}
    if not non_empty:
        return []
    lines = ["props:"]
    for group in ("str", "num", "date", "bool"):
        if group in non_empty:
            lines.append(f"  {group}: {', '.join(non_empty[group])}")
    return lines


def _concept_has_selectable_props(concept: str, ontology: Ontology) -> bool:
    """True if the concept has any property the serializer would emit at stage 1.

    Used to decide whether a stage-4 truncation marker is warranted: only emit
    `props: [hidden by cascade]` when the concept actually has props (otherwise
    a concept with zero props would falsely look truncated).
    """
    try:
        meta = ontology.get_concept_metadata(concept)
    except Exception:
        return False
    for prop in meta.properties.values():
        if prop.name in _ENTITY_COLS_TO_DROP:
            continue
        return True
    return False


def _concept_has_native_measures(concept: str, ontology: Ontology) -> bool:
    """True if the concept has any native (non-relationship-scoped) measure."""
    try:
        meta = ontology.get_concept_metadata(concept)
    except Exception:
        return False
    return any(
        not m.scoped_to_relationship for m in meta.measures.values()
    )


def _build_measures_line(concept: str, ontology: Ontology) -> str | None:
    """Return a single ``measures: a, b, c`` line listing ONLY native measures.

    Excludes relationship-scoped measures (``scoped_to_relationship is not None``).
    """
    try:
        meta = ontology.get_concept_metadata(concept)
    except Exception:
        return None
    native = sorted(
        m.name for m in meta.measures.values()
        if not m.scoped_to_relationship
    )
    if not native:
        return None
    return f"measures: {', '.join(native)}"


def _build_rels_block(
    outgoing: List[EdgeMeta],
    *,
    with_descriptions: bool = True,
    sentinel_when_empty: bool = False,
) -> List[str]:
    """Build the ``rels:`` block listing outgoing edges (including self-refs).

    Each line: ``  -[<rel>[*N], <card>]-> <target>[  # <annotations>]``
    where annotations combine ``recursive`` (for self-refs) and the inline
    relationship description (from ``EdgeMeta.description``) separated by
    `` | `` when both apply.

    Args:
        outgoing: edges leaving the concept. Used as-is — both full and
            minimal blocks render the concept's complete outgoing edge set
            (bounded earlier by the BFS-truncated edge list, which only
            retains edges with targets within ``max_graph_depth`` of the anchor).
        with_descriptions: when True (default), append inline
            ``# <description>`` annotation when ``EdgeMeta.description`` is
            non-empty.
        sentinel_when_empty: when True, emit ``rels: (none)`` for an empty
            edge list instead of returning an empty block. Used by detail
            blocks (Fix 2 — sentinel makes "leaf" distinguishable from
            "missing data") and minimal blocks (the same distinction
            matters for expand_to-promoted concepts).

    Returns ``[]`` when the concept has no outgoing rels AND
    ``sentinel_when_empty=False`` — matches legacy behavior for callers
    that haven't opted in.
    """
    if not outgoing:
        if sentinel_when_empty:
            return ["rels: (none)"]
        return []
    lines = ["rels:"]
    edges_to_emit = outgoing
    # Sort for deterministic output.
    ordered = sorted(edges_to_emit, key=lambda e: (e.relationship_name, e.to_concept))
    for e in ordered:
        rel_token = _format_rel_token(e)
        line = f"  -[{rel_token}, {e.cardinality}]-> {e.to_concept}"
        annotations: List[str] = []
        if e.is_self_ref:
            annotations.append("recursive")
        if with_descriptions and e.description:
            annotations.append(e.description)
        if annotations:
            line += "  # " + " | ".join(annotations)
        lines.append(line)
    return lines


def _get_concept_description(ontology: Ontology, concept: str) -> str | None:
    """Look up the concept's description from ontology metadata.

    Returns ``None`` when the concept is unknown to the ontology or has no
    description on file. Defensive against missing/raising lookups so that
    a metadata gap never breaks DDL rendering.
    """
    try:
        meta = ontology.get_concept_metadata(concept)
    except Exception:
        return None
    desc = getattr(meta, "description", None)
    if desc is None:
        return None
    desc = str(desc).strip()
    return desc or None


def _build_incoming_block(incoming: List[EdgeMeta]) -> List[str]:
    """Build the ``incoming:`` block listing subgraph-internal sources.

    Each line: ``  - <source_concept>.<rel_name>``. Self-refs are NOT listed
    here — they're already conveyed by the outgoing `# recursive` marker.

    Returns an empty list when the concept has no inbound edges from within
    the rendered subgraph.
    """
    if not incoming:
        return []
    lines = ["incoming:"]
    ordered = sorted(
        incoming, key=lambda e: (e.from_concept, e.relationship_name),
    )
    seen: Set[Tuple[str, str]] = set()
    for e in ordered:
        key = (e.from_concept, e.relationship_name)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"  - {e.from_concept}.{e.relationship_name}")
    return lines


def _format_rel_token(edge: EdgeMeta) -> str:
    """Format the rel-name token. Only the transitivity suffix is exposed;
    the inverse/canonical distinction is intentionally hidden because the
    LLM doesn't need to know — every edge in the rels list is a valid
    traversal target. Bounce-back inverses are still dropped upstream by
    ``should_include_in_ddl``."""
    name = edge.relationship_name
    if edge.transitivity > 1:
        name = f"{name}*{edge.transitivity}"
    return name


def _filter_edges_for_ddl(
    edges: List[EdgeMeta],
    predecessors: Dict[str, str | None],
) -> List[EdgeMeta]:
    """Apply the inverse-bounce-back filter using ``should_include_in_ddl``.

    Drops inverses pointing back to the previous-hop concept; keeps self-refs
    and unrelated inverses.
    """
    kept: List[EdgeMeta] = []
    for edge in edges:
        if should_include_in_ddl(
            _EdgeAsRel(edge),
            current_concept=edge.from_concept,
            previous_hop_concept=predecessors.get(edge.from_concept),
        ):
            kept.append(edge)
    return kept


class _EdgeAsRel:
    """Adapter so EdgeMeta can be passed to should_include_in_ddl unchanged."""

    __slots__ = ("target_concept", "is_inverse")

    def __init__(self, edge: EdgeMeta):
        self.target_concept = edge.to_concept
        self.is_inverse = edge.is_inverse


def _sub_concepts(ontology: Ontology, concept: str) -> List[str]:
    """Look up sub-concepts of ``concept`` via Plan 1's inheritance lookup."""
    try:
        return ontology.subconcepts_of(concept)
    except Exception:
        return []
