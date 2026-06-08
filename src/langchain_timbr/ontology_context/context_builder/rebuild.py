"""Build columns/measures/relationships for SQL-gen consumption.

Anti-hallucination guarantee (constructor approach):
  The rebuilt relationships dict is constructed FROM SCRATCH by walking the
  validated paths and emitting, at each segment endpoint, only that concept's
  DIRECT properties + measures (sourced from ontology metadata). No graph
  traversal happens here — off-path relationships literally cannot appear in
  the output because we never enumerate them.

Anchor-rooted prefix invariant:
  Every emitted property prefix MUST start at the anchor concept. The Step 1
  LLM is allowed to return paths in either style:
    - Full-chain style: one multi-segment SelectedPath starting at the anchor.
    - Single-hop style: several 1-segment SelectedPaths chained end-to-start,
      where path P_k+1 begins where P_k ends.
  Both styles must produce identical output. We enforce this with a
  normalization pre-pass (``_normalize_paths_to_anchor``) that prepends the
  segments of an earlier path onto any path whose first segment doesn't start
  at the anchor. After normalization every path begins at the anchor and the
  walk emits anchor-rooted prefixes by construction.

The Step 1 LLM's selected_properties / selected_measures are advisory only —
they don't filter here. We always emit ALL direct properties + measures of
each path concept.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, List, Optional, Set, Tuple

from .metadata_types import PathSegment, SelectedPath, TransitivityOverride

logger = logging.getLogger(__name__)


def collect_path_concepts(paths: Iterable[SelectedPath]) -> Set[str]:
    """Return the union of from/to concepts (+ sub-concept expansions) across all paths."""
    out: Set[str] = set()
    for path in paths:
        for seg in path.segments:
            out.add(seg.from_concept)
            out.add(seg.to_concept)
            for sub in getattr(seg, "expanded_sub_concepts", []) or []:
                out.add(sub)
            modeled = getattr(seg, "modeled_target", None)
            if modeled:
                out.add(modeled)
    return out


def collect_path_relationships(paths: Iterable[SelectedPath]) -> Set[Tuple[str, str, str]]:
    """Return the set of (from, rel, to) triples traversed by validated paths."""
    return {
        (s.from_concept, s.relationship_name, s.to_concept)
        for p in paths
        for s in p.segments
    }


def filter_columns_for_concepts(
    full_columns: List[dict],
    path_concepts: Set[str],
    *,
    concept_field: str = "concept",
) -> List[dict]:
    """Return columns whose ``concept_field`` is in ``path_concepts``.

    Used for the flat anchor columns/measures lists. Existing upstream code
    formats `columns` as list[dict] keyed by `col_name`, without a `concept`
    field — when the field is absent we keep the column (the caller has
    already scoped these to the anchor concept).
    """
    if not path_concepts:
        return list(full_columns)
    out: List[dict] = []
    for col in full_columns:
        concept = col.get(concept_field)
        if concept is None or concept in path_concepts:
            out.append(col)
    return out


def _normalize_paths_to_anchor(
    paths: Iterable[SelectedPath],
    anchor: Optional[str],
) -> List[Tuple[str, List[PathSegment]]]:
    """Make every path anchor-rooted by prepending an earlier path's segments.

    Processes paths in order. For each path P:
      - If P.segments[0].from == anchor, keep as-is.
      - Else find the most recent earlier (already-normalized) path Q whose
        last segment's `to` == P.segments[0].from. Prepend Q.segments to P.
        Since Q is already anchor-rooted, one prepend is sufficient.
      - If no such Q exists, drop P (logged at warning).

    Empty-segment paths are dropped silently. Returns a list of
    ``(path_id, segments)`` tuples — the path_id is preserved for logging.

    When ``anchor`` is None we skip normalization entirely (legacy callers).
    """
    out: List[Tuple[str, List[PathSegment]]] = []
    for path in paths:
        segments = list(path.segments)
        if not segments:
            continue
        if anchor is None or segments[0].from_concept == anchor:
            out.append((path.path_id, segments))
            continue
        prepend: Optional[List[PathSegment]] = None
        target = segments[0].from_concept
        for prior_id, prior_segs in reversed(out):
            if prior_segs and prior_segs[-1].to_concept == target:
                prepend = list(prior_segs)
                break
        if prepend is None:
            logger.warning(
                "rebuild: cannot anchor-root path %r (starts at %r, no earlier "
                "path ends there); dropping. anchor=%r",
                path.path_id, target, anchor,
            )
            continue
        out.append((path.path_id, prepend + segments))
    return out


def build_relationships_from_paths(
    paths: Iterable[SelectedPath],
    ontology,
    *,
    anchor: Optional[str] = None,
) -> dict:
    """Construct the SQL-gen relationships dict from scratch.

    For each path, walks segments in order. After each segment, the running
    prefix is extended by ``rel[to_concept]`` (with a ``*N`` transitivity
    marker baked in when ``RelationshipMeta.transitivity > 1``), and the
    target concept's DIRECT properties + measures are emitted as nested
    columns/measures under the first hop's rel name.

    Properties of the anchor (empty prefix) are NOT emitted here — they live
    in the flat ``columns``/``measures`` lists handled by
    ``filter_columns_for_concepts``.

    Both path styles produce the same output:
      - Full-chain: ``[customer → order → product → material]`` (1 path, 3 segs).
      - Single-hop: ``[customer → order], [order → product], [product → material]``
        (3 paths, 1 seg each).
    The normalization pre-pass prepends earlier paths' segments so that, in
    the single-hop case, P2 effectively becomes the 2-seg path
    ``customer → order → product`` and P3 the 3-seg path
    ``customer → order → product → material``. The dedup walk then skips
    prefixes already emitted by an earlier (shorter) path.

    Returns ``dict[first_hop_rel_name -> {description, columns, measures}]``
    matching the shape ``_build_rel_columns_str`` consumes. ``description``
    is the first hop's RelationshipMeta description (best-effort; first
    non-empty wins when multiple paths share the same first hop).

    The output is anti-hallucination by construction: only ``(prefix,
    concept)`` pairs that follow the selected paths exist, so off-path
    relationships, cross-product chains, and over-depth expansions are
    impossible.
    """
    normalized = _normalize_paths_to_anchor(paths, anchor)

    out: dict = {}
    seen_prefixes: Set[str] = set()

    for _path_id, segments in normalized:
        if not segments:
            continue

        prefix_parts: List[str] = []
        current_from = segments[0].from_concept
        for seg in segments:
            target = seg.to_concept
            transitivity = _lookup_transitivity(
                ontology, current_from, seg.relationship_name,
            )
            if transitivity > 1:
                prefix_parts.append(f"{seg.relationship_name}[{target}*{transitivity}]")
            else:
                prefix_parts.append(f"{seg.relationship_name}[{target}]")
            prefix_str = ".".join(prefix_parts)

            # Dedup: when single-hop paths get normalized into nested chains
            # that all share an anchor-rooted base, the same prefix would
            # otherwise be emitted by every path containing it.
            if prefix_str in seen_prefixes:
                current_from = target
                continue
            seen_prefixes.add(prefix_str)

            # Key buckets by FULL prefix (per-hop), not by the top-level
            # rel — so an N-hop chain produces N independent blocks in
            # the rendered SQL-gen prompt, each carrying ITS hop's own
            # description + cardinality. The renderer (`_build_rel_columns_str`)
            # iterates this dict and emits one block per key.
            bucket = out.setdefault(
                prefix_str,
                {"description": "", "columns": [], "measures": []},
            )
            # Compose THIS hop's description + cardinality (not the top
            # hop's). compose_..._with_cardinality returns "" when both
            # inputs are empty, so no stray "cardinality: " marker leaks
            # when the rel is unknown.
            if not bucket["description"]:
                bucket["description"] = compose_rel_description_with_cardinality(
                    _lookup_relationship_description(
                        ontology, current_from, seg.relationship_name,
                    ),
                    _safe_cardinality_of(
                        ontology, current_from, seg.relationship_name,
                    ),
                )

            meta = _safe_get_concept_metadata(ontology, target)
            if meta is None:
                current_from = target
                continue

            for prop in meta.properties.values():
                bucket["columns"].append({
                    "name": f"{prefix_str}.{prop.name}",
                    "col_name": prop.name,
                    "data_type": prop.data_type,
                    "description": prop.description,
                })
            # Emit ONLY direct measures of this concept. ConceptMetadata.measures
            # also holds ``measure_rel`` entries (scoped_to_relationship=<rel>)
            # for measures reached via 1-hop describe — those belong to other
            # concepts and would leak under this prefix. The parser keys
            # measure_rel entries by ``<rel>.<m_name>``, so the same measure
            # name can appear multiple times (direct + each rel alias);
            # without this filter that produces duplicates (count_of_product
            # 4× under product prefix in the user's bug report).
            for measure in meta.measures.values():
                if getattr(measure, "scoped_to_relationship", None) is not None:
                    continue
                bucket["measures"].append({
                    "name": f"measure.{prefix_str}.{measure.name}",
                    "col_name": measure.name,
                    "data_type": measure.data_type,
                    "description": measure.description,
                })

            current_from = target

    return out


def _safe_get_concept_metadata(ontology, concept: str):
    """Fetch concept metadata, returning None on any failure.

    The constructor must tolerate missing concepts — a malformed path
    shouldn't crash the rebuild, it should just produce an empty bucket
    entry for that segment.
    """
    try:
        return ontology.get_concept_metadata(concept)
    except Exception:
        return None


def _lookup_transitivity(ontology, from_concept: str, rel_name: str) -> int:
    """Return the declared transitivity (depth) for a relationship, or 1.

    ``1`` is the safe default — emit ``rel[target]`` without a ``*N`` marker.
    Higher values produce ``rel[target*N]``, which ``apply_transitivity_overrides``
    can later rewrite if the LLM requested a different depth.
    """
    meta = _safe_get_concept_metadata(ontology, from_concept)
    if meta is None:
        return 1
    rel_meta = meta.relationships.get(rel_name)
    if rel_meta is None:
        return 1
    try:
        return int(getattr(rel_meta, "transitivity", 1) or 1)
    except (TypeError, ValueError):
        return 1


def _lookup_relationship_description(
    ontology, from_concept: str, rel_name: str,
) -> str:
    """Return the relationship's description string, or '' on any failure."""
    meta = _safe_get_concept_metadata(ontology, from_concept)
    if meta is None:
        return ""
    rel_meta = meta.relationships.get(rel_name)
    if rel_meta is None:
        return ""
    return rel_meta.description or ""


def _safe_cardinality_of(ontology, from_concept: str, rel_name: str) -> Optional[str]:
    """Look up the relationship's cardinality (e.g. 'N:M'), or None on any
    failure. Mirrors ``_lookup_relationship_description``'s defensive
    pattern — the SQL-gen pipeline must never abort because cardinality
    couldn't be resolved for one relationship."""
    try:
        return ontology.cardinality_of(from_concept, rel_name)
    except Exception:
        return None


def compose_rel_description_with_cardinality(
    raw_desc: Optional[str], cardinality: Optional[str],
) -> str:
    """Append cardinality to an existing relationship description, or
    surface it on its own when there is no description. Returns ``""``
    when both inputs are empty/None.

    Used by both the dynamic-rebuild path (``build_relationships_from_paths``)
    and the static SQL-gen context path
    (``_build_sql_generation_context``) so the SQL-gen LLM sees the join
    multiplicity inline with the rel description in either mode.

    Examples:
        ("links A to B", "N:M") -> "links A to B (cardinality: N:M)"
        ("", "N:1")             -> "cardinality: N:1"
        (None, None)            -> ""
    """
    raw = (raw_desc or "").strip()
    card = (cardinality or "").strip()
    if raw and card:
        return f"{raw} (cardinality: {card})"
    if card:
        return f"cardinality: {card}"
    return raw


def apply_transitivity_overrides(
    text: str,
    overrides: Iterable[TransitivityOverride],
) -> str:
    """Rewrite ``<rel>[<target>*<old>]`` to ``<rel>[<target>*<level>]`` for each
    accepted override.

    Applied to the rebuilt SQL-gen context strings (columns_str, measures_str,
    rel_prop_str) so the depth chosen by the Step 1 LLM is baked into the
    prompt before SQL gen sees it.

    The override's ``rel`` matches the literal rel name in the text — a leading
    ``~`` is part of the name, not an inverse marker, and is not stripped.

    Empty / None overrides list is a no-op.
    """
    if not overrides:
        return text or ""
    if not text:
        return ""
    out = text
    for ov in overrides:
        if not ov.rel or not ov.target or ov.level is None:
            continue
        # Negative lookbehind: don't match when the rel name is preceded by
        # a name-character (word char or `~`). Prevents ``has_acquired``
        # from matching as a suffix of ``~has_acquired`` — they're distinct
        # literal names.
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_~]){re.escape(ov.rel)}\[{re.escape(ov.target)}\*\d+\]"
        )
        out = pattern.sub(f"{ov.rel}[{ov.target}*{ov.level}]", out)
    return out


# ---------------------------------------------------------------------------
# Waypoint filter — threshold-gated, precondition-protected
# ---------------------------------------------------------------------------

# Marker emitted by the DDL serializer when stage>=3 hides measures/props for
# a concept. Presence in the rendered DDL means the path-selection prompt
# was DEGRADED — the LLM did not have full visibility, so its
# ``is_intermediate`` flags are uninformed and the waypoint filter must skip.
_CASCADE_HIDDEN_MARKER = "[hidden by cascade"


def is_path_prompt_degraded(compact_ddl: str) -> bool:
    """True when the path-selection Compact DDL contained cascade-hidden
    markers (some concept's props or measures were trimmed). When True, the
    waypoint filter MUST NOT use ``is_intermediate`` flags — the LLM made
    its decisions without full visibility."""
    if not compact_ddl:
        return False
    return _CASCADE_HIDDEN_MARKER in compact_ddl


def compute_waypoint_strip_set(
    validated_paths: List[SelectedPath],
    anchor: str,
) -> Tuple[Set[str], Set[str]]:
    """Classify each concept reached by ``validated_paths`` and return
    ``(keep_set, strip_set)``.

    Algorithm — per the spec:
      - Anchor → always kept
      - Terminal of any path (last segment's ``to``) → always kept; if the
        LLM set ``is_intermediate=True`` on a terminal, the override is
        silently ignored (caller may log)
      - Otherwise, a concept appearing as ``to`` is kept iff ANY of its
        occurrences has ``is_intermediate=False`` (conflicting signals →
        keep wins)
      - A concept appearing ONLY in ``is_intermediate=True`` segments and
        never as a terminal is added to ``strip_set``

    Stripping a concept means: drop columns/measures whose nested-chain
    TERMINUS (last hop's target) is that concept.
    """
    keep: Set[str] = {anchor}
    terminals: Set[str] = set()
    appearances: dict = {}  # concept -> list[bool]  (is_intermediate per occurrence)

    for path in validated_paths or []:
        segs = getattr(path, "segments", []) or []
        if not segs:
            continue
        terminals.add(segs[-1].to_concept)
        for seg in segs:
            flag = bool(getattr(seg, "is_intermediate", False))
            appearances.setdefault(seg.to_concept, []).append(flag)

    keep |= terminals

    strip: Set[str] = set()
    for concept, flags in appearances.items():
        if concept in keep:
            continue
        if all(flags):
            strip.add(concept)
        else:
            keep.add(concept)

    return keep, strip


def _column_terminus(name: str) -> Optional[str]:
    """Return the concept the nested chain terminates at — the LAST
    ``[concept]`` bracket before the final property name. Returns None
    for direct attributes with no nested chain.

    Examples:
      ``of_customer[customer].name`` → ``customer``
      ``of_customer[customer].received[shipment].date`` → ``shipment``
      ``has_acquired[company*3].name`` → ``company`` (``*N`` stripped)
      ``name`` → None
    """
    if not name:
        return None
    s = name
    if s.startswith("measure."):
        s = s[len("measure."):]
    last_target: Optional[str] = None
    for part in s.split("."):
        if "[" not in part:
            continue
        bracket = part.split("[", 1)[1]
        target = bracket.split("]", 1)[0]
        if "*" in target:
            target = target.split("*", 1)[0]
        last_target = target
    return last_target


def strip_waypoint_columns(
    full_relationships: dict,
    strip_concepts: Set[str],
) -> dict:
    """Drop columns and measures whose nested-chain TERMINUS is in
    ``strip_concepts``. Top-level rel keys are retained even when emptied —
    the caller decides whether to drop empty entries.

    Direct (non-nested) attributes are untouched — they belong to the anchor,
    which is never in ``strip_concepts`` by construction.
    """
    if not strip_concepts:
        return dict(full_relationships)

    out: dict = {}
    for rel_key, rel_data in full_relationships.items():
        if not isinstance(rel_data, dict):
            out[rel_key] = rel_data
            continue
        new_rel = dict(rel_data)
        for field_name in ("columns", "measures"):
            entries = rel_data.get(field_name)
            if not isinstance(entries, list):
                continue
            new_rel[field_name] = [
                c for c in entries
                if _column_terminus(
                    c.get("name") or c.get("col_name") or ""
                ) not in strip_concepts
            ]
        out[rel_key] = new_rel
    return out
