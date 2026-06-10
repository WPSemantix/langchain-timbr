"""Path validator for Step 1 LLM output.

Six rules (numbered after the plan):
  1. Reachable start          — first segment's from == anchor OR == endpoint of
                                a previously-validated path in this batch
  2. Segment existence        — (from, rel, to) in edge_index
  3. Chain continuity         — segments[i].to == segments[i+1].from
  4. Hop budget               — len(segments) <= max_hop + 2 (slight overshoot allowed)
  5. (removed)                — INVALID_RECURSION validation dropped; is_recursive
                                is informational only
  6. Concept existence        — concepts in segments are known

Rule 1 was relaxed in the segmented-paths update to accept LLM output where each
path is a single hop (chained implicitly by ordering). Paths are topologically
sorted before validation so out-of-order emission still validates.

WRONG_DIRECTION is intentionally not a rule (canonical/inverse split is upstream).

Transitivity overrides (from Step1Output.transitivity_overrides) are validated
separately by validate_overrides() — invalid ones are silently dropped rather
than producing reason codes, since a bad override doesn't break SQL correctness
(the default *N is still valid).
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Tuple

from .edge_index import EdgeIndex
from .metadata_types import SelectedPath, TransitivityOverride, ValidationError

logger = logging.getLogger(__name__)


REASON_INVALID_START_CONCEPT = "INVALID_START_CONCEPT"
REASON_UNKNOWN_RELATIONSHIP = "UNKNOWN_RELATIONSHIP"
REASON_BROKEN_CHAIN = "BROKEN_CHAIN"
REASON_HOP_BUDGET_EXCEEDED = "HOP_BUDGET_EXCEEDED"
# Kept exported for backward compat; no longer emitted by validate_paths.
REASON_INVALID_RECURSION = "INVALID_RECURSION"
REASON_UNKNOWN_CONCEPT = "UNKNOWN_CONCEPT"


def _topological_sort_paths(
    paths: Iterable[SelectedPath],
    anchor: str,
) -> Tuple[List[SelectedPath], set]:
    """Reorder paths so each path's start is reachable from anchor or an
    earlier path's endpoint in the result. Unorderable paths (start never
    reachable) are appended at the end so the validator emits
    INVALID_START_CONCEPT for them.

    O(N^2) — N is small (typically <10 paths). Greedy: at each pass we move
    every path whose start is currently reachable into the sorted list and
    add its endpoint to the reachable set.

    Empty-segment paths are appended early so the validator can reject them
    on their own merits.
    """
    remaining: List[SelectedPath] = list(paths)
    ordered: List[SelectedPath] = []
    reachable = {anchor}
    progress = True
    while remaining and progress:
        progress = False
        for path in list(remaining):
            if not path.segments:
                ordered.append(path)
                remaining.remove(path)
                progress = True
                continue
            start = path.segments[0].from_concept
            if start in reachable:
                ordered.append(path)
                reachable.add(path.segments[-1].to_concept)
                remaining.remove(path)
                progress = True
    # Leftovers: start concept never reachable in any ordering. Append so the
    # validator can produce INVALID_START_CONCEPT with the final reachable set.
    return ordered + remaining, reachable


def _linear_runs(segments: List) -> List[List]:
    """Split a segment list into maximal linear runs at continuity breaks.

    A break is any point where ``segments[i].from != segments[i-1].to``. Each
    returned run is internally continuous (a valid linear chain).
    """
    runs: List[List] = []
    cur: List = [segments[0]]
    for prev, seg in zip(segments, segments[1:]):
        if prev.to_concept == seg.from_concept:
            cur.append(seg)
        else:
            runs.append(cur)
            cur = [seg]
    runs.append(cur)
    return runs


def split_branching_paths(
    paths: Iterable[SelectedPath],
    anchor: str,
) -> List[SelectedPath]:
    """Split a fork that was mis-packed into one ``path_id`` into separate
    linear paths.

    A ``SelectedPath`` must be a single linear chain (each segment's ``to`` is
    the next segment's ``from``) — the rebuild pipeline assumes this. LLMs
    sometimes pack a fork (two branches from a shared concept) into one
    ``path_id``, which would otherwise fail Rule 3 (BROKEN_CHAIN). This pass
    rewrites such a path into one linear path per branch BEFORE validation, so
    both validation and rebuild see clean linear chains.

    Conservative by design: a split is committed only when EVERY resulting run
    provably starts at a reachable concept — the anchor, or the terminal of an
    earlier run / earlier path. That is exactly the start set recognized by
    Rule 1 and by rebuild's ``_normalize_paths_to_anchor`` (both key off a
    prior path's *terminal*, never an intermediate). If a run would start at a
    non-reachable concept (e.g. a branch off a sibling run's intermediate), the
    path is left intact and the existing validator + retry path handles it —
    we never emit confusing synthetic-id errors.

    A legitimate waypoint full-chain (``a -> b -> c``) has no continuity break,
    so it is returned unchanged. Split path_ids are derived as ``<path_id>.<k>``
    to preserve traceability back to the LLM's original ``path_id``.
    """
    out: List[SelectedPath] = []
    reachable = {anchor}
    for path in paths:
        segments = list(path.segments)
        if not segments:
            out.append(path)
            continue
        runs = _linear_runs(segments)
        if len(runs) == 1:
            out.append(path)
            reachable.add(segments[-1].to_concept)
            continue
        # Only commit the split if every run starts at a reachable concept,
        # simulating the reachable set growing run-by-run.
        sim = set(reachable)
        startable = True
        for run in runs:
            if run[0].from_concept not in sim:
                startable = False
                break
            sim.add(run[-1].to_concept)
        if not startable:
            out.append(path)
            reachable.add(segments[-1].to_concept)
            continue
        for k, run in enumerate(runs, 1):
            out.append(SelectedPath(
                path_id=f"{path.path_id}.{k}",
                purpose=path.purpose,
                segments=run,
                is_recursive=path.is_recursive,
            ))
            reachable.add(run[-1].to_concept)
    return out


def validate_paths(
    paths: Iterable[SelectedPath],
    anchor: str,
    edge_index: EdgeIndex,
    *,
    max_hop: int,
    include_logic_concepts: bool = False,
) -> List[ValidationError]:
    """Validate selected paths; return a list of errors (empty list == all valid).

    Paths are topologically sorted by dependency BEFORE validation so the LLM
    can emit them in any order. Rule 1 (start concept) requires the first
    segment's from to equal the anchor OR an endpoint reached by an earlier
    validated path in this batch.
    """
    errors: List[ValidationError] = []
    sorted_paths, _ = _topological_sort_paths(paths, anchor)
    reachable = {anchor}

    for path in sorted_paths:
        segments = list(path.segments)
        if not segments:
            errors.append(ValidationError(
                path_id=path.path_id,
                segment_index=-1,
                reason_code=REASON_HOP_BUDGET_EXCEEDED,
                detail="Path has no segments",
            ))
            continue

        # Rule 1 — reachable start (anchor OR a previously-validated path's endpoint)
        if segments[0].from_concept not in reachable:
            errors.append(ValidationError(
                path_id=path.path_id,
                segment_index=0,
                reason_code=REASON_INVALID_START_CONCEPT,
                detail=(
                    f"Path must start from anchor {anchor!r} or from a concept "
                    f"reached by a previous path. Reachable so far: "
                    f"{sorted(reachable)}; got {segments[0].from_concept!r}"
                ),
            ))
            continue

        # Snapshot the error count so we can tell whether this path validated
        # cleanly (only then do we add its endpoint to the reachable set).
        _err_count_before = len(errors)

        # Rule 4 — hop budget (slight overshoot allowed)
        if len(segments) > max_hop + 2:
            errors.append(ValidationError(
                path_id=path.path_id,
                segment_index=-1,
                reason_code=REASON_HOP_BUDGET_EXCEEDED,
                detail=f"Path length {len(segments)} exceeds budget (max_hop={max_hop})",
            ))

        # Rule 5 removed in transitivity-overrides update — is_recursive is now
        # informational only. Depth control is per-relationship via the new
        # TransitivityOverride mechanism (validated by validate_overrides()).

        # Per-segment validation
        for i, seg in enumerate(segments):
            # Rule 3 — chain continuity
            if i > 0 and segments[i - 1].to_concept != seg.from_concept:
                errors.append(ValidationError(
                    path_id=path.path_id,
                    segment_index=i,
                    reason_code=REASON_BROKEN_CHAIN,
                    detail=(
                        f"Segment {i-1} ends at {segments[i-1].to_concept!r} but "
                        f"segment {i} starts at {seg.from_concept!r}. Each path_id "
                        f"must be ONE linear chain. If this is a fork from a shared "
                        f"concept, split it into separate path_ids — one linear chain each"
                    ),
                ))

            # Rule 6 — concept existence (via outbound_edges materialization)
            try:
                _ = edge_index.ontology.get_concept_metadata(seg.from_concept)
            except Exception:
                errors.append(ValidationError(
                    path_id=path.path_id,
                    segment_index=i,
                    reason_code=REASON_UNKNOWN_CONCEPT,
                    detail=seg.from_concept,
                ))
                continue
            try:
                _ = edge_index.ontology.get_concept_metadata(seg.to_concept)
            except Exception:
                # Concept might still be a sub-concept reachable via inheritance —
                # the sub-concept-aware fallback below catches that case.
                if not include_logic_concepts:
                    errors.append(ValidationError(
                        path_id=path.path_id,
                        segment_index=i,
                        reason_code=REASON_UNKNOWN_CONCEPT,
                        detail=seg.to_concept,
                    ))
                    continue

            # Rule 2 — segment existence (direct or sub-concept fallback)
            edge = edge_index.lookup(seg.from_concept, seg.relationship_name, seg.to_concept)
            if edge is not None:
                continue
            if include_logic_concepts:
                # Sub-concept-aware fallback: accept if `to` is a sub-concept of
                # the modeled target of (from, rel, ?).
                resolved = _resolve_sub_concept(
                    seg.from_concept, seg.relationship_name, seg.to_concept, edge_index
                )
                if resolved is not None:
                    seg.expanded_sub_concepts = [seg.to_concept]
                    seg.modeled_target = resolved
                    continue
            errors.append(ValidationError(
                path_id=path.path_id,
                segment_index=i,
                reason_code=REASON_UNKNOWN_RELATIONSHIP,
                detail=f"{seg.from_concept}.{seg.relationship_name}.{seg.to_concept}",
            ))

        # Only fully-valid paths contribute their endpoint to the reachable
        # set used by Rule 1 for subsequent paths.
        if len(errors) == _err_count_before:
            reachable.add(segments[-1].to_concept)

    return errors


def _resolve_sub_concept(
    from_concept: str,
    relationship_name: str,
    to_concept: str,
    edge_index: EdgeIndex,
) -> str | None:
    """Return the modeled target if ``to_concept`` is a recognized sub-concept of it.

    Requires Plan 1 to expose inheritance chains on ConceptMetadata. Until that
    surface is available, returns None — sub-concept resolution stays a no-op
    and the validator falls through to UNKNOWN_RELATIONSHIP.
    """
    return None


def validate_overrides(
    overrides: Iterable[TransitivityOverride],
    edge_index: EdgeIndex,
) -> List[TransitivityOverride]:
    """Filter LLM-emitted transitivity overrides down to the ones that timbr
    will actually honor at SQL time.

    Acceptance rules (silent drop on rejection — no reason codes; bad overrides
    don't break SQL correctness since the default *N is always valid):

      - level < 2                                  → drop (no-op)
      - rel.is_self_ref == True                    → accept (timbr allows
                                                     overriding self-ref edges
                                                     to any depth, even when
                                                     the default is *1)
      - rel.transitivity > 1 (real transitive)     → accept
      - non-self-ref AND transitivity == 1         → drop (flat relationship
                                                     can't be made transitive)
      - relationship not found in edge_index       → drop (phantom rel)

    Returns the list of accepted overrides in original order. Duplicates on the
    same (rel, target) are de-duped to the LAST entry (LLMs occasionally emit
    the same override twice).
    """
    accepted: List[TransitivityOverride] = []
    seen_keys: dict = {}

    for ov in overrides:
        if not ov.rel or not ov.target or ov.level is None:
            logger.debug("Dropping override with empty fields: %r", ov)
            continue
        if ov.level < 2:
            logger.debug("Dropping override with level<2 (no-op): %r", ov)
            continue

        # Find any edge with matching (rel, target) — applicability depends on
        # rel-level properties (is_self_ref, transitivity), which are the same
        # for every from_concept that uses this (rel, target) pair.
        matching_edge = _find_edge_by_rel_target(edge_index, ov.rel, ov.target)
        if matching_edge is None:
            logger.debug("Dropping override for unknown (rel, target): %r", ov)
            continue

        is_self_ref = matching_edge.is_self_ref
        is_transitive = matching_edge.transitivity > 1

        if not (is_self_ref or is_transitive):
            logger.debug(
                "Dropping override on flat non-self-ref relationship: %r", ov,
            )
            continue

        key = (ov.rel, ov.target)
        seen_keys[key] = ov

    # Preserve insertion order of last-seen entries.
    for key, ov in seen_keys.items():
        accepted.append(ov)

    return accepted


def _find_edge_by_rel_target(edge_index: EdgeIndex, rel: str, target: str):
    """Return the first EdgeMeta in the index matching (rel, target), or None.

    Iterates the internal edge_map; in practice this is small (only edges
    visited by BFS). For the override-validation path the small cost is fine.
    """
    for (from_c, rel_name, to_c), edge in edge_index._edge_map.items():
        if rel_name == rel and to_c == target:
            return edge
    return None
