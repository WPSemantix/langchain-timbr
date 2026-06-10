"""Orchestrator for the dynamic metadata-context pipeline.

Entry point: ``build_filtered_metadata`` — called from
``_build_sql_generation_context`` when ``metadata_context_mode`` is dynamic OR
``auto`` decides to switch.

Pipeline (current, minimal):
  Step 0  — anchor-rooted BFS subgraph + concept pre-filter + Compact DDL.
  Step 1  — LLM filter + path inference (single contract; ``paths_status`` /
            ``not_found`` removed per the action-grammar plan).
  Step 1b — Validator, bounded validation-retry, deterministic shortest-path
            floor over the LLM's ``selected_concepts`` (and the pre-filter's
            output as a last resort) when validation produces no valid paths.
  Output  — ``DynamicMetadataResult`` with filtered concepts + validated paths
            + telemetry. The caller consumes ``filtered_concepts`` and
            ``validated_paths`` to rebuild the SQL-gen context strings (see
            ``rebuild.py``).

The path-selection contract is in transition — see
[proactive-menu-path-selection.md](../../../../.claude/plans/proactive-menu-path-selection.md)
for the constrained action grammar that will replace Step 1's free-form
output in Phase 1 of that plan. The orchestrator surface (signature, result
envelope) is stable; only the LLM contract changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ..ontology.graph import Ontology
from .concept_prefilter import (
    run_concept_prefilter,
    should_trigger_concept_prefilter,
)
from .edge_index import EdgeIndex
from .fallback import generate_fallback_paths
from .llm_filter import run_step1_filter, run_step1_retry
from .menu_builder import (
    MenuEntry,
    build_hop_map,
    split_bands,
)
from .metadata_config import MetadataContextConfig
from .metadata_types import (
    EdgeMeta,
    PathSegment,
    SelectedPath,
    Step1Output,
    TransitivityOverride,
    ValidationError,
)
from .rebuild import collect_path_concepts, collect_path_relationships
from .subgraph import retrieve_subgraph, serialize_compact_ddl
from .validator import split_branching_paths, validate_overrides, validate_paths

# Per-request caps on the planner's non-build_path actions. Enforced via
# grammar narrowing — when a cap is hit, the next planner call's prompt
# omits the disallowed action from the allowed-action union so the LLM
# cannot request it. This is cheaper and tighter than emit-and-reject.
#
# Counter semantics:
#   - expand_count is per-round, not per-concept. One ``expand_to`` action
#     promotes ALL its named targets in a single round; the round is what
#     gets counted, regardless of how many targets are named.
#   - reanchor resets expand_count (new anchor → fresh expand budget) but
#     reanchor_count itself is monotonic across the request, so total swaps
#     stay bounded at _REANCHOR_CAP=1.
_EXPAND_CAP = 2
_REANCHOR_CAP = 1


def _allowed_actions(
    expand_count: int,
    reanchor_count: int,
    *,
    menu_size: int = 1,
) -> List[str]:
    """Return the action list the planner is allowed to emit this round.

    Caps are enforced at the prompt level: when a budget is exhausted, the
    corresponding action is dropped from the union shown to the LLM. The
    LLM literally cannot request it — no post-hoc rejection branch.

    ``menu_size`` is the count of concepts currently in the ``## REACHABLE``
    band. When it is ``0`` the menu is empty and ``expand_to`` has no possible
    valid target, so the action is dropped from the grammar — this is the
    structural counterpart of the prompt's HARD rule that expand_to targets
    come exclusively from ``## REACHABLE``. The default of ``1`` preserves
    legacy direct callers (e.g. unit tests) that don't know about the menu.
    """
    actions = ["build_path"]
    if expand_count < _EXPAND_CAP and menu_size > 0:
        actions.append("expand_to")
    if reanchor_count < _REANCHOR_CAP:
        actions.append("reanchor")
    return actions

logger = logging.getLogger(__name__)


@dataclass
class DynamicMetadataResult:
    """Result envelope from ``build_filtered_metadata``.

    The caller uses ``filtered_concepts`` and ``path_rel_keys`` to filter
    its existing columns/measures/relationships dicts (anti-hallucination
    guarantee: ALL properties/measures for path concepts are kept).
    """
    filtered_concepts: Set[str]
    path_rel_keys: Set[tuple]
    validated_paths: List[SelectedPath]
    compact_ddl: str
    stats: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    accepted_overrides: List[TransitivityOverride] = field(default_factory=list)
    # Retained for downstream wiring symmetry. Will carry the `reanchor`
    # action's target once the action-grammar plan lands; today it is always
    # None because anchor swaps no longer happen inside this module.
    effective_anchor: Optional[str] = None


def build_filtered_metadata(
    *,
    question: str,
    anchor: str,
    ontology: Ontology,
    llm,
    config: MetadataContextConfig,
    graph_depth: int,
    note: str = "",
) -> DynamicMetadataResult:
    """Run the dynamic pipeline and return a result envelope.

    Flow:
      1. Step 0 — build subgraph + DDL (concept pre-filter engages ONLY when
         token estimate > ``metadata_context_filter_max_tokens`` OR detail-band count >
         ``max_detail_concepts``; otherwise it is silent).
      2. Action loop — drive the planner LLM through ``build_path`` /
         ``expand_to`` / ``reanchor`` actions. Invalid ``expand_to`` is
         re-prompted within the expand budget; cap exhaustion drops the
         action from the grammar.
      3. Validation-retry — when ``build_path`` paths fail validation, retry
         the planner up to ``metadata_context_dynamic_retry`` times with
         structured error messages.
      4. Resolve via the four-way decision tree (see
         ``.claude/plans/retry-fallback-redesign.md``):
           - ``llm_paths``              — planner produced validated paths.
           - ``llm_paths_anchor_only``  — planner returned selected_paths=[]
             intentionally (or rescue ran and found nothing); lean rebuild
             using just ``{anchor}``.
           - ``bfs_selected_concepts``  — retry exhausted, but
             ``generate_fallback_paths`` enumerated rescue paths over the
             planner's ``selected_concepts``.
           - ``depth_capped_static``    — no concepts to rescue OR rescue
             raised; emit a depth-capped (``min(2, graph_depth)``)
             anchor-rooted subgraph.
           - ``empty``                  — anchor has zero reachable
             neighbors even at the depth cap; the wiring layer falls back
             to the unfiltered STATIC strings.

    On any uncaught failure inside the pipeline, returns a result with an
    ``error`` set and empty filtered_concepts/paths.
    """
    stats: Dict[str, Any] = _make_stats()
    stats["original_anchor"] = anchor
    warnings: List[str] = []

    # ---- Depth-ordering: clamp instead of raise --------------------------
    # graph_depth (the per-call detail bound) should be strictly less than
    # max_graph_depth (the outer reachability bound). When a caller passes a
    # graph_depth ≥ max_graph_depth, clamp to ``max_graph_depth - 1`` rather than
    # rejecting — be forgiving toward callers configured before the menu
    # band landed (e.g. a chain still asking for graph_depth=5 against
    # max_graph_depth=5). The floor of 1 keeps things sensible if max_graph_depth=1.
    if graph_depth >= config.max_graph_depth:
        original_graph_depth = graph_depth
        graph_depth = max(1, config.max_graph_depth - 1)
        warnings.append(
            f"graph_depth_clamped: requested {original_graph_depth}, "
            f"clamped to {graph_depth} (max_graph_depth={config.max_graph_depth}). "
            f"Raise MAX_GRAPH_DEPTH / per-chain max_graph_depth to use deeper detail."
        )
        logger.warning(
            "graph_depth (%d) >= max_graph_depth (%d); clamped to %d. Set "
            "MAX_GRAPH_DEPTH or per-chain max_graph_depth higher than the "
            "intended detail depth to opt out of clamping.",
            original_graph_depth, config.max_graph_depth, graph_depth,
        )
        stats["graph_depth_clamped_from"] = original_graph_depth
        stats["graph_depth_clamped_to"] = graph_depth

    try:
        edge_index = EdgeIndex(ontology)

        # Per-request planner state — bounds the non-build_path actions.
        expand_count = 0
        reanchor_count = 0
        # ``expand_targets`` is the set of menu-band concepts the planner has
        # requested via ``expand_to`` rounds. They are rendered as MINIMAL
        # blocks (description + connecting rels only) — not promoted to full
        # detail. See ``_build_subgraph_and_ddl`` for the connecting-chain
        # computation.
        expand_targets: Set[str] = set()
        current_anchor = anchor

        # First subgraph + DDL render.
        concepts, _predecessors, edges, compact_ddl, menu_entries = (
            _build_subgraph_and_ddl(
                question=question, anchor=current_anchor, ontology=ontology,
                edge_index=edge_index, llm=llm, config=config,
                graph_depth=graph_depth, stats=stats, warnings=warnings,
                note=note,
            )
        )

        # ---- Action loop: drive Step 1 until the planner emits build_path
        # (or until non-build_path actions are exhausted by caps).
        # The action union is narrowed each round based on remaining budget.
        # ``action_iterations`` is a hard safety bound; in practice it is
        # capped by ``_EXPAND_CAP + _REANCHOR_CAP + 1`` (each non-terminal
        # action consumes one budget unit, and build_path terminates).
        action_iterations = 0
        # Safety bound on loop iterations. Each iteration consumes either a
        # build_path (terminal), a valid/invalid expand_to (consumes one
        # expand budget unit), or a reanchor (consumes one reanchor unit).
        # The +2 leaves room for a final build_path after caps exhaust.
        max_action_iterations = _EXPAND_CAP + _REANCHOR_CAP + 2
        step1 = None
        valid_paths: List[SelectedPath] = []
        # Fix 2 of action-loop hardening: when an invalid expand_to fires,
        # we stash structured error lines here so the NEXT loop iteration's
        # planner call re-prompts via the retry path with these messages.
        pending_action_errors: Optional[List[str]] = None
        while action_iterations < max_action_iterations:
            action_iterations += 1
            allowed = _allowed_actions(
                expand_count, reanchor_count, menu_size=len(menu_entries),
            )
            stats.setdefault("allowed_actions_history", []).append(list(allowed))
            step1, valid_paths = _step1_with_validation_retries(
                llm=llm, question=question, anchor=current_anchor,
                compact_ddl=compact_ddl, edge_index=edge_index, config=config,
                graph_depth=graph_depth, stats=stats, warnings=warnings,
                note=note, allowed_actions=allowed,
                initial_action_errors=pending_action_errors,
            )
            # Consume any pending errors — they've been delivered to the LLM.
            pending_action_errors = None
            requested_action = getattr(step1, "action", "build_path") or "build_path"
            # Defensive: an LLM that names an unallowed action falls back to
            # build_path so the loop terminates rather than spinning.
            if requested_action not in allowed:
                warnings.append(
                    f"action_outside_allowed: requested={requested_action!r}, "
                    f"allowed={allowed}; defaulting to build_path"
                )
                requested_action = "build_path"
            stats.setdefault("action_history", []).append(requested_action)

            if requested_action == "expand_to":
                requested = list(getattr(step1, "expand_to", []) or [])
                # Fix 2 — three-way partition of requested targets:
                #   - valid: name is in the current menu band → promote
                #   - already_in_detail: name has a heading in ## CONCEPTS
                #     (FULL or MINIMAL) → invalid per the prompt's HARD rule
                #   - hallucinated: name appears nowhere → invalid
                detail_names = set(concepts)
                menu_names = {e.concept for e in menu_entries}
                already_in_detail = [c for c in requested if c in detail_names]
                valid_targets = [c for c in requested if c in menu_names]
                hallucinated = [
                    c for c in requested
                    if c not in detail_names and c not in menu_names
                ]
                if not valid_targets:
                    # Invalid expand_to — consume one expand budget unit,
                    # stash a structured re-prompt for the next loop iter.
                    # NEVER fall through to a deterministic pathfinder or
                    # standalone prefilter (Fix 3 deletes those branches).
                    expand_count += 1
                    stats["expand_rounds"] = expand_count
                    stats.setdefault("invalid_expand_to_history", []).append({
                        "requested": requested,
                        "already_in_detail": already_in_detail,
                        "hallucinated": hallucinated,
                    })
                    if already_in_detail:
                        err = (
                            f"expand_to: {already_in_detail!r} are already in "
                            "`## CONCEPTS` (FULL or MINIMAL blocks) and do not "
                            "need expansion. Use them directly in `build_path` "
                            "segments. If a different menu concept is genuinely "
                            "needed, pick a name from `## REACHABLE` instead."
                        )
                    else:
                        err = (
                            f"expand_to: {hallucinated!r} are not in "
                            "`## REACHABLE`. Pick a name listed there only."
                        )
                    warnings.append(err)
                    pending_action_errors = [err]
                    continue
                # Valid expand_to — fall through to the existing handler.
                promote_entries = [
                    e for e in menu_entries if e.concept in set(valid_targets)
                ]
                expand_count += 1
                stats["expand_rounds"] = expand_count
                promote_names = [e.concept for e in promote_entries]
                stats.setdefault("expanded_concepts", []).extend(promote_names)
                # Record the source breakdown for telemetry — the depth_band
                # vs prefilter_overflow split tells operators whether the
                # planner is mostly recovering from menu-band placement
                # (depth_band) or from prefilter-LLM judgment (overflow).
                round_breakdown = {
                    "depth_band": sum(
                        1 for e in promote_entries if e.source == "depth_band"
                    ),
                    "prefilter_overflow": sum(
                        1 for e in promote_entries if e.source == "prefilter_overflow"
                    ),
                }
                stats.setdefault("expand_source_breakdown", []).append(round_breakdown)
                expand_targets.update(promote_names)
                concepts, _predecessors, edges, compact_ddl, menu_entries = (
                    _build_subgraph_and_ddl(
                        question=question, anchor=current_anchor,
                        ontology=ontology, edge_index=edge_index, llm=llm,
                        config=config, graph_depth=graph_depth, stats=stats,
                        warnings=warnings, note=note,
                        expand_targets=expand_targets,
                    )
                )
                continue

            if requested_action == "reanchor":
                target = getattr(step1, "reanchor_to", None)
                visible = set(concepts) | {e.concept for e in menu_entries}
                if not target or target not in visible:
                    warnings.append(
                        f"reanchor: invalid target {target!r} not in visible "
                        f"subgraph; treating as build_path"
                    )
                    break
                reanchor_count += 1
                stats["reanchor_rounds"] = reanchor_count
                stats.setdefault("reanchor_history", []).append(
                    {"from": current_anchor, "to": target}
                )
                # Per the plan: a new anchor gets a fresh expand_to budget.
                # reanchor_count itself is monotonic so total swaps stay
                # bounded at _REANCHOR_CAP across the request.
                expand_count = 0
                expand_targets = set()
                current_anchor = target
                concepts, _predecessors, edges, compact_ddl, menu_entries = (
                    _build_subgraph_and_ddl(
                        question=question, anchor=current_anchor,
                        ontology=ontology, edge_index=edge_index, llm=llm,
                        config=config, graph_depth=graph_depth, stats=stats,
                        warnings=warnings, note=note,
                    )
                )
                continue

            # action == "build_path" — terminal.
            break

        stats["effective_anchor_after_actions"] = (
            current_anchor if current_anchor != anchor else None
        )

        # Defensive: if the action loop exited via the safety bound without
        # producing a step1, synthesize an empty one for the floor logic.
        if step1 is None:
            step1 = Step1Output()  # type: ignore[call-arg]

        # Validated transitivity overrides flow regardless of whether the
        # floor fired — they're a side channel on Step1Output.
        raw_overrides = getattr(step1, "transitivity_overrides", None) or []
        accepted_overrides = validate_overrides(raw_overrides, edge_index)
        stats["transitivity_overrides_emitted"] = len(raw_overrides)
        stats["transitivity_overrides_accepted"] = len(accepted_overrides)

        effective_anchor = current_anchor if current_anchor != anchor else None

        # Branch 1 — planner produced validated paths.
        if valid_paths:
            stats["resolved_by"] = "llm_paths"
            return _assemble_result(
                anchor=current_anchor, valid_paths=valid_paths,
                compact_ddl=compact_ddl, accepted_overrides=accepted_overrides,
                stats=stats, warnings=warnings,
                effective_anchor=effective_anchor,
            )

        # Branch 2 — anchor-only intentional. Planner returned an empty
        # selected_paths list AND validation produced no errors (because
        # there was nothing to validate). The planner is telling us no
        # joins are needed; honor it with a lean anchor-only rebuild
        # instead of falling back to the full static prompt.
        if stats.get("first_pass_valid") and not step1.selected_paths:
            stats["resolved_by"] = "llm_paths_anchor_only"
            return _assemble_result(
                anchor=current_anchor, valid_paths=[],
                compact_ddl=compact_ddl, accepted_overrides=accepted_overrides,
                stats=stats, warnings=warnings,
                effective_anchor=effective_anchor,
            )

        # Branch 3 — retry exhausted with paths-that-kept-failing. If the
        # planner left us selected_concepts, attempt the DFS rescue via
        # generate_fallback_paths (anchor -> each target, capped at
        # graph_depth). NOTE: this re-introduces the legacy
        # ``bfs_selected_concepts`` resolved_by value that Fix 3 of the
        # Action-Loop Hardening pass had removed — see
        # .claude/plans/retry-fallback-redesign.md for why we restored it.
        selected_concepts = list(getattr(step1, "selected_concepts", None) or [])
        if selected_concepts:
            try:
                rescue_paths = _bfs_paths_for_concepts(
                    anchor=current_anchor,
                    selected_concepts=selected_concepts,
                    edge_index=edge_index,
                    graph_depth=graph_depth or 1,
                )
                if rescue_paths:
                    stats["resolved_by"] = "bfs_selected_concepts"
                    stats["rescue_path_count"] = len(rescue_paths)
                    return _assemble_result(
                        anchor=current_anchor, valid_paths=rescue_paths,
                        compact_ddl=compact_ddl,
                        accepted_overrides=accepted_overrides,
                        stats=stats, warnings=warnings,
                        effective_anchor=effective_anchor,
                    )
                # Rescue ran but found zero paths despite the planner
                # naming concepts. Treat as anchor-only (lean rebuild) and
                # flag the case in stats for operator visibility — flip the
                # routing to depth_capped_static here if the user later
                # decides rescue-empty should fall through instead.
                stats["resolved_by"] = "llm_paths_anchor_only"
                stats["bfs_rescue_empty"] = True
                return _assemble_result(
                    anchor=current_anchor, valid_paths=[],
                    compact_ddl=compact_ddl,
                    accepted_overrides=accepted_overrides,
                    stats=stats, warnings=warnings,
                    effective_anchor=effective_anchor,
                )
            except Exception as rescue_exc:
                logger.warning(
                    "BFS rescue raised: %s — routing to depth_capped_static",
                    rescue_exc,
                )
                stats["bfs_rescue_failed"] = True
                warnings.append(f"bfs_rescue_error: {rescue_exc}")
                # Fall through to Branch 4.

        # Branch 4 — depth-capped dynamic rebuild. Either selected_concepts
        # was empty, or the rescue raised. Cap depth at 2 unless the
        # caller's graph_depth is already 1.
        gd = graph_depth or 1
        capped_depth = min(2, gd) if gd > 1 else 1
        return _build_depth_capped_result(
            anchor=current_anchor, edge_index=edge_index, config=config,
            capped_depth=capped_depth, compact_ddl=compact_ddl,
            accepted_overrides=accepted_overrides, stats=stats,
            warnings=warnings, effective_anchor=effective_anchor,
        )
    except Exception as exc:
        stats["metadata_context_dynamic_failed"] = True
        logger.warning("Dynamic metadata pipeline failed: %s", exc)
        return DynamicMetadataResult(
            filtered_concepts=set(),
            path_rel_keys=set(),
            validated_paths=[],
            compact_ddl="",
            stats=stats,
            warnings=warnings + [f"pipeline_error: {exc}"],
            error=str(exc),
        )


def _make_stats() -> Dict[str, Any]:
    return {
        "stage_0_ddl_stage": None,
        "stage_0_subgraph_size": 0,
        "stage_0_edge_count": 0,
        "first_pass_valid": False,
        "retry_used": False,
        "retry_attempts": 0,
        "retry_succeeded": False,
        "fallback_used": False,
        "fallback_empty": False,
        "metadata_context_dynamic_failed": False,
        # How the request resolved. Four-way decision tree (see
        # .claude/plans/retry-fallback-redesign.md):
        #   'llm_paths'             — planner emitted validated paths.
        #   'llm_paths_anchor_only' — planner intentionally returned
        #                             selected_paths=[]; lean rebuild
        #                             using only the anchor's properties.
        #   'bfs_selected_concepts' — retry exhausted; rescue DFS over
        #                             selected_concepts produced paths.
        #   'depth_capped_static'   — no selected_concepts (or rescue
        #                             raised); emit a 1- or 2-hop
        #                             anchor-rooted subgraph.
        #   'empty'                 — anchor has no neighbors at the cap;
        #                             wiring layer falls back to the
        #                             unfiltered STATIC strings.
        "resolved_by": None,
        "original_anchor": None,
        # Concept pre-filter telemetry — populated only when the pre-filter
        # engages (estimated DDL exceeds metadata_context_filter_max_tokens).
        "prefilter_used": False,
        "prefilter_input_count": 0,
        "prefilter_output_count": 0,
        "prefilter_latency_ms": 0,
        # Which trigger fired the pre-filter: 'token_overflow' | 'count_overflow'
        # | 'under_threshold' (didn't fire) | None (not checked yet).
        "prefilter_trigger": None,
        # Menu-band recovery: how many expand_to rounds the planner used
        # and which concepts it pulled back into detail.
        "expand_rounds": 0,
        "expanded_concepts": [],
        "promoted_via_expand_to": 0,
        # Action-grammar telemetry — populated by the per-request action loop.
        # ``action_history`` is the ordered list of actions the planner
        # actually emitted; ``allowed_actions_history`` is what the prompt
        # offered each round (grammar-narrowing trace). ``reanchor_*`` track
        # the monotonic anchor swap counter, and ``effective_anchor_after_actions``
        # is set when reanchor changed the FROM concept.
        "action_history": [],
        "allowed_actions_history": [],
        "reanchor_rounds": 0,
        "reanchor_history": [],
        "effective_anchor_after_actions": None,
    }


def _build_subgraph_and_ddl(
    *,
    question: str,
    anchor: str,
    ontology: Ontology,
    edge_index: EdgeIndex,
    llm,
    config: MetadataContextConfig,
    graph_depth: int,
    stats: Dict[str, Any],
    warnings: List[str],
    note: str = "",
    expand_targets: Optional[Set[str]] = None,
):
    """Step 0 — BFS to max_graph_depth + two-band split + DDL serialization.

    Returns ``(detail_concepts, predecessors, edges, compact_ddl, menu_entries)``.

    Two sources contribute to the menu band:
      - ``"depth_band"`` — concepts at ``graph_depth < hop ≤ max_graph_depth``
        from the anchor (computed by ``menu_builder.split_bands``).
      - ``"prefilter_overflow"`` — concepts in the detail band that the
        concept pre-filter demoted on size grounds (token or count
        overflow on the detail set only).

    ``expand_targets`` (Fix 4) — when the planner has used ``expand_to``
    on prior rounds, this carries the requested concept names. They are
    rendered as MINIMAL blocks in ``## CONCEPTS`` (description +
    connecting rels only — NOT full props/measures). The connecting-rel
    chain is computed here via a backward predecessors walk from each
    target to the detail frontier; the resulting ``(from, rel, to)``
    triples drive the minimal-block ``rels:`` filter inside
    ``serialize_compact_ddl``.

    Edges are kept whole — the planner sees structural rels even to menu
    concepts.
    """
    # BFS out to max_graph_depth — the outer reachability bound. The detail
    # band is hop ≤ graph_depth; the depth-band menu is the band beyond.
    concepts, predecessors, edges = retrieve_subgraph(
        anchor, edge_index, config, max_hop=config.max_graph_depth,
    )
    stats["stage_0_subgraph_size"] = len(concepts)
    stats["stage_0_edge_count"] = len(edges)

    # Compute hop distances and the depth-band split.
    hop_map = build_hop_map(anchor, edge_index, config.max_graph_depth)
    detail_in_band, depth_menu_entries = split_bands(
        hop_map, detail_depth=graph_depth, max_graph_depth=config.max_graph_depth,
    )

    # Pre-filter on the DETAIL band only — depth-band concepts are already
    # demoted by their position, no need to ask the LLM about them.
    detail_concepts: List[str] = list(detail_in_band)
    prefilter_demoted_entries: List[MenuEntry] = []
    should_fire, trigger_reason = should_trigger_concept_prefilter(
        candidate_concepts=detail_in_band, ontology=ontology, config=config,
    )
    if should_fire:
        stats["prefilter_trigger"] = trigger_reason
        pf = run_concept_prefilter(
            llm=llm,
            question=question,
            anchor=anchor,
            candidate_concepts=detail_in_band,
            ontology=ontology,
            config=config,
            note=note,
        )
        stats["prefilter_used"] = True
        stats["prefilter_input_count"] = pf.input_count
        stats["prefilter_output_count"] = pf.output_count
        stats["prefilter_latency_ms"] = pf.latency_ms
        if pf.fallback_used:
            warnings.append(
                "concept_prefilter_fallback: empty LLM output — using full detail-band set"
            )
        detail_concepts = list(pf.detail_concepts)
        for c in pf.menu_concepts:
            prefilter_demoted_entries.append(MenuEntry(
                concept=c,
                hop=hop_map.get(c, graph_depth),
                source="prefilter_overflow",
            ))
    else:
        # Stamp telemetry so "didn't fire" is distinguishable from "not yet
        # checked" (None). Integration tests assert on this value to confirm
        # the prefilter LLM call did not happen on small subgraphs.
        stats["prefilter_trigger"] = trigger_reason  # "under_threshold"

    # Combine the two menu sources (preserve the source tag — the caller
    # uses it to route expand_to correctly).
    menu_entries: List[MenuEntry] = depth_menu_entries + prefilter_demoted_entries

    # Compute the minimal-block set for any expand_targets the planner
    # accumulated on prior rounds. Each minimal block renders the
    # concept's FULL outgoing rels (description + complete outgoing edge
    # set, no props/measures/incoming) — one expand reveals the whole
    # chain in a single round. Targets that aren't actually visible in
    # the current menu band are dropped silently here (the planner's
    # invalid-target re-prompt has already been handled by the action
    # loop's three-way validator).
    expand_minimal: Set[str] = set()
    if expand_targets:
        visible_menu_names = {e.concept for e in menu_entries}
        valid_targets = {t for t in expand_targets if t in visible_menu_names}
        if valid_targets:
            expand_minimal = _compute_expand_minimal_concepts(
                expand_targets=valid_targets,
                detail_set=set(detail_concepts),
                predecessors=predecessors,
            )
            stats["promoted_via_expand_to"] = (
                stats.get("promoted_via_expand_to", 0) + len(valid_targets)
            )
        # Concepts now rendered as minimal blocks should be removed from the
        # menu band — they've earned a block (not "behind the curtain"
        # anymore). Intermediate concepts on the chain are also removed if
        # they happen to be in the menu band.
        if expand_minimal:
            menu_entries = [
                e for e in menu_entries if e.concept not in expand_minimal
            ]

    # Render: detail band gets full Compact-DDL, expand minimals get
    # minimal blocks (description + full outgoing rels), menu band gets
    # names-only.
    menu_names = [e.concept for e in menu_entries]
    compact_ddl, ddl_stage = serialize_compact_ddl(
        detail_concepts, edges, ontology, predecessors, config,
        menu_concepts=menu_names,
        expand_minimal_concepts=sorted(expand_minimal),
    )
    stats["stage_0_ddl_stage"] = ddl_stage
    stats["max_graph_depth_band_count"] = sum(
        1 for e in menu_entries if e.source == "depth_band"
    )
    stats["menu_prefilter_overflow_count"] = sum(
        1 for e in menu_entries if e.source == "prefilter_overflow"
    )
    return detail_concepts, predecessors, edges, compact_ddl, menu_entries


def _compute_expand_minimal_concepts(
    *,
    expand_targets: Set[str],
    detail_set: Set[str],
    predecessors: Dict[str, Optional[str]],
) -> Set[str]:
    """Walk predecessors backward from each expand target to the detail
    frontier; collect the concepts that need MINIMAL blocks.

    Returns ``expand_minimal_concepts`` — every concept on the backward
    chain that is NOT already in ``detail_set`` (i.e. the requested
    target plus any intermediates whose minimal block needs to render so
    the chain from detail-frontier to target is visible).

    Each minimal block is rendered with its CONCEPT'S FULL OUTGOING REL
    SET (subject to BFS truncation at ``max_graph_depth``). One expand reveals
    the full chain — no on-path/connecting filter is applied at render
    time anymore; this function only decides WHICH concepts get a
    minimal block, not WHICH edges they show.

    The walk uses BFS ``predecessors`` (already populated by
    ``retrieve_subgraph``), so it follows the path the BFS first found
    from anchor to target — one chain per target, even if multiple
    paths exist.
    """
    minimal: Set[str] = set()
    for target in expand_targets:
        current = target
        # Bound the walk by predecessors chain length (defensive — should
        # terminate at a detail concept or anchor in practice).
        steps = 0
        max_steps = len(predecessors) + 1
        while (
            current is not None
            and current not in detail_set
            and steps < max_steps
        ):
            pred = predecessors.get(current)
            if pred is None:
                # Reached the anchor (which should be in detail_set; if not,
                # this is a degenerate case — stop walking).
                break
            minimal.add(current)
            current = pred
            steps += 1
    return minimal


def _step1_with_validation_retries(
    *,
    llm,
    question: str,
    anchor: str,
    compact_ddl: str,
    edge_index: EdgeIndex,
    config: MetadataContextConfig,
    graph_depth: int,
    stats: Dict[str, Any],
    warnings: List[str],
    note: str = "",
    allowed_actions: Optional[List[str]] = None,
    initial_action_errors: Optional[List[str]] = None,
) -> tuple[Step1Output, List[SelectedPath]]:
    """Step 1 filter + bounded validation-retry loop. Returns (step1, valid_paths).

    ``allowed_actions`` narrows the planner's action union for this call
    (default: all three of build_path/expand_to/reanchor). Validation only
    runs when the planner returns ``build_path`` — non-terminal actions
    (expand_to, reanchor) have no paths to validate and return immediately.

    ``initial_action_errors`` (Fix 2 of the action-loop hardening pass) — when
    non-empty, the FIRST call to the planner uses the retry-prompt path with
    these error lines injected, rather than the plain filter path. The action
    loop sets this when a prior round emitted an invalid ``expand_to`` — the
    planner sees a structured rejection message and is re-prompted within the
    same call shape.
    """
    if initial_action_errors:
        step1 = run_step1_retry(
            llm=llm, question=question, anchor=anchor, compact_ddl=compact_ddl,
            error_lines=initial_action_errors,
            note=note, allowed_actions=allowed_actions,
        )
    else:
        step1 = run_step1_filter(
            llm=llm, question=question, anchor=anchor, compact_ddl=compact_ddl,
            note=note, allowed_actions=allowed_actions,
        )

    # Non-build_path actions have no selected_paths to validate.
    action = getattr(step1, "action", "build_path") or "build_path"
    if action != "build_path":
        return step1, []

    # Split any fork mis-packed into one path_id into separate linear paths so
    # both validation and rebuild see clean linear chains (see split_branching_paths).
    step1.selected_paths = split_branching_paths(step1.selected_paths, anchor)

    errors = validate_paths(
        step1.selected_paths,
        anchor=anchor,
        edge_index=edge_index,
        max_hop=graph_depth or 1,
        include_logic_concepts=config.include_logic_concepts,
    )
    stats["first_pass_valid"] = not errors

    retry_budget = max(0, int(config.metadata_context_dynamic_retry or 0))
    attempt = 0
    while errors and attempt < retry_budget:
        attempt += 1
        stats["retry_used"] = True
        stats["retry_attempts"] = attempt
        try:
            step1 = run_step1_retry(
                llm=llm,
                question=question,
                anchor=anchor,
                compact_ddl=compact_ddl,
                errors=errors,
                note=note,
                allowed_actions=allowed_actions,
            )
            # Validation retries are only meaningful for build_path; if the
            # planner switches to expand_to/reanchor on retry, propagate that.
            if getattr(step1, "action", "build_path") != "build_path":
                return step1, []
            step1.selected_paths = split_branching_paths(step1.selected_paths, anchor)
            errors = validate_paths(
                step1.selected_paths,
                anchor=anchor,
                edge_index=edge_index,
                max_hop=graph_depth or 1,
                include_logic_concepts=config.include_logic_concepts,
            )
        except Exception as exc:
            logger.warning("Step 1 retry attempt %d failed: %s", attempt, exc)
            warnings.append(f"retry_error_attempt_{attempt}: {exc}")
            break
    if not errors and stats["retry_used"]:
        stats["retry_succeeded"] = True

    return step1, _valid_paths(step1.selected_paths, errors)


def _bfs_paths_for_concepts(
    anchor: str,
    selected_concepts: List[str],
    edge_index: EdgeIndex,
    *,
    graph_depth: int,
) -> List[SelectedPath]:
    """Rescue path enumeration for retry-exhausted planner outcomes.

    Despite the name (kept for telemetry symmetry with the
    ``bfs_selected_concepts`` resolved_by value), the underlying enumeration
    is DFS via ``generate_fallback_paths`` — simple paths from anchor to each
    target concept, bounded by ``graph_depth`` segments and the existing
    safety cap. Returns an empty list when no paths exist within the cap.
    """
    return generate_fallback_paths(
        anchor=anchor,
        selected_concepts=selected_concepts,
        edge_index=edge_index,
        length_cap=graph_depth,
    )


def _edge_to_singleton_path(edge: EdgeMeta, idx: int) -> SelectedPath:
    """Wrap a single edge into a one-segment SelectedPath. Used to feed the
    depth-capped subgraph through ``_assemble_result`` which expects paths."""
    return SelectedPath(
        path_id=f"D{idx + 1}",
        purpose=f"depth-capped edge {edge.from_concept}->{edge.to_concept}",
        segments=[
            PathSegment(
                **{
                    "from": edge.from_concept,
                    "rel": edge.relationship_name,
                    "to": edge.to_concept,
                }
            )
        ],
        is_recursive=False,
    )


def _build_depth_capped_result(
    *,
    anchor: str,
    edge_index: EdgeIndex,
    config: MetadataContextConfig,
    capped_depth: int,
    compact_ddl: str,
    accepted_overrides: List[TransitivityOverride],
    stats: Dict[str, Any],
    warnings: List[str],
    effective_anchor: Optional[str] = None,
) -> DynamicMetadataResult:
    """Last-resort dynamic rebuild bounded by ``capped_depth`` hops from anchor.

    Fires when the BFS rescue cannot run (no ``selected_concepts``) or
    raises. Synthesizes single-segment paths from the retained edges so
    downstream assembly walks every relationship in the depth-capped
    neighborhood. If even this produces nothing (anchor has zero outbound
    edges), routes to ``resolved_by="empty"`` and lets the wiring layer
    fall back to the unfiltered static strings.
    """
    concepts, _preds, edges = retrieve_subgraph(
        anchor=anchor, edge_index=edge_index, config=config, max_hop=capped_depth,
    )
    if not edges:
        stats["resolved_by"] = "empty"
        stats["fallback_empty"] = True
        warnings.append(
            f"depth_capped_static found no edges within {capped_depth} hop(s) "
            f"of {anchor!r}; routing to STATIC fallback"
        )
        return _assemble_result(
            anchor=anchor, valid_paths=[], compact_ddl=compact_ddl,
            accepted_overrides=accepted_overrides, stats=stats, warnings=warnings,
            effective_anchor=effective_anchor,
        )
    synthetic_paths = [
        _edge_to_singleton_path(e, idx) for idx, e in enumerate(edges)
    ]
    stats["resolved_by"] = "depth_capped_static"
    stats["depth_capped_at"] = capped_depth
    stats["depth_capped_concept_count"] = len(concepts)
    stats["depth_capped_edge_count"] = len(edges)
    return _assemble_result(
        anchor=anchor, valid_paths=synthetic_paths, compact_ddl=compact_ddl,
        accepted_overrides=accepted_overrides, stats=stats, warnings=warnings,
        effective_anchor=effective_anchor,
    )


def _assemble_result(
    *,
    anchor: str,
    valid_paths: List[SelectedPath],
    compact_ddl: str,
    accepted_overrides: List[TransitivityOverride],
    stats: Dict[str, Any],
    warnings: List[str],
    effective_anchor: Optional[str] = None,
) -> DynamicMetadataResult:
    filtered_concepts = collect_path_concepts(valid_paths) if valid_paths else set()
    filtered_concepts.add(anchor)
    return DynamicMetadataResult(
        filtered_concepts=filtered_concepts,
        path_rel_keys=collect_path_relationships(valid_paths),
        validated_paths=valid_paths,
        compact_ddl=compact_ddl,
        stats=stats,
        warnings=warnings,
        error=None,
        accepted_overrides=accepted_overrides,
        effective_anchor=effective_anchor,
    )


def _valid_paths(
    paths: List[SelectedPath],
    errors: List[ValidationError],
) -> List[SelectedPath]:
    bad = {e.path_id for e in errors}
    return [p for p in paths if p.path_id not in bad]
