"""Type definitions for the dynamic metadata-context pipeline.

EdgeMeta is a derived view over Plan 1's RelationshipMeta — it bundles the
cardinality + description + bookkeeping the BFS / DDL / validator layers need.
SelectedPath / PathSegment are pydantic models so they can be the LLM filter's
output schema (Step 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

try:
    from pydantic import BaseModel, ConfigDict, Field
    _PYDANTIC_AVAILABLE = True
except Exception:  # pragma: no cover — pydantic is a hard dep but stay defensive
    _PYDANTIC_AVAILABLE = False
    BaseModel = object  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class EdgeMeta:
    """A single relationship edge as seen by the BFS / validator / DDL layers."""

    from_concept: str
    relationship_name: str
    to_concept: str
    transitivity: int
    is_mtm: bool
    is_inverse: bool
    cardinality: str            # 'N:M' | 'N:1' | '1:N' | '1:1'
    description: Optional[str]
    is_self_ref: bool


@dataclass(frozen=True)
class ValidationError:
    """Structured error from validate_paths(); fed into the retry prompt."""

    path_id: str
    segment_index: int          # -1 for path-level errors
    reason_code: str
    detail: str


# ---- Pydantic models for Step 1 LLM output -------------------------------

if _PYDANTIC_AVAILABLE:

    class PathSegment(BaseModel):
        """One hop within a SelectedPath. Field aliases match the LLM JSON schema."""

        model_config = ConfigDict(populate_by_name=True)
        from_concept: str = Field(alias="from")
        relationship_name: str = Field(alias="rel")
        to_concept: str = Field(alias="to")
        # Waypoint hint — when True, the segment's ``to`` concept is a routing
        # waypoint whose data the user does NOT expect in the result. Consulted
        # only by the rebuild's threshold-gated waypoint filter when the path-
        # selection prompt was served with full visibility (no cascade trimming).
        # Default False is backward-compat: legacy responses omit it.
        is_intermediate: bool = False
        # Populated by the validator at runtime when sub-concept expansion fires
        # (see ontology_context/validator.py). Not part of the LLM schema.
        expanded_sub_concepts: list[str] = Field(default_factory=list)
        modeled_target: Optional[str] = None

    class SelectedPath(BaseModel):
        path_id: str
        purpose: str = ""
        segments: list[PathSegment]
        # Informational only since Plan 2 update — no longer validated.
        is_recursive: bool = False

    class TransitivityOverride(BaseModel):
        """LLM-emitted override of a transitive relationship's depth.

        Applied per (rel, target) pair at the rebuild step: every occurrence
        of `<rel>[<target>*<old>]` in the SQL-gen context strings is rewritten
        to `<rel>[<target>*<level>]`.
        """
        rel: str
        target: str
        level: int

    class Step1Output(BaseModel):
        # Action grammar — the planner emits EXACTLY ONE action per response:
        #   "build_path"  → use selected_paths + selected_concepts + overrides
        #   "expand_to"   → use expand_to (list of menu-band names to promote)
        #   "reanchor"    → use reanchor_to (single concept name to swap to)
        # Default ``build_path`` preserves backward compatibility with legacy
        # responses that omit the field. (``clarify`` deliberately omitted —
        # chat pipelines handle ambiguity via re-ask, not an in-band action.)
        action: str = "build_path"

        # build_path payload — used only when action == "build_path"
        selected_concepts: list[str] = Field(default_factory=list)
        selected_properties: list[str] = Field(default_factory=list)
        selected_measures: list[str] = Field(default_factory=list)
        selected_paths: list[SelectedPath] = Field(default_factory=list)
        transitivity_overrides: list[TransitivityOverride] = Field(default_factory=list)

        # expand_to payload — used only when action == "expand_to". Lists
        # ## REACHABLE band concepts the planner needs promoted to detail.
        # The orchestrator re-renders the DDL with those concepts pinned to
        # the detail band and re-calls Step 1. Per-request count is bounded
        # by ``_EXPAND_CAP`` and enforced via grammar narrowing — when the
        # cap is hit, the next prompt omits expand_to from the allowed
        # actions so the LLM literally cannot request it.
        expand_to: list[str] = Field(default_factory=list)

        # reanchor payload — used only when action == "reanchor". Names a
        # concept in the current subgraph the planner judges to be a better
        # SQL FROM root for the question. The orchestrator restarts the
        # pipeline with the new anchor. Per-request count is bounded by
        # ``_REANCHOR_CAP`` (monotonic across the request) and enforced via
        # grammar narrowing on subsequent calls.
        reanchor_to: Optional[str] = None

else:  # pragma: no cover

    @dataclass
    class PathSegment:
        from_concept: str
        relationship_name: str
        to_concept: str
        is_intermediate: bool = False
        expanded_sub_concepts: list = field(default_factory=list)
        modeled_target: Optional[str] = None

    @dataclass
    class SelectedPath:
        path_id: str
        segments: list
        purpose: str = ""
        is_recursive: bool = False

    @dataclass
    class TransitivityOverride:
        rel: str
        target: str
        level: int

    @dataclass
    class Step1Output:
        action: str = "build_path"
        selected_concepts: list = field(default_factory=list)
        selected_properties: list = field(default_factory=list)
        selected_measures: list = field(default_factory=list)
        selected_paths: list = field(default_factory=list)
        transitivity_overrides: list = field(default_factory=list)
        expand_to: list = field(default_factory=list)
        reanchor_to: Optional[str] = None
