"""Concept pre-filter — LLM-driven narrowing of the candidate concept set.

Engages between ``retrieve_subgraph`` and ``serialize_compact_ddl`` when the
estimated full-output DDL would exceed ``metadata_context_filter_max_tokens``. The pre-filter
asks the LLM "which of these concepts is your question actually about?" and
narrows the concept list before serialization.

Public entry point: ``run_concept_prefilter``. Token estimation: ``estimate_full_ddl_tokens``.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from ..ontology.graph import Ontology
from .metadata_config import MetadataContextConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def _encode_count(text: str) -> int:
    """cl100k_base token count with a length-based fallback when tiktoken is unavailable."""
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, len(text) // 4)


def estimate_full_ddl_tokens(
    concepts: Sequence[str],
    ontology: Ontology,
) -> int:
    """Rough estimate of full-output Compact DDL cost for ``concepts``.

    Constants are tokens-per-line approximations calibrated against actual
    cl100k_base encoding of small samples. Used as the trigger for the
    concept pre-filter — if this exceeds ``metadata_context_filter_max_tokens``, narrow
    the concept set before serializing.
    """
    tokens = 0
    for c in concepts:
        try:
            meta = ontology.get_concept_metadata(c)
        except Exception:
            tokens += 20
            continue
        tokens += 15
        tokens += 4 * len(meta.properties) + 3
        tokens += 4 * len(meta.measures) + 2
        tokens += 8 * len(meta.relationships) + 2
    tokens += 100
    return tokens


# ---------------------------------------------------------------------------
# Candidate-rendering helpers
# ---------------------------------------------------------------------------

@dataclass
class _Candidate:
    """Mutable view of a candidate concept used during prompt construction."""
    name: str
    description: str


def _gather_candidates(
    concepts: Sequence[str],
    ontology: Ontology,
) -> List[_Candidate]:
    out: List[_Candidate] = []
    for name in concepts:
        try:
            meta = ontology.get_concept_metadata(name)
            desc = (meta.description or "").strip()
        except Exception:
            desc = ""
        out.append(_Candidate(name=name, description=desc))
    return out


_WORD_BOUNDARY = re.compile(r"\s+")


def truncate_to_tokens(text: str, target: int) -> str:
    """Truncate ``text`` to roughly ``target`` tokens, respecting word boundaries.

    Never cuts mid-word. If the first word alone exceeds the target, returns
    the empty string (the description is dropped) rather than mid-word output.
    """
    if not text or target <= 0:
        return ""
    if _encode_count(text) <= target:
        return text
    words = _WORD_BOUNDARY.split(text.strip())
    out_parts: List[str] = []
    running = 0
    for w in words:
        if not w:
            continue
        next_token_count = _encode_count(" " + w if out_parts else w)
        if running + next_token_count > target:
            break
        out_parts.append(w)
        running += next_token_count
    return " ".join(out_parts)


def render_with_descriptions(candidates: Iterable[_Candidate]) -> str:
    """Render candidates as ``- <name>: <description>`` lines.

    Empty descriptions render as ``- <name>`` (no trailing colon).
    """
    lines: List[str] = []
    for c in candidates:
        if c.description:
            lines.append(f"- {c.name}: {c.description}")
        else:
            lines.append(f"- {c.name}")
    return "\n".join(lines)


def render_names_only(candidates: Iterable[_Candidate]) -> str:
    """Render candidates as bare names (one per line)."""
    return "\n".join(f"- {c.name}" for c in candidates)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_FIXED_OVERHEAD_TOKENS = 400
_NAME_OVERHEAD_TOKENS_PER_CANDIDATE = 2
_SOFT_CAP_TOKENS = 25
_MIN_TRUNCATED_DESC_TOKENS = 5
_MIN_REDISTRIBUTED_DESC_TOKENS = 10
_PATHOLOGICAL_CANDIDATE_THRESHOLD = 500


def apply_truncation_and_render(
    candidates: List[_Candidate],
    desc_tokens: dict,
    available: int,
) -> str:
    """Apply the soft-cap-with-redistribution strategy and render with descriptions.

    Mutates ``candidates`` in place (description fields are truncated)."""
    under_cap = [c for c in candidates if desc_tokens[c.name] <= _SOFT_CAP_TOKENS]
    over_cap = [c for c in candidates if desc_tokens[c.name] > _SOFT_CAP_TOKENS]
    used_by_under = sum(desc_tokens[c.name] for c in under_cap)
    remaining = available - used_by_under

    if not over_cap:
        # All under the soft cap but total still over — uniform proportional reduction.
        total = sum(desc_tokens.values()) or 1
        ratio = available / total
        for c in candidates:
            target = max(_MIN_TRUNCATED_DESC_TOKENS, int(desc_tokens[c.name] * ratio))
            c.description = truncate_to_tokens(c.description, target)
    elif remaining <= 0:
        # No budget left for over-cap concepts — drop their descriptions.
        for c in over_cap:
            c.description = ""
    else:
        per_over_cap = remaining // len(over_cap)
        if per_over_cap < _MIN_REDISTRIBUTED_DESC_TOKENS:
            for c in over_cap:
                c.description = ""
        else:
            for c in over_cap:
                c.description = truncate_to_tokens(c.description, per_over_cap)

    return render_with_descriptions(candidates)


def build_prefilter_prompt(
    candidates: List[_Candidate],
    max_prompt_tokens: int,
) -> Tuple[str, bool]:
    """Build the CANDIDATE CONCEPTS block, deciding whether to include descriptions.

    Returns ``(candidates_block, with_descriptions)`` so the caller can pick
    the right user-template (names-only vs. with-descriptions) when building
    the LLM messages.
    """
    bare_concept_tokens = len(candidates) * _NAME_OVERHEAD_TOKENS_PER_CANDIDATE
    available_for_descriptions = (
        max_prompt_tokens - _FIXED_OVERHEAD_TOKENS - bare_concept_tokens
    )

    if available_for_descriptions <= 0:
        if len(candidates) > _PATHOLOGICAL_CANDIDATE_THRESHOLD:
            logger.warning(
                "Concept pre-filter has %d candidates (names only). "
                "Estimated token count: ~%d. Latency will be elevated.",
                len(candidates),
                bare_concept_tokens + _FIXED_OVERHEAD_TOKENS,
            )
        return render_names_only(candidates), False

    desc_token_counts = {
        c.name: _encode_count(c.description) if c.description else 0
        for c in candidates
    }
    total_desc_tokens = sum(desc_token_counts.values())

    if total_desc_tokens <= available_for_descriptions:
        return render_with_descriptions(candidates), True

    block = apply_truncation_and_render(
        candidates, desc_token_counts, available_for_descriptions,
    )
    return block, True


# ---------------------------------------------------------------------------
# LLM round-trip
# ---------------------------------------------------------------------------

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_relevant_concepts(raw: str) -> List[str]:
    text = raw.strip()
    if not text:
        return []
    fence_match = _JSON_FENCE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return []
        text = text[start:end + 1]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, dict):
        return []
    raw_list = parsed.get("relevant_concepts") or []
    if not isinstance(raw_list, list):
        return []
    out: List[str] = []
    seen = set()
    for item in raw_list:
        if isinstance(item, str):
            name = item.strip()
            if name and name not in seen:
                seen.add(name)
                out.append(name)
    return out


def _invoke_prefilter_llm(llm, messages: List[dict], *, timeout: int) -> str:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    role_to_cls = {
        "system": SystemMessage,
        "user": HumanMessage,
        "human": HumanMessage,
        "assistant": AIMessage,
    }
    lc_messages = [
        role_to_cls.get(m["role"], HumanMessage)(content=m["content"])
        for m in messages
    ]
    try:
        from ...utils.timbr_llm_utils import _call_llm_with_timeout
        response = _call_llm_with_timeout(llm, lc_messages, timeout=timeout)
    except Exception:
        response = llm.invoke(lc_messages)

    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


@dataclass
class PrefilterResult:
    """Return envelope from ``run_concept_prefilter``.

    The pre-filter is non-destructive — every input candidate ends up in
    EITHER ``detail_concepts`` (full Compact-DDL block) OR ``menu_concepts``
    (names-only band the LLM can recover via ``expand_to``). Together they
    always equal the input candidate set; a concept never silently disappears.

    ``filtered_concepts`` is the legacy alias for ``detail_concepts`` retained
    for downstream callers that haven't switched to the demote semantics yet.
    """
    detail_concepts: List[str]
    menu_concepts: List[str]
    input_count: int
    output_count: int
    latency_ms: int
    with_descriptions: bool
    fallback_used: bool = False

    @property
    def filtered_concepts(self) -> List[str]:
        """Legacy alias for ``detail_concepts`` — preserved for callers that
        haven't migrated to the (detail, menu) split."""
        return self.detail_concepts


def run_concept_prefilter(
    *,
    llm,
    question: str,
    anchor: str,
    candidate_concepts: Sequence[str],
    ontology: Ontology,
    config: MetadataContextConfig,
    timeout: int = 60,
    note: str = "",
) -> PrefilterResult:
    """Run the pre-filter LLM call to split ``candidate_concepts`` into a
    detail band (full DDL render) + a menu band (names-only, recoverable
    via ``expand_to``).

    Guarantees:
      - The anchor is ALWAYS in ``detail_concepts`` (auto-promoted if the LLM
        dropped it).
      - Every input candidate ends up in exactly one of ``detail_concepts``
        or ``menu_concepts`` — no silent disappearance.
      - Hallucinated names (not in the input set) are discarded with a log.
      - Empty LLM result falls back to "everything in detail" rather than
        producing a zero-detail downstream DDL.
    """
    from .prompts import build_prefilter_messages

    started = time.perf_counter()
    candidates_view = _gather_candidates(candidate_concepts, ontology)
    candidates_block, with_descriptions = build_prefilter_prompt(
        candidates_view, max_prompt_tokens=config.max_concept_prefilter_token,
    )
    messages = build_prefilter_messages(
        question=question,
        anchor=anchor,
        candidates_block=candidates_block,
        with_descriptions=with_descriptions,
        note=note,
    )

    candidate_set = {c.name for c in candidates_view}
    try:
        raw = _invoke_prefilter_llm(llm, messages, timeout=timeout)
        names = _parse_relevant_concepts(raw)
    except Exception as exc:
        logger.warning("Concept pre-filter LLM call failed: %s", exc)
        names = []

    # Validate output: drop hallucinated names.
    kept: List[str] = []
    seen = set()
    for name in names:
        if name in candidate_set and name not in seen:
            kept.append(name)
            seen.add(name)
    dropped = [n for n in names if n not in candidate_set]
    if dropped:
        logger.warning(
            "Concept pre-filter dropped %d hallucinated concept(s): %s",
            len(dropped), dropped,
        )

    fallback_used = False
    if not kept:
        # LLM returned nothing valid — defensive fallback to "everything in
        # detail" rather than emitting a zero-detail DDL.
        logger.warning(
            "Concept pre-filter returned no valid concepts — falling back to "
            "full candidate set in detail (%d concepts).", len(candidate_set),
        )
        kept = list(candidate_concepts)
        fallback_used = True
    elif anchor in candidate_set and anchor not in seen:
        kept.insert(0, anchor)
        seen.add(anchor)
        logger.warning(
            "Concept pre-filter omitted anchor %r — auto-promoted to detail.",
            anchor,
        )

    # Demote, don't drop — every candidate ends up in detail OR menu.
    detail_set = set(kept)
    menu_concepts = [c for c in candidate_concepts if c not in detail_set]

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return PrefilterResult(
        detail_concepts=kept,
        menu_concepts=menu_concepts,
        input_count=len(candidate_concepts),
        output_count=len(kept),
        latency_ms=elapsed_ms,
        with_descriptions=with_descriptions,
        fallback_used=fallback_used,
    )


def should_trigger_concept_prefilter(
    *,
    candidate_concepts: Sequence[str],
    ontology: Ontology,
    config: MetadataContextConfig,
) -> tuple[bool, str]:
    """Return (should_fire, reason) for the two prefilter triggers.

    Two independent reasons fire the pre-filter:
      1. ``token_overflow`` — estimated full DDL exceeds ``metadata_context_filter_max_tokens``.
      2. ``count_overflow`` — detail-band concept count would meet/exceed
         ``max_detail_concepts``, independent of token size. Rationale:
         lost-in-the-middle path-selection degradation kicks in well before
         the token budget on dense ontologies.

    Either trigger fires the pre-filter; the overflow concepts demote into
    the menu band (never silently disappear). The caller should fire when
    this returns ``(True, ...)``.
    """
    if len(candidate_concepts) >= max(1, config.max_detail_concepts):
        return True, "count_overflow"
    if estimate_full_ddl_tokens(candidate_concepts, ontology) > config.metadata_context_filter_max_tokens_hard_ceiling:
        return True, "token_overflow"
    return False, "under_threshold"


__all__ = [
    "PrefilterResult",
    "apply_truncation_and_render",
    "build_prefilter_prompt",
    "estimate_full_ddl_tokens",
    "render_names_only",
    "render_with_descriptions",
    "run_concept_prefilter",
    "should_trigger_concept_prefilter",
    "truncate_to_tokens",
]
