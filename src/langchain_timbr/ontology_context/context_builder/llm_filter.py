"""Step 1 — LLM filter + path inference. One LLM call, optional retry on validation errors.

Returns a parsed Step1Output. The caller (the orchestrator) drives validate→retry→
fallback decisions; this module only handles the LLM round-trip + JSON parsing.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, List, Optional

from .metadata_types import Step1Output, ValidationError


def run_step1_filter(
    *,
    llm,
    question: str,
    anchor: str,
    compact_ddl: str,
    timeout: int = 60,
    note: str = "",
    allowed_actions: Optional[List[str]] = None,
) -> Step1Output:
    """Invoke the LLM once with the filter prompt and parse the JSON output.

    ``allowed_actions`` (default: all three) narrows the documented action
    union for this round. The orchestrator passes a shorter list when a cap
    has been exhausted — the LLM is shown only what it can still request
    so capped actions are structurally un-emittable (grammar narrowing).

    ``note`` carries conversation memory + caller notes (Additional Notes
    block); passed verbatim to the prompt builder.

    Raises:
        ValueError: if the LLM output cannot be parsed as Step1Output.
    """
    from .prompts import build_filter_messages

    messages = build_filter_messages(
        question=question, anchor=anchor, compact_ddl=compact_ddl, note=note,
        allowed_actions=allowed_actions,
    )
    raw = _invoke_llm(llm, messages, timeout=timeout)
    return _parse_step1(raw)


def run_step1_retry(
    *,
    llm,
    question: str,
    anchor: str,
    compact_ddl: str,
    errors: Optional[Iterable[ValidationError]] = None,
    error_lines: Optional[List[str]] = None,
    timeout: int = 60,
    note: str = "",
    allowed_actions: Optional[List[str]] = None,
) -> Step1Output:
    """Retry with error messages injected from the previous round.

    Two error-source flavors:
      - ``errors`` (default): a list of ``ValidationError`` objects produced
        by ``validate_paths``. Rendered via ``_format_errors_for_retry``.
      - ``error_lines`` (override): pre-formatted strings. Used by the action
        loop when re-prompting the planner about an invalid ``expand_to``
        (e.g. target already in ``## CONCEPTS`` or hallucinated). Bypasses
        the ValidationError formatter and uses the provided strings as-is.

    Exactly one of ``errors`` / ``error_lines`` should be non-empty; when
    both are provided the ``error_lines`` override wins. Same
    ``allowed_actions`` semantics as ``run_step1_filter``.
    """
    from .prompts import build_retry_messages

    if error_lines is None:
        error_lines = _format_errors_for_retry(errors or [])
    messages = build_retry_messages(
        question=question,
        anchor=anchor,
        compact_ddl=compact_ddl,
        error_lines=error_lines,
        note=note,
        allowed_actions=allowed_actions,
    )
    raw = _invoke_llm(llm, messages, timeout=timeout)
    return _parse_step1(raw)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _format_errors_for_retry(errors: Iterable[ValidationError]) -> List[str]:
    out: List[str] = []
    for e in errors:
        seg_part = f"segment {e.segment_index}" if e.segment_index >= 0 else "path-level"
        out.append(f"Path {e.path_id}, {seg_part}: {e.reason_code} — {e.detail}")
    return out


def _invoke_llm(llm, messages: List[dict], *, timeout: int) -> str:
    """Invoke the LLM. Accepts a langchain BaseChatModel or anything with .invoke()."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    role_to_cls = {
        "system": SystemMessage,
        "user": HumanMessage,
        "human": HumanMessage,
        "assistant": AIMessage,
    }
    langchain_messages = [
        role_to_cls.get(m["role"], HumanMessage)(content=m["content"])
        for m in messages
    ]

    # Use the same timeout helper the rest of the codebase uses where possible.
    try:
        from ...utils.timbr_llm_utils import _call_llm_with_timeout
        response = _call_llm_with_timeout(llm, langchain_messages, timeout=timeout)
    except Exception:
        response = llm.invoke(langchain_messages)

    content = getattr(response, "content", response)
    if isinstance(content, list):
        # Some LLM wrappers return content as a list of parts; join text parts.
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_step1(raw: str) -> Step1Output:
    text = raw.strip()
    if not text:
        raise ValueError("Step 1 LLM returned empty output")

    # Strip ```json ... ``` fences if present.
    fence_match = _JSON_FENCE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    # Locate the outermost JSON object if there's surrounding text.
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Step 1 output is not valid JSON: {raw!r}")
        text = text[start:end + 1]

    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Step 1 JSON parse failed: {exc.msg}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"Step 1 output must be a JSON object, got {type(parsed).__name__}")

    # Validate via pydantic if available; otherwise build the dataclass manually.
    try:
        return Step1Output.model_validate(parsed)  # type: ignore[attr-defined]
    except AttributeError:
        # Pydantic v1 or the dataclass fallback path in metadata_types
        return Step1Output(**parsed)  # type: ignore[call-arg]
