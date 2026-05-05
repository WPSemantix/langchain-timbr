"""Token-budget trimming for technical context annotations.

Operates on STRUCTURED ColumnPayloads (not formatted strings).
Reduces K per column (200→100→50→20→10→5) before dropping any column.

Two-tier budget:
- max_tokens (soft): trigger K reduction per column
- safety_ceiling (hard): trigger column dropping as last resort
"""

from __future__ import annotations

import logging

from ..config import TechnicalContextConfig
from ..types import ColumnPayload, ColumnRef

logger = logging.getLogger(__name__)

# Protected format_hints — never trimmed, never dropped
_PROTECTED_HINTS = frozenset({"all", "min_max", "name_only", "count_only", "boolean"})


def trim_to_budget(
    payloads: dict[str, ColumnPayload],
    column_refs: dict[str, ColumnRef],
    matched_keys: set[str],
    config: TechnicalContextConfig,
) -> dict[str, ColumnPayload]:
    """Trim per-column value payloads to fit within token budget.

    Operates on STRUCTURED PAYLOADS, not formatted strings.
    Reduces K per column (trim_sequence) before dropping any column.

    PROTECTED COLUMNS (never trimmed, never dropped):
    - Columns in matched_keys (had at least one match)
    - Columns where format_hint in {all, min_max, name_only, count_only, boolean}

    TRIMMABLE COLUMNS:
    - Only those with format_hint == "top_k" AND not in matched_keys

    PHASE 1 — Reduce K per column:
    Walk trim_sequence. For each step k, reduce trimmable columns
    (highest band first, then highest cardinality) until under max_tokens.

    PHASE 2 — Check safety_ceiling:
    If total <= safety_ceiling after Phase 1, return as-is.

    PHASE 3 — Drop columns entirely:
    Replace lowest-priority non-protected columns with None until under safety_ceiling.

    Args:
        payloads: Column name -> ColumnPayload (structured, pre-format).
        column_refs: Column name -> ColumnRef (for priority_band, distinct_count).
        matched_keys: Column names that had at least one match.
        config: Configuration with max_tokens, safety_ceiling, trim_sequence.

    Returns:
        Modified payloads dict (may have fewer entries or reduced values).
    """
    if not payloads:
        return payloads

    # Quick check: if already within soft budget, no trimming needed
    total = _estimate_total_tokens(payloads)
    if total <= config.max_tokens:
        return payloads

    # Identify trimmable columns: top_k hint AND not matched
    trimmable = _get_trimmable_sorted(payloads, column_refs, matched_keys)

    # PHASE 1: Reduce K per column through trim_sequence
    # Check budget once per K level (not per column) to reduce tiktoken calls.
    for k in config.trim_sequence:
        changed = False
        for col_name in trimmable:
            payload = payloads.get(col_name)
            if payload is None:
                continue
            if len(payload.values) > k:
                # Simple slicing preserves matched values at front
                # (assembly guarantees matched-first ordering)
                payload.values = payload.values[:k]
                changed = True

        if changed:
            total = _estimate_total_tokens(payloads)
            if total <= config.max_tokens:
                return payloads

    # After full trim_sequence, recompute
    total = _estimate_total_tokens(payloads)
    if total <= config.max_tokens:
        return payloads

    # PHASE 2: Accept if under safety_ceiling
    if total <= config.safety_ceiling:
        return payloads

    # PHASE 3: Replace columns with name_only fallback (lowest priority first)
    # Preserves column visibility for the LLM while dropping all values.
    droppable = _get_trimmable_sorted(payloads, column_refs, matched_keys)
    for col_name in droppable:
        if col_name not in payloads:
            continue
        payloads[col_name] = ColumnPayload(
            format_hint="name_only",
            values=[],
            distinct_count=payloads[col_name].distinct_count,
        )
        total = _estimate_total_tokens(payloads)
        if total <= config.safety_ceiling:
            break

    if total > config.safety_ceiling:
        logger.warning(
            "Could not trim below safety_ceiling (%d tokens estimated, ceiling=%d). "
            "All droppable columns exhausted.",
            total, config.safety_ceiling,
        )

    return payloads


def _is_protected(col_name: str, payload: ColumnPayload, matched_keys: set[str]) -> bool:
    """Check if a column is protected from trimming/dropping."""
    if col_name in matched_keys:
        return True
    return payload.format_hint in _PROTECTED_HINTS


def _get_trimmable_sorted(
    payloads: dict[str, ColumnPayload],
    column_refs: dict[str, ColumnRef],
    matched_keys: set[str],
) -> list[str]:
    """Get trimmable column names sorted by: priority_band DESC, distinct_count DESC.

    Highest band (lowest priority) trimmed first.
    Within same band, highest cardinality trimmed first (loses less per K reduction).
    """
    trimmable = [
        name for name, payload in payloads.items()
        if not _is_protected(name, payload, matched_keys)
    ]
    trimmable.sort(
        key=lambda name: (
            -(column_refs[name].priority_band if name in column_refs else 5),
            -(payloads[name].distinct_count if payloads[name].distinct_count > 0 else 0),
        ),
    )
    return trimmable


def _estimate_total_tokens(payloads: dict[str, ColumnPayload]) -> int:
    """Estimate total tokens from all payloads using tiktoken (cl100k_base).

    Falls back to chars/4 if tiktoken is unavailable.
    """
    total_text = " ".join(
        _render_payload_text(p) for p in payloads.values()
    )
    if not total_text:
        return 0

    enc = _get_encoding()
    if enc is not None:
        return len(enc.encode(total_text))
    # Fallback: chars / 4
    return len(total_text) // 4


def _get_encoding():
    """Get tiktoken encoding, cached. Returns None if unavailable."""
    if not hasattr(_get_encoding, "_enc"):
        try:
            import tiktoken
            _get_encoding._enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _get_encoding._enc = None
    return _get_encoding._enc


def _render_payload_text(payload: ColumnPayload) -> str:
    """Render payload to representative text for token counting."""
    hint = payload.format_hint

    if hint == "name_only":
        if payload.values:
            formatted = [f"'{v}'" for v in payload.values]
            return f"matched values from prompt: [{', '.join(formatted)}]"
        return ""

    if hint == "count_only":
        if payload.distinct_count > 0:
            return f"({payload.distinct_count} distinct values)"
        return ""

    if hint == "min_max":
        if payload.min_value is not None and payload.max_value is not None:
            return f"{payload.range_label}: {payload.min_value} to {payload.max_value}"
        return ""

    if hint == "boolean":
        if payload.values:
            return f"values: [{', '.join(payload.values)}]"
        return ""

    # top_k or all: known values list
    if not payload.values:
        return ""
    formatted = [f"'{v}'" for v in payload.values]
    result = f"known values: [{', '.join(formatted)}]"
    if payload.distinct_count > 0 and payload.distinct_count > len(payload.values):
        result += f" ({payload.distinct_count} distinct total)"
    return result
