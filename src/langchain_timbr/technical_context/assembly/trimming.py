"""Token-budget trimming for technical context annotations.

Operates on STRUCTURED ColumnPayloads (not formatted strings).
Searches for the largest per-column value cap that fits the budget before giving up
any column's values.

Two-tier budget:
- max_tokens (soft): the cap the search aims at
- safety_ceiling (hard): the fallback target when even one value per column misses the
  soft budget, and the bar the last-resort degradation has to clear
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
    Searches for the largest per-column value cap that fits before any column gives up
    its values, so every budget increase shows up in the output.

    PROTECTED COLUMNS (never trimmed, never dropped):
    - Columns in matched_keys (had at least one match)
    - Columns where format_hint in {all, min_max, name_only, count_only, boolean}

    TRIMMABLE COLUMNS:
    - Only those with format_hint == "top_k" AND not in matched_keys

    PHASE 1 — Largest uniform cap that fits max_tokens:
    Binary search the per-column cap. Uniform across trimmable columns: the priority
    order decides who is sacrificed in Phase 3, not who keeps more values here.

    PHASE 2 — One value per column:
    Reached when the budget is unreachable — even a single value each misses it, or the
    protected columns alone exceed it. Emitting the minimum beats both emitting more than
    was asked for and gutting the context. Accepted if it clears safety_ceiling.

    PHASE 3 — Give up values, lowest priority first:
    With every trimmable column already down to one value and still over the hard cap,
    degrade the lowest-priority columns — first to their distinct count alone, then to
    the bare column name — until the total clears the cap.

    Args:
        payloads: Column name -> ColumnPayload (structured, pre-format).
        column_refs: Column name -> ColumnRef (for priority_band, distinct_count).
        matched_keys: Column names that had at least one match.
        config: Configuration with max_tokens and safety_ceiling.

    Returns:
        Modified payloads dict (values reduced, and in Phase 3 some payloads replaced).
    """
    if not payloads:
        return payloads

    # Quick check: if already within soft budget, no trimming needed
    if _estimate_total_tokens(payloads) <= config.max_tokens:
        return payloads

    # Identify trimmable columns: top_k hint AND not matched. Lowest priority first.
    trimmable = _get_trimmable_sorted(payloads, column_refs, matched_keys)
    if not trimmable:
        return payloads

    # The search moves the cap both up and down, so it cannot slice in place the way a
    # one-way walk can — keep every original list to re-slice from.
    originals = {name: list(payloads[name].values) for name in trimmable}
    max_k = max((len(values) for values in originals.values()), default=0)

    # PHASE 1: largest uniform cap that fits the soft budget.
    k = _largest_k_within(payloads, originals, trimmable, max_k, config.max_tokens)
    if k >= 1:
        _apply_k(payloads, originals, trimmable, k)
        return payloads

    # PHASE 2: the budget is unreachable — one value per column already misses it, or the
    # protected columns alone exceed it. Emit the least we can (one value each) rather than
    # more than was asked for, and accept it as long as it clears the hard cap.
    _apply_k(payloads, originals, trimmable, 1)
    if _estimate_total_tokens(payloads) <= config.safety_ceiling:
        return payloads

    # PHASE 3: even one value per column is over the hard cap. Keep that one value for as
    # many columns as fit and sacrifice the lowest-priority ones, in two steps.
    for degrade in (_to_count_only, _to_bare):
        for col_name in trimmable:
            total = _estimate_total_tokens(payloads)
            if total <= config.safety_ceiling:
                return payloads
            degrade(payloads, col_name)

    total = _estimate_total_tokens(payloads)
    if total > config.safety_ceiling:
        logger.warning(
            "Could not trim below safety_ceiling (%d tokens estimated, ceiling=%d). "
            "Every trimmable column is already reduced — the protected columns alone "
            "exceed the ceiling.",
            total, config.safety_ceiling,
        )

    return payloads


def _apply_k(
    payloads: dict[str, ColumnPayload],
    originals: dict[str, list],
    trimmable: list[str],
    k: int,
) -> None:
    """Re-slice every trimmable column from its ORIGINAL list to k values.

    Slicing from the original (not from the current, already-sliced list) is what lets
    the search raise the cap again after overshooting. Front-of-list slicing preserves
    matched values — assembly guarantees matched-first ordering.
    """
    for col_name in trimmable:
        payload = payloads.get(col_name)
        if payload is not None:
            payload.values = originals[col_name][:k]


def _largest_k_within(
    payloads: dict[str, ColumnPayload],
    originals: dict[str, list],
    trimmable: list[str],
    max_k: int,
    budget: int,
) -> int:
    """Largest uniform per-column cap whose rendered total fits ``budget``.

    Returns 0 when not even one value per column fits. Costs ~log2(max_k) tiktoken
    calls — the same order as the fixed six-level walk this replaces.

    Rendered length grows with the cap, so the fit is monotonic apart from the
    "(N distinct total)" suffix, which a column drops once the cap reaches its own full
    length (~5 tokens each). The untrimmed case is already returned by the caller's quick
    check, so that can only cost a value or two of precision in a mixed-length set.
    """
    lo, hi, best = 1, max_k, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        _apply_k(payloads, originals, trimmable, mid)
        if _estimate_total_tokens(payloads) <= budget:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


def _to_count_only(payloads: dict[str, ColumnPayload], col_name: str) -> None:
    """Give up a column's values but keep its cardinality hint."""
    payload = payloads.get(col_name)
    if payload is None or payload.format_hint == "count_only":
        return
    payloads[col_name] = ColumnPayload(
        format_hint="count_only",
        values=[],
        distinct_count=payload.distinct_count,
    )


def _to_bare(payloads: dict[str, ColumnPayload], col_name: str) -> None:
    """Give up everything but the column itself, which the DDL still lists."""
    payload = payloads.get(col_name)
    if payload is None:
        return
    payloads[col_name] = ColumnPayload(
        format_hint="name_only",
        values=[],
        distinct_count=payload.distinct_count,
    )


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
