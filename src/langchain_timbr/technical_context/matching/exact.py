"""Exact matching of prompt tokens against column top-K values."""

from __future__ import annotations

from ..types import MatchResult
from .normalize import normalize


def exact_match(
    prompt_tokens: list[str],
    column_name: str,
    known_values: list[str],
    *,
    normalized: list[str] | None = None,
) -> list[MatchResult]:
    """Match prompt tokens against known values using exact (normalized) equality.

    Args:
        prompt_tokens: Normalized tokens extracted from the user prompt.
        column_name: Column name for result attribution.
        known_values: Known values from column statistics top_k.
        normalized: Optional pre-normalized forms of ``known_values``, index-aligned.
            Supplied by :func:`run_all_matchers` so the three matchers normalize
            each value once between them. Computed here when omitted.

    Returns:
        List of MatchResult with match_type="exact" and score=100.
    """
    results: list[MatchResult] = []
    # Build a lookup: normalized_value -> original_value
    norm_to_original: dict[str, str] = {}
    if normalized is None:
        for v in known_values:
            sv = str(v)
            nv = normalize(sv)
            if nv:
                norm_to_original[nv] = sv
    else:
        for v, nv in zip(known_values, normalized):
            if nv:
                norm_to_original[nv] = str(v)

    for token in prompt_tokens:
        norm_token = normalize(token)
        if not norm_token:
            continue
        if norm_token in norm_to_original:
            results.append(MatchResult(
                column_name=column_name,
                matched_value=norm_to_original[norm_token],
                score=100,
                match_type="exact",
                candidate=token,
            ))

    return results
