"""Exact matching of prompt tokens against column top-K values."""

from __future__ import annotations

from ..types import MatchResult
from .normalize import normalize


def exact_match(
    prompt_tokens: list[str],
    column_name: str,
    known_values: list[str],
) -> list[MatchResult]:
    """Match prompt tokens against known values using exact (normalized) equality.

    Args:
        prompt_tokens: Normalized tokens extracted from the user prompt.
        column_name: Column name for result attribution.
        known_values: Known values from column statistics top_k.

    Returns:
        List of MatchResult with match_type="exact" and score=100.
    """
    results: list[MatchResult] = []
    # Build a lookup: normalized_value -> original_value
    norm_to_original: dict[str, str] = {}
    for v in known_values:
        nv = normalize(str(v))
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
