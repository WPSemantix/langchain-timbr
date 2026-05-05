"""Fuzzy matching using rapidfuzz for approximate string comparison."""

from __future__ import annotations

from ..types import MatchResult
from .normalize import normalize


def fuzzy_match(
    prompt_tokens: list[str],
    column_name: str,
    known_values: list[str],
    *,
    threshold: int = 88,
) -> list[MatchResult]:
    """Match prompt tokens against known values using fuzzy string similarity.

    Uses rapidfuzz's fuzz.ratio for token-level similarity scoring.
    Only imports rapidfuzz when called (lazy import for optional dependency).

    Args:
        prompt_tokens: Tokens extracted from the user prompt.
        column_name: Column name for result attribution.
        known_values: Known values from column statistics top_k.
        threshold: Minimum similarity score (0-100) to consider a match.

    Returns:
        List of MatchResult with match_type="fuzzy" and the similarity score.
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:
        # rapidfuzz not available — skip fuzzy matching
        return []

    results: list[MatchResult] = []
    # Pre-normalize known values
    norm_known: list[tuple[str, str]] = []
    for v in known_values:
        nv = normalize(str(v))
        if nv and len(nv) >= 3:  # Skip very short values for fuzzy
            norm_known.append((nv, str(v)))

    for token in prompt_tokens:
        norm_token = normalize(token)
        if not norm_token or len(norm_token) < 3:
            continue

        best_score = 0
        best_original = ""
        for nv, original in norm_known:
            score = fuzz.ratio(norm_token, nv)
            if score > best_score:
                best_score = score
                best_original = original

        if best_score >= threshold:
            results.append(MatchResult(
                column_name=column_name,
                matched_value=best_original,
                score=int(best_score),
                match_type="fuzzy",
                candidate=token,
            ))

    return results
