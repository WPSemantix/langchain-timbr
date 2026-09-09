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
    normalized: list[str] | None = None,
) -> list[MatchResult]:
    """Match prompt tokens against known values using fuzzy string similarity.

    Uses rapidfuzz's fuzz.ratio for token-level similarity scoring. Scoring runs
    through ``process.extractOne``, which does the whole token-vs-values sweep
    inside rapidfuzz's C++ extension instead of a Python loop: same scores, same
    first-best tie-break, about a quarter of the wall time. It does not release
    the GIL, but it holds it for a quarter as long, which is what other request
    threads in a shared server process actually feel.
    Only imports rapidfuzz when called (lazy import for optional dependency).

    Args:
        prompt_tokens: Tokens extracted from the user prompt.
        column_name: Column name for result attribution.
        known_values: Known values from column statistics top_k.
        threshold: Minimum similarity score (0-100) to consider a match.
        normalized: Optional pre-normalized forms of ``known_values``, index-aligned.
            Supplied by :func:`run_all_matchers` so the three matchers normalize
            each value once between them. Computed here when omitted.

    Returns:
        List of MatchResult with match_type="fuzzy" and the similarity score.
    """
    try:
        from rapidfuzz import fuzz, process
    except ImportError:
        # rapidfuzz not available — skip fuzzy matching
        return []

    # Pre-normalize known values
    norm_choices: list[str] = []
    originals: list[str] = []
    if normalized is None:
        normalized = (normalize(str(v)) for v in known_values)
    for v, nv in zip(known_values, normalized):
        if nv and len(nv) >= 3:  # Skip very short values for fuzzy
            norm_choices.append(nv)
            originals.append(str(v))

    if not norm_choices:
        return []

    results: list[MatchResult] = []
    for token in prompt_tokens:
        norm_token = normalize(token)
        if not norm_token or len(norm_token) < 3:
            continue

        best = process.extractOne(norm_token, norm_choices, scorer=fuzz.ratio)
        if best is None:
            continue
        best_score, best_index = best[1], best[2]

        if best_score >= threshold:
            results.append(MatchResult(
                column_name=column_name,
                matched_value=originals[best_index],
                score=int(best_score),
                match_type="fuzzy",
                candidate=token,
            ))

    return results
