"""Aho-Corasick based substring matching for detecting known values within prompt text."""

from __future__ import annotations

from ..types import MatchResult
from .normalize import normalize_keep_spaces


def substring_match(
    prompt_text: str,
    column_name: str,
    known_values: list[str],
    *,
    min_length: int = 3,
    normalized: list[str] | None = None,
) -> list[MatchResult]:
    """Find known values that appear as substrings within the prompt text.

    Uses pyahocorasick for efficient multi-pattern search (O(n+m+z) where
    n=text length, m=total pattern length, z=number of matches).

    Args:
        prompt_text: Full user prompt text.
        column_name: Column name for result attribution.
        known_values: Known values from column statistics top_k.
        min_length: Minimum normalized value length to include in search.
        normalized: Optional pre-normalized (space-preserving) forms of
            ``known_values``, index-aligned. Supplied by :func:`run_all_matchers`
            so the three matchers normalize each value once between them.
            Computed here when omitted.

    Returns:
        List of MatchResult with match_type="substring" and score=95.
    """
    try:
        import ahocorasick
    except ImportError:
        # pyahocorasick not available — skip substring matching
        return []

    if not prompt_text or not known_values:
        return []

    # Build automaton from known values
    automaton = ahocorasick.Automaton()
    norm_to_original: dict[str, str] = {}

    if normalized is None:
        normalized = (normalize_keep_spaces(str(v)) for v in known_values)
    for v, nv in zip(known_values, normalized):
        if nv and len(nv) >= min_length:
            norm_to_original[nv] = str(v)
            automaton.add_word(nv, nv)

    if not norm_to_original:
        return []

    automaton.make_automaton()

    # Search the normalized prompt
    norm_prompt = normalize_keep_spaces(prompt_text)
    results: list[MatchResult] = []
    seen: set[str] = set()

    for end_idx, matched_norm in automaton.iter(norm_prompt):
        if matched_norm in seen:
            continue
        seen.add(matched_norm)
        original = norm_to_original[matched_norm]
        results.append(MatchResult(
            column_name=column_name,
            matched_value=original,
            score=95,
            match_type="substring",
            candidate=matched_norm,
        ))

    return results
