"""Aho-Corasick based substring matching for detecting known values within prompt text."""

from __future__ import annotations

from ..types import MatchResult
from .normalize import normalize_keep_spaces


def build_automaton(
    known_values: list[str],
    normalized: list[str] | None = None,
    min_length: int = 3,
):
    """Build the Aho-Corasick automaton and its value lookup for one column.

    Question-independent — a pure function of the column's values — which is why
    :mod:`..assembly.structure_cache` can hold the result across requests. This
    is the single most expensive thing the matchers do: 567.8 ms of the 1440.7 ms
    a 400-column question spent in matching, rebuilt identically every time.

    Returns:
        ``(automaton, {normalized: [originals, in value order]})``, or None when
        no value is long enough to search for. A built automaton is safe to
        share across threads for reading.

    Note the value is a **list**, not a single original. The uncached code built
    this dict over only the values earlier stages had not already claimed, so a
    later duplicate could win — which is how ``'phd'`` yields ``'Ph.D.'`` from
    exact and then ``'PhD'`` from substring. Caching one winner per column would
    silently break that cascade, so the whole group is kept and the caller picks
    from it (see ``exclude`` in :func:`substring_match`).
    """
    try:
        import ahocorasick
    except ImportError:
        return None

    automaton = ahocorasick.Automaton()
    norm_to_originals: dict[str, list[str]] = {}

    if normalized is None:
        normalized = (normalize_keep_spaces(str(v)) for v in known_values)
    for v, nv in zip(known_values, normalized):
        if nv and len(nv) >= min_length:
            group = norm_to_originals.get(nv)
            if group is None:
                norm_to_originals[nv] = [str(v)]
                automaton.add_word(nv, nv)
            else:
                group.append(str(v))

    if not norm_to_originals:
        return None

    automaton.make_automaton()
    return automaton, norm_to_originals


def substring_match(
    prompt_text: str,
    column_name: str,
    known_values: list[str],
    *,
    min_length: int = 3,
    normalized: list[str] | None = None,
    normalized_prompt: str | None = None,
    built: tuple | None = None,
    exclude: set[str] | None = None,
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
    if not prompt_text or not known_values:
        return []

    if built is None:
        built = build_automaton(known_values, normalized, min_length)
    if built is None:
        # pyahocorasick unavailable, or no value long enough to search for.
        return []
    automaton, norm_to_originals = built

    # Search the normalized prompt
    norm_prompt = (
        normalized_prompt if normalized_prompt is not None
        else normalize_keep_spaces(prompt_text)
    )
    results: list[MatchResult] = []
    seen: set[str] = set()

    for end_idx, matched_norm in automaton.iter(norm_prompt):
        if matched_norm in seen:
            continue
        seen.add(matched_norm)
        # LAST value in the group that an earlier stage has not claimed. Last,
        # not first: the uncached code rebuilt this dict per request and later
        # assignments overwrote earlier ones, so the last surviving duplicate
        # won. Reproduced exactly so the cache changes no results.
        original = None
        for candidate_value in reversed(norm_to_originals[matched_norm]):
            if exclude is None or candidate_value not in exclude:
                original = candidate_value
                break
        if original is None:
            continue
        results.append(MatchResult(
            column_name=column_name,
            matched_value=original,
            score=95,
            match_type="substring",
            candidate=matched_norm,
        ))

    return results
