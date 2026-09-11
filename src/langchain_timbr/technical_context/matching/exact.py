"""Exact matching of prompt tokens against column top-K values."""

from __future__ import annotations

from ..types import MatchResult
from .normalize import normalize


def build_exact_lookup(
    known_values: list[str],
    normalized: list[str] | None = None,
    normalized_stripped: list[str] | None = None,
) -> dict[str, str]:
    """Build the ``{normalized_value: original}`` lookup for one column.

    Question-independent — a pure function of the column's values — which is why
    :mod:`..assembly.structure_cache` can hold the result across requests.
    """
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

    # Second key per value: its zero-stripped form. A real value wins over a
    # stripped one, so `10` present as itself is preferred to `000010` stripped.
    if normalized_stripped is not None:
        for v, sv in zip(known_values, normalized_stripped):
            if sv and sv not in norm_to_original:
                norm_to_original[sv] = str(v)

    return norm_to_original


def exact_match(
    prompt_tokens: list[str],
    column_name: str,
    known_values: list[str],
    *,
    normalized: list[str] | None = None,
    normalized_tokens: list[str] | None = None,
    normalized_stripped: list[str] | None = None,
    lookup: dict[str, str] | None = None,
) -> list[MatchResult]:
    """Match prompt tokens against known values using exact (normalized) equality.

    Args:
        prompt_tokens: Normalized tokens extracted from the user prompt.
        column_name: Column name for result attribution.
        known_values: Known values from column statistics top_k.
        normalized: Optional pre-normalized forms of ``known_values``, index-aligned.
            Supplied by :func:`run_all_matchers` so the three matchers normalize
            each value once between them. Computed here when omitted.
        normalized_stripped: Optional leading-zero-stripped forms, index-aligned,
            supplied for numeric columns only. Zero-padded identifiers are the
            norm in this data and nobody asks for them that way, so ``10`` has to
            be able to find ``000010``. Exact matching is whole-token by
            construction, which is why the stripped form is safe here and is
            deliberately *not* added to the substring automaton — ``10`` there
            would match inside ``2010``.

    Returns:
        List of MatchResult with match_type="exact" and score=100.
    """
    results: list[MatchResult] = []
    if lookup is None:
        lookup = build_exact_lookup(known_values, normalized, normalized_stripped)
    norm_to_original = lookup

    norm_tokens = (
        normalized_tokens if normalized_tokens is not None
        else [normalize(t) for t in prompt_tokens]
    )

    for token, norm_token in zip(prompt_tokens, norm_tokens):
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
