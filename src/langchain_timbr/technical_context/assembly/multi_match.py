"""Multi-match coordination.

Runs all matching strategies (exact → substring → fuzzy) in priority order,
collecting results and deduplicating across strategies.
"""

from __future__ import annotations

from collections import OrderedDict

from ..config import TechnicalContextConfig
from ..types import MatchResult, SemanticType
from ..matching.exact import exact_match
from ..matching.normalize import normalize, normalize_keep_spaces
from ..matching.rapidfuzz_matcher import fuzzy_match
from ..matching.ahocorasick_matcher import substring_match
from .structure_cache import get_entry

# Results memo. ``run_all_matchers`` is a pure function of its inputs, and one
# SQL-generation run calls it with the SAME (question, column, values) two or
# three times — the dynamic metadata-context top-up and the reasoning pass both
# re-enter ``build_technical_context`` from scratch. Keyed on the values tuple
# itself (verified on hit, so a hash collision can't return a wrong answer);
# the statistics cache hands back the same string objects between passes, which
# makes both the hash and the verification a pointer walk.
# Bounded by total cached values rather than entry count: an entry only holds
# references to strings the statistics cache already owns, so its real cost is
# one pointer per value.
_RESULT_CACHE_MAX_VALUES = 2_000_000
# Second bound on entry count: the value counter is incremented without a lock,
# so concurrent callers can lose an update and let it drift low. len() cannot.
_RESULT_CACHE_MAX_ENTRIES = 4096
_RESULT_CACHE: "OrderedDict[tuple, tuple[tuple, list]]" = OrderedDict()
_RESULT_CACHE_VALUES = 0


def clear_matcher_cache() -> None:
    """Drop every memoized matcher result. Used by tests for isolation."""
    global _RESULT_CACHE_VALUES
    _RESULT_CACHE.clear()
    _RESULT_CACHE_VALUES = 0


def run_all_matchers(
    prompt_text: str,
    prompt_tokens: list[str],
    column_name: str,
    known_values: list[str],
    config: TechnicalContextConfig,
    semantic_type: SemanticType | None = None,
    normalized: list[str] | None = None,
    normalized_space: list[str] | None = None,
    normalized_tokens: list[str] | None = None,
    normalized_prompt: str | None = None,
    normalized_stripped: list[str] | None = None,
    value_kind: str = "text",
    ontology: str | None = None,
    schema: str | None = None,
    concept: str | None = None,
) -> list[MatchResult]:
    """Run all matchers in priority order and deduplicate results.

    Priority: exact (100) > substring (95) > fuzzy (sort_threshold-based)

    Uses the sort threshold (surface - gap) as the fuzzy floor so that both
    strong and weak matches are returned with their scores. Downstream code
    uses scores to bucket into strong/weak tiers.

    Args:
        prompt_text: Full user prompt (for substring matching).
        prompt_tokens: Extracted tokens/n-grams OR LLM candidates (for exact and fuzzy).
        column_name: Column being matched against.
        known_values: Known values from statistics top_k.
        config: Configuration with thresholds.
        semantic_type: Column semantic type (affects threshold selection).
        normalized: Pre-normalized forms of ``known_values``, index-aligned.
            Carried on ``TopKEntry`` and derived once when the statistics row was
            parsed, so the hot path does no normalization at all. Computed here
            when omitted, which is what happens for the branches that feed
            already-matched values back in.
        normalized_space: as ``normalized``, space-preserving form.
        normalized_tokens: Pre-normalized ``prompt_tokens``, index-aligned.
            The tokens are identical for every column in a request, so the caller
            normalizes them once rather than once per column.
        normalized_prompt: Pre-normalized (space-preserving) ``prompt_text``.
        normalized_stripped: Leading-zero-stripped forms, for numeric columns.
        value_kind: "numeric", "date" or "text", derived when the row was parsed.
            Numeric and date columns skip fuzzy matching: edit distance between
            two numbers is meaningless and actively harmful — at the default
            floor of 70, `1500` matches `1600` and `20240301` matches `20240302`,
            so a question about one offers the other as a filter literal.

    Returns:
        Combined, deduplicated list of MatchResults.
    """
    if not known_values or (not prompt_text and not prompt_tokens):
        return []

    # Select the surface (strong) threshold based on semantic type; the sort
    # threshold is the weak bar below it.
    if semantic_type in (SemanticType.CODE_LIKE, SemanticType.BUSINESS_KEY_LIKE):
        surface_threshold = config.fuzzy_threshold_strict
    else:
        surface_threshold = config.fuzzy_threshold_default
    sort_threshold = surface_threshold - config.fuzzy_sort_gap

    values_key = tuple(known_values)
    cache_key = (
        prompt_text, tuple(prompt_tokens), column_name,
        semantic_type, sort_threshold, len(known_values), value_kind,
    )
    cached = _RESULT_CACHE.get(cache_key)
    if cached is not None and cached[0] == values_key:
        _RESULT_CACHE.move_to_end(cache_key)
        return list(cached[1])

    all_results: list[MatchResult] = []
    matched_values: set[str] = set()

    # Normalize each known value once and share the result across all three
    # matchers. Previously every stage re-derived it from the raw text, so a
    # single column's values went through NFKC + casefold + regex two or three
    # times per pass — the dominant CPU cost of the whole technical-context build.
    values = [str(v) for v in known_values]
    norm_values = (
        normalized if normalized is not None else [normalize(v) for v in values]
    )
    norm_space_values = (
        normalized_space if normalized_space is not None
        else [normalize_keep_spaces(v) for v in values]
    )
    norm_tokens = (
        normalized_tokens if normalized_tokens is not None
        else [normalize(t) for t in prompt_tokens]
    )
    norm_prompt = (
        normalized_prompt if normalized_prompt is not None
        else normalize_keep_spaces(prompt_text)
    )

    # The per-column structures each stage needs are pure functions of the
    # column's values, so they are built once and reused across requests. The
    # stages used to receive a pre-filtered value list ("everything the previous
    # stage did not claim"), which is question-dependent and therefore uncachable
    # — so the filtering moved into the stages as ``exclude``. Same results, see
    # the note in ``build_automaton``.
    entry = get_entry(column_name, known_values, value_kind, values_key,
                      ontology=ontology, schema=schema, concept=concept)

    # 1. Exact matching
    exact_results = exact_match(
        prompt_tokens, column_name, values,
        normalized=norm_values, normalized_tokens=norm_tokens,
        normalized_stripped=normalized_stripped,
        lookup=(entry.exact(values, norm_values, normalized_stripped)
                if entry is not None else None),
    )
    for r in exact_results:
        if r.matched_value not in matched_values:
            matched_values.add(r.matched_value)
            all_results.append(r)

    # 2. Substring matching (Aho-Corasick)
    if prompt_text and len(matched_values) < len(values):
        sub_results = substring_match(
            prompt_text,
            column_name,
            values,
            normalized=norm_space_values,
            normalized_prompt=norm_prompt,
            built=(entry.automaton(values, norm_space_values, 3)
                   if entry is not None else None),
            exclude=matched_values,
        )
        for r in sub_results:
            if r.matched_value not in matched_values:
                matched_values.add(r.matched_value)
                all_results.append(r)

    # 3. Fuzzy matching (using sort_threshold as the floor).
    # Not for numbers or dates — see ``value_kind`` in the docstring.
    if prompt_tokens and value_kind == "text" and len(matched_values) < len(values):
        fuzzy_results = fuzzy_match(
            prompt_tokens,
            column_name,
            values,
            threshold=sort_threshold,
            strong_threshold=surface_threshold,
            normalized=norm_values,
            normalized_tokens=norm_tokens,
            choices=(entry.choices(values, norm_values)
                     if entry is not None else None),
            exclude=matched_values,
        )
        for r in fuzzy_results:
            if r.matched_value not in matched_values:
                matched_values.add(r.matched_value)
                all_results.append(r)

    # LRU rather than clear-on-overflow: wiping the whole memo mid-request made
    # a request larger than the cap retain nothing at all, which measured worse
    # than having no memo. Drop the coldest entries until we fit instead.
    global _RESULT_CACHE_VALUES
    while _RESULT_CACHE and (
        _RESULT_CACHE_VALUES >= _RESULT_CACHE_MAX_VALUES
        or len(_RESULT_CACHE) >= _RESULT_CACHE_MAX_ENTRIES
    ):
        _, (evicted_values_key, _) = _RESULT_CACHE.popitem(last=False)
        _RESULT_CACHE_VALUES = max(0, _RESULT_CACHE_VALUES - len(evicted_values_key))
    _RESULT_CACHE[cache_key] = (values_key, all_results)
    _RESULT_CACHE_VALUES += len(values_key)

    return list(all_results)
