"""Multi-match coordination.

Runs all matching strategies (exact → substring → fuzzy) in priority order,
collecting results and deduplicating across strategies.
"""

from __future__ import annotations

from ..config import TechnicalContextConfig
from ..types import MatchResult, SemanticType
from ..matching.exact import exact_match
from ..matching.normalize import normalize, normalize_keep_spaces
from ..matching.rapidfuzz_matcher import fuzzy_match
from ..matching.ahocorasick_matcher import substring_match

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
_RESULT_CACHE: dict[tuple, tuple[tuple, list]] = {}
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

    Returns:
        Combined, deduplicated list of MatchResults.
    """
    if not known_values or (not prompt_text and not prompt_tokens):
        return []

    # Select the sort threshold (lower bar) based on semantic type
    if semantic_type in (SemanticType.CODE_LIKE, SemanticType.BUSINESS_KEY_LIKE):
        sort_threshold = config.fuzzy_threshold_strict - config.fuzzy_sort_gap
    else:
        sort_threshold = config.fuzzy_threshold_default - config.fuzzy_sort_gap

    values_key = tuple(known_values)
    cache_key = (
        prompt_text, tuple(prompt_tokens), column_name,
        semantic_type, sort_threshold, len(known_values),
    )
    cached = _RESULT_CACHE.get(cache_key)
    if cached is not None and cached[0] == values_key:
        return list(cached[1])

    all_results: list[MatchResult] = []
    matched_values: set[str] = set()

    # Normalize each known value once and share the result across all three
    # matchers. Previously every stage re-derived it from the raw text, so a
    # single column's values went through NFKC + casefold + regex two or three
    # times per pass — the dominant CPU cost of the whole technical-context build.
    values = [str(v) for v in known_values]
    norm_values = [normalize(v) for v in values]
    norm_space_values = [normalize_keep_spaces(v) for v in values]

    # 1. Exact matching
    exact_results = exact_match(
        prompt_tokens, column_name, values, normalized=norm_values,
    )
    for r in exact_results:
        if r.matched_value not in matched_values:
            matched_values.add(r.matched_value)
            all_results.append(r)

    # 2. Substring matching (Aho-Corasick)
    remaining = [i for i, v in enumerate(values) if v not in matched_values]
    if remaining and prompt_text:
        sub_results = substring_match(
            prompt_text,
            column_name,
            [values[i] for i in remaining],
            normalized=[norm_space_values[i] for i in remaining],
        )
        for r in sub_results:
            if r.matched_value not in matched_values:
                matched_values.add(r.matched_value)
                all_results.append(r)

    # 3. Fuzzy matching (using sort_threshold as the floor)
    remaining = [i for i, v in enumerate(values) if v not in matched_values]
    if remaining and prompt_tokens:
        fuzzy_results = fuzzy_match(
            prompt_tokens,
            column_name,
            [values[i] for i in remaining],
            threshold=sort_threshold,
            normalized=[norm_values[i] for i in remaining],
        )
        for r in fuzzy_results:
            if r.matched_value not in matched_values:
                matched_values.add(r.matched_value)
                all_results.append(r)

    global _RESULT_CACHE_VALUES
    if (_RESULT_CACHE_VALUES >= _RESULT_CACHE_MAX_VALUES
            or len(_RESULT_CACHE) >= _RESULT_CACHE_MAX_ENTRIES):
        _RESULT_CACHE.clear()
        _RESULT_CACHE_VALUES = 0
    _RESULT_CACHE[cache_key] = (values_key, all_results)
    _RESULT_CACHE_VALUES += len(values_key)

    return list(all_results)
