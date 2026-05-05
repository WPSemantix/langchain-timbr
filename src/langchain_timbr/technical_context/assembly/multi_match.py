"""Multi-match coordination.

Runs all matching strategies (exact → substring → fuzzy) in priority order,
collecting results and deduplicating across strategies.
"""

from __future__ import annotations

from ..config import TechnicalContextConfig
from ..types import MatchResult, SemanticType
from ..matching.exact import exact_match
from ..matching.rapidfuzz_matcher import fuzzy_match
from ..matching.ahocorasick_matcher import substring_match


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

    all_results: list[MatchResult] = []
    matched_values: set[str] = set()

    # 1. Exact matching
    exact_results = exact_match(prompt_tokens, column_name, known_values)
    for r in exact_results:
        if r.matched_value not in matched_values:
            matched_values.add(r.matched_value)
            all_results.append(r)

    # 2. Substring matching (Aho-Corasick)
    remaining_values = [v for v in known_values if v not in matched_values]
    if remaining_values and prompt_text:
        sub_results = substring_match(prompt_text, column_name, remaining_values)
        for r in sub_results:
            if r.matched_value not in matched_values:
                matched_values.add(r.matched_value)
                all_results.append(r)

    # 3. Fuzzy matching (using sort_threshold as the floor)
    remaining_values = [v for v in known_values if v not in matched_values]
    if remaining_values and prompt_tokens:
        fuzzy_results = fuzzy_match(
            prompt_tokens,
            column_name,
            remaining_values,
            threshold=sort_threshold,
        )
        for r in fuzzy_results:
            if r.matched_value not in matched_values:
                matched_values.add(r.matched_value)
                all_results.append(r)

    return all_results
