"""LLM-based candidate extraction for the Technical Context Builder.

Extracts filter literal candidates + synonyms from the user's question.
The LLM receives ONLY the question — no column info, no values.
Output feeds into the same matching cascade as heuristic n-grams.

Falls back gracefully (returns []) on any failure.
"""

from __future__ import annotations

import concurrent.futures
import contextvars
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def extract_candidates_with_llm(
    question: str,
    *,
    llm=None,
    timeout: int = 30,
) -> list[str]:
    """Extract focused filter candidate strings from the question using an LLM.

    The LLM uses world knowledge to identify literals that would appear in
    WHERE clauses and provides synonyms/alternate forms for each.

    Args:
        question: User's natural language question.
        llm: LLM instance with an .invoke() method. If None, returns [].
        timeout: Timeout in seconds for the LLM call.

    Returns:
        Flat list of candidate literal strings + their synonyms.
        Returns [] on any failure.
    """
    if llm is None:
        return []

    if not question or not question.strip():
        return []

    prompt_text = _build_candidate_extraction_prompt(question)

    try:
        response_text = _call_llm(llm, prompt_text, timeout=timeout)
    except Exception as e:
        logger.warning("LLM candidate extraction call failed: %s", e)
        return []

    return _parse_candidates_response(response_text)


def _build_candidate_extraction_prompt(question: str) -> str:
    """Build the extraction prompt — asks LLM for filter literals + synonyms.
    
    Synonyms should be semantically distinct alternate forms (different
    spellings, abbreviations, translations) — NOT case variants. Matching
    layer handles case-insensitivity via normalization.
    """
    return (
        "Extract filter literals from the user's question — values that would appear "
        "in SQL WHERE clauses (names, codes, places, categories, IDs, statuses, dates, numbers).\n\n"
        "For each literal, include semantically distinct synonyms — alternate spellings, "
        "abbreviations, codes, or expansions the database might use. "
        "DO NOT include case variants (matching is case-insensitive). "
        "Skip generic words (verbs, articles, descriptors with no filter meaning).\n\n"
        "Examples:\n"
        '- "active customers in the US" → [{"literal": "active", "synonyms": []}, '
        '{"literal": "US", "synonyms": ["USA", "United States", "United States of America"]}]\n'
        '- "orders shipped last month with status delivered" → [{"literal": "delivered", "synonyms": ["shipped"]}]\n'
        '- "transactions over 5000 USD from Q3" → [{"literal": "5000", "synonyms": []}, '
        '{"literal": "USD", "synonyms": ["US Dollar", "$"]}]\n'
        '- "premium tier subscribers" → [{"literal": "premium", "synonyms": ["pro", "enterprise"]}]\n'
        '- "products in the Electronics category" → [{"literal": "Electronics", "synonyms": []}]\n\n'
        'Return ONLY valid JSON: {"candidates": [{"literal": str, "synonyms": [str, ...]}]}\n\n'
        f"Question: {question}"
    )


def _call_llm(llm: Any, prompt_text: str, *, timeout: int = 30) -> str:
    """Call the LLM with timeout using ThreadPoolExecutor + contextvars."""
    ctx = contextvars.copy_context()

    def _invoke():
        return ctx.run(llm.invoke, prompt_text)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_invoke)
        try:
            response = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"LLM candidate extraction timed out after {timeout}s")

    # Handle different response types
    if hasattr(response, "content"):
        return str(response.content)
    return str(response)


def _parse_candidates_response(response_text: str) -> list[str]:
    """Parse JSON from LLM response and flatten literals + synonyms.

    Returns flat list of all candidate strings (each literal + its synonyms).
    Returns [] on any parse failure.
    """
    if not response_text:
        return []

    # Strip markdown code fences if present
    text = response_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        logger.warning("LLM candidate extraction returned invalid JSON")
        return []

    if not isinstance(parsed, dict):
        logger.warning("LLM candidate extraction returned non-dict JSON: %s", type(parsed).__name__)
        return []

    candidates_list = parsed.get("candidates", [])
    if not isinstance(candidates_list, list):
        return []

    # Flatten: literal + synonyms for each entry
    result: list[str] = []
    seen: set[str] = set()

    for entry in candidates_list:
        if not isinstance(entry, dict):
            continue

        literal = entry.get("literal", "")
        if isinstance(literal, str) and literal.strip():
            lit = literal.strip()
            if lit not in seen:
                seen.add(lit)
                result.append(lit)

        synonyms = entry.get("synonyms", [])
        if isinstance(synonyms, list):
            for syn in synonyms:
                if isinstance(syn, str) and syn.strip():
                    s = syn.strip()
                    if s not in seen:
                        seen.add(s)
                        result.append(s)

    return result
