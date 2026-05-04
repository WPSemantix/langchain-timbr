"""N-gram extraction from user prompts for value matching.

Extracts candidate tokens and multi-word n-grams from the prompt that can be
matched against column known values.
"""

from __future__ import annotations

import re

_QUOTED_RE = re.compile(r"""(?:"([^"]+)"|'([^']+)')""")
_SPLIT_RE = re.compile(r"[\s,;]+")


def extract_prompt_tokens(
    prompt: str,
    *,
    max_ngram: int = 4,
    min_token_length: int = 2,
) -> list[str]:
    """Extract candidate tokens and n-grams from a user prompt.

    Extraction strategy:
    1. Extract quoted strings as-is (high priority candidates)
    2. Split remaining text into words
    3. Generate n-grams up to max_ngram word length

    Args:
        prompt: The user's natural language question.
        max_ngram: Maximum number of words in an n-gram.
        min_token_length: Minimum character length for single tokens.

    Returns:
        Deduplicated list of candidate strings for matching.
    """
    if not prompt:
        return []

    candidates: list[str] = []
    seen: set[str] = set()

    def _add(token: str) -> None:
        t = token.strip()
        if t and t not in seen:
            seen.add(t)
            candidates.append(t)

    # 1. Extract quoted strings
    for match in _QUOTED_RE.finditer(prompt):
        quoted = match.group(1) or match.group(2)
        _add(quoted)

    # 2. Remove quoted parts and split into words
    cleaned = _QUOTED_RE.sub(" ", prompt)
    words = [w for w in _SPLIT_RE.split(cleaned) if len(w) >= min_token_length]

    # 3. Single words
    for w in words:
        _add(w)

    # 4. N-grams (2..max_ngram)
    for n in range(2, min(max_ngram + 1, len(words) + 1)):
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i:i + n])
            _add(ngram)

    return candidates
