"""Text normalization for value matching.

Applies NFKC normalization, casefolding, and strips non-alphanumeric characters
to produce a canonical form for comparison.
"""

from __future__ import annotations

import re
import unicodedata

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]")


def normalize(text: str) -> str:
    """Normalize a string for matching: NFKC + casefold + strip non-alphanumerics.

    Args:
        text: Input string (may be None or empty).

    Returns:
        Normalized lowercase alphanumeric-only string.

    Examples:
        >>> normalize("Café Latte")
        'cafelatte'
        >>> normalize("U.S.A.")
        'usa'
        >>> normalize("  Hello World  ")
        'helloworld'
    """
    if not text:
        return ""
    # NFKC: normalize unicode (e.g., ﬁ → fi, ² → 2)
    text = unicodedata.normalize("NFKC", text)
    # Casefold: aggressive lowercase (e.g., ß → ss)
    text = text.casefold()
    # Strip non-alphanumeric
    return _NON_ALNUM_RE.sub("", text)


def normalize_keep_spaces(text: str) -> str:
    """Normalize but preserve word boundaries as single spaces.

    Useful for substring matching where word order matters.

    Examples:
        >>> normalize_keep_spaces("New  York  City")
        'new york city'
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    # Replace non-alnum with space, collapse multiple spaces
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()
