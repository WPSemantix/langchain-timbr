"""Text normalization for value matching.

Applies NFKC normalization, casefolding, and strips non-alphanumeric characters
to produce a canonical form for comparison.

Both functions are pure ``str -> str`` maps, so results are memoized in a
process-wide cache. Column top-K value lists repeat heavily — across columns of
the same ontology, across the 2-3 ``build_technical_context`` passes inside one
invoke, and across invokes (the statistics cache hands back the *same* string
objects, so their hashes are already computed). The cache is cleared wholesale
when it grows past ``_CACHE_MAX`` rather than maintaining LRU order, which keeps
the hit path a single dict lookup.
"""

from __future__ import annotations

import re
import unicodedata

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]")
_NON_ALNUM_SPACE_RE = re.compile(r"[^a-z0-9]+")

# Bounded memo caches. Entries are short strings; the bound keeps worst-case
# footprint predictable on very wide ontologies.
_CACHE_MAX = 250_000
_NORM_CACHE: dict[str, str] = {}
_NORM_SPACE_CACHE: dict[str, str] = {}


def clear_normalize_cache() -> None:
    """Drop every memoized normalization. Used by tests for isolation."""
    _NORM_CACHE.clear()
    _NORM_SPACE_CACHE.clear()


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
    cached = _NORM_CACHE.get(text)
    if cached is not None:
        return cached
    # NFKC: normalize unicode (e.g., ﬁ → fi, ² → 2)
    result = unicodedata.normalize("NFKC", text)
    # Casefold: aggressive lowercase (e.g., ß → ss)
    result = result.casefold()
    # Strip non-alphanumeric
    result = _NON_ALNUM_RE.sub("", result)
    if len(_NORM_CACHE) >= _CACHE_MAX:
        _NORM_CACHE.clear()
    _NORM_CACHE[text] = result
    return result


def normalize_keep_spaces(text: str) -> str:
    """Normalize but preserve word boundaries as single spaces.

    Useful for substring matching where word order matters.

    Examples:
        >>> normalize_keep_spaces("New  York  City")
        'new york city'
    """
    if not text:
        return ""
    cached = _NORM_SPACE_CACHE.get(text)
    if cached is not None:
        return cached
    result = unicodedata.normalize("NFKC", text)
    result = result.casefold()
    # Replace non-alnum with space, collapse multiple spaces
    result = _NON_ALNUM_SPACE_RE.sub(" ", result).strip()
    if len(_NORM_SPACE_CACHE) >= _CACHE_MAX:
        _NORM_SPACE_CACHE.clear()
    _NORM_SPACE_CACHE[text] = result
    return result
