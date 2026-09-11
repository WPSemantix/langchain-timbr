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

from ...config import normalize_unicode

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]")
_NON_ALNUM_SPACE_RE = re.compile(r"[^a-z0-9]+")

# Unicode-aware equivalents. ``\w`` under Python 3 is already Unicode-aware, so
# it keeps CJK, Cyrillic, Greek and the rest; ``_`` is excluded explicitly
# because it is punctuation for our purposes, not a letter.
_NON_WORD_RE = re.compile(r"[\W_]", re.UNICODE)
_NON_WORD_SPACE_RE = re.compile(r"[\W_]+", re.UNICODE)


def _fold(text: str) -> str:
    """NFKC -> casefold -> strip accents, preserving non-Latin scripts.

    The accent strip is NFKD (decompose) -> drop combining marks -> NFC
    (recompose). The recompose matters: NFKD splits Hangul syllables into jamo,
    which are letters rather than combining marks, so they survive the drop and
    would leave Korean text decomposed. Latin accents do not come back, because
    their combining mark is gone — which is the point.
    """
    result = unicodedata.normalize("NFKC", text)
    result = result.casefold()
    decomposed = unicodedata.normalize("NFKD", result)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", stripped)

# Bounded memo caches. Entries are short strings; the bound keeps worst-case
# footprint predictable on very wide ontologies.
def clear_normalize_cache() -> None:
    """No-op, kept for callers that still invoke it.

    Normalization used to be memoized in two process-wide dicts that wiped
    themselves wholesale on overflow — so a single request bigger than the cap
    thrashed and retained nothing. Values are now normalized once where they are
    parsed and carried on ``TopKEntry``; prompt tokens are normalized once per
    request. Nothing is memoized here any more, so there is nothing to clear.
    """


def normalize(text: str) -> str:
    """Normalize a string for matching: fold case and accents, keep alphanumerics.

    "Alphanumeric" means Unicode alphanumeric. The previous implementation kept
    only ASCII ``[a-z0-9]``, which silently destroyed international values

    Args:
        text: Input string (may be None or empty).

    Returns:
        Normalized lowercase alphanumeric string.

    Examples:
        >>> normalize("Café Latte")
        'cafelatte'
        >>> normalize("U.S.A.")
        'usa'
        >>> normalize("  Hello World  ")
        'helloworld'
        >>> normalize("Müller GmbH")
        'mullergmbh'
    """
    if not text:
        return ""
    # Fast path, and it is the overwhelmingly common one: 99% of the values in
    # the captured fixture are pure ASCII, where every step of the Unicode
    # pipeline is a no-op — NFKC, NFKD and NFC are all identities, there are no
    # combining marks to strip, and `[\W_]` and `[^a-z0-9]` select the same
    # characters once casefolded. Skipping it is exact, not an approximation.
    # Without this the fix cost 3.6x on normalization, which parse time pays for
    # every statistics value.
    if text.isascii():
        return _NON_ALNUM_RE.sub("", text.casefold())
    if not normalize_unicode:
        result = unicodedata.normalize("NFKC", text).casefold()
        return _NON_ALNUM_RE.sub("", result)
    return _NON_WORD_RE.sub("", _fold(text))


def normalize_keep_spaces(text: str) -> str:
    """Normalize but preserve word boundaries as single spaces.

    Useful for substring matching where word order matters.

    Examples:
        >>> normalize_keep_spaces("New  York  City")
        'new york city'
    """
    if not text:
        return ""
    if text.isascii():  # see the note in normalize()
        return _NON_ALNUM_SPACE_RE.sub(" ", text.casefold()).strip()
    if not normalize_unicode:
        result = unicodedata.normalize("NFKC", text).casefold()
        return _NON_ALNUM_SPACE_RE.sub(" ", result).strip()
    return _NON_WORD_SPACE_RE.sub(" ", _fold(text)).strip()
