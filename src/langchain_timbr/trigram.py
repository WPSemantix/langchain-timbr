"""Pure, dependency-free trigram matcher.

Used by the identify-concept context builder for two query-conditioned decisions:

* **Sub-type hints** — does an out-of-list sub-type look relevant to the question?
* **Relationship-axis trim** — under token pressure, keep a relationship whose
  name / target / source trigram-matches the question.

The matcher is morphology-tolerant (``snake_case`` and ``camelCase`` are split
into words before 3-gramming) so ``ProductReturns`` matches a question that
mentions "returned products". It is intentionally cheap and deterministic — no
stemming, no external corpora.
"""

import re

# Split snake_case / camelCase / digit runs into individual words.
_WORD_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")

# Tiny function words that carry no anchoring signal. Kept deliberately short —
# trigramming already dilutes common glue words, this just avoids false matches
# on very short names.
DEFAULT_STOPWORDS = frozenset({
    "the", "a", "an", "of", "for", "to", "in", "on", "at", "by", "and", "or",
    "is", "are", "be", "as", "with", "from", "that", "this", "which", "what",
    "how", "many", "much", "do", "does", "did", "was", "were", "has", "have",
})

# Defaults mirror config; callers pass config values explicitly where available.
DEFAULT_THRESHOLD = 0.65
DEFAULT_FLOOR = 3


def _tokenize(text):
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def to_tokens(text):
    """Normalized word tokens of ``text`` (snake/camel split, lowercased).

    Unlike :func:`to_trigram_set` these are *not* stopword-filtered — the
    whole-word short-name fallback (:func:`matches`) matches on the literal
    tokens, so even a name that is itself a glue word can still be matched.
    """
    return _tokenize(text)


# Collapse every non-alphanumeric run (punctuation, underscores, whitespace) to a
# single space so the question can be scanned for whole-word occurrences.
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize(text):
    """Lowercase ``text`` and collapse non-alphanumeric runs to single spaces."""
    return _NON_ALNUM_RE.sub(" ", (text or "").lower()).strip()


def pad_question(question):
    """Normalized question wrapped in sentinel spaces for whole-word containment.

    Compute once per query and reuse for every name match via :func:`matches`.
    """
    return " " + normalize(question) + " "


def to_trigram_set(text, stopwords=DEFAULT_STOPWORDS):
    """Return the set of 3-grams for ``text``.

    Words shorter than three characters are kept whole (a name like ``id`` would
    otherwise produce no grams and never match).
    """
    grams = set()
    for word in _tokenize(text):
        if word in stopwords:
            continue
        if len(word) < 3:
            grams.add(word)
        else:
            for i in range(len(word) - 2):
                grams.add(word[i:i + 3])
    return frozenset(grams)


def contains(name_tri, q_tri, threshold=DEFAULT_THRESHOLD, floor=DEFAULT_FLOOR):
    """Does the question (``q_tri``) plausibly reference this name (``name_tri``)?

    * Tiny names (<=2 trigrams) use a subset test — the whole name must appear in
      the question, which keeps short names from matching on a single shared gram.
    * Larger names use an overlap ratio ``|name ∩ q| / |name|`` gated by both
      ``threshold`` and an absolute ``floor`` of shared grams.
    """
    if not name_tri or not q_tri:
        return False
    shared = len(name_tri & q_tri)
    if len(name_tri) <= 2:
        return name_tri <= q_tri
    if shared < floor:
        return False
    return shared / len(name_tri) >= threshold


def matches(name_norm, name_tri, q_padded, q_tri,
            threshold=DEFAULT_THRESHOLD, floor=DEFAULT_FLOOR):
    """Single entry point for "does the question reference this name?".

    Dispatches on the *name* side:

    * Names with ``>= floor`` trigrams take the trigram-containment path
      (:func:`contains`) — the existing subset / overlap-ratio behavior.
    * Shorter names can never clear the shared-gram floor (a 3-char name has one
      trigram, a 4-char name two), so they fall back to high-precision
      whole-word containment: every normalized name token must appear as a whole
      word in the space-padded question. This lets a ``lrt`` measure match
      "average lrt per station" without matching "filtration systems".

    ``name_norm`` is the space-joined normalized name tokens; ``q_padded`` is the
    once-per-query :func:`pad_question` output.
    """
    if name_tri and len(name_tri) >= floor:
        return contains(name_tri, q_tri, threshold, floor)
    toks = name_norm.split()
    if not toks:
        return False
    return all(f" {t} " in q_padded for t in toks)
