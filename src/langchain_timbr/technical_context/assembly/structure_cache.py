"""Per-column matcher structures, built once and shared across requests.

The three matchers each build a lookup structure from a column's values before
they ever look at the question:

    exact       {normalized_value: original}
    substring   an Aho-Corasick automaton over the normalized values
    fuzzy       the (normalized choices, originals) split

All three are **pure functions of the column's values**, so they are identical
on every request — and until now all three were rebuilt on every request.
Measured on the captured fixture, that rebuild is **51% of all matcher CPU**,
and the automaton alone is 40% of it (567.8 ms of 1440.7 ms over 400 columns).

Caching them changes no results: the matchers receive the same structures they
would have built. The whole ontology's structures measured ~65 MB by RSS, 93%
of it the automaton — the exact dict and the fuzzy choices hold references to
strings the statistics cache already owns, so their real cost is a pointer per
value.

Keying follows ``multi_match._RESULT_CACHE``: cheap fields in the key, the
values tuple verified on hit. The statistics cache hands back the *same* string
objects between passes, so both the hash and the verification are pointer walks.

Thread-safety: entries are immutable once built and a shared automaton is safe
for concurrent ``iter()`` (verified: 32 threads x 20 iterations over one
automaton, results identical to serial). Two threads may build the same entry
concurrently and one write wins; both structures are equivalent, so the race is
benign and a lock on this path would serialise exactly what we are trying to
parallelise.
"""

from __future__ import annotations

from collections import OrderedDict

from ...config import cache_match_structures
from ..matching.ahocorasick_matcher import build_automaton
from ..matching.exact import build_exact_lookup
from ..matching.rapidfuzz_matcher import build_fuzzy_choices

# Bounded by total cached values as well as entry count. The value counter is
# incremented without a lock, so concurrent callers can lose an update and let
# it drift low; ``len()`` cannot. Same belt-and-braces as _RESULT_CACHE.
#
# 500k values is ~100 MB at the 195 bytes/value measured, comfortably above the
# 336k values of the whole captured ontology and well under StatsCache's 500 MB.
_MAX_VALUES = 500_000
_MAX_ENTRIES = 4096


class _Entry:
    """Lazily-built structures for one column's values."""
    __slots__ = ("values_key", "n_values", "_exact", "_automaton", "_choices")

    def __init__(self, values_key: tuple, n_values: int):
        self.values_key = values_key
        self.n_values = n_values
        self._exact = None
        self._automaton = _UNSET
        self._choices = None

    def exact(self, values, normalized, normalized_stripped):
        if self._exact is None:
            self._exact = build_exact_lookup(values, normalized, normalized_stripped)
        return self._exact

    def automaton(self, values, normalized_space, min_length):
        # None is a real answer here (no value long enough), so a distinct
        # sentinel is needed to tell "not built" from "built, and empty".
        if self._automaton is _UNSET:
            self._automaton = build_automaton(values, normalized_space, min_length)
        return self._automaton

    def choices(self, values, normalized):
        if self._choices is None:
            self._choices = build_fuzzy_choices(values, normalized)
        return self._choices


_UNSET = object()

_CACHE: "OrderedDict[tuple, _Entry]" = OrderedDict()
_CACHE_VALUES = 0


def clear_structure_cache() -> None:
    """Drop every cached structure. Used by tests for isolation."""
    global _CACHE_VALUES
    _CACHE.clear()
    _CACHE_VALUES = 0


def cache_stats() -> dict:
    """Entry count and total cached values — for tests and diagnostics."""
    return {"entries": len(_CACHE), "values": _CACHE_VALUES}


def get_entry(
    column_name: str,
    known_values: list,
    value_kind: str,
    values_key: tuple | None = None,
    ontology: str | None = None,
    schema: str | None = None,
    concept: str | None = None,
) -> _Entry | None:
    """The cache entry for this column's values, or None when caching is off.

    The key identifies **which column this is**, not what is in it:
    ``(ontology, schema, concept, column_name, len, value_kind)``. All six are
    O(1) to key on. What the column *contains* is settled by the ``values_key``
    comparison below, which is what catches a statistics refresh or a version
    change within one identity.

    Every part of that identity earns its place:

    - ``ontology`` — two tenants sharing a column name and value count would
      otherwise collide, fail each other's verification, and rebuild on **every**
      request. Correct, but the cache silently stops caching and it shows up only
      as latency in production.
    - ``schema`` / ``concept`` — ``column_name`` is not unique within an
      ontology. The same property name exists under different concepts
      (``status`` on ``orders`` vs on ``deliveries``) and different schemas
      (``dtimbr`` vs ``vtimbr``), with different values.
    - ``value_kind`` — decides whether ``normalized_stripped`` contributes to the
      exact lookup, so an entry built under one kind must not serve another.

    Keying on ``hash(values)`` instead was the first implementation and was
    worse on both counts: O(n) per lookup (measured 2.8% of matcher stage time),
    and a version change would mint a new key and leave the previous generation
    orphaned until the LRU reclaimed it, rather than replacing it in place.

    ``values_key`` lets the caller pass the values tuple it has already built
    (``run_all_matchers`` builds one for its own memo), so a cache hit costs one
    dict lookup and a pointer-wise tuple compare rather than a second tuple.

    The identity arguments are optional so that tests and probes can call the
    matchers directly. Omitting them only makes the key less discriminating —
    never incorrect, because the values comparison is the backstop.
    """
    if not cache_match_structures:
        return None

    global _CACHE_VALUES
    if values_key is None:
        values_key = tuple(known_values)
    key = (ontology, schema, concept, column_name, len(known_values), value_kind)

    entry = _CACHE.get(key)
    if entry is not None and entry.values_key == values_key:
        _CACHE.move_to_end(key)
        return entry

    # Miss, or the column's values changed under the same name (a new ontology
    # version) — rebuild rather than serve a stale structure.
    if entry is not None:
        # Replacing an entry under an existing key. Subtract what it contributed
        # before adding the replacement, or the counter drifts UP by the old
        # entry's size on every ontology version change — and a counter that
        # reads high causes premature eviction, emptying a cache that is not
        # actually full. (The `len()` bound below guards the opposite drift,
        # where a lost concurrent increment reads low.)
        _CACHE_VALUES = max(0, _CACHE_VALUES - entry.n_values)

    entry = _Entry(values_key, len(known_values))
    while _CACHE and (
        _CACHE_VALUES >= _MAX_VALUES or len(_CACHE) >= _MAX_ENTRIES
    ):
        _, evicted = _CACHE.popitem(last=False)
        _CACHE_VALUES = max(0, _CACHE_VALUES - evicted.n_values)
    _CACHE[key] = entry
    _CACHE_VALUES += entry.n_values
    return entry
