"""Ontology graph — version-keyed lazy cache over concept describe output.

A single Ontology instance should be shared per (conn_params) tuple. The version
check is throttled to once per `cache_timeout` seconds (env: CACHE_TIMEOUT,
default 120s) so cache hits are pure in-memory dict lookups.
"""

from __future__ import annotations

import time
from collections import OrderedDict

from ...config import cache_timeout, filtered_cache_size, metadata_prefetch_workers
from .cardinality import derive_cardinality
from .models import ConceptMetadata, RelationshipLookupEntry
from .parser import parse_describe_output


def _split_csv(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = str(value).split(",")
    return [s.strip() for s in items if s and str(s).strip()]


def _to_bool(value) -> bool:
    """Coerce timbr bool-ish values (1/0, '1'/'0', True/False, 'true'/'false') to bool."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    return s in ("1", "true", "t", "yes", "y")


class Ontology:
    """Lazy, version-keyed concept-metadata cache.

    The constructor takes any client exposing the three required methods. Use
    the module-level factory (see Plan 3's get_shared_ontology) to share one
    Ontology per connection rather than instantiating directly in consumers.
    """

    def __init__(self, client, version_ttl_seconds: int | None = None):
        self._client = client
        self._version_id: str | None = None
        self._last_version_check: float = 0.0
        self._version_ttl: int = (
            version_ttl_seconds if version_ttl_seconds is not None else cache_timeout
        )
        self._cache: dict[str, ConceptMetadata] = {}
        # Concepts whose describe output failed to parse. Without this, a single
        # malformed concept costs a fresh `describe concept` round-trip on every
        # lookup — callers such as edge_index.outbound_edges swallow the error and
        # ask again for each relationship pointing at it. Cleared with _cache.
        self._failed: dict[str, Exception] = {}
        self._rel_lookup: dict[tuple[str, str], RelationshipLookupEntry] | None = None
        # concept_name -> tuple of parent concepts (root-to-direct order trimmed
        # below). Populated lazily alongside _rel_lookup. Empty tuple means
        # "no inheritance info" (either not in sys_ontology or the column is
        # blank for that concept).
        self._inheritance_lookup: dict[str, tuple[str, ...]] | None = None
        # Side cache for Plan 2 filtered-metadata results (Step 0+1 outputs).
        # Keyed by an arbitrary tuple (question, anchor, graph_depth, ...) chosen
        # by the caller. Invalidated together with concept cache on version change.
        # One entry per distinct question (~12 KB), so a worker that runs for
        # days without an ontology change needs a size bound as well: LRU, capped
        # at ``filtered_cache_size`` (env TIMBR_FILTERED_CACHE_SIZE).
        self._filtered_cache: "OrderedDict[tuple, object]" = OrderedDict()

    # ---- public API --------------------------------------------------------

    def show_version(self) -> str:
        self._ensure_version()
        return self._version_id or "unknown"

    def get_concept_metadata(self, name: str) -> ConceptMetadata:
        self._ensure_version()
        cached = self._cache.get(name)
        if cached is not None:
            return cached
        failure = self._failed.get(name)
        if failure is not None:
            # Same error the first attempt raised, without re-querying. The
            # traceback is reset so repeated raises can't accumulate frames.
            raise failure.with_traceback(None)
        try:
            rows = self._client.describe_concept(name)
            # Look up the inheritance chain BEFORE parsing so the parser can
            # fall back through parent concepts when resolving relationship
            # metadata (a relationship declared on ``organization`` must be
            # found when parsing ``company`` so its is_mtm / join-key signal
            # flows through to cardinality derivation).
            chain = (self._inheritance_lookup or {}).get(name, ())
            meta = parse_describe_output(
                name, rows,
                relationship_meta_lookup=self._rel_lookup,
                inheritance_chain=chain,
            )
        except Exception as exc:
            self._failed[name] = exc
            raise
        # Also surface the chain on ConceptMetadata for downstream callers
        # (the parser itself doesn't set this field — kept separate so the
        # parser stays independent of sys_ontology fetching).
        if chain:
            from dataclasses import replace
            meta = replace(meta, inheritance_chain=chain)
        self._cache[name] = meta
        return meta

    def prefetch(self, names, *, max_workers: int | None = None) -> None:
        """Warm the concept-metadata cache for ``names`` concurrently.

        ``describe concept`` is one HTTP round-trip per concept and a single
        subgraph walk needs dozens of them, strictly one after another. Callers
        that already know which concepts they are about to need can issue them
        as a few parallel waves instead.

        Best-effort and side-effect-free from the caller's point of view: a
        concept that fails to describe or parse is recorded exactly as a normal
        lookup would record it, and the error surfaces when that concept is
        actually requested. Never raises.
        """
        self._ensure_version()
        pending = [
            n for n in dict.fromkeys(names)
            if n and n not in self._cache and n not in self._failed
        ]
        if not pending:
            return

        workers = metadata_prefetch_workers if max_workers is None else max_workers
        if workers <= 1 or len(pending) == 1:
            for name in pending:
                self._prefetch_one(name)
            return

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(
            max_workers=min(workers, len(pending)),
            thread_name_prefix="timbr-describe",
        ) as pool:
            list(pool.map(self._prefetch_one, pending))

    def _prefetch_one(self, name: str) -> None:
        try:
            self.get_concept_metadata(name)
        except Exception:
            # Recorded in self._failed by get_concept_metadata; re-raised when
            # the concept is genuinely requested.
            pass

    def inheritance_chain_of(self, concept: str) -> tuple[str, ...]:
        """Return the cached inheritance chain for ``concept``.

        Direct accessor for callers (e.g. the concept-centric serializer) that
        need inheritance info for concepts they haven't fetched yet via
        get_concept_metadata. Returns empty tuple when the concept has no
        inheritance entry.
        """
        self._ensure_version()
        return (self._inheritance_lookup or {}).get(concept, ())

    def subconcepts_of(self, concept: str) -> list[str]:
        """Return concept names whose inheritance_chain contains ``concept``.

        Used by the serializer when include_logic_concepts is enabled to emit
        the per-concept ``subconcepts:`` line. Excludes ``concept`` itself.
        Result is lex-sorted for deterministic output.
        """
        self._ensure_version()
        out = [
            c for c, chain in (self._inheritance_lookup or {}).items()
            if c != concept and concept in chain
        ]
        return sorted(out)

    def cardinality_of(self, concept: str, relationship_name: str) -> str:
        """Return one of 'N:M' | 'N:1' | '1:N' | '1:1' for the relationship.

        Fetches source + target concept metadata (cached after first call).
        Raises KeyError if the relationship is not present on the source concept.
        """
        source_meta = self.get_concept_metadata(concept)
        if relationship_name not in source_meta.relationships:
            raise KeyError(
                f"Relationship {relationship_name!r} not found on concept {concept!r}"
            )
        rel = source_meta.relationships[relationship_name]
        target_meta = self.get_concept_metadata(rel.target_concept)
        source_pks = {p.name for p in source_meta.properties.values() if p.is_pk}
        target_pks = {p.name for p in target_meta.properties.values() if p.is_pk}
        return derive_cardinality(rel, source_pks=source_pks, target_pks=target_pks)

    def invalidate(self) -> None:
        """Force a fresh version check + relationship-lookup rebuild on next call."""
        self._version_id = None
        self._last_version_check = 0.0
        self._cache.clear()
        self._rel_lookup = None
        self._inheritance_lookup = None
        self._filtered_cache.clear()

    # ---- side cache for Plan 2 filtered-metadata results -------------------

    def get_filtered_cache(self, key: tuple):
        """Return a previously-stored filtered-metadata result, or None."""
        value = self._filtered_cache.get(key)
        if value is not None:
            self._filtered_cache.move_to_end(key)
        return value

    def set_filtered_cache(self, key: tuple, value) -> None:
        """Store a filtered-metadata result.

        Evicted on ontology version change, and least-recently-used first once
        the cache holds more than ``filtered_cache_size`` questions.
        """
        self._filtered_cache[key] = value
        self._filtered_cache.move_to_end(key)
        while len(self._filtered_cache) > filtered_cache_size:
            self._filtered_cache.popitem(last=False)

    # ---- internals ---------------------------------------------------------

    def _ensure_version(self) -> None:
        now = time.time()
        # Throttle SHOW VERSION to once per TTL window; matches the semantics of
        # cache_with_version_check at timbr_utils.py:96.
        if (now - self._last_version_check) > self._version_ttl:
            current = self._client.fetch_version_id()
            self._last_version_check = now
            if current != self._version_id:
                self._cache.clear()
                self._failed.clear()
                self._rel_lookup = None
                self._inheritance_lookup = None
                self._filtered_cache.clear()
                self._version_id = current
        if self._rel_lookup is None:
            self._rel_lookup = self._build_rel_lookup()
        if self._inheritance_lookup is None:
            self._inheritance_lookup = self._build_inheritance_lookup()

    def _build_rel_lookup(self) -> dict[tuple[str, str], RelationshipLookupEntry]:
        rows = self._client.fetch_relationships_meta() or []
        lookup: dict[tuple[str, str], RelationshipLookupEntry] = {}
        for r in rows:
            concept = r.get("concept")
            rel_name = r.get("relationship_name")
            if not concept or not rel_name:
                continue
            description_raw = r.get("description")
            description = (
                str(description_raw).strip() or None
                if description_raw is not None
                else None
            )
            lookup[(concept, rel_name)] = RelationshipLookupEntry(
                is_mtm=_to_bool(r.get("is_mtm")),
                is_inverse=_to_bool(r.get("is_inverse")),
                description=description,
                source_join_keys=tuple(_split_csv(r.get("source_properties"))),
                target_join_keys=tuple(_split_csv(r.get("target_properties"))),
            )
        return lookup

    def _build_inheritance_lookup(self) -> dict[str, tuple[str, ...]]:
        """Fetch ``sys_ontology.inheritance`` once per ontology version.

        Returns ``{concept_name: (parent_1, parent_2, ..., 'thing')}`` where the
        tuple is the parent chain as timbr emits it (typically root-direction
        ordered, ending at ``thing``).

        Tolerates the absence of ``fetch_inheritance_meta`` on the client —
        in that case, returns an empty dict (inheritance section in the
        serializer will degrade to ``(none)``).
        """
        fetcher = getattr(self._client, "fetch_inheritance_meta", None)
        if fetcher is None:
            return {}
        try:
            rows = fetcher() or []
        except Exception:
            return {}
        out: dict[str, tuple[str, ...]] = {}
        for r in rows:
            concept = r.get("concept")
            if not concept:
                continue
            chain = tuple(_split_csv(r.get("inheritance")))
            if chain:
                out[concept] = chain
        return out
