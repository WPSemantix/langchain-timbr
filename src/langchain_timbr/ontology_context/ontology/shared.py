"""Process-wide ``Ontology`` factory.

Both the static and dynamic SQL-generation paths benefit from a single shared
``Ontology`` per (url, token, ontology) triple — concept describes and
relationship-lookup fetches are then cached across every SQL-gen call inside
one process. This also makes the Plan 2 filtered-metadata cache
(``Ontology._filtered_cache``) reusable across the reasoning-retry loop in
``handle_generate_sql_reasoning`` — without it, each retry would re-run the
Step 1 LLM filter, doubling token spend per attempt. (The validation-retry loop
in ``handle_validate_generate_sql`` no longer rebuilds the context at all — it
regenerates from the context the invalid SQL came from.)
"""

from __future__ import annotations

from threading import Lock
from typing import Any, Dict, Tuple

from ...utils.timbr_utils import get_ontology_version
from .client import TimbrOntologyClient
from .graph import Ontology


_instances: Dict[Tuple[Any, ...], Ontology] = {}
_instances_lock = Lock()


def _cache_key(conn_params: dict) -> Tuple[Any, ...]:
    """Stable cache key for an Ontology instance.

    Three fields, and only three: what a graph *is* — a server, an ontology on
    it, and the tenant that ontology resolves to. Everything else describes how
    a caller reached it. An ``Ontology`` only ever fetches concept-level
    metadata — ``describe concept dtimbr.<name>``, ``sys_concept_relationships``
    and ``sys_ontology`` — and never views, so nothing it holds is
    permission-filtered: ``token`` is out, and so are ``verify_ssl`` (transport)
    and ``is_jwt`` (an authentication mechanism, not a data boundary — it cannot
    tell two JWT tenants apart, only ``jwt_tenant_id`` does that).
    ``additional_headers`` is likewise out: it varies per request (e.g.
    results-limit) but identifies the same backend graph.
    """
    return (
        conn_params.get("url"),
        conn_params.get("ontology"),
        conn_params.get("jwt_tenant_id"),
    )


def get_shared_ontology(conn_params: dict) -> Ontology:
    """Return the process-wide Ontology for this connection's current version.

    Thread-safe — the lookup-and-insert is guarded by a lock to avoid two threads
    creating duplicate instances during a cold start.

    **Replacement, not invalidation.** When the ontology version moves, a fresh
    Ontology is published here and the previous one is left alone. A caller that
    already holds the old instance keeps using it until its request finishes, so
    it sees one consistent generation of metadata rather than caches being
    emptied underneath it mid-flight. The old instance is collected once the last
    holder releases it.

    ``get_ontology_version`` returning ``None`` means the shared probe has no
    value and could not obtain one right now. That is "no information", not a
    change: keep serving whatever instance we have.
    """
    key = _cache_key(conn_params)
    version = get_ontology_version(conn_params)

    cached = _instances.get(key)
    if cached is not None and (version is None or cached.version_id == version):
        return cached

    with _instances_lock:
        cached = _instances.get(key)
        if cached is not None and (version is None or cached.version_id == version):
            return cached
        fresh = Ontology(TimbrOntologyClient(conn_params), version_id=version)
        _instances[key] = fresh
        return fresh


def reset_shared_ontologies() -> None:
    """Clear all shared Ontology instances. Intended for tests."""
    with _instances_lock:
        _instances.clear()
