"""Process-wide ``Ontology`` factory.

Both the static and dynamic SQL-generation paths benefit from a single shared
``Ontology`` per (url, token, ontology) triple — concept describes and
relationship-lookup fetches are then cached across every SQL-gen call inside
one process. This also makes the Plan 2 filtered-metadata cache
(``Ontology._filtered_cache``) reusable across the validation-retry loop in
``handle_validate_generate_sql`` — without it, each retry would re-run the
Step 1 LLM filter, doubling token spend per attempt.
"""

from __future__ import annotations

from threading import Lock
from typing import Any, Dict, Tuple

from .client import TimbrOntologyClient
from .graph import Ontology


_instances: Dict[Tuple[Any, ...], Ontology] = {}
_instances_lock = Lock()


def _cache_key(conn_params: dict) -> Tuple[Any, ...]:
    """Stable cache key for an Ontology instance.

    Tied to the connection identity (url, token, ontology, verify_ssl). We
    intentionally do NOT include things like additional_headers because those
    vary per request (e.g. results-limit) but identify the same backend graph.
    """
    return (
        conn_params.get("url"),
        conn_params.get("token"),
        conn_params.get("ontology"),
        bool(conn_params.get("verify_ssl", True)),
        bool(conn_params.get("is_jwt", False)),
        conn_params.get("jwt_tenant_id"),
    )


def get_shared_ontology(conn_params: dict) -> Ontology:
    """Return a process-wide Ontology shared across callers with the same conn.

    Thread-safe — the lookup-and-insert is guarded by a lock to avoid two
    threads creating duplicate instances during a cold start.
    """
    key = _cache_key(conn_params)
    cached = _instances.get(key)
    if cached is not None:
        return cached
    with _instances_lock:
        cached = _instances.get(key)
        if cached is None:
            cached = Ontology(TimbrOntologyClient(conn_params))
            _instances[key] = cached
    return cached


def reset_shared_ontologies() -> None:
    """Clear all shared Ontology instances. Intended for tests."""
    with _instances_lock:
        _instances.clear()
