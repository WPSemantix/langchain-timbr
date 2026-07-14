"""Thin, dependency-light HTTP client for the Timbr knowledge-base search API.

`kbclient.py` is a clean, typed surface over ``POST /timbr/api/kb/search``. It
knows nothing about retrieval internals (no models, BM25, or embeddings) — its
only job is to be an ergonomic Python client with typed results, typed
exceptions, and an opt-in LRU cache that invalidates itself by polling
``MAX(changed_on)`` on Timbr system tables directly.

The module reuses existing repo connectivity patterns:
  * HTTP via ``requests`` with the same ``x-api-key`` / ``x-jwt-*`` header
    scheme used by ``utils.prompt_service.PromptService``.
  * SQL (cache invalidation + KB resolution) via ``utils.timbr_utils.run_query``
    (``conn_params`` -> ``pytimbr_api.timbr_http_connector``).

It is intentionally a separate, independent client — nothing else in the
package depends on it, and it depends only on ``config``, ``run_query``, and
``requests``.
"""

from __future__ import annotations

import hashlib
import logging
import random
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, FrozenSet, Iterable, List, Literal, Optional, Tuple

import requests
from requests.exceptions import (
    ConnectionError as _RequestsConnectionError,
    RequestException as _RequestException,
    Timeout as _RequestsTimeout,
)

from . import config
from .utils.timbr_utils import run_query

logger = logging.getLogger(__name__)

# Endpoint + system tables (single source of truth).
_SEARCH_PATH = "/timbr/api/kb/search"
_EXAMPLES_TABLE = "timbr.sys_knowledgebase_examples"
_AGENTS_TABLE = "timbr.sys_agents"
_KB_TABLE = "timbr.sys_knowledgebase"
_RULES_TABLE = "timbr.sys_knowledgebase_rules"

# API constraints (mirror the server contract so we can fail fast client-side).
_TOP_K_MIN = 1
_TOP_K_MAX = 20

# Knowledge-base rules: map the table's rule_type to an internal kind bucket, and
# the render labels/order for each kind (matches the frozen injection contract).
_RULE_KIND_BY_TYPE = {
    "SELECTION_RULE": "selection",
    "INSTRUCTION": "instruction",
    "VALIDATION": "validation",
}
_RULE_KINDS = ("selection", "instruction", "validation")
_RULE_LABELS = (
    ("selection", "selection_rules"),
    ("instruction", "instructions"),
    ("validation", "validation_rules"),
)


# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class KBMatch:
    """A single knowledge-base example returned by the search endpoint."""

    knowledge_base: str
    example_name: str
    question: str
    query: Optional[str]
    instructions: Optional[str]
    validate_sql: int
    confidence: float                       # sigmoid rerank score in [0, 1]
    changed_on: Optional[datetime]

    @property
    def has_query(self) -> bool:
        return bool(self.query and self.query.strip())

    @property
    def has_instructions(self) -> bool:
        return bool(self.instructions and self.instructions.strip())


@dataclass(frozen=True)
class KBSearchResult:
    """The parsed envelope of a successful ``/kb/search`` response.

    ``match_level`` and match filtering are decided server-side; the client
    only surfaces what the server returned.
    """

    matches: List[KBMatch]
    match_level: Literal["high", "medium", "none"]
    per_kb_max_changed_on: Dict[str, datetime]
    latency_ms: float

    @property
    def has_matches(self) -> bool:
        return len(self.matches) > 0

    @property
    def top(self) -> Optional[KBMatch]:
        return self.matches[0] if self.matches else None

    @property
    def is_high_confidence(self) -> bool:
        return self.match_level == "high"


@dataclass(frozen=True)
class RuleSet:
    """Knowledge-base rules indexed by ``(target_type, target_name)``.

    Threaded through the NL2SQL pipeline; each stage renders the
    matrix-appropriate subset per in-play object. An empty ``RuleSet`` is a
    no-op everywhere, which is how backward-compatibility / safe-fail is
    guaranteed (older backends without the rules table yield an empty set).

    Each ``by_target`` value maps an internal kind (``selection`` /
    ``instruction`` / ``validation``) to the list of that kind's rule texts for
    the object.
    """

    by_target: Dict[Tuple[str, str], Dict[str, List[str]]]
    kb_names: Tuple[str, ...]
    version: Optional[datetime]

    def is_empty(self) -> bool:
        return not self.by_target

    def rules_for(
        self,
        name: Optional[str],
        allowed_types: Iterable[str],
        kinds: Iterable[str],
    ) -> Dict[str, List[str]]:
        """Union of rules for ``name`` across ``allowed_types``, limited to ``kinds``.

        Kinds with no content are dropped, so an object with no matching rules
        yields ``{}`` and its render call returns ``""`` (empty-key suppression /
        no-op).
        """
        if not name or not self.by_target:
            return {}
        key_name = name.strip().lower()
        wanted = [k for k in kinds if k in _RULE_KINDS]
        out: Dict[str, List[str]] = {}
        for ttype in allowed_types:
            bucket = self.by_target.get((ttype.strip().lower(), key_name))
            if not bucket:
                continue
            for kind in wanted:
                texts = bucket.get(kind)
                if texts:
                    out.setdefault(kind, []).extend(texts)
        return out


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class KBClientError(Exception):
    """Base class for all kbclient errors.

    Carries ``status_code`` (HTTP status, or ``None`` for transport errors),
    ``message`` (the server's ``data`` field when available), and
    ``server_error_id`` (reserved for future correlation; ``None`` today).
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        server_error_id: Optional[str] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.server_error_id = server_error_id


class KBAuthError(KBClientError):
    """401 / 403 — invalid token or no access to the requested KB(s)."""


class KBBadRequestError(KBClientError):
    """400 — malformed request (bad question, top_k, thresholds, ...)."""


class KBOverloadError(KBClientError):
    """429 — rerank queue timeout / too many requests. Never auto-retried."""


class KBServerError(KBClientError):
    """5xx — model/dependency failure or unexpected server error."""


class KBTimeoutError(KBClientError):
    """Network / connection / read timeout talking to the API."""


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
class KBClient:
    """Sync HTTP client for the Timbr knowledge-base search endpoint.

    Safe to share across threads: the underlying ``requests.Session`` is reused
    and the LRU cache is guarded by an ``RLock``. Not safe across processes —
    construct one ``KBClient`` per process.
    """

    def __init__(
        self,
        base_url: str,
        auth_token: str,
        *,
        is_jwt: bool = False,
        jwt_tenant_id: Optional[str] = None,
        verify_ssl: bool = True,
        ontology: Optional[str] = None,
        conn_params: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = config.kb_search_timeout,
        max_retries: int = config.kb_max_retries,
        retry_backoff_seconds: float = 0.5,
        cache_ttl_seconds: Optional[int] = config.kb_cache_ttl_seconds,
        cache_max_entries: int = config.kb_cache_max_entries,
    ):
        if not base_url:
            raise ValueError("base_url is required")

        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self.is_jwt = is_jwt
        self.jwt_tenant_id = jwt_tenant_id
        self.verify_ssl = verify_ssl
        self.ontology = ontology
        # A caller-supplied conn_params overrides the assembled one; either way
        # it drives run_query for cache invalidation + KB resolution.
        self._conn_params_override = conn_params
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.cache_ttl_seconds = cache_ttl_seconds
        self.cache_max_entries = cache_max_entries

        self._session = requests.Session()
        self._lock = threading.RLock()
        # Cache maps key -> (stored_at_epoch, KBSearchResult).
        self._cache: "OrderedDict[tuple, tuple]" = OrderedDict()
        # Cache of the full KB -> ontologies map, keyed by that table's
        # MAX(changed_on) so it refreshes only when the KB catalog changes.
        self._kb_ontology_map: Optional[Dict[str, List[str]]] = None
        self._kb_ontology_map_stamp: Optional[datetime] = None

        if self.cache_ttl_seconds is not None and self._resolve_conn_params() is None:
            logger.info(
                "KBClient: cache enabled without DB access (no conn_params); "
                "running in pure-TTL mode. Cached results may be stale for up "
                "to cache_ttl_seconds."
            )

    @classmethod
    def from_conn_params(
        cls,
        conn_params: Dict[str, Any],
        ontology: Optional[str] = None,
    ) -> "KBClient":
        """Build a client from a pipeline ``conn_params`` dict.

        Single construction point reused by the memory KB flow and the rules
        fetch so auth/SSL/ontology are wired identically everywhere.
        """
        return cls(
            base_url=(conn_params.get("url") or config.url or "").rstrip("/"),
            auth_token=conn_params.get("token") or config.token or "",
            is_jwt=conn_params.get("is_jwt", False),
            jwt_tenant_id=conn_params.get("jwt_tenant_id"),
            verify_ssl=conn_params.get("verify_ssl", True),
            ontology=ontology or conn_params.get("ontology"),
            conn_params=conn_params,
        )

    # ---- connectivity helpers -------------------------------------------- #
    def _get_headers(self) -> Dict[str, str]:
        """Auth headers, matching utils.prompt_service.PromptService."""
        headers = {"Content-Type": "application/json"}
        if self.is_jwt:
            headers["x-jwt-token"] = self.auth_token
            if self.jwt_tenant_id:
                headers["x-jwt-tenant-id"] = self.jwt_tenant_id
        elif self.auth_token:
            headers["x-api-key"] = self.auth_token
        return headers

    def _resolve_conn_params(self) -> Optional[Dict[str, Any]]:
        """Assemble conn_params for run_query, or None if DB access is absent.

        A caller can pass a fully-formed ``conn_params`` dict; otherwise one is
        built from the client's own connection settings. Returns None only when
        neither is usable (so caching degrades to pure TTL).
        """
        if self._conn_params_override:
            return self._conn_params_override
        if not self.base_url or not self.auth_token:
            return None
        return {
            "url": self.base_url,
            "token": self.auth_token,
            "ontology": self.ontology or config.ontology,
            "is_jwt": self.is_jwt,
            "jwt_tenant_id": self.jwt_tenant_id,
            "verify_ssl": self.verify_ssl,
        }

    # ---- KB resolution (fail-safe: warn + continue, never break flow) ---- #
    def resolve_knowledge_bases(
        self,
        *,
        agent: Optional[str] = None,
        ontology: Optional[str] = None,
    ) -> List[str]:
        """Resolve the knowledge_base names available to a Data Agent / ontology.

        * ``agent`` given -> read ``sys_agents.knowledge_bases`` (CSV).
        * else ``ontology`` given -> keep KBs whose ``sys_knowledgebase.ontologies``
          CSV contains that ontology.

        Every failure mode (missing column, missing table, no DB access, query
        error) is swallowed with a warning and yields ``[]`` so the caller's
        flow is never broken.
        """
        conn_params = self._resolve_conn_params()
        if conn_params is None:
            logger.warning("KBClient: cannot resolve knowledge bases without DB access.")
            return []
        if agent:
            return self._resolve_from_agent(agent, conn_params)
        if ontology:
            return self._resolve_from_ontology(ontology, conn_params)
        return []

    def _resolve_from_agent(self, agent: str, conn_params: Dict[str, Any]) -> List[str]:
        safe_agent = agent.replace("'", "''")
        sql = (
            f"SELECT knowledge_bases FROM {_AGENTS_TABLE} "
            f"WHERE LOWER(agent_name) = LOWER('{safe_agent}')"
        )
        try:
            rows = run_query(sql, conn_params)
        except Exception as exc:  # old versions lack the column / table
            logger.warning(
                "KBClient: could not read knowledge_bases for agent '%s' "
                "(older Timbr version?); continuing without KB context: %s",
                agent,
                exc,
            )
            return []
        if not rows:
            return []
        return _split_csv(rows[0].get("knowledge_bases"))

    def _resolve_from_ontology(self, ontology: str, conn_params: Dict[str, Any]) -> List[str]:
        kb_map = self._get_kb_ontology_map(conn_params)
        if not kb_map:
            return []
        target = ontology.strip().lower()
        return [kb for kb, onts in kb_map.items() if target in onts]

    def _get_kb_ontology_map(self, conn_params: Dict[str, Any]) -> Dict[str, List[str]]:
        """KB -> [lowercased ontology names], cached by MAX(changed_on).

        Fetches the whole ``sys_knowledgebase`` table in one query so a single
        round trip covers every KB. The cached copy is reused until the table's
        ``MAX(changed_on)`` advances.
        """
        sql = f"SELECT knowledge_base, ontologies, changed_on FROM {_KB_TABLE}"
        try:
            rows = run_query(sql, conn_params)
        except Exception as exc:  # table absent on older versions
            logger.warning(
                "KBClient: could not read %s (older Timbr version?); "
                "continuing without KB context: %s",
                _KB_TABLE,
                exc,
            )
            return {}

        max_changed = _max_changed_on(rows)
        with self._lock:
            if (
                self._kb_ontology_map is not None
                and self._kb_ontology_map_stamp == max_changed
            ):
                return self._kb_ontology_map

            kb_map: Dict[str, List[str]] = {}
            for row in rows:
                name = row.get("knowledge_base")
                if not name:
                    continue
                kb_map[name] = [o.lower() for o in _split_csv(row.get("ontologies"))]
            self._kb_ontology_map = kb_map
            self._kb_ontology_map_stamp = max_changed
            return kb_map

    # ---- search ---------------------------------------------------------- #
    def search(
        self,
        question: str,
        knowledge_bases: Optional[List[str]] = None,
        *,
        agent: Optional[str] = None,
        ontology: Optional[str] = None,
        top_k: int = config.kb_top_k,
        high_threshold: float = config.kb_high_threshold,
        medium_threshold: float = config.kb_medium_threshold,
    ) -> KBSearchResult:
        """Search the given knowledge bases for examples matching ``question``.

        When ``knowledge_bases`` is omitted it is auto-resolved from ``agent`` /
        ``ontology`` (see ``resolve_knowledge_bases``). Raises ``KBClientError``
        subclasses on API failure — callers decide how to degrade.
        """
        if not question or not question.strip():
            raise KBBadRequestError("'question' is required and must be a non-empty string")

        if knowledge_bases is None:
            knowledge_bases = self.resolve_knowledge_bases(agent=agent, ontology=ontology)
        if not knowledge_bases or not all(isinstance(k, str) for k in knowledge_bases):
            raise KBBadRequestError(
                "'knowledge_bases' is required and must be a non-empty list of strings"
            )

        self._validate_params(top_k, high_threshold, medium_threshold)

        kb_names = sorted(set(knowledge_bases))
        cache_key = self._cache_key(question, kb_names, top_k, high_threshold, medium_threshold)

        cached = self._cache_get(cache_key, kb_names)
        if cached is not None:
            return cached

        result = self._post_search(
            question, kb_names, top_k, high_threshold, medium_threshold
        )
        self._cache_put(cache_key, result)
        return result

    @staticmethod
    def _validate_params(top_k: int, high_threshold: float, medium_threshold: float) -> None:
        if not isinstance(top_k, int):
            raise KBBadRequestError("'top_k' must be an integer")
        if not (_TOP_K_MIN <= top_k <= _TOP_K_MAX):
            raise KBBadRequestError(f"'top_k' must be between {_TOP_K_MIN} and {_TOP_K_MAX}")
        if not (0.0 <= high_threshold <= 1.0):
            raise KBBadRequestError("'high_threshold' must be between 0.0 and 1.0")
        if not (0.0 <= medium_threshold <= high_threshold):
            raise KBBadRequestError(
                "'medium_threshold' must be between 0.0 and high_threshold"
            )

    def _post_search(
        self,
        question: str,
        knowledge_bases: List[str],
        top_k: int,
        high_threshold: float,
        medium_threshold: float,
    ) -> KBSearchResult:
        url = f"{self.base_url}{_SEARCH_PATH}"
        body = {
            "question": question,
            "knowledge_bases": knowledge_bases,
            "top_k": top_k,
            "high_threshold": high_threshold,
            "medium_threshold": medium_threshold,
        }
        headers = self._get_headers()

        attempt = 0
        while True:
            try:
                response = self._session.post(
                    url,
                    json=body,
                    headers=headers,
                    timeout=self.timeout_seconds,
                    verify=self.verify_ssl,
                )
            except (_RequestsTimeout, _RequestsConnectionError) as exc:
                # Connection errors are retryable; a pure Timeout is not.
                if isinstance(exc, _RequestsConnectionError) and attempt < self.max_retries:
                    self._sleep_backoff(attempt)
                    attempt += 1
                    continue
                raise KBTimeoutError(f"KB search request failed: {exc}") from exc
            except _RequestException as exc:
                raise KBClientError(f"KB search request failed: {exc}") from exc

            status = response.status_code
            if status == 200:
                return self._parse_result(response)

            if status in (502, 503, 504) and attempt < self.max_retries:
                self._sleep_backoff(attempt)
                attempt += 1
                continue

            raise self._map_error(response)

    def _sleep_backoff(self, attempt: int) -> None:
        delay = self.retry_backoff_seconds * (2 ** attempt)
        delay += random.uniform(0, self.retry_backoff_seconds)
        time.sleep(delay)

    @staticmethod
    def _map_error(response: requests.Response) -> KBClientError:
        status = response.status_code
        message = KBClient._error_message(response)
        if status == 400:
            return KBBadRequestError(message, status_code=status)
        if status in (401, 403):
            return KBAuthError(message, status_code=status)
        if status == 429:
            return KBOverloadError(message, status_code=status)
        if status >= 500:
            return KBServerError(message, status_code=status)
        return KBClientError(message, status_code=status)

    @staticmethod
    def _error_message(response: requests.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            return response.text or f"HTTP {response.status_code}"
        if isinstance(body, dict) and "data" in body:
            return str(body["data"])
        return str(body)

    @staticmethod
    def _parse_result(response: requests.Response) -> KBSearchResult:
        try:
            body = response.json()
        except ValueError as exc:
            raise KBServerError(f"Invalid JSON in KB search response: {exc}") from exc
        if not isinstance(body, dict):
            raise KBServerError("Unexpected KB search response shape (expected object)")

        matches = [
            KBMatch(
                knowledge_base=m.get("knowledge_base"),
                example_name=m.get("example_name"),
                question=m.get("question"),
                query=m.get("query"),
                instructions=m.get("instructions"),
                validate_sql=int(m.get("validate_sql", 0)),
                confidence=float(m.get("confidence", 0.0)),
                changed_on=_parse_dt(m.get("changed_on")),
            )
            for m in body.get("matches", [])
        ]
        per_kb = {
            kb: dt
            for kb, raw in (body.get("per_kb_max_changed_on") or {}).items()
            if (dt := _parse_dt(raw)) is not None
        }
        return KBSearchResult(
            matches=matches,
            match_level=body.get("match_level", "none"),
            per_kb_max_changed_on=per_kb,
            latency_ms=float(body.get("latency_ms", 0.0)),
        )

    # ---- cache ----------------------------------------------------------- #
    def _cache_key(
        self,
        question: str,
        kb_names: List[str],
        top_k: int,
        high_threshold: float,
        medium_threshold: float,
    ) -> tuple:
        q_hash = hashlib.sha256(question.strip().lower().encode("utf-8")).hexdigest()
        auth_hash = hashlib.sha256((self.auth_token or "").encode("utf-8")).hexdigest()
        return (q_hash, tuple(kb_names), top_k, high_threshold, medium_threshold, auth_hash)

    def _cache_get(self, key: tuple, kb_names: List[str]) -> Optional[KBSearchResult]:
        if self.cache_ttl_seconds is None:
            return None
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            stored_at, result = entry
            if (time.time() - stored_at) > self.cache_ttl_seconds:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)

        # TTL still valid — verify against the DB when we can (safe mode).
        if not self._is_still_fresh(result, kb_names):
            with self._lock:
                self._cache.pop(key, None)
            return None
        return result

    def _is_still_fresh(self, result: KBSearchResult, kb_names: List[str]) -> bool:
        """True if no involved KB has changed since the result was cached.

        Runs a single grouped ``MAX(changed_on)`` query over all KBs. Without DB
        access we fall back to pure TTL (already checked) and treat as fresh.
        """
        conn_params = self._resolve_conn_params()
        if conn_params is None:
            return True
        current = self._fetch_max_changed_on(kb_names, conn_params)
        if current is None:  # query failed — don't serve possibly-stale data
            return False
        for kb, cached_dt in result.per_kb_max_changed_on.items():
            server_dt = current.get(kb)
            if server_dt is not None and server_dt > cached_dt:
                return False
        return True

    def _fetch_max_changed_on(
        self, kb_names: List[str], conn_params: Dict[str, Any]
    ) -> Optional[Dict[str, datetime]]:
        in_list = ", ".join("'" + k.replace("'", "''") + "'" for k in kb_names)
        sql = (
            f"SELECT knowledge_base, MAX(changed_on) AS max_changed_on "
            f"FROM {_EXAMPLES_TABLE} "
            f"WHERE knowledge_base IN ({in_list}) "
            f"AND (query IS NOT NULL OR instructions IS NOT NULL) "
            f"GROUP BY knowledge_base"
        )
        try:
            rows = run_query(sql, conn_params)
        except Exception as exc:
            logger.warning("KBClient: cache-invalidation query failed: %s", exc)
            return None
        out: Dict[str, datetime] = {}
        for row in rows:
            name = row.get("knowledge_base")
            dt = _parse_dt(row.get("max_changed_on"))
            if name and dt is not None:
                out[name] = dt
        return out

    def _cache_put(self, key: tuple, result: KBSearchResult) -> None:
        if self.cache_ttl_seconds is None:
            return
        with self._lock:
            self._cache[key] = (time.time(), result)
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_max_entries:
                self._cache.popitem(last=False)

    def invalidate_cache(self, knowledge_bases: Optional[List[str]] = None) -> None:
        """Force-drop cached entries. ``None`` drops all; a list drops entries
        that touch any of the named KBs."""
        with self._lock:
            if knowledge_bases is None:
                self._cache.clear()
                return
            targets = set(knowledge_bases)
            for key in [k for k in self._cache if targets.intersection(k[1])]:
                del self._cache[key]

    # ---- lifecycle ------------------------------------------------------- #
    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "KBClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# Module helpers
# --------------------------------------------------------------------------- #
def _split_csv(value: Optional[str]) -> List[str]:
    if not value or not isinstance(value, str):
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_dt(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp (Z-suffixed or naive) into a datetime.

    Returns the value unchanged if already a datetime, and ``None`` for null /
    unparseable input. The trailing ``Z`` is normalized to ``+00:00`` for
    Python 3.10 ``fromisoformat`` compatibility.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _max_changed_on(rows: List[Dict[str, Any]]) -> Optional[datetime]:
    stamps = [dt for row in rows if (dt := _parse_dt(row.get("changed_on"))) is not None]
    return max(stamps) if stamps else None


# --------------------------------------------------------------------------- #
# Knowledge-base rules — fetch, cache, render
# --------------------------------------------------------------------------- #
# Session-level cache keyed by the resolved KB set. Lives at module scope (not on
# a KBClient instance) so it survives the per-stage client rebuilds across the
# identify + generate chains within one process. Guarded independently of the
# per-instance search cache.
_rules_cache: "Dict[FrozenSet[str], tuple]" = {}
_rules_cache_lock = threading.Lock()
_EMPTY_RULESET = RuleSet(by_target={}, kb_names=(), version=None)


def fetch_rules(
    conn_params: Optional[Dict[str, Any]],
    *,
    agent: Optional[str] = None,
    ontology: Optional[str] = None,
) -> RuleSet:
    """Fetch + index knowledge-base rules for the current session.

    Resolves the applicable KBs (agent-first, else ontology) exactly like the
    memory KB flow, queries ``sys_knowledgebase_rules`` once, and caches the
    indexed result keyed by the resolved KB set (short TTL + ``MAX(changed_on)``
    re-validation). Every failure mode — kill-switch off, no DB access, KB
    resolution failure, missing table on an older backend — returns an empty
    ``RuleSet`` so the pipeline runs exactly as before.
    """
    if not getattr(config, "enable_knowledge_base", True):
        return _EMPTY_RULESET
    if not conn_params:
        return _EMPTY_RULESET

    try:
        client = KBClient.from_conn_params(conn_params, ontology)
    except Exception as exc:
        logger.debug("KBClient: rules client construction failed: %s", exc)
        return _EMPTY_RULESET
    try:
        effective_ontology = ontology or conn_params.get("ontology")
        kb_names = client.resolve_knowledge_bases(
            agent=agent, ontology=None if agent else effective_ontology
        )
    finally:
        client.close()

    kb_set = frozenset(k for k in (kb_names or []) if k)
    if not kb_set:
        return _EMPTY_RULESET
    return _get_or_load_rules(kb_set, conn_params)


def _get_or_load_rules(
    kb_set: FrozenSet[str], conn_params: Dict[str, Any]
) -> RuleSet:
    ttl = getattr(config, "kb_rules_cache_ttl_seconds", 60) or 0
    now = time.time()

    with _rules_cache_lock:
        entry = _rules_cache.get(kb_set)
    if entry is not None:
        stored_at, cached = entry
        if ttl <= 0 or (now - stored_at) <= ttl:
            return cached
        # TTL expired — re-validate cheaply against MAX(changed_on) (no lock held
        # during the query). Reload only when the stamp advanced.
        current_version = _fetch_rules_version(kb_set, conn_params)
        if (
            current_version is not None
            and cached.version is not None
            and current_version <= cached.version
        ):
            with _rules_cache_lock:
                _rules_cache[kb_set] = (time.time(), cached)
            return cached

    ruleset = _load_rules(kb_set, conn_params)
    with _rules_cache_lock:
        _rules_cache[kb_set] = (time.time(), ruleset)
    return ruleset


def _rules_in_list(kb_set: FrozenSet[str]) -> str:
    return ", ".join("'" + k.replace("'", "''") + "'" for k in sorted(kb_set))


def _fetch_rules_version(
    kb_set: FrozenSet[str], conn_params: Dict[str, Any]
) -> Optional[datetime]:
    sql = (
        f"SELECT MAX(changed_on) AS max_changed_on FROM {_RULES_TABLE} "
        f"WHERE knowledge_base IN ({_rules_in_list(kb_set)})"
    )
    try:
        rows = run_query(sql, conn_params)
    except Exception as exc:
        logger.debug("KBClient: rules version query failed: %s", exc)
        return None
    if not rows:
        return None
    return _parse_dt(rows[0].get("max_changed_on"))


def _load_rules(kb_set: FrozenSet[str], conn_params: Dict[str, Any]) -> RuleSet:
    sql = (
        f"SELECT knowledge_base, rule_name, rule_type, target_name, target_type, "
        f"instructions, changed_on FROM {_RULES_TABLE} "
        f"WHERE knowledge_base IN ({_rules_in_list(kb_set)})"
    )
    try:
        rows = run_query(sql, conn_params)
    except Exception as exc:  # table absent on older versions
        logger.debug(
            "KBClient: could not read %s (older Timbr version?); "
            "continuing without KB rules: %s",
            _RULES_TABLE,
            exc,
        )
        return RuleSet(by_target={}, kb_names=tuple(sorted(kb_set)), version=None)

    by_target: Dict[Tuple[str, str], Dict[str, List[str]]] = {}
    for row in rows or []:
        target_type = (row.get("target_type") or "").strip().lower()
        target_name = (row.get("target_name") or "").strip().lower()
        rule_type = (row.get("rule_type") or "").strip().upper()
        text = row.get("instructions")
        kind = _RULE_KIND_BY_TYPE.get(rule_type)
        if not target_type or not target_name or not kind or not (text and text.strip()):
            continue
        bucket = by_target.setdefault((target_type, target_name), {})
        bucket.setdefault(kind, []).append(text.strip())

    return RuleSet(
        by_target=by_target,
        kb_names=tuple(sorted(kb_set)),
        version=_max_changed_on(rows or []),
    )


def render_object_rules(rules: Dict[str, List[str]], *, indent: str = "") -> str:
    """Render one object's rules as the discrete sub-block (empty-key suppressed).

    Returns ``""`` when no kind has content — this is the single guarantee that
    keeps every prompt byte-identical when rules are absent.
    """
    lines: List[str] = []
    for kind, label in _RULE_LABELS:
        texts = [t.strip() for t in rules.get(kind, []) if t and t.strip()]
        if texts:
            lines.append(f"{indent}{label}: {'; '.join(texts)}")
    return "\n".join(lines)


def clear_rules_cache() -> None:
    """Drop the session rules cache (test hook / manual invalidation)."""
    with _rules_cache_lock:
        _rules_cache.clear()
