import os
import re
from typing import Optional, Any
import inspect
import threading
import time
import base64
import hashlib
import requests
from pytimbr_api import timbr_http_connector
from functools import wraps
from cryptography.fernet import Fernet

from ..config import (
    cache_timeout, per_user_cache_ttl, ignore_tags, ignore_tags_prefix,
    http_keepalive, http_pool_maxsize, http_connect_timeout, http_read_timeout,
)
from .general import to_boolean

class _Snapshot:
    """One generation of cached entries for one ontology.

    The three maps travel together and are swapped as a single reference. Keeping
    ``inflight`` outside the snapshot would be a live bug rather than untidiness:
    a waiter would block on a marker whose result is written into a generation it
    no longer reads, and wait out the full ``waiter.wait(timeout=cache_timeout)``
    — two minutes.
    """

    __slots__ = ("cache", "expiry", "inflight")

    def __init__(self):
        self.cache: dict = {}
        # Absolute deadlines for the entries that have one (the per-user tier);
        # shared entries are absent and live until the version changes.
        self.expiry: dict = {}
        self.inflight: dict = {}


# Version-gated entries, one snapshot per ontology. A version change *replaces*
# an ontology's snapshot; readers already holding the previous one finish against
# it, so a slow call cannot write a pre-change result into the fresh generation.
_snapshots: dict = {}

# Entries declared ``version_gated=False`` — read from the statistics table,
# which is recomputed on its own schedule. A DDL version bump is neither
# necessary nor sufficient to invalidate them, so this store
# is **never swapped**; its entries are governed by their own per-user TTL.
_exempt_store: dict = {}

# The version each ontology's snapshot was built against. A single scalar here
# meant that with two ontologies in rotation the recorded version regularly
# belonged to the other one, so the comparison failed and the whole process-wide
# cache was dropped roughly every ``cache_timeout``.
_ontology_version: dict = {}

# Guards creation and replacement of snapshot objects — held only around dict
# operations on the maps above, never across a call.
_snapshots_lock = threading.Lock()

# Guards the entry-level maps inside a snapshot. Under a threaded server every
# request thread misses a cold cache at the same moment and every one of them
# runs the underlying query — measured at 150 Timbr queries for a single cached
# call with 50 threads. ``_cache_lock`` makes the read/write atomic; each
# snapshot's ``inflight`` turns a key into a single-flight, so the first thread
# computes and the rest wait for its result instead of duplicating it.
#
# This is a **leaf lock**: nothing is ever called while it is held, which is what
# keeps it out of any deadlock cycle. Keep it that way.
_cache_lock = threading.Lock()

# --- Timbr SQL transport -----------------------------------------------------
# pytimbr_api issues every query through the module-level ``requests.post``,
# which builds a throwaway Session — a fresh TCP connection and TLS handshake
# per query — and asks the server to close it (``Connection: close``). A single
# NL2SQL run makes dozens of queries, so on an HTTPS endpoint that handshake
# accounts for most of the Timbr wait. We keep one pooled Session for the
# process instead. Same URL, same headers minus ``Connection``, same response
# parsing (``timbr_http_connector._parse_response``), so behaviour is unchanged
# apart from the socket being reused.
_session: Optional[requests.Session] = None
_session_lock = threading.Lock()


def _get_session() -> requests.Session:
    """Return the process-wide pooled Session, creating it on first use."""
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                session = requests.Session()
                # connect-only retries: a pooled socket the server closed while
                # idle fails before the request is sent, so retrying cannot
                # re-execute a query.
                retry = requests.adapters.Retry(
                    total=1, connect=1, read=0, status=0,
                    allowed_methods=False, backoff_factor=0.1,
                )
                adapter = requests.adapters.HTTPAdapter(
                    pool_connections=http_pool_maxsize,
                    pool_maxsize=http_pool_maxsize,
                    max_retries=retry,
                )
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                _session = session
    return _session


def close_session() -> None:
    """Close the pooled Session and drop it. Next query opens a fresh one."""
    global _session
    with _session_lock:
        if _session is not None:
            try:
                _session.close()
            finally:
                _session = None


def _run_query_pooled(
    url: str,
    ontology: str,
    token: str,
    query,
    datasource: str = None,
    nested: str = 'false',
    verify_ssl: bool = True,
    enable_IPv6: bool = False,
    is_jwt: bool = False,
    jwt_tenant_id: str = None,
    additional_headers: dict = None,
):
    """Keep-alive equivalent of ``timbr_http_connector.run_query``.

    Mirrors that function's URL, headers, body encoding and response parsing
    exactly; the only differences are the reused connection pool and the absent
    ``Connection: close`` header.
    """
    datasource_addition = f'?datasource={datasource}' if datasource else ''

    base_url = url if url.endswith('/') else url + '/'

    headers = {
        'Content-Type': 'application/text',
        'nested': nested,
    }

    if is_jwt:
        headers['x-jwt-token'] = token
        if jwt_tenant_id:
            headers['x-jwt-tenant-id'] = jwt_tenant_id
    else:
        headers['x-api-key'] = token

    if additional_headers:
        for key, value in additional_headers.items():
            headers[key.replace('_', '-')] = value

    requests.packages.urllib3.util.connection.HAS_IPV6 = enable_IPv6
    response = _get_session().post(
        f'{base_url}timbr/openapi/ontology/{ontology}/query{datasource_addition}',
        headers=headers,
        data=query.encode('utf-8') if isinstance(query, str) else query,
        verify=verify_ssl,
        timeout=(http_connect_timeout, http_read_timeout),
    )
    return timbr_http_connector._parse_response(response)


def build_server_url(url: str, thrift_host: str, thrift_port: int) -> str:
    """Build the timbr server URL from thrift host and port for API logging calls."""
    if ("0.0.0.0" in url or "localhost" in url or os.environ.get("THRIFT_HOST")):
        thrift_host = os.environ.get('THRIFT_HOST', 'timbr-server')
        if thrift_host.startswith("http"):
            url = f"{thrift_host.rstrip('/')}:{os.environ.get('THRIFT_PORT', '11000')}"
        else:
            url = f"http://{thrift_host}:{os.environ.get('THRIFT_PORT', '11000')}"
    
    return url


def clear_cache():
    """Drop every cached entry, in both stores, for every ontology.

    An explicit, process-wide reset. Note what it is *not*: the ontology-version
    path no longer comes through here — a version change replaces one ontology's
    snapshot (``_swap_snapshot``) and leaves every other ontology, and the exempt
    store, untouched.

    A thread already holding a snapshot finishes its call against it. That is the
    point of replacing rather than emptying, so this is not an immediate barrier
    for work already in flight.
    """
    with _snapshots_lock:
        _snapshots.clear()
        _exempt_store.clear()
        _ontology_version.clear()


def _get_ontology_version(conn_params) -> str:
    """Fetch the current ontology version."""
    query = "SHOW VERSION"
    res = run_query(query, conn_params)
    return res[0].get("id") if res else "unknown"


# --- shared ontology-version probe -------------------------------------------
# Two consumers ask the server "what version is this ontology?" — this module's
# query cache and ontology_context's Ontology graph — and each used to throttle
# its own SHOW VERSION independently, so a cold run paid the round-trip twice.
# They now share one probe. Only the *fetch* is shared: each consumer keeps its
# own recorded version and its own decision about what to invalidate.
#
# Keyed on the ontology identity, not the version id: to find the entry for an
# incoming request we have only conn_params, and learning the version requires
# the probe itself. Same triple as ontology_context.shared._cache_key.
_version_probe: dict = {}                 # identity -> (version_id, fetched_at)
# When a probe last *failed*, per ontology. Without this, a server that is down
# would add a full ``http_connect_timeout`` to every question, because the
# once-per-question refresh bypasses the throttle by design. After a failure we
# back off for ``cache_timeout`` and behave exactly as the throttled path did.
_version_probe_failed: dict = {}
_version_probe_lock = threading.Lock()


def _ontology_identity(conn_params) -> tuple:
    """The (url, ontology, jwt_tenant_id) triple a version belongs to.

    Mirrors ``ontology_context.ontology.shared._cache_key``. Datasource is out:
    it selects where data is read from, not which DDL version is deployed.
    """
    if not isinstance(conn_params, dict):
        return (None, None, None)
    return (
        conn_params.get("url"),
        conn_params.get("ontology"),
        conn_params.get("jwt_tenant_id"),
    )


def _probe_and_record(key: tuple, conn_params) -> str:
    """Fetch the version, recording success or failure. Caller holds the lock."""
    try:
        version = _get_ontology_version(conn_params)
    except Exception:
        # Remember the failure so the next question backs off instead of paying
        # the connect timeout again, then let the caller see the error — the
        # throttled path has always propagated it.
        _version_probe_failed[key] = time.time()
        raise
    _version_probe_failed.pop(key, None)
    _version_probe[key] = (version, time.time())
    return version


def get_ontology_version(conn_params, force: bool = False) -> Optional[str]:
    """Return the ontology's current version id, or ``None`` for "no opinion".

    ``None`` means the probe holds no value and could not obtain one right now —
    another thread is mid-fetch. Callers must treat it as "no information" and
    leave their cache alone, which is exactly what the old non-blocking version
    check did when it lost the race.

    ``force=True`` bypasses the throttle for one call, for the once-per-question
    refresh: concurrent forcing threads collapse onto a single fetch, since any
    value produced after this call started is accepted.

    The read path takes no lock — the gate fires on every decorated call and
    from five sites in the Ontology graph, so making it contend would cost more
    than the round-trip it saves. A plain dict get is atomic under CPython.
    """
    key = _ontology_identity(conn_params)
    entry = _version_probe.get(key)

    if not force and entry is not None and (time.time() - entry[1]) <= cache_timeout:
        return entry[0]

    # A recent failure means the server is not answering. Do not pay the connect
    # timeout again on this question — serve what we have, or admit to nothing.
    failed_at = _version_probe_failed.get(key)
    if failed_at is not None and (time.time() - failed_at) <= cache_timeout:
        return entry[0] if entry is not None else None

    if force:
        # Blocking: the caller explicitly asked for a fresh value. Whoever holds
        # the lock is already fetching one, so waiting for it is cheaper than a
        # second round-trip.
        #
        # "Did someone refresh while we waited?" is decided by tuple *identity*,
        # not by comparing timestamps: entries are replaced wholesale, so a
        # different object means a different fetch. Timestamps cannot answer it —
        # time.time() has ~15ms resolution on Windows, so a refresh and the call
        # that follows it routinely land on the same tick.
        with _version_probe_lock:
            current = _version_probe.get(key)
            if current is not None and current is not entry:
                return current[0]
            version = _probe_and_record(key, conn_params)
            return version

    # Throttled path: non-blocking, so the thread that gets the lock refreshes
    # for everyone and the rest carry on with what they have rather than
    # queueing behind a network round-trip.
    if _version_probe_lock.acquire(blocking=False):
        try:
            entry = _version_probe.get(key)
            if entry is None or (time.time() - entry[1]) > cache_timeout:
                return _probe_and_record(key, conn_params)
            return entry[0]
        finally:
            _version_probe_lock.release()

    entry = _version_probe.get(key)
    return entry[0] if entry is not None else None


def clear_version_probe() -> None:
    """Drop every cached version and failure record. Intended for tests."""
    with _version_probe_lock:
        _version_probe.clear()
        _version_probe_failed.clear()


def _get_snapshot(identity: tuple, store: dict) -> _Snapshot:
    """Return this ontology's snapshot from ``store``, creating it if absent."""
    snap = store.get(identity)          # lock-free fast path; dict get is atomic
    if snap is not None:
        return snap
    with _snapshots_lock:
        snap = store.get(identity)
        if snap is None:
            snap = _Snapshot()
            store[identity] = snap
        return snap


def _swap_snapshot(identity: tuple) -> None:
    """Publish a fresh, empty snapshot for one ontology.

    Nothing is copied out of the outgoing generation — not entries, not expiry
    deadlines. Exempt entries are unaffected because they live in a different
    store that is never swapped, so there is no filtering pass over a dict other
    threads are still writing to, and therefore no
    ``RuntimeError: dictionary changed size during iteration`` to avoid.
    """
    with _snapshots_lock:
        _snapshots[identity] = _Snapshot()


def _all_expiry_maps() -> list:
    """Every expiry map across both stores. Test helper."""
    return [s.expiry for s in (*_snapshots.values(), *_exempt_store.values())]


def _all_cache_maps() -> list:
    """Every entry map across both stores. Test helper."""
    return [s.cache for s in (*_snapshots.values(), *_exempt_store.values())]


# ``conn_params`` fields that never change a metadata response. ``token`` and the
# impersonation header identify the *caller*, not the data: everything except the
# mapping list and the view list is the same for every user with access to the
# ontology, so they are projected out of the shared key. The rest say how the
# caller reached the server, not what it can see. What survives is the data
# identity — url, ontology, jwt_tenant_id, datasource, and any extra a caller
# adds — so an unrecognised parameter fragments the cache rather than silently
# colliding.
_NON_IDENTITY_CONN_FIELDS = (
    "token", "additional_headers",   # the caller
    "verify_ssl", "enable_IPv6",     # transport
    "is_jwt",                        # how the caller authenticated, not what it can see
)
_IMPERSONATE_HEADER = "x-api-impersonate-user"


def _caller_identity(conn_params: dict) -> list:
    """The caller identity the per-user tier keys on.

    ``x-api-impersonate-user`` is a second token as far as visibility goes — it
    selects whose mapping and view lists the server returns — so it keys the
    per-user tier alongside the token itself. Header names are normalised the
    same way the transport normalises them before sending.
    """
    headers = conn_params.get("additional_headers") or {}
    impersonated = None
    for key, value in headers.items():
        if str(key).replace("_", "-").lower() == _IMPERSONATE_HEADER:
            impersonated = value

    return [conn_params.get("token"), impersonated]


def _project_conn_params(conn_params: dict, per_user: bool) -> dict:
    """Reduce ``conn_params`` to the fields that belong in a cache key."""
    if not isinstance(conn_params, dict):
        return conn_params

    projected = {k: v for k, v in conn_params.items() if k not in _NON_IDENTITY_CONN_FIELDS}
    if per_user:
        projected["_caller"] = _caller_identity(conn_params)

    return projected


def _serialize_cache_key(*args, **kwargs):
    """Serialize arguments into a hashable cache key."""
    def serialize(obj):
        if isinstance(obj, dict):
            return tuple(sorted((k, serialize(v)) for k, v in obj.items()))
        elif isinstance(obj, list):
            return tuple(serialize(x) for x in obj)
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        raise TypeError(f"Unsupported type for caching: {type(obj)}")

    return (tuple(serialize(arg) for arg in args), tuple((k, serialize(v)) for k, v in kwargs.items()))


def generate_key() -> bytes:
    """Generate a new Fernet secret key."""
    passcode = b"lucylit2025"
    hlib = hashlib.md5()
    hlib.update(passcode)
    return base64.urlsafe_b64encode(hlib.hexdigest().encode('utf-8'))


ENCRYPT_KEY = generate_key()


def encrypt_prompt(prompt: Any, key: Optional[bytes] = ENCRYPT_KEY) -> bytes:
    """Serialize & encrypt the prompt; returns a URL-safe token."""
    if isinstance(prompt, str):
        text = prompt
    elif isinstance(prompt, list):
        parts = []
        for message in prompt:
            if hasattr(message, "content"):
                parts.append(f"{message.type}: {message.content}")
            else:
                parts.append(str(message))
        text = "\n".join(parts)
    else:
        text = str(prompt)

    f = Fernet(key)
    return f.encrypt(text.encode()).decode('utf-8')


def decrypt_prompt(token: bytes, key: bytes) -> str:
    """Decrypt the token and return the original prompt string."""
    f = Fernet(key)
    return f.decrypt(token).decode()


def cache_with_version_check(func=None, *, per_user: bool = False, version_gated: bool = True):
    """Decorator to cache function results and invalidate if ontology version changes.

    Two tiers, because Timbr permission-filters some system tables:

    - shared (default) — the key drops the caller identity (token,
      ``x-api-impersonate-user``) and the transport-only connection fields, so
      every user of an ontology version shares one entry.
    - ``per_user=True`` — the key keeps the caller identity and the entry also
      expires after ``per_user_cache_ttl``. Use it for anything whose rows the
      server filters by permission (the mapping list, the view list); sharing
      those across users hands one user another's rows.

    ``version_gated=False`` exempts the entry from version-change eviction, for
    data whose freshness is decided by its own source rather than by the DDL
    version. It does **not** exempt it from the per-user TTL.

    Usable bare (``@cache_with_version_check``) or called
    (``@cache_with_version_check(per_user=True)``).
    """
    if func is None:
        return lambda f: cache_with_version_check(f, per_user=per_user, version_gated=version_gated)

    # Where ``conn_params`` sits in the signature, resolved once at decoration
    # time so the wrapper can find it whether it was passed positionally or by
    # keyword.
    _params = list(inspect.signature(func).parameters)
    conn_index = _params.index("conn_params") if "conn_params" in _params else None

    def _conn_params_of(args, kwargs):
        if "conn_params" in kwargs:
            return kwargs["conn_params"]
        if conn_index is not None and conn_index < len(args):
            return args[conn_index]
        return args[-1] if args else {}

    def _key_of(args, kwargs):
        """Cache key with conn_params reduced to its tier-appropriate identity."""
        if "conn_params" in kwargs:
            kwargs = {**kwargs, "conn_params": _project_conn_params(kwargs["conn_params"], per_user)}
        elif conn_index is not None and conn_index < len(args):
            args = list(args)
            args[conn_index] = _project_conn_params(args[conn_index], per_user)

        return (func.__name__, _serialize_cache_key(*args, **kwargs))

    @wraps(func)
    def wrapper(*args, **kwargs):
        # The throttle now lives inside the shared probe, so this runs on every
        # call and is a dict read in the common case. ``None`` means the probe
        # has no value and could not get one right now — no information, so
        # leave the cache alone rather than treating it as a change.
        conn_params = _conn_params_of(args, kwargs)
        identity = _ontology_identity(conn_params)
        current_version = get_ontology_version(conn_params)
        if current_version is not None and _ontology_version.get(identity) != current_version:
            # Only this ontology's gated snapshot is replaced. Every other
            # ontology, and the exempt store, are untouched.
            _swap_snapshot(identity)
            _ontology_version[identity] = current_version

        # Capture the snapshot ONCE, into a local, and read *and write* through
        # this reference for the rest of the call. This single line is what makes
        # the stale-write race structurally impossible: if the version moves
        # while ``func`` is out, the result lands in the generation it was
        # computed against — one nobody reads any more — instead of the fresh
        # one. Reaching for the module global again anywhere below would silently
        # reinstate the bug.
        snap = _get_snapshot(identity, _snapshots if version_gated else _exempt_store)

        # Generate a cache key based on function name and arguments
        cache_key = _key_of(args, kwargs)

        while True:
            with _cache_lock:
                if cache_key in snap.cache:
                    expires_at = snap.expiry.get(cache_key)
                    if expires_at is None or expires_at > time.time():
                        return snap.cache[cache_key]
                    # Per-user entry past its TTL — drop it and re-fetch, so a
                    # permission change that did not move the ontology version
                    # is still picked up.
                    del snap.cache[cache_key]
                    del snap.expiry[cache_key]
                waiter = snap.inflight.get(cache_key)
                if waiter is None:
                    waiter = threading.Event()
                    snap.inflight[cache_key] = waiter
                    break                     # this thread computes it
            # Someone else is already computing this key — wait, then re-check.
            waiter.wait(timeout=cache_timeout)

        try:
            # Call the function and store the result in the cache. An empty
            # result is cached like any other: "this ontology has no property
            # descriptions" is an answer, and re-asking on every call cost a
            # round-trip per invoke for each such query.
            result = func(*args, **kwargs)
            with _cache_lock:
                snap.cache[cache_key] = result
                if per_user:
                    snap.expiry[cache_key] = time.time() + per_user_cache_ttl
            return result
        finally:
            # Always release the waiters. On failure they find no cache entry
            # and no in-flight marker, so one of them retries the call.
            with _cache_lock:
                snap.inflight.pop(cache_key, None)
            waiter.set()

    def invalidate(*args, **kwargs) -> bool:
        """Drop this caller's entry for these arguments. Returns True if one went.

        Scoped exactly like a read: same ontology identity, same tier, same key.
        A per-user entry therefore only ever drops for the caller whose
        ``conn_params`` are passed, which is what lets a freshness probe act on
        one user's view without touching anybody else's.
        """
        identity = _ontology_identity(_conn_params_of(args, kwargs))
        store = _snapshots if version_gated else _exempt_store
        snap = store.get(identity)
        if snap is None:
            return False
        cache_key = _key_of(args, kwargs)
        with _cache_lock:
            snap.expiry.pop(cache_key, None)
            return snap.cache.pop(cache_key, None) is not None

    def cache_key(*args, **kwargs):
        """The key these arguments resolve to, for callers that need to hang
        their own per-entry state off the same identity."""
        return _key_of(args, kwargs)

    wrapper.invalidate = invalidate
    wrapper.cache_key = cache_key
    return wrapper


def _send_query(**kwargs):
    """Single transport seam for every Timbr query."""
    if http_keepalive:
        return _run_query_pooled(**kwargs)
    return timbr_http_connector.run_query(**kwargs)


# A read of a `timbr.sys_*` table, however the schema is quoted. The plain
# ``'.SYS' in query`` test this replaces missed the backtick-quoted form —
# ``timbr`.`sys_x`` puts a backtick between the dot and SYS — so those reads kept
# the caller's results-limit and came back truncated. A partial metadata answer
# is not a smaller answer, it is a wrong one: a missing sys_concept_relationships
# row costs a relationship its is_mtm flag and its cardinality.
#
# Only reached when use_query_limit is False, i.e. for the library's own metadata
# reads. Generated SQL runs with use_query_limit=True and is always capped,
# whatever tables it happens to name.
# The trailing underscore matters: every sys object is ``sys_<something>``, and
# without it a data query selecting a column like ``t.system_id`` matched and
# escaped its row cap.
_SYS_TABLE_RE = re.compile(r'\.\s*[`"\[]?SYS_')


def run_query(sql: str, conn_params: dict, llm_prompt: Optional[str] = None, use_query_limit = False) -> list[list]:
    if not conn_params:
        raise("Please provide connection params.")

    query = sql
    if llm_prompt:
        clean_prompt = llm_prompt.replace('\r\n', ' ').replace('\n', ' ').replace('?', '')
        query = f"-- LLM: {clean_prompt}\n{sql}"

    # Fix for multiple ontologies - load the prompt template using a specific ontology connection
    query_conn_params = conn_params.copy()
    if 'ontology' in conn_params and conn_params['ontology'] and ',' in conn_params['ontology']:
        query_conn_params['ontology'] = query_conn_params['ontology'].split(',')[0].strip()

    if not use_query_limit:
        # Remove results-limit
        if 'additional_headers' in conn_params and 'results-limit' in conn_params['additional_headers']:
            query_upper = query.strip().upper()
            if query_upper.startswith('SHOW') or query_upper.startswith('DESC') or _SYS_TABLE_RE.search(query_upper):
                query_conn_params = conn_params.copy()
                query_conn_params['additional_headers'] = conn_params['additional_headers'].copy()
                del query_conn_params['additional_headers']['results-limit']
                # If no other additional_headers remain, delete the key entirely
                if not query_conn_params['additional_headers']:
                    del query_conn_params['additional_headers']

    results = _send_query(
        query=query,
        **query_conn_params,
    )

    return results


def get_ontologies(conn_params: dict) -> list[str]:
    query = "SELECT ontology FROM timbr.sys_ontologies"
    res = run_query(query, conn_params)
    return [row.get('ontology') for row in res]


@cache_with_version_check
def get_datasources(conn_params: dict, filter_active: Optional[bool] = False) -> list[dict]:
    query = "SHOW DATASOURCES"
    res = run_query(query, conn_params)
    if filter_active:
        res = [row for row in res if to_boolean(row.get('is_active'))]
    
    return res


def get_timbr_agent_options(agent_name: str, conn_params: dict) -> dict:
    """
    Get all options for a specific agent from timbr sys_agents_options table.
    
    Args:
        agent_name: Name of the agent to get options for
        conn_params: Connection parameters for Timbr system engine
        
    Returns:
        Dictionary of option_name -> option_value pairs
    """
    options = {}
    
    # Query agent options (case-insensitive match on agent_name)
    options_query = f"SELECT option_name, option_value FROM timbr.sys_agents_options WHERE LOWER(agent_name) = LOWER('{agent_name}')"
    
    # Fix for multiple ontologies - load the prompt template using a specific ontology connection
    use_ontology = "system_db" if conn_params.get("ontology") and ',' in conn_params.get("ontology") else conn_params.get("ontology")
    results = run_query(options_query, {**conn_params, "ontology": use_ontology})
    if len(results) == 0:
        raise Exception(f'Agent "{agent_name}" not found or has no options defined.')
    for row in results:
        option_name = row.get('option_name')
        option_value = row.get('option_value', '')
        options[option_name] = option_value

    return options


def get_timbr_benchmark_info(benchmark_name: str, conn_params: dict) -> dict:
    """
    Get benchmark info from timbr SYS_AGENTS_BENCHMARKS table.

    Args:
        benchmark_name: Name of the benchmark to retrieve
        conn_params: Connection parameters for Timbr system engine

    Returns:
        Dictionary with benchmark fields: benchmark_name, agent_name, description,
        number_of_questions, benchmark (questions JSON string), updated_at, changed_by_user

    Raises:
        Exception: If the benchmark is not found in SYS_AGENTS_BENCHMARKS.
    """
    query = f"SELECT * FROM timbr.SYS_AGENTS_BENCHMARKS WHERE LOWER(benchmark_name) = LOWER('{benchmark_name}')"
    results = run_query(query, conn_params)
    if not results:
        raise Exception(f'Benchmark "{benchmark_name}" not found in SYS_AGENTS_BENCHMARKS.')
    row = results[0]
    return {k.lower(): v for k, v in row.items()}


def _validate(sql: str, conn_params: dict) -> bool:
    explain_sql = f"EXPLAIN {sql}"
    explain_res = run_query(explain_sql, conn_params)
    
    query_sql = f"SELECT * FROM ({sql.replace(';', '')}) explainable_query WHERE 1=0"
    query_res = run_query(query_sql, conn_params)

    return to_boolean(explain_res and explain_res[0].get('PLAN') and query_res is not None)


def validate_sql(sql: str, conn_params: dict) -> tuple[bool, str, str]:
    if not sql:
        raise Exception("Please provide SQL to validate.")
    
    is_valid = False
    error = None

    try:
        is_valid = _validate(sql, conn_params)
    except Exception as e:
        error = str(getattr(e, 'doc', e))
        if not sql.upper().startswith("SELECT") and not sql.upper().startswith("WITH"):
            test_sql = sql[sql.upper().index("SELECT"):]
            try:
                is_valid = _validate(test_sql, conn_params)
                if is_valid:
                    error = None
                    sql = test_sql
            except Exception:
                pass

    return is_valid, error, sql


def _should_ignore_tag(tag_name: str) -> bool:
    if not tag_name:
        return True
    
    tag_name_lower = tag_name.lower()
    if tag_name_lower in ignore_tags:
        return True
    
    for prefix in ignore_tags_prefix:
        if tag_name_lower.startswith(prefix.lower()):
            return True
            
    return False


def _prepare_tags_dict(
    type: Optional[str] = 'concept',
    tags_list: Optional[list] = [],
    include_tags: Optional[list] = [],
) -> dict:
    tags_dict = {}
    if not include_tags:
        return tags_dict # currently empty
    
    for tag in tags_list:
        # Make sure that the tag is of the correct type
        if type != tag.get('target_type'):
            continue

        tag_name = tag.get('tag_name')

        # Check if the tag is included
        if (not _should_select_all(include_tags) and tag_name not in include_tags) or _should_ignore_tag(tag_name):
            continue
        
        key = tag.get('target_name')
        tag_value = tag.get('tag_value')

        if key not in tags_dict:
            tags_dict[key] = {}
        
        tags_dict[key][tag_name] = tag_value
        
    return tags_dict


@cache_with_version_check
def get_ontology_tags(conn_params: dict) -> list[dict]:
    """Every tag row in the ontology.

    ``SHOW TAGS`` takes no filter, so the server returns the same rows whatever
    the caller means to keep — hence a key without ``include_tags``, and one
    query per ontology version rather than one per distinct tag selection.
    """
    query = "SHOW TAGS"

    return run_query(query, conn_params)


@cache_with_version_check
def get_tags(conn_params: dict, include_tags: Optional[Any] = None) -> dict:
    """Tag rows pivoted per target type, keeping only ``include_tags``.

    Still cached per selection so the pivot is not redone on every call; the
    rows underneath it come from the one shared ``SHOW TAGS`` fetch.
    """
    if not to_boolean(include_tags):
        return {
            "concept_tags": {},
            "view_tags": {},
            "property_tags": {},
            # "relationship_tags": {},
        }

    ontology_tags = get_ontology_tags(conn_params=conn_params)

    return {
        "concept_tags": _prepare_tags_dict('concept', ontology_tags, include_tags),
        "view_tags": _prepare_tags_dict('ontology view', ontology_tags, include_tags),
        "property_tags": _prepare_tags_dict('property', ontology_tags, include_tags),
        # "relationship_tags": _prepare_tags_dict('relationship', ontology_tags, include_tags),
    }


def _should_ignore_list(list: list[Any] | None) -> bool:
    return bool(list and len(list) == 1 and (list[0].lower() in ['none', 'null']))


def _should_select_all(list: list[Any] | None) -> bool:
    return bool(list and len(list) == 1 and list[0] == '*')


@cache_with_version_check(per_user=True)
def _has_dtimbr_permissions(conn_params: dict) -> bool:
    has_perms = True
    dtimbr_query = "SHOW TABLES IN dtimbr"
    try:
        dtimbr_tables = run_query(dtimbr_query, conn_params)
        has_perms = len(dtimbr_tables) > 0
    except Exception:
        has_perms = False

    return has_perms


@cache_with_version_check
def get_concepts_only(conn_params: dict, filter_concepts: str = "") -> list[dict]:
    """Concept rows from timbr.sys_concepts.

    Shared tier: access to an ontology grants access to every concept in it, so
    these rows are identical for every caller on the same ontology version.
    """
    query = f"""
        SELECT concept, description, 'false' AS is_view 
        FROM timbr.sys_concepts{filter_concepts}
        ORDER BY is_view ASC
    """.strip()

    return run_query(query, conn_params)


@cache_with_version_check(per_user=True)
def get_views_only(conn_params: dict) -> list[dict]:
    """Every row of timbr.sys_views, unfiltered.

    Per-user tier: the server filters this table by permission, so one caller's
    view list is not another's. Fetched whole and once per (caller, ontology,
    version) — ``get_concepts``, ``load_view_row_counts`` and the
    identify-concept catalog all read it and each projects the columns it needs
    in Python, so ``sys_views`` costs one query instead of three.
    """
    query = """
        SELECT view_name, description, is_cube, tables, number_of_rows
        FROM timbr.sys_views
    """.strip()

    return run_query(query, conn_params)


def get_concepts(
    conn_params,
    concepts_list: Optional[list[Any]] = None,
    views_list: Optional[list[Any]] = None,
    include_logic_concepts: Optional[bool] = False,
) -> dict:
    """Fetch concepts (or views) from timbr.sys_concepts and/or timbr.sys_views.

    The two halves are fetched separately — concepts are shared across callers,
    views are permission-filtered — and combined here, so the return value is
    the same dict of ``{concept: {concept, description, is_view}}`` the single
    UNION query produced: concepts first, views second, first name wins.
    """
    should_ignore_concepts = _should_ignore_list(concepts_list) or not _has_dtimbr_permissions(conn_params)
    should_ignore_views = _should_ignore_list(views_list)

    # Which halves to read — mirrors the three branches the UNION query chose
    # between. Anything the SQL would have filtered out with `WHERE 1 = 0` is
    # simply not asked for.
    only_concepts = bool(concepts_list) and not should_ignore_concepts and not views_list
    only_views = bool(views_list) and not should_ignore_views and not concepts_list

    rows = []

    if not only_views and not (concepts_list and should_ignore_concepts):
        filter_concepts = " WHERE concept IN (SELECT DISTINCT concept FROM timbr.sys_concept_properties)" if not include_logic_concepts else ""
        if concepts_list:
            if _should_select_all(concepts_list):
                filter_concepts = ""
            else:
                joined_concepts = ','.join(f"'{c}'" for c in concepts_list)
                filter_concepts = f" WHERE concept IN ({joined_concepts})"

        rows.extend(get_concepts_only(conn_params=conn_params, filter_concepts=filter_concepts))

    if not only_concepts and not should_ignore_views:
        view_rows = get_views_only(conn_params=conn_params)
        if views_list and not _should_select_all(views_list):
            wanted = set(views_list)
            view_rows = [row for row in view_rows if row.get('view_name') in wanted]

        rows.extend(
            {
                'concept': row.get('view_name'),
                'description': row.get('description'),
                'is_view': 'true',
            }
            for row in view_rows
        )

    uniq_concepts = {}
    for row in rows:
        concept = row.get('concept')
        if concept not in uniq_concepts and concept != 'thing':
            # Copied: callers annotate these rows in place (see the `tags` key in
            # timbr_llm_utils), and the concept half now lives in a cache shared
            # by every user of the ontology.
            uniq_concepts[concept] = dict(row)

    return uniq_concepts


@cache_with_version_check
def get_ontology_description(conn_params):
    
    ontology = conn_params.get("ontology")
    query = f"SELECT description FROM timbr.sys_ontologies WHERE ontology = '{ontology}'"
    
    res = run_query(query, conn_params)
    ontology_description = ""

    for row in res:
        ontology_description = row.get('description') or ""
    
    query = f"SELECT name, description, ontologies FROM timbr.sys_domains WHERE ontologies like '%{ontology}%'"
    
    res = run_query(query, conn_params)
    domain_description = ""

    for row in res:
        ontologies_list = row.get("ontologies").split(",")
        if ontology in ontologies_list:
            domain_name = row.get("name")
            domain_additional_description = row.get("description") or ""
            if domain_additional_description != "":
                if domain_description != "":
                    domain_description += " "
                domain_description += f"{domain_name} - {domain_additional_description}."

    return ontology_description, domain_description

def _generate_column_relationship_description(column_name):
    """
    Generates a concise description for a column used in text-to-SQL generation.

    Expected column name formats:
      relationship_name[target_concept].property_name
      or
      relationship_name[target_concept].relationship_name[target_concept].property_name
      (and potentially more nested relationships)

    For example:
      "includes_product[product].contains[material].material_name"

    Output example:
      "This column represents the material name from table material using the relationship contains from table product from relationship includes product."
    """

    try:
        # Split the column name into parts using the period as a delimiter.
        parts = column_name.split('.')
        # The final part is the property name; replace underscores with spaces.
        property_name = parts[-1].replace('_', ' ')

        # Extract relationships (each part before the final property)
        relationships = []
        for part in parts[:-1]:
            if '[' in part and ']' in part:
                relationship, target_concept = part.split('[')
                target_concept = target_concept.rstrip(']')
                # Replace underscores with spaces.
                relationship = relationship.replace('_', ' ')
                target_concept = target_concept.replace('_', ' ')
                relationships.append((relationship, target_concept))

        col_type = "column"
        if column_name.startswith("measure."):
            col_type = "measure"

        # Build the description.
        if relationships:
            # The final table is taken from the target of the last relationship.
            final_table = relationships[-1][1]
            description = f"This {col_type} represents the {property_name} from table {final_table}"
            if len(relationships) == 1:
                # Only one relationship in the chain.
                description += f" using the relationship {relationships[0][0]}."
            else:
                # For two or more relationships:
                # The last relationship is applied on the table from the previous relationship.
                # For example, for two relationships:
                #   relationships[0] = ("includes product", "product")
                #   relationships[1] = ("contains", "material")
                # We want: "using the relationship contains from table product from relationship includes product."
                last_rel, _ = relationships[-1]
                base_table = relationships[-2][1]
                derivation = f" using the relationship {last_rel} from table {base_table}"
                # For any additional relationships (if more than two), append them in order.
                for i in range(len(relationships) - 2, -1, -1):
                    derivation += f" from relationship {relationships[i][0]}"
                description += derivation + "."
        else:
            description = f"This {col_type} represents the {property_name}."

        return description
    except Exception as exp:
      return ""

@cache_with_version_check
def get_relationships_description(conn_params: dict) -> dict:
    """Fetch relationships data."""
    query = f"""
        SELECT 
            relationship_name,
            description
        FROM `timbr`.`SYS_CONCEPT_RELATIONSHIPS`
        WHERE description is not null
    """.strip()

    res = run_query(query, conn_params)
    relationships_desc = {}
    for row in res:
        relationships_desc[row['relationship_name']] = row['description']
    
    return relationships_desc

@cache_with_version_check
def get_properties_description(conn_params: dict) -> dict:
    query = f"""
        SELECT property_name, description
        FROM `timbr`.`SYS_PROPERTIES`
        WHERE description is not null
    """.strip()

    res = run_query(query, conn_params)
    properties_desc = {}
    for row in res:
        properties_desc[row['property_name']] = row['description']
    
    return properties_desc


def _add_relationship_column(
    relationship_name: str,
    relationship_desc: str,
    col_dict: dict,
    relationships: dict,
) -> None:
    """Add a column to the specified relationship."""
    col_name = col_dict.get('name')
    if col_name:
        if relationship_name not in relationships:
            is_transitive = '*' in col_name
            relationships[relationship_name] = {
                "relationship_name": relationship_name,
                "description": relationship_desc,
                "columns": [],
                "measures": [],
                "is_transitive": is_transitive,
            }

        if col_name.startswith('measure.'):
            relationships[relationship_name]['measures'].append(col_dict)
        else:
            relationships[relationship_name]['columns'].append(col_dict)

@cache_with_version_check
def get_concept_properties(
    concept_name: str,
    conn_params: dict,
    properties_desc: dict,
    relationships_desc: dict,
    schema: Optional[str] = 'dtimbr',
    graph_depth: Optional[int] = 1,
) -> dict:
    rows = []
    desc_query = f"describe concept `{schema}`.`{concept_name}`"
    if schema == 'dtimbr':
        desc_query += f" options (graph_depth='{graph_depth}')"

    try:
        rows = run_query(desc_query, conn_params)
    except Exception as e:
        # skipping new describe concept syntax
        pass

    if not rows:
        legacy_desc_query = f"desc `{schema}`.`{concept_name}`"
        try:
            rows = run_query(legacy_desc_query, conn_params)
        except Exception as e:
            print(f"Error describing concept using legacy desc stmt: {e}")

    relationships = {}
    columns = []
    measures = []
    
    for column in rows:
        col_name = column.get('col_name')
        comment = properties_desc.get(col_name)

        if col_name:
            if "_type_of_" in col_name:
                comment = f"if 1, the row is of type {col_name.split('_type_of_')[1]}"
            # elif (comment is None or comment == "") and "[" in col_name and "]" in col_name:
            #     comment = _generate_column_relationship_description(col_name)
            elif col_name.startswith("~"):
                rel_name = col_name[1:].split('[')[0]
                comment = comment + "; " if comment else ''
                comment = comment + f"This columns means the inverse of `{rel_name}`"
            
            if "." in col_name and (comment is None or comment == ""): 
                    comment = properties_desc.get(col_name.split(".")[-1])

            if '[' in col_name:
                # This is a relationship column
                rel_path, rel_col_name = col_name.rsplit('.', 1) if '.' in col_name else col_name.rsplit('_', 1) if '_' in col_name else col_name
                rel_name = rel_path.split('[', 1)[0]

                if rel_name:
                    if rel_name.startswith('measure.'):
                        rel_name = rel_name.replace('measure.', '')
            
                    comment = properties_desc.get(rel_col_name, '')

                    rel_col_dict = {
                        'name': col_name,
                        'col_name': rel_col_name,
                        'type': column.get('data_type', 'string').lower(),
                        'data_type': column.get('data_type', 'string').lower(),
                        'comment': comment,
                    }
                    _add_relationship_column(
                        relationship_name=rel_name,
                        relationship_desc=relationships_desc.get(rel_name, ''),
                        col_dict=rel_col_dict,
                        relationships=relationships
                    )

            elif col_name.startswith("measure."):
                measures.append({ **column, 'comment': comment })
            else:
                columns.append({ **column, 'comment': comment })

    return {
        "columns": columns,
        "measures": measures,
        "relationships": relationships,
    }
