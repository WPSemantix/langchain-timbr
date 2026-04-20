"""
Conversation memory subsystem for langchain-timbr.

Provides opt-in follow-up detection and context propagation so that
downstream chains (identify-concept, generate-sql, answer) can produce
coherent results across a multi-turn conversation.

All public entry points are accessed through ``resolve_memory()``.
Everything else in this module is an implementation detail.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import requests
from langchain_core.language_models.llms import LLM

from .. import config
from .prompt_service import get_memory_classifier_prompt_template

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal tuning constants (not user-facing)
# ---------------------------------------------------------------------------
_SOFT_SQL_CONTEXT_LIMIT = 5
"""Soft cap applied after classification on SQL queries included in
generate-sql / identify-concept context.  The classifier may signal
``requires_extended_context=true`` to override.  These are not transport
limits — the API has its own hard caps."""

_SOFT_QA_CONTEXT_LIMIT = 15
"""Soft cap applied after classification on Q&A pairs included in the
answer chain context.  Same override semantics as *_SOFT_SQL_CONTEXT_LIMIT*."""

_HISTORY_FETCH_TIMEOUT = 5
"""Timeout in seconds for the conversation-history HTTP call."""


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------
class MemoryDisabledSentinel:
    """Singleton-like marker indicating memory is inactive for an invocation."""
    __slots__ = ()
    _instance: Optional["MemoryDisabledSentinel"] = None

    def __new__(cls) -> "MemoryDisabledSentinel":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "MemoryDisabledSentinel()"

    def __bool__(self) -> bool:
        return False


MEMORY_DISABLED = MemoryDisabledSentinel()


@dataclass
class MemoryContext:
    """Immutable-by-convention result of ``resolve_memory()``."""
    is_follow_up: bool
    summary: str = ""
    parent_message_id: Optional[str] = None
    relevant_message_ids: List[str] = field(default_factory=list)
    requires_extended_context: bool = False
    sql_context: List[Dict[str, Any]] = field(default_factory=list)
    qa_context: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def resolve_memory(
    llm: LLM,
    conn_params: dict,
    conversation_id: Optional[str],
    prompt: str,
    enable_memory: bool,
    memory_window_size: int,
    concept_names: Optional[List[str]] = None,
    timeout: Optional[int] = None,
) -> Union[MemoryContext, MemoryDisabledSentinel]:
    """Resolve conversation memory for the current invocation.

    Returns a ``MemoryContext`` when memory is active and classification
    succeeds, or ``MEMORY_DISABLED`` when any precondition is unmet or an
    error occurs.  Failures are always silent (DEBUG-logged).
    """
    # ---- activation gate ------------------------------------------------
    if not enable_memory:
        return MEMORY_DISABLED
    if not conversation_id or not conversation_id.strip():
        return MEMORY_DISABLED
    if not prompt or not prompt.strip():
        return MEMORY_DISABLED

    try:
        return _resolve_memory_impl(
            llm=llm,
            conn_params=conn_params,
            conversation_id=conversation_id,
            prompt=prompt,
            memory_window_size=memory_window_size,
            concept_names=concept_names,
            timeout=timeout,
        )
    except Exception as exc:
        logger.debug("Memory disabled for this invocation due to error: %s", exc)
        return MEMORY_DISABLED


def _resolve_memory_impl(
    llm: LLM,
    conn_params: dict,
    conversation_id: str,
    prompt: str,
    memory_window_size: int,
    concept_names: Optional[List[str]],
    timeout: Optional[int],
) -> Union[MemoryContext, MemoryDisabledSentinel]:
    # Step 1 — fetch history
    messages = fetch_conversation_history(conn_params, conversation_id, memory_window_size)
    if not messages:
        return MEMORY_DISABLED

    # Step 2 — build id_map once (API returns complete chains)
    id_map: Dict[str, Dict[str, Any]] = {
        m["message_id"]: m for m in messages if "message_id" in m
    }

    # Step 3 — classify
    classifier_output = classify_follow_up(
        llm=llm,
        conn_params=conn_params,
        prompt=prompt,
        messages=messages,
        id_map=id_map,
        concept_names=concept_names,
        timeout=timeout,
    )
    if classifier_output is None:
        return MEMORY_DISABLED

    is_follow_up = classifier_output.get("is_follow_up", False)
    relevant_ids = classifier_output.get("relevant_message_ids", [])

    if not is_follow_up or not relevant_ids:
        return MemoryContext(is_follow_up=False)

    # Step 4 — build contexts
    sql_ctx = build_sql_context(id_map, classifier_output)
    qa_ctx = build_qa_context(id_map, classifier_output)

    return MemoryContext(
        is_follow_up=True,
        summary=classifier_output.get("summary", ""),
        parent_message_id=classifier_output.get("parent_message_id"),
        relevant_message_ids=relevant_ids,
        requires_extended_context=classifier_output.get("requires_extended_context", False),
        sql_context=sql_ctx,
        qa_context=qa_ctx,
    )


# ---------------------------------------------------------------------------
# Step 1 — Fetch history
# ---------------------------------------------------------------------------
def fetch_conversation_history(
    conn_params: dict,
    conversation_id: str,
    top: int,
) -> Optional[List[Dict[str, Any]]]:
    """GET conversation history from the Timbr server.

    Returns ``None`` on any failure (logged at DEBUG).
    """
    base_url = (conn_params.get("url") or config.url or "").rstrip("/")
    if not base_url:
        logger.debug("Memory: no base URL configured, skipping history fetch")
        return None

    url = f"{base_url}/timbr/api/fetch_conversation_history"
    headers = _build_auth_headers(conn_params)
    params = {"conversation_id": conversation_id, "top": top}

    try:
        response = requests.get(
            url, headers=headers, params=params, timeout=_HISTORY_FETCH_TIMEOUT,
        )
        if not response.ok:
            logger.debug(
                "Memory: history endpoint returned %s for conversation %s",
                response.status_code, conversation_id,
            )
            return None
        data = response.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            if data.get("truncated", False):
                logger.debug(
                    "Memory: API truncated history response, context may be incomplete"
                )
            return data["data"]
        logger.debug("Memory: unexpected history response shape: %s", type(data))
        return None
    except Exception as exc:
        logger.debug("Memory: history fetch failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Parent-chain walking helper
# ---------------------------------------------------------------------------
def _walk_parent_chain(
    msg_id: str,
    id_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Walk ``parent_query_id`` links from *msg_id* back to root.

    Returns a chronological list (root first) of ancestor messages,
    **excluding** *msg_id* itself.  The API guarantees complete chains, so
    missing parents should not occur; a ``seen`` set is kept defensively to
    prevent infinite loops if data is malformed.
    """
    ancestors: list[Dict[str, Any]] = []
    seen: set[str] = {msg_id}
    current_id = msg_id

    while True:
        current = id_map.get(current_id)
        if current is None:
            break
        parent_id = current.get("parent_query_id")
        if not parent_id or parent_id in seen:
            break
        parent = id_map.get(parent_id)
        if parent is None:
            break
        seen.add(parent_id)
        ancestors.append(parent)
        current_id = parent_id

    ancestors.reverse()  # chronological: root first
    return ancestors


# ---------------------------------------------------------------------------
# Step 3 — Classify the current question
# ---------------------------------------------------------------------------
def classify_follow_up(
    llm: LLM,
    conn_params: dict,
    prompt: str,
    messages: List[Dict[str, Any]],
    id_map: Dict[str, Dict[str, Any]],
    concept_names: Optional[List[str]] = None,
    timeout: Optional[int] = None,
) -> Optional[dict]:
    """Run the memory-classifier LLM call.

    Returns the parsed+validated classifier dict, or ``None`` on any failure.
    """
    if timeout is None:
        timeout = config.llm_timeout

    # Fetch the classifier prompt template
    try:
        classifier_prompt_wrapper = get_memory_classifier_prompt_template(conn_params)
    except Exception as exc:
        logger.debug("Memory: classifier prompt fetch failed: %s", exc)
        return None

    # Build chronological Q&A text for the classifier (uses sequential IDs)
    conversation_history, seq_to_guid = _format_history_for_classifier(messages, id_map)
    concepts_str = ", ".join(concept_names) if concept_names else ""

    try:
        formatted_prompt = classifier_prompt_wrapper.format_messages(
            question=prompt.strip(),
            conversation_history=conversation_history,
            concept_names=concepts_str,
        )
    except Exception as exc:
        logger.debug("Memory: classifier prompt formatting failed: %s", exc)
        return None

    # Call LLM
    try:
        from .timbr_llm_utils import _call_llm_with_timeout
        response = _call_llm_with_timeout(llm, formatted_prompt, timeout=timeout)
    except Exception as exc:
        logger.debug("Memory: classifier LLM call failed: %s", exc)
        return None

    # Extract response text
    if hasattr(response, "content"):
        response_text = response.content
    elif isinstance(response, str):
        response_text = response
    else:
        logger.debug("Memory: unexpected classifier response type: %s", type(response))
        return None

    # Parse + validate (translate sequential IDs back to real GUIDs)
    history_ids = {m["message_id"] for m in messages if "message_id" in m}
    return _validate_classifier_output(response_text, history_ids, seq_to_guid)


def _format_history_for_classifier(
    messages: List[Dict[str, Any]],
    id_map: Dict[str, Dict[str, Any]],
) -> tuple:
    """Build a chronological Q&A block for the classifier prompt.

    Walks each message's parent chain via *id_map* so the classifier sees
    full follow-up threads, then deduplicates messages that appear in
    multiple chains.

    Returns a tuple of (formatted_text, seq_to_guid) where seq_to_guid maps
    sequential number strings ("1", "2", ...) to actual message_id GUIDs.
    """
    seen: set[str] = set()
    ordered_ids: list[str] = []
    ordered_entries: list[Dict[str, Any]] = []

    for msg in messages:
        mid = msg.get("message_id", "")
        # Expand ancestor chain for this message
        for ancestor in _walk_parent_chain(mid, id_map):
            aid = ancestor.get("message_id", "")
            if aid and aid not in seen:
                seen.add(aid)
                ordered_ids.append(aid)
                ordered_entries.append(ancestor)
        if mid and mid not in seen:
            seen.add(mid)
            ordered_ids.append(mid)
            ordered_entries.append(msg)

    # Assign sequential numbers (1-based) in chronological order
    seq_to_guid: Dict[str, str] = {}
    lines: list[str] = []
    for idx, (guid, entry) in enumerate(zip(ordered_ids, ordered_entries), start=1):
        seq_id = str(idx)
        seq_to_guid[seq_id] = guid
        lines.append(
            f"[{seq_id}] Q: {entry.get('question', '')}\n"
            f"A: {entry.get('answer', '')}"
        )

    return "\n---\n".join(lines), seq_to_guid


def _validate_classifier_output(
    raw_text: str,
    history_ids: set,
    seq_to_guid: Optional[Dict[str, str]] = None,
) -> Optional[dict]:
    """Parse and validate classifier JSON.  Returns ``None`` on any problem.

    When *seq_to_guid* is provided, the classifier's sequential IDs ("1", "2", ...)
    are translated back to real message GUIDs before validation.
    """
    # Strip markdown code fences if present
    text = raw_text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
    if text.endswith("```"):
        text = "\n".join(text.split("\n")[:-1])
    text = text.strip()

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        logger.debug("Memory: classifier returned invalid JSON: %s", raw_text[:500])
        return None

    if not isinstance(parsed, dict):
        logger.debug("Memory: classifier returned non-dict JSON: %s", raw_text[:500])
        return None

    is_follow_up = parsed.get("is_follow_up", False)
    relevant_ids = parsed.get("relevant_message_ids", [])
    parent_id = parsed.get("parent_message_id")

    # Normalize types
    if not isinstance(relevant_ids, list):
        relevant_ids = []
    relevant_ids = [str(rid) for rid in relevant_ids]

    if not isinstance(is_follow_up, bool):
        is_follow_up = bool(is_follow_up)

    # Translate sequential IDs back to real GUIDs
    if seq_to_guid:
        relevant_ids = [seq_to_guid.get(rid, rid) for rid in relevant_ids]
        if parent_id is not None:
            parent_id = seq_to_guid.get(str(parent_id), str(parent_id))

    # is_follow_up=True but no relevant IDs → force to False
    if is_follow_up and not relevant_ids:
        logger.debug("Memory: classifier said follow-up but gave no relevant IDs")
        is_follow_up = False

    if not is_follow_up:
        return {
            "is_follow_up": False,
            "summary": "",
            "parent_message_id": None,
            "relevant_message_ids": [],
            "requires_extended_context": False,
        }

    # Validate parent_message_id
    if parent_id is not None:
        parent_id = str(parent_id)
    if parent_id and parent_id not in relevant_ids:
        logger.debug(
            "Memory: classifier parent_message_id %s not in relevant_message_ids %s",
            parent_id, relevant_ids,
        )
        return None
    if parent_id and parent_id not in history_ids:
        logger.debug(
            "Memory: classifier parent_message_id %s not found in fetched history",
            parent_id,
        )
        return None

    # Validate all relevant IDs exist in history
    for rid in relevant_ids:
        if rid not in history_ids:
            logger.debug(
                "Memory: classifier relevant_message_id %s not found in fetched history",
                rid,
            )
            return None

    requires_extended = parsed.get("requires_extended_context", False)
    if not isinstance(requires_extended, bool):
        requires_extended = bool(requires_extended)

    if requires_extended:
        logger.debug(
            "Memory: extended context override requested by classifier"
        )

    return {
        "is_follow_up": True,
        "summary": str(parsed.get("summary", "")),
        "parent_message_id": parent_id,
        "relevant_message_ids": relevant_ids,
        "requires_extended_context": requires_extended,
    }


# ---------------------------------------------------------------------------
# Step 4a — Build SQL context
# ---------------------------------------------------------------------------
def build_sql_context(
    id_map: Dict[str, Dict[str, Any]],
    classifier_output: dict,
) -> List[Dict[str, Any]]:
    """Construct the SQL context for identify-concept and generate-sql.

    Walks the primary ancestor chain from ``parent_message_id`` back to root
    in chronological order, then appends sibling chains.
    """
    parent_id = classifier_output.get("parent_message_id")
    relevant_ids = classifier_output.get("relevant_message_ids", [])
    extended = classifier_output.get("requires_extended_context", False)

    # Primary chain: parent's ancestors + parent itself
    primary_chain: list[Dict[str, Any]] = []
    primary_ids: set[str] = set()

    if parent_id and parent_id in id_map:
        for anc in _walk_parent_chain(parent_id, id_map):
            aid = anc.get("message_id", "")
            if aid and aid not in primary_ids:
                primary_ids.add(aid)
                primary_chain.append(_sql_entry(anc))
        if parent_id not in primary_ids:
            primary_ids.add(parent_id)
            primary_chain.append(_sql_entry(id_map[parent_id]))

    # Sibling chains: relevant IDs not in primary ancestry
    siblings: list[Dict[str, Any]] = []
    for rid in relevant_ids:
        if rid in primary_ids:
            continue
        msg = id_map.get(rid)
        if msg:
            for anc in _walk_parent_chain(rid, id_map):
                aid = anc.get("message_id", "")
                if aid and aid not in primary_ids and not any(s.get("message_id") == aid for s in siblings):
                    siblings.append(_sql_entry(anc))
            if not any(s.get("message_id") == rid for s in siblings):
                siblings.append(_sql_entry(msg))

    combined = primary_chain + siblings

    # Apply soft limit (drop oldest) unless extended
    if not extended and len(combined) > _SOFT_SQL_CONTEXT_LIMIT:
        combined = combined[-_SOFT_SQL_CONTEXT_LIMIT:]
    elif extended:
        logger.debug(
            "Memory: extended SQL context override activated, including %d entries",
            len(combined),
        )

    return combined


def _sql_entry(msg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "message_id": msg.get("message_id", ""),
        "question": msg.get("question", ""),
        "sql": msg.get("sql", ""),
    }


# ---------------------------------------------------------------------------
# Step 4b — Build Q&A context
# ---------------------------------------------------------------------------
def build_qa_context(
    id_map: Dict[str, Dict[str, Any]],
    classifier_output: dict,
) -> List[Dict[str, Any]]:
    """Construct the Q&A context for the answer chain.

    Uses ``relevant_message_ids`` in the order the classifier ranked them.
    """
    relevant_ids = classifier_output.get("relevant_message_ids", [])
    extended = classifier_output.get("requires_extended_context", False)

    entries: list[Dict[str, Any]] = []
    for rid in relevant_ids:
        msg = id_map.get(rid)
        if msg:
            entries.append({
                "message_id": msg.get("message_id", ""),
                "question": msg.get("question", ""),
                "answer": msg.get("answer", ""),
            })

    # Apply soft limit (truncate tail) unless extended
    if not extended and len(entries) > _SOFT_QA_CONTEXT_LIMIT:
        entries = entries[:_SOFT_QA_CONTEXT_LIMIT]
    elif extended:
        logger.debug(
            "Memory: extended Q&A context override activated, including %d entries",
            len(entries),
        )

    return entries


# ---------------------------------------------------------------------------
# Formatters — produce text injected into existing prompt template variables
# ---------------------------------------------------------------------------
def format_memory_note_for_sql(memory_context: MemoryContext) -> str:
    """Format memory context for injection into the ``{note}`` template var
    of identify-concept and generate-sql prompts.

    Returns an empty string when memory is inactive or not a follow-up.
    """
    if not memory_context or not memory_context.is_follow_up:
        return ""

    parts: list[str] = [
        "[CONVERSATION MEMORY]",
        "This is a follow-up question.",
    ]
    if memory_context.summary:
        parts.append(f"Context summary: {memory_context.summary}")

    if memory_context.sql_context:
        parts.append("\nPrior SQL queries (chronological):")
        for idx, entry in enumerate(memory_context.sql_context, start=1):
            question = entry.get("question", "")
            sql = entry.get("sql", "")
            parts.append(f'--- [{idx}] Q: "{question}" ---')
            parts.append(sql)
            parts.append("--- End ---")

    return "\n".join(parts)


def format_memory_note_for_answer(memory_context: MemoryContext) -> str:
    """Format memory context for injection into the ``{additional_context}``
    template var of the answer prompt.

    Returns an empty string when memory is inactive or not a follow-up.
    """
    if not memory_context or not memory_context.is_follow_up:
        return ""

    parts: list[str] = [
        "[CONVERSATION MEMORY]",
        "This is a follow-up question.",
    ]
    if memory_context.summary:
        parts.append(f"Context summary: {memory_context.summary}")

    if memory_context.qa_context:
        parts.append("\nPrior conversation:")
        for entry in memory_context.qa_context:
            parts.append(f"Q: {entry.get('question', '')}")
            parts.append(f"A: {entry.get('answer', '')}")
            parts.append("---")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_auth_headers(conn_params: dict) -> Dict[str, str]:
    """Build auth headers consistent with PromptService pattern."""
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    is_jwt = conn_params.get("is_jwt", False)
    token = conn_params.get("token") or config.token or ""

    if is_jwt:
        headers["x-jwt-token"] = token
        jwt_tenant_id = conn_params.get("jwt_tenant_id")
        if jwt_tenant_id:
            headers["x-jwt-tenant-id"] = jwt_tenant_id
    elif token:
        headers["x-api-key"] = token

    return headers
