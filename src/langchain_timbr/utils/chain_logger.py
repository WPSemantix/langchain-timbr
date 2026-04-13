import logging
import uuid6
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

try:
    from importlib.metadata import version as _pkg_version
    _LANGCHAIN_TIMBR_VERSION = _pkg_version("langchain_timbr")
except Exception:
    _LANGCHAIN_TIMBR_VERSION = "unknown"


@dataclass
class AgentLogContext:
    """Carries all runtime state needed for logging a single agent/chain execution."""
    query_id: str
    agent_name: str
    url: str
    token: str
    chain_type: str
    start_time: datetime
    prompt: str
    enable_trace: bool
    current_step: Optional[str] = None
    retry_count: int = 0
    no_results_retry_count: int = 0
    concept: Optional[str] = None
    is_delegated: bool = False
    trace_sequence: int = 0
    conversation_id: Optional[str] = None


def new_query_id() -> str:
    return str(uuid6.uuid7())


def new_trace_id() -> str:
    return str(uuid6.uuid7())


def _now() -> datetime:
    """UTC datetime."""
    return datetime.now(timezone.utc)


def _fmt(dt: datetime) -> str:
    """Format datetime as MySQL DATETIME string."""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _clean(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Strip None values so the server uses its column defaults."""
    return {k: v for k, v in payload.items() if v is not None}


def _safe_post(url: str, token: str, endpoint_path: str, payload: Dict[str, Any]) -> None:
    """Fire-and-forget HTTP POST. Never raises; logs failures at WARNING level."""
    try:
        endpoint = f"{url.rstrip('/')}{endpoint_path}"
        response = requests.post(endpoint, json=payload, auth=("token", token), timeout=5)
        if not response.ok:
            logger.warning(
                "Chain log request to %s returned %s: %s",
                endpoint_path, response.status_code, response.text[:200],
            )
    except Exception as exc:
        logger.warning("Chain log request to %s failed: %s", endpoint_path, exc)


def log_agent_start(
    ctx: AgentLogContext,
    ontology: Optional[str] = None,
    schema: Optional[str] = None,
    additional_options: Optional[str] = "{}",
) -> None:
    """POST to sys_agents_running — called when execution begins."""
    _safe_post(ctx.url, ctx.token, "/timbr-server/log_agent/running", _clean({
        "query_id":               ctx.query_id,
        "agent_name":             ctx.agent_name,
        "chain_type":             ctx.chain_type,
        "ontology":               ontology or "",
        "schema":                 schema or "",
        "question":               ctx.prompt,
        "start_time":             _fmt(ctx.start_time),
        "current_step":           ctx.current_step or "",
        "retry_count":            ctx.retry_count,
        "no_results_retry_count": ctx.no_results_retry_count,
        "concept":                ctx.concept,
        "conversation_id":        ctx.conversation_id,
        "additional_options":     additional_options,
    }))


def log_agent_step(ctx: AgentLogContext) -> None:
    """POST step update to sys_agents_running — called at each step transition."""
    _safe_post(ctx.url, ctx.token, "/timbr-server/log_agent/running_update_step", _clean({
        "query_id":               ctx.query_id,
        "agent_name":             ctx.agent_name,
        "current_step":           ctx.current_step or "",
        "retry_count":            ctx.retry_count,
        "no_results_retry_count": ctx.no_results_retry_count,
        "concept":                ctx.concept,
    }))


def log_agent_history(
    ctx: AgentLogContext,
    ontology: Optional[str],
    schema: Optional[str],
    concept: Optional[str],
    generated_sql: Optional[str],
    rows_returned: Optional[int],
    status: str,
    failed_at_step: Optional[str],
    error: Optional[str],
    reasoning_status: Optional[str],
    usage_metadata: dict,
    answer_generated: bool,
    llm_type: Optional[str],
    llm_model: Optional[str],
    identify_concept_reason: Optional[str] = None,
    generate_sql_reason: Optional[str] = None,
    identify_concept_chain_duration: Optional[int] = None,
    generate_sql_chain_duration: Optional[int] = None,
    answer_chain_duration: Optional[int] = None,
    reasoning_duration: Optional[int] = None,
    answer: Optional[str] = None,
    has_results: Optional[bool] = None,
    results: Optional[Any] = None,
    additional_options: Optional[str] = "{}",
) -> None:
    """POST to sys_agents_history — triggers server-side deletion of the running row."""
    end_time = _now()
    duration_ms = int((end_time - ctx.start_time).total_seconds() * 1000)

    post_params = {
        "query_id":                        ctx.query_id,
        "agent_name":                      ctx.agent_name,
        "ontology":                        ontology or "",
        "schema":                          schema or "",
        "question":                        ctx.prompt,
        "start_time":                      _fmt(ctx.start_time),
        "end_time":                        _fmt(end_time),
        "duration":                        duration_ms,
        "identify_concept_chain_duration": identify_concept_chain_duration,
        "generate_sql_chain_duration":     generate_sql_chain_duration,
        "answer_chain_duration":           answer_chain_duration,
        "reasoning_duration":              reasoning_duration,
        "status":                          status,
        "failed_at_step":                  failed_at_step,
        "concept":                         concept,
        "generated_sql":                   generated_sql,
        "rows_returned":                   rows_returned,
        "error":                           error,
        "reasoning_status":                reasoning_status,
        "total_tokens":                    _sum_token_field(usage_metadata, "total_tokens", "approximate"),
        "input_tokens":                    _sum_token_field(usage_metadata, "input_tokens"),
        "output_tokens":                   _sum_token_field(usage_metadata, "output_tokens"),
        "retry_count":                     ctx.retry_count,
        "no_results_retry_count":          ctx.no_results_retry_count,
        "answer_generated":                answer_generated,
        "chain_trace_enabled":             ctx.enable_trace,
        "has_results":                     has_results,
        # "is_follow_up":                  None,   # future
        # "summarized_question":           None,   # future
        # "summarized_answer":             None,   # future
        # "parent_query_id":               None,   # future
        "langchain_timbr_version":         _LANGCHAIN_TIMBR_VERSION,
        "llm_type":                        llm_type or "",
        "llm_model":                       llm_model or "",
        "identify_concept_reason":         identify_concept_reason,
        "generate_sql_reason":             generate_sql_reason,
        "answer":                          answer,
        "conversation_id":                 ctx.conversation_id,
        "additional_options":              additional_options,
    }

    if has_results and results is not None:
        post_params["results"] = results

    _safe_post(ctx.url, ctx.token, "/timbr-server/log_agent/history", _clean(post_params))


def log_chain_trace(
    ctx: AgentLogContext,
    chain_type: str,
    start_time: datetime,
    status: str,
    ontology: Optional[str] = None,
    concept: Optional[str] = None,
    schema: Optional[str] = None,
    question: Optional[str] = None,
    chain_output: Optional[dict] = None,
    generated_sql: Optional[str] = None,
    is_sql_valid: Optional[bool] = None,
    rows_returned: Optional[int] = None,
    error: Optional[str] = None,
    reasoning_status: Optional[str] = None,
    usage_metadata: Optional[dict] = None,
    retry_attempt: int = 0,
    additional_options: Optional[str] = "{}",
) -> None:
    """POST a single chain step row to sys_agents_chain_trace_log. No-op when trace is disabled."""
    if not ctx.enable_trace:
        return

    ctx.trace_sequence += 1
    end_time = _now()
    duration_ms = int((end_time - start_time).total_seconds() * 1000)
    meta = usage_metadata or {}

    _safe_post(ctx.url, ctx.token, "/timbr-server/log_agent/trace", _clean({
        "trace_id":           new_trace_id(),
        "query_id":           ctx.query_id,
        "agent_name":         ctx.agent_name,
        "chain_type":         chain_type,
        "ontology":           ontology,
        "sequence":           ctx.trace_sequence,
        "retry_attempt":      retry_attempt,
        "start_time":         _fmt(start_time),
        "end_time":           _fmt(end_time),
        "duration":           duration_ms,
        "status":             status,
        "concept":            concept,
        "schema":             schema,
        "question":           question,
        "chain_output":       chain_output,
        "generated_sql":      generated_sql,
        "is_sql_valid":       is_sql_valid,
        "rows_returned":      rows_returned,
        "error":              error,
        "reasoning_status":   reasoning_status,
        "input_tokens":       _sum_token_field(meta, "input_tokens"),
        "output_tokens":      _sum_token_field(meta, "output_tokens"),
        "total_tokens":       _sum_token_field(meta, "total_tokens", "approximate"),
        "additional_options": additional_options,
    }))


def determine_status(rows: Optional[list], error: Optional[str]) -> str:
    """Map execution outcome to a status string for sys_agents_history."""
    if error and "timed out" in str(error).lower():
        return "timeout"
    if error:
        return "failed"
    if not rows or all(all(v is None for v in row.values()) for row in rows):
        return "completed_no_results"
    return "completed"


def get_llm_type(llm) -> Optional[str]:
    if llm is None:
        return None
    for attr in ("_llm_type", "llm_type"):
        try:
            val = getattr(llm, attr, None)
            if val:
                return str(val)
        except Exception:
            continue
    return None


def get_llm_model(llm) -> Optional[str]:
    if llm is None:
        return None
    for attr in ("model_name", "model", "deployment_name", "model_id"):
        try:
            val = getattr(llm, attr, None)
            if val:
                return str(val)
            elif hasattr(llm, "client") and llm.client and hasattr(llm.client, attr):
                val = getattr(llm.client, attr, None)
                if val:
                    return str(val)
        except Exception:
            continue
    return None


def _sum_token_field(usage_metadata: dict, field: str, fallback_field: Optional[str] = None) -> int:
    """Sum a token count field across all nested usage metadata dicts."""
    total = 0
    for value in usage_metadata.values():
        if isinstance(value, dict):
            val = value.get(field, value.get(fallback_field, 0) if fallback_field else 0)
            if isinstance(val, (int, float)):
                total += int(val)
    return total
