"""
LLM Benchmark utility for evaluating Timbr SQL agent performance.

Runs a set of questions through a Timbr agent and scores each result with an
LLM judge. When the expected rows are available, the judge compares the agent's
rows with them blind, in a fresh context, without knowing which set is which.

Usage::

    from langchain_timbr import run_benchmark

    # Queries provided inline (questions-enhanced dict format)
    results = run_benchmark(
        agent="my_agent",
        queries={
            "Q1": {"question": "How many active policies are there?"},
            "Q2": {"question": "Total premium amount?", "correct_sql": "SELECT SUM(...)"}
        }
    )

    # Queries pulled from the agent's 'questions' option in sys_agents_options
    results = run_benchmark(agent="my_agent")
"""

import base64
import copy
import json
import logging
import random
import re
import uuid
import warnings
from datetime import datetime
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

from pytimbr_api import timbr_http_connector

from .. import config
from langchain_timbr import create_timbr_sql_agent, GenerateTimbrSqlChain
from ..llm_wrapper.llm_wrapper import LlmWrapper
from .prompt_service import (
    get_benchmark_judge_prompt_template,
    get_benchmark_judge_results_prompt_template,
)
from .timbr_utils import get_timbr_agent_options, get_timbr_benchmark_info, build_server_url
from .timbr_llm_utils import (
    LLM_OUTPUT_RETRIES, LLMOutputError, _extract_json, _response_text, _with_output_feedback,
)

try:
    # from .._version import __version__ as _langchain_timbr_version
    from importlib.metadata import version
    _langchain_timbr_version = version("langchain_timbr")
    if '.dev' in _langchain_timbr_version:
        _langchain_timbr_version = _langchain_timbr_version.split('.dev')[0] + '.dev'
except ImportError:
    _langchain_timbr_version = "unknown"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Expected-value helpers (concept / ontology checks)
# ---------------------------------------------------------------------------

def _normalize_value(value: Any) -> Any:
    """Normalize a value for comparison (handle None, convert numbers, etc.)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return value.strip().lower()
    return value


def _matches_expected_value(expected_value: Any, selected_value: Any) -> Optional[bool]:
    """Compare expected and selected values using normalization.

    Returns:
        ``True`` when expected value is missing, otherwise ``True``/``False``.
    """
    if expected_value is None:
        return True
    return _normalize_value(expected_value) == _normalize_value(selected_value)


# ---------------------------------------------------------------------------
# Judge helpers
# ---------------------------------------------------------------------------

_DEPRECATED_MECHANISM_MESSAGE = (
    "Benchmark scoring always uses the LLM judge; '{name}' is deprecated and ignored."
)

# "Result A" / "result B" in the judge's reasoning, mapped back to the real origin.
_RESULT_LABEL_RE = re.compile(r"\bresult\s+([AB])\b", re.IGNORECASE)


def _warn_deprecated_mechanism(name: str, stacklevel: int = 3) -> None:
    warnings.warn(
        _DEPRECATED_MECHANISM_MESSAGE.format(name=name),
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def _rows_to_json(rows: List[Dict[str, Any]]) -> str:
    """Serialize result rows for the judge. Non-JSON values (dates, decimals) become strings."""
    return json.dumps(rows, default=str, ensure_ascii=False)


def _restore_result_labels(reasoning: str, labels: Dict[str, str]) -> str:
    """Rewrite the blind "Result A/B" labels in the judge's reasoning to the real origins."""
    def _replace(match: "re.Match[str]") -> str:
        label = labels[match.group(1).upper()]
        before = reasoning[:match.start()].rstrip()
        sentence_start = not before or before.endswith((".", "!", "?", ":"))
        return label[0].upper() + label[1:] if sentence_start else label
    return _RESULT_LABEL_RE.sub(_replace, reasoning)


# ---------------------------------------------------------------------------
# BenchmarkScorer
# ---------------------------------------------------------------------------

class BenchmarkScorer:
    """
    Score benchmark results with an LLM judge.

    Every verdict is a fresh, stateless LLM call with no chain or memory context.

    * When both the agent's rows and the expected rows (from ``correct_sql``) are
      available, the judge compares the two JSON row sets **blind**: they are
      labelled "Result A" / "Result B" in random order, with no SQL and no hint of
      which set came from the agent.
    * Otherwise (``generate_sql_only`` execution, no ``correct_sql``, or the
      expected query failed) the judge evaluates the candidate SQL against the
      question, alongside the reference SQL when one exists.
    """

    def __init__(
        self,
        conn_params: Dict[str, Any],
        llm: Optional[Any] = None,
        use_deterministic: Optional[bool] = None,
        use_llm_judge: Optional[bool] = None,
        rng: Optional[random.Random] = None,
    ):
        """
        Args:
            conn_params: Timbr connection parameters (url, token, ontology, …).
                         Used to fetch the benchmark-judge prompt templates from the API.
            llm: An LlmWrapper instance (or compatible LangChain LLM) used as the judge.
            use_deterministic: Deprecated and ignored.
            use_llm_judge: Deprecated and ignored.
            rng: Random source for the blind A/B ordering (injectable for tests).
        """
        if use_deterministic is not None:
            _warn_deprecated_mechanism("use_deterministic")
        if use_llm_judge is not None:
            _warn_deprecated_mechanism("use_llm_judge")
        self.conn_params = conn_params
        self.llm = llm
        self._rng = rng or random.Random()

        # Lazy-loaded template wrappers (fetched on first judge call)
        self._results_prompt_template: Optional[Any] = None
        self._sql_prompt_template: Optional[Any] = None
        self._results_template_warned = False

    # ------------------------------------------------------------------
    # Public scoring entry-point
    # ------------------------------------------------------------------

    def score_result(
        self,
        question: str,
        generated_sql: str,
        answer: str = "",
        generated_rows: Optional[List[Dict[str, Any]]] = None,
        expected_sql: Optional[str] = None,
        expected_answer: Optional[str] = None,
        expected_rows: Optional[List[Dict[str, Any]]] = None,
        execution_error: Optional[str] = None,
        execution_mode: str = "full",
        expected_rows_error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Score a single benchmark result.

        ``answer`` and ``expected_answer`` are accepted for compatibility; the judge
        evaluates result rows or SQL, never answer text.

        Args:
            expected_rows: Rows returned by ``correct_sql``; ``None`` when unavailable.
            expected_rows_error: Why ``correct_sql`` could not be executed, if it failed.

        Returns a dict with keys:
            ``assessment``    – ``"correct"``, ``"partial"``, or ``"incorrect"``
            ``scoring_method``– ``"llm_judge"``, or ``"error"`` when no verdict was produced
            ``reasoning``     – human-readable explanation string
        """
        if not generated_sql or not generated_sql.strip():
            return _error_result("No SQL query was generated")

        if execution_mode == "full" and execution_error:
            return _error_result(f"Query failed to execute: {execution_error}"[:5000])

        if self.llm is None:
            return _error_result("No LLM is configured for the benchmark judge")

        try:
            if execution_mode == "full" and expected_rows is not None:
                verdict = self._judge_results_blind(question, generated_rows or [], expected_rows)
                if verdict is not None:
                    return verdict
                note = "The results judge prompt is unavailable, so the SQL was judged against the question."
            elif execution_mode == "full" and expected_rows_error:
                note = (
                    f"The reference SQL failed to execute ({expected_rows_error}), "
                    "so the SQL was judged against the question."
                )
            elif execution_mode == "full":
                note = "No reference SQL was provided, so the SQL was judged against the question."
            else:
                note = ""

            verdict = self._judge_sql(question, generated_sql, expected_sql)
            if note:
                verdict["reasoning"] = f"{verdict['reasoning']} ({note})".strip()
            return verdict

        except Exception as exc:
            logger.warning(f"LLM judge scoring failed: {exc}")
            return _error_result(f"LLM judge scoring failed: {str(exc)}")

    # ------------------------------------------------------------------
    # Judge implementations
    # ------------------------------------------------------------------

    def _judge_results_blind(
        self,
        question: str,
        generated_rows: List[Dict[str, Any]],
        expected_rows: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Compare the two row sets blind. Returns ``None`` when the template is unavailable."""
        generated_first = self._rng.random() < 0.5
        if generated_first:
            rows_a, rows_b = generated_rows, expected_rows
            labels = {"A": "agent result", "B": "expected result"}
        else:
            rows_a, rows_b = expected_rows, generated_rows
            labels = {"A": "expected result", "B": "agent result"}

        try:
            if self._results_prompt_template is None:
                self._results_prompt_template = get_benchmark_judge_results_prompt_template(
                    conn_params=self.conn_params
                )
            messages = self._results_prompt_template.format_messages(
                question=question,
                result_a=_rows_to_json(rows_a),
                result_a_row_count=len(rows_a),
                result_b=_rows_to_json(rows_b),
                result_b_row_count=len(rows_b),
            )
        except Exception as exc:
            # An older platform does not serve this template; judge the SQL instead.
            if not self._results_template_warned:
                logger.warning(f"Benchmark results judge template unavailable, judging SQL instead: {exc}")
                self._results_template_warned = True
            return None

        assessment, reasoning = self._ask_judge(messages)
        return {
            "assessment": assessment,
            "reasoning": _restore_result_labels(reasoning, labels),
            "scoring_method": "llm_judge",
        }

    def _judge_sql(
        self,
        question: str,
        generated_sql: str,
        expected_sql: Optional[str],
    ) -> Dict[str, Any]:
        """Judge the candidate SQL against the question (and the reference SQL, when given)."""
        if self._sql_prompt_template is None:
            self._sql_prompt_template = get_benchmark_judge_prompt_template(
                conn_params=self.conn_params
            )

        expected_sql_context = (
            f"**Reference SQL Query:**\n```sql\n{expected_sql}\n```\n\n" if expected_sql else ""
        )
        messages = self._sql_prompt_template.format_messages(
            question=question,
            generated_sql=generated_sql,
            answer_context="",
            expected_sql_context=expected_sql_context,
            expected_answer_context="",
        )
        assessment, reasoning = self._ask_judge(messages)
        return {
            "assessment": assessment,
            "reasoning": reasoning,
            "scoring_method": "llm_judge",
        }

    def _ask_judge(self, messages: List[Any]) -> Tuple[str, str]:
        """Send *messages* to the judge in a single fresh call; returns (assessment, reasoning).

        A verdict that is not a JSON object is sent back to the judge, with the
        parser error, for correction.
        """
        attempt_messages = messages
        for attempt in range(LLM_OUTPUT_RETRIES + 1):
            response = self.llm(attempt_messages)
            try:
                evaluation = _extract_json(_response_text(response))
                if not isinstance(evaluation, dict):
                    raise LLMOutputError("the response must be a JSON object with 'assessment' and 'reasoning'.")
                break
            except LLMOutputError as exc:
                if attempt == LLM_OUTPUT_RETRIES:
                    raise
                attempt_messages = _with_output_feedback(self.llm, messages, response, exc)

        assessment = str(evaluation.get("assessment", "incorrect")).strip().lower()
        if assessment not in ("correct", "partial", "incorrect"):
            assessment = "incorrect"
        return assessment, str(evaluation.get("reasoning", ""))[:5000]


def _error_result(reasoning: str) -> Dict[str, Any]:
    return {
        "assessment": "incorrect",
        "reasoning": reasoning,
        "scoring_method": "error",
    }


# ---------------------------------------------------------------------------
# Benchmark run logging helpers
# ---------------------------------------------------------------------------

def _build_benchmark_log_headers(token: str) -> Dict[str, str]:
    """Build Basic Auth headers for the benchmark logging endpoints."""
    encoded = base64.b64encode(f"token:{token}".encode()).decode()
    return {
        "Authorization": f"Basic {encoded}",
        "Content-Type": "application/json",
    }


def _log_benchmark_running(url: str, token: str, payload: Dict[str, Any], verify_ssl: bool = False) -> None:
    """POST to /timbr-server/log_benchmark/running to register a benchmark run start."""
    endpoint = f"{url.rstrip('/')}/timbr-server/log_benchmark/running"
    headers = _build_benchmark_log_headers(token)
    response = requests.post(endpoint, json=payload, headers=headers, verify=verify_ssl)
    if not response.ok:
        raise RuntimeError(
            f"Failed to log benchmark start [{response.status_code}]: {response.text}"
        )


def _log_benchmark_update_completed(url: str, token: str, run_id: str, completed: int, agent_name: str, verify_ssl: bool = False) -> None:
    """POST to /timbr-server/log_benchmark/running_update_completed to update progress."""
    endpoint = f"{url.rstrip('/')}/timbr-server/log_benchmark/running_update_completed"
    headers = _build_benchmark_log_headers(token)
    payload = {"run_id": run_id, "completed": completed, "agent_name": agent_name}
    response = requests.post(endpoint, json=payload, headers=headers, verify=verify_ssl)
    if not response.ok:
        raise RuntimeError(
            f"Failed to update benchmark progress [{response.status_code}]: {response.text}"
        )


def _log_benchmark_history(url: str, token: str, payload: Dict[str, Any], verify_ssl: bool = False) -> None:
    """POST to /timbr-server/log_benchmark/history to finalise a benchmark run."""
    endpoint = f"{url.rstrip('/')}/timbr-server/log_benchmark/history"
    headers = _build_benchmark_log_headers(token)
    response = requests.post(endpoint, json=payload, headers=headers, verify=verify_ssl)
    if not response.ok:
        raise RuntimeError(
            f"Failed to log benchmark history [{response.status_code}]: {response.text}"
        )


# ---------------------------------------------------------------------------
# run_benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    benchmark_name: str,
    queries: Optional[Union[Dict[str, Any], List[str]]] = None,
    url: Optional[str] = None,
    token: Optional[str] = None,
    ontology: Optional[str] = None,
    use_deterministic: Optional[bool] = None,
    use_llm_judge: Optional[bool] = None,
    execution: str = "full",
    number_of_iterations: int = 1,
    verify_ssl: bool = False,
    is_jwt: Optional[bool] = None,
    jwt_tenant_id: Optional[str] = None,
    llm_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run an LLM benchmark against a Timbr agent and return scored results.

    Args:
        benchmark_name: **Mandatory.** The benchmark name (looked up in ``SYS_AGENTS_BENCHMARKS``).
            The associated agent name and default questions are retrieved from this table.
        queries: Questions to evaluate.  Must follow the *questions-enhanced* format::

                {
                    "Q1": {
                        "question": "How many active policies are there?",
                        "correct_sql": "SELECT COUNT(*) FROM Policy WHERE ...",  # optional
                        "expected_answer": "42"  # optional
                    },
                    ...
                }

            When omitted the questions are read from the benchmark's ``benchmark`` field
            in ``SYS_AGENTS_BENCHMARKS`` (stored as a JSON string in the same format).

        url: Timbr server URL.  Defaults to the ``TIMBR_URL`` environment variable.
        token: Timbr authentication token.  Defaults to ``TIMBR_TOKEN``.
        ontology: Ontology / knowledge-graph name.  Defaults to
            ``TIMBR_ONTOLOGY`` / ``ONTOLOGY`` environment variable.
        use_deterministic: Deprecated and ignored. Scoring always uses the LLM judge.
        use_llm_judge: Deprecated and ignored. Scoring always uses the LLM judge.
        verify_ssl: Whether to verify SSL certificates when connecting to Timbr.
            Defaults to ``False``.
        is_jwt: Whether to use JWT authentication. Defaults to ``None``.
        jwt_tenant_id: JWT tenant ID. Defaults to ``None``.
        number_of_iterations: How many times each question is executed. Defaults to ``1``.
            When greater than 1, the benchmark checks result *consistency*: a question is
            marked ``"inconsistent"`` when different iterations produce different assessments,
            and ``consistent`` / ``iterations_detail`` fields are added to each question
            result.
        execution: Execution method. One of:

            * ``"full"`` *(default)* – run the full Timbr SQL agent (concept
              identification → SQL generation → execution → answer generation).
              When ``correct_sql`` is given it is executed too, and the LLM judge
              compares the two JSON row sets blind. Without ``correct_sql`` (or
              when it fails to execute) the judge evaluates the SQL against the
              question.
            * ``"generate_sql_only"`` – use GenerateTimbrSqlChain to generate
              SQL without executing it. The LLM judge evaluates the SQL against
              the question, alongside ``correct_sql`` when it is given.

    Returns:
        A dictionary with one entry per question ID (matching the input *queries*
        keys), each containing the original question data enriched with:

        * ``generated_sql`` – SQL produced by the agent
        * ``selected_entity`` – Timbr concept chosen by the agent
        * ``selected_ontology`` – Timbr ontology selected by the agent
        * ``answer`` – natural-language answer (if the agent generates one)
        * ``timbr_reasoning_status`` – Timbr's own SQL reasoning assessment
        * ``identify_concept_reason`` – reason returned by concept identification step
        * ``generate_sql_reason`` – reason returned by SQL generation step
        * ``correct_concept`` – ``True``/``False`` if expected concept is provided, else ``None``
        * ``correct_ontology`` – ``True``/``False`` if expected ontology is provided, else ``None``
        * ``tokens_used`` – total tokens consumed for this question
        * ``duration_ms`` – time (ms) the agent / chain took to complete this
          question, averaged across iterations (each iteration's own time is in
          ``iterations_detail``). Scoring time is not included.
        * ``status`` – ``"correct"``, ``"partial"``, or ``"incorrect"``
        * ``scoring_method`` – ``"llm_judge"``, or ``"error"`` when no verdict
          was produced (no SQL, a failed query, or a judge failure)
        * ``score_reasoning`` – human-readable scoring explanation

        A special ``"_summary"`` key contains aggregate statistics and the run
        configuration.

    Raises:
        ValueError: If *benchmark_name* is missing, if *queries* cannot be resolved, or if
                    required connection parameters are not available.
        RuntimeError: If any benchmark logging HTTP call fails.
    """
    if not benchmark_name:
        raise ValueError("The 'benchmark_name' parameter is mandatory.")

    _valid_execution_modes = ("full", "generate_sql_only")
    if execution not in _valid_execution_modes:
        raise ValueError(
            f"Invalid execution mode '{execution}'. Must be one of: {_valid_execution_modes}."
        )

    # Build system-level conn_params for calls to timbr schema (sys_agents_options)
    thrift_host = config.thrift_host
    thrift_port = config.thrift_port

    if not thrift_host or not thrift_port:
        raise ValueError(
            "Thrift host and port are required for benchmark execution. "
            "Set THRIFT_HOST and THRIFT_PORT environment variables."
        )

    resolved_url = url or config.url
    resolved_token = token or config.token
    server_url = build_server_url(resolved_url, thrift_host, thrift_port)

    if not resolved_url:
        raise ValueError(
            "Timbr URL is required. Pass 'url' or set the TIMBR_URL environment variable."
        )
    if not resolved_token:
        raise ValueError(
            "Timbr token is required. Pass 'token' or set the TIMBR_TOKEN environment variable."
        )

    system_conn_params: Dict[str, Any] = {
        "url": resolved_url,
        "token": resolved_token,
        "ontology": "system_db",
        "verify_ssl": verify_ssl,
        "is_jwt": is_jwt if is_jwt is not None else config.is_jwt,
        "jwt_tenant_id": jwt_tenant_id if jwt_tenant_id is not None else config.jwt_tenant_id,
    }

    # ------------------------------------------------------------------
    # Resolve benchmark info
    # ------------------------------------------------------------------
    logger.info(f"Fetching benchmark info for '{benchmark_name}'…")
    benchmark_info = get_timbr_benchmark_info(benchmark_name, system_conn_params)
    agent_name: str = benchmark_info["agent_name"]

    # ------------------------------------------------------------------
    # Resolve agent options
    # ------------------------------------------------------------------
    logger.info(f"Fetching agent options for '{agent_name}'…")
    agent_options = get_timbr_agent_options(agent_name, system_conn_params)

    # ------------------------------------------------------------------
    # Resolve queries
    # ------------------------------------------------------------------
    def _load_queries_from_benchmark_info() -> Dict[str, Any]:
        """Load and normalise the full questions dict from SYS_AGENTS_BENCHMARKS."""
        raw_questions = benchmark_info.get("benchmark")
        if not raw_questions:
            raise ValueError(
                f"No 'queries' argument provided and benchmark '{benchmark_name}' has no "
                "'benchmark' (questions) field in SYS_AGENTS_BENCHMARKS."
            )
        try:
            loaded = json.loads(raw_questions)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Benchmark field 'benchmark' is not valid JSON: {exc}"
            ) from exc

        if isinstance(loaded, list):
            return {f"Q{i + 1}": {"question": q} for i, q in enumerate(loaded)}
        if isinstance(loaded, dict):
            return loaded
        raise ValueError(
            "Benchmark field 'benchmark' must be a JSON object (dict) or array (list)."
        )

    if queries is None:
        queries = _load_queries_from_benchmark_info()

    elif isinstance(queries, list):
        # List of question ID strings — load full set from DB then filter to the requested IDs.
        question_ids: List[str] = queries
        full_queries = _load_queries_from_benchmark_info()
        missing = [qid for qid in question_ids if qid not in full_queries]
        if missing:
            logger.warning(
                "The following question IDs were not found in benchmark '%s' and will be "
                "skipped: %s",
                benchmark_name,
                missing,
            )
        queries = {qid: full_queries[qid] for qid in question_ids if qid in full_queries}
        if not queries:
            raise ValueError(
                f"None of the provided question IDs exist in benchmark '{benchmark_name}'. "
                f"Requested: {question_ids}. Available: {list(full_queries.keys())}"
            )

    if not queries:
        raise ValueError("No questions to benchmark.")

    # ------------------------------------------------------------------
    # Scoring always uses the LLM judge. The old mechanism params and agent
    # options (use_deterministic_scoring / use_llm_judge_scoring) are ignored.
    # ------------------------------------------------------------------
    if use_deterministic is not None:
        _warn_deprecated_mechanism("use_deterministic")
    if use_llm_judge is not None:
        _warn_deprecated_mechanism("use_llm_judge")

    # ------------------------------------------------------------------
    # Resolve ontology and agent-level connection details
    # ------------------------------------------------------------------
    resolved_ontology = (
        ontology
        or agent_options.get("ontology")
        or config.ontology
    )

    agent_conn_params: Dict[str, Any] = {
        "url": resolved_url,
        "token": resolved_token,
        "ontology": resolved_ontology,
        "verify_ssl": verify_ssl,
        "is_jwt": is_jwt if is_jwt is not None else config.is_jwt,
        "jwt_tenant_id": jwt_tenant_id if jwt_tenant_id is not None else config.jwt_tenant_id,
    }

    # ------------------------------------------------------------------
    # Resolve LLM info (used for judge scoring and benchmark logging)
    # ------------------------------------------------------------------
    llm_type: str = agent_options.get("llm_type") or config.llm_type or ""
    llm_model: str = agent_options.get("llm_model") or config.llm_model or ""
    llm_api_key: Optional[str] = agent_options.get("llm_api_key") or config.llm_api_key

    # Runtime override: if explicit llm_params passed
    # they take precedence over agent_options and config.
    # Keys match LlmWrapper constructor: llm_type, model, api_key (+ extras like endpoint).
    _use_llm_params = bool(llm_params)
    if llm_params:
        # Extract named LlmWrapper params — pop so they don't leak into **llm_params
        llm_type    = llm_params.pop("llm_type",    None) or llm_type
        llm_model   = llm_params.pop("llm_model",   None) or llm_params.pop("model",   None) or llm_model
        llm_api_key = llm_params.pop("llm_api_key", None) or llm_params.pop("api_key", None) or llm_api_key
        # Temperature is always forced to 0 for judge / SQL generation
        llm_params.pop("temperature",     None)
        llm_params.pop("llm_temperature", None)
        # Timeout is handled separately
        llm_params.pop("llm_timeout", None)
        additional_params = llm_params.pop("llm_additional_params", None)
        if additional_params and isinstance(additional_params, dict):
            llm_params.update(additional_params)

    # ------------------------------------------------------------------
    # Build the judge LLM. It is its own wrapper, called with stateless
    # one-shot prompts, so the judge never shares the agent's context.
    # ------------------------------------------------------------------
    judge_llm = LlmWrapper(
        llm_type=llm_type,
        model=llm_model,
        api_key=llm_api_key,
        **(llm_params or {}),
    )

    # ------------------------------------------------------------------
    # Instantiate scorer and agent executor / SQL-only chain
    # ------------------------------------------------------------------
    scorer = BenchmarkScorer(conn_params=agent_conn_params, llm=judge_llm)

    agent_executor = None
    sql_chain = None

    if execution == "generate_sql_only":
        # Build an LLM for the chain (same config as the judge LLM, but a separate
        # instance so the chain and the judge never share state)
        sql_chain_llm: Optional[Any] = None
        if llm_type and llm_api_key:
            sql_chain_llm = LlmWrapper(
                llm_type=llm_type,
                model=llm_model,
                api_key=llm_api_key,
                **(llm_params or {}),
            )
        logger.info(f"Creating GenerateTimbrSqlChain for '{agent_name}'…")
        sql_chain = GenerateTimbrSqlChain(
            url=resolved_url,
            token=resolved_token,
            agent=agent_name,
            verify_ssl=verify_ssl,
            is_jwt=is_jwt,
            jwt_tenant_id=jwt_tenant_id,
            llm=sql_chain_llm,
        )
    else:
        logger.info(f"Creating Timbr SQL agent for '{agent_name}'…")
        override_llm = LlmWrapper(
            llm_type=llm_type,
            model=llm_model,
            api_key=llm_api_key,
            **llm_params,
        ) if _use_llm_params else None
        agent_executor = create_timbr_sql_agent(
            llm=override_llm,
            url=resolved_url,
            token=resolved_token,
            # ontology=resolved_ontology,
            agent=agent_name,
            # include_tags="*",
            # enable_reasoning=to_boolean(agent_options.get("enable_reasoning", config.enable_reasoning)),
            # graph_depth=to_integer(agent_options.get("graph_depth", 1)),
            verify_ssl=verify_ssl,
            is_jwt=is_jwt,
            jwt_tenant_id=jwt_tenant_id,
            generate_answer=True,
        )

    # ------------------------------------------------------------------
    # Initialise run tracking
    # ------------------------------------------------------------------
    run_id = str(uuid.uuid4())
    start_time = datetime.now()
    total_questions = len(queries)

    _log_benchmark_running(
        url=server_url,
        token=resolved_token,
        payload={
            "benchmark_name": benchmark_name,
            "agent_name": agent_name,
            "run_id": run_id,
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "number_of_questions": total_questions,
            "execution": execution,
            "number_of_iterations": number_of_iterations,
            "completed": 0,
        },
        verify_ssl=verify_ssl,
    )

    # ------------------------------------------------------------------
    # Benchmark loop
    # ------------------------------------------------------------------
    benchmark_results: Dict[str, Any] = copy.deepcopy(queries)
    correct_count = partial_count = incorrect_count = inconsistent_count = error_count = 0
    total_tokens_used = 0
    completed_count = 0

    logger.info(
        f"Running benchmark '{benchmark_name}' on {total_questions} question(s) "
        f"[execution={execution}, iterations={number_of_iterations}]…"
    )

    for question_id, question_data in queries.items():
        question_text: str = question_data.get("question", "")
        correct_sql: Optional[str] = question_data.get("correct_sql")
        expected_answer: Optional[str] = question_data.get("expected_answer")
        expected_concept: Optional[str] = question_data.get("correct_concept")
        expected_ontology: Optional[str] = question_data.get("correct_ontology")

        logger.info(f"  [{question_id}] {question_text[:100]}…")

        # Execute correct SQL once (outside iteration loop) so the judge can compare
        # the agent's rows with the expected rows. In generate_sql_only mode we never
        # execute SQL, so skip this. On failure the judge falls back to judging SQL.
        expected_rows: Optional[List[Dict[str, Any]]] = None
        expected_rows_error: Optional[str] = None
        if execution == "full" and correct_sql:
            try:
                expected_rows = timbr_http_connector.run_query(
                    query=correct_sql.replace(";", ""),
                    url=resolved_url,
                    token=resolved_token,
                    ontology=resolved_ontology,
                    verify_ssl=verify_ssl,
                    is_jwt=is_jwt,
                    jwt_tenant_id=jwt_tenant_id,
                ) or []
            except Exception as exc:
                logger.warning(f"  [{question_id}] Failed to execute correct SQL: {exc}")
                expected_rows = None
                expected_rows_error = str(exc)[:500]

        # ------------------------------------------------------------------
        # Iterations inner loop
        # ------------------------------------------------------------------
        iteration_results: List[Dict[str, Any]] = []
        question_tokens_total: int = 0
        question_duration_total_ms: int = 0
        had_error_in_any_iteration = False
        last_llm_result: Dict[str, Any] = {}
        last_score_result: Dict[str, Any] = {}

        for iter_num in range(1, number_of_iterations + 1):
            iter_label = f"[{question_id}][{iter_num}/{number_of_iterations}]" if number_of_iterations > 1 else f"[{question_id}]"

            # ---- Execute via agent or SQL-only chain ----
            # Only the agent / chain call is timed: expected-SQL execution and
            # scoring are benchmark overhead, not part of answering the input.
            iter_start = perf_counter()
            try:
                if execution == "generate_sql_only":
                    raw = sql_chain.invoke({"prompt": question_text})  # type: ignore[union-attr]
                    # Normalise the chain result to the same shape as agent_executor output
                    llm_result: Dict[str, Any] = {
                        "sql": raw.get("sql"),
                        "rows": [],
                        "answer": "",
                        "ontology": raw.get("ontology") or resolved_ontology,
                        "concept": raw.get("concept"),
                        "schema": raw.get("schema"),
                        "error": raw.get("error"),
                        "reasoning_status": raw.get("reasoning_status"),
                        "identify_concept_reason": raw.get("identify_concept_reason"),
                        "generate_sql_reason": raw.get("generate_sql_reason"),
                        "usage_metadata": raw.get("generate_sql_usage_metadata") or {},
                    }
                else:
                    llm_result = agent_executor.invoke({"input": question_text})  # type: ignore[misc]
            except Exception as exc:
                llm_result = {"sql": None, "rows": [], "error": str(exc), "usage_metadata": {}}
            iter_duration_ms = int((perf_counter() - iter_start) * 1000)
            question_duration_total_ms += iter_duration_ms

            if llm_result.get("error"):
                had_error_in_any_iteration = True

            generated_sql: str = llm_result.get("sql") or ""
            generated_rows: List[Dict[str, Any]] = llm_result.get("rows") or []

            # ---- Collect token usage ----
            usage_metadata = llm_result.get("usage_metadata") or {}
            iter_tokens = sum(
                v.get("total_tokens", v.get("approximate", 0))
                for v in usage_metadata.values()
                if isinstance(v, dict)
            )
            question_tokens_total += iter_tokens

            # ---- Score ----
            score_result = scorer.score_result(
                question=question_text,
                generated_sql=generated_sql,
                answer=llm_result.get("answer") or "",
                generated_rows=generated_rows,
                expected_sql=correct_sql,
                expected_answer=expected_answer,
                expected_rows=expected_rows,
                execution_error=llm_result.get("error"),
                execution_mode=execution,
                expected_rows_error=expected_rows_error,
            )

            iter_status: str = score_result["assessment"]
            iter_entry: Dict[str, Any] = {
                "iteration": iter_num,
                "generated_sql": generated_sql,
                "status": iter_status,
                "scoring_method": score_result["scoring_method"],
                "tokens_used": iter_tokens,
                "duration_ms": iter_duration_ms,
            }
            if "reasoning" in score_result:
                iter_entry["score_reasoning"] = score_result["reasoning"]
            iteration_results.append(iter_entry)

            last_llm_result = llm_result
            last_score_result = score_result

            logger.info(f"  {iter_label} → {iter_status.upper()}")

        # ---- Determine final status and consistency ----
        if number_of_iterations == 1:
            result_status: str = iteration_results[0]["status"]
            consistent = True
        else:
            statuses = [r["status"] for r in iteration_results]
            consistent = len(set(statuses)) == 1
            if consistent:
                result_status = statuses[0]
            else:
                result_status = "inconsistent"
                inconsistent_count += 1

        if had_error_in_any_iteration:
            error_count += 1

        total_tokens_used += question_tokens_total

        # ---- Write question results ----
        selected_concept = last_llm_result.get("concept")
        selected_ontology = last_llm_result.get("ontology")
        benchmark_results[question_id]["generated_sql"] = last_llm_result.get("sql") or ""
        benchmark_results[question_id]["selected_entity"] = selected_concept or ""
        benchmark_results[question_id]["selected_ontology"] = selected_ontology or ""
        benchmark_results[question_id]["answer"] = last_llm_result.get("answer") or ""
        benchmark_results[question_id]["timbr_reasoning_status"] = last_llm_result.get("reasoning_status") or ""
        benchmark_results[question_id]["identify_concept_reason"] = last_llm_result.get("identify_concept_reason")
        benchmark_results[question_id]["generate_sql_reason"] = last_llm_result.get("generate_sql_reason")
        benchmark_results[question_id]["correct_concept"] = _matches_expected_value(expected_concept, selected_concept)
        benchmark_results[question_id]["correct_ontology"] = _matches_expected_value(expected_ontology, selected_ontology)
        benchmark_results[question_id]["tokens_used"] = question_tokens_total
        # Average across iterations; per-iteration times are in iterations_detail.
        benchmark_results[question_id]["duration_ms"] = round(question_duration_total_ms / number_of_iterations)
        benchmark_results[question_id]["status"] = result_status
        if last_llm_result.get("error"):
            benchmark_results[question_id]["error"] = last_llm_result.get("error")
        benchmark_results[question_id]["scoring_method"] = last_score_result.get("scoring_method", "error")
        if "reasoning" in last_score_result:
            benchmark_results[question_id]["score_reasoning"] = last_score_result["reasoning"]

        # Store per-iteration detail and consistency flag when iterations > 1
        if number_of_iterations > 1:
            benchmark_results[question_id]["iterations_detail"] = iteration_results
            benchmark_results[question_id]["consistent"] = consistent

        if result_status == "correct":
            correct_count += 1
        elif result_status == "partial":
            partial_count += 1
        elif result_status == "incorrect":
            incorrect_count += 1
        # "inconsistent" is already counted above

        completed_count += 1
        _log_benchmark_update_completed(
            url=server_url,
            token=resolved_token,
            run_id=run_id,
            completed=completed_count,
            agent_name=agent_name,
            verify_ssl=verify_ssl,
        )

        logger.info(f"  [{question_id}] FINAL → {result_status.upper()}")

    # ------------------------------------------------------------------
    # Finalise run and log history
    # ------------------------------------------------------------------
    end_time = datetime.now()
    duration_ms = int((end_time - start_time).total_seconds() * 1000)
    correct_rate = round((correct_count + partial_count) / total_questions * 100, 2) if total_questions > 0 else 0.0

    _log_benchmark_history(
        url=server_url,
        token=resolved_token,
        payload={
            "benchmark_name": benchmark_name,
            "agent_name": agent_name,
            "run_id": run_id,
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": duration_ms,
            "number_of_questions": total_questions,
            "correct_count": correct_count + partial_count,  # Count partial as correct for the summary log
            "partial_count": partial_count,
            "incorrect_count": incorrect_count,
            "inconsistent_count": inconsistent_count,
            "error_count": error_count,
            "correct_rate": correct_rate,
            "execution": execution,
            "number_of_iterations": number_of_iterations,
            "total_tokens_used": total_tokens_used,
            "langchain_timbr_version": _langchain_timbr_version,
            "llm_type": llm_type,
            "llm_model": llm_model,
            "result": benchmark_results,
        },
        verify_ssl=verify_ssl,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    benchmark_results["_summary"] = {
        "total_questions": total_questions,
        "correct_count": correct_count + partial_count,  # Count partial as correct for the summary log
        "partial_count": partial_count,
        "incorrect_count": incorrect_count,
        "inconsistent_count": inconsistent_count,
        "error_count": error_count,
        "correct_rate": correct_rate,
        "total_tokens_used": total_tokens_used,
        "timestamp": start_time.isoformat(),
        "duration_ms": duration_ms,
        "config": {
            "benchmark_name": benchmark_name,
            "agent_name": agent_name,
            "ontology": resolved_ontology,
            "timbr_url": resolved_url,
            "scoring_method": "llm_judge",
            "execution": execution,
            "number_of_iterations": number_of_iterations,
        },
    }

    return benchmark_results
