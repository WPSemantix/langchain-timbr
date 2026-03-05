"""
LLM Benchmark utility for evaluating Timbr SQL agent performance.

Runs a set of questions through a Timbr agent and scores the generated SQL and
answers using basic heuristics, deterministic row-comparison, or an LLM-as-judge
approach (all three modes are configurable).

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
import uuid
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import requests

from pytimbr_api import timbr_http_connector

from .. import config
from ..langchain.timbr_sql_agent import create_timbr_sql_agent
from ..llm_wrapper.llm_wrapper import LlmWrapper
from .general import to_boolean, to_integer
from .prompt_service import get_benchmark_judge_prompt_template
from .timbr_utils import get_timbr_agent_options, get_timbr_benchmark_info

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
# Result comparison helpers (ported from LLM Benchmark utils.py)
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


def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a row by lowercasing keys and normalizing values."""
    return {
        k.lower().strip().replace(' ', '_'): _normalize_value(v)
        for k, v in row.items()
    }


def _normalize_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize a list of rows and sort them for order-independent comparison."""
    if not results:
        return []
    normalized = [_normalize_row(r) for r in results]
    normalized.sort(key=lambda r: str(sorted(r.items())))
    return normalized


def _compare_results(
    llm_results: List[Dict[str, Any]],
    correct_results: List[Dict[str, Any]],
) -> Tuple[bool, str]:
    """
    Compare LLM result rows with expected rows.

    Column names do not need to match – only the *values* must match.
    Rows may appear in any order.

    Returns:
        Tuple of (is_match: bool, error_message: str)
    """
    try:
        norm_llm = _normalize_results(llm_results)
        norm_correct = _normalize_results(correct_results)

        if len(norm_llm) != len(norm_correct):
            return (
                False,
                f"Row count mismatch: LLM returned {len(norm_llm)} rows, expected {len(norm_correct)} rows",
            )

        if len(norm_llm) == 0 and len(norm_correct) == 0:
            return True, ""

        used_llm_rows: set = set()
        for i, correct_row in enumerate(norm_correct):
            correct_values = list(correct_row.values())
            row_matched = False
            for j, llm_row in enumerate(norm_llm):
                if j in used_llm_rows:
                    continue
                llm_values = list(llm_row.values())
                if all(cv in llm_values for cv in correct_values):
                    used_llm_rows.add(j)
                    row_matched = True
                    break
            if not row_matched:
                return (
                    False,
                    f"No matching row found for correct row {i + 1} with values: "
                    f"{dict(zip(list(correct_row.keys()), correct_values))}",
                )

        return True, ""
    except Exception as exc:
        return False, f"Error during comparison: {str(exc)}"


# ---------------------------------------------------------------------------
# BenchmarkScorer
# ---------------------------------------------------------------------------

class BenchmarkScorer:
    """
    Score benchmark results using a configurable combination of:

    * **basic** – pure heuristics (no expected data required)
    * **deterministic** – row comparison between LLM and expected results
    * **llm_judge** – LLM evaluates the SQL + answer quality

    When both *deterministic* and *llm_judge* are enabled the deterministic
    assessment takes priority and the scoring method is reported as ``"hybrid"``.
    """

    def __init__(
        self,
        conn_params: Dict[str, Any],
        llm: Optional[Any] = None,
        use_deterministic: bool = False,
        use_llm_judge: bool = False,
    ):
        """
        Args:
            conn_params: Timbr connection parameters (url, token, ontology, …).
                         Used to fetch the benchmark-judge prompt template from the API.
            llm: An LlmWrapper instance (or compatible LangChain LLM).
                 Required only when *use_llm_judge* is ``True``.
            use_deterministic: Enable deterministic row comparison scoring.
            use_llm_judge: Enable LLM-as-judge scoring (requires *llm*).
        """
        self.conn_params = conn_params
        self.use_deterministic = use_deterministic
        self.use_llm_judge = use_llm_judge
        self.llm = llm if use_llm_judge else None

        # Lazy-loaded template wrapper (fetched on first judge call)
        self._judge_prompt_template = None

    # ------------------------------------------------------------------
    # Public scoring entry-point
    # ------------------------------------------------------------------

    def score_result(
        self,
        question: str,
        generated_sql: str,
        answer: str,
        generated_rows: Optional[List[Dict[str, Any]]] = None,
        expected_sql: Optional[str] = None,
        expected_answer: Optional[str] = None,
        expected_rows: Optional[List[Dict[str, Any]]] = None,
        execution_error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Score a single benchmark result.

        Returns a dict with keys:
            ``assessment``    – ``"correct"``, ``"partial"``, or ``"incorrect"``
            ``breakdown``     – method-specific detail (populated by deterministic mode)
            ``scoring_method``– one of ``"basic"``, ``"deterministic"``,
                                ``"llm_judge"``, ``"hybrid"``, ``"error"``
            ``reasoning``     – optional human-readable explanation string
        """
        if not generated_sql or not generated_sql.strip():
            return {
                "assessment": "incorrect",
                "breakdown": {},
                "reasoning": "No SQL query was generated",
                "scoring_method": "error",
            }

        methods_to_use: List[str] = []
        if self.use_deterministic:
            methods_to_use.append("deterministic")
        if self.use_llm_judge:
            methods_to_use.append("llm_judge")

        if not methods_to_use:
            return self._basic_score(generated_sql, answer, execution_error)

        all_assessments: List[Dict[str, Any]] = []
        combined_breakdown: Dict[str, Any] = {}
        combined_reasoning: List[str] = []

        if "deterministic" in methods_to_use:
            det_result = self._deterministic_score(
                generated_sql,
                answer,
                generated_rows,
                expected_sql,
                expected_answer,
                expected_rows,
                execution_error,
            )
            all_assessments.append(det_result)
            combined_breakdown.update(
                {f"det_{k}": v for k, v in det_result["breakdown"].items()}
            )
            if det_result.get("reasoning"):
                combined_reasoning.append(f"[Deterministic] {det_result['reasoning']}")

        if "llm_judge" in methods_to_use:
            llm_result = self._llm_judge_score(question, generated_sql, answer)
            all_assessments.append(llm_result)
            combined_breakdown.update(
                {f"llm_{k}": v for k, v in llm_result["breakdown"].items()}
            )
            if llm_result.get("reasoning"):
                combined_reasoning.append(f"[LLM Judge] {llm_result['reasoning']}")

        if len(all_assessments) > 1:
            # Deterministic takes priority when both are enabled
            if "deterministic" in methods_to_use:
                final_assessment = all_assessments[0]["assessment"]
            else:
                assessments = [a["assessment"] for a in all_assessments]
                if "incorrect" in assessments:
                    final_assessment = "incorrect"
                elif "partial" in assessments:
                    final_assessment = "partial"
                else:
                    final_assessment = "correct"
            scoring_method = "hybrid"
        else:
            final_assessment = all_assessments[0]["assessment"]
            scoring_method = all_assessments[0]["scoring_method"]

        result: Dict[str, Any] = {
            "assessment": final_assessment,
            "breakdown": combined_breakdown,
            "scoring_method": scoring_method,
        }
        if combined_reasoning:
            result["reasoning"] = " | ".join(combined_reasoning)

        return result

    # ------------------------------------------------------------------
    # Private scoring implementations
    # ------------------------------------------------------------------

    def _deterministic_score(
        self,
        generated_sql: str,
        answer: str,
        generated_rows: Optional[List[Dict[str, Any]]],
        expected_sql: Optional[str],
        expected_answer: Optional[str],
        expected_rows: Optional[List[Dict[str, Any]]],
        execution_error: Optional[str],
    ) -> Dict[str, Any]:
        breakdown: Dict[str, Any] = {}
        assessment = "correct"
        reasoning_parts: List[str] = []

        if execution_error:
            breakdown["execution_success"] = "failed"
            assessment = "incorrect"
            reasoning_parts.append("Query failed to execute")
        else:
            breakdown["execution_success"] = "passed"

        if generated_rows is not None and expected_rows is not None:
            is_match, error_message = _compare_results(generated_rows, expected_rows)
            if is_match:
                breakdown["result_accuracy"] = "matched"
                assessment = "correct"
                reasoning_parts.append("Results match expected output")
            else:
                breakdown["result_accuracy"] = "mismatched"
                assessment = "incorrect"
                reasoning_parts.append(f"Results mismatch: {error_message}")

        if expected_sql:
            breakdown["query_similarity"] = round(
                self._score_query_similarity(generated_sql, expected_sql), 2
            )

        if expected_answer:
            breakdown["answer_similarity"] = round(
                self._score_answer_similarity(answer, expected_answer), 2
            )

        return {
            "assessment": assessment,
            "breakdown": breakdown,
            "scoring_method": "deterministic",
            "reasoning": " | ".join(reasoning_parts) if reasoning_parts else "Deterministic comparison",
        }

    def _llm_judge_score(
        self,
        question: str,
        generated_sql: str,
        answer: str,
    ) -> Dict[str, Any]:
        """Use LLM (via the benchmark-judge template from timbr-api) to score the result."""
        try:
            if self._judge_prompt_template is None:
                self._judge_prompt_template = get_benchmark_judge_prompt_template(
                    conn_params=self.conn_params
                )

            messages = self._judge_prompt_template.format_messages(
                question=question,
                generated_sql=generated_sql,
                answer=answer or "(no answer generated)",
                expected_sql_context="",
                expected_answer_context="",
            )

            response = self.llm(messages)

            content = response.content.strip()
            # Strip markdown code fences if present
            for fence in ("```json", "```"):
                if content.startswith(fence):
                    content = content[len(fence):]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            evaluation = json.loads(content)
            assessment = evaluation.get("assessment", "incorrect").lower()
            if assessment not in ("correct", "partial", "incorrect"):
                assessment = "incorrect"

            return {
                "assessment": assessment,
                "breakdown": {},
                "reasoning": str(evaluation.get("reasoning", ""))[:200],
                "scoring_method": "llm_judge",
            }

        except Exception as exc:
            logger.warning(f"LLM judge scoring failed, falling back to basic: {exc}")
            return self._basic_score(generated_sql, answer)

    def _basic_score(
        self,
        generated_sql: str,
        answer: str,
        execution_error: Optional[str] = None,
    ) -> Dict[str, Any]:
        if execution_error:
            return {"assessment": "incorrect", "breakdown": {}, "reasoning": "Query execution failed", "scoring_method": "basic"}
        if not generated_sql or not generated_sql.strip():
            return {"assessment": "incorrect", "breakdown": {}, "reasoning": "No SQL generated", "scoring_method": "basic"}
        if not answer or not answer.strip():
            return {"assessment": "partial", "breakdown": {}, "reasoning": "SQL generated but no answer provided", "scoring_method": "basic"}

        sql_lower = generated_sql.lower()
        if "select" in sql_lower and "from" in sql_lower:
            return {"assessment": "partial", "breakdown": {}, "reasoning": "Basic SQL structure valid, but cannot verify correctness without expected results", "scoring_method": "basic"}

        return {"assessment": "incorrect", "breakdown": {}, "reasoning": "Invalid SQL structure", "scoring_method": "basic"}

    # ------------------------------------------------------------------
    # Similarity helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_query_similarity(query1: str, query2: str) -> float:
        if not query1 or not query2:
            return 0.0
        q1 = " ".join(query1.lower().split())
        q2 = " ".join(query2.lower().split())
        if q1 == q2:
            return 10.0
        return round(SequenceMatcher(None, q1, q2).ratio() * 10, 2)

    @staticmethod
    def _score_answer_similarity(answer1: str, answer2: str) -> float:
        if not answer1 or not answer2:
            return 0.0
        a1 = answer1.strip().lower()
        a2 = answer2.strip().lower()
        if a1 == a2:
            return 10.0
        try:
            num1 = float(a1.replace(",", ""))
            num2 = float(a2.replace(",", ""))
            pct = abs(num1 - num2) / num2 * 100 if num2 != 0 else 100
            if pct < 0.01:
                return 10.0
            elif pct < 1:
                return 9.0
            elif pct < 5:
                return 7.0
            elif pct < 10:
                return 5.0
            else:
                return max(3.0, 10.0 - pct / 10)
        except ValueError:
            pass
        return round(SequenceMatcher(None, a1, a2).ratio() * 10, 2)


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


def _log_benchmark_running(url: str, token: str, payload: Dict[str, Any]) -> None:
    """POST to /timbr-server/log_benchmark/running to register a benchmark run start."""
    endpoint = f"{url.rstrip('/')}/timbr-server/log_benchmark/running"
    headers = _build_benchmark_log_headers(token)
    response = requests.post(endpoint, json=payload, headers=headers)
    if not response.ok:
        raise RuntimeError(
            f"Failed to log benchmark start [{response.status_code}]: {response.text}"
        )


def _log_benchmark_update_completed(url: str, token: str, run_id: str, completed: int, agent_name: str) -> None:
    """POST to /timbr-server/log_benchmark/running_update_completed to update progress."""
    endpoint = f"{url.rstrip('/')}/timbr-server/log_benchmark/running_update_completed"
    headers = _build_benchmark_log_headers(token)
    payload = {"run_id": run_id, "completed": completed, "agent_name": agent_name}
    response = requests.post(endpoint, json=payload, headers=headers)
    if not response.ok:
        raise RuntimeError(
            f"Failed to update benchmark progress [{response.status_code}]: {response.text}"
        )


def _log_benchmark_history(url: str, token: str, payload: Dict[str, Any]) -> None:
    """POST to /timbr-server/log_benchmark/history to finalise a benchmark run."""
    endpoint = f"{url.rstrip('/')}/timbr-server/log_benchmark/history"
    headers = _build_benchmark_log_headers(token)
    response = requests.post(endpoint, json=payload, headers=headers)
    if not response.ok:
        raise RuntimeError(
            f"Failed to log benchmark history [{response.status_code}]: {response.text}"
        )


# ---------------------------------------------------------------------------
# run_benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    benchmark_name: str,
    queries: Optional[Dict[str, Any]] = None,
    url: Optional[str] = None,
    token: Optional[str] = None,
    ontology: Optional[str] = None,
    use_deterministic: Optional[bool] = None,
    use_llm_judge: Optional[bool] = None,
    verify_ssl: bool = False,
    is_jwt: Optional[bool] = None,
    jwt_tenant_id: Optional[str] = None,
    number_of_iterations: int = 1,
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
        use_deterministic: Enable deterministic row-comparison scoring.
            Overrides the agent option ``use_deterministic_scoring``.
            Defaults to ``False``.
        use_llm_judge: Enable LLM-as-judge scoring.
            Overrides the agent option ``use_llm_judge_scoring``.
            Defaults to ``False``.
        verify_ssl: Whether to verify SSL certificates when connecting to Timbr.
            Defaults to ``False``.
        is_jwt: Whether to use JWT authentication. Defaults to ``None``.
        jwt_tenant_id: JWT tenant ID. Defaults to ``None``.
        number_of_iterations: Number of iterations for the benchmark run. Defaults to ``1``.

    Returns:
        A dictionary with one entry per question ID (matching the input *queries*
        keys), each containing the original question data enriched with:

        * ``generated_sql`` – SQL produced by the agent
        * ``selected_concept`` – Timbr concept chosen by the agent
        * ``answer`` – natural-language answer (if the agent generates one)
        * ``timbr_reasoning_status`` – Timbr's own SQL reasoning assessment
        * ``tokens_used`` – total tokens consumed for this question
        * ``status`` – ``"correct"``, ``"partial"``, or ``"incorrect"``
        * ``score_breakdown`` – method-specific detail dict
        * ``scoring_method`` – ``"basic"``, ``"deterministic"``, ``"llm_judge"``,
          ``"hybrid"``, or ``"error"``
        * ``score_reasoning`` – (optional) human-readable scoring explanation

        A special ``"_summary"`` key contains aggregate statistics and the run
        configuration.

    Raises:
        ValueError: If *benchmark_name* is missing, if *queries* cannot be resolved, or if
                    required connection parameters are not available.
        RuntimeError: If any benchmark logging HTTP call fails.
    """
    if not benchmark_name:
        raise ValueError("The 'benchmark_name' parameter is mandatory.")

    # Build system-level conn_params for calls to timbr schema (sys_agents_options)
    thrift_host = config.thrift_host
    thrift_port = config.thrift_port

    if not thrift_host or not thrift_port:
        raise ValueError(
            "Thrift host and port are required for benchmark execution. "
            "Set THRIFT_HOST and THRIFT_PORT environment variables."
        )

    server_url = f"{thrift_host}:{thrift_port}"
    resolved_url = url or config.url
    resolved_token = token or config.token

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
    if queries is None:
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
            queries = {f"Q{i + 1}": {"question": q} for i, q in enumerate(loaded)}
        elif isinstance(loaded, dict):
            queries = loaded
        else:
            raise ValueError(
                "Benchmark field 'benchmark' must be a JSON object (dict) or array (list)."
            )

    if not queries:
        raise ValueError("No questions to benchmark.")

    # ------------------------------------------------------------------
    # Resolve scoring modes
    # param > agent option > default (False)
    # ------------------------------------------------------------------
    def _resolve_flag(param_val: Optional[bool], option_key: str) -> bool:
        if param_val is not None:
            return param_val
        raw = agent_options.get(option_key)
        if raw is not None:
            return to_boolean(raw)
        return False

    resolved_use_deterministic = _resolve_flag(use_deterministic, "use_deterministic_scoring")
    resolved_use_llm_judge = _resolve_flag(use_llm_judge, "use_llm_judge_scoring")

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

    # ------------------------------------------------------------------
    # Build LLM wrapper for judge scoring (if needed)
    # ------------------------------------------------------------------
    judge_llm: Optional[Any] = None
    if resolved_use_llm_judge:
        judge_llm = LlmWrapper(
            llm_type=llm_type,
            model=llm_model,
            api_key=llm_api_key,
            temperature=0,
        )

    # ------------------------------------------------------------------
    # Instantiate scorer and agent executor
    # ------------------------------------------------------------------
    scorer = BenchmarkScorer(
        conn_params=agent_conn_params,
        llm=judge_llm,
        use_deterministic=resolved_use_deterministic,
        use_llm_judge=resolved_use_llm_judge,
    )

    logger.info(f"Creating Timbr SQL agent for '{agent_name}'…")
    agent_executor = create_timbr_sql_agent(
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
            "completed": 0,
        },
    )

    # ------------------------------------------------------------------
    # Benchmark loop
    # ------------------------------------------------------------------
    benchmark_results: Dict[str, Any] = copy.deepcopy(queries)
    correct_count = partial_count = incorrect_count = inconsistent_count = error_count = 0
    total_tokens_used = 0
    completed_count = 0

    logger.info(f"Running benchmark '{benchmark_name}' on {total_questions} question(s)…")

    for question_id, question_data in queries.items():
        question_text: str = question_data.get("question", "")
        correct_sql: Optional[str] = question_data.get("correct_sql")
        expected_answer: Optional[str] = question_data.get("expected_answer")

        logger.info(f"  [{question_id}] {question_text[:100]}…")

        # Execute question via agent
        had_error = False
        try:
            llm_result = agent_executor({"input": question_text})
        except Exception as exc:
            llm_result = {"sql": None, "rows": [], "error": str(exc)}
            had_error = True

        if had_error or llm_result.get("error"):
            error_count += 1

        generated_sql: str = llm_result.get("sql") or ""
        generated_rows: List[Dict[str, Any]] = llm_result.get("rows") or []

        # Collect token usage
        usage_metadata = llm_result.get("usage_metadata") or {}
        question_tokens = sum(
            v.get("total_tokens", v.get("approximate", 0))
            for v in usage_metadata.values()
            if isinstance(v, dict)
        )
        total_tokens_used += question_tokens

        # Write agent outputs back to results
        benchmark_results[question_id]["generated_sql"] = generated_sql
        benchmark_results[question_id]["selected_concept"] = llm_result.get("concept") or ""
        benchmark_results[question_id]["answer"] = llm_result.get("answer") or ""
        benchmark_results[question_id]["timbr_reasoning_status"] = llm_result.get("reasoning_status") or ""
        benchmark_results[question_id]["tokens_used"] = question_tokens

        # Execute correct SQL to obtain expected rows (deterministic mode)
        expected_rows: Optional[List[Dict[str, Any]]] = None
        if correct_sql and resolved_use_deterministic:
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
                expected_rows = []

        # Score
        score_result = scorer.score_result(
            question=question_text,
            generated_sql=generated_sql,
            answer=llm_result.get("answer") or "",
            generated_rows=generated_rows,
            expected_sql=correct_sql,
            expected_answer=expected_answer,
            expected_rows=expected_rows,
            execution_error=llm_result.get("error"),
        )

        result_status: str = score_result["assessment"]
        benchmark_results[question_id]["status"] = result_status
        benchmark_results[question_id]["score_breakdown"] = score_result["breakdown"]
        benchmark_results[question_id]["scoring_method"] = score_result["scoring_method"]
        if "reasoning" in score_result:
            benchmark_results[question_id]["score_reasoning"] = score_result["reasoning"]

        if result_status == "correct":
            correct_count += 1
        elif result_status == "partial":
            partial_count += 1
        elif result_status == "incorrect":
            incorrect_count += 1
        else:
            inconsistent_count += 1

        completed_count += 1
        _log_benchmark_update_completed(
            url=server_url,
            token=resolved_token,
            run_id=run_id,
            completed=completed_count,
            agent_name=agent_name,
        )

        logger.info(f"  [{question_id}] → {result_status.upper()}")

    # ------------------------------------------------------------------
    # Finalise run and log history
    # ------------------------------------------------------------------
    end_time = datetime.now()
    duration_ms = int((end_time - start_time).total_seconds() * 1000)
    correct_rate = round((correct_count / total_questions) * 100, 2) if total_questions > 0 else 0.0

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
            "correct_count": correct_count,
            "partial_count": partial_count,
            "incorrect_count": incorrect_count,
            "inconsistent_count": inconsistent_count,
            "error_count": error_count,
            "correct_rate": correct_rate,
            "number_of_iterations": number_of_iterations,
            "total_tokens_used": total_tokens_used,
            "langchain_timbr_version": _langchain_timbr_version,
            "llm_type": llm_type,
            "llm_model": llm_model,
            "result": benchmark_results,
        },
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    benchmark_results["_summary"] = {
        "total_questions": total_questions,
        "correct_count": correct_count,
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
            "use_deterministic_scoring": resolved_use_deterministic,
            "use_llm_judge_scoring": resolved_use_llm_judge,
            "number_of_iterations": number_of_iterations,
        },
    }

    return benchmark_results
