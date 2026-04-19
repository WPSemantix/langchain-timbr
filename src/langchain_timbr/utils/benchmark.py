"""
LLM Benchmark utility for evaluating Timbr SQL agent performance.

Runs a set of questions through a Timbr agent and scores the generated SQL and
answers using deterministic row-comparison, an LLM-as-judge approach, or the
combined full mode.

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
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

from pytimbr_api import timbr_http_connector

from .. import config
from langchain_timbr import create_timbr_sql_agent, GenerateTimbrSqlChain
from ..llm_wrapper.llm_wrapper import LlmWrapper
from .general import to_boolean
from .prompt_service import get_benchmark_judge_prompt_template
from .timbr_utils import get_timbr_agent_options, get_timbr_benchmark_info, build_server_url

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


def _matches_expected_value(expected_value: Any, selected_value: Any) -> Optional[bool]:
    """Compare expected and selected values using normalization.

    Returns:
        ``None`` when expected value is missing, otherwise ``True``/``False``.
    """
    if expected_value is None:
        return None
    return _normalize_value(expected_value) == _normalize_value(selected_value)


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
# SQL normalization helpers
# ---------------------------------------------------------------------------

# Similarity threshold for SQL-to-SQL partial match (generate_sql_only mode)
SQL_PARTIAL_MATCH_THRESHOLD = 0.85


def _normalize_sql(sql: str) -> str:
    """Normalise SQL for comparison: lowercase, collapse all whitespace to single
    spaces, and strip trailing semicolons."""
    if not sql:
        return ""
    return " ".join(sql.lower().split()).rstrip(";").strip()


# ---------------------------------------------------------------------------
# BenchmarkScorer
# ---------------------------------------------------------------------------

class BenchmarkScorer:
    """
    Score benchmark results using a configurable combination of:

    * **deterministic** – row comparison between LLM and expected results
    * **llm_judge** – LLM evaluates the SQL + answer quality
    * **full** – deterministic and llm_judge together (deterministic assessment wins)

    When both *deterministic* and *llm_judge* are enabled the deterministic
    assessment takes priority and the scoring method is reported as ``"full"``.
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
        execution_mode: str = "full",
    ) -> Dict[str, Any]:
        """
        Score a single benchmark result.

        Returns a dict with keys:
            ``assessment``    – ``"correct"``, ``"partial"``, or ``"incorrect"``
            ``breakdown``     – method-specific detail (populated by deterministic mode)
            ``scoring_method``– one of ``"deterministic"``, ``"llm_judge"``,
                                ``"full"``, ``"error"``
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
            return {
                "assessment": "incorrect",
                "breakdown": {},
                "reasoning": "No scoring method enabled",
                "scoring_method": "error",
            }

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
                execution_mode=execution_mode,
            )
            all_assessments.append(det_result)
            combined_breakdown.update(
                {f"det_{k}": v for k, v in det_result["breakdown"].items()}
            )
            if det_result.get("reasoning"):
                combined_reasoning.append(f"[Deterministic] {det_result['reasoning']}")

        if "llm_judge" in methods_to_use:
            llm_result = self._llm_judge_score(question, generated_sql, answer, execution_mode=execution_mode)
            all_assessments.append(llm_result)
            combined_breakdown.update(
                {f"llm_{k}": v for k, v in llm_result["breakdown"].items()}
            )
            if llm_result.get("reasoning"):
                combined_reasoning.append(f"[LLM Judge] {llm_result['reasoning']}")

        if len(all_assessments) > 1:
            # Filter out skipped/error assessments — they carry no signal
            meaningful = [a for a in all_assessments if a["scoring_method"] not in ("skipped", "error")]
            if not meaningful:
                # Nothing produced a real result; fall back to incorrect
                final_assessment = "incorrect"
                scoring_method = "error"
            elif len(meaningful) == 1:
                final_assessment = meaningful[0]["assessment"]
                scoring_method = meaningful[0]["scoring_method"]
            else:
                # Deterministic takes priority when it produced a meaningful result
                det_results = [a for a in meaningful if a["scoring_method"] == "deterministic"]
                if det_results:
                    final_assessment = det_results[0]["assessment"]
                else:
                    assessments = [a["assessment"] for a in meaningful]
                    if "incorrect" in assessments:
                        final_assessment = "incorrect"
                    elif "partial" in assessments:
                        final_assessment = "partial"
                    else:
                        final_assessment = "correct"
                scoring_method = "full"
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
        execution_mode: str = "full",
    ) -> Dict[str, Any]:
        breakdown: Dict[str, Any] = {}
        reasoning_parts: List[str] = []

        # ---- SQL-only mode: compare generated SQL against expected SQL directly ----
        if execution_mode == "generate_sql_only":
            if not expected_sql:
                return {
                    "assessment": "skipped",
                    "breakdown": {},
                    "reasoning": "No expected SQL provided for SQL comparison",
                    "scoring_method": "skipped",
                }
            norm_generated = _normalize_sql(generated_sql)
            norm_expected = _normalize_sql(expected_sql)
            ratio = round(SequenceMatcher(None, norm_expected, norm_generated).ratio(), 4)
            breakdown["sql_similarity"] = ratio
            if ratio == 1.0:
                breakdown["sql_match"] = "exact"
                assessment = "correct"
                reasoning_parts.append("SQL matches expected exactly")
            elif ratio >= SQL_PARTIAL_MATCH_THRESHOLD:
                breakdown["sql_match"] = "partial"
                assessment = "partial"
                reasoning_parts.append(f"SQL partially matches expected (similarity: {ratio:.2f})")
            else:
                breakdown["sql_match"] = "none"
                assessment = "incorrect"
                reasoning_parts.append(f"SQL does not match expected (similarity: {ratio:.2f})")
            return {
                "assessment": assessment,
                "breakdown": breakdown,
                "scoring_method": "deterministic",
                "reasoning": " | ".join(reasoning_parts),
            }

        # ---- Full mode: row-comparison scoring ----
        assessment = "correct"

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
        execution_mode: str = "full",
    ) -> Dict[str, Any]:
        """Use LLM (via the benchmark-judge template from timbr-api) to score the result."""
        try:
            if self._judge_prompt_template is None:
                self._judge_prompt_template = get_benchmark_judge_prompt_template(
                    conn_params=self.conn_params
                )

            # In SQL-only mode there is no executed answer; pass an empty context so
            # the template section is absent and the judge evaluates SQL alone.
            if execution_mode == "generate_sql_only":
                answer_context = ""
            else:
                answer_context = f"**Generated Answer:**\n{answer or '(no answer generated)'}\n\n"

            messages = self._judge_prompt_template.format_messages(
                question=question,
                generated_sql=generated_sql,
                answer_context=answer_context,
                expected_sql_context="",
                expected_answer_context="",
            )

            response = self.llm(messages)

            content = response if isinstance(response, str) else response.content.strip()
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
            logger.warning(f"LLM judge scoring failed: {exc}")
            return {
                "assessment": "incorrect",
                "breakdown": {},
                "reasoning": f"LLM judge scoring failed: {str(exc)}",
                "scoring_method": "error",
            }

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
            Defaults to ``True``.
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
              identification → SQL generation → execution → answer generation)
              and score using row comparison and/or LLM judge.
            * ``"generate_sql_only"`` – use GenerateTimbrSqlChain to generate
              SQL without executing it. Deterministic scoring compares the generated SQL
              against ``correct_sql`` (exact match → *correct*, similarity ≥
              :data:`SQL_PARTIAL_MATCH_THRESHOLD` → *partial*, else → *incorrect*).
              LLM-judge scoring evaluates whether the SQL looks correct for the question.

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
        * ``status`` – ``"correct"``, ``"partial"``, or ``"incorrect"``
        * ``score_breakdown`` – method-specific detail dict
        * ``scoring_method`` – ``"deterministic"``, ``"llm_judge"``, ``"full"``,
                    or ``"error"``
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

    server_url = build_server_url(thrift_host, thrift_port)
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
    # Resolve scoring modes
    # param > agent option > default (False)
    # ------------------------------------------------------------------
    def _resolve_flag(param_val: Optional[bool], option_key: str, default: bool) -> bool:
        if param_val is not None:
            return param_val
        raw = agent_options.get(option_key)
        if raw is not None:
            return to_boolean(raw)
        return default

    resolved_use_deterministic = _resolve_flag(use_deterministic, "use_deterministic_scoring", False)
    resolved_use_llm_judge = _resolve_flag(use_llm_judge, "use_llm_judge_scoring", True)

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
    # Instantiate scorer and agent executor / SQL-only chain
    # ------------------------------------------------------------------
    scorer = BenchmarkScorer(
        conn_params=agent_conn_params,
        llm=judge_llm,
        use_deterministic=resolved_use_deterministic,
        use_llm_judge=resolved_use_llm_judge,
    )

    agent_executor = None
    sql_chain = None

    if execution == "generate_sql_only":
        # Build an LLM for the chain (same config as judge LLM; create separately so
        # generate_sql_only mode works even when use_llm_judge is False)
        sql_chain_llm: Optional[Any] = None
        if llm_type and llm_api_key:
            sql_chain_llm = LlmWrapper(
                llm_type=llm_type,
                model=llm_model,
                api_key=llm_api_key,
                temperature=0,
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
            "execution": execution,
            "number_of_iterations": number_of_iterations,
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

        # Execute correct SQL once (outside iteration loop) for deterministic full-mode scoring.
        # In generate_sql_only mode we never execute SQL, so skip this.
        expected_rows: Optional[List[Dict[str, Any]]] = None
        if execution == "full" and correct_sql and resolved_use_deterministic:
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

        # ------------------------------------------------------------------
        # Iterations inner loop
        # ------------------------------------------------------------------
        iteration_results: List[Dict[str, Any]] = []
        question_tokens_total: int = 0
        had_error_in_any_iteration = False
        last_llm_result: Dict[str, Any] = {}
        last_score_result: Dict[str, Any] = {}

        for iter_num in range(1, number_of_iterations + 1):
            iter_label = f"[{question_id}][{iter_num}/{number_of_iterations}]" if number_of_iterations > 1 else f"[{question_id}]"

            # ---- Execute via agent or SQL-only chain ----
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
                        # Wrap into the nested format used by the token-counting logic
                        "usage_metadata": {
                            "generate_sql": raw.get("generate_sql_usage_metadata") or {}
                        },
                    }
                else:
                    llm_result = agent_executor.invoke({"input": question_text})  # type: ignore[misc]
            except Exception as exc:
                llm_result = {"sql": None, "rows": [], "error": str(exc), "usage_metadata": {}}

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
            )

            iter_status: str = score_result["assessment"]
            iter_entry: Dict[str, Any] = {
                "iteration": iter_num,
                "generated_sql": generated_sql,
                "status": iter_status,
                "scoring_method": score_result["scoring_method"],
                "score_breakdown": score_result["breakdown"],
                "tokens_used": iter_tokens,
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
        benchmark_results[question_id]["status"] = result_status
        benchmark_results[question_id]["score_breakdown"] = last_score_result.get("breakdown", {})
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
            "use_deterministic_scoring": resolved_use_deterministic,
            "use_llm_judge_scoring": resolved_use_llm_judge,
            "execution": execution,
            "number_of_iterations": number_of_iterations,
        },
    }

    return benchmark_results
