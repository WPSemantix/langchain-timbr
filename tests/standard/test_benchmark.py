"""Unit tests for the run_benchmark utility and BenchmarkScorer."""
import json
import pytest
from unittest.mock import MagicMock, patch

from langchain_timbr.utils.benchmark import (
    BenchmarkScorer,
    _compare_results,
    _normalize_results,
    _normalize_value,
    run_benchmark,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_URL = "http://test.timbr.ai"
MOCK_TOKEN = "tk_test_token"
MOCK_ONTOLOGY = "test_ontology"
MOCK_AGENT = "test_agent"
MOCK_BENCHMARK_NAME = "test_benchmark"

SAMPLE_QUESTIONS = {
    "Q1": {
        "question": "How many active policies are there?",
        "correct_sql": "SELECT COUNT(*) FROM Policy WHERE status = 'active'",
        "expected_answer": "42",
    },
    "Q2": {
        "question": "List all customers.",
    },
}

SAMPLE_AGENT_OPTIONS = {
    "ontology": MOCK_ONTOLOGY,
    "graph_depth": "3",
    "enable_reasoning": "true",
    "include_tags": "*",
}

SAMPLE_BENCHMARK_INFO = {
    "benchmark_name": MOCK_BENCHMARK_NAME,
    "agent_name": MOCK_AGENT,
    "description": "Test benchmark",
    "number_of_questions": 2,
    "benchmark": None,  # no default questions; individual tests override as needed
    "updated_at": "2025-01-01 00:00:00",
    "changed_by_user": "admin",
}


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_normalize_value_none(self):
        assert _normalize_value(None) is None

    def test_normalize_value_numeric(self):
        assert _normalize_value(42) == 42.0
        assert _normalize_value(3.14) == 3.14

    def test_normalize_value_string(self):
        assert _normalize_value("  Hello  ") == "hello"

    def test_normalize_results_empty(self):
        assert _normalize_results([]) == []

    def test_normalize_results_sorted(self):
        rows = [{"b": 2, "a": 1}, {"b": 4, "a": 3}]
        result = _normalize_results(rows)
        # Both rows must be present after normalization
        assert len(result) == 2


class TestCompareResults:
    def test_exact_match(self):
        rows = [{"name": "Alice"}, {"name": "Bob"}]
        ok, msg = _compare_results(rows, rows)
        assert ok is True
        assert msg == ""

    def test_row_count_mismatch(self):
        ok, msg = _compare_results([{"a": 1}], [{"a": 1}, {"a": 2}])
        assert ok is False
        assert "Row count mismatch" in msg

    def test_value_mismatch(self):
        ok, msg = _compare_results([{"a": 1}], [{"a": 2}])
        assert ok is False

    def test_column_names_ignored(self):
        """Values must match regardless of column name."""
        llm_rows = [{"col_x": "Alice"}]
        exp_rows = [{"full_name": "Alice"}]
        ok, _ = _compare_results(llm_rows, exp_rows)
        assert ok is True

    def test_both_empty(self):
        ok, msg = _compare_results([], [])
        assert ok is True


# ---------------------------------------------------------------------------
# BenchmarkScorer
# ---------------------------------------------------------------------------

class TestBenchmarkScorer:
    def _make_scorer(self, use_deterministic=False, use_llm_judge=False, llm=None):
        conn_params = {"url": MOCK_URL, "token": MOCK_TOKEN, "ontology": MOCK_ONTOLOGY}
        return BenchmarkScorer(
            conn_params=conn_params,
            llm=llm,
            use_deterministic=use_deterministic,
            use_llm_judge=use_llm_judge,
        )

    # -- No-scoring mode -----------------------------------------------------

    def test_basic_no_sql(self):
        scorer = self._make_scorer()
        result = scorer.score_result(question="q", generated_sql="", answer="")
        assert result["assessment"] == "incorrect"
        assert result["scoring_method"] == "error"

    def test_no_scoring_methods_with_sql_no_answer(self):
        scorer = self._make_scorer()
        result = scorer.score_result(question="q", generated_sql="SELECT 1", answer="")
        assert result["assessment"] == "incorrect"
        assert result["scoring_method"] == "error"

    def test_no_scoring_methods_with_execution_error(self):
        scorer = self._make_scorer()
        result = scorer.score_result(
            question="q",
            generated_sql="SELECT 1",
            answer="some answer",
            execution_error="Table not found",
        )
        assert result["assessment"] == "incorrect"
        assert result["scoring_method"] == "error"

    def test_no_scoring_methods_with_valid_sql_and_answer(self):
        scorer = self._make_scorer()
        result = scorer.score_result(
            question="q",
            generated_sql="SELECT * FROM Policy",
            answer="There are 5 policies",
        )
        assert result["assessment"] == "incorrect"
        assert result["scoring_method"] == "error"

    # -- Deterministic mode --------------------------------------------------

    def test_deterministic_match(self):
        scorer = self._make_scorer(use_deterministic=True)
        rows = [{"count": 42}]
        result = scorer.score_result(
            question="q",
            generated_sql="SELECT COUNT(*) FROM Policy",
            answer="42",
            generated_rows=rows,
            expected_rows=rows,
        )
        assert result["assessment"] == "correct"
        assert result["scoring_method"] == "deterministic"
        assert result["breakdown"]["det_result_accuracy"] == "matched"

    def test_deterministic_mismatch(self):
        scorer = self._make_scorer(use_deterministic=True)
        result = scorer.score_result(
            question="q",
            generated_sql="SELECT COUNT(*) FROM Policy",
            answer="10",
            generated_rows=[{"count": 10}],
            expected_rows=[{"count": 42}],
        )
        assert result["assessment"] == "incorrect"
        assert result["breakdown"]["det_result_accuracy"] == "mismatched"

    def test_deterministic_execution_error(self):
        scorer = self._make_scorer(use_deterministic=True)
        result = scorer.score_result(
            question="q",
            generated_sql="SELECT COUNT(*) FROM Policy",
            answer="",
            execution_error="Syntax error",
        )
        assert result["assessment"] == "incorrect"
        assert result["breakdown"]["det_execution_success"] == "failed"

    # -- LLM judge mode ------------------------------------------------------

    def test_llm_judge_correct(self):
        mock_llm = MagicMock()
        mock_llm.return_value.content = json.dumps({
            "assessment": "correct",
            "reasoning": "The answer fully addresses the question.",
        })

        scorer = self._make_scorer(use_llm_judge=True, llm=mock_llm)

        with patch(
            "langchain_timbr.utils.benchmark.get_benchmark_judge_prompt_template"
        ) as mock_template_getter:
            mock_template = MagicMock()
            mock_template.format_messages.return_value = [MagicMock(), MagicMock()]
            mock_template_getter.return_value = mock_template

            result = scorer.score_result(
                question="How many policies?",
                generated_sql="SELECT COUNT(*) FROM Policy",
                answer="There are 42 policies.",
            )

        assert result["assessment"] == "correct"
        assert result["scoring_method"] == "llm_judge"
        assert "reasoning" in result

    def test_llm_judge_fallback_on_error(self):
        mock_llm = MagicMock()
        mock_llm.side_effect = Exception("LLM unavailable")

        scorer = self._make_scorer(use_llm_judge=True, llm=mock_llm)

        with patch(
            "langchain_timbr.utils.benchmark.get_benchmark_judge_prompt_template"
        ) as mock_template_getter:
            mock_template = MagicMock()
            mock_template.format_messages.return_value = []
            mock_template_getter.return_value = mock_template

            result = scorer.score_result(
                question="q",
                generated_sql="SELECT * FROM Policy",
                answer="some answer",
            )

        assert result["scoring_method"] == "error"

    # -- Full mode ------------------------------------------------------------

    def test_full_deterministic_wins(self):
        """When both modes are enabled, deterministic assessment takes priority."""
        mock_llm = MagicMock()
        mock_llm.return_value.content = json.dumps({
            "assessment": "incorrect",
            "reasoning": "LLM says no.",
        })

        scorer = self._make_scorer(use_deterministic=True, use_llm_judge=True, llm=mock_llm)

        with patch(
            "langchain_timbr.utils.benchmark.get_benchmark_judge_prompt_template"
        ) as mock_template_getter:
            mock_template = MagicMock()
            mock_template.format_messages.return_value = []
            mock_template_getter.return_value = mock_template

            rows = [{"count": 42}]
            result = scorer.score_result(
                question="q",
                generated_sql="SELECT COUNT(*) FROM Policy",
                answer="42",
                generated_rows=rows,
                expected_rows=rows,
            )

        assert result["assessment"] == "correct"  # deterministic wins
        assert result["scoring_method"] == "full"


# ---------------------------------------------------------------------------
# run_benchmark
# ---------------------------------------------------------------------------

class TestRunBenchmark:
    """Tests for the top-level run_benchmark function."""

    def setup_method(self, method):
        """Set up patches for benchmark logging and benchmark info for every test."""
        from unittest.mock import patch as _patch
        self._patches = [
            _patch("langchain_timbr.utils.benchmark.get_timbr_benchmark_info"),
            _patch("langchain_timbr.utils.benchmark._log_benchmark_running"),
            _patch("langchain_timbr.utils.benchmark._log_benchmark_update_completed"),
            _patch("langchain_timbr.utils.benchmark._log_benchmark_history"),
        ]
        self.mock_get_bm_info = self._patches[0].start()
        self.mock_get_bm_info.return_value = SAMPLE_BENCHMARK_INFO.copy()
        self.mock_log_running = self._patches[1].start()
        self.mock_log_update = self._patches[2].start()
        self.mock_log_history = self._patches[3].start()

    def teardown_method(self, method):
        for p in self._patches:
            p.stop()

    def _make_agent_result(self, sql="SELECT * FROM Policy", concept="Policy"):
        return {
            "sql": sql,
            "rows": [{"count": 42}],
            "answer": "There are 42 policies.",
            "concept": concept,
            "ontology": "Insurance",
            "reasoning_status": "correct",
            "identify_concept_reason": "Question maps best to Policy concept.",
            "generate_sql_reason": "Using aggregate count pattern.",
            "error": None,
            "usage_metadata": {
                "identify_concept": {"total_tokens": 100},
                "generate_sql": {"total_tokens": 200},
            },
        }

    @patch("langchain_timbr.utils.benchmark.create_timbr_sql_agent")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_basic_run(self, mock_get_options, mock_create_agent):
        mock_get_options.return_value = SAMPLE_AGENT_OPTIONS.copy()

        mock_agent_executor = MagicMock()
        mock_agent_executor.return_value = self._make_agent_result()
        mock_create_agent.return_value = mock_agent_executor

        results = run_benchmark(
            benchmark_name=MOCK_BENCHMARK_NAME,
            queries=SAMPLE_QUESTIONS,
            url=MOCK_URL,
            token=MOCK_TOKEN,
            use_llm_judge=False,
        )

        assert "_summary" in results
        assert results["_summary"]["total_questions"] == 2
        assert (
            results["_summary"]["correct_count"]
            + results["_summary"]["partial_count"]
            + results["_summary"]["incorrect_count"]
            == 2
        )
        assert "Q1" in results
        assert "Q2" in results
        assert "generated_sql" in results["Q1"]
        assert "selected_concept" in results["Q1"]
        assert "selected_ontology" in results["Q1"]
        assert "identify_concept_reason" in results["Q1"]
        assert "generate_sql_reason" in results["Q1"]
        assert "correct_concept" in results["Q1"]
        assert "correct_ontology" in results["Q1"]
        assert "status" in results["Q1"]

        # Logging helpers should have been called
        self.mock_log_running.assert_called_once()
        assert self.mock_log_update.call_count == 2  # once per question
        self.mock_log_history.assert_called_once()

    @patch("langchain_timbr.utils.benchmark.create_timbr_sql_agent")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_queries_from_benchmark_info(self, mock_get_options, mock_create_agent):
        """When no queries are passed, read from the benchmark's 'benchmark' field."""
        bm_info = SAMPLE_BENCHMARK_INFO.copy()
        bm_info["benchmark"] = json.dumps({
            "Q1": {"question": "How many policies?"},
        })
        self.mock_get_bm_info.return_value = bm_info
        mock_get_options.return_value = SAMPLE_AGENT_OPTIONS.copy()

        mock_agent_executor = MagicMock()
        mock_agent_executor.return_value = self._make_agent_result()
        mock_create_agent.return_value = mock_agent_executor

        results = run_benchmark(
            benchmark_name=MOCK_BENCHMARK_NAME,
            url=MOCK_URL,
            token=MOCK_TOKEN,
            use_llm_judge=False,
        )

        assert results["_summary"]["total_questions"] == 1
        assert "Q1" in results

    @patch("langchain_timbr.utils.benchmark.create_timbr_sql_agent")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_scoring_mode_from_agent_options(self, mock_get_options, mock_create_agent):
        """Scoring flags from agent options are respected when no explicit params given."""
        options = SAMPLE_AGENT_OPTIONS.copy()
        options["use_llm_judge_scoring"] = "false"
        options["use_deterministic_scoring"] = "false"
        mock_get_options.return_value = options

        mock_agent_executor = MagicMock()
        mock_agent_executor.return_value = self._make_agent_result()
        mock_create_agent.return_value = mock_agent_executor

        results = run_benchmark(
            benchmark_name=MOCK_BENCHMARK_NAME,
            queries={"Q1": {"question": "test?"}},
            url=MOCK_URL,
            token=MOCK_TOKEN,
        )
        assert results["Q1"]["scoring_method"] == "error"

    @patch("langchain_timbr.utils.benchmark.create_timbr_sql_agent")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_explicit_params_override_agent_options(self, mock_get_options, mock_create_agent):
        """Explicit use_deterministic param overrides any agent option value."""
        options = SAMPLE_AGENT_OPTIONS.copy()
        options["use_deterministic_scoring"] = "false"
        mock_get_options.return_value = options

        mock_agent_executor = MagicMock()
        agent_result = self._make_agent_result()
        agent_result["rows"] = [{"count": 42}]
        mock_agent_executor.return_value = agent_result
        mock_create_agent.return_value = mock_agent_executor

        results = run_benchmark(
            benchmark_name=MOCK_BENCHMARK_NAME,
            queries={"Q1": {"question": "How many?", "correct_sql": "SELECT COUNT(*) FROM P"}},
            url=MOCK_URL,
            token=MOCK_TOKEN,
            use_deterministic=True,  # explicit override
            use_llm_judge=False,
        )
        # Deterministic mode was forced, scoring_method should be deterministic
        assert results["Q1"]["scoring_method"] == "deterministic"

    def test_missing_benchmark_name_raises(self):
        with pytest.raises(ValueError, match="mandatory"):
            run_benchmark(benchmark_name="", url=MOCK_URL, token=MOCK_TOKEN)

    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_no_queries_and_no_benchmark_field_raises(self, mock_get_options):
        """When no queries are passed and the benchmark has no 'benchmark' field, raise."""
        mock_get_options.return_value = SAMPLE_AGENT_OPTIONS.copy()
        # SAMPLE_BENCHMARK_INFO already has benchmark=None
        with pytest.raises(ValueError, match="benchmark"):
            run_benchmark(
                benchmark_name=MOCK_BENCHMARK_NAME,
                url=MOCK_URL,
                token=MOCK_TOKEN,
                use_llm_judge=False,
            )

    @patch("langchain_timbr.utils.benchmark.create_timbr_sql_agent")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_summary_structure(self, mock_get_options, mock_create_agent):
        mock_get_options.return_value = SAMPLE_AGENT_OPTIONS.copy()

        mock_agent_executor = MagicMock()
        mock_agent_executor.return_value = self._make_agent_result()
        mock_create_agent.return_value = mock_agent_executor

        results = run_benchmark(
            benchmark_name=MOCK_BENCHMARK_NAME,
            queries=SAMPLE_QUESTIONS,
            url=MOCK_URL,
            token=MOCK_TOKEN,
            use_llm_judge=False,
        )

        summary = results["_summary"]
        assert "total_questions" in summary
        assert "correct_count" in summary
        assert "partial_count" in summary
        assert "incorrect_count" in summary
        assert "inconsistent_count" in summary
        assert "error_count" in summary
        assert "correct_rate" in summary
        assert "total_tokens_used" in summary
        assert "duration_ms" in summary
        assert "timestamp" in summary
        assert "config" in summary
        assert summary["config"]["benchmark_name"] == MOCK_BENCHMARK_NAME
        assert summary["config"]["agent_name"] == MOCK_AGENT

    @patch("langchain_timbr.utils.benchmark.create_timbr_sql_agent")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_error_count_tracking(self, mock_get_options, mock_create_agent):
        """Questions where the agent throws an exception increment error_count."""
        mock_get_options.return_value = SAMPLE_AGENT_OPTIONS.copy()

        mock_agent_executor = MagicMock()
        mock_agent_executor.side_effect = Exception("Agent failure")
        mock_create_agent.return_value = mock_agent_executor

        results = run_benchmark(
            benchmark_name=MOCK_BENCHMARK_NAME,
            queries={"Q1": {"question": "test?"}},
            url=MOCK_URL,
            token=MOCK_TOKEN,
            use_llm_judge=False,
        )
        assert results["_summary"]["error_count"] == 1

    @patch("langchain_timbr.utils.benchmark.create_timbr_sql_agent")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_total_tokens_summed(self, mock_get_options, mock_create_agent):
        """total_tokens_used in summary is the sum of all per-question tokens."""
        mock_get_options.return_value = SAMPLE_AGENT_OPTIONS.copy()

        mock_agent_executor = MagicMock()
        mock_agent_executor.return_value = self._make_agent_result()  # 100 + 200 = 300 tokens
        mock_create_agent.return_value = mock_agent_executor

        results = run_benchmark(
            benchmark_name=MOCK_BENCHMARK_NAME,
            queries=SAMPLE_QUESTIONS,  # 2 questions
            url=MOCK_URL,
            token=MOCK_TOKEN,
            use_llm_judge=False,
        )
        assert results["_summary"]["total_tokens_used"] == 600  # 300 × 2

    @patch("langchain_timbr.utils.benchmark.create_timbr_sql_agent")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_log_history_payload_fields(self, mock_get_options, mock_create_agent):
        """The history logging call receives all required fields."""
        mock_get_options.return_value = SAMPLE_AGENT_OPTIONS.copy()

        mock_agent_executor = MagicMock()
        mock_agent_executor.return_value = self._make_agent_result()
        mock_create_agent.return_value = mock_agent_executor

        run_benchmark(
            benchmark_name=MOCK_BENCHMARK_NAME,
            queries=SAMPLE_QUESTIONS,
            url=MOCK_URL,
            token=MOCK_TOKEN,
            number_of_iterations=1,
            use_llm_judge=False,
        )

        _, kwargs = self.mock_log_history.call_args
        payload = kwargs["payload"]
        for field in (
            "benchmark_name", "agent_name", "run_id", "start_time", "end_time",
            "duration", "number_of_questions", "correct_count", "partial_count",
            "incorrect_count", "inconsistent_count", "error_count", "correct_rate",
            "number_of_iterations", "total_tokens_used", "langchain_timbr_version",
            "llm_type", "llm_model", "result",
        ):
            assert field in payload, f"Missing field in history payload: {field}"
        assert payload["benchmark_name"] == MOCK_BENCHMARK_NAME
        assert payload["number_of_iterations"] == 1

    @patch("langchain_timbr.utils.benchmark.create_timbr_sql_agent")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_concept_ontology_checks_normalized(self, mock_get_options, mock_create_agent):
        """Concept/ontology checks use normalized comparison and null when missing expected values."""
        mock_get_options.return_value = SAMPLE_AGENT_OPTIONS.copy()

        mock_agent_executor = MagicMock()
        mock_agent_executor.side_effect = [
            self._make_agent_result(concept=" policy "),
            self._make_agent_result(concept="Customer"),
        ]
        mock_create_agent.return_value = mock_agent_executor

        queries = {
            "Q1": {
                "question": "How many active policies are there?",
                "correct_concept": "POLICY",
                "correct_ontology": "insurance",
            },
            "Q2": {
                "question": "List all customers.",
            },
        }

        results = run_benchmark(
            benchmark_name=MOCK_BENCHMARK_NAME,
            queries=queries,
            url=MOCK_URL,
            token=MOCK_TOKEN,
            use_llm_judge=False,
        )

        assert results["Q1"]["correct_concept"] is True
        assert results["Q1"]["correct_ontology"] is True
        assert results["Q2"]["correct_concept"] is None
        assert results["Q2"]["correct_ontology"] is None

    @patch("langchain_timbr.utils.benchmark.LlmWrapper")
    @patch("langchain_timbr.utils.benchmark.create_timbr_sql_agent")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_llm_judge_is_default_when_not_configured(self, mock_get_options, mock_create_agent, mock_llm_wrapper):
        """When neither param nor option is provided, llm_judge is enabled by default."""
        options = SAMPLE_AGENT_OPTIONS.copy()
        options.pop("use_llm_judge_scoring", None)
        options.pop("use_deterministic_scoring", None)
        mock_get_options.return_value = options

        mock_agent_executor = MagicMock()
        mock_agent_executor.return_value = self._make_agent_result()
        mock_create_agent.return_value = mock_agent_executor

        mock_llm = MagicMock()
        mock_llm.return_value.content = '{"assessment":"correct","reasoning":"ok"}'
        mock_llm_wrapper.return_value = mock_llm

        with patch("langchain_timbr.utils.benchmark.get_benchmark_judge_prompt_template") as mock_template_getter:
            mock_template = MagicMock()
            mock_template.format_messages.return_value = []
            mock_template_getter.return_value = mock_template

            results = run_benchmark(
                benchmark_name=MOCK_BENCHMARK_NAME,
                queries={"Q1": {"question": "How many?"}},
                url=MOCK_URL,
                token=MOCK_TOKEN,
            )

        assert results["Q1"]["scoring_method"] == "llm_judge"
