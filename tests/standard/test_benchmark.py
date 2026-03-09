"""Unit tests for the run_benchmark utility and BenchmarkScorer."""
import json
import pytest
from unittest.mock import MagicMock, patch

from langchain_timbr.utils.benchmark import (
    BenchmarkScorer,
    SQL_PARTIAL_MATCH_THRESHOLD,
    _compare_results,
    _normalize_results,
    _normalize_sql,
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


# ---------------------------------------------------------------------------
# _normalize_sql
# ---------------------------------------------------------------------------

class TestNormalizeSql:
    """Unit tests for the _normalize_sql helper."""

    def test_empty_string(self):
        assert _normalize_sql("") == ""

    def test_none_like_empty(self):
        # Function accepts str; verify empty-ish inputs return ""
        assert _normalize_sql("   ") == ""

    def test_lowercases(self):
        assert _normalize_sql("SELECT * FROM Policy") == "select * from policy"

    def test_collapses_whitespace(self):
        assert _normalize_sql("SELECT\n  *\n  FROM\t\tPolicy") == "select * from policy"

    def test_strips_trailing_semicolon(self):
        assert _normalize_sql("SELECT 1;") == "select 1"

    def test_strips_trailing_semicolon_with_spaces(self):
        assert _normalize_sql("SELECT 1 ;  ") == "select 1"

    def test_no_mutation_on_clean_sql(self):
        sql = "select count(*) from policy where status = 'active'"
        assert _normalize_sql(sql) == sql

    def test_identical_after_normalisation(self):
        a = "SELECT\n  COUNT(*)\nFROM\n  Policy\nWHERE status = 'active';"
        b = "select count(*) from policy where status = 'active'"
        assert _normalize_sql(a) == _normalize_sql(b)


# ---------------------------------------------------------------------------
# BenchmarkScorer – generate_sql_only mode
# ---------------------------------------------------------------------------

class TestBenchmarkScorerSqlOnlyMode:
    """Tests for _deterministic_score and _llm_judge_score with execution_mode='generate_sql_only'."""

    def _make_scorer(self, use_deterministic=False, use_llm_judge=False, llm=None):
        conn_params = {"url": MOCK_URL, "token": MOCK_TOKEN, "ontology": MOCK_ONTOLOGY}
        return BenchmarkScorer(
            conn_params=conn_params,
            llm=llm,
            use_deterministic=use_deterministic,
            use_llm_judge=use_llm_judge,
        )

    # -- Deterministic scoring via SQL comparison ----------------------------

    def test_exact_sql_match_is_correct(self):
        scorer = self._make_scorer(use_deterministic=True)
        sql = "SELECT COUNT(*) FROM Policy WHERE status = 'active'"
        result = scorer.score_result(
            question="How many active policies?",
            generated_sql=sql,
            answer="",
            expected_sql=sql,
            execution_mode="generate_sql_only",
        )
        assert result["assessment"] == "correct"
        assert result["scoring_method"] == "deterministic"
        assert result["breakdown"]["det_sql_match"] == "exact"
        assert result["breakdown"]["det_sql_similarity"] == 1.0

    def test_exact_sql_match_normalised_whitespace(self):
        """Whitespace and case differences should still be an exact match."""
        scorer = self._make_scorer(use_deterministic=True)
        expected = "SELECT COUNT(*) FROM Policy;"
        generated = "select  count(*)  from  policy"
        result = scorer.score_result(
            question="q",
            generated_sql=generated,
            answer="",
            expected_sql=expected,
            execution_mode="generate_sql_only",
        )
        assert result["assessment"] == "correct"
        assert result["breakdown"]["det_sql_match"] == "exact"

    def test_partial_sql_match(self):
        """A SQL that is close but not identical should yield 'partial'."""
        scorer = self._make_scorer(use_deterministic=True)
        expected = "SELECT COUNT(*) FROM Policy WHERE status = 'active'"
        # Very similar but missing the WHERE clause
        generated = "SELECT COUNT(*) FROM Policy WHERE status = 'inactive'"
        result = scorer.score_result(
            question="q",
            generated_sql=generated,
            answer="",
            expected_sql=expected,
            execution_mode="generate_sql_only",
        )
        # Similarity should be ≥ SQL_PARTIAL_MATCH_THRESHOLD for this pair
        assert result["assessment"] in ("partial", "correct")  # depends on ratio
        assert result["breakdown"]["det_sql_match"] in ("partial", "exact")

    def test_completely_different_sql_is_incorrect(self):
        scorer = self._make_scorer(use_deterministic=True)
        result = scorer.score_result(
            question="q",
            generated_sql="SELECT name FROM Customer LIMIT 5",
            answer="",
            expected_sql="SELECT COUNT(*) FROM Policy WHERE status = 'active'",
            execution_mode="generate_sql_only",
        )
        assert result["assessment"] == "incorrect"
        assert result["breakdown"]["det_sql_match"] == "none"
        assert result["breakdown"]["det_sql_similarity"] < SQL_PARTIAL_MATCH_THRESHOLD

    def test_missing_expected_sql_returns_error(self):
        """Deterministic SQL-only scoring without expected_sql should return error."""
        scorer = self._make_scorer(use_deterministic=True)
        result = scorer.score_result(
            question="q",
            generated_sql="SELECT 1",
            answer="",
            expected_sql=None,
            execution_mode="generate_sql_only",
        )
        assert result["assessment"] == "incorrect"
        assert result["scoring_method"] == "error"

    def test_sql_similarity_stored_as_float(self):
        scorer = self._make_scorer(use_deterministic=True)
        result = scorer.score_result(
            question="q",
            generated_sql="SELECT * FROM Policy",
            answer="",
            expected_sql="SELECT * FROM Customer",
            execution_mode="generate_sql_only",
        )
        assert isinstance(result["breakdown"]["det_sql_similarity"], float)

    # -- LLM judge scoring in SQL-only mode ----------------------------------

    def test_llm_judge_sql_only_passes_empty_answer_context(self):
        """In generate_sql_only mode the judge template receives an empty answer_context."""
        mock_llm = MagicMock()
        mock_llm.return_value.content = json.dumps({
            "assessment": "correct",
            "reasoning": "SQL looks right.",
        })
        scorer = self._make_scorer(use_llm_judge=True, llm=mock_llm)

        with patch(
            "langchain_timbr.utils.benchmark.get_benchmark_judge_prompt_template"
        ) as mock_template_getter:
            mock_template = MagicMock()
            mock_template.format_messages.return_value = [MagicMock()]
            mock_template_getter.return_value = mock_template

            result = scorer.score_result(
                question="How many active policies?",
                generated_sql="SELECT COUNT(*) FROM Policy",
                answer="",
                execution_mode="generate_sql_only",
            )

        # Verify answer_context="" was passed (not a populated string)
        _, call_kwargs = mock_template.format_messages.call_args
        assert call_kwargs.get("answer_context") == ""
        assert result["assessment"] == "correct"
        assert result["scoring_method"] == "llm_judge"

    def test_llm_judge_full_mode_passes_populated_answer_context(self):
        """In full mode the judge template receives a populated answer_context string."""
        mock_llm = MagicMock()
        mock_llm.return_value.content = json.dumps({
            "assessment": "correct",
            "reasoning": "Good answer.",
        })
        scorer = self._make_scorer(use_llm_judge=True, llm=mock_llm)

        with patch(
            "langchain_timbr.utils.benchmark.get_benchmark_judge_prompt_template"
        ) as mock_template_getter:
            mock_template = MagicMock()
            mock_template.format_messages.return_value = [MagicMock()]
            mock_template_getter.return_value = mock_template

            scorer.score_result(
                question="How many active policies?",
                generated_sql="SELECT COUNT(*) FROM Policy",
                answer="There are 42 policies.",
                execution_mode="full",
            )

        _, call_kwargs = mock_template.format_messages.call_args
        assert call_kwargs.get("answer_context") != ""
        assert "42" in call_kwargs["answer_context"]


# ---------------------------------------------------------------------------
# run_benchmark – execution parameter
# ---------------------------------------------------------------------------

class TestRunBenchmarkExecutionMode:
    """Tests for the execution='generate_sql_only' path in run_benchmark."""

    def setup_method(self, method):
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

    def _make_chain_result(self, sql="SELECT COUNT(*) FROM Policy", concept="Policy"):
        """Simulates the dict returned by GenerateTimbrSqlChain.invoke()."""
        return {
            "sql": sql,
            "schema": "dtimbr",
            "concept": concept,
            "is_sql_valid": True,
            "error": None,
            "reasoning_status": None,
            "identify_concept_reason": "Matched Policy concept.",
            "generate_sql_reason": "Count pattern applied.",
            "generate_sql_usage_metadata": {"total_tokens": 150},
        }

    @patch("langchain_timbr.utils.benchmark.GenerateTimbrSqlChain")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_generate_sql_only_uses_chain_not_agent(self, mock_get_options, mock_chain_cls):
        """When execution='generate_sql_only', GenerateTimbrSqlChain is used and create_timbr_sql_agent is NOT called."""
        mock_get_options.return_value = SAMPLE_AGENT_OPTIONS.copy()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = self._make_chain_result()
        mock_chain_cls.return_value = mock_chain

        with patch("langchain_timbr.utils.benchmark.create_timbr_sql_agent") as mock_create_agent:
            run_benchmark(
                benchmark_name=MOCK_BENCHMARK_NAME,
                queries={"Q1": {"question": "How many policies?"}},
                url=MOCK_URL,
                token=MOCK_TOKEN,
                use_llm_judge=False,
                execution="generate_sql_only",
            )
            mock_create_agent.assert_not_called()

        mock_chain_cls.assert_called_once()
        mock_chain.invoke.assert_called_once_with({"prompt": "How many policies?"})

    @patch("langchain_timbr.utils.benchmark.GenerateTimbrSqlChain")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_generate_sql_only_result_fields_populated(self, mock_get_options, mock_chain_cls):
        """generate_sql_only mode populates standard result fields (excluding rows/answer)."""
        mock_get_options.return_value = SAMPLE_AGENT_OPTIONS.copy()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = self._make_chain_result(sql="SELECT COUNT(*) FROM Policy", concept="Policy")
        mock_chain_cls.return_value = mock_chain

        results = run_benchmark(
            benchmark_name=MOCK_BENCHMARK_NAME,
            queries={"Q1": {"question": "How many policies?"}},
            url=MOCK_URL,
            token=MOCK_TOKEN,
            use_llm_judge=False,
            execution="generate_sql_only",
        )

        q = results["Q1"]
        assert q["generated_sql"] == "SELECT COUNT(*) FROM Policy"
        assert q["selected_concept"] == "Policy"
        assert q["answer"] == ""  # no execution, no answer
        assert "status" in q
        assert "scoring_method" in q

    @patch("langchain_timbr.utils.benchmark.GenerateTimbrSqlChain")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_generate_sql_only_deterministic_exact_match(self, mock_get_options, mock_chain_cls):
        """SQL-only + deterministic: exact SQL match produces status='correct'."""
        mock_get_options.return_value = SAMPLE_AGENT_OPTIONS.copy()
        expected_sql = "SELECT COUNT(*) FROM Policy WHERE status = 'active'"
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = self._make_chain_result(sql=expected_sql)
        mock_chain_cls.return_value = mock_chain

        results = run_benchmark(
            benchmark_name=MOCK_BENCHMARK_NAME,
            queries={"Q1": {"question": "How many active policies?", "correct_sql": expected_sql}},
            url=MOCK_URL,
            token=MOCK_TOKEN,
            use_deterministic=True,
            use_llm_judge=False,
            execution="generate_sql_only",
        )

        assert results["Q1"]["status"] == "correct"
        assert results["Q1"]["scoring_method"] == "deterministic"
        assert results["Q1"]["score_breakdown"]["det_sql_match"] == "exact"

    @patch("langchain_timbr.utils.benchmark.GenerateTimbrSqlChain")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_generate_sql_only_deterministic_mismatch(self, mock_get_options, mock_chain_cls):
        """SQL-only + deterministic: very different SQL produces status='incorrect'."""
        mock_get_options.return_value = SAMPLE_AGENT_OPTIONS.copy()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = self._make_chain_result(sql="SELECT name FROM Customer LIMIT 10")
        mock_chain_cls.return_value = mock_chain

        results = run_benchmark(
            benchmark_name=MOCK_BENCHMARK_NAME,
            queries={
                "Q1": {
                    "question": "How many active policies?",
                    "correct_sql": "SELECT COUNT(*) FROM Policy WHERE status = 'active'",
                }
            },
            url=MOCK_URL,
            token=MOCK_TOKEN,
            use_deterministic=True,
            use_llm_judge=False,
            execution="generate_sql_only",
        )

        assert results["Q1"]["status"] == "incorrect"
        assert results["Q1"]["score_breakdown"]["det_sql_match"] == "none"

    def test_invalid_execution_mode_raises(self):
        """Passing an unrecognised execution value raises ValueError."""
        with pytest.raises(ValueError, match="execution"):
            run_benchmark(
                benchmark_name=MOCK_BENCHMARK_NAME,
                queries={"Q1": {"question": "test?"}},
                url=MOCK_URL,
                token=MOCK_TOKEN,
                execution="unknown_mode",
            )

    @patch("langchain_timbr.utils.benchmark.GenerateTimbrSqlChain")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_execution_and_iterations_in_log_running_payload(self, mock_get_options, mock_chain_cls):
        """execution and number_of_iterations are sent in the log_running payload."""
        mock_get_options.return_value = SAMPLE_AGENT_OPTIONS.copy()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = self._make_chain_result()
        mock_chain_cls.return_value = mock_chain

        run_benchmark(
            benchmark_name=MOCK_BENCHMARK_NAME,
            queries={"Q1": {"question": "test?"}},
            url=MOCK_URL,
            token=MOCK_TOKEN,
            use_llm_judge=False,
            execution="generate_sql_only",
            number_of_iterations=2,
        )

        _, kwargs = self.mock_log_running.call_args
        payload = kwargs["payload"]
        assert payload["execution"] == "generate_sql_only"
        assert payload["number_of_iterations"] == 2

    @patch("langchain_timbr.utils.benchmark.GenerateTimbrSqlChain")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_execution_in_summary_config(self, mock_get_options, mock_chain_cls):
        """Summary config block contains the execution mode."""
        mock_get_options.return_value = SAMPLE_AGENT_OPTIONS.copy()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = self._make_chain_result()
        mock_chain_cls.return_value = mock_chain

        results = run_benchmark(
            benchmark_name=MOCK_BENCHMARK_NAME,
            queries={"Q1": {"question": "test?"}},
            url=MOCK_URL,
            token=MOCK_TOKEN,
            use_llm_judge=False,
            execution="generate_sql_only",
        )

        assert results["_summary"]["config"]["execution"] == "generate_sql_only"

    @patch("langchain_timbr.utils.benchmark.GenerateTimbrSqlChain")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_execution_in_log_history_payload(self, mock_get_options, mock_chain_cls):
        """execution is included in the history log payload."""
        mock_get_options.return_value = SAMPLE_AGENT_OPTIONS.copy()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = self._make_chain_result()
        mock_chain_cls.return_value = mock_chain

        run_benchmark(
            benchmark_name=MOCK_BENCHMARK_NAME,
            queries={"Q1": {"question": "test?"}},
            url=MOCK_URL,
            token=MOCK_TOKEN,
            use_llm_judge=False,
            execution="generate_sql_only",
        )

        _, kwargs = self.mock_log_history.call_args
        assert kwargs["payload"]["execution"] == "generate_sql_only"

    @patch("langchain_timbr.utils.benchmark.create_timbr_sql_agent")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_full_execution_does_not_create_sql_chain(self, mock_get_options, mock_create_agent):
        """Default execution='full' creates agent_executor and does NOT instantiate GenerateTimbrSqlChain."""
        mock_get_options.return_value = SAMPLE_AGENT_OPTIONS.copy()
        mock_agent = MagicMock()
        mock_agent.return_value = {
            "sql": "SELECT 1", "rows": [], "answer": "", "concept": "Policy",
            "ontology": MOCK_ONTOLOGY, "error": None, "usage_metadata": {},
            "reasoning_status": None, "identify_concept_reason": None, "generate_sql_reason": None,
        }
        mock_create_agent.return_value = mock_agent

        with patch("langchain_timbr.utils.benchmark.GenerateTimbrSqlChain") as mock_chain_cls:
            run_benchmark(
                benchmark_name=MOCK_BENCHMARK_NAME,
                queries={"Q1": {"question": "test?"}},
                url=MOCK_URL,
                token=MOCK_TOKEN,
                use_llm_judge=False,
                execution="full",
            )
            mock_chain_cls.assert_not_called()


# ---------------------------------------------------------------------------
# run_benchmark – iterations
# ---------------------------------------------------------------------------

class TestRunBenchmarkIterations:
    """Tests for number_of_iterations behaviour in run_benchmark."""

    def setup_method(self, method):
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

    def _make_agent_result(self, sql="SELECT * FROM Policy"):
        return {
            "sql": sql,
            "rows": [{"count": 1}],
            "answer": "One policy.",
            "concept": "Policy",
            "ontology": MOCK_ONTOLOGY,
            "error": None,
            "usage_metadata": {"generate_sql": {"total_tokens": 100}},
            "reasoning_status": None,
            "identify_concept_reason": None,
            "generate_sql_reason": None,
        }

    @patch("langchain_timbr.utils.benchmark.create_timbr_sql_agent")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_single_iteration_no_iterations_detail(self, mock_get_options, mock_create_agent):
        """With number_of_iterations=1 (default), no iterations_detail or consistent field is added."""
        mock_get_options.return_value = SAMPLE_AGENT_OPTIONS.copy()
        mock_agent = MagicMock()
        mock_agent.return_value = self._make_agent_result()
        mock_create_agent.return_value = mock_agent

        results = run_benchmark(
            benchmark_name=MOCK_BENCHMARK_NAME,
            queries={"Q1": {"question": "test?"}},
            url=MOCK_URL,
            token=MOCK_TOKEN,
            use_llm_judge=False,
            number_of_iterations=1,
        )

        assert "iterations_detail" not in results["Q1"]
        assert "consistent" not in results["Q1"]

    @patch("langchain_timbr.utils.benchmark.create_timbr_sql_agent")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_multiple_iterations_consistent(self, mock_get_options, mock_create_agent):
        """When all iterations produce the same status, consistent=True and status is set normally."""
        mock_get_options.return_value = SAMPLE_AGENT_OPTIONS.copy()
        mock_agent = MagicMock()
        # All 3 iterations return the same SQL → all will score identically
        mock_agent.return_value = self._make_agent_result()
        mock_create_agent.return_value = mock_agent

        results = run_benchmark(
            benchmark_name=MOCK_BENCHMARK_NAME,
            queries={"Q1": {"question": "test?"}},
            url=MOCK_URL,
            token=MOCK_TOKEN,
            use_llm_judge=False,
            number_of_iterations=3,
        )

        assert results["Q1"]["consistent"] is True
        assert results["Q1"]["status"] != "inconsistent"
        assert len(results["Q1"]["iterations_detail"]) == 3

    @patch("langchain_timbr.utils.benchmark.timbr_http_connector")
    @patch("langchain_timbr.utils.benchmark.create_timbr_sql_agent")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_multiple_iterations_inconsistent(self, mock_get_options, mock_create_agent, mock_connector):
        """When iterations produce different statuses, status='inconsistent' and inconsistent_count increments.

        Strategy:
          - iter 1 & 3: return rows matching expected_rows  → deterministic "correct"
          - iter 2: return no SQL                            → "incorrect" (early-exit in scorer)
        Statuses across iterations: ["correct", "incorrect", "correct"] → inconsistent.
        """
        mock_get_options.return_value = SAMPLE_AGENT_OPTIONS.copy()

        expected_rows = [{"count": 42}]
        mock_connector.run_query.return_value = expected_rows

        mock_agent = MagicMock()
        mock_agent.side_effect = [
            {**self._make_agent_result(), "rows": [{"count": 42}]},  # iter 1 → rows match expected → correct
            {**self._make_agent_result(sql=""), "rows": []},         # iter 2 → no SQL → incorrect
            {**self._make_agent_result(), "rows": [{"count": 42}]},  # iter 3 → rows match expected → correct
        ]
        mock_create_agent.return_value = mock_agent

        results = run_benchmark(
            benchmark_name=MOCK_BENCHMARK_NAME,
            queries={"Q1": {
                "question": "test?",
                "correct_sql": "SELECT COUNT(*) FROM Policy",
            }},
            url=MOCK_URL,
            token=MOCK_TOKEN,
            use_deterministic=True,
            use_llm_judge=False,
            number_of_iterations=3,
        )

        assert results["Q1"]["consistent"] is False
        assert results["Q1"]["status"] == "inconsistent"
        assert results["_summary"]["inconsistent_count"] == 1

    @patch("langchain_timbr.utils.benchmark.create_timbr_sql_agent")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_iterations_detail_structure(self, mock_get_options, mock_create_agent):
        """Each entry in iterations_detail has the required fields."""
        mock_get_options.return_value = SAMPLE_AGENT_OPTIONS.copy()
        mock_agent = MagicMock()
        mock_agent.return_value = self._make_agent_result()
        mock_create_agent.return_value = mock_agent

        results = run_benchmark(
            benchmark_name=MOCK_BENCHMARK_NAME,
            queries={"Q1": {"question": "test?"}},
            url=MOCK_URL,
            token=MOCK_TOKEN,
            use_llm_judge=False,
            number_of_iterations=2,
        )

        detail = results["Q1"]["iterations_detail"]
        assert len(detail) == 2
        for i, entry in enumerate(detail, start=1):
            assert entry["iteration"] == i
            assert "generated_sql" in entry
            assert "status" in entry
            assert "scoring_method" in entry
            assert "score_breakdown" in entry
            assert "tokens_used" in entry

    @patch("langchain_timbr.utils.benchmark.create_timbr_sql_agent")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_tokens_summed_across_iterations(self, mock_get_options, mock_create_agent):
        """Total tokens = per-iteration tokens × number_of_iterations for a single question."""
        mock_get_options.return_value = SAMPLE_AGENT_OPTIONS.copy()
        mock_agent = MagicMock()
        mock_agent.return_value = self._make_agent_result()  # 100 tokens per call
        mock_create_agent.return_value = mock_agent

        results = run_benchmark(
            benchmark_name=MOCK_BENCHMARK_NAME,
            queries={"Q1": {"question": "test?"}},
            url=MOCK_URL,
            token=MOCK_TOKEN,
            use_llm_judge=False,
            number_of_iterations=4,
        )

        assert results["Q1"]["tokens_used"] == 400  # 100 × 4
        assert results["_summary"]["total_tokens_used"] == 400

    @patch("langchain_timbr.utils.benchmark.create_timbr_sql_agent")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_agent_called_once_per_iteration_per_question(self, mock_get_options, mock_create_agent):
        """Agent executor is called exactly number_of_iterations × number_of_questions times."""
        mock_get_options.return_value = SAMPLE_AGENT_OPTIONS.copy()
        mock_agent = MagicMock()
        mock_agent.return_value = self._make_agent_result()
        mock_create_agent.return_value = mock_agent

        run_benchmark(
            benchmark_name=MOCK_BENCHMARK_NAME,
            queries={"Q1": {"question": "q1?"}, "Q2": {"question": "q2?"}},
            url=MOCK_URL,
            token=MOCK_TOKEN,
            use_llm_judge=False,
            number_of_iterations=3,
        )

        assert mock_agent.call_count == 6  # 2 questions × 3 iterations

    @patch("langchain_timbr.utils.benchmark.create_timbr_sql_agent")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_number_of_iterations_in_summary_config(self, mock_get_options, mock_create_agent):
        """number_of_iterations appears in _summary.config."""
        mock_get_options.return_value = SAMPLE_AGENT_OPTIONS.copy()
        mock_agent = MagicMock()
        mock_agent.return_value = self._make_agent_result()
        mock_create_agent.return_value = mock_agent

        results = run_benchmark(
            benchmark_name=MOCK_BENCHMARK_NAME,
            queries={"Q1": {"question": "test?"}},
            url=MOCK_URL,
            token=MOCK_TOKEN,
            use_llm_judge=False,
            number_of_iterations=5,
        )

        assert results["_summary"]["config"]["number_of_iterations"] == 5

    @patch("langchain_timbr.utils.benchmark.GenerateTimbrSqlChain")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_generate_sql_only_with_multiple_iterations(self, mock_get_options, mock_chain_cls):
        """generate_sql_only + iterations: chain is invoked N times per question."""
        mock_get_options.return_value = SAMPLE_AGENT_OPTIONS.copy()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "sql": "SELECT COUNT(*) FROM Policy",
            "concept": "Policy",
            "schema": "dtimbr",
            "is_sql_valid": True,
            "error": None,
            "reasoning_status": None,
            "identify_concept_reason": None,
            "generate_sql_reason": None,
            "generate_sql_usage_metadata": {"total_tokens": 80},
        }
        mock_chain_cls.return_value = mock_chain

        results = run_benchmark(
            benchmark_name=MOCK_BENCHMARK_NAME,
            queries={"Q1": {"question": "How many?"}},
            url=MOCK_URL,
            token=MOCK_TOKEN,
            use_llm_judge=False,
            execution="generate_sql_only",
            number_of_iterations=3,
        )

        assert mock_chain.invoke.call_count == 3
        assert results["Q1"]["consistent"] is True
        assert results["Q1"]["tokens_used"] == 240  # 80 × 3

