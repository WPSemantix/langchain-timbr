"""Unit tests for the run_benchmark utility and BenchmarkScorer."""
import datetime
import decimal
import json
from types import SimpleNamespace

import pytest
from unittest.mock import MagicMock, patch

from langchain_timbr.utils.benchmark import (
    BenchmarkScorer,
    _matches_expected_value,
    _normalize_value,
    _restore_result_labels,
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

CORRECT_VERDICT = '{"assessment": "correct", "reasoning": "Both results match."}'


@pytest.fixture(autouse=True)
def judge():
    """Stub the judge LLM, both judge prompt templates and expected-SQL execution.

    Every test gets a judge that answers "correct". Tests that need something
    else reconfigure the returned mocks (or patch the same names themselves).
    """
    with patch("langchain_timbr.utils.benchmark.LlmWrapper") as llm_wrapper, \
            patch("langchain_timbr.utils.benchmark.get_benchmark_judge_prompt_template") as sql_getter, \
            patch("langchain_timbr.utils.benchmark.get_benchmark_judge_results_prompt_template") as results_getter, \
            patch("langchain_timbr.utils.benchmark.timbr_http_connector") as connector:
        llm = MagicMock(return_value=CORRECT_VERDICT)
        llm_wrapper.return_value = llm

        sql_template = MagicMock()
        sql_template.format_messages.return_value = ["sql-system", "sql-human"]
        sql_getter.return_value = sql_template

        results_template = MagicMock()
        results_template.format_messages.return_value = ["results-system", "results-human"]
        results_getter.return_value = results_template

        connector.run_query.return_value = [{"count": 42}]

        yield SimpleNamespace(
            llm_wrapper=llm_wrapper,
            llm=llm,
            sql_template=sql_template,
            results_template=results_template,
            connector=connector,
        )


def _fixed_rng(value):
    """A random source whose random() always returns *value* (< 0.5 → agent rows are "Result A")."""
    rng = MagicMock()
    rng.random.return_value = value
    return rng


# ---------------------------------------------------------------------------
# Normalisation helpers (concept / ontology checks)
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_normalize_value_none(self):
        assert _normalize_value(None) is None

    def test_normalize_value_numeric(self):
        assert _normalize_value(42) == 42.0
        assert _normalize_value(3.14) == 3.14

    def test_normalize_value_string(self):
        assert _normalize_value("  Hello  ") == "hello"

    def test_matches_expected_value_missing_expected(self):
        assert _matches_expected_value(None, "anything") is True

    def test_matches_expected_value_normalized(self):
        assert _matches_expected_value("POLICY", " policy ") is True
        assert _matches_expected_value("Policy", "Customer") is False


class TestRestoreResultLabels:
    def test_maps_labels_to_origins(self):
        labels = {"A": "agent result", "B": "expected result"}
        text = "Result A has 3 rows while result B has 2."
        assert _restore_result_labels(text, labels) == "Agent result has 3 rows while expected result has 2."

    def test_capitalizes_only_at_sentence_start(self):
        labels = {"A": "agent result", "B": "expected result"}
        text = "Rows differ from Result B. Result A misses one."
        assert _restore_result_labels(text, labels) == "Rows differ from expected result. Agent result misses one."

    def test_leaves_unrelated_text_alone(self):
        labels = {"A": "expected result", "B": "agent result"}
        assert _restore_result_labels("A total of 5 rows", labels) == "A total of 5 rows"


# ---------------------------------------------------------------------------
# BenchmarkScorer
# ---------------------------------------------------------------------------

class TestBenchmarkScorer:
    def _make_scorer(self, llm, rng=None):
        conn_params = {"url": MOCK_URL, "token": MOCK_TOKEN, "ontology": MOCK_ONTOLOGY}
        return BenchmarkScorer(conn_params=conn_params, llm=llm, rng=rng)

    # -- Short-circuits (no judge call) ---------------------------------------

    def test_no_sql_is_incorrect_without_judge(self, judge):
        scorer = self._make_scorer(judge.llm)
        result = scorer.score_result(question="q", generated_sql="", answer="")
        assert result["assessment"] == "incorrect"
        assert result["scoring_method"] == "error"
        judge.llm.assert_not_called()

    def test_execution_error_is_incorrect_without_judge(self, judge):
        scorer = self._make_scorer(judge.llm)
        result = scorer.score_result(
            question="q",
            generated_sql="SELECT 1",
            execution_error="Table not found",
            expected_rows=[{"a": 1}],
        )
        assert result["assessment"] == "incorrect"
        assert result["scoring_method"] == "error"
        assert "Query failed to execute" in result["reasoning"]
        judge.llm.assert_not_called()

    def test_error_in_sql_only_mode_still_judges_sql(self, judge):
        """generate_sql_only has nothing to execute; a chain error does not skip the judge."""
        scorer = self._make_scorer(judge.llm)
        result = scorer.score_result(
            question="q",
            generated_sql="SELECT 1",
            execution_error="validation warning",
            execution_mode="generate_sql_only",
        )
        assert result["scoring_method"] == "llm_judge"
        judge.sql_template.format_messages.assert_called_once()

    def test_no_llm_returns_error(self):
        scorer = self._make_scorer(None)
        result = scorer.score_result(question="q", generated_sql="SELECT 1", expected_rows=[])
        assert result["scoring_method"] == "error"

    # -- Blind results judge --------------------------------------------------

    def test_blind_judge_receives_only_question_and_rows(self, judge):
        scorer = self._make_scorer(judge.llm, rng=_fixed_rng(0.1))
        generated = [{"cnt": 42}]
        expected = [{"count": 42}]
        result = scorer.score_result(
            question="How many policies?",
            generated_sql="SELECT COUNT(*) FROM Policy",
            answer="There are 42 policies.",
            generated_rows=generated,
            expected_sql="SELECT COUNT(*) FROM Policy",
            expected_rows=expected,
        )

        assert result == {
            "assessment": "correct",
            "reasoning": "Both results match.",
            "scoring_method": "llm_judge",
        }
        kwargs = judge.results_template.format_messages.call_args.kwargs
        assert kwargs == {
            "question": "How many policies?",
            "result_a": json.dumps(generated),
            "result_a_row_count": 1,
            "result_b": json.dumps(expected),
            "result_b_row_count": 1,
        }
        # Nothing about SQL, answers or the agent reaches the blind judge
        assert "SELECT" not in json.dumps(kwargs)
        assert "42 policies" not in json.dumps(kwargs)
        judge.sql_template.format_messages.assert_not_called()
        judge.llm.assert_called_once_with(["results-system", "results-human"])

    def test_blind_order_can_put_expected_rows_first(self, judge):
        scorer = self._make_scorer(judge.llm, rng=_fixed_rng(0.9))
        scorer.score_result(
            question="q",
            generated_sql="SELECT 1",
            generated_rows=[{"x": "agent"}],
            expected_rows=[{"x": "expected"}, {"x": "more"}],
        )
        kwargs = judge.results_template.format_messages.call_args.kwargs
        assert json.loads(kwargs["result_a"]) == [{"x": "expected"}, {"x": "more"}]
        assert kwargs["result_a_row_count"] == 2
        assert json.loads(kwargs["result_b"]) == [{"x": "agent"}]

    def test_blind_order_is_randomized(self, judge):
        """With a real random source both orders occur."""
        scorer = self._make_scorer(judge.llm)
        firsts = set()
        for _ in range(40):
            scorer.score_result(
                question="q",
                generated_sql="SELECT 1",
                generated_rows=[{"x": "agent"}],
                expected_rows=[{"x": "expected"}],
            )
            firsts.add(json.loads(judge.results_template.format_messages.call_args.kwargs["result_a"])[0]["x"])
        assert firsts == {"agent", "expected"}

    def test_reasoning_labels_restored_to_origins(self, judge):
        judge.llm.return_value = json.dumps({
            "assessment": "incorrect",
            "reasoning": "Result A has 3 rows but Result B has 2.",
        })
        scorer = self._make_scorer(judge.llm, rng=_fixed_rng(0.9))  # expected rows are A
        result = scorer.score_result(
            question="q", generated_sql="SELECT 1", generated_rows=[{"a": 1}], expected_rows=[{"a": 1}],
        )
        assert result["assessment"] == "incorrect"
        assert result["reasoning"] == "Expected result has 3 rows but agent result has 2."

    def test_non_json_values_are_serialized(self, judge):
        scorer = self._make_scorer(judge.llm, rng=_fixed_rng(0.1))
        rows = [{"amount": decimal.Decimal("1.50"), "day": datetime.date(2026, 1, 2)}]
        scorer.score_result(question="q", generated_sql="SELECT 1", generated_rows=rows, expected_rows=rows)
        kwargs = judge.results_template.format_messages.call_args.kwargs
        assert json.loads(kwargs["result_a"]) == [{"amount": "1.50", "day": "2026-01-02"}]

    def test_missing_generated_rows_compared_as_empty(self, judge):
        scorer = self._make_scorer(judge.llm, rng=_fixed_rng(0.1))
        scorer.score_result(question="q", generated_sql="SELECT 1", generated_rows=None, expected_rows=[{"a": 1}])
        kwargs = judge.results_template.format_messages.call_args.kwargs
        assert kwargs["result_a"] == "[]"
        assert kwargs["result_a_row_count"] == 0

    def test_empty_expected_rows_still_use_blind_judge(self, judge):
        scorer = self._make_scorer(judge.llm, rng=_fixed_rng(0.1))
        scorer.score_result(question="q", generated_sql="SELECT 1", generated_rows=[], expected_rows=[])
        judge.results_template.format_messages.assert_called_once()
        judge.sql_template.format_messages.assert_not_called()

    # -- SQL fallback judge ---------------------------------------------------

    def test_results_template_unavailable_falls_back_to_sql(self, judge):
        judge.results_template.format_messages.side_effect = Exception("Prompt service request failed: 400")
        scorer = self._make_scorer(judge.llm)
        result = scorer.score_result(
            question="q",
            generated_sql="SELECT 1",
            generated_rows=[{"a": 1}],
            expected_sql="SELECT 2",
            expected_rows=[{"a": 1}],
        )
        assert result["scoring_method"] == "llm_judge"
        assert "results judge prompt is unavailable" in result["reasoning"]
        judge.sql_template.format_messages.assert_called_once()
        judge.llm.assert_called_once_with(["sql-system", "sql-human"])

    def test_expected_sql_failure_falls_back_to_sql(self, judge):
        scorer = self._make_scorer(judge.llm)
        result = scorer.score_result(
            question="q",
            generated_sql="SELECT 1",
            generated_rows=[{"a": 1}],
            expected_sql="SELECT broken",
            expected_rows=None,
            expected_rows_error="Syntax error near broken",
        )
        assert result["scoring_method"] == "llm_judge"
        assert "Syntax error near broken" in result["reasoning"]
        judge.results_template.format_messages.assert_not_called()
        kwargs = judge.sql_template.format_messages.call_args.kwargs
        assert "SELECT broken" in kwargs["expected_sql_context"]

    def test_no_reference_sql_in_full_mode_falls_back_to_sql(self, judge):
        scorer = self._make_scorer(judge.llm)
        result = scorer.score_result(question="q", generated_sql="SELECT 1", generated_rows=[{"a": 1}])
        assert "No reference SQL was provided" in result["reasoning"]
        judge.results_template.format_messages.assert_not_called()

    def test_sql_only_mode_judges_sql_with_reference(self, judge):
        scorer = self._make_scorer(judge.llm)
        result = scorer.score_result(
            question="How many active policies?",
            generated_sql="SELECT COUNT(*) FROM Policy",
            answer="ignored",
            expected_sql="SELECT COUNT(*) FROM Policy WHERE status = 'active'",
            execution_mode="generate_sql_only",
        )
        kwargs = judge.sql_template.format_messages.call_args.kwargs
        assert kwargs["question"] == "How many active policies?"
        assert kwargs["generated_sql"] == "SELECT COUNT(*) FROM Policy"
        assert kwargs["answer_context"] == ""
        assert kwargs["expected_answer_context"] == ""
        assert "status = 'active'" in kwargs["expected_sql_context"]
        # No fallback note in SQL-only mode: judging SQL is the expected path
        assert result["reasoning"] == "Both results match."

    def test_sql_only_mode_without_reference(self, judge):
        scorer = self._make_scorer(judge.llm)
        scorer.score_result(question="q", generated_sql="SELECT 1", execution_mode="generate_sql_only")
        assert judge.sql_template.format_messages.call_args.kwargs["expected_sql_context"] == ""

    # -- Judge output handling ------------------------------------------------

    def test_invalid_json_is_sent_back_for_correction(self, judge):
        judge.llm._llm_type = "test"
        judge.llm.side_effect = [
            "The answer looks right to me.",
            '{"assessment": "correct", "reasoning": "ok"}',
        ]
        scorer = self._make_scorer(judge.llm, rng=_fixed_rng(0.1))
        result = scorer.score_result(
            question="q", generated_sql="SELECT 1", generated_rows=[], expected_rows=[],
        )

        assert result["assessment"] == "correct"
        assert judge.llm.call_count == 2
        retry_messages = judge.llm.call_args_list[1].args[0]
        assert retry_messages[:2] == ["results-system", "results-human"]
        assert retry_messages[2].content == "The answer looks right to me."
        assert "not valid JSON" in retry_messages[3].content

    def test_unknown_assessment_maps_to_incorrect(self, judge):
        judge.llm.return_value = '{"assessment": "maybe", "reasoning": "unsure"}'
        scorer = self._make_scorer(judge.llm)
        result = scorer.score_result(question="q", generated_sql="SELECT 1", expected_rows=[])
        assert result["assessment"] == "incorrect"
        assert result["scoring_method"] == "llm_judge"

    def test_llm_failure_returns_error(self, judge):
        judge.llm.side_effect = Exception("LLM unavailable")
        scorer = self._make_scorer(judge.llm)
        result = scorer.score_result(question="q", generated_sql="SELECT 1", expected_rows=[])
        assert result["assessment"] == "incorrect"
        assert result["scoring_method"] == "error"
        assert "LLM unavailable" in result["reasoning"]

    def test_each_verdict_is_a_fresh_call(self, judge):
        """Successive questions never carry earlier messages into the judge's context."""
        scorer = self._make_scorer(judge.llm, rng=_fixed_rng(0.1))
        for _ in range(3):
            scorer.score_result(question="q", generated_sql="SELECT 1", generated_rows=[], expected_rows=[])
        assert [c.args[0] for c in judge.llm.call_args_list] == [["results-system", "results-human"]] * 3

    def test_result_has_no_breakdown(self, judge):
        scorer = self._make_scorer(judge.llm)
        result = scorer.score_result(question="q", generated_sql="SELECT 1", expected_rows=[])
        assert "breakdown" not in result

    # -- Deprecated arguments -------------------------------------------------

    def test_deprecated_mechanism_args_warn_and_are_ignored(self, judge):
        with pytest.warns(DeprecationWarning) as record:
            scorer = BenchmarkScorer(conn_params={}, llm=judge.llm, use_deterministic=True, use_llm_judge=False)
        messages = " ".join(str(w.message) for w in record)
        assert "use_deterministic" in messages and "use_llm_judge" in messages
        result = scorer.score_result(question="q", generated_sql="SELECT 1", expected_rows=[])
        assert result["scoring_method"] == "llm_judge"


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

    def _run(self, mock_get_options, mock_create_agent, queries=None, options=None, **kwargs):
        mock_get_options.return_value = options or SAMPLE_AGENT_OPTIONS.copy()
        mock_agent_executor = MagicMock()
        mock_agent_executor.invoke.return_value = self._make_agent_result()
        mock_create_agent.return_value = mock_agent_executor
        return run_benchmark(
            benchmark_name=MOCK_BENCHMARK_NAME,
            queries=queries if queries is not None else SAMPLE_QUESTIONS,
            url=MOCK_URL,
            token=MOCK_TOKEN,
            **kwargs,
        )

    @patch("langchain_timbr.utils.benchmark.create_timbr_sql_agent")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_basic_run(self, mock_get_options, mock_create_agent):
        results = self._run(mock_get_options, mock_create_agent)

        assert "_summary" in results
        assert results["_summary"]["total_questions"] == 2
        assert (
            results["_summary"]["correct_count"]
            + results["_summary"]["incorrect_count"]
            == 2
        )
        assert "Q1" in results
        assert "Q2" in results
        assert "generated_sql" in results["Q1"]
        assert "selected_entity" in results["Q1"]
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
        mock_agent_executor.invoke.return_value = self._make_agent_result()
        mock_create_agent.return_value = mock_agent_executor

        results = run_benchmark(
            benchmark_name=MOCK_BENCHMARK_NAME,
            url=MOCK_URL,
            token=MOCK_TOKEN,
        )

        assert results["_summary"]["total_questions"] == 1
        assert "Q1" in results

    @patch("langchain_timbr.utils.benchmark.create_timbr_sql_agent")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_correct_sql_rows_are_judged_blind(self, mock_get_options, mock_create_agent, judge):
        """correct_sql is executed and the judge compares both row sets."""
        judge.connector.run_query.return_value = [{"total": 42}]
        results = self._run(
            mock_get_options, mock_create_agent,
            queries={"Q1": {"question": "How many?", "correct_sql": "SELECT COUNT(*) FROM P;"}},
        )

        judge.connector.run_query.assert_called_once()
        assert judge.connector.run_query.call_args.kwargs["query"] == "SELECT COUNT(*) FROM P"
        kwargs = judge.results_template.format_messages.call_args.kwargs
        assert {kwargs["result_a"], kwargs["result_b"]} == {
            json.dumps([{"count": 42}]), json.dumps([{"total": 42}]),
        }
        judge.sql_template.format_messages.assert_not_called()
        assert results["Q1"]["status"] == "correct"
        assert results["Q1"]["scoring_method"] == "llm_judge"

    @patch("langchain_timbr.utils.benchmark.create_timbr_sql_agent")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_correct_sql_failure_falls_back_to_sql_judge(self, mock_get_options, mock_create_agent, judge):
        judge.connector.run_query.side_effect = Exception("Table not found")
        results = self._run(
            mock_get_options, mock_create_agent,
            queries={"Q1": {"question": "How many?", "correct_sql": "SELECT COUNT(*) FROM Missing"}},
        )
        judge.results_template.format_messages.assert_not_called()
        judge.sql_template.format_messages.assert_called_once()
        assert "Table not found" in results["Q1"]["score_reasoning"]

    @patch("langchain_timbr.utils.benchmark.create_timbr_sql_agent")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_question_without_correct_sql_judges_sql(self, mock_get_options, mock_create_agent, judge):
        results = self._run(mock_get_options, mock_create_agent, queries={"Q1": {"question": "List all?"}})
        judge.connector.run_query.assert_not_called()
        judge.sql_template.format_messages.assert_called_once()
        assert "No reference SQL was provided" in results["Q1"]["score_reasoning"]

    @patch("langchain_timbr.utils.benchmark.create_timbr_sql_agent")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_mechanism_agent_options_are_ignored(self, mock_get_options, mock_create_agent):
        """The old scoring agent options no longer turn the judge off."""
        options = SAMPLE_AGENT_OPTIONS.copy()
        options["use_llm_judge_scoring"] = "false"
        options["use_deterministic_scoring"] = "true"
        results = self._run(mock_get_options, mock_create_agent, queries={"Q1": {"question": "test?"}}, options=options)
        assert results["Q1"]["scoring_method"] == "llm_judge"

    @patch("langchain_timbr.utils.benchmark.create_timbr_sql_agent")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_deprecated_mechanism_params_warn_and_are_ignored(self, mock_get_options, mock_create_agent):
        with pytest.warns(DeprecationWarning) as record:
            results = self._run(
                mock_get_options, mock_create_agent,
                queries={"Q1": {"question": "test?"}},
                use_deterministic=True,
                use_llm_judge=False,
            )
        messages = " ".join(str(w.message) for w in record)
        assert "use_deterministic" in messages
        assert "use_llm_judge" in messages
        assert results["Q1"]["scoring_method"] == "llm_judge"

    @patch("langchain_timbr.utils.benchmark.create_timbr_sql_agent")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_judge_uses_its_own_llm_instance(self, mock_get_options, mock_create_agent, judge):
        """The judge LLM is always built, separately from the agent."""
        self._run(mock_get_options, mock_create_agent, queries={"Q1": {"question": "test?"}})
        judge.llm_wrapper.assert_called_once()
        assert mock_create_agent.call_args.kwargs["llm"] is None

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
            )

    @patch("langchain_timbr.utils.benchmark.create_timbr_sql_agent")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_summary_structure(self, mock_get_options, mock_create_agent):
        results = self._run(mock_get_options, mock_create_agent)

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
        assert summary["config"]["scoring_method"] == "llm_judge"
        assert "use_deterministic_scoring" not in summary["config"]

    @patch("langchain_timbr.utils.benchmark.create_timbr_sql_agent")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_results_have_no_score_breakdown(self, mock_get_options, mock_create_agent):
        results = self._run(mock_get_options, mock_create_agent)
        for qid in ("Q1", "Q2"):
            assert "score_breakdown" not in results[qid]
            assert results[qid]["score_reasoning"]

    @patch("langchain_timbr.utils.benchmark.create_timbr_sql_agent")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_error_count_tracking(self, mock_get_options, mock_create_agent):
        """Questions where the agent throws an exception increment error_count."""
        mock_get_options.return_value = SAMPLE_AGENT_OPTIONS.copy()

        mock_agent_executor = MagicMock()
        mock_agent_executor.invoke.side_effect = Exception("Agent failure")
        mock_create_agent.return_value = mock_agent_executor

        results = run_benchmark(
            benchmark_name=MOCK_BENCHMARK_NAME,
            queries={"Q1": {"question": "test?"}},
            url=MOCK_URL,
            token=MOCK_TOKEN,
        )
        assert results["_summary"]["error_count"] == 1
        assert results["Q1"]["status"] == "incorrect"

    @patch("langchain_timbr.utils.benchmark.create_timbr_sql_agent")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_agent_query_error_is_incorrect_without_judge(self, mock_get_options, mock_create_agent, judge):
        mock_get_options.return_value = SAMPLE_AGENT_OPTIONS.copy()
        agent_result = self._make_agent_result()
        agent_result["error"] = "Column not found"
        mock_create_agent.return_value.invoke.return_value = agent_result

        results = run_benchmark(
            benchmark_name=MOCK_BENCHMARK_NAME,
            queries={"Q1": {"question": "test?", "correct_sql": "SELECT 1"}},
            url=MOCK_URL,
            token=MOCK_TOKEN,
        )
        assert results["Q1"]["status"] == "incorrect"
        assert "Column not found" in results["Q1"]["score_reasoning"]
        judge.llm.assert_not_called()

    @patch("langchain_timbr.utils.benchmark.create_timbr_sql_agent")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_total_tokens_summed(self, mock_get_options, mock_create_agent):
        """total_tokens_used in summary is the sum of all per-question tokens."""
        results = self._run(mock_get_options, mock_create_agent)  # 2 questions × 300 tokens
        assert results["_summary"]["total_tokens_used"] == 600

    @patch("langchain_timbr.utils.benchmark.create_timbr_sql_agent")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_log_history_payload_fields(self, mock_get_options, mock_create_agent):
        """The history logging call receives all required fields."""
        self._run(mock_get_options, mock_create_agent, number_of_iterations=1)

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
        mock_agent_executor.invoke.side_effect = [
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
        )

        assert results["Q1"]["correct_concept"] is True
        assert results["Q1"]["correct_ontology"] is True
        assert results["Q2"]["correct_concept"] is True
        assert results["Q2"]["correct_ontology"] is True


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
            "generate_sql_usage_metadata": {"generate_sql": {"total_tokens": 150}},
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
            execution="generate_sql_only",
        )

        q = results["Q1"]
        assert q["generated_sql"] == "SELECT COUNT(*) FROM Policy"
        assert q["selected_entity"] == "Policy"
        assert q["answer"] == ""  # no execution, no answer
        assert "status" in q
        assert "scoring_method" in q

    @patch("langchain_timbr.utils.benchmark.GenerateTimbrSqlChain")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_generate_sql_only_judges_sql_against_reference(self, mock_get_options, mock_chain_cls, judge):
        """SQL-only: nothing is executed; the judge sees the candidate and reference SQL."""
        mock_get_options.return_value = SAMPLE_AGENT_OPTIONS.copy()
        expected_sql = "SELECT COUNT(*) FROM Policy WHERE status = 'active'"
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = self._make_chain_result(sql="SELECT name FROM Customer LIMIT 10")
        mock_chain_cls.return_value = mock_chain
        judge.llm.return_value = '{"assessment": "incorrect", "reasoning": "Wrong concept."}'

        results = run_benchmark(
            benchmark_name=MOCK_BENCHMARK_NAME,
            queries={"Q1": {"question": "How many active policies?", "correct_sql": expected_sql}},
            url=MOCK_URL,
            token=MOCK_TOKEN,
            execution="generate_sql_only",
        )

        judge.connector.run_query.assert_not_called()
        judge.results_template.format_messages.assert_not_called()
        kwargs = judge.sql_template.format_messages.call_args.kwargs
        assert kwargs["generated_sql"] == "SELECT name FROM Customer LIMIT 10"
        assert expected_sql in kwargs["expected_sql_context"]
        assert results["Q1"]["status"] == "incorrect"
        assert results["Q1"]["scoring_method"] == "llm_judge"
        assert results["Q1"]["score_reasoning"] == "Wrong concept."

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
        mock_agent.invoke.return_value = {
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
        mock_agent.invoke.return_value = self._make_agent_result()
        mock_create_agent.return_value = mock_agent

        results = run_benchmark(
            benchmark_name=MOCK_BENCHMARK_NAME,
            queries={"Q1": {"question": "test?"}},
            url=MOCK_URL,
            token=MOCK_TOKEN,
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
          - iter 1 & 3: the judge returns "correct"
          - iter 2: return no SQL → "incorrect" (early-exit in scorer, no judge call)
        Statuses across iterations: ["correct", "incorrect", "correct"] → inconsistent.
        """
        mock_get_options.return_value = SAMPLE_AGENT_OPTIONS.copy()

        expected_rows = [{"count": 42}]
        mock_connector.run_query.return_value = expected_rows

        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = [
            {**self._make_agent_result(), "rows": [{"count": 42}]},  # iter 1 → judged correct
            {**self._make_agent_result(sql=""), "rows": []},         # iter 2 → no SQL → incorrect
            {**self._make_agent_result(), "rows": [{"count": 42}]},  # iter 3 → judged correct
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
        mock_agent.invoke.return_value = self._make_agent_result()
        mock_create_agent.return_value = mock_agent

        results = run_benchmark(
            benchmark_name=MOCK_BENCHMARK_NAME,
            queries={"Q1": {"question": "test?"}},
            url=MOCK_URL,
            token=MOCK_TOKEN,
            number_of_iterations=2,
        )

        detail = results["Q1"]["iterations_detail"]
        assert len(detail) == 2
        for i, entry in enumerate(detail, start=1):
            assert entry["iteration"] == i
            assert "generated_sql" in entry
            assert "status" in entry
            assert "scoring_method" in entry
            assert "score_breakdown" not in entry
            assert "tokens_used" in entry

    @patch("langchain_timbr.utils.benchmark.create_timbr_sql_agent")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_tokens_summed_across_iterations(self, mock_get_options, mock_create_agent):
        """Total tokens = per-iteration tokens × number_of_iterations for a single question."""
        mock_get_options.return_value = SAMPLE_AGENT_OPTIONS.copy()
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = self._make_agent_result()  # 100 tokens per call
        mock_create_agent.return_value = mock_agent

        results = run_benchmark(
            benchmark_name=MOCK_BENCHMARK_NAME,
            queries={"Q1": {"question": "test?"}},
            url=MOCK_URL,
            token=MOCK_TOKEN,
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
        mock_agent.invoke.return_value = self._make_agent_result()
        mock_create_agent.return_value = mock_agent

        run_benchmark(
            benchmark_name=MOCK_BENCHMARK_NAME,
            queries={"Q1": {"question": "q1?"}, "Q2": {"question": "q2?"}},
            url=MOCK_URL,
            token=MOCK_TOKEN,
            number_of_iterations=3,
        )

        assert mock_agent.invoke.call_count == 6  # 2 questions × 3 iterations

    @patch("langchain_timbr.utils.benchmark.create_timbr_sql_agent")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_number_of_iterations_in_summary_config(self, mock_get_options, mock_create_agent):
        """number_of_iterations appears in _summary.config."""
        mock_get_options.return_value = SAMPLE_AGENT_OPTIONS.copy()
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = self._make_agent_result()
        mock_create_agent.return_value = mock_agent

        results = run_benchmark(
            benchmark_name=MOCK_BENCHMARK_NAME,
            queries={"Q1": {"question": "test?"}},
            url=MOCK_URL,
            token=MOCK_TOKEN,
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
            "generate_sql_usage_metadata": {"generate_sql": {"total_tokens": 80}},
        }
        mock_chain_cls.return_value = mock_chain

        results = run_benchmark(
            benchmark_name=MOCK_BENCHMARK_NAME,
            queries={"Q1": {"question": "How many?"}},
            url=MOCK_URL,
            token=MOCK_TOKEN,
            execution="generate_sql_only",
            number_of_iterations=3,
        )

        assert mock_chain.invoke.call_count == 3
        assert results["Q1"]["consistent"] is True
        assert results["Q1"]["tokens_used"] == 240  # 80 × 3



class TestRunBenchmarkDuration:
    """Per-question completion time (duration_ms) in run_benchmark results."""

    def setup_method(self, method):
        from unittest.mock import patch as _patch
        self._patches = [
            _patch("langchain_timbr.utils.benchmark.get_timbr_benchmark_info"),
            _patch("langchain_timbr.utils.benchmark._log_benchmark_running"),
            _patch("langchain_timbr.utils.benchmark._log_benchmark_update_completed"),
            _patch("langchain_timbr.utils.benchmark._log_benchmark_history"),
            _patch("langchain_timbr.utils.benchmark.perf_counter"),
        ]
        self.mock_get_bm_info = self._patches[0].start()
        self.mock_get_bm_info.return_value = SAMPLE_BENCHMARK_INFO.copy()
        self._patches[1].start()
        self._patches[2].start()
        self.mock_log_history = self._patches[3].start()
        self.mock_perf_counter = self._patches[4].start()

    def teardown_method(self, method):
        for p in self._patches:
            p.stop()

    def _set_clock(self, *iteration_seconds):
        """Make perf_counter yield (start, end) pairs giving the given per-iteration durations."""
        ticks, now = [], 0.0
        for secs in iteration_seconds:
            ticks.extend([now, now + secs])
            now += secs + 100.0  # gap between iterations must not be counted
        self.mock_perf_counter.side_effect = ticks

    def _make_agent_result(self):
        return {
            "sql": "SELECT * FROM Policy",
            "rows": [{"count": 1}],
            "answer": "One policy.",
            "concept": "Policy",
            "ontology": MOCK_ONTOLOGY,
            "error": None,
            "usage_metadata": {"generate_sql": {"total_tokens": 100}},
        }

    def _run(self, mock_get_options, mock_create_agent, queries, **kwargs):
        mock_get_options.return_value = SAMPLE_AGENT_OPTIONS.copy()
        if not isinstance(mock_create_agent.return_value.invoke.side_effect, list):
            mock_create_agent.return_value.invoke.return_value = self._make_agent_result()
        return run_benchmark(
            benchmark_name=MOCK_BENCHMARK_NAME,
            queries=queries,
            url=MOCK_URL,
            token=MOCK_TOKEN,
            **kwargs,
        )

    @patch("langchain_timbr.utils.benchmark.create_timbr_sql_agent")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_single_iteration_duration_per_question(self, mock_get_options, mock_create_agent):
        self._set_clock(1.5, 0.25)
        results = self._run(
            mock_get_options, mock_create_agent,
            {"Q1": {"question": "q1?"}, "Q2": {"question": "q2?"}},
        )
        assert results["Q1"]["duration_ms"] == 1500
        assert results["Q2"]["duration_ms"] == 250

    @patch("langchain_timbr.utils.benchmark.create_timbr_sql_agent")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_multiple_iterations_average_and_detail(self, mock_get_options, mock_create_agent):
        self._set_clock(1.0, 2.0, 4.5)
        results = self._run(
            mock_get_options, mock_create_agent,
            {"Q1": {"question": "q1?"}},
            number_of_iterations=3,
        )
        assert [d["duration_ms"] for d in results["Q1"]["iterations_detail"]] == [1000, 2000, 4500]
        assert results["Q1"]["duration_ms"] == 2500  # (1000 + 2000 + 4500) / 3

    @patch("langchain_timbr.utils.benchmark.create_timbr_sql_agent")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_duration_recorded_when_agent_raises(self, mock_get_options, mock_create_agent):
        self._set_clock(3.0)
        mock_create_agent.return_value.invoke.side_effect = [RuntimeError("boom")]
        results = self._run(mock_get_options, mock_create_agent, {"Q1": {"question": "q1?"}})
        assert results["Q1"]["error"] == "boom"
        assert results["Q1"]["duration_ms"] == 3000

    @patch("langchain_timbr.utils.benchmark.GenerateTimbrSqlChain")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_generate_sql_only_duration(self, mock_get_options, mock_chain_cls):
        self._set_clock(0.75)
        mock_get_options.return_value = SAMPLE_AGENT_OPTIONS.copy()
        mock_chain_cls.return_value.invoke.return_value = {"sql": "SELECT 1", "error": None}
        results = run_benchmark(
            benchmark_name=MOCK_BENCHMARK_NAME,
            queries={"Q1": {"question": "q1?"}},
            url=MOCK_URL,
            token=MOCK_TOKEN,
            execution="generate_sql_only",
        )
        assert results["Q1"]["duration_ms"] == 750

    @patch("langchain_timbr.utils.benchmark.create_timbr_sql_agent")
    @patch("langchain_timbr.utils.benchmark.get_timbr_agent_options")
    def test_duration_in_log_history_payload(self, mock_get_options, mock_create_agent):
        self._set_clock(2.0)
        self._run(mock_get_options, mock_create_agent, {"Q1": {"question": "q1?"}})
        payload = self.mock_log_history.call_args.kwargs["payload"]
        assert payload["result"]["Q1"]["duration_ms"] == 2000


# ---------------------------------------------------------------------------
# benchmark_cli
# ---------------------------------------------------------------------------

class TestBenchmarkCli:
    def _main(self, argv, monkeypatch):
        from langchain_timbr import benchmark_cli
        monkeypatch.setattr("sys.argv", ["benchmark_cli", *argv])
        with patch.object(benchmark_cli, "run_benchmark") as mock_run:
            exit_code = benchmark_cli.main()
        return exit_code, mock_run

    def test_runs_without_mechanism(self, monkeypatch):
        exit_code, mock_run = self._main(
            ["--benchmark-name", "b", "--url", MOCK_URL, "--token", MOCK_TOKEN], monkeypatch,
        )
        assert exit_code == 0
        kwargs = mock_run.call_args.kwargs
        assert "use_deterministic" not in kwargs
        assert "use_llm_judge" not in kwargs

    @pytest.mark.parametrize("mechanism", ["full", "deterministic", "llm_judge"])
    def test_deprecated_mechanism_is_accepted_and_ignored(self, mechanism, monkeypatch, caplog):
        exit_code, mock_run = self._main(
            ["--benchmark-name", "b", "--url", MOCK_URL, "--token", MOCK_TOKEN, "--mechanism", mechanism],
            monkeypatch,
        )
        assert exit_code == 0
        assert "use_deterministic" not in mock_run.call_args.kwargs
        assert "--mechanism is deprecated" in caplog.text
