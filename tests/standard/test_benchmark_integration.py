"""
Manual integration tests for run_benchmark().

These tests make real HTTP calls to a live Timbr server.
They are NOT run in CI (no mock patches) — execute them manually when you
want to validate end-to-end benchmark behaviour.

Usage
-----
Set the required environment variables (defaults defined in tests/conftest.py),
then run:

    python -m pytest tests/standard/test_benchmark_integration.py -v -s

or run a single test by name:

    python -m pytest tests/standard/test_benchmark_integration.py::TestBenchmarkIntegration::test_llm_judge_scoring -v -s

Configuration
-------------
All connection settings are read from the shared ``config`` fixture in
``tests/conftest.py``.  The relevant env vars are:

    TIMBR_URL, TIMBR_TOKEN, TIMBR_ONTOLOGY, TIMBR_BENCHMARK, VERIFY_SSL,
    JWT_TENANT_ID
"""

import pprint

import pytest

from langchain_timbr import run_benchmark

# ---------------------------------------------------------------------------
# Sample inline queries — edit to match your ontology
# ---------------------------------------------------------------------------
SAMPLE_QUERIES = {
    "Q1": {
        "question": "Most valueable item of the oldest person",
        # Optionally include correct_sql so the LLM judge compares result rows blind:
        # "correct_sql": "SELECT COUNT(*) AS cnt FROM <SomeConcept>",
        # "expected_answer": "42",
    },
    "Q2": {
        "question": "Who follows the youngest?",
        # "correct_sql": "SELECT * FROM <SomeConcept> LIMIT 5",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skip_if_not_configured(config: dict):
    """Skip the test if required connection values are missing."""
    if not config.get("timbr_token"):
        pytest.skip(
            "Integration test skipped — set TIMBR_TOKEN (and optionally "
            "TIMBR_URL, TIMBR_ONTOLOGY, TIMBR_BENCHMARK) environment variables."
        )
    if not config.get("timbr_benchmark"):
        pytest.skip(
            "Integration test skipped — set TIMBR_BENCHMARK environment variable."
        )


def _print_results(results: dict):
    """Pretty-print benchmark results for manual inspection."""
    summary = results.pop("_summary", None)
    pprint.pprint(results, width=120)
    if summary:
        print("\n--- SUMMARY ---")
        pprint.pprint(summary, width=120)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class Skip_TestBenchmarkIntegration:

    def skip_test_llm_judge_scoring(self, config):
        """
        Run benchmark with the default (and only) scoring: the LLM judge.
        Questions with correct_sql are judged by blind row comparison; the rest by
        judging the SQL against the question. Requires the benchmark_judge and
        benchmark_judge_results prompt templates to be served by the platform.
        """
        _skip_if_not_configured(config)

        results = run_benchmark(
            benchmark_name=config["timbr_benchmark"],
            queries=SAMPLE_QUERIES,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            verify_ssl=config["verify_ssl"],
            jwt_tenant_id=config.get("jwt_tenant_id"),
        )

        _print_results(results)

        for q_id, r in results.items():
            for field in (
                "generated_sql", "selected_entity", "selected_ontology", "identify_concept_reason",
                "generate_sql_reason", "correct_concept", "correct_ontology", "status",
                "scoring_method", "score_reasoning",
            ):
                assert field in r, f"{q_id}: missing {field}"
            assert "score_breakdown" not in r, f"{q_id}: score_breakdown should no longer be written"
            assert r["scoring_method"] in ("llm_judge", "error"),                 f"{q_id}: unexpected scoring_method '{r['scoring_method']}'"

    def skip_test_queries_from_benchmark_options(self, config):
        """
        Run benchmark without providing inline queries.
        Questions will be loaded from the benchmark's 'questions' option in sys_agent_benchmarks.
        The benchmark must have a 'questions' option set (JSON string in questions-enhanced format).
        """
        _skip_if_not_configured(config)

        results = run_benchmark(
            benchmark_name=config["timbr_benchmark"],
            # queries intentionally omitted — loaded from sys_agent_benchmarks
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            verify_ssl=config["verify_ssl"],
            jwt_tenant_id=config.get("jwt_tenant_id"),
        )

        _print_results(results)

        assert isinstance(results, dict)
        assert len(results) > 0, "Expected at least one result from agent questions option"

    def skip_test_generate_sql_only_execution(self, config):
        """
        Run benchmark with execution='generate_sql_only'.
        Uses GenerateTimbrSqlChain — no SQL execution, no answer generation.
        The generated_sql field should be populated; the answer field should be empty.
        """
        _skip_if_not_configured(config)

        results = run_benchmark(
            benchmark_name=config["timbr_benchmark"],
            queries=SAMPLE_QUERIES,
            url=config["timbr_url"],
            token=config["timbr_token"],
            verify_ssl=config["verify_ssl"],
            execution="generate_sql_only",
        )

        _print_results(results)

        for q_id, r in results.items():
            assert "generated_sql" in r, f"{q_id}: missing generated_sql"
            assert "answer" in r, f"{q_id}: missing answer field"
            assert r["answer"] == "", f"{q_id}: answer should be empty in generate_sql_only mode"
            assert r["scoring_method"] in ("llm_judge", "error"), f"{q_id}: unexpected scoring_method"

    def skip_test_generate_sql_only_with_llm_judge(self, config):
        """
        Run benchmark with execution='generate_sql_only'.
        The judge evaluates the SQL against the question (and correct_sql, when given).
        """
        _skip_if_not_configured(config)

        results = run_benchmark(
            benchmark_name=config["timbr_benchmark"],
            queries=SAMPLE_QUERIES,
            url=config["timbr_url"],
            token=config["timbr_token"],
            verify_ssl=config["verify_ssl"],
            execution="generate_sql_only",
        )

        _print_results(results)

        for q_id, r in results.items():
            assert r["scoring_method"] in ("llm_judge", "error"), \
                f"{q_id}: unexpected scoring_method '{r['scoring_method']}'"
            if r["scoring_method"] == "llm_judge":
                assert r["status"] in ("correct", "partial", "incorrect"), \
                    f"{q_id}: unexpected status '{r['status']}'"

    def skip_test_full_execution_with_iterations(self, config):
        """
        Run benchmark with execution='full' and number_of_iterations=3.
        Each question is executed 3 times; results should include:
          - 'consistent'       (bool) on each question
          - 'iterations_detail' (list of 3 entries) on each question
          - 'inconsistent_count' in _summary (>= 0)
        """
        _skip_if_not_configured(config)

        results = run_benchmark(
            benchmark_name=config["timbr_benchmark"],
            queries=SAMPLE_QUERIES,
            url=config["timbr_url"],
            token=config["timbr_token"],
            verify_ssl=config["verify_ssl"],
            execution="full",
            number_of_iterations=3,
        )

        summary = results.pop("_summary", {})
        _print_results(results)

        for q_id, r in results.items():
            assert "consistent" in r, f"{q_id}: missing 'consistent' field"
            assert isinstance(r["consistent"], bool), f"{q_id}: 'consistent' should be a bool"
            assert "iterations_detail" in r, f"{q_id}: missing 'iterations_detail' field"
            assert len(r["iterations_detail"]) == 3, \
                f"{q_id}: expected 3 iteration entries, got {len(r['iterations_detail'])}"
            for entry in r["iterations_detail"]:
                assert "iteration" in entry
                assert "generated_sql" in entry
                assert "status" in entry

        assert "inconsistent_count" in summary
        assert isinstance(summary["inconsistent_count"], int)
        print(f"\n--- ITERATION SUMMARY ---\nconsistent_count={summary.get('correct_count')}, "
              f"inconsistent_count={summary['inconsistent_count']}")

    def skip_test_generate_sql_only_with_iterations(self, config):
        """
        Run benchmark with execution='generate_sql_only' and number_of_iterations=3.
        Verifies that the SQL chain is called N times and consistency tracking works.
        """
        _skip_if_not_configured(config)

        results = run_benchmark(
            benchmark_name=config["timbr_benchmark"],
            queries=SAMPLE_QUERIES,
            url=config["timbr_url"],
            token=config["timbr_token"],
            verify_ssl=config["verify_ssl"],
            execution="generate_sql_only",
            number_of_iterations=3,
        )

        summary = results.pop("_summary", {})
        _print_results(results)

        for q_id, r in results.items():
            assert "consistent" in r, f"{q_id}: missing 'consistent' field"
            assert "iterations_detail" in r, f"{q_id}: missing 'iterations_detail' field"
            assert len(r["iterations_detail"]) == 3, \
                f"{q_id}: expected 3 iteration entries"
            # In SQL-only mode each iteration answer should be empty
            for entry in r["iterations_detail"]:
                assert entry.get("generated_sql") is not None

        assert summary.get("config", {}).get("execution") == "generate_sql_only"
        assert summary.get("config", {}).get("number_of_iterations") == 3

