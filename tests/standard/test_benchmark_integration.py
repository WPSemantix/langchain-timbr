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

    python -m pytest tests/standard/test_benchmark_integration.py::TestBenchmarkIntegration::test_basic_scoring -v -s

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
        # Optionally include correct_sql and expected_answer for deterministic / LLM-judge scoring:
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

class TestBenchmarkIntegration:

    def test_basic_scoring(self, config):
        """
        Run benchmark with basic scoring only (no deterministic, no LLM judge).
        Score is based purely on whether the agent produced valid SQL and an answer.
        """
        _skip_if_not_configured(config)

        results = run_benchmark(
            benchmark_name=config["timbr_benchmark"],
            queries=SAMPLE_QUERIES,
            url=config["timbr_url"],
            token=config["timbr_token"],
            # ontology=config["timbr_ontology"],
            use_deterministic=False,
            use_llm_judge=False,
            verify_ssl=config["verify_ssl"],
        )

        _print_results(results)

        assert isinstance(results, dict), "Expected a dict result"
        assert "_summary" not in results, "_summary should have been popped by _print_results"

        # Each question key should have at minimum these fields
        for q_id, r in results.items():
            assert "generated_sql" in r, f"{q_id}: missing generated_sql"
            assert "selected_concept" in r, f"{q_id}: missing selected_concept"
            assert "selected_ontology" in r, f"{q_id}: missing selected_ontology"
            assert "identify_concept_reason" in r, f"{q_id}: missing identify_concept_reason"
            assert "generate_sql_reason" in r, f"{q_id}: missing generate_sql_reason"
            assert "correct_concept" in r, f"{q_id}: missing correct_concept"
            assert "correct_ontology" in r, f"{q_id}: missing correct_ontology"
            assert "status" in r, f"{q_id}: missing status"
            assert r["status"] in ("correct", "partial", "incorrect", "error"), \
                f"{q_id}: unexpected status '{r['status']}'"
            assert "scoring_method" in r, f"{q_id}: missing scoring_method"

    def test_deterministic_scoring(self, config):
        """
        Run benchmark with deterministic row-comparison scoring.
        Requires correct_sql in the query definitions to be meaningful.
        """
        _skip_if_not_configured(config)

        queries_with_sql = {
            k: v for k, v in SAMPLE_QUERIES.items() if v.get("correct_sql")
        }
        if not queries_with_sql:
            pytest.skip(
                "No queries with 'correct_sql' defined — add correct_sql to "
                "SAMPLE_QUERIES to enable deterministic scoring."
            )

        results = run_benchmark(
            benchmark_name=config["timbr_benchmark"],
            queries=queries_with_sql,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            use_deterministic=True,
            use_llm_judge=False,
            verify_ssl=config["verify_ssl"],
            jwt_tenant_id=config.get("jwt_tenant_id"),
        )

        _print_results(results)

        for q_id, r in results.items():
            assert r["scoring_method"] in ("deterministic", "hybrid", "basic", "error"), \
                f"{q_id}: unexpected scoring_method '{r['scoring_method']}'"

    def test_llm_judge_scoring(self, config):
        """
        Run benchmark with LLM-as-judge scoring.
        The LLM evaluates the generated answer against the expected answer / SQL context.
        Requires the benchmark_judge prompt template to be deployed in timbr-api.
        """
        _skip_if_not_configured(config)

        results = run_benchmark(
            benchmark_name=config["timbr_benchmark"],
            queries=SAMPLE_QUERIES,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            use_deterministic=False,
            use_llm_judge=True,
            verify_ssl=config["verify_ssl"],
            jwt_tenant_id=config.get("jwt_tenant_id"),
        )

        _print_results(results)

        for q_id, r in results.items():
            assert r["scoring_method"] in ("llm_judge", "hybrid", "basic", "error"), \
                f"{q_id}: unexpected scoring_method '{r['scoring_method']}'"

    def test_all_scoring_modes(self, config):
        """
        Run benchmark with both deterministic and LLM-judge enabled (hybrid mode).
        Deterministic result takes precedence when correct_sql is available.
        """
        _skip_if_not_configured(config)

        results = run_benchmark(
            benchmark_name=config["timbr_benchmark"],
            queries=SAMPLE_QUERIES,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            use_deterministic=True,
            use_llm_judge=True,
            verify_ssl=config["verify_ssl"],
            jwt_tenant_id=config.get("jwt_tenant_id"),
        )

        _print_results(results)

        assert isinstance(results, dict)

    def test_queries_from_benchmark_options(self, config):
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
            use_deterministic=False,
            use_llm_judge=False,
            verify_ssl=config["verify_ssl"],
            jwt_tenant_id=config.get("jwt_tenant_id"),
        )

        _print_results(results)

        assert isinstance(results, dict)
        assert len(results) > 0, "Expected at least one result from agent questions option"
