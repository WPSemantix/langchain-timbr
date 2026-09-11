"""Unit tests for the technical-context budget at the real SQL-generation call site.

`_build_sql_generation_context` runs unmodified; only its metadata lookups and the
statistics loader are stubbed, so these cover the path a chain actually takes:
budget -> config -> annotation -> the column block of the prompt.

Two properties matter here and cannot be seen from the trimmer alone:
- a budget the caller considers generous must not end up with no statistics at all;
- when the pass does fail, it stays non-fatal but becomes visible.
"""

import logging
import re

import pytest

import langchain_timbr.technical_context as technical_context
import langchain_timbr.utils.timbr_llm_utils as llm_utils
from langchain_timbr.technical_context.config import TechnicalContextConfig
from langchain_timbr.technical_context.statistics_loader.types import (
    ColumnStatistics,
    TopKEntry,
)

COLUMNS = ("region", "product_line", "status_code", "channel", "segment", "site", "reason")
VALUES_PER_COLUMN = 200


def _statistics() -> dict:
    """Value domains large enough that the full set cannot fit any budget under the cap,
    so the budget always has to choose how many values to keep."""
    words = ("northern", "central", "coastal", "upper", "lower", "greater")
    stats = {}
    for col_idx, name in enumerate(COLUMNS):
        values = [
            f"{words[(i + col_idx) % len(words)].title()} {name.title()} Division {i:03d}"
            for i in range(VALUES_PER_COLUMN)
        ]
        stats[name] = ColumnStatistics(
            distinct_count=VALUES_PER_COLUMN + 50,
            non_null_count=1_000_000,
            total_source_rows=1_000_000,
            top_k=[TopKEntry(value=v, count=1000 - i) for i, v in enumerate(values)],
        )
    return stats


@pytest.fixture
def offline_context(monkeypatch):
    """Build contexts with no network: every metadata lookup and the statistics loader
    are replaced, the rest of the function is the real thing."""
    columns = [{"name": c, "col_name": c, "data_type": "string"} for c in COLUMNS]
    stats = _statistics()

    monkeypatch.setattr(
        llm_utils, "_get_active_datasource", lambda conn_params: {"target_type": "databricks"},
    )
    monkeypatch.setattr(llm_utils, "get_properties_description", lambda conn_params: {})
    monkeypatch.setattr(llm_utils, "get_relationships_description", lambda conn_params: {})
    monkeypatch.setattr(
        llm_utils,
        "get_concept_properties",
        lambda **kw: {"columns": [dict(c) for c in columns], "measures": [], "relationships": {}},
    )
    monkeypatch.setattr(llm_utils, "get_tags", lambda **kw: {"property_tags": {}})
    monkeypatch.setattr(
        technical_context,
        "load_column_statistics",
        lambda **kw: {c["name"]: stats[c["name"]] for c in kw["columns"] if c["name"] in stats},
    )

    def build(budget, status_sink=None, **overrides):
        kwargs = dict(
            question="Which channels had the most activity last month?",
            conn_params={},
            schema="vtimbr",
            concept="reporting_cube",
            concept_metadata={},
            graph_depth=1,
            include_tags=None,
            exclude_properties=[],
            db_is_case_sensitive=False,
            max_limit=100,
            llm=None,
            enable_technical_context=True,
            technical_context_mode="include_all",
            technical_context_max_tokens=budget,
            technical_context_properties=list(COLUMNS),
            metadata_context_mode="static",
            status_sink=status_sink,
        )
        kwargs.update(overrides)
        return llm_utils._build_sql_generation_context(**kwargs)

    return build


def _values_in_prompt(context: dict) -> int:
    """Count the values the column block actually carries, the way a reader of the prompt
    would count them."""
    return sum(
        len(re.findall(r"'[^']*'", listing))
        for listing in re.findall(r"known values: \[(.*?)\]", context["columns_str"])
    )


def _columns_with_values(context: dict) -> int:
    return len(re.findall(r"known values: \[", context["columns_str"]))


class TestBudgetReachesThePrompt:
    @pytest.mark.parametrize("budget", (4000, 9000, 15000))
    def test_columns_are_annotated(self, offline_context, budget):
        context = offline_context(budget)
        assert _columns_with_values(context) == len(COLUMNS)
        assert _values_in_prompt(context) > 0

    def test_a_budget_at_or_above_the_cap_still_annotates(self, offline_context):
        """A budget the caller thinks is generous must not read as "no statistics". This is
        the whole point of clamping instead of rejecting."""
        for budget in (
            TechnicalContextConfig.safety_ceiling,
            TechnicalContextConfig.safety_ceiling + 1,
            TechnicalContextConfig.safety_ceiling * 3,
        ):
            context = offline_context(budget)
            assert _columns_with_values(context) == len(COLUMNS), f"budget={budget}"
            assert _values_in_prompt(context) > 0, f"budget={budget}"

    def test_a_bigger_budget_puts_more_values_in_the_prompt(self, offline_context):
        counts = [_values_in_prompt(offline_context(b)) for b in (4000, 9000, 14000)]
        assert counts[0] < counts[1] < counts[2]

    def test_the_prompt_still_names_the_true_cardinality(self, offline_context):
        """Trimmed columns keep telling the model how many values really exist."""
        context = offline_context(4000)
        assert f"({VALUES_PER_COLUMN + 50} distinct total)" in context["columns_str"]


class TestDisabledByBudget:
    def test_non_positive_budget_disables_without_failing(self, offline_context, caplog):
        with caplog.at_level(logging.INFO):
            context = offline_context(0)
        assert _columns_with_values(context) == 0
        assert context["columns_str"], "the columns themselves must still reach the prompt"
        assert "Technical context disabled" in caplog.text


class TestFailureIsVisible:
    def test_failure_is_logged_and_reported_but_not_fatal(
        self, offline_context, monkeypatch, caplog,
    ):
        def explode(**kwargs):
            raise RuntimeError("statistics backend unavailable")

        monkeypatch.setattr(technical_context, "build_technical_context", explode)
        status: dict = {}
        with caplog.at_level(logging.WARNING):
            context = offline_context(9000, status_sink=status)

        # Non-fatal: the prompt is still built, just without statistics.
        assert context["columns_str"]
        assert _columns_with_values(context) == 0
        # Visible: named in the log and handed back to the caller.
        assert "Technical context skipped" in caplog.text
        assert status.get("technical_context_degraded") is True
        assert "statistics backend unavailable" in status["technical_context_error"]

    def test_no_error_is_reported_on_the_happy_path(self, offline_context):
        status: dict = {}
        offline_context(9000, status_sink=status)
        assert "technical_context_error" not in status
