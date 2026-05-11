"""Integration tests for the Technical Context Builder orchestrator."""

import pytest
from unittest.mock import patch, MagicMock

from langchain_timbr.technical_context import build_technical_context
from langchain_timbr.technical_context.config import TechnicalContextConfig
from langchain_timbr.technical_context.types import TechnicalContextResult


def _make_top_k_entry(value, count=10):
    entry = MagicMock()
    entry.value = value
    entry.count = count
    return entry


def _make_column_stats(distinct_count=5, non_null_count=100, top_k_values=None, min_value=None, max_value=None):
    stats = MagicMock()
    stats.distinct_count = distinct_count
    stats.non_null_count = non_null_count
    stats.top_k = [_make_top_k_entry(v) for v in (top_k_values or [])]
    stats.min_value = min_value
    stats.max_value = max_value
    stats.updated_at = None
    stats.approx_union = None
    stats.total_source_rows = non_null_count
    stats.contributing_mappings = []
    return stats


class TestBuildTechnicalContext:
    """Tests for build_technical_context() orchestrator."""

    @patch("langchain_timbr.technical_context.load_column_statistics")
    def test_basic_categorical_column(self, mock_load_stats):
        """A categorical column with known values should produce an annotation."""
        mock_load_stats.return_value = {
            "status": _make_column_stats(
                distinct_count=3,
                top_k_values=["Active", "Inactive", "Pending"],
            ),
        }

        columns = [{"name": "status", "type": "varchar(50)"}]
        result = build_technical_context(
            question="Show active orders",
            columns=columns,
            schema="test_schema",
            concept="orders",
            conn_params={"ontology": "test"},
        )

        assert not result.is_empty
        assert "status" in result.column_annotations
        assert "known values:" in result.column_annotations["status"]
        assert "'Active'" in result.column_annotations["status"]

    @patch("langchain_timbr.technical_context.load_column_statistics")
    def test_matched_value_from_prompt(self, mock_load_stats):
        """When a prompt token matches a known value, it should sort first in known values."""
        mock_load_stats.return_value = {
            "country": _make_column_stats(
                distinct_count=10,
                top_k_values=["USA", "France", "Germany", "Japan", "Brazil"],
            ),
        }

        columns = [{"name": "country", "type": "varchar(100)"}]
        result = build_technical_context(
            question="Show me customers from USA",
            columns=columns,
            schema="test_schema",
            concept="customers",
            conn_params={"ontology": "test"},
        )

        assert "country" in result.column_annotations
        assert "known values:" in result.column_annotations["country"]
        # Matched value 'USA' should appear first in the value list
        annotation = result.column_annotations["country"]
        assert "'USA'" in annotation
        assert annotation.index("USA") < annotation.index("France")

    @patch("langchain_timbr.technical_context.load_column_statistics")
    def test_numeric_column_range(self, mock_load_stats):
        """Numeric columns should get value range annotation."""
        mock_load_stats.return_value = {
            "total": _make_column_stats(
                distinct_count=500,
                non_null_count=1000,
                min_value=0.50,
                max_value=9999.99,
            ),
        }

        columns = [{"name": "total", "type": "decimal(18,2)"}]
        result = build_technical_context(
            question="Show orders over 500",
            columns=columns,
            schema="test_schema",
            concept="orders",
            conn_params={"ontology": "test"},
        )

        assert "total" in result.column_annotations
        assert "value range:" in result.column_annotations["total"]

    @patch("langchain_timbr.technical_context.load_column_statistics")
    def test_empty_question(self, mock_load_stats):
        """Empty question should return empty result."""
        result = build_technical_context(
            question="",
            columns=[{"name": "col", "type": "int"}],
            schema="s",
            concept="c",
            conn_params={},
        )
        assert result.is_empty
        mock_load_stats.assert_not_called()

    @patch("langchain_timbr.technical_context.load_column_statistics")
    def test_empty_columns(self, mock_load_stats):
        """Empty columns should return empty result."""
        result = build_technical_context(
            question="show me stuff",
            columns=[],
            schema="s",
            concept="c",
            conn_params={},
        )
        assert result.is_empty
        mock_load_stats.assert_not_called()

    @patch("langchain_timbr.technical_context.load_column_statistics")
    def test_stats_load_failure_graceful(self, mock_load_stats):
        """Stats loading failure should not crash — return empty result."""
        mock_load_stats.side_effect = Exception("DB connection failed")

        result = build_technical_context(
            question="show orders",
            columns=[{"name": "status", "type": "varchar"}],
            schema="s",
            concept="c",
            conn_params={},
        )

        assert result.is_empty
        assert "error" in result.metadata

    @patch("langchain_timbr.technical_context.load_column_statistics")
    def test_multiple_columns(self, mock_load_stats):
        """Multiple columns with different types should be handled."""
        mock_load_stats.return_value = {
            "status": _make_column_stats(
                distinct_count=3,
                top_k_values=["Active", "Inactive", "Pending"],
            ),
            "amount": _make_column_stats(
                distinct_count=1000,
                non_null_count=5000,
                min_value=1,
                max_value=50000,
            ),
            "created_at": _make_column_stats(
                distinct_count=365,
                min_value="2023-01-01",
                max_value="2023-12-31",
            ),
        }

        columns = [
            {"name": "status", "type": "varchar(20)"},
            {"name": "amount", "type": "decimal(18,2)"},
            {"name": "created_at", "type": "timestamp"},
        ]
        result = build_technical_context(
            question="Show active orders",
            columns=columns,
            schema="test",
            concept="orders",
            conn_params={"ontology": "test"},
        )

        assert not result.is_empty
        assert result.metadata["total_columns"] == 3

    @patch("langchain_timbr.technical_context.load_column_statistics")
    def test_filter_matched_mode(self, mock_load_stats):
        """filter_matched mode without LLM falls back to include_all — all columns annotated."""
        mock_load_stats.return_value = {
            "country": _make_column_stats(
                distinct_count=10,
                top_k_values=["USA", "France"],
            ),
            "status": _make_column_stats(
                distinct_count=3,
                top_k_values=["Active", "Inactive"],
            ),
        }

        columns = [
            {"name": "country", "type": "varchar(100)"},
            {"name": "status", "type": "varchar(50)"},
        ]
        config = TechnicalContextConfig(mode="filter_matched")
        result = build_technical_context(
            question="Show customers from USA",
            columns=columns,
            schema="test",
            concept="customers",
            conn_params={"ontology": "test"},
            config=config,
        )

        # Without LLM, filter_matched falls back to include_all
        # Both columns get annotated
        assert "country" in result.column_annotations
        assert "status" in result.column_annotations
        assert result.metadata.get("effective_mode") == "include_all"


class TestBuildSqlGenerationContextTC:
    """Tests for technical context integration in _build_sql_generation_context()."""

    @patch("langchain_timbr.technical_context.build_technical_context")
    @patch("langchain_timbr.utils.timbr_llm_utils.get_tags")
    @patch("langchain_timbr.utils.timbr_llm_utils.get_concept_properties")
    @patch("langchain_timbr.utils.timbr_llm_utils.get_relationships_description")
    @patch("langchain_timbr.utils.timbr_llm_utils.get_properties_description")
    @patch("langchain_timbr.utils.timbr_llm_utils._get_active_datasource")
    def test_build_technical_context_called_with_question(
        self, mock_ds, mock_props_desc, mock_rels_desc, mock_get_props, mock_tags, mock_build_tc
    ):
        """build_technical_context should be called with the user question."""
        mock_ds.return_value = {"target_type": "spark"}
        mock_props_desc.return_value = {}
        mock_rels_desc.return_value = {}
        mock_get_props.return_value = {
            "columns": [{"name": "status", "col_name": "status", "data_type": "varchar"}],
            "measures": [],
            "relationships": {},
        }
        mock_tags.return_value = {"property_tags": {}}
        mock_build_tc.return_value = TechnicalContextResult(
            column_annotations={"status": "known values: ['Active', 'Inactive']"}
        )

        from langchain_timbr.utils.timbr_llm_utils import _build_sql_generation_context
        result = _build_sql_generation_context(
            question="Show active orders",
            conn_params={"ontology": "test"},
            schema="dtimbr",
            concept="orders",
            concept_metadata={"description": "Orders"},
            graph_depth=1,
            include_tags=None,
            exclude_properties=None,
            db_is_case_sensitive=False,
            max_limit=100,
        )

        mock_build_tc.assert_called_once()
        call_kwargs = mock_build_tc.call_args
        assert call_kwargs[1]["question"] == "Show active orders" or call_kwargs[0][0] == "Show active orders"
        # The annotation should flow into columns_str via _build_columns_str
        assert "statistics:" in result["columns_str"]
        assert "known values:" in result["columns_str"]

    @patch("langchain_timbr.technical_context.build_technical_context")
    @patch("langchain_timbr.utils.timbr_llm_utils.get_tags")
    @patch("langchain_timbr.utils.timbr_llm_utils.get_concept_properties")
    @patch("langchain_timbr.utils.timbr_llm_utils.get_relationships_description")
    @patch("langchain_timbr.utils.timbr_llm_utils.get_properties_description")
    @patch("langchain_timbr.utils.timbr_llm_utils._get_active_datasource")
    def test_tc_failure_does_not_break_context(
        self, mock_ds, mock_props_desc, mock_rels_desc, mock_get_props, mock_tags, mock_build_tc
    ):
        """Technical context failure should not prevent SQL generation context from being built."""
        mock_ds.return_value = {"target_type": "spark"}
        mock_props_desc.return_value = {}
        mock_rels_desc.return_value = {}
        mock_get_props.return_value = {
            "columns": [{"name": "col1", "col_name": "col1", "data_type": "int"}],
            "measures": [],
            "relationships": {},
        }
        mock_tags.return_value = {"property_tags": {}}
        mock_build_tc.side_effect = Exception("TC crashed")

        from langchain_timbr.utils.timbr_llm_utils import _build_sql_generation_context
        result = _build_sql_generation_context(
            question="Show data",
            conn_params={"ontology": "test"},
            schema="dtimbr",
            concept="t",
            concept_metadata={},
            graph_depth=1,
            include_tags=None,
            exclude_properties=None,
            db_is_case_sensitive=False,
            max_limit=100,
        )

        # Should still return a valid context dict
        assert "columns_str" in result
        assert "col1" in result["columns_str"]

    @patch("langchain_timbr.technical_context.build_technical_context")
    @patch("langchain_timbr.utils.timbr_llm_utils.get_tags")
    @patch("langchain_timbr.utils.timbr_llm_utils.get_concept_properties")
    @patch("langchain_timbr.utils.timbr_llm_utils.get_relationships_description")
    @patch("langchain_timbr.utils.timbr_llm_utils.get_properties_description")
    @patch("langchain_timbr.utils.timbr_llm_utils._get_active_datasource")
    def test_empty_annotations_no_technical_context_in_output(
        self, mock_ds, mock_props_desc, mock_rels_desc, mock_get_props, mock_tags, mock_build_tc
    ):
        """When TC returns no annotations, columns_str should not contain technical context."""
        mock_ds.return_value = {"target_type": "spark"}
        mock_props_desc.return_value = {}
        mock_rels_desc.return_value = {}
        mock_get_props.return_value = {
            "columns": [{"name": "id", "col_name": "id", "data_type": "bigint"}],
            "measures": [],
            "relationships": {},
        }
        mock_tags.return_value = {"property_tags": {}}
        mock_build_tc.return_value = TechnicalContextResult(column_annotations={})

        from langchain_timbr.utils.timbr_llm_utils import _build_sql_generation_context
        result = _build_sql_generation_context(
            question="Show IDs",
            conn_params={"ontology": "test"},
            schema="dtimbr",
            concept="t",
            concept_metadata={},
            graph_depth=1,
            include_tags=None,
            exclude_properties=None,
            db_is_case_sensitive=False,
            max_limit=100,
        )

        assert "technical context" not in result["columns_str"]


class TestLlmWiringInOrchestrator:
    """Tests for LLM integration in build_technical_context()."""

    @patch("langchain_timbr.technical_context.load_column_statistics")
    @patch("langchain_timbr.technical_context.extract_candidates_with_llm")
    def test_include_all_mode_never_calls_llm(self, mock_llm_extract, mock_load_stats):
        """include_all mode should never call LLM even when llm is provided."""
        mock_load_stats.return_value = {
            "status": _make_column_stats(distinct_count=3, top_k_values=["Active", "Inactive"]),
        }
        llm = MagicMock()

        config = TechnicalContextConfig(mode="include_all")
        result = build_technical_context(
            question="Show orders",
            columns=[{"name": "status", "type": "varchar"}],
            schema="s",
            concept="c",
            conn_params={},
            config=config,
            llm=llm,
        )

        mock_llm_extract.assert_not_called()
        assert not result.is_empty
        assert result.metadata.get("llm_used") is False

    @patch("langchain_timbr.technical_context.load_column_statistics")
    @patch("langchain_timbr.technical_context.extract_candidates_with_llm")
    def test_filter_matched_mode_calls_llm(self, mock_llm_extract, mock_load_stats):
        """filter_matched mode should call LLM when llm is provided."""
        mock_load_stats.return_value = {
            "status": _make_column_stats(distinct_count=3, top_k_values=["Active", "Inactive"]),
        }
        mock_llm_extract.return_value = ["Active"]
        llm = MagicMock()

        config = TechnicalContextConfig(mode="filter_matched")
        result = build_technical_context(
            question="Show active orders",
            columns=[{"name": "status", "type": "varchar"}],
            schema="s",
            concept="c",
            conn_params={},
            config=config,
            llm=llm,
        )

        mock_llm_extract.assert_called_once()
        assert result.metadata.get("llm_used") is True
        assert "status" in result.column_annotations

    @patch("langchain_timbr.technical_context.load_column_statistics")
    @patch("langchain_timbr.technical_context.extract_candidates_with_llm")
    def test_filter_matched_no_llm_heuristic_only(self, mock_llm_extract, mock_load_stats):
        """filter_matched without llm falls back to include_all (degraded)."""
        mock_load_stats.return_value = {
            "status": _make_column_stats(distinct_count=3, top_k_values=["Active", "Inactive"]),
        }
        config = TechnicalContextConfig(mode="filter_matched")
        result = build_technical_context(
            question="Show active orders",
            columns=[{"name": "status", "type": "varchar"}],
            schema="s",
            concept="c",
            conn_params={},
            config=config,
            llm=None,
        )

        mock_llm_extract.assert_not_called()
        assert result.metadata.get("llm_used") is False
        # Falls back to include_all when no LLM available
        assert result.metadata.get("effective_mode") == "include_all"

    @patch("langchain_timbr.technical_context.load_column_statistics")
    @patch("langchain_timbr.technical_context.extract_candidates_with_llm")
    @patch("langchain_timbr.technical_context.estimate_include_all_cost")
    def test_auto_under_budget_acts_as_include_all(self, mock_cost, mock_llm_extract, mock_load_stats):
        """auto mode under budget should act as include_all (no LLM)."""
        mock_load_stats.return_value = {
            "status": _make_column_stats(distinct_count=3, top_k_values=["Active", "Inactive"]),
        }
        mock_cost.return_value = 100  # Well under default max_tokens=3000
        llm = MagicMock()

        config = TechnicalContextConfig(mode="auto")
        result = build_technical_context(
            question="Show orders",
            columns=[{"name": "status", "type": "varchar"}],
            schema="s",
            concept="c",
            conn_params={},
            config=config,
            llm=llm,
        )

        mock_llm_extract.assert_not_called()
        assert result.metadata.get("effective_mode") == "include_all"
        assert result.metadata.get("llm_used") is False

    @patch("langchain_timbr.technical_context.load_column_statistics")
    @patch("langchain_timbr.technical_context.extract_candidates_with_llm")
    @patch("langchain_timbr.technical_context.estimate_include_all_cost")
    def test_auto_over_budget_with_llm_falls_back_to_filter_matched(self, mock_cost, mock_llm_extract, mock_load_stats):
        """auto mode over budget with llm should fallback to filter_matched."""
        mock_load_stats.return_value = {
            "status": _make_column_stats(distinct_count=3, top_k_values=["Active", "Inactive"]),
        }
        mock_cost.return_value = 5000  # Over default max_tokens=3000
        mock_llm_extract.return_value = ["Active"]
        llm = MagicMock()

        config = TechnicalContextConfig(mode="auto")
        result = build_technical_context(
            question="Show active orders",
            columns=[{"name": "status", "type": "varchar"}],
            schema="s",
            concept="c",
            conn_params={},
            config=config,
            llm=llm,
        )

        mock_llm_extract.assert_called_once()
        assert result.metadata.get("effective_mode") == "filter_matched"
        assert result.metadata.get("llm_used") is True

    @patch("langchain_timbr.technical_context.load_column_statistics")
    @patch("langchain_timbr.technical_context.extract_candidates_with_llm")
    @patch("langchain_timbr.technical_context.estimate_include_all_cost")
    def test_auto_over_budget_no_llm_uses_heuristic(self, mock_cost, mock_llm_extract, mock_load_stats):
        """auto mode over budget without llm should use include_all (degraded path)."""
        mock_load_stats.return_value = {
            "status": _make_column_stats(distinct_count=3, top_k_values=["Active", "Inactive"]),
        }
        mock_cost.return_value = 5000  # Over budget

        config = TechnicalContextConfig(mode="auto")
        result = build_technical_context(
            question="Show active orders",
            columns=[{"name": "status", "type": "varchar"}],
            schema="s",
            concept="c",
            conn_params={},
            config=config,
            llm=None,
        )

        mock_llm_extract.assert_not_called()
        # Degraded path: can't escalate without LLM, falls back to include_all
        assert result.metadata.get("effective_mode") == "include_all"
        assert result.metadata.get("llm_used") is False

    @patch("langchain_timbr.technical_context.load_column_statistics")
    @patch("langchain_timbr.technical_context.extract_candidates_with_llm")
    def test_llm_failure_graceful_fallback(self, mock_llm_extract, mock_load_stats):
        """LLM extraction failure should not crash — falls back to heuristic."""
        mock_load_stats.return_value = {
            "status": _make_column_stats(distinct_count=3, top_k_values=["Active", "Inactive"]),
        }
        mock_llm_extract.side_effect = RuntimeError("LLM crashed")
        llm = MagicMock()

        config = TechnicalContextConfig(mode="filter_matched")
        result = build_technical_context(
            question="Show active orders",
            columns=[{"name": "status", "type": "varchar"}],
            schema="s",
            concept="c",
            conn_params={},
            config=config,
            llm=llm,
        )

        # Should still work (heuristic matching finds "Active")
        assert "status" in result.column_annotations

    @patch("langchain_timbr.technical_context.build_technical_context")
    @patch("langchain_timbr.utils.timbr_llm_utils.get_tags")
    @patch("langchain_timbr.utils.timbr_llm_utils.get_concept_properties")
    @patch("langchain_timbr.utils.timbr_llm_utils.get_relationships_description")
    @patch("langchain_timbr.utils.timbr_llm_utils.get_properties_description")
    @patch("langchain_timbr.utils.timbr_llm_utils._get_active_datasource")
    def test_build_sql_generation_context_passes_llm(
        self, mock_ds, mock_props_desc, mock_rels_desc, mock_get_props, mock_tags, mock_build_tc
    ):
        """_build_sql_generation_context should pass llm to build_technical_context."""
        mock_ds.return_value = {"target_type": "spark"}
        mock_props_desc.return_value = {}
        mock_rels_desc.return_value = {}
        mock_get_props.return_value = {
            "columns": [{"name": "status", "col_name": "status", "data_type": "varchar"}],
            "measures": [],
            "relationships": {},
        }
        mock_tags.return_value = {"property_tags": {}}
        mock_build_tc.return_value = TechnicalContextResult(column_annotations={})
        fake_llm = MagicMock()

        from langchain_timbr.utils.timbr_llm_utils import _build_sql_generation_context
        _build_sql_generation_context(
            question="Show data",
            conn_params={"ontology": "test"},
            schema="dtimbr",
            concept="t",
            concept_metadata={},
            graph_depth=1,
            include_tags=None,
            exclude_properties=None,
            db_is_case_sensitive=False,
            max_limit=100,
            llm=fake_llm,
        )

        mock_build_tc.assert_called_once()
        call_kwargs = mock_build_tc.call_args[1]
        assert call_kwargs.get("llm") is fake_llm
