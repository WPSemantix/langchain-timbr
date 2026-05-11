"""Integration test: failure modes."""

import pytest
from unittest.mock import patch
from datetime import datetime

from langchain_timbr.technical_context.statistics_loader import (
    load_column_statistics,
    StatisticsLoaderConfig,
)
from langchain_timbr.technical_context.statistics_loader.path_parser import ColumnPathParseError


@pytest.mark.integration
class TestFailureModes:
    """Test graceful degradation on various failures."""

    def _base_mock(self, query, conn_params, **kwargs):
        """Base mock returning minimal valid responses."""
        if "SHOW VERSION" in query:
            return [{"id": "v1.0"}]
        if "sys_ontology" in query:
            return [{"concept": "test", "inheritance": "", "query": None}]
        if "sys_concept_mappings" in query:
            return [{"concept": "test", "mapping_name": "map_test", "number_of_rows": 100}]
        if "sys_views" in query:
            return [{"view_name": "v_test", "number_of_rows": 500}]
        if "sys_properties_statistics" in query:
            return []
        return []

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_malformed_stats_json(self, mock_run_query, conn_params):
        """Malformed stats JSON degrades to sentinel, never raises."""
        def mock_query(query, conn_params, **kwargs):
            if "sys_properties_statistics" in query:
                return [
                    {
                        "property_name": "col_a",
                        "target_name": "map_test",
                        "target_type": "mapping",
                        "distinct_count": 10,
                        "non_null_count": 100,
                        "stats": "{invalid json!!!!}",
                        "updated_at": "2024-01-01T00:00:00",
                    }
                ]
            return self._base_mock(query, conn_params, **kwargs)

        mock_run_query.side_effect = mock_query

        result = load_column_statistics(
            schema="dtimbr",
            table_name="test",
            columns=[{"name": "col_a", "type": "int"}],
            conn_params=conn_params,
        )

        # Should still have a result, with parsed counts but no top_k/min_max
        assert "col_a" in result
        assert result["col_a"].distinct_count == 10
        assert result["col_a"].top_k is None
        assert result["col_a"].min_value is None

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_missing_concept_in_ontology(self, mock_run_query, conn_params):
        """Concept not in ontology produces sentinel for all columns."""
        def mock_query(query, conn_params, **kwargs):
            if "sys_ontology" in query:
                return []  # empty ontology
            return self._base_mock(query, conn_params, **kwargs)

        mock_run_query.side_effect = mock_query

        result = load_column_statistics(
            schema="dtimbr",
            table_name="nonexistent",
            columns=[{"name": "col_a", "type": "int"}],
            conn_params=conn_params,
        )

        assert result["col_a"].distinct_count == -1

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_db_error_propagates(self, mock_run_query, conn_params):
        """Database errors propagate — loader cannot succeed without DB."""
        mock_run_query.side_effect = Exception("Connection refused")

        with pytest.raises(Exception, match="Connection refused"):
            load_column_statistics(
                schema="dtimbr",
                table_name="test",
                columns=[{"name": "col_a", "type": "int"}],
                conn_params=conn_params,
            )

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_malformed_column_path_sentinel(self, mock_run_query, conn_params):
        """Malformed column path produces sentinel with default config."""
        mock_run_query.side_effect = self._base_mock

        result = load_column_statistics(
            schema="dtimbr",
            table_name="test",
            columns=[{"name": "bad[.col", "type": "int"}],
            conn_params=conn_params,
        )

        assert "bad[.col" in result
        assert result["bad[.col"].distinct_count == -1

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_malformed_column_path_raise(self, mock_run_query, conn_params):
        """Malformed column path raises with on_missing_stats='raise'."""
        mock_run_query.side_effect = self._base_mock

        config = StatisticsLoaderConfig(on_missing_stats="raise")
        with pytest.raises(ColumnPathParseError):
            load_column_statistics(
                schema="dtimbr",
                table_name="test",
                columns=[{"name": "bad[.col", "type": "int"}],
                conn_params=conn_params,
                config=config,
            )
