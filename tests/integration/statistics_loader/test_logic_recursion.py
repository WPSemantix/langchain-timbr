"""Integration test: logic query recursion and cycle detection."""

import pytest
from unittest.mock import patch
from datetime import datetime

from langchain_timbr.technical_context.statistics_loader import (
    load_column_statistics,
    StatisticsLoaderConfig,
)


@pytest.mark.integration
class TestLogicRecursion:
    """Test logic-query resolution chains and cycle detection."""

    def test_single_logic_hop(self, conn_params):
        """A→logic→B: A gets B's mappings via logic query."""
        def mock_query(query, conn_params, **kwargs):
            if "SHOW VERSION" in query:
                return [{"id": "v1.0"}]
            if "sys_ontology" in query:
                return [
                    {"concept": "logic_a", "inheritance": "", "query": "SELECT * FROM dtimbr.base_b WHERE x = 1"},
                    {"concept": "base_b", "inheritance": "", "query": None},
                ]
            if "sys_concept_mappings" in query:
                return [
                    {"concept": "base_b", "mapping_name": "map_b", "number_of_rows": 1000},
                ]
            if "sys_views" in query:
                return []
            if "sys_properties_statistics" in query:
                return [
                    {
                        "property_name": "col_x",
                        "target_name": "map_b",
                        "target_type": "mapping",
                        "distinct_count": 100,
                        "non_null_count": 1000,
                        "stats": '{"min_value": "1", "max_value": "100"}',
                        "updated_at": "2024-01-01T00:00:00",
                    }
                ]
            return []

        with patch("langchain_timbr.utils.timbr_utils.run_query", side_effect=mock_query):
            result = load_column_statistics(
                schema="dtimbr",
                table_name="logic_a",
                columns=[{"name": "col_x", "type": "int"}],
                conn_params=conn_params,
            )

        assert result["col_x"].min_value == 1
        assert result["col_x"].max_value == 100

    def test_two_logic_hops(self, conn_params):
        """A→logic→B→logic→C: A ultimately gets C's mappings."""
        def mock_query(query, conn_params, **kwargs):
            if "SHOW VERSION" in query:
                return [{"id": "v1.0"}]
            if "sys_ontology" in query:
                return [
                    {"concept": "concept_a", "inheritance": "", "query": "SELECT * FROM dtimbr.concept_b WHERE x = 1"},
                    {"concept": "concept_b", "inheritance": "", "query": "SELECT * FROM dtimbr.concept_c WHERE y = 2"},
                    {"concept": "concept_c", "inheritance": "", "query": None},
                ]
            if "sys_concept_mappings" in query:
                return [
                    {"concept": "concept_c", "mapping_name": "map_c", "number_of_rows": 5000},
                ]
            if "sys_views" in query:
                return []
            if "sys_properties_statistics" in query:
                return [
                    {
                        "property_name": "col_y",
                        "target_name": "map_c",
                        "target_type": "mapping",
                        "distinct_count": 500,
                        "non_null_count": 5000,
                        "stats": '{"top_k": [{"value": "DONE", "count": 3000}]}',
                        "updated_at": "2024-01-01T00:00:00",
                    }
                ]
            return []

        with patch("langchain_timbr.utils.timbr_utils.run_query", side_effect=mock_query):
            result = load_column_statistics(
                schema="dtimbr",
                table_name="concept_a",
                columns=[{"name": "col_y", "type": "varchar(50)"}],
                conn_params=conn_params,
            )

        assert result["col_y"].top_k is not None
        assert result["col_y"].top_k[0].value == "DONE"

    def test_cycle_a_to_a(self, conn_params):
        """A→logic→A: Self-referencing cycle terminates gracefully."""
        def mock_query(query, conn_params, **kwargs):
            if "SHOW VERSION" in query:
                return [{"id": "v1.0"}]
            if "sys_ontology" in query:
                return [
                    {"concept": "self_ref", "inheritance": "", "query": "SELECT * FROM dtimbr.self_ref WHERE x = 1"},
                ]
            if "sys_concept_mappings" in query:
                return [
                    {"concept": "self_ref", "mapping_name": "map_self", "number_of_rows": 100},
                ]
            if "sys_views" in query:
                return []
            if "sys_properties_statistics" in query:
                return [
                    {
                        "property_name": "col_z",
                        "target_name": "map_self",
                        "target_type": "mapping",
                        "distinct_count": 10,
                        "non_null_count": 100,
                        "stats": '{"top_k": [{"value": "A", "count": 50}]}',
                        "updated_at": "2024-01-01T00:00:00",
                    }
                ]
            return []

        with patch("langchain_timbr.utils.timbr_utils.run_query", side_effect=mock_query):
            # Should not hang or raise
            result = load_column_statistics(
                schema="dtimbr",
                table_name="self_ref",
                columns=[{"name": "col_z", "type": "varchar(10)"}],
                conn_params=conn_params,
            )

        # Should still get the direct mapping stats
        assert result["col_z"].distinct_count == 10

    def test_mutual_cycle(self, conn_params):
        """A→logic→B→logic→A: Mutual cycle terminates."""
        def mock_query(query, conn_params, **kwargs):
            if "SHOW VERSION" in query:
                return [{"id": "v1.0"}]
            if "sys_ontology" in query:
                return [
                    {"concept": "cycle_a", "inheritance": "", "query": "SELECT * FROM dtimbr.cycle_b WHERE x = 1"},
                    {"concept": "cycle_b", "inheritance": "", "query": "SELECT * FROM dtimbr.cycle_a WHERE y = 2"},
                ]
            if "sys_concept_mappings" in query:
                return [
                    {"concept": "cycle_a", "mapping_name": "map_a", "number_of_rows": 100},
                    {"concept": "cycle_b", "mapping_name": "map_b", "number_of_rows": 200},
                ]
            if "sys_views" in query:
                return []
            if "sys_properties_statistics" in query:
                return [
                    {
                        "property_name": "col_m",
                        "target_name": "map_a",
                        "target_type": "mapping",
                        "distinct_count": 10,
                        "non_null_count": 100,
                        "stats": '{"top_k": [{"value": "X", "count": 50}]}',
                        "updated_at": "2024-01-01T00:00:00",
                    },
                    {
                        "property_name": "col_m",
                        "target_name": "map_b",
                        "target_type": "mapping",
                        "distinct_count": 20,
                        "non_null_count": 200,
                        "stats": '{"top_k": [{"value": "Y", "count": 100}]}',
                        "updated_at": "2024-02-01T00:00:00",
                    },
                ]
            return []

        with patch("langchain_timbr.utils.timbr_utils.run_query", side_effect=mock_query):
            result = load_column_statistics(
                schema="dtimbr",
                table_name="cycle_a",
                columns=[{"name": "col_m", "type": "varchar(10)"}],
                conn_params=conn_params,
            )

        # cycle_a should get both map_a (direct) and map_b (logic from cycle_b)
        assert result["col_m"].distinct_count == 30  # 10 + 20
        assert result["col_m"].approx_union is True
