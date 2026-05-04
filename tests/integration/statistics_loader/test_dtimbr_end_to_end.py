"""Integration test: dtimbr end-to-end flow."""

import pytest
from unittest.mock import patch
from datetime import datetime
from decimal import Decimal

from langchain_timbr.technical_context.statistics_loader import (
    load_column_statistics,
    StatisticsLoaderConfig,
)


@pytest.mark.integration
class TestDtimbrEndToEnd:
    """Full dtimbr path with direct + derived + logic mappings."""

    def _mock_run_query(self, query, conn_params, **kwargs):
        """Simulate timbr query responses for a complex ontology."""
        if "SHOW VERSION" in query:
            return [{"id": "v2.0"}]

        if "sys_ontology" in query:
            return [
                {"concept": "customer", "inheritance": "person", "query": None},
                {"concept": "person", "inheritance": "", "query": None},
                {"concept": "order", "inheritance": "", "query": None},
                {"concept": "order_item", "inheritance": "", "query": None},
                {"concept": "active_customer", "inheritance": "", "query": "SELECT * FROM dtimbr.customer WHERE status = 'active'"},
            ]

        if "sys_concept_mappings" in query:
            return [
                {"concept": "customer", "mapping_name": "map_cust_main", "number_of_rows": 5000},
                {"concept": "customer", "mapping_name": "map_cust_archive", "number_of_rows": 2000},
                {"concept": "person", "mapping_name": "map_person", "number_of_rows": 10000},
                {"concept": "order", "mapping_name": "map_order", "number_of_rows": 50000},
                {"concept": "order_item", "mapping_name": "map_item_a", "number_of_rows": 100000},
                {"concept": "order_item", "mapping_name": "map_item_b", "number_of_rows": 50000},
            ]

        if "sys_views" in query:
            return []

        if "sys_properties_statistics" in query:
            # Return stats for various properties across mappings
            return [
                # customer_name from both customer mappings
                {
                    "property_name": "customer_name",
                    "target_name": "map_cust_main",
                    "target_type": "mapping",
                    "distinct_count": 4500,
                    "non_null_count": 5000,
                    "stats": '{"top_k": [{"value": "Acme Corp", "count": 50}, {"value": "Global Inc", "count": 30}]}',
                    "updated_at": "2024-05-01T10:00:00",
                },
                {
                    "property_name": "customer_name",
                    "target_name": "map_cust_archive",
                    "target_type": "mapping",
                    "distinct_count": 1800,
                    "non_null_count": 2000,
                    "stats": '{"top_k": [{"value": "Acme Corp", "count": 20}, {"value": "Old LLC", "count": 15}]}',
                    "updated_at": "2024-03-01T10:00:00",
                },
                # total from order mapping
                {
                    "property_name": "total",
                    "target_name": "map_order",
                    "target_type": "mapping",
                    "distinct_count": 30000,
                    "non_null_count": 50000,
                    "stats": '{"min_value": "0.01", "max_value": "99999.99"}',
                    "updated_at": "2024-06-01T10:00:00",
                },
                # quantity from both order_item mappings
                {
                    "property_name": "quantity",
                    "target_name": "map_item_a",
                    "target_type": "mapping",
                    "distinct_count": 50,
                    "non_null_count": 100000,
                    "stats": '{"min_value": "1", "max_value": "500"}',
                    "updated_at": "2024-04-01T10:00:00",
                },
                {
                    "property_name": "quantity",
                    "target_name": "map_item_b",
                    "target_type": "mapping",
                    "distinct_count": 30,
                    "non_null_count": 50000,
                    "stats": '{"min_value": "1", "max_value": "1000"}',
                    "updated_at": "2024-05-01T10:00:00",
                },
            ]

        return []

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_full_dtimbr_flow(self, mock_run_query, conn_params):
        """Full dtimbr flow with multi-hop columns and cross-mapping merge."""
        mock_run_query.side_effect = self._mock_run_query

        columns = [
            {"name": "customer_name", "type": "varchar(255)"},
            {"name": "orders[order].total", "type": "decimal(18,2)"},
            {"name": "orders[order].items[order_item].quantity", "type": "int"},
        ]

        result = load_column_statistics(
            schema="dtimbr",
            table_name="customer",
            columns=columns,
            conn_params=conn_params,
        )

        # customer_name: merged from 2 customer mappings
        assert "customer_name" in result
        stats = result["customer_name"]
        assert stats.approx_union is True
        assert stats.distinct_count == 4500 + 1800  # upper bound sum
        assert stats.top_k is not None
        # Acme Corp count merged: 50 + 20 = 70
        topk_dict = {e.value: e.count for e in stats.top_k}
        assert topk_dict["Acme Corp"] == 70
        assert topk_dict["Global Inc"] == 30
        assert topk_dict["Old LLC"] == 15

        # orders[order].total: from order mapping (single)
        assert "orders[order].total" in result
        stats = result["orders[order].total"]
        assert stats.min_value == Decimal("0.01")
        assert stats.max_value == Decimal("99999.99")
        assert stats.total_source_rows == 50000

        # orders[order].items[order_item].quantity: merged from 2 item mappings
        assert "orders[order].items[order_item].quantity" in result
        stats = result["orders[order].items[order_item].quantity"]
        assert stats.min_value == 1  # min(1, 1)
        assert stats.max_value == 1000  # max(500, 1000)
        assert stats.approx_union is True
        assert stats.total_source_rows == 150000  # 100000 + 50000
