"""Integration test: vtimbr end-to-end flow."""

import pytest
from unittest.mock import patch
from datetime import datetime

from langchain_timbr.technical_context.statistics_loader import (
    load_column_statistics,
    StatisticsLoaderConfig,
    ColumnStatistics,
    TopKEntry,
)


@pytest.mark.integration
class TestVtimbrEndToEnd:
    """Full vtimbr path with realistic data shapes."""

    def _mock_run_query(self, query, conn_params, **kwargs):
        """Simulate timbr query responses."""
        if "sys_views" in query:
            return [
                {"view_name": "v_sales", "number_of_rows": 10000},
                {"view_name": "v_inventory", "number_of_rows": 50000},
            ]
        if "sys_properties_statistics" in query and "v_sales" in query:
            return [
                {
                    "property_name": "region",
                    "target_name": "v_sales",
                    "target_type": "view",
                    "distinct_count": 12,
                    "non_null_count": 10000,
                    "stats": '{"top_k": [{"value": "US", "count": 5000}, {"value": "EU", "count": 3000}, {"value": "APAC", "count": 2000}]}',
                    "updated_at": "2024-06-01T12:00:00",
                },
                {
                    "property_name": "amount",
                    "target_name": "v_sales",
                    "target_type": "view",
                    "distinct_count": 8000,
                    "non_null_count": 9500,
                    "stats": '{"min_value": "-50.00", "max_value": "100000.00"}',
                    "updated_at": "2024-06-01T12:00:00",
                },
                {
                    "property_name": "sale_date",
                    "target_name": "v_sales",
                    "target_type": "view",
                    "distinct_count": 365,
                    "non_null_count": 10000,
                    "stats": '{"min_value": "2023-01-01", "max_value": "2023-12-31"}',
                    "updated_at": "2024-06-01T12:00:00",
                },
            ]
        if "SHOW VERSION" in query:
            return [{"id": "v1.0"}]
        return []

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_full_vtimbr_flow(self, mock_run_query, conn_params):
        """Full vtimbr flow: view row count + stats for multiple columns."""
        mock_run_query.side_effect = self._mock_run_query

        columns = [
            {"name": "region", "type": "varchar(50)"},
            {"name": "amount", "type": "decimal(18,2)"},
            {"name": "sale_date", "type": "date"},
            {"name": "no_stats_col", "type": "varchar(100)"},
        ]

        result = load_column_statistics(
            schema="vtimbr",
            table_name="v_sales",
            columns=columns,
            conn_params=conn_params,
        )

        # region: top_k
        assert result["region"].distinct_count == 12
        assert result["region"].top_k is not None
        assert len(result["region"].top_k) == 3
        assert result["region"].top_k[0].value == "US"
        assert result["region"].total_source_rows == 10000

        # amount: min/max decimal
        from decimal import Decimal
        assert result["amount"].min_value == Decimal("-50.00")
        assert result["amount"].max_value == Decimal("100000.00")

        # sale_date: min/max date
        from datetime import date
        assert result["sale_date"].min_value == date(2023, 1, 1)
        assert result["sale_date"].max_value == date(2023, 12, 31)

        # missing column: sentinel
        assert result["no_stats_col"].distinct_count == -1
        assert result["no_stats_col"].top_k is None
