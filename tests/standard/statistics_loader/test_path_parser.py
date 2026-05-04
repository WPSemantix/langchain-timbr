"""Tests for path_parser module."""

import pytest

from langchain_timbr.technical_context.statistics_loader.path_parser import (
    parse_column_path,
    ColumnPathParseError,
)


class TestParseColumnPath:
    """Test column path parsing."""

    def test_direct_column(self):
        """Simple column with no relationship hops."""
        path = parse_column_path("customer_id", "customer")
        assert path.raw == "customer_id"
        assert path.hops == []
        assert path.final_property == "customer_id"
        assert path.owning_concept == "customer"

    def test_single_hop(self):
        """Single relationship hop: orders[order].total"""
        path = parse_column_path("orders[order].total", "customer")
        assert path.raw == "orders[order].total"
        assert path.hops == [("orders", "order")]
        assert path.final_property == "total"
        assert path.owning_concept == "order"

    def test_multi_hop(self):
        """Multi-hop: orders[order].items[order_item].quantity"""
        path = parse_column_path("orders[order].items[order_item].quantity", "customer")
        assert path.raw == "orders[order].items[order_item].quantity"
        assert path.hops == [("orders", "order"), ("items", "order_item")]
        assert path.final_property == "quantity"
        assert path.owning_concept == "order_item"

    def test_three_hops(self):
        """Three relationship hops."""
        col = "a[b].c[d].e[f].prop"
        path = parse_column_path(col, "root")
        assert path.hops == [("a", "b"), ("c", "d"), ("e", "f")]
        assert path.final_property == "prop"
        assert path.owning_concept == "f"

    def test_measure_prefix_direct(self):
        """Direct column with measure. prefix."""
        path = parse_column_path("measure.total_sales", "customer")
        assert path.raw == "measure.total_sales"
        assert path.hops == []
        assert path.final_property == "measure.total_sales"
        assert path.owning_concept == "customer"

    def test_underscore_in_names(self):
        """Underscores in relationship and concept names."""
        path = parse_column_path("has_orders[sales_order].line_total", "customer")
        assert path.hops == [("has_orders", "sales_order")]
        assert path.final_property == "line_total"
        assert path.owning_concept == "sales_order"

    def test_empty_brackets_raises(self):
        """Empty brackets should raise ColumnPathParseError."""
        with pytest.raises(ColumnPathParseError, match="Empty brackets"):
            parse_column_path("orders[].total", "customer")

    def test_unbalanced_opening_bracket(self):
        """Unbalanced opening bracket."""
        with pytest.raises(ColumnPathParseError, match="Unbalanced opening bracket"):
            parse_column_path("orders[order.total", "customer")

    def test_unbalanced_closing_bracket(self):
        """Unbalanced closing bracket."""
        with pytest.raises(ColumnPathParseError, match="Unbalanced closing bracket"):
            parse_column_path("orders]order[.total", "customer")

    def test_empty_column_name(self):
        """Empty column name should raise."""
        with pytest.raises(ColumnPathParseError, match="Empty column name"):
            parse_column_path("", "customer")

    def test_whitespace_only(self):
        """Whitespace-only column name should raise."""
        with pytest.raises(ColumnPathParseError, match="Empty column name"):
            parse_column_path("   ", "customer")

    def test_concept_with_numbers(self):
        """Concept names with numbers."""
        path = parse_column_path("rel1[concept2].prop3", "root")
        assert path.hops == [("rel1", "concept2")]
        assert path.final_property == "prop3"
        assert path.owning_concept == "concept2"

    def test_raw_preserved_verbatim(self):
        """The raw field preserves the exact input string."""
        col = "orders[order].items[order_item].quantity"
        path = parse_column_path(col, "customer")
        assert path.raw == col
