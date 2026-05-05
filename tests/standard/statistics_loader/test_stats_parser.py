"""Tests for stats_parser module."""

import pytest
from decimal import Decimal
from datetime import date, datetime

from langchain_timbr.technical_context.statistics_loader.stats_parser import (
    parse_stats_json,
)
from langchain_timbr.technical_context.statistics_loader.types import TopKEntry


class TestParseStatsJson:
    """Test stats JSON parsing."""

    def test_top_k_basic(self):
        """Basic top_k JSON parsing."""
        stats = '{"top_k": [{"value": "COMPLETE", "count": 29754}, {"value": "PENDING", "count": 1500}]}'
        top_k, min_val, max_val = parse_stats_json(stats)

        assert top_k is not None
        assert len(top_k) == 2
        assert top_k[0].value == "COMPLETE"
        assert top_k[0].count == 29754
        assert top_k[1].value == "PENDING"
        assert top_k[1].count == 1500
        assert min_val is None
        assert max_val is None

    def test_min_max_decimal(self):
        """Min/max with decimal type."""
        stats = '{"min_value": "-2.750000000000000", "max_value": "0.500000000000000"}'
        top_k, min_val, max_val = parse_stats_json(stats, sql_type="decimal(18,6)")

        assert top_k is None
        assert min_val == Decimal("-2.750000000000000")
        assert max_val == Decimal("0.500000000000000")

    def test_min_max_integer(self):
        """Min/max with integer type."""
        stats = '{"min_value": "1", "max_value": "99999"}'
        top_k, min_val, max_val = parse_stats_json(stats, sql_type="bigint")

        assert top_k is None
        assert min_val == 1
        assert max_val == 99999
        assert isinstance(min_val, int)

    def test_min_max_negative_integer(self):
        """Negative integer min/max."""
        stats = '{"min_value": "-100", "max_value": "50"}'
        top_k, min_val, max_val = parse_stats_json(stats, sql_type="int")

        assert min_val == -100
        assert max_val == 50

    def test_min_max_date(self):
        """Min/max with date type."""
        stats = '{"min_value": "2020-01-01", "max_value": "2024-12-31"}'
        top_k, min_val, max_val = parse_stats_json(stats, sql_type="date")

        assert top_k is None
        assert min_val == date(2020, 1, 1)
        assert max_val == date(2024, 12, 31)

    def test_min_max_timestamp(self):
        """Min/max with timestamp type."""
        stats = '{"min_value": "2020-01-01T00:00:00", "max_value": "2024-12-31T23:59:59"}'
        top_k, min_val, max_val = parse_stats_json(stats, sql_type="timestamp")

        assert top_k is None
        assert min_val == datetime(2020, 1, 1, 0, 0, 0)
        assert max_val == datetime(2024, 12, 31, 23, 59, 59)

    def test_min_max_unknown_type(self):
        """Unknown SQL type keeps values as strings."""
        stats = '{"min_value": "abc", "max_value": "xyz"}'
        top_k, min_val, max_val = parse_stats_json(stats, sql_type="varchar(255)")

        assert top_k is None
        assert min_val == "abc"
        assert max_val == "xyz"

    def test_min_max_no_type(self):
        """No SQL type keeps values as strings."""
        stats = '{"min_value": "100", "max_value": "200"}'
        top_k, min_val, max_val = parse_stats_json(stats, sql_type=None)

        assert min_val == "100"
        assert max_val == "200"

    def test_malformed_json(self):
        """Malformed JSON returns all None."""
        top_k, min_val, max_val = parse_stats_json("{invalid json}")
        assert top_k is None
        assert min_val is None
        assert max_val is None

    def test_none_input(self):
        """None input returns all None."""
        top_k, min_val, max_val = parse_stats_json(None)
        assert top_k is None
        assert min_val is None
        assert max_val is None

    def test_empty_string(self):
        """Empty string returns all None."""
        top_k, min_val, max_val = parse_stats_json("")
        assert top_k is None
        assert min_val is None
        assert max_val is None

    def test_empty_top_k_list(self):
        """Empty top_k list returns None."""
        stats = '{"top_k": []}'
        top_k, min_val, max_val = parse_stats_json(stats)
        assert top_k is None

    def test_top_k_with_invalid_entries(self):
        """Invalid top_k entries are skipped."""
        stats = '{"top_k": [{"value": "GOOD", "count": 10}, {"bad": "entry"}, {"value": "OK", "count": 5}]}'
        top_k, _, _ = parse_stats_json(stats)
        assert top_k is not None
        assert len(top_k) == 2

    def test_unparseable_decimal(self):
        """Unparseable decimal value falls back to string."""
        stats = '{"min_value": "not_a_number", "max_value": "0.5"}'
        top_k, min_val, max_val = parse_stats_json(stats, sql_type="decimal(10,2)")

        assert min_val == "not_a_number"  # fallback
        assert max_val == Decimal("0.5")  # parsed successfully

    def test_negative_decimal_comparison(self):
        """Verify negative decimals are properly comparable after parsing."""
        stats1 = '{"min_value": "-10.5", "max_value": "5.0"}'
        stats2 = '{"min_value": "-2.0", "max_value": "100.0"}'

        _, min1, max1 = parse_stats_json(stats1, sql_type="decimal(10,2)")
        _, min2, max2 = parse_stats_json(stats2, sql_type="decimal(10,2)")

        # Verify correct numeric comparison (not string comparison)
        assert min1 < min2  # -10.5 < -2.0
        assert max2 > max1  # 100.0 > 5.0

    def test_neither_top_k_nor_minmax(self):
        """JSON with unrecognized keys returns all None."""
        stats = '{"something_else": "value"}'
        top_k, min_val, max_val = parse_stats_json(stats)
        assert top_k is None
        assert min_val is None
        assert max_val is None

    def test_float_type(self):
        """Float type parses to Decimal."""
        stats = '{"min_value": "1.5", "max_value": "9.9"}'
        _, min_val, max_val = parse_stats_json(stats, sql_type="float")
        assert isinstance(min_val, Decimal)
        assert isinstance(max_val, Decimal)

    def test_only_min_value(self):
        """Only min_value present, max_value absent."""
        stats = '{"min_value": "10"}'
        top_k, min_val, max_val = parse_stats_json(stats, sql_type="int")
        assert min_val == 10
        assert max_val is None
