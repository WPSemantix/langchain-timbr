"""Tests for stats_parser module."""

import pytest
from decimal import Decimal
from datetime import date, datetime

from langchain_timbr.technical_context.statistics_loader.stats_parser import (
    parse_stats_json,
    classify_values,
    index_date_years,
    annotate_stripped_forms,
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


class TestNormalizedFormsCarriedOnEntries:
    """Normalization is derived once, where the row is parsed.

    It used to happen in the matcher on every request, memoized in a
    process-wide cache that wiped itself wholesale on overflow. Carrying it on
    the entry ties its lifetime to the value it belongs to.
    """

    def test_top_k_entries_carry_normalized_forms(self):
        stats = '{"top_k": [{"value": "Good Place", "count": 9}]}'
        top_k, _, _ = parse_stats_json(stats)

        assert top_k[0].value == "Good Place"
        assert top_k[0].norm == "goodplace"
        assert top_k[0].norm_space == "good place"

    def test_normalization_matches_the_matcher_functions(self):
        """The stored form must equal what the matcher would have derived."""
        from langchain_timbr.technical_context.matching.normalize import (
            normalize, normalize_keep_spaces,
        )
        raw = "U.S.A.  Café  Latte²"
        stats = '{"top_k": [{"value": "%s", "count": 1}]}' % raw
        top_k, _, _ = parse_stats_json(stats)

        assert top_k[0].norm == normalize(raw)
        assert top_k[0].norm_space == normalize_keep_spaces(raw)

    def test_memo_shares_one_object_across_rows(self):
        """A value in several mappings' top-K holds one copy, not one per row.

        Measured 6.5x duplication across columns on the captured fixture; without
        this the normalized strings cost 17.7M chars instead of 2.1M.
        """
        memo: dict = {}
        stats_a = '{"top_k": [{"value": "Indolene", "count": 4}]}'
        stats_b = '{"top_k": [{"value": "Indolene", "count": 2}]}'

        a, _, _ = parse_stats_json(stats_a, None, memo)
        b, _, _ = parse_stats_json(stats_b, None, memo)

        assert a[0].norm is b[0].norm
        assert a[0].norm_space is b[0].norm_space
        assert len(memo) == 1

    def test_without_a_memo_values_are_still_correct(self):
        """The memo is an optimization, never a correctness requirement."""
        stats = '{"top_k": [{"value": "Indolene", "count": 4}]}'
        with_memo, _, _ = parse_stats_json(stats, None, {})
        without, _, _ = parse_stats_json(stats)

        assert with_memo[0].norm == without[0].norm
        assert with_memo[0].norm_space == without[0].norm_space


class TestValueClassification:
    """Numbers and dates are recognised from the values, not the declared type.

    The declared SQL type finds only 8 of the 148 columns that are numeric in
    practice, because zero-padded identifiers are varchar.
    """

    def test_numeric_date_and_text(self):
        assert classify_values(["1", "2", "0003"]) == "numeric"
        assert classify_values(["-5", "1.5", "1e3"]) == "numeric"
        assert classify_values(["2024-03-01", "2019-12-31"]) == "date"
        assert classify_values(["Good Place", "Test1"]) == "text"

    def test_one_text_value_disqualifies_the_column(self):
        """Mixed content is text — the safe direction, it keeps the full cascade."""
        assert classify_values(["1", "2", "n/a"]) == "text"
        assert classify_values(["2024-03-01", "unknown"]) == "text"

    def test_empty_is_text(self):
        assert classify_values([]) == "text"

    def test_classification_short_circuits_on_text(self):
        """A text column must not be scanned past its first value.

        Three quarters of columns are text. If they were scanned in full, the
        cost would land on every fetch instead of being abandoned immediately.
        Proven by making every value after the first raise if it is inspected.
        """
        class Boom(str):
            def __new__(cls):
                return super().__new__(cls, "xxxxxxxxxx")

            def __getitem__(self, item):
                raise AssertionError("classify_values kept scanning after text")

        values = ["definitely text"] + [Boom() for _ in range(500)]
        assert classify_values(values) == "text"

    def test_classification_cost_stays_bounded(self):
        """Guard against a future rewrite reintroducing an O(values) regex.

        Measured at 30.8 ms for 872,953 values; this asserts the same order of
        magnitude on a smaller sample so it stays fast in CI.
        """
        import time
        cols = [[f"text value {i}"] * 50 for i in range(400)]
        cols += [[str(i) for i in range(50)] for _ in range(100)]
        t = time.perf_counter()
        for c in cols:
            classify_values(c)
        elapsed = time.perf_counter() - t
        assert elapsed < 0.5, f"classification took {elapsed:.3f}s for 25k values"


class TestStrippedFormsAndYearIndex:

    def test_stripped_form_keeps_a_digit(self):
        top_k, _, _ = parse_stats_json(
            '{"top_k": [{"value": "000010", "count": 3}, {"value": "0000", "count": 1}]}'
        )
        annotate_stripped_forms(top_k)
        assert top_k[0].norm_stripped == "10"
        assert top_k[1].norm_stripped == "0"      # not the empty string

    def test_only_integers_get_a_stripped_form(self):
        """`0.5` would strip to `.5` and `-0010` to `-0010` — keys no normalized
        token can ever match, so they are dead weight. Integers only."""
        top_k, _, _ = parse_stats_json(
            '{"top_k": [{"value": "0.5", "count": 1}, {"value": "-0010", "count": 1},'
            ' {"value": "000010", "count": 1}]}'
        )
        annotate_stripped_forms(top_k)
        assert top_k[0].norm_stripped is None      # 0.5
        assert top_k[1].norm_stripped is None      # -0010
        assert top_k[2].norm_stripped == "10"      # 000010

    def test_year_index_is_months_and_count_not_dates(self):
        values = ["2024-01-05", "2024-01-19", "2024-03-02", "2019-07-01"]
        idx = index_date_years(values)
        assert idx["2024"] == (frozenset({"01", "03"}), 3)
        assert idx["2019"] == (frozenset({"07"}), 1)
        # the whole point: 12 entries instead of 1,408 values
        assert len(idx) == 2
