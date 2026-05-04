"""Tests for logic_query module."""

import pytest

from langchain_timbr.technical_context.statistics_loader.logic_query import (
    parse_logic_query,
    LogicQueryParseError,
)


class TestParseLogicQuery:
    """Test logic query parsing."""

    def test_plain_dtimbr(self):
        """Basic dtimbr.concept WHERE pattern."""
        result = parse_logic_query("SELECT * FROM dtimbr.inventory WHERE status = 'active'")
        assert result == "inventory"

    def test_plain_timbr(self):
        """Basic timbr.concept WHERE pattern."""
        result = parse_logic_query("SELECT * FROM timbr.inventory WHERE status = 'active'")
        assert result == "inventory"

    def test_backtick_both(self):
        """Backtick-quoted schema and concept."""
        result = parse_logic_query("SELECT * FROM `dtimbr`.`inventory` WHERE x > 0")
        assert result == "inventory"

    def test_double_quote_both(self):
        """Double-quote schema and concept."""
        result = parse_logic_query('SELECT * FROM "dtimbr"."inventory" WHERE x > 0')
        assert result == "inventory"

    def test_backtick_concept_only(self):
        """Backtick on concept only."""
        result = parse_logic_query("SELECT * FROM dtimbr.`my_concept` WHERE x = 1")
        assert result == "my_concept"

    def test_mixed_quoting(self):
        """Double-quote schema, backtick concept."""
        result = parse_logic_query('SELECT * FROM "dtimbr".`inventory` WHERE active = true')
        assert result == "inventory"

    def test_case_insensitive_where(self):
        """WHERE keyword is case-insensitive."""
        result = parse_logic_query("SELECT * FROM dtimbr.test_concept where col = 1")
        assert result == "test_concept"

    def test_case_insensitive_schema(self):
        """Schema name is case-insensitive."""
        result = parse_logic_query("SELECT * FROM DTIMBR.inventory WHERE x = 1")
        assert result == "inventory"

    def test_no_match_returns_none(self):
        """No match returns None."""
        result = parse_logic_query("SELECT * FROM some_table WHERE x = 1")
        assert result is None

    def test_none_input(self):
        """None input returns None."""
        result = parse_logic_query(None)
        assert result is None

    def test_empty_string(self):
        """Empty string returns None."""
        result = parse_logic_query("")
        assert result is None

    def test_first_match_wins(self):
        """When query references multiple concepts, first match wins."""
        query = "SELECT * FROM dtimbr.first_concept WHERE x IN (SELECT y FROM dtimbr.second_concept WHERE z = 1)"
        result = parse_logic_query(query)
        assert result == "first_concept"

    def test_concept_with_underscores(self):
        """Concept name with underscores."""
        result = parse_logic_query("SELECT * FROM dtimbr.my_long_concept_name WHERE x = 1")
        assert result == "my_long_concept_name"

    def test_concept_with_numbers(self):
        """Concept name with numbers."""
        result = parse_logic_query("SELECT * FROM dtimbr.concept123 WHERE x = 1")
        assert result == "concept123"

    def test_whitespace_before_where(self):
        """Multiple spaces before WHERE."""
        result = parse_logic_query("SELECT * FROM dtimbr.inventory   WHERE x > 0")
        assert result == "inventory"

    def test_no_where_clause(self):
        """Query without WHERE clause should not match."""
        result = parse_logic_query("SELECT * FROM dtimbr.inventory")
        assert result is None

    def test_newline_before_where(self):
        """Newline counts as whitespace before WHERE."""
        result = parse_logic_query("SELECT * FROM dtimbr.inventory\nWHERE x > 0")
        assert result == "inventory"
