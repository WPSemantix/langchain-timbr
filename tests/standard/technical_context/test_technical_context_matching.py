"""Unit tests for matching modules: normalize, exact, fuzzy, substring."""

import pytest
from langchain_timbr.technical_context.matching.normalize import normalize, normalize_keep_spaces
from langchain_timbr.technical_context.matching.exact import exact_match
from langchain_timbr.technical_context.matching.rapidfuzz_matcher import fuzzy_match
from langchain_timbr.technical_context.matching.ahocorasick_matcher import substring_match


class TestNormalize:
    """Tests for normalize()."""

    def test_basic(self):
        assert normalize("Hello World") == "helloworld"

    def test_unicode_nfkc(self):
        # Non-ASCII letters are stripped by the [^a-z0-9] regex after casefold
        assert normalize("Café") == "caf"
        # ß casefolded to ss (German sharp s)
        assert normalize("Straße") == "strasse"

    def test_punctuation_stripped(self):
        assert normalize("U.S.A.") == "usa"

    def test_empty(self):
        assert normalize("") == ""

    def test_none(self):
        assert normalize(None) == ""

    def test_numbers_kept(self):
        assert normalize("ABC-123") == "abc123"

    def test_casefold(self):
        assert normalize("Straße") == "strasse"

    def test_whitespace(self):
        assert normalize("  spaces  ") == "spaces"


class TestNormalizeKeepSpaces:
    """Tests for normalize_keep_spaces()."""

    def test_basic(self):
        assert normalize_keep_spaces("New  York  City") == "new york city"

    def test_punctuation_becomes_space(self):
        assert normalize_keep_spaces("hello-world") == "hello world"

    def test_empty(self):
        assert normalize_keep_spaces("") == ""

    def test_none(self):
        assert normalize_keep_spaces(None) == ""


class TestExactMatch:
    """Tests for exact_match()."""

    def test_exact_match_found(self):
        results = exact_match(["USA", "Germany"], "country", ["USA", "France", "Germany"])
        assert len(results) == 2
        matched_values = {r.matched_value for r in results}
        assert "USA" in matched_values
        assert "Germany" in matched_values
        assert all(r.score == 100 for r in results)
        assert all(r.match_type == "exact" for r in results)

    def test_case_insensitive(self):
        results = exact_match(["usa"], "country", ["USA"])
        assert len(results) == 1
        assert results[0].matched_value == "USA"

    def test_no_match(self):
        results = exact_match(["xyz"], "country", ["USA", "France"])
        assert results == []

    def test_empty_tokens(self):
        results = exact_match([], "country", ["USA"])
        assert results == []

    def test_empty_values(self):
        results = exact_match(["USA"], "country", [])
        assert results == []


class TestFuzzyMatch:
    """Tests for fuzzy_match()."""

    def test_close_match(self):
        results = fuzzy_match(["Unitd States"], "country", ["United States"], threshold=80)
        assert len(results) == 1
        assert results[0].matched_value == "United States"
        assert results[0].match_type == "fuzzy"
        assert results[0].score >= 80

    def test_no_match_below_threshold(self):
        results = fuzzy_match(["xyz"], "country", ["United States"], threshold=88)
        assert results == []

    def test_short_tokens_skipped(self):
        results = fuzzy_match(["ab"], "country", ["ab"], threshold=88)
        assert results == []

    def test_empty_inputs(self):
        results = fuzzy_match([], "country", ["USA"])
        assert results == []


class TestSubstringMatch:
    """Tests for substring_match()."""

    def test_substring_found(self):
        results = substring_match(
            "Show me orders from United States",
            "country",
            ["United States", "France", "Germany"],
        )
        assert len(results) == 1
        assert results[0].matched_value == "United States"
        assert results[0].match_type == "substring"

    def test_no_substring(self):
        results = substring_match(
            "Show me all orders",
            "country",
            ["United States", "France"],
        )
        assert results == []

    def test_multiple_substrings(self):
        results = substring_match(
            "Compare France and Germany sales",
            "country",
            ["France", "Germany", "Italy"],
        )
        matched_values = {r.matched_value for r in results}
        assert "France" in matched_values
        assert "Germany" in matched_values
        assert len(results) == 2

    def test_min_length_filter(self):
        # "US" is only 2 chars, should be filtered at min_length=3
        results = substring_match("US orders", "country", ["US"], min_length=3)
        assert results == []

    def test_empty_prompt(self):
        results = substring_match("", "country", ["USA"])
        assert results == []
