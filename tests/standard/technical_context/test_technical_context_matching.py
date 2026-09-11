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
        # Accents fold to their base letter rather than being deleted. This
        # asserted "caf" until the normalizer was fixed — the old ASCII-only
        # strip silently mangled every non-ASCII value, and this test pinned
        # that behaviour instead of catching it.
        assert normalize("Café") == "cafe"
        # ß casefolded to ss (German sharp s)
        assert normalize("Straße") == "strasse"

    def test_non_latin_scripts_survive(self):
        """CJK and Hangul must remain matchable.

        Previously `[^a-z0-9]` reduced these to '' or to whatever Latin fragment
        they happened to contain, so a Japanese ontology's values were either
        unmatchable or all collapsed onto the same key — 24 distinct locations
        in the captured fixture shared the normalized form 'c'.
        """
        assert normalize("北京") == "北京"
        assert normalize("한국") == "한국"
        # Full-width Latin folds to ASCII via NFKC, and the rest is preserved,
        # so these two no longer collide.
        assert normalize("あらた神") == "あらた神"
        assert normalize("あらた四") == "あらた四"
        assert normalize("あらた神奈川") != normalize("あらた四国")

    def test_accents_fold_for_european_data(self):
        assert normalize("Müller GmbH") == "mullergmbh"
        assert normalize("São Paulo") == "saopaulo"
        assert normalize("naïve") == "naive"

    def test_underscore_is_punctuation(self):
        """`\\w` would keep '_'; it is a separator here, not a letter."""
        assert normalize("order_id") == "orderid"
        assert normalize_keep_spaces("order_id") == "order id"

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


class TestResultCacheLRU:
    """The matcher memo evicts its coldest entries instead of wiping itself.

    It used to call ``.clear()`` on overflow, so a single request larger than the
    cap filled it, wiped it, refilled it and retained nothing — measurably worse
    than having no memo at all.
    """

    def _run(self, mm, col, value, config):
        return mm.run_all_matchers(
            prompt_text="find " + value,
            prompt_tokens=[value],
            column_name=col,
            known_values=[value, "other"],
            config=config,
        )

    def test_overflow_evicts_oldest_not_everything(self):
        from langchain_timbr.technical_context.assembly import multi_match as mm
        from langchain_timbr.technical_context.config import TechnicalContextConfig

        config = TechnicalContextConfig()
        mm.clear_matcher_cache()
        original = mm._RESULT_CACHE_MAX_ENTRIES
        mm._RESULT_CACHE_MAX_ENTRIES = 3
        try:
            for i in range(10):
                self._run(mm, f"col_{i}", f"value_{i}", config)

            # Old behaviour: .clear() on overflow, so this could be as low as 1.
            assert len(mm._RESULT_CACHE) == 3
        finally:
            mm._RESULT_CACHE_MAX_ENTRIES = original
            mm.clear_matcher_cache()

    def test_recently_used_entry_survives_overflow(self):
        from langchain_timbr.technical_context.assembly import multi_match as mm
        from langchain_timbr.technical_context.config import TechnicalContextConfig

        config = TechnicalContextConfig()
        mm.clear_matcher_cache()
        original = mm._RESULT_CACHE_MAX_ENTRIES
        mm._RESULT_CACHE_MAX_ENTRIES = 3
        try:
            for i in range(3):
                self._run(mm, f"col_{i}", f"value_{i}", config)

            # Touch the oldest so it becomes the most recently used...
            self._run(mm, "col_0", "value_0", config)
            # ...then push two more in, evicting two entries.
            self._run(mm, "col_9", "value_9", config)
            self._run(mm, "col_8", "value_8", config)

            surviving = {k[2] for k in mm._RESULT_CACHE}
            assert "col_0" in surviving      # kept: recently used
            assert "col_1" not in surviving  # dropped: coldest
        finally:
            mm._RESULT_CACHE_MAX_ENTRIES = original
            mm.clear_matcher_cache()

    def test_value_counter_tracks_evictions(self):
        from langchain_timbr.technical_context.assembly import multi_match as mm
        from langchain_timbr.technical_context.config import TechnicalContextConfig

        config = TechnicalContextConfig()
        mm.clear_matcher_cache()
        original = mm._RESULT_CACHE_MAX_ENTRIES
        mm._RESULT_CACHE_MAX_ENTRIES = 2
        try:
            for i in range(6):
                self._run(mm, f"col_{i}", f"value_{i}", config)

            # Two entries of two values each; the counter must have come back
            # down with the evictions rather than growing without bound.
            assert mm._RESULT_CACHE_VALUES == sum(
                len(v[0]) for v in mm._RESULT_CACHE.values()
            )
        finally:
            mm._RESULT_CACHE_MAX_ENTRIES = original
            mm.clear_matcher_cache()


class TestNumbersAndDatesAreNotFuzzy:
    """Edit distance between two numbers is meaningless and actively harmful.

    At the default fuzzy floor of 70, `1500` scores 75 against `1600` and
    `20240301` scores 87.5 against `20240302` — so asking about one offers the
    other to the model as a filter literal.
    """

    @staticmethod
    def _run(values, tokens, kind, stripped=None):
        from langchain_timbr.technical_context.assembly.multi_match import (
            run_all_matchers, clear_matcher_cache,
        )
        from langchain_timbr.technical_context.config import TechnicalContextConfig
        clear_matcher_cache()
        return run_all_matchers(
            prompt_text=" ".join(tokens), prompt_tokens=tokens,
            column_name="c", known_values=values,
            config=TechnicalContextConfig(), value_kind=kind,
            normalized_stripped=stripped,
        )

    def test_absent_number_produces_no_match_at_all(self):
        """Was: '777777' annotated the prompt with '91772777' at score 71."""
        res = self._run(["91772777", "91772778", "12345678"], ["777777"], "numeric")
        assert res == []

    def test_neighbouring_number_is_not_offered(self):
        res = self._run(["000010", "000020"], ["000010"], "numeric")
        assert [(r.match_type, r.matched_value) for r in res] == [("exact", "000010")]

    def test_neighbouring_date_is_not_offered(self):
        res = self._run(["2024-03-01", "2024-03-02"], ["2024-03-01"], "date")
        assert all(r.match_type != "fuzzy" for r in res)
        assert "2024-03-02" not in [r.matched_value for r in res]

    def test_text_columns_keep_fuzzy(self):
        """The cascade is unchanged off the numeric/date path."""
        res = self._run(["Good Place", "Test1"], ["Good Pla"], "text")
        assert any(r.match_type == "fuzzy" for r in res)


class TestZeroPaddedNumbersMatch:
    """Zero-padded identifiers are the norm here — 82 of 148 numeric columns —
    and nobody asks for them that way."""

    @staticmethod
    def _run(values, token):
        from langchain_timbr.technical_context.assembly.multi_match import (
            run_all_matchers, clear_matcher_cache,
        )
        from langchain_timbr.technical_context.config import TechnicalContextConfig
        from langchain_timbr.technical_context.matching.normalize import normalize
        clear_matcher_cache()
        stripped = [v.lstrip("0") or "0" for v in values]
        return run_all_matchers(
            prompt_text=f"item {token}", prompt_tokens=[token], column_name="c",
            known_values=values, config=TechnicalContextConfig(),
            value_kind="numeric", normalized_stripped=stripped,
        )

    def test_ten_finds_zero_padded_ten(self):
        res = self._run(["000010", "000020", "000030"], "10")
        assert [(r.match_type, r.matched_value) for r in res] == [("exact", "000010")]

    def test_one_finds_zero_padded_one(self):
        res = self._run(["0001", "0002", "0003"], "1")
        assert [r.matched_value for r in res] == ["0001"]

    def test_two_finds_zero_padded_two_in_a_small_enum(self):
        res = self._run(["0000", "0001", "0002", "0003"], "2")
        assert [r.matched_value for r in res] == ["0002"]

    def test_unpadded_values_still_match_directly(self):
        res = self._run(["1", "2", "3", "10"], "1")
        assert [r.matched_value for r in res] == ["1"]

    def test_a_real_value_beats_a_stripped_one(self):
        """If the column holds both `10` and `000010`, `10` means `10`."""
        res = self._run(["10", "000010"], "10")
        assert [r.matched_value for r in res] == ["10"]

    def test_stripped_form_is_not_findable_inside_a_longer_number(self):
        """`10` must not match inside `2010`.

        The stripped form goes into exact matching only — which is whole-token
        by construction — and deliberately not into the substring automaton.
        """
        from langchain_timbr.technical_context.assembly.multi_match import (
            run_all_matchers, clear_matcher_cache,
        )
        from langchain_timbr.technical_context.config import TechnicalContextConfig
        clear_matcher_cache()
        res = run_all_matchers(
            prompt_text="orders in 2010", prompt_tokens=["2010"], column_name="c",
            known_values=["000010", "000020"], config=TechnicalContextConfig(),
            value_kind="numeric", normalized_stripped=["10", "20"],
        )
        assert res == []


class TestDateYearMatching:
    """A year in the question promotes date columns covering it — as a summary.

    Listing the dates instead would be ~184 values for one year of one column,
    and across all date columns roughly sixteen times the whole context budget.
    """

    YEARS = {"2024": (frozenset({"01", "03", "12"}), 184),
             "2019": (frozenset({"07"}), 12)}

    def test_year_hit_returns_a_summary_not_dates(self):
        from langchain_timbr.technical_context import match_date_year
        m = match_date_year(["Indonesia", "2024"], self.YEARS, "delivery_date")
        assert m is not None
        assert m.match_type == "year"
        assert m.candidate == "2024"
        assert m.matched_value == "2024: months 01,03,12 (184 dates)"
        # the whole point — no date is listed
        assert "2024-01" not in m.matched_value

    def test_absent_year_does_not_promote_the_column(self):
        from langchain_timbr.technical_context import match_date_year
        assert match_date_year(["2031"], self.YEARS, "delivery_date") is None

    def test_non_year_candidates_are_ignored(self):
        from langchain_timbr.technical_context import match_date_year
        assert match_date_year(
            ["Good Place", "past 3 months"], self.YEARS, "delivery_date"
        ) is None

    def test_first_matching_year_wins(self):
        from langchain_timbr.technical_context import match_date_year
        m = match_date_year(["2019", "2024"], self.YEARS, "delivery_date")
        assert m.candidate == "2019"
