"""Behavioural pins for Plan 07 — cdist, structure cache, and multi-match caps.

These cover the properties that make the changes safe, not the speedups:

  Part 1  cdist must reproduce extractOne's answer, including its tie-break
  Part 2  a cached structure must not change what the cascade returns
  Part 4  multiplicity is bounded, and unbounded cases degrade to the old
          single-best behaviour rather than flooding the prompt
"""

import pytest

from langchain_timbr.technical_context.config import TechnicalContextConfig
from langchain_timbr.technical_context.types import SemanticType
from langchain_timbr.technical_context.assembly.multi_match import (
    run_all_matchers, clear_matcher_cache,
)
from langchain_timbr.technical_context.assembly.structure_cache import (
    clear_structure_cache, cache_stats, get_entry,
)
from langchain_timbr.technical_context.matching.rapidfuzz_matcher import fuzzy_match


CFG = TechnicalContextConfig()

# Five spellings that all normalize to 'phd'. The three-stage cascade can only
# surface three of them (one per stage), which is the gap Part 4 closes.
PHD_VARIANTS = ["PhD", "Ph.D", "Ph.D.", "P.H.D.", "Ph D", "BS", "BA", "MS"]


@pytest.fixture(autouse=True)
def _isolate():
    clear_matcher_cache()
    clear_structure_cache()
    yield
    clear_matcher_cache()
    clear_structure_cache()


def _match(values, tokens, semantic_type=SemanticType.CATEGORICAL_ENUM,
           value_kind="text", prompt=None):
    return run_all_matchers(
        prompt_text=prompt if prompt is not None else " ".join(tokens),
        prompt_tokens=tokens,
        column_name="col",
        known_values=values,
        config=CFG,
        semantic_type=semantic_type,
        value_kind=value_kind,
    )


class TestCdistEquivalence:
    """Part 1 — cdist must be a drop-in for extractOne."""

    def test_matches_extract_one_on_ties(self):
        """extractOne returns the FIRST best; argmax must agree.

        All three score identically, so the tie-break is the whole test: the
        winner must be the lowest-indexed one, and when multiplicity is off it
        must be the ONLY one.
        """
        values = ["abcd", "abce", "abcf"]
        results = fuzzy_match(["abcx"], "col", values, threshold=70)
        assert results[0].matched_value == "abcd"

        # strong_threshold above every score => weak band => single best only,
        # which is exactly the pre-Part-4 contract.
        single = fuzzy_match(["abcx"], "col", values,
                             threshold=70, strong_threshold=90)
        assert [r.matched_value for r in single] == ["abcd"]

    def test_empty_column_returns_nothing(self):
        assert fuzzy_match(["anything"], "col", [], threshold=70) == []

    def test_values_below_min_length_are_skipped(self):
        """Values under 3 chars never enter the fuzzy choice list."""
        assert fuzzy_match(["ab"], "col", ["ab", "ac"], threshold=70) == []

    def test_no_match_above_threshold(self):
        assert fuzzy_match(["zzzzzzzz"], "col", ["aaaaaaaa"], threshold=70) == []


class TestStructureCache:
    """Part 2 — caching must not change the cascade's answer."""

    def test_cached_and_uncached_agree(self):
        first = _match(PHD_VARIANTS, ["phd"])
        second = _match(PHD_VARIANTS, ["phd"])  # served from the structure cache
        assert [(r.match_type, r.matched_value) for r in first] == \
               [(r.match_type, r.matched_value) for r in second]

    def test_cascade_still_walks_duplicate_groups(self):
        """The bug a naive cache would introduce.

        'PhD', 'Ph.D' and 'Ph.D.' share one normalized form. Each stage must
        claim a DIFFERENT one — if the cached automaton kept a single winner
        per normalized form, substring would re-offer what exact already took
        and the later variants would vanish.
        """
        results = _match(PHD_VARIANTS, ["phd"])
        by_stage = {r.match_type: r.matched_value for r in results}
        assert "exact" in by_stage and "substring" in by_stage
        assert by_stage["exact"] != by_stage["substring"]

    def test_same_column_name_different_values_is_not_reused(self):
        """An ontology version change reuses column names with new values."""
        entry_a = get_entry("col", ["alpha", "beta"], "text")
        entry_b = get_entry("col", ["gamma", "delta"], "text")
        assert entry_a is not entry_b

    def test_version_change_does_not_serve_stale_matches(self):
        """The invalidation that matters: results must follow the new values.

        Content verification, not version keying — a changed value list fails
        the tuple check and rebuilds, so a stale structure can never be served
        even if the caller never tells the cache a version changed.
        """
        v1 = ["Cancelled", "Shipped", "Pending"]
        assert [r.matched_value for r in _match(v1, ["Shipped"])] == ["Shipped"]

        # Same column name, same length, different values.
        v2 = ["Delivered", "Returned", "Draft"]
        clear_matcher_cache()  # results memo only; structure cache stays warm
        assert _match(v2, ["Shipped"]) == []
        assert [r.matched_value for r in _match(v2, ["Returned"])] == ["Returned"]

    def test_two_tenants_sharing_a_column_name_do_not_thrash(self):
        """Multi-tenant: same column name, same value count, different values.

        With only (column_name, len, value_kind) in the key these collide, fail
        each other's verification, and rebuild on every request — correct, but
        the cache stops caching and only ever shows up as latency.
        """
        clear_structure_cache()
        a = ["Cancelled", "Shipped", "Pending"]
        b = ["Aprobado", "Rechazado", "Enviado"]
        ka = dict(ontology="tenant_a", schema="dtimbr", concept="orders")
        kb = dict(ontology="tenant_b", schema="dtimbr", concept="orders")
        first_a = get_entry("status", a, "text", **ka)
        first_b = get_entry("status", b, "text", **kb)
        for _ in range(3):
            assert get_entry("status", a, "text", **ka) is first_a
            assert get_entry("status", b, "text", **kb) is first_b
        assert cache_stats()["entries"] == 2

    def test_same_column_name_under_different_concepts_is_separate(self):
        """`column_name` is not unique within one ontology."""
        clear_structure_cache()
        base = dict(ontology="ont", schema="dtimbr")
        orders = get_entry("status", ["New", "Paid"], "text",
                           concept="orders", **base)
        deliveries = get_entry("status", ["Sent", "Lost"], "text",
                               concept="deliveries", **base)
        assert orders is not deliveries
        assert cache_stats()["entries"] == 2

    def test_new_values_replace_in_place_and_keep_the_counter_honest(self):
        """A version change reuses the key, so the entry is REPLACED.

        That is the point of keying on identity rather than on the values' hash:
        the previous generation is not left orphaned for the LRU to reclaim.
        Replacing used to add the new entry's size without subtracting the old
        one's, and a counter that reads high evicts a cache that is not full.
        """
        clear_structure_cache()
        ident = dict(ontology="ont", schema="dtimbr", concept="orders")
        for values in (["a1", "b1", "c1"], ["a2", "b2", "c2"], ["a3", "b3", "c3"]):
            get_entry("same_column", values, "text", **ident)
        assert cache_stats() == {"entries": 1, "values": 3}, (
            "successive value sets should replace, not accumulate"
        )

        for _ in range(3):  # hits
            get_entry("same_column", ["a3", "b3", "c3"], "text", **ident)
        assert cache_stats() == {"entries": 1, "values": 3}, (
            "counter grew on cache hits"
        )

    def test_cache_is_bounded(self):
        for i in range(50):
            get_entry(f"col{i}", [f"value{i}_{j}" for j in range(10)], "text")
        assert cache_stats()["entries"] <= 4096


class TestMultiMatchCaps:
    """Part 4 — bounded multiplicity, with a safe degradation."""

    def test_all_five_variants_surface(self):
        """Was 3 of 5: exact, substring and fuzzy each took one and stopped."""
        results = _match(PHD_VARIANTS, ["phd"])
        found = {r.matched_value for r in results}
        for variant in ("PhD", "Ph.D", "Ph.D.", "P.H.D.", "Ph D"):
            assert variant in found, f"{variant} was dropped"

    def test_weak_matches_stay_capped_at_one(self):
        """The 70-87 band is where fuzzy invents values — do not multiply it."""
        values = ["Damage Stock", "MRP ASI Versteck", "Widget", "Gadget"]
        results = _match(values, ["master stock"])
        fuzzy = [r for r in results if r.match_type == "fuzzy"]
        assert len(fuzzy) == 1
        assert fuzzy[0].score < CFG.fuzzy_threshold_default

    def test_non_selective_token_falls_back_to_single_best(self):
        """1,000 near-identical values must not all reach the prompt.

        Trimming is forbidden from shrinking a matched column
        (``trimming._is_protected``), so this cap is the only bound.
        """
        values = [f"Location {i:03d}" for i in range(1000)]
        results = _match(values, ["Location 500"])
        fuzzy = [r for r in results if r.match_type == "fuzzy"]
        assert len(fuzzy) <= 1
        assert len(results) <= 2

    def test_per_column_backstop(self):
        values = [f"Variant{i}" for i in range(40)]
        tokens = [f"Variant{i}" for i in range(30)]
        results = _match(values, tokens)
        fuzzy = [r for r in results if r.match_type == "fuzzy"]
        assert len(fuzzy) <= 20

    def test_numeric_columns_still_skip_fuzzy(self):
        """Plan 04b must survive Parts 1 and 4."""
        values = ["000010", "000020", "000030"]
        results = _match(values, ["000011"], value_kind="numeric")
        assert not [r for r in results if r.match_type == "fuzzy"]
