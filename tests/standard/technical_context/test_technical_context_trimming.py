"""Unit tests for budget-driven trimming of technical context payloads.

The contract these lock down:
- whatever the budget is, the result fits it and is the largest fit available;
- a bigger budget never yields fewer values (and, while values remain to be trimmed,
  a meaningfully bigger budget yields more);
- a budget past the hard cap still produces values;
- when even one value per column is too much, columns give up their values in priority
  order instead of all at once.
"""

import pytest

from langchain_timbr.technical_context.assembly.trimming import (
    _estimate_total_tokens,
    trim_to_budget,
)
from langchain_timbr.technical_context.config import TechnicalContextConfig
from langchain_timbr.technical_context.types import ColumnPayload, ColumnRef, SemanticType

VALUES_PER_COLUMN = 200
TRIMMABLE_COLUMNS = 7


def _values(prefix: str, count: int = VALUES_PER_COLUMN) -> list[str]:
    """Value lists with realistic length and variety — a token count over uniform short
    values would not exercise the search the way real value domains do."""
    words = ("northern", "central", "coastal", "upper", "lower", "greater")
    return [
        f"{words[i % len(words)].title()} {prefix.title()} Division {i:03d}"
        for i in range(count)
    ]


def _payloads(n_trimmable: int = TRIMMABLE_COLUMNS) -> dict[str, ColumnPayload]:
    return {
        f"dim_{i}": ColumnPayload(
            format_hint="top_k",
            values=_values(f"dim_{i}"),
            distinct_count=VALUES_PER_COLUMN + 50,
        )
        for i in range(n_trimmable)
    }


def _refs(payloads: dict[str, ColumnPayload]) -> dict[str, ColumnRef]:
    """Half the columns in a high-priority band, half in a low-priority one."""
    return {
        name: ColumnRef(
            name=name,
            sql_type="varchar",
            ontology_distance=0 if i % 2 == 0 else 2,
            priority_band=1 if i % 2 == 0 else 5,
            semantic_type=SemanticType.CATEGORICAL_TEXT,
        )
        for i, name in enumerate(payloads)
    }


def _trim(budget: int, n_trimmable: int = TRIMMABLE_COLUMNS, **config_kwargs):
    payloads = _payloads(n_trimmable)
    result = trim_to_budget(
        payloads, _refs(payloads), set(), TechnicalContextConfig(max_tokens=budget, **config_kwargs)
    )
    return result, sum(len(p.values) for p in result.values()), _estimate_total_tokens(result)


def _budgets(start: int, stop: int, step: int) -> list[int]:
    return list(range(start, stop + 1, step))


class TestBudgetIsRespected:
    @pytest.mark.parametrize("budget", _budgets(1000, 20000, 500))
    def test_result_fits_the_budget(self, budget):
        _result, _values_kept, tokens = _trim(budget)
        assert tokens <= budget

    @pytest.mark.parametrize("budget", (2000, 6000, 9000, 12000))
    def test_the_fit_is_maximal(self, budget):
        """One more value per trimmable column would not fit — the budget is spent, not
        rounded down to a preset size."""
        result, _values_kept, _tokens = _trim(budget)
        assert all(len(p.values) < VALUES_PER_COLUMN for p in result.values()), (
            "budget is high enough to keep every value, so this proves nothing"
        )
        for payload in result.values():
            payload.values = payload.values + ["One More Value For Every Column 999"]
        assert _estimate_total_tokens(result) > budget

    def test_untouched_when_everything_already_fits(self):
        result, values_kept, _tokens = _trim(TechnicalContextConfig.safety_ceiling, n_trimmable=1)
        assert values_kept == VALUES_PER_COLUMN
        assert all(len(p.values) == VALUES_PER_COLUMN for p in result.values())


class TestBudgetChangesTheOutput:
    """The budget has to behave like a dial. A fixed ladder of value counts makes wide
    ranges of it indistinguishable, which is invisible to whoever set it."""

    def test_value_count_never_decreases_as_the_budget_grows(self):
        counts = [_trim(budget)[1] for budget in _budgets(1000, 20000, 500)]
        assert counts == sorted(counts)

    @pytest.mark.parametrize(
        "lower,higher",
        ((4000, 5000), (5000, 6000), (6000, 7000), (7000, 8000), (8000, 9000)),
    )
    def test_a_bigger_budget_yields_more_values(self, lower, higher):
        assert _trim(lower)[1] < _trim(higher)[1]

    def test_every_trimmable_column_contributes(self):
        result, _values_kept, _tokens = _trim(9000)
        assert all(len(p.values) > 0 for p in result.values())


class TestHardCap:
    def test_budget_above_the_cap_still_emits_values(self):
        """Asking for more than the cap must not empty the context."""
        _result, values_kept, tokens = _trim(TechnicalContextConfig.safety_ceiling * 3)
        assert values_kept > 0
        assert tokens <= TechnicalContextConfig.safety_ceiling

    def test_budget_above_the_cap_matches_the_cap(self):
        at_cap = _trim(TechnicalContextConfig.safety_ceiling)
        above_cap = _trim(TechnicalContextConfig.safety_ceiling * 3)
        assert above_cap[1] == at_cap[1]

    @pytest.mark.parametrize("budget", (1, 10, 3000, 20000, 10 ** 6))
    def test_never_raises(self, budget):
        _trim(budget)


class TestUnreachableBudget:
    """A budget can be impossible to meet — most often because a protected column alone
    exceeds it. The trimmable columns then fall to a single value each: handing back more
    just because the hard cap is far away would emit values the budget never authorized."""

    @staticmethod
    def _with_protected_giant():
        payloads = _payloads(n_trimmable=2)
        payloads["protected_giant"] = ColumnPayload(
            format_hint="all",  # protected hint: never trimmed
            values=_values("protected"),
            distinct_count=VALUES_PER_COLUMN,
        )
        config = TechnicalContextConfig(max_tokens=50)
        return trim_to_budget(payloads, _refs(payloads), set(), config), config

    def test_protected_column_keeps_everything(self):
        result, _config = self._with_protected_giant()
        assert len(result["protected_giant"].values) == VALUES_PER_COLUMN

    def test_trimmable_columns_fall_to_a_single_value(self):
        result, _config = self._with_protected_giant()
        assert [len(result[name].values) for name in ("dim_0", "dim_1")] == [1, 1]

    def test_nothing_is_emptied_while_the_ceiling_allows_a_value(self):
        result, config = self._with_protected_giant()
        assert _estimate_total_tokens(result) <= config.safety_ceiling
        assert all(p.values for p in result.values())


def _information_rank(payload: ColumnPayload) -> int:
    """How much a column still says: its values, only its cardinality, or nothing."""
    if payload.values:
        return 2
    return 1 if payload.format_hint == "count_only" else 0


class TestLastResortDegradation:
    """Reached only when one value per column is already over the hard cap. Columns must
    be sacrificed in priority order, and gradually, rather than all losing everything."""

    @staticmethod
    def _crowded(ceiling: int):
        """Enough columns that one value each cannot fit the ceiling."""
        payloads = {
            f"dim_{i}": ColumnPayload(
                format_hint="top_k", values=_values(f"dim_{i}", count=5), distinct_count=500,
            )
            for i in range(60)
        }
        refs = {
            name: ColumnRef(
                name=name,
                sql_type="varchar",
                ontology_distance=i % 3,
                # Lower band = higher priority; the last third is the least relevant.
                priority_band=1 if i < 20 else (3 if i < 40 else 5),
                semantic_type=SemanticType.CATEGORICAL_TEXT,
            )
            for i, name in enumerate(payloads)
        }
        config = TechnicalContextConfig(max_tokens=ceiling, safety_ceiling=ceiling)
        return trim_to_budget(payloads, refs, set(), config), refs, config

    @pytest.mark.parametrize("ceiling", (600, 100))
    def test_information_never_increases_with_lower_priority(self, ceiling):
        """The invariant: no column outranks a higher-priority one in what it kept."""
        result, refs, _config = self._crowded(ceiling)
        by_band: dict[int, set[int]] = {}
        for name, payload in result.items():
            by_band.setdefault(refs[name].priority_band, set()).add(_information_rank(payload))
        for band, ranks in by_band.items():
            better = {
                rank
                for other_band, other in by_band.items()
                if other_band > band
                for rank in other
            }
            assert not better or max(better) <= max(ranks)

    @pytest.mark.parametrize("ceiling", (600, 100))
    def test_result_clears_the_ceiling(self, ceiling):
        result, _refs, config = self._crowded(ceiling)
        assert _estimate_total_tokens(result) <= config.safety_ceiling

    def test_highest_priority_columns_keep_their_values(self):
        result, refs, _config = self._crowded(600)
        kept = [name for name, p in result.items() if p.values]
        assert kept, "every column was emptied — degradation was not gradual"
        assert min(refs[name].priority_band for name in kept) == 1

    def test_cardinality_survives_where_values_cannot(self):
        """A sacrificed column reports its distinct count before it goes fully bare."""
        result, _refs, _config = self._crowded(600)
        sacrificed = [p for p in result.values() if not p.values]
        assert sacrificed, "nothing was sacrificed, so this scenario proves nothing"
        assert any(p.format_hint == "count_only" for p in sacrificed)

    def test_extreme_ceiling_keeps_the_most_relevant_columns_talking(self):
        """Even when no column can keep a value, the ones that still report a cardinality
        are the highest-priority ones."""
        result, refs, _config = self._crowded(100)
        speaking = [name for name, p in result.items() if _information_rank(p) > 0]
        assert speaking
        assert max(refs[name].priority_band for name in speaking) <= 3
