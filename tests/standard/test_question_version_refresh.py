"""Once-per-question ontology-version refresh (Plan 06 ruling 2b, step 7).

An ontology edit should be visible to the *next question*, not up to
CACHE_TIMEOUT later. The refresh is a hook on the base Chain: chains that hold
connection parameters override it, and one that does not degrades safely to the
shared probe's throttle.
"""

from typing import Any, Dict

import pytest

from langchain_timbr.utils import timbr_utils as tu
from langchain_timbr.utils._base_chain import Chain


CONN = {"url": "https://x/", "ontology": "onto_a", "token": "t1"}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    # The suite disables the per-question refresh (unit tests must not do
    # network I/O); this module is the one that exercises it.
    from langchain_timbr import config

    monkeypatch.setattr(config, "version_refresh_per_question", True)
    tu.clear_cache()
    tu.clear_version_probe()
    yield
    tu.clear_cache()
    tu.clear_version_probe()


class _RefreshingChain(Chain):
    """Stands in for the real chains, which override the hook identically."""

    def __init__(self, inner=None):
        super().__init__()
        self._inner = inner

    def _get_conn_params(self) -> dict:
        return CONN

    def _refresh_ontology_version(self) -> None:
        self._refresh_version_for(self._get_conn_params())

    def _call(self, inputs: Dict[str, Any], run_manager=None) -> Dict[str, Any]:
        if self._inner is not None:
            self._inner.invoke({})
        return {}


class _PlainChain(Chain):
    """A chain that never overrides the hook."""

    def _call(self, inputs: Dict[str, Any], run_manager=None) -> Dict[str, Any]:
        return {}


@pytest.fixture
def fetches(monkeypatch):
    seen = []

    def fake(_conn):
        seen.append(1)
        return "v1"

    monkeypatch.setattr(tu, "_get_ontology_version", fake)
    return seen


class TestRefresh:
    def test_a_question_forces_one_refresh(self, fetches):
        _RefreshingChain().invoke({})
        assert len(fetches) == 1

    def test_a_second_question_refreshes_again(self, fetches):
        chain = _RefreshingChain()
        chain.invoke({})
        chain.invoke({})
        assert len(fetches) == 2, "each question must see a fresh version"

    def test_an_edit_between_questions_is_visible_immediately(self, monkeypatch):
        version = ["v1"]
        monkeypatch.setattr(tu, "_get_ontology_version", lambda _c: version[0])

        calls = []

        @tu.cache_with_version_check
        def cached(conn_params):
            calls.append(1)
            return "x"

        class C(_RefreshingChain):
            def _call(self, inputs, run_manager=None):
                cached(conn_params=CONN)
                return {}

        chain = C()
        chain.invoke({})
        chain.invoke({})
        assert len(calls) == 1, "unchanged version should stay cached"

        # Ontology edited between questions — no waiting for the throttle.
        version[0] = "v2"
        chain.invoke({})
        assert len(calls) == 2, "the next question must see the edit"


class TestQuestionScope:
    def test_nested_chains_refresh_once_per_question(self, fetches):
        """An agent runs generate -> validate -> execute for one question."""
        inner = _RefreshingChain()
        middle = _RefreshingChain(inner=inner)
        outer = _RefreshingChain(inner=middle)

        outer.invoke({})

        assert len(fetches) == 1, (
            f"one question forced {len(fetches)} probes; the re-entrancy guard is not holding"
        )

    def test_scope_is_released_after_a_question(self, fetches):
        outer = _RefreshingChain(inner=_RefreshingChain())
        outer.invoke({})
        outer.invoke({})
        assert len(fetches) == 2

    def test_scope_is_released_even_when_the_chain_raises(self, fetches):
        class Boom(_RefreshingChain):
            def _call(self, inputs, run_manager=None):
                raise RuntimeError("boom")

        chain = Boom()
        with pytest.raises(RuntimeError):
            chain.invoke({})
        # A failed question must not leave the guard latched.
        _RefreshingChain().invoke({})
        assert len(fetches) == 2


class TestDegradation:
    def test_a_chain_without_the_override_does_not_probe(self, fetches):
        _PlainChain().invoke({})
        assert len(fetches) == 0, "the base hook must be a no-op"

    def test_a_probe_failure_does_not_fail_the_question(self, monkeypatch):
        """The refresh runs before the question — it must not be what breaks it.

        A real connectivity problem surfaces on the next actual query, with a
        better message than this pre-flight probe would give.
        """
        def explode(_conn):
            raise RuntimeError("timbr unreachable")

        monkeypatch.setattr(tu, "_get_ontology_version", explode)

        result = _RefreshingChain().invoke({})
        assert "chain_context" in result
