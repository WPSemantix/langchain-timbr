"""Shared Ontology factory — version-driven instance replacement (Plan 06, step 6).

When the ontology version moves the factory publishes a *new* Ontology and
leaves the previous one alone. A caller already holding the old instance keeps
using it until its request finishes, so it sees one consistent generation
instead of caches being emptied underneath it mid-flight.
"""

import threading

import pytest

from langchain_timbr.ontology_context.ontology.shared import (
    get_shared_ontology,
    reset_shared_ontologies,
)
from langchain_timbr.utils import timbr_utils as tu


CONN_A = {"url": "https://x/", "ontology": "onto_a", "token": "t1"}
CONN_B = {"url": "https://x/", "ontology": "onto_b", "token": "t1"}


@pytest.fixture(autouse=True)
def _clean():
    reset_shared_ontologies()
    tu.clear_version_probe()
    yield
    reset_shared_ontologies()
    tu.clear_version_probe()


@pytest.fixture
def version(monkeypatch):
    """A mutable server-side version, shared by every ontology unless keyed."""
    state = {"onto_a": "v1", "onto_b": "b1"}
    monkeypatch.setattr(tu, "_get_ontology_version", lambda c: state[c["ontology"]])
    return state


class TestReuse:
    def test_same_version_returns_the_same_instance(self, version):
        first = get_shared_ontology(CONN_A)
        second = get_shared_ontology(CONN_A)
        assert first is second

    def test_different_ontologies_get_different_instances(self, version):
        assert get_shared_ontology(CONN_A) is not get_shared_ontology(CONN_B)

    def test_caller_identity_does_not_split_the_instance(self, version):
        """token is not part of the graph's identity — only url/ontology/tenant."""
        a1 = get_shared_ontology({**CONN_A, "token": "aaa"})
        a2 = get_shared_ontology({**CONN_A, "token": "bbb"})
        assert a1 is a2


class TestReplacement:
    def test_version_change_publishes_a_new_instance(self, version):
        before = get_shared_ontology(CONN_A)
        version["onto_a"] = "v2"
        tu.clear_version_probe()
        after = get_shared_ontology(CONN_A)

        assert after is not before
        assert before.version_id == "v1"
        assert after.version_id == "v2"

    def test_the_previous_instance_stays_usable(self, version):
        """The whole point: an in-flight request finishes on its own generation."""
        held = get_shared_ontology(CONN_A)
        held.set_filtered_cache(("q",), "answer-under-v1")

        version["onto_a"] = "v2"
        tu.clear_version_probe()
        fresh = get_shared_ontology(CONN_A)

        assert held.get_filtered_cache(("q",)) == "answer-under-v1", (
            "the previous instance must not be emptied underneath its holder"
        )
        assert fresh.get_filtered_cache(("q",)) is None, (
            "the new generation must start clean"
        )

    def test_a_version_change_on_one_ontology_does_not_replace_another(self, version):
        a_before = get_shared_ontology(CONN_A)
        b_before = get_shared_ontology(CONN_B)

        version["onto_a"] = "v2"
        tu.clear_version_probe()

        assert get_shared_ontology(CONN_A) is not a_before
        assert get_shared_ontology(CONN_B) is b_before, (
            "B's graph must survive A's version change"
        )

    def test_probe_returning_none_keeps_the_current_instance(self, version, monkeypatch):
        """None is "no information", not a change — it must not churn instances."""
        held = get_shared_ontology(CONN_A)
        monkeypatch.setattr(tu, "get_ontology_version", lambda *_a, **_k: None)
        monkeypatch.setattr(
            "langchain_timbr.ontology_context.ontology.shared.get_ontology_version",
            lambda *_a, **_k: None,
        )
        assert get_shared_ontology(CONN_A) is held


class TestConcurrency:
    def test_concurrent_cold_callers_get_one_instance(self, version):
        start = threading.Barrier(10)
        seen = []

        def worker():
            start.wait()
            seen.append(get_shared_ontology(CONN_A))

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len({id(o) for o in seen}) == 1, "cold start produced duplicate graphs"

    def test_concurrent_callers_across_a_version_change_see_at_most_two(self, version):
        """Two threads detecting the same new version must not each publish."""
        get_shared_ontology(CONN_A)
        version["onto_a"] = "v2"
        tu.clear_version_probe()

        start = threading.Barrier(10)
        seen = []

        def worker():
            start.wait()
            seen.append(get_shared_ontology(CONN_A))

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len({id(o) for o in seen}) == 1, "duplicate replacements published"
        assert all(o.version_id == "v2" for o in seen)
