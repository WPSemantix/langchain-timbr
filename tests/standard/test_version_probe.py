"""Shared ontology-version probe (Plan 06 #9, step 3).

The probe collapses the two independent SHOW VERSION round-trips (timbr_utils'
query cache and ontology_context's Ontology graph) into one shared fetch, while
each consumer keeps its own invalidation decision.
"""

import threading
import time
from unittest.mock import patch

import pytest

from langchain_timbr.utils import timbr_utils as tu


CONN_A = {"url": "https://x/", "ontology": "onto_a", "token": "t1"}
CONN_B = {"url": "https://x/", "ontology": "onto_b", "token": "t1"}


@pytest.fixture(autouse=True)
def _clean_probe():
    tu.clear_version_probe()
    yield
    tu.clear_version_probe()


class TestIdentity:
    def test_key_is_url_ontology_tenant(self):
        assert tu._ontology_identity(
            {"url": "u", "ontology": "o", "jwt_tenant_id": "t"}
        ) == ("u", "o", "t")

    def test_caller_and_transport_fields_are_not_part_of_identity(self):
        """Two callers on the same ontology must share one probe entry."""
        base = {"url": "u", "ontology": "o", "jwt_tenant_id": "t"}
        assert tu._ontology_identity({**base, "token": "aaa", "verify_ssl": False}) == \
               tu._ontology_identity({**base, "token": "bbb", "verify_ssl": True})

    def test_datasource_does_not_split_the_key(self):
        """Datasource selects where data is read, not which DDL version is deployed."""
        base = {"url": "u", "ontology": "o"}
        assert tu._ontology_identity({**base, "datasource": "d1"}) == \
               tu._ontology_identity({**base, "datasource": "d2"})

    def test_non_dict_conn_params_do_not_raise(self):
        assert tu._ontology_identity(None) == (None, None, None)


class TestThrottle:
    def test_second_call_inside_the_ttl_does_not_refetch(self):
        with patch.object(tu, "_get_ontology_version", return_value="v1") as fetch:
            assert tu.get_ontology_version(CONN_A) == "v1"
            assert tu.get_ontology_version(CONN_A) == "v1"
            assert tu.get_ontology_version(CONN_A) == "v1"
        assert fetch.call_count == 1

    def test_expired_entry_refetches(self):
        with patch.object(tu, "_get_ontology_version", side_effect=["v1", "v2"]) as fetch:
            assert tu.get_ontology_version(CONN_A) == "v1"
            # Age the entry past the throttle window.
            key = tu._ontology_identity(CONN_A)
            version, _ = tu._version_probe[key]
            tu._version_probe[key] = (version, time.time() - tu.cache_timeout - 1)
            assert tu.get_ontology_version(CONN_A) == "v2"
        assert fetch.call_count == 2

    def test_two_ontologies_are_probed_separately(self):
        with patch.object(tu, "_get_ontology_version", side_effect=["va", "vb"]) as fetch:
            assert tu.get_ontology_version(CONN_A) == "va"
            assert tu.get_ontology_version(CONN_B) == "vb"
        assert fetch.call_count == 2
        # ...and neither evicts the other.
        with patch.object(tu, "_get_ontology_version", side_effect=AssertionError):
            assert tu.get_ontology_version(CONN_A) == "va"
            assert tu.get_ontology_version(CONN_B) == "vb"


class TestForce:
    def test_force_bypasses_the_throttle(self):
        with patch.object(tu, "_get_ontology_version", side_effect=["v1", "v2"]) as fetch:
            assert tu.get_ontology_version(CONN_A) == "v1"
            assert tu.get_ontology_version(CONN_A, force=True) == "v2"
        assert fetch.call_count == 2

    def test_concurrent_forced_refreshes_collapse_to_one_fetch(self):
        """The once-per-question refresh must not become N round-trips."""
        calls = []
        start = threading.Barrier(8)

        def slow_fetch(_conn):
            calls.append(1)
            time.sleep(0.05)
            return "v1"

        def worker():
            start.wait()
            results.append(tu.get_ontology_version(CONN_A, force=True))

        results = []
        with patch.object(tu, "_get_ontology_version", side_effect=slow_fetch):
            threads = [threading.Thread(target=worker) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert results == ["v1"] * 8
        assert len(calls) == 1, f"expected one shared fetch, got {len(calls)}"


class TestBothConsumersShareOneFetch:
    """The point of #9: two consumers, one SHOW VERSION.

    ``timbr_utils``' query cache and ``ontology_context``'s Ontology graph used
    to throttle their own probes independently, so a cold run paid the ~1.3s
    round-trip twice.
    """

    def _count_show_version(self, monkeypatch):
        seen = []

        def fake_run_query(query, conn_params=None, *a, **kw):
            if "SHOW VERSION" in str(query).upper():
                seen.append(query)
                return [{"id": "v1"}]
            return []

        monkeypatch.setattr(tu, "run_query", fake_run_query)
        return seen

    def test_cold_run_issues_one_show_version_for_both_consumers(self, monkeypatch):
        from langchain_timbr.ontology_context.ontology.client import TimbrOntologyClient
        from langchain_timbr.ontology_context.ontology.graph import Ontology

        seen = self._count_show_version(monkeypatch)

        # Consumer 1 — the module-level query cache.
        assert tu.get_ontology_version(CONN_A) == "v1"

        # Consumer 2 — the Ontology graph, through its real client.
        ontology = Ontology(TimbrOntologyClient(CONN_A))
        assert ontology._client.fetch_version_id() == "v1"

        assert len(seen) == 1, f"expected one shared SHOW VERSION, got {len(seen)}"

    def test_each_ontology_still_gets_its_own_probe(self, monkeypatch):
        from langchain_timbr.ontology_context.ontology.client import TimbrOntologyClient

        seen = self._count_show_version(monkeypatch)

        tu.get_ontology_version(CONN_A)
        TimbrOntologyClient(CONN_B).fetch_version_id()

        assert len(seen) == 2, "two distinct ontologies must not share a version"


class TestFailureBackoff:
    """A server that is not answering must not be probed on every question.

    The once-per-question refresh bypasses the throttle by design, so without a
    backoff an unreachable Timbr would add a full connect timeout to every
    question — where the old throttled-only behaviour paid it once per window.
    """

    def test_a_failed_probe_is_not_retried_immediately(self):
        attempts = []

        def explode(_conn):
            attempts.append(1)
            raise RuntimeError("timbr unreachable")

        with patch.object(tu, "_get_ontology_version", side_effect=explode):
            with pytest.raises(RuntimeError):
                tu.get_ontology_version(CONN_A, force=True)
            # Subsequent questions back off rather than paying the timeout again.
            assert tu.get_ontology_version(CONN_A, force=True) is None
            assert tu.get_ontology_version(CONN_A) is None

        assert len(attempts) == 1, f"probed {len(attempts)} times while down"

    def test_backoff_serves_the_last_known_version(self):
        with patch.object(tu, "_get_ontology_version", return_value="v1"):
            assert tu.get_ontology_version(CONN_A) == "v1"

        def explode(_conn):
            raise RuntimeError("timbr unreachable")

        # Age the entry so the fast path misses and a real probe is attempted.
        key = tu._ontology_identity(CONN_A)
        tu._version_probe[key] = ("v1", time.time() - tu.cache_timeout - 1)

        with patch.object(tu, "_get_ontology_version", side_effect=explode):
            with pytest.raises(RuntimeError):
                tu.get_ontology_version(CONN_A, force=True)
            # Degrades to the last known value rather than to "no information".
            assert tu.get_ontology_version(CONN_A, force=True) == "v1"

    def test_backoff_is_per_ontology(self):
        """One ontology being down must not silence probes for another."""
        def only_a_fails(conn):
            if conn["ontology"] == "onto_a":
                raise RuntimeError("down")
            return "vb"

        with patch.object(tu, "_get_ontology_version", side_effect=only_a_fails):
            with pytest.raises(RuntimeError):
                tu.get_ontology_version(CONN_A, force=True)
            assert tu.get_ontology_version(CONN_B, force=True) == "vb"

    def test_a_recovered_server_clears_the_backoff(self):
        state = {"up": False}

        def flaky(_conn):
            if not state["up"]:
                raise RuntimeError("down")
            return "v9"

        with patch.object(tu, "_get_ontology_version", side_effect=flaky):
            with pytest.raises(RuntimeError):
                tu.get_ontology_version(CONN_A, force=True)
            state["up"] = True
            # Simulate the backoff window elapsing.
            tu._version_probe_failed[tu._ontology_identity(CONN_A)] = (
                time.time() - tu.cache_timeout - 1
            )
            assert tu.get_ontology_version(CONN_A, force=True) == "v9"
            assert tu._ontology_identity(CONN_A) not in tu._version_probe_failed


class TestConcurrency:
    def test_cold_concurrent_callers_produce_one_fetch(self):
        calls = []
        start = threading.Barrier(10)

        def slow_fetch(_conn):
            calls.append(1)
            time.sleep(0.05)
            return "v1"

        results = []

        def worker():
            start.wait()
            results.append(tu.get_ontology_version(CONN_A))

        with patch.object(tu, "_get_ontology_version", side_effect=slow_fetch):
            threads = [threading.Thread(target=worker) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert len(calls) == 1, f"expected one shared fetch, got {len(calls)}"

    def test_loser_of_a_cold_race_returns_none_not_a_guess(self):
        """None means 'no information' — the consumer must leave its cache alone.

        A guessed value here would look like a version change to the consumer and
        evict a cache that is perfectly good.
        """
        tu._version_probe_lock.acquire()
        try:
            assert tu.get_ontology_version(CONN_A) is None
        finally:
            tu._version_probe_lock.release()

    def test_loser_of_a_warm_race_returns_the_cached_value(self):
        with patch.object(tu, "_get_ontology_version", return_value="v1"):
            tu.get_ontology_version(CONN_A)
        # Age it out so the fast path misses, then hold the lock.
        key = tu._ontology_identity(CONN_A)
        tu._version_probe[key] = ("v1", time.time() - tu.cache_timeout - 1)
        tu._version_probe_lock.acquire()
        try:
            assert tu.get_ontology_version(CONN_A) == "v1"
        finally:
            tu._version_probe_lock.release()
