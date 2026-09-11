"""Per-ontology cache snapshots (Plan 06 #9 + C2, step 5).

The cache is replaced, not emptied in place. A caller captures its snapshot into
a local and reads *and writes* through that reference, so a slow request whose
version moved while it was out writes into the snapshot it started with — one
nobody reads any more — instead of contaminating the fresh one.
"""

import threading

import pytest

from langchain_timbr.utils import timbr_utils as tu


CONN = {"url": "https://x/", "ontology": "onto_a", "token": "t1"}
CONN_OTHER = {"url": "https://x/", "ontology": "onto_b", "token": "t1"}


@pytest.fixture(autouse=True)
def _clean():
    tu.clear_cache()
    tu.clear_version_probe()
    yield
    tu.clear_cache()
    tu.clear_version_probe()


class TestRacingReader:
    """The failure that passes every other test.

    Without the captured-pointer discipline this is silent: the multi-tenant
    test passes, the round-trip count passes, the whole suite passes, and the
    race the change exists to fix is still there.
    """

    def test_slow_writer_does_not_contaminate_the_new_snapshot(self, monkeypatch):
        version = ["v1"]
        monkeypatch.setattr(tu, "_get_ontology_version", lambda _c: version[0])

        inside = threading.Event()
        release = threading.Event()
        slow_calls = []

        @tu.cache_with_version_check
        def slow(conn_params):
            slow_calls.append(1)
            inside.set()
            release.wait(5)
            return "COMPUTED_UNDER_V1"

        @tu.cache_with_version_check
        def other(conn_params):
            return "other"

        # Thread A: misses the cache, enters the function, and parks there
        # holding the v1 snapshot.
        worker = threading.Thread(target=lambda: slow(conn_params=CONN))
        worker.start()
        assert inside.wait(5), "worker never entered the function"

        # The ontology version moves while A is still out.
        version[0] = "v2"
        tu.clear_version_probe()
        other(conn_params=CONN)  # trips the version change and swaps the snapshot

        # A comes back and writes its v1 result.
        release.set()
        worker.join(5)
        assert not worker.is_alive()

        # The stale value must not be readable from the current cache: asking
        # again has to recompute rather than hand back the pre-change answer.
        release.set()
        slow(conn_params=CONN)
        assert len(slow_calls) == 2, (
            "stale result computed under v1 was served from the post-change cache"
        )


class TestPerOntologyScoping:
    """C2 — a version change on one ontology must not wipe another's entries."""

    def test_version_bump_on_a_leaves_b_alone(self, monkeypatch):
        versions = {"onto_a": "a1", "onto_b": "b1"}
        monkeypatch.setattr(
            tu, "_get_ontology_version", lambda c: versions[c["ontology"]]
        )

        calls = {"a": 0, "b": 0}

        @tu.cache_with_version_check
        def load(conn_params):
            calls[conn_params["ontology"][-1]] += 1
            return conn_params["ontology"]

        load(conn_params=CONN)
        load(conn_params=CONN_OTHER)
        assert calls == {"a": 1, "b": 1}

        # Both are cached now.
        load(conn_params=CONN)
        load(conn_params=CONN_OTHER)
        assert calls == {"a": 1, "b": 1}

        # Move only A's version.
        versions["onto_a"] = "a2"
        tu.clear_version_probe()

        load(conn_params=CONN)
        assert calls["a"] == 2, "A's entry should have been dropped"
        load(conn_params=CONN_OTHER)
        assert calls["b"] == 1, "B's entry must survive A's version change"

    def test_alternating_ontologies_do_not_thrash(self, monkeypatch):
        """The C2 bug: one global scalar meant the recorded version usually
        belonged to the *other* ontology, so the comparison failed and the whole
        cache was dropped — roughly every cache_timeout, forever."""
        versions = {"onto_a": "a1", "onto_b": "b1"}
        monkeypatch.setattr(
            tu, "_get_ontology_version", lambda c: versions[c["ontology"]]
        )

        calls = []

        @tu.cache_with_version_check
        def load(conn_params):
            calls.append(conn_params["ontology"])
            return conn_params["ontology"]

        for _ in range(10):
            load(conn_params=CONN)
            load(conn_params=CONN_OTHER)

        assert len(calls) == 2, f"expected 2 cold fetches, got {len(calls)}: {calls}"


class TestExemptStore:
    """Plan 02 Ruling 5 — statistics-derived entries stay off the DDL version."""

    def test_version_change_does_not_evict_an_exempt_entry(self, monkeypatch):
        version = ["v1"]
        monkeypatch.setattr(tu, "_get_ontology_version", lambda _c: version[0])

        gated_calls, exempt_calls = [], []

        @tu.cache_with_version_check
        def gated(conn_params):
            gated_calls.append(1)
            return "gated"

        @tu.cache_with_version_check(version_gated=False)
        def exempt(conn_params):
            exempt_calls.append(1)
            return "exempt"

        gated(conn_params=CONN)
        exempt(conn_params=CONN)
        assert (len(gated_calls), len(exempt_calls)) == (1, 1)

        version[0] = "v2"
        tu.clear_version_probe()

        gated(conn_params=CONN)
        exempt(conn_params=CONN)
        assert len(gated_calls) == 2, "version-gated entry should be dropped"
        assert len(exempt_calls) == 1, (
            "exempt entry must survive a DDL version change (Plan 02 Ruling 5)"
        )

    def test_exempt_entry_survives_two_consecutive_version_changes(self, monkeypatch):
        """The bug that only shows up on the second swap."""
        version = ["v1"]
        monkeypatch.setattr(tu, "_get_ontology_version", lambda _c: version[0])

        exempt_calls = []

        @tu.cache_with_version_check(version_gated=False)
        def exempt(conn_params):
            exempt_calls.append(1)
            return "exempt"

        exempt(conn_params=CONN)
        for new_version in ("v2", "v3"):
            version[0] = new_version
            tu.clear_version_probe()
            exempt(conn_params=CONN)

        assert len(exempt_calls) == 1, "exemption was lost on a later swap"

    def test_exempt_per_user_entry_keeps_its_ttl(self, monkeypatch):
        """Losing the deadline would make it immortal — permission changes for
        that user would never be picked up."""
        monkeypatch.setattr(tu, "_get_ontology_version", lambda _c: "v1")

        calls = []

        @tu.cache_with_version_check(per_user=True, version_gated=False)
        def exempt_per_user(conn_params):
            calls.append(1)
            return "x"

        exempt_per_user(conn_params=CONN)
        assert len(calls) == 1

        # The entry must carry a deadline; age it out and it must refetch.
        aged = False
        for store in tu._all_expiry_maps():
            for key in list(store):
                store[key] = 0
                aged = True
        assert aged, "exempt per-user entry recorded no expiry deadline"

        exempt_per_user(conn_params=CONN)
        assert len(calls) == 2, "TTL did not apply to the exempt entry"
