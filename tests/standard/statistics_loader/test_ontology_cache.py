"""Tests for ontology_cache module."""

import pytest
from unittest.mock import patch, MagicMock

from langchain_timbr.technical_context.statistics_loader.ontology_cache import (
    load_ontology_concepts,
    load_concept_mappings,
    load_view_row_counts,
)
from langchain_timbr.technical_context.statistics_loader.types import (
    OntologyConceptRow,
    ConceptMappingRow,
)


class TestLoadOntologyConcepts:
    """Test ontology concepts loading."""

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_basic_load(self, mock_run_query, conn_params):
        """Loads concepts from sys_ontology."""
        mock_run_query.return_value = [
            {"concept": "customer", "inheritance": "person", "query": None},
            {"concept": "person", "inheritance": "thing", "query": None},
            {"concept": "logic_concept", "inheritance": "", "query": "SELECT * FROM dtimbr.customer WHERE active = 1"},
        ]

        # Need to bypass cache for testing
        result = load_ontology_concepts.__wrapped__(conn_params)

        assert "customer" in result
        assert result["customer"].inheritance == "person"
        assert result["customer"].query is None
        assert result["logic_concept"].query == "SELECT * FROM dtimbr.customer WHERE active = 1"

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_missing_concept_field(self, mock_run_query, conn_params):
        """Rows with missing concept field are skipped."""
        mock_run_query.return_value = [
            {"concept": "valid", "inheritance": "", "query": None},
            {"concept": None, "inheritance": "", "query": None},
            {"inheritance": "thing", "query": None},  # no concept key
        ]

        result = load_ontology_concepts.__wrapped__(conn_params)
        assert len(result) == 1
        assert "valid" in result

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_empty_result(self, mock_run_query, conn_params):
        """Empty query result returns empty dict."""
        mock_run_query.return_value = []
        result = load_ontology_concepts.__wrapped__(conn_params)
        assert result == {}


class TestLoadConceptMappings:
    """Test concept mappings loading."""

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_basic_load(self, mock_run_query, conn_params):
        """Loads mappings grouped by concept."""
        mock_run_query.return_value = [
            {"concept": "customer", "mapping_name": "map_a", "number_of_rows": 1000},
            {"concept": "customer", "mapping_name": "map_b", "number_of_rows": 500},
            {"concept": "order", "mapping_name": "map_c", "number_of_rows": 5000},
        ]

        result = load_concept_mappings.__wrapped__(conn_params)

        assert len(result["customer"]) == 2
        assert result["customer"][0].mapping_name == "map_a"
        assert result["customer"][0].number_of_rows == 1000
        assert len(result["order"]) == 1

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_null_number_of_rows(self, mock_run_query, conn_params):
        """NULL number_of_rows defaults to -1."""
        mock_run_query.return_value = [
            {"concept": "customer", "mapping_name": "map_a", "number_of_rows": None},
        ]

        result = load_concept_mappings.__wrapped__(conn_params)
        assert result["customer"][0].number_of_rows == -1

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_empty_result(self, mock_run_query, conn_params):
        """Empty query result returns empty dict."""
        mock_run_query.return_value = []
        result = load_concept_mappings.__wrapped__(conn_params)
        assert result == {}


class TestLoadViewRowCounts:
    """Test view row counts loading.

    Row counts are projected from the shared per-user `get_views_only` fetch, so
    that is what these patch rather than the transport.
    """

    @patch("langchain_timbr.technical_context.statistics_loader.ontology_cache.get_views_only")
    def test_basic_load(self, mock_get_views, conn_params):
        """Loads view row counts."""
        mock_get_views.return_value = [
            {"view_name": "v_customers", "number_of_rows": 1000},
            {"view_name": "v_orders", "number_of_rows": 5000},
        ]

        result = load_view_row_counts.__wrapped__(conn_params)

        assert result["v_customers"] == 1000
        assert result["v_orders"] == 5000

    @patch("langchain_timbr.technical_context.statistics_loader.ontology_cache.get_views_only")
    def test_null_rows(self, mock_get_views, conn_params):
        """NULL number_of_rows defaults to -1."""
        mock_get_views.return_value = [
            {"view_name": "v_test", "number_of_rows": None},
        ]

        result = load_view_row_counts.__wrapped__(conn_params)
        assert result["v_test"] == -1


# ─── Properties-index freshness (Plan 02b) ──────────────────────────────────


class TestPropertiesIndexFreshness:
    """Two users, different permissions, one ontology.

    The mapping→properties index is a gate: a column absent from it is never
    asked about, so a column that gained statistics contributes no value hints
    until the index catches up. Its only other invalidation is the 1-hour
    per-user TTL. A probe — ``COUNT(*)`` and ``MAX(updated_at)`` over the rows
    *that caller can see* — narrows the window to the probe interval.

    Both aggregates are permission-filtered, which is why the probe is per user:
    a caller who cannot see the most recently updated mapping reports an older
    watermark, so one shared value would under-invalidate for everyone who can
    see more. These tests pin that the two users never disturb each other.
    """

    X = {"url": "http://t:11000", "token": "user-x", "ontology": "ont"}
    Y = {"url": "http://t:11000", "token": "user-y", "ontology": "ont"}

    # X sees two mappings, Y sees three. Y's extra one is the newest.
    X_ROWS = [{"target_name": "map_a", "property_name": "col_1"},
              {"target_name": "map_b", "property_name": "col_1"}]
    Y_ROWS = X_ROWS + [{"target_name": "map_c", "property_name": "col_1"}]

    def _world(self, state):
        """run_query stand-in: answers the probe and the index per caller."""
        def _dispatch(query, conn_params=None, *a, **k):
            who = (conn_params or {}).get("token")
            if "COUNT(*)" in query:
                state["probes"].append(who)
                n, wm = state["fingerprint"][who]
                return [{"n": n, "watermark": wm}]
            state["fetches"].append(who)
            return list(state["rows"][who])
        return _dispatch

    def _fresh_state(self):
        return {
            "probes": [], "fetches": [],
            "rows": {"user-x": list(self.X_ROWS), "user-y": list(self.Y_ROWS)},
            "fingerprint": {"user-x": (2, "2026-08-11 10:00:00"),
                            "user-y": (3, "2026-08-11 12:00:00")},
        }

    @pytest.fixture(autouse=True)
    def _isolate(self):
        from langchain_timbr.technical_context.statistics_loader import ontology_cache as oc
        from langchain_timbr.utils import timbr_utils as tu
        tu.clear_cache()
        oc._reset_index_probe_state()
        yield
        tu.clear_cache()
        oc._reset_index_probe_state()

    def _load(self, who):
        from langchain_timbr.technical_context.statistics_loader.ontology_cache import (
            load_mapping_properties_index,
        )
        return load_mapping_properties_index(who)

    @patch("langchain_timbr.utils.timbr_utils.get_ontology_version", return_value="v1")
    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_each_user_gets_only_their_own_mappings(self, run_query, _ver):
        state = self._fresh_state()
        run_query.side_effect = self._world(state)

        assert set(self._load(self.X)) == {"map_a", "map_b"}
        assert set(self._load(self.Y)) == {"map_a", "map_b", "map_c"}
        # one fetch each — the entries are separate, not shared
        assert state["fetches"] == ["user-x", "user-y"]

    @patch("langchain_timbr.utils.timbr_utils.get_ontology_version", return_value="v1")
    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_unchanged_fingerprint_never_refetches(self, run_query, _ver):
        state = self._fresh_state()
        run_query.side_effect = self._world(state)

        self._load(self.X)
        self._load(self.X)
        self._load(self.X)

        assert state["fetches"] == ["user-x"]          # fetched once
        assert state["probes"].count("user-x") == 1    # probed once, inside the interval

    @patch("langchain_timbr.utils.timbr_utils.get_ontology_version", return_value="v1")
    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_new_mapping_for_Y_refetches_only_Y(self, run_query, _ver):
        """Y gains a mapping — count moves for Y, X is untouched."""
        state = self._fresh_state()
        run_query.side_effect = self._world(state)
        self._load(self.X)
        self._load(self.Y)
        state["fetches"].clear()

        # a fourth mapping appears for Y only
        state["rows"]["user-y"].append({"target_name": "map_d", "property_name": "col_1"})
        state["fingerprint"]["user-y"] = (4, "2026-08-12 09:00:00")
        self._force_probe()

        assert set(self._load(self.Y)) == {"map_a", "map_b", "map_c", "map_d"}
        assert set(self._load(self.X)) == {"map_a", "map_b"}
        assert state["fetches"] == ["user-y"]          # X was never refetched

    @patch("langchain_timbr.utils.timbr_utils.get_ontology_version", return_value="v1")
    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_revoked_permission_for_X_refetches_only_X(self, run_query, _ver):
        """X loses a mapping — the count falls, which MAX alone would miss."""
        state = self._fresh_state()
        run_query.side_effect = self._world(state)
        self._load(self.X)
        self._load(self.Y)
        state["fetches"].clear()

        state["rows"]["user-x"] = [{"target_name": "map_a", "property_name": "col_1"}]
        # note the watermark is unchanged — only the count reveals this
        state["fingerprint"]["user-x"] = (1, "2026-08-11 10:00:00")
        self._force_probe()

        assert set(self._load(self.X)) == {"map_a"}
        assert set(self._load(self.Y)) == {"map_a", "map_b", "map_c"}
        assert state["fetches"] == ["user-x"]

    @patch("langchain_timbr.utils.timbr_utils.get_ontology_version", return_value="v1")
    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_new_stats_on_an_existing_mapping_refetches(self, run_query, _ver):
        """Same mappings, same row count — only the watermark moves."""
        state = self._fresh_state()
        run_query.side_effect = self._world(state)
        self._load(self.X)
        state["fetches"].clear()

        state["rows"]["user-x"] = [{"target_name": "map_a", "property_name": "col_1"},
                                   {"target_name": "map_b", "property_name": "col_2"}]
        state["fingerprint"]["user-x"] = (2, "2026-08-12 03:00:00")
        self._force_probe()

        assert self._load(self.X)["map_b"] == {"col_2"}
        assert state["fetches"] == ["user-x"]

    @patch("langchain_timbr.utils.timbr_utils.get_ontology_version", return_value="v1")
    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_probe_failure_keeps_the_cached_index(self, run_query, _ver):
        """No information is not the same as a change — fail open."""
        state = self._fresh_state()
        dispatch = self._world(state)

        def _flaky(query, conn_params=None, *a, **k):
            if "COUNT(*)" in query and state["fetches"]:
                raise Exception("probe blew up")
            return dispatch(query, conn_params, *a, **k)

        run_query.side_effect = _flaky
        self._load(self.X)
        state["fetches"].clear()
        self._force_probe()

        assert set(self._load(self.X)) == {"map_a", "map_b"}
        assert state["fetches"] == []                 # nothing was thrown away

    @patch("langchain_timbr.utils.timbr_utils.get_ontology_version", return_value="v1")
    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_ontology_with_no_statistics_is_a_real_answer(self, run_query, _ver):
        """NULL watermark means "no mapping statistics", not "unknown"."""
        state = self._fresh_state()
        state["rows"]["user-x"] = []
        state["fingerprint"]["user-x"] = (0, None)
        run_query.side_effect = self._world(state)

        assert self._load(self.X) == {}
        state["fetches"].clear()

        # statistics appear — NULL -> a timestamp must register as a change
        state["rows"]["user-x"] = list(self.X_ROWS)
        state["fingerprint"]["user-x"] = (2, "2026-08-12 04:00:00")
        self._force_probe()

        assert set(self._load(self.X)) == {"map_a", "map_b"}
        assert state["fetches"] == ["user-x"]

    @staticmethod
    def _force_probe():
        """Age every recorded probe past the interval."""
        from langchain_timbr.technical_context.statistics_loader import ontology_cache as oc
        for key, (_, n, wm) in list(oc._index_probe_state.items()):
            oc._index_probe_state[key] = (0.0, n, wm)
