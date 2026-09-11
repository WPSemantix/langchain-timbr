"""Cache-key tiering for the version-checked metadata cache.

Shared metadata is keyed on the ontology, not on the caller, so ten users on one
ontology fetch it once. The permission-filtered tables (the mapping list, the
view list, and what is derived from them) keep the caller in the key, so one
user never inherits another's rows.
"""

import pytest
from unittest.mock import patch

from langchain_timbr.utils import timbr_utils as tu
from langchain_timbr.technical_context.statistics_loader.ontology_cache import (
    load_concept_mappings,
    load_view_row_counts,
    load_ontology_concepts,
)
from langchain_timbr.ontology_context.ontology.shared import _cache_key


BASE = {
    "url": "http://timbr:11000",
    "token": "token-a",
    "ontology": "ont",
    "verify_ssl": True,
    "is_jwt": False,
    "jwt_tenant_id": None,
    "additional_headers": {"results-limit": "100"},
}


def conn(**overrides):
    """BASE with overrides; `headers` replaces additional_headers wholesale."""
    params = {**BASE, "additional_headers": dict(BASE["additional_headers"])}
    headers = overrides.pop("headers", None)
    if headers is not None:
        params["additional_headers"] = headers
    params.update(overrides)
    return params


class FakeTransport:
    """Counts Timbr queries and answers them per token.

    `rows_by_token` lets a test give two callers different server-side answers,
    which is how the permission tests tell a shared entry from a per-user one.
    """

    def __init__(self, rows_by_token=None, rows=None):
        self.rows_by_token = rows_by_token or {}
        self.rows = rows or []
        self.queries = []

    def __call__(self, **kwargs):
        query = kwargs["query"]
        if query.strip().upper() == "SHOW VERSION":
            return [{"id": "v1"}]

        self.queries.append(query)
        if query.strip().upper().startswith("SHOW TABLES IN DTIMBR"):
            return [{"tab_name": "customer"}]

        token = kwargs.get("token")
        if token in self.rows_by_token:
            return self.rows_by_token[token]
        return self.rows

    @property
    def count(self):
        return len(self.queries)


@pytest.fixture(autouse=True)
def _clean_cache():
    """Cold cache and a version probe that is due, before and after each test."""
    tu.clear_cache()
    tu.clear_version_probe()
    yield
    tu.clear_cache()
    tu.clear_version_probe()


# --------------------------------------------------------------------------- #
# shared tier - the caller is not part of the key
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "second",
    [
        pytest.param(conn(headers={"results-limit": "500"}), id="results-limit"),
        pytest.param(conn(token="token-b"), id="different-token"),
        pytest.param(
            conn(headers={"results-limit": "100", "x-api-impersonate-user": "bob"}),
            id="impersonated-user",
        ),
        pytest.param(conn(verify_ssl=False), id="verify-ssl"),
        pytest.param(conn(is_jwt=True), id="auth-mechanism"),
    ],
)
def test_shared_metadata_is_fetched_once_per_ontology(second):
    fake = FakeTransport(rows=[{"concept": "customer", "inheritance": "thing", "query": None}])
    with patch.object(tu, "_send_query", fake):
        load_ontology_concepts(conn_params=conn())
        load_ontology_concepts(conn_params=second)

    assert fake.count == 1


def test_shared_tier_still_separates_ontologies_and_tenants():
    """What a graph *is*: a server, an ontology on it, and the tenant behind it."""
    fake = FakeTransport(rows=[])
    with patch.object(tu, "_send_query", fake):
        load_ontology_concepts(conn_params=conn())
        load_ontology_concepts(conn_params=conn(ontology="other"))
        load_ontology_concepts(conn_params=conn(is_jwt=True, jwt_tenant_id="t1"))
        load_ontology_concepts(conn_params=conn(url="http://elsewhere:11000"))

    assert fake.count == 4


# --------------------------------------------------------------------------- #
# per-user tier - the caller stays in the key
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "second",
    [
        pytest.param(conn(token="token-b"), id="different-token"),
        pytest.param(
            conn(headers={"results-limit": "100", "x-api-impersonate-user": "bob"}),
            id="impersonated-user",
        ),
    ],
)
def test_permission_filtered_tables_are_fetched_per_caller(second):
    fake = FakeTransport(rows=[])
    with patch.object(tu, "_send_query", fake):
        load_concept_mappings(conn_params=conn())
        load_concept_mappings(conn_params=second)

    assert fake.count == 2


def test_per_user_tier_ignores_results_limit():
    """results-limit is stripped before the query is sent, so it cannot key anything."""
    fake = FakeTransport(rows=[])
    with patch.object(tu, "_send_query", fake):
        load_concept_mappings(conn_params=conn())
        load_concept_mappings(conn_params=conn(headers={"results-limit": "500"}))

    assert fake.count == 1


def test_mappings_of_one_user_never_reach_another():
    """The regression this whole split exists to prevent."""
    fake = FakeTransport(
        rows_by_token={
            "token-a": [
                {"concept": "customer", "mapping_name": "map_public", "number_of_rows": 10},
                {"concept": "customer", "mapping_name": "map_secret", "number_of_rows": 20},
            ],
            "token-b": [
                {"concept": "customer", "mapping_name": "map_public", "number_of_rows": 10},
            ],
        }
    )
    with patch.object(tu, "_send_query", fake):
        privileged = load_concept_mappings(conn_params=conn(token="token-a"))
        restricted = load_concept_mappings(conn_params=conn(token="token-b"))
        impersonated = load_concept_mappings(
            conn_params=conn(token="token-a", headers={"x-api-impersonate-user": "bob"})
        )

    assert {m.mapping_name for m in privileged["customer"]} == {"map_public", "map_secret"}
    assert {m.mapping_name for m in restricted["customer"]} == {"map_public"}
    # Impersonation is a caller identity too: it gets its own fetch, not token-a's.
    assert fake.count == 3
    assert impersonated is not privileged


def test_per_user_entry_expires_after_ttl():
    fake = FakeTransport(rows=[])
    with patch.object(tu, "_send_query", fake):
        load_concept_mappings(conn_params=conn())
        load_concept_mappings(conn_params=conn())
        assert fake.count == 1

        # Age every per-user entry past its deadline. The version probe is warm
        # from the call above and stays inside its throttle window, so the
        # re-fetch below is the TTL's doing and nothing else.
        for expiry in tu._all_expiry_maps():
            for key in list(expiry):
                expiry[key] = 0

        load_concept_mappings(conn_params=conn())

    assert fake.count == 2


def test_shared_entry_has_no_expiry():
    fake = FakeTransport(rows=[])
    with patch.object(tu, "_send_query", fake):
        load_ontology_concepts(conn_params=conn())

    assert all(expiry == {} for expiry in tu._all_expiry_maps())
    assert sum(len(cache) for cache in tu._all_cache_maps()) == 1


# --------------------------------------------------------------------------- #
# one sys_views query per caller, shared by every consumer
# --------------------------------------------------------------------------- #
VIEW_ROWS = [
    {
        "view_name": "v_sales",
        "description": "sales view",
        "is_cube": "false",
        "tables": "dtimbr.order",
        "number_of_rows": 42,
    }
]


def test_sys_views_is_queried_once_for_every_consumer():
    fake = FakeTransport(rows=VIEW_ROWS)
    with patch.object(tu, "_send_query", fake):
        tu.get_concepts(conn_params=conn(), views_list="*")
        counts = load_view_row_counts(conn_params=conn())
        again = tu.get_views_only(conn_params=conn())

    assert counts == {"v_sales": 42}
    assert len(again) == 1
    assert sum("sys_views" in q.lower() for q in fake.queries) == 1


def test_get_concepts_composes_concepts_then_views():
    concept_rows = [{"concept": "customer", "description": "a customer", "is_view": "false"}]

    def transport(**kwargs):
        query = kwargs["query"]
        if query.strip().upper() == "SHOW VERSION":
            return [{"id": "v1"}]
        if "sys_views" in query.lower():
            return VIEW_ROWS
        if query.strip().upper().startswith("SHOW TABLES IN DTIMBR"):
            return [{"tab_name": "customer"}]
        return concept_rows

    with patch.object(tu, "_send_query", transport):
        result = tu.get_concepts(conn_params=conn())

    assert list(result) == ["customer", "v_sales"]
    assert result["customer"]["is_view"] == "false"
    assert result["v_sales"] == {
        "concept": "v_sales",
        "description": "sales view",
        "is_view": "true",
    }


def test_get_concepts_rows_are_not_shared_between_calls():
    """Callers annotate these rows in place; the concept half is a shared cache entry."""
    fake = FakeTransport(
        rows=[{"concept": "customer", "description": "a customer", "is_view": "false"}]
    )
    with patch.object(tu, "_send_query", fake):
        first = tu.get_concepts(conn_params=conn(), concepts_list="*")
        first["customer"]["tags"] = "leaked"
        second = tu.get_concepts(conn_params=conn(token="token-b"), concepts_list="*")

    assert "tags" not in second["customer"]


def test_identify_concept_catalog_shares_its_concept_half():
    """Four concept-side queries for everyone; views and view properties per caller."""
    from langchain_timbr import identify_concept_context as icc

    fake = FakeTransport(rows=[])
    with patch.object(tu, "_send_query", fake):
        icc._load_catalog(conn())
        after_first = fake.count
        icc._load_catalog(conn())
        cached = fake.count
        icc._load_catalog(conn(token="token-b"))

    # concepts + properties + concept_properties + relationships + views + view_properties
    assert after_first == 6
    assert cached == 6
    # The second caller re-reads only the two permission-filtered sections.
    assert fake.count == 8
    assert sum("sys_view" in q.lower() for q in fake.queries) == 4


def test_show_tags_is_queried_once_per_ontology_version():
    """SHOW TAGS takes no filter, so include_tags must not key the fetch."""
    tag_rows = [
        {"target_type": "concept", "target_name": "customer",
         "tag_name": "alias", "tag_value": "client"},
        {"target_type": "concept", "target_name": "customer",
         "tag_name": "context", "tag_value": "crm"},
    ]
    fake = FakeTransport(rows=tag_rows)
    with patch.object(tu, "_send_query", fake):
        both = tu.get_tags(conn_params=conn(), include_tags=["alias", "context"])
        tu.get_tags(conn_params=conn(token="token-b"), include_tags=["alias", "context"])
        one = tu.get_tags(conn_params=conn(), include_tags=["alias"])
        none = tu.get_tags(conn_params=conn(), include_tags=None)

    assert sum("SHOW TAGS" in q.upper() for q in fake.queries) == 1
    # The pivot still honours each selection.
    assert both["concept_tags"]["customer"] == {"alias": "client", "context": "crm"}
    assert one["concept_tags"]["customer"] == {"alias": "client"}
    assert none == {"concept_tags": {}, "view_tags": {}, "property_tags": {}}


# --------------------------------------------------------------------------- #
# Ontology instances
# --------------------------------------------------------------------------- #
def test_ontology_instances_are_keyed_on_the_graph_only():
    # How a caller reached the graph does not make it a different graph.
    assert _cache_key(conn()) == _cache_key(conn(token="token-b"))
    assert _cache_key(conn()) == _cache_key(conn(verify_ssl=False))
    assert _cache_key(conn()) == _cache_key(conn(is_jwt=True))
    # What the graph is, does.
    assert _cache_key(conn()) != _cache_key(conn(ontology="other"))
    assert _cache_key(conn()) != _cache_key(conn(jwt_tenant_id="t1"))
    assert _cache_key(conn()) != _cache_key(conn(url="http://elsewhere:11000"))
