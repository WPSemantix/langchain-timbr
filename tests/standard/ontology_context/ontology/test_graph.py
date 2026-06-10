"""Unit tests for ontology_context.graph.Ontology — uses an in-memory FakeClient."""

from __future__ import annotations

import time

from langchain_timbr.ontology_context.ontology.graph import Ontology


def _row(col_name: str, *, data_type: str = "varchar", comment: str = "",
         inheritance_marker: str = "", pk_marker: str = "") -> dict:
    return {
        "col_name": col_name,
        "data_type": data_type,
        "comment": comment,
        "inheritance_marker": inheritance_marker,
        "pk_marker": pk_marker,
    }


CUSTOMER_DESCRIBE = [
    _row("id", data_type="bigint", pk_marker="PK"),
    _row("name", data_type="varchar"),
    _row("made_order[order].order_date", data_type="date"),
]

ORDER_DESCRIBE = [
    _row("id", data_type="bigint", pk_marker="PK"),
    _row("customer_id", data_type="bigint", pk_marker="FK"),
    _row("total", data_type="decimal"),
    _row("~of_customer[customer].name", data_type="varchar"),
]


class FakeClient:
    def __init__(self, version="v1", relationships=None, fixtures=None):
        self._version = version
        self._relationships = relationships or []
        self._fixtures = fixtures or {
            "customer": CUSTOMER_DESCRIBE,
            "order": ORDER_DESCRIBE,
        }
        self.version_calls = 0
        self.describe_calls: list[str] = []
        self.rels_calls = 0

    def fetch_version_id(self):
        self.version_calls += 1
        return self._version

    def describe_concept(self, name):
        self.describe_calls.append(name)
        return list(self._fixtures.get(name, []))

    def fetch_relationships_meta(self):
        self.rels_calls += 1
        return list(self._relationships)

    # test-only mutator
    def bump_version(self, new_version: str):
        self._version = new_version


def _ontology(client, *, ttl=120):
    return Ontology(client, version_ttl_seconds=ttl)


class TestColdStart:
    def test_first_call_fetches_describe_and_rels_once(self):
        client = FakeClient()
        ontology = _ontology(client)
        meta = ontology.get_concept_metadata("customer")
        assert meta.name == "customer"
        assert client.describe_calls == ["customer"]
        assert client.rels_calls == 1
        assert client.version_calls == 1


class TestCaching:
    def test_repeat_call_same_concept_hits_cache(self):
        client = FakeClient()
        ontology = _ontology(client)
        ontology.get_concept_metadata("customer")
        ontology.get_concept_metadata("customer")
        # Describe called once; rels lookup built once.
        assert client.describe_calls == ["customer"]
        assert client.rels_calls == 1

    def test_different_concept_reuses_relationship_lookup(self):
        client = FakeClient()
        ontology = _ontology(client)
        ontology.get_concept_metadata("customer")
        ontology.get_concept_metadata("order")
        assert client.describe_calls == ["customer", "order"]
        assert client.rels_calls == 1, "rels lookup must NOT refetch across concepts"

    def test_version_check_throttled_within_ttl(self):
        client = FakeClient()
        ontology = _ontology(client, ttl=3600)  # 1h window
        ontology.get_concept_metadata("customer")
        ontology.get_concept_metadata("customer")
        ontology.get_concept_metadata("order")
        # Only one version call total, because all three calls fall in the TTL window.
        assert client.version_calls == 1


class TestVersionChange:
    def test_version_change_clears_caches(self):
        client = FakeClient(version="v1")
        ontology = _ontology(client, ttl=0)  # always re-check on each public call
        ontology.get_concept_metadata("customer")
        ontology.get_concept_metadata("customer")  # cache hit
        assert client.describe_calls.count("customer") == 1

        # Now bump server-side version; next call should re-fetch.
        # Small sleep to ensure (now - last_check) > 0 even at ttl=0.
        time.sleep(0.01)
        client.bump_version("v2")
        ontology.get_concept_metadata("customer")
        assert client.describe_calls.count("customer") == 2
        assert client.rels_calls == 2


class TestInvalidate:
    def test_invalidate_forces_refetch(self):
        client = FakeClient()
        ontology = _ontology(client)
        ontology.get_concept_metadata("customer")
        assert client.describe_calls.count("customer") == 1

        ontology.invalidate()
        ontology.get_concept_metadata("customer")
        assert client.describe_calls.count("customer") == 2
        assert client.rels_calls == 2


class TestShowVersion:
    def test_returns_current_version(self):
        client = FakeClient(version="abc123")
        ontology = _ontology(client)
        assert ontology.show_version() == "abc123"


class TestCardinalityOf:
    def test_cardinality_fetches_source_and_target(self):
        # Set lookup so order.customer_id is FK; cardinality should resolve to N:1.
        rels = [
            {
                "concept": "customer",
                "relationship_name": "made_order",
                "target_concept": "order",
                "is_inverse": 0,
                "is_mtm": 0,
                "source_properties": "id",
                "target_properties": "customer_id",
                "description": "Customer's orders",
            },
        ]
        client = FakeClient(relationships=rels)
        ontology = _ontology(client)
        # We made join_keys: source=("id",) target=("customer_id",)
        # source PKs={id}, target PKs={id}
        # source_match = True (id == {id})
        # target_match = False ({customer_id} != {id})
        # → '1:N' (source-match rule)
        result = ontology.cardinality_of("customer", "made_order")
        assert result == "1:N"
        # Both concepts described once each, rels fetched once.
        assert sorted(client.describe_calls) == ["customer", "order"]
        assert client.rels_calls == 1

    def test_cardinality_of_mtm_returns_n_to_m(self):
        rels = [
            {
                "concept": "customer",
                "relationship_name": "made_order",
                "target_concept": "order",
                "is_inverse": 0,
                "is_mtm": 1,
                "source_properties": "",
                "target_properties": "",
                "description": "",
            },
        ]
        client = FakeClient(relationships=rels)
        ontology = _ontology(client)
        assert ontology.cardinality_of("customer", "made_order") == "N:M"


class TestRelationshipLookup:
    def test_inverse_flag_carried_from_lookup_to_relationship_meta(self):
        rels = [
            {
                "concept": "order",
                "relationship_name": "of_customer",
                "target_concept": "customer",
                "is_inverse": 1,
                "is_mtm": 0,
                "source_properties": "customer_id",
                "target_properties": "id",
                "description": "",
            },
        ]
        client = FakeClient(relationships=rels)
        ontology = _ontology(client)
        order_meta = ontology.get_concept_metadata("order")
        # The describe output already carries ~of_customer — and the lookup also
        # marks is_inverse=1; both should agree.
        assert "of_customer" in order_meta.relationships
        assert order_meta.relationships["of_customer"].is_inverse is True
