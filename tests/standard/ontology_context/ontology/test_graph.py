"""Unit tests for ontology_context.graph.Ontology — uses an in-memory FakeClient."""

from __future__ import annotations

from langchain_timbr.ontology_context.ontology.graph import Ontology
from .._bulk_fixtures import bulk_rows_from_describe, merge_rel_rows


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
    def __init__(self, version="v1", relationships=None, fixtures=None,
                 inheritance_rows=None):
        self._version = version
        _derived_onto, _derived_rels = bulk_rows_from_describe(
            fixtures or {"customer": CUSTOMER_DESCRIBE, "order": ORDER_DESCRIBE}
        )
        self._relationships = merge_rel_rows(relationships, _derived_rels)
        self._fixtures = fixtures or {
            "customer": CUSTOMER_DESCRIBE,
            "order": ORDER_DESCRIBE,
        }
        self.version_calls = 0
        self.describe_calls: list[str] = []
        self.rels_calls = 0
        derived, _ = bulk_rows_from_describe(self._fixtures)
        # Tests that assert bulk-vs-describe agreement pass their own rows —
        # deriving them here would make the assertion circular.
        self._onto_rows = (
            inheritance_rows if inheritance_rows is not None else derived
        )

    def fetch_version_id(self):
        self.version_calls += 1
        return self._version

    def describe_concept(self, name):
        self.describe_calls.append(name)
        return list(self._fixtures.get(name, []))

    def fetch_relationships_meta(self):
        self.rels_calls += 1
        return list(self._relationships)

    def fetch_inheritance_meta(self):
        return list(self._onto_rows)

    # test-only mutator
    def bump_version(self, new_version: str):
        self._version = new_version


def _ontology(client):
    return Ontology(client)


class TestColdStart:
    def test_first_call_fetches_describe_and_rels_once(self):
        client = FakeClient()
        ontology = _ontology(client)
        meta = ontology.get_concept_metadata("customer")
        assert meta.name == "customer"
        assert client.describe_calls == ["customer"]
        assert client.rels_calls == 1
        # An Ontology belongs to one version and never probes for itself; the
        # version is decided when the instance is built (see shared.py).
        assert client.version_calls == 0


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

    def test_instance_never_probes_for_a_version(self):
        """Version handling belongs to shared.get_shared_ontology, not here.

        The instance used to hold its own TTL gate and empty itself in place.
        Two gates — one here, one in the shared probe — compound, so a change
        could take up to twice the window to notice; and clearing in place let a
        reader see a half-built lookup. Both are gone: the instance is now
        replaced wholesale. See test_shared_ontology.py.
        """
        client = FakeClient()
        ontology = _ontology(client)
        ontology.get_concept_metadata("customer")
        ontology.get_concept_metadata("customer")
        ontology.get_concept_metadata("order")
        assert client.version_calls == 0
        assert client.describe_calls == ["customer", "order"]
        assert client.rels_calls == 1


class TestVersionChange:
    """An instance does not react to a version change — it *is* replaced.

    The replacement itself is tested in test_shared_ontology.py; what matters
    here is that the instance keeps serving its own consistent generation, which
    is exactly what lets an in-flight request finish undisturbed.
    """

    def test_instance_keeps_serving_its_own_generation(self):
        client = FakeClient(version="v1")
        ontology = _ontology(client)
        ontology.get_concept_metadata("customer")
        ontology.get_concept_metadata("customer")  # cache hit
        assert client.describe_calls.count("customer") == 1

        client.bump_version("v2")
        ontology.get_concept_metadata("customer")
        assert client.describe_calls.count("customer") == 1, (
            "the instance must not empty itself underneath a reader"
        )
        assert client.rels_calls == 1


class TestRebuildSingleFlight:
    """Ruling 6 — one thread rebuilds, the rest wait.

    A fresh instance is handed to every in-flight thread at once. Without the
    single-flight each would run its own sys_concept_relationships fetch.
    """

    def test_concurrent_cold_callers_build_the_lookup_once(self):
        import threading

        client = FakeClient()
        ontology = _ontology(client)
        start = threading.Barrier(8)

        def worker():
            start.wait()
            ontology.get_concept_metadata("customer")

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert client.rels_calls == 1, (
            f"relationship lookup rebuilt {client.rels_calls} times, expected 1"
        )


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
    def test_returns_the_version_the_instance_was_built_for(self):
        ontology = Ontology(FakeClient(), version_id="abc123")
        assert ontology.show_version() == "abc123"
        assert ontology.version_id == "abc123"

    def test_version_id_does_not_trigger_a_build(self):
        """get_shared_ontology reads this on every call to decide staleness."""
        client = FakeClient()
        ontology = Ontology(client, version_id="v1")
        assert ontology.version_id == "v1"
        assert client.rels_calls == 0, "reading version_id must do no work"


class TestCardinalityOf:
    def test_cardinality_describes_nothing(self):
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
        # Nothing is described at all: the relationship comes from
        # sys_concept_relationships and both key sets from sys_ontology, so
        # cardinality costs no round-trip beyond the two bulk fetches.
        assert client.describe_calls == []
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


# ---------------------------------------------------------------------------
# Primary keys and relationship descriptions from the bulk tables
#
# These read sys_ontology / sys_concept_relationships instead of paying a
# `describe concept` round-trip, so the rows below are written by hand rather
# than derived from the describe fixtures — deriving them would make the
# assertions circular.
# ---------------------------------------------------------------------------


def _rel(concept, name, target, *, is_mtm=0, src="", tgt="", desc=""):
    return {
        "concept": concept, "relationship_name": name,
        "target_concept": target, "is_inverse": 0, "is_mtm": is_mtm,
        "source_properties": src, "target_properties": tgt,
        "description": desc,
    }


class TestPksOf:
    def test_reads_declared_keys_without_describing(self):
        client = FakeClient(inheritance_rows=[
            {"concept": "customer", "inheritance": "", "primary_keys": "id"},
            {"concept": "order", "inheritance": "", "primary_keys": "id,line"},
        ])
        ontology = _ontology(client)
        assert ontology.pks_of("customer") == {"id"}
        assert ontology.pks_of("order") == {"id", "line"}
        assert client.describe_calls == []

    def test_blank_keys_resolve_through_the_inheritance_chain(self):
        client = FakeClient(inheritance_rows=[
            {"concept": "thing", "inheritance": "", "primary_keys": "entity_id"},
            {"concept": "organization", "inheritance": "thing",
             "primary_keys": "org_id"},
            # Inherits its PK: the column is blank, the chain supplies it.
            {"concept": "company", "inheritance": "organization,thing",
             "primary_keys": ""},
        ])
        ontology = _ontology(client)
        assert ontology.pks_of("company") == {"org_id"}
        assert client.describe_calls == []

    def test_own_keys_win_over_an_ancestors(self):
        client = FakeClient(inheritance_rows=[
            {"concept": "organization", "inheritance": "", "primary_keys": "org_id"},
            {"concept": "company", "inheritance": "organization",
             "primary_keys": "company_number"},
        ])
        ontology = _ontology(client)
        assert ontology.pks_of("company") == {"company_number"}

    def test_unknown_concept_yields_an_empty_set(self):
        """No describe fallback: an absent concept degrades that relationship's
        cardinality to the default rather than costing a round-trip."""
        ontology = _ontology(FakeClient(inheritance_rows=[]))
        assert ontology.pks_of("nope") == set()


class TestCardinalityFromBulkPks:
    def _ontology_with(self, rels, pk_rows):
        return _ontology(FakeClient(relationships=rels, inheritance_rows=pk_rows))

    def test_mtm_wins_before_keys_are_consulted(self):
        ontology = self._ontology_with(
            [_rel("customer", "made_order", "order", is_mtm=1)],
            [{"concept": "customer", "inheritance": "", "primary_keys": "id"},
             {"concept": "order", "inheritance": "", "primary_keys": "id"}],
        )
        assert ontology.cardinality_of("customer", "made_order") == "N:M"

    def test_both_sides_match_their_keys_gives_one_to_one(self):
        ontology = self._ontology_with(
            [_rel("customer", "made_order", "order", src="id", tgt="id")],
            [{"concept": "customer", "inheritance": "", "primary_keys": "id"},
             {"concept": "order", "inheritance": "", "primary_keys": "id"}],
        )
        assert ontology.cardinality_of("customer", "made_order") == "1:1"

    def test_target_side_match_gives_n_to_one(self):
        ontology = self._ontology_with(
            [_rel("customer", "made_order", "order", src="name", tgt="id")],
            [{"concept": "customer", "inheritance": "", "primary_keys": "id"},
             {"concept": "order", "inheritance": "", "primary_keys": "id"}],
        )
        assert ontology.cardinality_of("customer", "made_order") == "N:1"

    def test_source_side_match_gives_one_to_n(self):
        ontology = self._ontology_with(
            [_rel("customer", "made_order", "order", src="id", tgt="total")],
            [{"concept": "customer", "inheritance": "", "primary_keys": "id"},
             {"concept": "order", "inheritance": "", "primary_keys": "id"}],
        )
        assert ontology.cardinality_of("customer", "made_order") == "1:N"

    def test_target_is_never_described(self):
        client = FakeClient(
            relationships=[_rel("customer", "made_order", "order",
                                src="name", tgt="id")],
            inheritance_rows=[
                {"concept": "customer", "inheritance": "", "primary_keys": "id"},
                {"concept": "order", "inheritance": "", "primary_keys": "id"},
            ],
        )
        ontology = _ontology(client)
        assert ontology.cardinality_of("customer", "made_order") == "N:1"
        assert "order" not in client.describe_calls

    def test_inherited_target_keys_still_resolve(self):
        client = FakeClient(
            relationships=[_rel("customer", "made_order", "order",
                                src="name", tgt="id")],
            inheritance_rows=[
                {"concept": "customer", "inheritance": "", "primary_keys": "name"},
                {"concept": "document", "inheritance": "", "primary_keys": "id"},
                {"concept": "order", "inheritance": "document", "primary_keys": ""},
            ],
        )
        ontology = _ontology(client)
        # target_join_keys {id} == order's inherited PKs {id} -> target match.
        assert ontology.cardinality_of("customer", "made_order") == "1:1"


class TestRelationshipDescription:
    def test_read_from_the_lookup_without_describing(self):
        client = FakeClient(relationships=[
            _rel("customer", "made_order", "order", desc="Customer's orders"),
        ])
        ontology = _ontology(client)
        assert (
            ontology.relationship_description("customer", "made_order")
            == "Customer's orders"
        )
        assert client.describe_calls == []

    def test_resolves_through_the_inheritance_chain(self):
        client = FakeClient(
            relationships=[_rel("organization", "has_employee", "person",
                                desc="staff")],
            inheritance_rows=[
                {"concept": "organization", "inheritance": "", "primary_keys": ""},
                {"concept": "company", "inheritance": "organization",
                 "primary_keys": ""},
            ],
        )
        ontology = _ontology(client)
        # Declared on the parent; the child must still resolve it.
        assert ontology.relationship_description("company", "has_employee") == "staff"

    def test_missing_relationship_yields_empty_string(self):
        ontology = _ontology(FakeClient(relationships=[]))
        assert ontology.relationship_description("customer", "nope") == ""

    def test_blank_description_yields_empty_string(self):
        client = FakeClient(relationships=[
            _rel("customer", "made_order", "order", desc=""),
        ])
        ontology = _ontology(client)
        assert ontology.relationship_description("customer", "made_order") == ""


# ---------------------------------------------------------------------------
# Walking the graph without describing it
# ---------------------------------------------------------------------------


class TestOutboundRelationships:
    def test_built_from_bulk_rows_without_describing(self):
        client = FakeClient(relationships=[
            _rel("customer", "made_order", "order", desc="orders"),
            _rel("customer", "lives_at", "address"),
        ])
        ontology = _ontology(client)
        rels = ontology.outbound_relationships("customer")
        assert set(rels) == {"made_order", "lives_at"}
        assert rels["made_order"].target_concept == "order"
        assert rels["made_order"].description == "orders"
        assert client.describe_calls == []

    def test_transitivity_sentinel_normalizes_to_one(self):
        """sys_concept_relationships uses -1 for "default"; describe reports 1.
        Read literally, every edge in the ontology looks different."""
        row = _rel("customer", "made_order", "order")
        row["transitivity"] = -1
        ontology = _ontology(FakeClient(relationships=[row]))
        assert ontology.outbound_relationships("customer")["made_order"].transitivity == 1

    def test_explicit_transitivity_is_preserved(self):
        row = _rel("customer", "made_order", "order")
        row["transitivity"] = 3
        ontology = _ontology(FakeClient(relationships=[row]))
        assert ontology.outbound_relationships("customer")["made_order"].transitivity == 3

    def test_inherited_relationships_resolve_for_the_child(self):
        client = FakeClient(
            relationships=[_rel("organization", "has_employee", "person")],
            inheritance_rows=[
                {"concept": "organization", "inheritance": "", "primary_keys": ""},
                {"concept": "company", "inheritance": "organization",
                 "primary_keys": ""},
            ],
        )
        ontology = _ontology(client)
        assert "has_employee" in ontology.outbound_relationships("company")

    def test_child_overrides_an_inherited_relationship(self):
        client = FakeClient(
            relationships=[
                _rel("organization", "has_employee", "person"),
                _rel("company", "has_employee", "staff_member"),
            ],
            inheritance_rows=[
                {"concept": "organization", "inheritance": "", "primary_keys": ""},
                {"concept": "company", "inheritance": "organization",
                 "primary_keys": ""},
            ],
        )
        ontology = _ontology(client)
        rels = ontology.outbound_relationships("company")
        assert rels["has_employee"].target_concept == "staff_member"

    def test_concept_with_no_rows_yields_no_relationships(self):
        ontology = _ontology(FakeClient(relationships=[]))
        assert ontology.outbound_relationships("nobody") == {}

    def test_a_malformed_row_costs_only_its_own_relationship(self):
        """The regression the describe-parser's poison-drop was written for: one
        bad definition must never take the concept's other relationships with it."""
        broken = _rel("customer", "broken_rel", "")   # no target
        client = FakeClient(relationships=[
            _rel("customer", "made_order", "order"),
            broken,
            _rel("customer", "lives_at", "address"),
        ])
        ontology = _ontology(client)
        rels = ontology.outbound_relationships("customer")
        assert set(rels) == {"made_order", "lives_at"}


class TestEdgeIndexIssuesNoDescribe:
    def test_materialize_walks_from_bulk_rows_alone(self):
        from langchain_timbr.ontology_context.context_builder.edge_index import (
            EdgeIndex,
        )
        client = FakeClient(
            relationships=[
                _rel("customer", "made_order", "order", src="name", tgt="id"),
                _rel("order", "contains", "product"),
            ],
            inheritance_rows=[
                {"concept": "customer", "inheritance": "", "primary_keys": "id"},
                {"concept": "order", "inheritance": "", "primary_keys": "id"},
                {"concept": "product", "inheritance": "", "primary_keys": "id"},
            ],
        )
        index = EdgeIndex(_ontology(client))
        edges = index.outbound_edges("customer")
        assert [(e.from_concept, e.relationship_name, e.to_concept) for e in edges] == [
            ("customer", "made_order", "order"),
        ]
        assert edges[0].cardinality == "N:1"
        assert client.describe_calls == [], (
            "walking the graph must not describe anything"
        )

    def test_unknown_concept_yields_no_edges_rather_than_raising(self):
        from langchain_timbr.ontology_context.context_builder.edge_index import (
            EdgeIndex,
        )
        index = EdgeIndex(_ontology(FakeClient(relationships=[])))
        assert index.outbound_edges("ghost") == []
