"""Unit tests for ontology_context.parser — classify + parse_describe_output."""

from __future__ import annotations

import pytest

from langchain_timbr.ontology_context.ontology.models import RelationshipLookupEntry
from langchain_timbr.ontology_context.ontology.parser import classify, parse_describe_output


class TestClassify:
    """One assertion per pattern. Mirrors the Plan 1 spec table."""

    def test_direct_property(self):
        assert classify("status") == ("direct", "status")

    def test_type_discriminator(self):
        cls = classify("_type_of_consumer_customer")
        assert cls[0] == "type_discriminator"

    def test_measure_direct(self):
        assert classify("measure.total_sales") == ("measure_direct", "total_sales")

    def test_measure_relationship_scoped(self):
        assert classify("measure.of_customer[customer].count_of_customer") == (
            "measure_rel",
            "of_customer",
            "customer",
            1,
            "count_of_customer",
            False,
        )

    def test_relationship_target_property(self):
        assert classify("made_order[order].customer_name") == (
            "rel_target_prop",
            "made_order",
            "order",
            1,
            "customer_name",
            False,
        )

    def test_relationship_additional_property(self):
        assert classify("has_employee[person]_title") == (
            "rel_additional",
            "has_employee",
            "person",
            1,
            "title",
            False,
        )

    def test_transitive_relationship_target_property(self):
        assert classify("has_acquired[company*2].twitter_username") == (
            "rel_target_prop",
            "has_acquired",
            "company",
            2,
            "twitter_username",
            False,
        )

    def test_transitive_relationship_additional_property(self):
        assert classify("has_acquired[company*2]_acquisition_id") == (
            "rel_additional",
            "has_acquired",
            "company",
            2,
            "acquisition_id",
            False,
        )

    def test_transitivity_level_marker(self):
        assert classify("has_acquired[company*2]_transitivity_level") == (
            "rel_transitivity_marker",
            "has_acquired",
            "company",
            2,
            False,
        )

    # ---- inverse-marker (~) cases -----------------------------------------

    def test_inverse_relationship_target_property(self):
        assert classify("~of_customer[customer].name") == (
            "rel_target_prop",
            "of_customer",
            "customer",
            1,
            "name",
            True,
        )

    def test_inverse_relationship_additional_property(self):
        assert classify("~of_employer[company]_role") == (
            "rel_additional",
            "of_employer",
            "company",
            1,
            "role",
            True,
        )

    # ---- error cases ------------------------------------------------------

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            classify("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError):
            classify("   ")

    def test_missing_close_bracket_raises(self):
        with pytest.raises(ValueError):
            classify("made_order[order.customer_name")

    def test_bad_transitivity_raises(self):
        with pytest.raises(ValueError):
            classify("has_acquired[company*notanint].x")


# ---- parse_describe_output ------------------------------------------------


def _row(col_name: str, *, data_type: str = "varchar", comment: str = "",
         inheritance_marker: str = "", key: str = "") -> dict:
    """Build a describe-output row. ``key`` carries the PK/FK signal —
    matches the live ``describe concept`` column name returned by Timbr."""
    return {
        "col_name": col_name,
        "data_type": data_type,
        "comment": comment,
        "inheritance_marker": inheritance_marker,
        "key": key,
    }


class TestParseDescribeOutputDirectProperties:
    def test_simple_property_collected(self):
        rows = [_row("status", data_type="varchar", comment="order status")]
        meta = parse_describe_output("order", rows, relationship_meta_lookup={})
        assert "status" in meta.properties
        p = meta.properties["status"]
        assert p.name == "status"
        assert p.data_type == "varchar"
        assert p.description == "order status"
        assert p.is_inherited is False
        assert p.is_pk is False
        assert p.is_fk is False

    def test_pk_flag_propagated(self):
        rows = [_row("id", data_type="bigint", key="PK")]
        meta = parse_describe_output("order", rows, relationship_meta_lookup={})
        assert meta.properties["id"].is_pk is True
        assert meta.properties["id"].is_fk is False

    def test_fk_flag_propagated(self):
        rows = [_row("customer_id", data_type="bigint", key="FK")]
        meta = parse_describe_output("order", rows, relationship_meta_lookup={})
        assert meta.properties["customer_id"].is_pk is False
        assert meta.properties["customer_id"].is_fk is True

    def test_inheritance_marker_propagated(self):
        rows = [_row("name", inheritance_marker="inherited")]
        meta = parse_describe_output("order", rows, relationship_meta_lookup={})
        assert meta.properties["name"].is_inherited is True

    def test_whitespace_description_becomes_none(self):
        rows = [_row("status", comment="   ")]
        meta = parse_describe_output("order", rows, relationship_meta_lookup={})
        assert meta.properties["status"].description is None

    def test_type_discriminator_skipped(self):
        rows = [_row("_type_of_consumer_customer")]
        meta = parse_describe_output("customer", rows, relationship_meta_lookup={})
        assert meta.properties == {}


class TestParseDescribeOutputMeasures:
    def test_direct_measure(self):
        rows = [_row("measure.total_sales", data_type="decimal")]
        meta = parse_describe_output("customer", rows, relationship_meta_lookup={})
        assert "total_sales" in meta.measures
        m = meta.measures["total_sales"]
        assert m.scoped_to_relationship is None
        assert m.data_type == "decimal"

    def test_relationship_scoped_measure(self):
        rows = [_row("measure.of_customer[customer].count_of_customer", data_type="bigint")]
        meta = parse_describe_output("order", rows, relationship_meta_lookup={})
        assert "of_customer.count_of_customer" in meta.measures
        m = meta.measures["of_customer.count_of_customer"]
        assert m.scoped_to_relationship == "of_customer"


class TestParseDescribeOutputRelationships:
    def test_relationship_target_property_collected(self):
        rows = [_row("made_order[order].customer_name", data_type="varchar")]
        meta = parse_describe_output("customer", rows, relationship_meta_lookup={})
        assert "made_order" in meta.relationships
        rel = meta.relationships["made_order"]
        assert rel.target_concept == "order"
        assert rel.transitivity == 1
        assert "customer_name" in rel.target_properties

    def test_relationship_additional_property_collected(self):
        rows = [_row("has_employee[person]_title", data_type="varchar")]
        meta = parse_describe_output("company", rows, relationship_meta_lookup={})
        rel = meta.relationships["has_employee"]
        assert len(rel.additional_properties) == 1
        assert rel.additional_properties[0].name == "title"
        assert rel.additional_properties[0].data_type == "varchar"

    def test_transitive_relationship(self):
        rows = [_row("has_acquired[company*2].twitter_username")]
        meta = parse_describe_output("company", rows, relationship_meta_lookup={})
        rel = meta.relationships["has_acquired"]
        assert rel.transitivity == 2

    def test_transitivity_level_marker_skipped(self):
        rows = [
            _row("has_acquired[company*2].twitter_username"),
            _row("has_acquired[company*2]_transitivity_level"),
        ]
        meta = parse_describe_output("company", rows, relationship_meta_lookup={})
        # The marker doesn't add an additional property
        rel = meta.relationships["has_acquired"]
        names = [ap.name for ap in rel.additional_properties]
        assert "transitivity_level" not in names

    def test_inverse_marker_from_column_prefix(self):
        rows = [_row("~of_customer[customer].name")]
        meta = parse_describe_output("order", rows, relationship_meta_lookup={})
        rel = meta.relationships["of_customer"]
        assert rel.is_inverse is True

    def test_missing_relationship_in_lookup_uses_defaults(self):
        rows = [_row("made_order[order].customer_name")]
        meta = parse_describe_output("customer", rows, relationship_meta_lookup={})
        rel = meta.relationships["made_order"]
        assert rel.is_mtm is False
        assert rel.description is None
        assert rel.source_join_keys == ()
        assert rel.target_join_keys == ()

    def test_lookup_entry_populates_relationship_meta(self):
        # RelationshipLookupEntry.description is expected to be pre-normalized
        # by the graph layer (graph.Ontology._build_rel_lookup strips and Noneify-s
        # blanks). The parser trusts the lookup as-is.
        rows = [_row("made_order[order].customer_name")]
        lookup = {
            ("customer", "made_order"): RelationshipLookupEntry(
                is_mtm=True,
                is_inverse=False,
                description="Customer's orders",
                source_join_keys=("customer_id",),
                target_join_keys=("customer_id",),
            )
        }
        meta = parse_describe_output("customer", rows, relationship_meta_lookup=lookup)
        rel = meta.relationships["made_order"]
        assert rel.is_mtm is True
        assert rel.description == "Customer's orders"
        assert rel.source_join_keys == ("customer_id",)
        assert rel.target_join_keys == ("customer_id",)

    def test_lookup_inverse_or_prefix_inverse_or(self):
        """is_inverse is True if EITHER the ~ prefix OR the lookup entry says so."""
        rows = [_row("made_order[order].name")]  # no ~ prefix
        lookup = {
            ("customer", "made_order"): RelationshipLookupEntry(
                is_mtm=False, is_inverse=True, description=None,
                source_join_keys=(), target_join_keys=(),
            )
        }
        meta = parse_describe_output("customer", rows, relationship_meta_lookup=lookup)
        assert meta.relationships["made_order"].is_inverse is True

    def test_inconsistent_target_raises(self):
        rows = [
            _row("rel_x[concept_a].p1"),
            _row("rel_x[concept_b].p2"),
        ]
        with pytest.raises(ValueError, match="inconsistent target_concept"):
            parse_describe_output("source", rows, relationship_meta_lookup={})

    def test_inconsistent_transitivity_raises(self):
        rows = [
            _row("has_x[c].p1"),
            _row("has_x[c*2].p2"),
        ]
        with pytest.raises(ValueError, match="inconsistent transitivity"):
            parse_describe_output("source", rows, relationship_meta_lookup={})


# ---------------------------------------------------------------------------
# Bug 1 — PK signal lives in the ``key`` column of describe output (NOT
# ``pk_marker``). Without this, every property's is_pk silently degrades
# to False, zeroing out PK sets downstream in derive_cardinality and
# collapsing every non-mtm relationship to the default '1:N'.
# ---------------------------------------------------------------------------


class TestParseDescribeOutputPkKeyField:
    def test_pk_from_key_field_sets_is_pk_true(self):
        rows = [{
            "col_name": "fund_id",
            "data_type": "varchar",
            "comment": "fund id",
            "key": "PK",
        }]
        meta = parse_describe_output("fund", rows, relationship_meta_lookup={})
        assert meta.properties["fund_id"].is_pk is True
        assert meta.properties["fund_id"].is_fk is False

    def test_fk_from_key_field_sets_is_fk_true(self):
        rows = [{
            "col_name": "fund_id",
            "data_type": "varchar",
            "comment": "fund id",
            "key": "FK",
        }]
        meta = parse_describe_output("company", rows, relationship_meta_lookup={})
        assert meta.properties["fund_id"].is_pk is False
        assert meta.properties["fund_id"].is_fk is True

    def test_none_in_key_field_yields_neither_pk_nor_fk(self):
        rows = [{
            "col_name": "name",
            "data_type": "varchar",
            "comment": "",
            "key": None,
        }]
        meta = parse_describe_output("any", rows, relationship_meta_lookup={})
        p = meta.properties["name"]
        assert p.is_pk is False
        assert p.is_fk is False

    def test_legacy_pk_marker_field_is_ignored(self):
        """Defensive: the OLD field name 'pk_marker' (which Timbr does NOT
        return) must NOT be picked up, even if a future caller emits it
        alongside the real ``key`` field."""
        rows = [{
            "col_name": "id",
            "data_type": "varchar",
            "comment": "",
            "pk_marker": "PK",   # legacy / wrong field — must be ignored
            # ``key`` intentionally absent
        }]
        meta = parse_describe_output("any", rows, relationship_meta_lookup={})
        assert meta.properties["id"].is_pk is False


# ---------------------------------------------------------------------------
# Bug 2 — Relationship meta lookup falls back through ``inheritance_chain``.
# A relationship declared on ``organization`` must be available on its
# child ``company`` with the parent's is_mtm / join-key signal intact, so
# inherited many-to-many cardinality derives correctly downstream.
# ---------------------------------------------------------------------------


class TestParseDescribeOutputInheritanceChain:
    def test_inherited_relationship_resolves_through_parent(self):
        """``has_employee`` is declared on organization; when parsing
        company, the parser must fall back to ``('organization',
        'has_employee')`` and surface the parent's is_mtm + join keys."""
        rows = [_row("has_employee[person].entity_id")]
        lookup = {
            ("organization", "has_employee"): RelationshipLookupEntry(
                is_mtm=True,
                is_inverse=False,
                description="employs",
                source_join_keys=("organization_id",),
                target_join_keys=("entity_id",),
            ),
        }
        meta = parse_describe_output(
            "company", rows,
            relationship_meta_lookup=lookup,
            inheritance_chain=("organization", "thing"),
        )
        rel = meta.relationships["has_employee"]
        assert rel.is_mtm is True
        assert rel.source_join_keys == ("organization_id",)
        assert rel.target_join_keys == ("entity_id",)
        assert rel.description == "employs"

    def test_direct_entry_wins_over_inherited(self):
        """When BOTH ``(child, rel)`` and ``(parent, rel)`` exist in the
        lookup, the direct entry must take precedence."""
        rows = [_row("has_office[office].entity_id")]
        lookup = {
            ("company", "has_office"): RelationshipLookupEntry(
                is_mtm=False,
                is_inverse=False,
                description="direct",
                source_join_keys=("company_id",),
                target_join_keys=("office_id",),
            ),
            ("organization", "has_office"): RelationshipLookupEntry(
                is_mtm=True,   # parent says m2m — but direct child wins
                is_inverse=False,
                description="inherited",
                source_join_keys=("parent",),
                target_join_keys=("parent",),
            ),
        }
        meta = parse_describe_output(
            "company", rows,
            relationship_meta_lookup=lookup,
            inheritance_chain=("organization",),
        )
        rel = meta.relationships["has_office"]
        assert rel.is_mtm is False
        assert rel.source_join_keys == ("company_id",)
        assert rel.description == "direct"

    def test_empty_chain_falls_through_to_default(self):
        """Backward-compat: when no chain is supplied AND no direct entry
        exists, the relationship gets the default branch (is_mtm=False,
        empty join keys) — unchanged from the legacy behavior."""
        rows = [_row("has_employee[person].entity_id")]
        # parent has the entry, but no inheritance chain is passed
        lookup = {
            ("organization", "has_employee"): RelationshipLookupEntry(
                is_mtm=True,
                is_inverse=False,
                description="employs",
                source_join_keys=("organization_id",),
                target_join_keys=("entity_id",),
            ),
        }
        meta = parse_describe_output(
            "company", rows,
            relationship_meta_lookup=lookup,
            # inheritance_chain defaults to () — no fallback walk
        )
        rel = meta.relationships["has_employee"]
        assert rel.is_mtm is False
        assert rel.source_join_keys == ()
        assert rel.target_join_keys == ()

    def test_chain_walked_in_order_stops_at_first_hit(self):
        """When the chain has multiple ancestors and BOTH parents have an
        entry, the FIRST (closer ancestor) wins."""
        rows = [_row("ancestor_rel[other].entity_id")]
        lookup = {
            ("immediate_parent", "ancestor_rel"): RelationshipLookupEntry(
                is_mtm=True, is_inverse=False, description="from_parent",
                source_join_keys=("a",), target_join_keys=("b",),
            ),
            ("thing", "ancestor_rel"): RelationshipLookupEntry(
                is_mtm=False, is_inverse=False, description="from_thing",
                source_join_keys=("x",), target_join_keys=("y",),
            ),
        }
        meta = parse_describe_output(
            "child", rows,
            relationship_meta_lookup=lookup,
            inheritance_chain=("immediate_parent", "thing"),
        )
        rel = meta.relationships["ancestor_rel"]
        assert rel.description == "from_parent"
        assert rel.is_mtm is True
