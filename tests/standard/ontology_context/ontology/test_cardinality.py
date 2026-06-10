"""Unit tests for ontology_context.cardinality.derive_cardinality."""

from __future__ import annotations

from langchain_timbr.ontology_context.ontology.cardinality import derive_cardinality
from langchain_timbr.ontology_context.ontology.models import RelationshipMeta


def _rel(
    *,
    is_mtm: bool = False,
    source_join_keys: tuple[str, ...] = (),
    target_join_keys: tuple[str, ...] = (),
) -> RelationshipMeta:
    return RelationshipMeta(
        name="r",
        target_concept="t",
        transitivity=1,
        is_mtm=is_mtm,
        is_inverse=False,
        description=None,
        source_join_keys=source_join_keys,
        target_join_keys=target_join_keys,
        additional_properties=(),
        target_properties=(),
    )


class TestDeriveCardinality:
    def test_mtm_flag_wins(self):
        rel = _rel(is_mtm=True)
        assert derive_cardinality(rel, source_pks=set(), target_pks=set()) == "N:M"

    def test_mtm_wins_even_when_pks_match(self):
        rel = _rel(is_mtm=True, source_join_keys=("a",), target_join_keys=("b",))
        result = derive_cardinality(rel, source_pks={"a"}, target_pks={"b"})
        assert result == "N:M"

    def test_foreign_key_join_is_n_to_1(self):
        # source joins on FK (not PK), target joins on PK
        rel = _rel(source_join_keys=("customer_id",), target_join_keys=("id",))
        result = derive_cardinality(rel, source_pks={"order_id"}, target_pks={"id"})
        assert result == "N:1"

    def test_reverse_join_is_1_to_n(self):
        # source joins on PK, target joins on FK
        rel = _rel(source_join_keys=("id",), target_join_keys=("customer_id",))
        result = derive_cardinality(rel, source_pks={"id"}, target_pks={"order_id"})
        assert result == "1:N"

    def test_pk_to_pk_is_1_to_1(self):
        rel = _rel(source_join_keys=("id",), target_join_keys=("id",))
        result = derive_cardinality(rel, source_pks={"id"}, target_pks={"id"})
        assert result == "1:1"

    def test_empty_join_keys_default_to_1_to_n(self):
        rel = _rel()
        result = derive_cardinality(rel, source_pks={"id"}, target_pks={"id"})
        assert result == "1:N"

    def test_empty_target_keys_against_empty_target_pks_default(self):
        # Both empty must not "match" — default applies.
        rel = _rel(source_join_keys=("x",), target_join_keys=())
        result = derive_cardinality(rel, source_pks={"x"}, target_pks=set())
        # source match present, target empty → 1:N (source-match rule)
        assert result == "1:N"

    def test_composite_keys_set_equality(self):
        rel = _rel(source_join_keys=("a", "b"), target_join_keys=("c", "d"))
        result = derive_cardinality(
            rel, source_pks={"a", "b"}, target_pks={"c", "d"}
        )
        assert result == "1:1"

    def test_composite_keys_partial_match_is_default(self):
        rel = _rel(source_join_keys=("a",), target_join_keys=("c", "d"))
        result = derive_cardinality(
            rel, source_pks={"a", "b"}, target_pks={"c", "d"}
        )
        # source PKs are {a,b}, but join keys are only {a} — does NOT equal
        # target PKs match exactly → N:1
        assert result == "N:1"
