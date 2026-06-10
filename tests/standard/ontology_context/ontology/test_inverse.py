"""Unit tests for ontology_context.inverse.should_include_in_ddl."""

from __future__ import annotations

from langchain_timbr.ontology_context.ontology.inverse import should_include_in_ddl
from langchain_timbr.ontology_context.ontology.models import RelationshipMeta


def _rel(
    name: str = "r",
    target_concept: str = "t",
    *,
    is_inverse: bool = False,
) -> RelationshipMeta:
    return RelationshipMeta(
        name=name,
        target_concept=target_concept,
        transitivity=1,
        is_mtm=False,
        is_inverse=is_inverse,
        description=None,
        source_join_keys=(),
        target_join_keys=(),
        additional_properties=(),
        target_properties=(),
    )


class TestShouldIncludeInDDL:
    def test_anchor_keeps_everything(self):
        # No previous hop → anchor view, include every relationship.
        rel = _rel("of_customer", "customer", is_inverse=True)
        assert should_include_in_ddl(rel, current_concept="order", previous_hop_concept=None)

    def test_canonical_forward_edge_kept(self):
        rel = _rel("contains_product", "product", is_inverse=False)
        assert should_include_in_ddl(
            rel, current_concept="order", previous_hop_concept="customer"
        )

    def test_bounce_back_inverse_dropped(self):
        rel = _rel("of_customer", "customer", is_inverse=True)
        assert not should_include_in_ddl(
            rel, current_concept="order", previous_hop_concept="customer"
        )

    def test_self_ref_canonical_kept(self):
        rel = _rel("has_child", "work_item", is_inverse=False)
        assert should_include_in_ddl(
            rel, current_concept="work_item", previous_hop_concept="project"
        )

    def test_self_ref_inverse_kept(self):
        # The crucial case: has_parent IS an inverse but still reaches different
        # instances of work_item — must NOT be dropped.
        rel = _rel("has_parent", "work_item", is_inverse=True)
        assert should_include_in_ddl(
            rel, current_concept="work_item", previous_hop_concept="project"
        )
        assert should_include_in_ddl(
            rel, current_concept="work_item", previous_hop_concept="work_item"
        )

    def test_inverse_to_non_previous_concept_kept(self):
        # The inverse points somewhere we did NOT just come from → keep.
        rel = _rel("of_vendor", "vendor", is_inverse=True)
        assert should_include_in_ddl(
            rel, current_concept="order", previous_hop_concept="customer"
        )

    def test_canonical_edge_to_previous_hop_kept(self):
        # Canonical (non-inverse) edge that happens to target the previous hop —
        # this can occur in legitimate ontologies; the rule only filters INVERSE
        # bounce-backs.
        rel = _rel("contains_customer", "customer", is_inverse=False)
        assert should_include_in_ddl(
            rel, current_concept="order", previous_hop_concept="customer"
        )
