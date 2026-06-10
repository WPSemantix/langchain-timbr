"""Unit tests for ontology_context.paths."""

from __future__ import annotations

import pytest

from langchain_timbr.ontology_context.ontology.models import (
    ConceptMetadata,
    RelationshipAdditionalProperty,
    RelationshipMeta,
)
from langchain_timbr.ontology_context.ontology.paths import (
    format_relationship_path,
    list_relationship_paths,
)


def _rel(
    name: str,
    target: str,
    *,
    transitivity: int = 1,
    is_mtm: bool = False,
    target_properties: tuple[str, ...] = (),
    additional_properties: tuple[RelationshipAdditionalProperty, ...] = (),
) -> RelationshipMeta:
    return RelationshipMeta(
        name=name,
        target_concept=target,
        transitivity=transitivity,
        is_mtm=is_mtm,
        is_inverse=False,
        description=None,
        source_join_keys=(),
        target_join_keys=(),
        additional_properties=additional_properties,
        target_properties=target_properties,
    )


class TestFormatRelationshipPath:
    def test_bare(self):
        rel = _rel("made_order", "order")
        assert format_relationship_path(rel) == "made_order[order]"

    def test_target_property(self):
        rel = _rel("made_order", "order")
        assert format_relationship_path(rel, target_property="customer_name") == (
            "made_order[order].customer_name"
        )

    def test_additional_property(self):
        rel = _rel("has_employee", "person")
        assert format_relationship_path(rel, additional_property="title") == (
            "has_employee[person]_title"
        )

    def test_transitive_target_property(self):
        rel = _rel("has_acquired", "company", transitivity=2)
        assert format_relationship_path(rel, target_property="twitter_username") == (
            "has_acquired[company*2].twitter_username"
        )

    def test_transitive_additional_property(self):
        rel = _rel("has_acquired", "company", transitivity=2)
        assert format_relationship_path(rel, additional_property="acquisition_id") == (
            "has_acquired[company*2]_acquisition_id"
        )

    def test_both_target_and_additional_raises(self):
        rel = _rel("r", "t")
        with pytest.raises(ValueError):
            format_relationship_path(rel, target_property="x", additional_property="y")


class TestListRelationshipPaths:
    def _concept_meta(self) -> ConceptMetadata:
        rels = {
            "made_order": _rel(
                "made_order", "order",
                target_properties=("customer_name", "order_date"),
            ),
            "has_acquired": _rel(
                "has_acquired", "company", transitivity=2, is_mtm=True,
                target_properties=("name",),
                additional_properties=(
                    RelationshipAdditionalProperty(name="acquisition_id", data_type="bigint"),
                ),
            ),
            "owns_account": _rel(
                "owns_account", "account",
                additional_properties=(
                    RelationshipAdditionalProperty(name="role", data_type="varchar"),
                ),
            ),
        }
        return ConceptMetadata(
            name="customer",
            description=None,
            properties={},
            measures={},
            relationships=rels,
        )

    def test_full_listing(self):
        out = list_relationship_paths(self._concept_meta())
        # made_order bare + 2 target props
        assert "made_order[order]" in out
        assert "made_order[order].customer_name" in out
        assert "made_order[order].order_date" in out
        # has_acquired bare + 1 target + 1 additional
        assert "has_acquired[company*2]" in out
        assert "has_acquired[company*2].name" in out
        assert "has_acquired[company*2]_acquisition_id" in out
        # owns_account bare + 1 additional
        assert "owns_account[account]" in out
        assert "owns_account[account]_role" in out

    def test_filter_to_transitive_only(self):
        out = list_relationship_paths(
            self._concept_meta(), filter_fn=lambda r: r.transitivity > 1
        )
        assert all("[company*2" in p for p in out)
        assert not any(p.startswith("made_order") for p in out)
        assert not any(p.startswith("owns_account") for p in out)

    def test_filter_to_mtm_only(self):
        out = list_relationship_paths(
            self._concept_meta(), filter_fn=lambda r: r.is_mtm
        )
        assert all(p.startswith("has_acquired") for p in out)

    def test_disable_target_properties(self):
        out = list_relationship_paths(
            self._concept_meta(), include_target_properties=False
        )
        # No `.prop` suffix appears
        assert not any("." in p for p in out)
        # Bare and additional still appear
        assert "made_order[order]" in out
        assert "has_acquired[company*2]_acquisition_id" in out

    def test_disable_additional_properties(self):
        out = list_relationship_paths(
            self._concept_meta(), include_additional_properties=False
        )
        # No `_addname` suffix on bare bases
        assert "has_acquired[company*2]_acquisition_id" not in out
        assert "owns_account[account]_role" not in out
