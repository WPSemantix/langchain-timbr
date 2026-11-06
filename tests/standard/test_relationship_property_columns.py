"""Many-to-many relationship properties (``has_employee[person]_title``) reach the
SQL-generation prompt on the static and the dynamic metadata paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import patch

from langchain_timbr.ontology_context.context_builder.metadata_types import (
    EdgeMeta,
    PathSegment,
    SelectedPath,
)
from langchain_timbr.ontology_context.context_builder.rebuild import build_relationships_from_paths
from langchain_timbr.ontology_context.context_builder.subgraph import (
    _build_rels_block,
    _relationship_props,
)
from langchain_timbr.ontology_context.ontology.models import RelationshipAdditionalProperty
from langchain_timbr.utils import timbr_utils
from langchain_timbr.utils.timbr_llm_utils import _partition_static_relationships_by_prefix
from langchain_timbr.utils.timbr_utils import split_relationship_column


@dataclass
class _Prop:
    name: str
    data_type: str = "varchar"
    description: str | None = None


@dataclass
class _Rel:
    name: str
    target_concept: str
    transitivity: int = 1
    additional_properties: tuple = ()


@dataclass
class _Concept:
    name: str
    properties: dict = field(default_factory=dict)
    measures: dict = field(default_factory=dict)
    relationships: dict = field(default_factory=dict)


class _Ontology:
    def __init__(self, concepts):
        self._concepts = concepts

    def get_concept_metadata(self, name):
        return self._concepts[name]

    def relationship_description(self, from_concept, rel_name):
        return ""

    def cardinality_of(self, from_concept, rel_name):
        return "N:M"


_TITLE = RelationshipAdditionalProperty(name="title", data_type="varchar")
_START = RelationshipAdditionalProperty(name="start_at", data_type="date")


def _crunchbase_ontology():
    return _Ontology({
        "organization": _Concept("organization", relationships={
            "has_employee": _Rel("has_employee", "person", additional_properties=(_TITLE, _START)),
        }),
        "person": _Concept("person", properties={"full_name": _Prop("full_name")}),
    })


class TestSplitRelationshipColumn:
    def test_target_property(self):
        assert split_relationship_column("has_employee[person].full_name") == ("has_employee[person]", "full_name")

    def test_relationship_property(self):
        assert split_relationship_column("has_employee[person]_start_at") == ("has_employee[person]", "start_at")

    def test_transitive_relationship_property(self):
        assert split_relationship_column("has_acquired[company*2]_acquisition_id") == (
            "has_acquired[company*2]", "acquisition_id",
        )

    def test_nested_relationship_property(self):
        assert split_relationship_column("a[x].has_employee[person]_title") == ("a[x].has_employee[person]", "title")

    def test_type_flag_of_target_is_a_target_property(self):
        assert split_relationship_column("has_employee[person]._type_of_usa_person") == (
            "has_employee[person]", "_type_of_usa_person",
        )


class TestStaticPath:
    def test_get_concept_properties_keeps_relationship_property_name(self):
        rows = [
            {"col_name": "has_employee[person]_start_at", "data_type": "date"},
            {"col_name": "has_employee[person].full_name", "data_type": "varchar"},
        ]
        with patch.object(timbr_utils, "run_query", return_value=rows):
            result = timbr_utils.get_concept_properties.__wrapped__(
                "organization", {}, properties_desc={"start_at": "When the job started"},
                relationships_desc={},
            )
        cols = result["relationships"]["has_employee"]["columns"]
        start = next(c for c in cols if c["name"] == "has_employee[person]_start_at")
        assert start["col_name"] == "start_at"
        assert start["comment"] == "When the job started"

    def test_partition_keeps_relationship_property_columns(self):
        static = {"has_employee": {"columns": [
            {"name": "has_employee[person]_title", "col_name": "title"},
            {"name": "has_employee[person].full_name", "col_name": "full_name"},
        ], "measures": []}}
        out = _partition_static_relationships_by_prefix(
            static, ontology=_crunchbase_ontology(), anchor="organization",
        )
        names = [c["name"] for c in out["has_employee[person]"]["columns"]]
        assert names == ["has_employee[person]_title", "has_employee[person].full_name"]


class TestDynamicPath:
    def test_rebuild_emits_relationship_properties(self):
        path = SelectedPath(path_id="P1", segments=[
            PathSegment(**{"from": "organization", "rel": "has_employee", "to": "person"}),
        ])
        out = build_relationships_from_paths([path], _crunchbase_ontology(), anchor="organization")
        cols = {c["name"]: c for c in out["has_employee[person]"]["columns"]}
        assert cols["has_employee[person]_title"]["col_name"] == "title"
        assert cols["has_employee[person]_start_at"]["data_type"] == "date"
        assert "has_employee[person].full_name" in cols

    def test_planner_rels_line_lists_relationship_properties(self):
        edge = EdgeMeta(
            from_concept="organization", relationship_name="has_employee", to_concept="person",
            transitivity=1, is_mtm=True, is_inverse=True, cardinality="N:M",
            description=None, is_self_ref=False,
        )
        rel_props = _relationship_props("organization", _crunchbase_ontology())
        assert rel_props == {"has_employee": ["start_at", "title"]}
        lines = _build_rels_block([edge], rel_props=rel_props)
        assert lines[1] == "  -[has_employee, N:M]-> person  # props: start_at, title"
