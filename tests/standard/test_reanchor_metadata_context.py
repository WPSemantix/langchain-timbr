"""Reanchor metadata-context regression test.

Mocks the "Which customer segment purchased the most metal material" scenario
where the Step-1 planner reanchors customer -> material. Asserts the rebuilt
SQL-gen context for the NEW anchor:

  1. The new anchor (material) DIRECT props/measures are present in the flat
     columns/measures blocks (Bug A — previously the stale customer columns
     leaked through).
  2. The intermediate concepts (product, order — marked is_intermediate) are
     stripped from the relationship block (waypoint filter, size-gate forced
     open here).
  3. The selected-path terminal (customer) columns remain, under their full
     nested prefix.
  4. effective_anchor is propagated as material.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import langchain_timbr.ontology_context as oc
from langchain_timbr.ontology_context import DynamicMetadataResult
from langchain_timbr.ontology_context.context_builder.metadata_types import (
    PathSegment,
    SelectedPath,
)
from langchain_timbr.utils.timbr_llm_utils import _apply_dynamic_metadata_context


# ---- Fake ontology helpers (mirror test_rebuild.py) -----------------------


@dataclass
class _FakeProp:
    name: str
    data_type: str = "string"
    description: str | None = None


@dataclass
class _FakeMeasure:
    name: str
    data_type: str = "int"
    description: str | None = None
    scoped_to_relationship: str | None = None


@dataclass
class _FakeRel:
    name: str
    target_concept: str
    transitivity: int = 1
    description: str | None = None


@dataclass
class _FakeConcept:
    name: str
    description: str | None = None
    properties: dict = field(default_factory=dict)
    measures: dict = field(default_factory=dict)
    relationships: dict = field(default_factory=dict)


class _FakeOntology:
    def __init__(self, concepts: dict[str, _FakeConcept]):
        self._concepts = concepts

    def get_concept_metadata(self, name: str):
        if name not in self._concepts:
            raise KeyError(name)
        return self._concepts[name]

    def cardinality_of(self, from_concept: str, rel_name: str):
        return "N:1"

    # _apply_dynamic_metadata_context memoizes via these — keep them no-ops.
    def get_filtered_cache(self, key):
        return None

    def set_filtered_cache(self, key, entry):
        return None


def _props(*names):
    return {n: _FakeProp(name=n) for n in names}


def _measures(*names):
    return {n: _FakeMeasure(name=n) for n in names}


def _build_ontology() -> _FakeOntology:
    material = _FakeConcept(
        name="material",
        description="materials used in products",
        properties=_props(
            "material_name", "materials", "supplier_location", "supplier_name",
            "price_per_1_kg", "material_id",
        ),
        measures=_measures("total_price_per_1_kg"),
        relationships={"used_in": _FakeRel("used_in", "product")},
    )
    product = _FakeConcept(
        name="product",
        properties=_props("category", "department", "image", "product_name", "product_id"),
        measures=_measures("count_of_product", "total_price"),
        relationships={"of_order": _FakeRel("of_order", "order")},
    )
    order = _FakeConcept(
        name="order",
        properties=_props("order_status", "order_city", "market", "order_id"),
        measures=_measures("total_sales", "count_of_order"),
        relationships={"of_customer": _FakeRel("of_customer", "customer")},
    )
    customer = _FakeConcept(
        name="customer",
        description="customer related info",
        properties=_props("customer_segment", "customer_name", "customer_email", "customer_id"),
        measures=_measures("count_of_customer", "count_first_names"),
        relationships={},
    )
    return _FakeOntology({
        "material": material, "product": product, "order": order, "customer": customer,
    })


def _seg(a, r, b, *, is_intermediate=False):
    return PathSegment(**{"from": a, "rel": r, "to": b, "is_intermediate": is_intermediate})


def _customer_flat_columns():
    # The flat columns the upstream static fetch loaded for the ORIGINAL anchor.
    return [
        {"name": "customer_segment", "col_name": "customer_segment", "data_type": "varchar"},
        {"name": "customer_name", "col_name": "customer_name", "data_type": "varchar"},
    ]


def _customer_flat_measures():
    return [
        {"name": "measure.count_of_customer", "col_name": "count_of_customer", "data_type": "bigint"},
    ]


class TestReanchorMetadataContext:
    def test_reanchor_refreshes_anchor_columns_and_strips_intermediates(self, monkeypatch):
        ontology = _build_ontology()

        path = SelectedPath(path_id="P1", segments=[
            _seg("material", "used_in", "product", is_intermediate=True),
            _seg("product", "of_order", "order", is_intermediate=True),
            _seg("order", "of_customer", "customer", is_intermediate=False),
        ])
        result = DynamicMetadataResult(
            filtered_concepts={"material", "product", "order", "customer"},
            path_rel_keys=set(),
            validated_paths=[path],
            compact_ddl="## CONCEPTS\n### material [anchor]\n",  # non-degraded (no cascade marker)
            stats={"resolved_by": "llm_paths"},
            warnings=[],
            error=None,
            accepted_overrides=[],
            effective_anchor="material",
        )

        monkeypatch.setattr(oc, "get_shared_ontology", lambda conn_params: ontology)
        monkeypatch.setattr(oc, "build_filtered_metadata", lambda **kwargs: result)

        columns_str, measures_str, rel_prop_str, effective_anchor = _apply_dynamic_metadata_context(
            mode="dynamic",
            question="Which customer segment purchased the most metal material",
            anchor="customer",                       # ORIGINAL anchor
            conn_params={},
            graph_depth=3,
            columns=_customer_flat_columns(),        # stale original-anchor flat cols
            measures=_customer_flat_measures(),
            tags={},
            exclude_properties=[],
            static_columns_str="customer_static_cols",
            static_measures_str="customer_static_meas",
            static_rel_prop_str="customer_static_rels",
            llm=None,
            config_overrides=dict(
                metadata_context_max_tokens=1,       # force the size gate open
                max_graph_depth=None,
                include_logic_concepts=None,
            ),
            tc_topup=None,
        )

        # 4. effective_anchor propagated.
        assert effective_anchor == "material"

        # 1. New anchor (material) flat props/measures present; stale customer
        #    flat columns gone.
        assert "material_name" in columns_str
        assert "supplier_name" in columns_str
        assert "customer_segment" not in columns_str          # stale flat col removed
        assert "total_price_per_1_kg" in measures_str
        assert "count_of_customer" not in measures_str          # stale flat measure removed

        # 2. Intermediate concepts (product, order) stripped from the rel block.
        assert "product_name" not in rel_prop_str
        assert "category" not in rel_prop_str
        assert "order_status" not in rel_prop_str
        assert "total_sales" not in rel_prop_str
        # material is the anchor — its own props never appear in the rel block.
        assert "material_name" not in rel_prop_str

        # 3. Selected-path terminal (customer) kept under its full nested prefix.
        assert "of_customer[customer].customer_segment" in rel_prop_str

    def test_no_reanchor_leaves_flat_columns_untouched(self, monkeypatch):
        """When the pipeline did NOT reanchor (effective_anchor is None), the
        passed-in flat columns must be used as-is (no re-source)."""
        ontology = _build_ontology()
        path = SelectedPath(path_id="P1", segments=[
            _seg("customer", "of_customer", "customer"),  # trivial; not exercised deeply
        ])
        result = DynamicMetadataResult(
            filtered_concepts={"customer"},
            path_rel_keys=set(),
            validated_paths=[path],
            compact_ddl="## CONCEPTS\n### customer [anchor]\n",
            stats={"resolved_by": "llm_paths"},
            effective_anchor=None,                    # no reanchor
        )
        monkeypatch.setattr(oc, "get_shared_ontology", lambda conn_params: ontology)
        monkeypatch.setattr(oc, "build_filtered_metadata", lambda **kwargs: result)

        columns_str, _measures_str, _rel, effective_anchor = _apply_dynamic_metadata_context(
            mode="dynamic",
            question="how many customers",
            anchor="customer",
            conn_params={},
            graph_depth=2,
            columns=_customer_flat_columns(),
            measures=_customer_flat_measures(),
            tags={},
            exclude_properties=[],
            static_columns_str="x",
            static_measures_str="x",
            static_rel_prop_str="x",
            llm=None,
            config_overrides=dict(metadata_context_max_tokens=12000),
            tc_topup=None,
        )
        assert effective_anchor is None
        assert "customer_segment" in columns_str        # original flat cols preserved
        assert "material_name" not in columns_str

    def test_selected_path_columns_get_description_and_tags(self, monkeypatch):
        """Non-reanchor: rebuilt selected-path columns must render descriptions
        (from properties_desc) and tags (from the property_tags dict). The fake
        ontology leaves prop.description=None, so descriptions can ONLY come from
        the properties_desc injection."""
        ontology = _build_ontology()
        path = SelectedPath(path_id="P1", segments=[
            _seg("material", "used_in", "product"),  # product is the terminal
        ])
        result = DynamicMetadataResult(
            filtered_concepts={"material", "product"},
            path_rel_keys=set(),
            validated_paths=[path],
            compact_ddl="## CONCEPTS\n### material [anchor]\n",
            stats={"resolved_by": "llm_paths"},
            effective_anchor=None,                    # no reanchor
        )
        monkeypatch.setattr(oc, "get_shared_ontology", lambda conn_params: ontology)
        monkeypatch.setattr(oc, "build_filtered_metadata", lambda **kwargs: result)

        _columns_str, _measures_str, rel_prop_str, _eff = _apply_dynamic_metadata_context(
            mode="dynamic",
            question="products of metal materials",
            anchor="material",
            conn_params={},
            graph_depth=2,
            columns=[{"name": "material_name", "col_name": "material_name", "data_type": "varchar"}],
            measures=[],
            tags={"product_name": {"PII": "no"}},
            exclude_properties=[],
            static_columns_str="x",
            static_measures_str="x",
            static_rel_prop_str="x",
            llm=None,
            config_overrides=dict(metadata_context_max_tokens=12000),  # no waypoint strip
            tc_topup=None,
            properties_desc={"product_name": "the product display name"},
        )

        assert "used_in[product].product_name" in rel_prop_str
        assert "description: the product display name" in rel_prop_str
        assert "annotations and constraints:" in rel_prop_str
        assert "PII" in rel_prop_str

    def test_reanchor_new_anchor_and_path_get_description_and_tags(self, monkeypatch):
        """Reanchor customer->material: the NEW anchor flat columns AND the
        selected-path terminal columns must render descriptions + tags."""
        ontology = _build_ontology()
        path = SelectedPath(path_id="P1", segments=[
            _seg("material", "used_in", "product", is_intermediate=True),
            _seg("product", "of_order", "order", is_intermediate=True),
            _seg("order", "of_customer", "customer", is_intermediate=False),
        ])
        result = DynamicMetadataResult(
            filtered_concepts={"material", "product", "order", "customer"},
            path_rel_keys=set(),
            validated_paths=[path],
            compact_ddl="## CONCEPTS\n### material [anchor]\n",
            stats={"resolved_by": "llm_paths"},
            effective_anchor="material",
        )
        monkeypatch.setattr(oc, "get_shared_ontology", lambda conn_params: ontology)
        monkeypatch.setattr(oc, "build_filtered_metadata", lambda **kwargs: result)

        columns_str, _measures_str, rel_prop_str, effective_anchor = _apply_dynamic_metadata_context(
            mode="dynamic",
            question="Which customer segment purchased the most metal material",
            anchor="customer",                       # ORIGINAL anchor
            conn_params={},
            graph_depth=3,
            columns=_customer_flat_columns(),
            measures=_customer_flat_measures(),
            tags={
                "material_name": {"Domain": "supply"},
                "customer_segment": {"PII": "yes"},
            },
            exclude_properties=[],
            static_columns_str="x",
            static_measures_str="x",
            static_rel_prop_str="x",
            llm=None,
            config_overrides=dict(metadata_context_max_tokens=12000),  # no waypoint strip
            tc_topup=None,
            properties_desc={
                "material_name": "the material name",
                "customer_segment": "segment of customer",
            },
        )

        assert effective_anchor == "material"
        # New anchor (material) flat column: description + tag.
        assert "material_name" in columns_str
        assert "description: the material name" in columns_str
        assert "Domain" in columns_str  # rendered "annotations and constraints: Domain - supply"
        # Selected-path terminal (customer) column: description + tag.
        assert "of_customer[customer].customer_segment" in rel_prop_str
        assert "description: segment of customer" in rel_prop_str
        assert "PII" in rel_prop_str

    def test_reanchor_direct_columns_get_technical_context(self, monkeypatch):
        """After reanchor, the NEW anchor's DIRECT (bare) columns must resolve
        their stats against the NEW anchor — the tc_topup closure captured the
        ORIGINAL anchor, so they were silently dropped. The fake tc_topup returns
        a stats annotation for `material_name` ONLY when bound to `material`,
        proving the effective anchor is threaded through."""
        ontology = _build_ontology()
        path = SelectedPath(path_id="P1", segments=[
            _seg("material", "used_in", "product"),
        ])
        result = DynamicMetadataResult(
            filtered_concepts={"material", "product"},
            path_rel_keys=set(),
            validated_paths=[path],
            compact_ddl="## CONCEPTS\n### material [anchor]\n",
            stats={"resolved_by": "llm_paths"},
            effective_anchor="material",
        )
        monkeypatch.setattr(oc, "get_shared_ontology", lambda conn_params: ontology)
        monkeypatch.setattr(oc, "build_filtered_metadata", lambda **kwargs: result)

        calls = {}

        def fake_tc_topup(cols, bound_concept=None):
            calls["bound_concept"] = bound_concept
            names = {c["name"] for c in cols}
            # The loader only resolves material's direct columns under material.
            if bound_concept == "material" and "material_name" in names:
                return {"material_name": "known values: ['Metal', 'Plastic']"}
            return {}

        columns_str, _measures_str, _rel, effective_anchor = _apply_dynamic_metadata_context(
            mode="dynamic",
            question="Which customer segment purchased the most metal material",
            anchor="customer",                       # ORIGINAL anchor
            conn_params={},
            graph_depth=3,
            columns=_customer_flat_columns(),
            measures=_customer_flat_measures(),
            tags={},
            exclude_properties=[],
            static_columns_str="x",
            static_measures_str="x",
            static_rel_prop_str="x",
            llm=None,
            config_overrides=dict(metadata_context_max_tokens=12000),  # no waypoint strip
            tc_topup=fake_tc_topup,
            tc_seen_names=set(),                     # so the flat columns are gathered
        )

        assert effective_anchor == "material"
        # The effective (reanchored) anchor was threaded to the top-up call.
        assert calls["bound_concept"] == "material"
        # The reanchored direct column rendered its statistics block.
        assert "material_name" in columns_str
        assert "statistics: known values: ['Metal', 'Plastic']" in columns_str
