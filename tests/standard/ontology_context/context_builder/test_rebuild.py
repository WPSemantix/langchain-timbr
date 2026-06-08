"""Tests for rebuild.py — anti-hallucination construction helpers.

The constructor builds the SQL-gen relationships dict FROM the validated
paths + ontology metadata. Off-path chains literally cannot appear in the
output because no graph traversal happens here — we only emit direct
properties for each segment endpoint along each selected path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_timbr.ontology_context.context_builder.metadata_types import (
    PathSegment,
    SelectedPath,
    TransitivityOverride,
)
from langchain_timbr.ontology_context.context_builder.rebuild import (
    apply_transitivity_overrides,
    build_relationships_from_paths,
    collect_path_concepts,
    collect_path_relationships,
    filter_columns_for_concepts,
)


# ---- Fake ontology helpers ------------------------------------------------


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
    # Mirrors MeasureMeta: ``None`` for direct measures of this concept;
    # the rel name for measures reached via that relationship (parser stores
    # both kinds in ConceptMetadata.measures — see parser.py:204-213).
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
    properties: dict = field(default_factory=dict)
    measures: dict = field(default_factory=dict)
    relationships: dict = field(default_factory=dict)


class _FakeOntology:
    """Minimal stand-in for Ontology.get_concept_metadata."""

    def __init__(self, concepts: dict[str, _FakeConcept]):
        self._concepts = concepts

    def get_concept_metadata(self, name: str):
        if name not in self._concepts:
            raise KeyError(name)
        return self._concepts[name]


def _seg(a, r, b):
    return PathSegment(**{"from": a, "rel": r, "to": b})


# ---- Existing helper tests (unchanged behavior) ---------------------------


class TestCollectPathConcepts:
    def test_collects_both_endpoints(self):
        path = SelectedPath(path_id="P1", segments=[
            _seg("customer", "made_order", "order"),
            _seg("order", "contains_product", "product"),
        ])
        result = collect_path_concepts([path])
        assert result == {"customer", "order", "product"}

    def test_collects_sub_concept_expansions(self):
        path = SelectedPath(path_id="P1", segments=[
            _seg("customer", "made_order", "europe_order"),
        ])
        path.segments[0].expanded_sub_concepts = ["europe_order"]
        path.segments[0].modeled_target = "order"
        result = collect_path_concepts([path])
        assert "europe_order" in result
        assert "order" in result

    def test_empty_paths_returns_empty_set(self):
        assert collect_path_concepts([]) == set()


class TestCollectPathRelationships:
    def test_collects_triples(self):
        path = SelectedPath(path_id="P1", segments=[
            _seg("customer", "made_order", "order"),
            _seg("order", "contains_product", "product"),
        ])
        triples = collect_path_relationships([path])
        assert triples == {
            ("customer", "made_order", "order"),
            ("order", "contains_product", "product"),
        }


class TestFilterColumnsForConcepts:
    def test_keeps_matching_concepts(self):
        cols = [
            {"col_name": "x", "concept": "customer"},
            {"col_name": "y", "concept": "vendor"},
            {"col_name": "z", "concept": "order"},
        ]
        result = filter_columns_for_concepts(cols, {"customer", "order"})
        names = [c["col_name"] for c in result]
        assert names == ["x", "z"]

    def test_keeps_columns_without_concept_field(self):
        cols = [{"col_name": "x"}, {"col_name": "y"}]
        result = filter_columns_for_concepts(cols, {"order"})
        assert len(result) == 2

    def test_empty_path_concepts_returns_all(self):
        cols = [{"col_name": "x", "concept": "customer"}]
        assert filter_columns_for_concepts(cols, set()) == cols


# ---- Constructor tests ----------------------------------------------------


class TestBuildRelationshipsFromPaths:
    """The core anti-hallucination contract: output contains exactly the
    (prefix, concept) entries dictated by the paths, and nothing else."""

    def _organization_universe(self):
        """Shared fixture mirroring the user's crunchbase-like ontology.

        Includes a bunch of off-path concepts/rels (person, office, milestone,
        ipo, etc.) so we can assert they NEVER leak through."""
        organization = _FakeConcept(
            name="organization",
            properties={"organization_id": _FakeProp("organization_id")},
            relationships={
                "made_investment": _FakeRel(
                    "made_investment", "funding_round",
                    description="org wrote a funding round",
                ),
                # Off-path rels — must never be walked.
                "has_employee": _FakeRel("has_employee", "person"),
                "has_office": _FakeRel("has_office", "office"),
            },
        )
        funding_round = _FakeConcept(
            name="funding_round",
            properties={
                "funding_round_id": _FakeProp("funding_round_id"),
                "raised_amount_usd": _FakeProp("raised_amount_usd", "number"),
            },
            measures={"total_raised": _FakeMeasure("total_raised")},
            relationships={
                "funding_of": _FakeRel("funding_of", "company"),
                # Off-path: investor person — must not leak.
                "invested_by": _FakeRel("invested_by", "person"),
                "invested_from": _FakeRel("invested_from", "organization"),
            },
        )
        company = _FakeConcept(
            name="company",
            properties={
                "organization_name": _FakeProp("organization_name"),
                "_type_of_social_company": _FakeProp("_type_of_social_company"),
            },
            relationships={
                "acquired_by": _FakeRel("acquired_by", "company"),
                # Off-path rels at company.
                "reached_milestone": _FakeRel("reached_milestone", "milestone"),
                "basis_for_ipo": _FakeRel("basis_for_ipo", "ipo"),
            },
        )
        person = _FakeConcept(
            name="person",
            properties={"name": _FakeProp("name"), "title": _FakeProp("title")},
            relationships={"works_at2": _FakeRel("works_at2", "organization")},
        )
        # Off-path concepts that must never appear in any column name.
        office = _FakeConcept(name="office", properties={"city": _FakeProp("city")})
        milestone = _FakeConcept(name="milestone", properties={"name": _FakeProp("name")})
        ipo = _FakeConcept(name="ipo", properties={"date": _FakeProp("date")})
        return _FakeOntology({
            "organization": organization,
            "funding_round": funding_round,
            "company": company,
            "person": person,
            "office": office,
            "milestone": milestone,
            "ipo": ipo,
        })

    def test_three_segment_path_produces_exactly_three_prefix_buckets(self):
        """User's Sensobi case: a 3-segment path produces exactly 3 nested
        prefix entries (anchor properties are emitted separately into the
        flat columns list, NOT here). The constructor must NOT emit any
        4th-level or 5th-level nesting, nor any column referencing
        off-path rels (invested_from, invested_by, works_at2, has_employee,
        reached_milestone, basis_for_ipo, etc.)."""
        ontology = self._organization_universe()
        path = SelectedPath(path_id="P1", segments=[
            _seg("organization", "made_investment", "funding_round"),
            _seg("funding_round", "funding_of", "company"),
            _seg("company", "acquired_by", "company"),
        ])
        result = build_relationships_from_paths(
            [path], ontology, anchor="organization",
        )

        # Each hop is its own bucket keyed by the FULL accumulated prefix
        # (per-hop description + cardinality drove the re-keying away from
        # the original top-level-rel key). In this ontology
        # (_organization_universe), ``acquired_by`` has the default
        # transitivity=1 so no ``*N`` marker shows up on the last hop.
        assert set(result.keys()) == {
            "made_investment[funding_round]",
            "made_investment[funding_round].funding_of[company]",
            (
                "made_investment[funding_round].funding_of[company]"
                ".acquired_by[company]"
            ),
        }

        # The per-hop column / measure presence checks below are
        # structural — pool every bucket's items so the same assertions
        # apply regardless of which bucket each item lives in.
        col_names = [
            c["name"]
            for bucket in result.values()
            for c in bucket["columns"]
        ]
        meas_names = [
            m["name"]
            for bucket in result.values()
            for m in bucket["measures"]
        ]

        # All three segment endpoints contribute their DIRECT properties.
        assert "made_investment[funding_round].funding_round_id" in col_names
        assert "made_investment[funding_round].raised_amount_usd" in col_names
        assert (
            "made_investment[funding_round].funding_of[company].organization_name"
            in col_names
        )
        assert (
            "made_investment[funding_round].funding_of[company]"
            ".acquired_by[company].organization_name" in col_names
        )

        # Measures: funding_round has a measure, so it shows up under the
        # 1-hop prefix.
        assert (
            "measure.made_investment[funding_round].total_raised" in meas_names
        )

        # ANTI-HALLUCINATION: no off-path rels appear anywhere.
        forbidden = [
            "invested_from", "invested_by", "works_at2", "has_employee",
            "has_office", "reached_milestone", "basis_for_ipo",
        ]
        for forbidden_rel in forbidden:
            for n in col_names + meas_names:
                assert forbidden_rel not in n, (
                    f"off-path rel {forbidden_rel!r} leaked into {n!r}"
                )

        # ANTI-HALLUCINATION: no off-path concepts appear in any prefix.
        for forbidden_concept in ["person", "office", "milestone", "ipo"]:
            for n in col_names + meas_names:
                assert f"[{forbidden_concept}" not in n, (
                    f"off-path concept {forbidden_concept!r} in {n!r}"
                )

        # No chain has more than 3 bracket pairs — the path is 3 segments.
        for n in col_names + meas_names:
            assert n.count("[") <= 3, f"over-depth chain emitted: {n}"

    def test_anchor_properties_not_in_relationships_output(self):
        """The flat anchor columns go through filter_columns_for_concepts;
        the constructor must NOT also emit them, otherwise SQL-gen sees
        the anchor's own properties twice."""
        ontology = self._organization_universe()
        path = SelectedPath(path_id="P1", segments=[
            _seg("organization", "made_investment", "funding_round"),
        ])
        result = build_relationships_from_paths(
            [path], ontology, anchor="organization",
        )
        # Walk every per-prefix bucket — the anchor's columns must NOT
        # appear anywhere in the relationships dict.
        for bucket in result.values():
            for col in bucket.get("columns", []):
                assert col["col_name"] != "organization_id"

    def test_first_hop_description_populated_from_ontology(self):
        ontology = self._organization_universe()
        path = SelectedPath(path_id="P1", segments=[
            _seg("organization", "made_investment", "funding_round"),
        ])
        result = build_relationships_from_paths(
            [path], ontology, anchor="organization",
        )
        # Per-hop key: a single segment produces one bucket keyed by the
        # full prefix ``rel[target]``.
        assert (
            result["made_investment[funding_round]"]["description"]
            == "org wrote a funding round"
        )

    def test_transitivity_emitted_as_star_n_when_greater_than_one(self):
        """RelationshipMeta.transitivity > 1 must emit `rel[target*N]`. This
        keeps apply_transitivity_overrides useful — it can rewrite the
        existing *N marker if the LLM picked a different depth."""
        company = _FakeConcept(
            name="company",
            properties={"organization_name": _FakeProp("organization_name")},
            relationships={
                "has_acquired": _FakeRel(
                    "has_acquired", "company", transitivity=2,
                ),
            },
        )
        ontology = _FakeOntology({"company": company})
        path = SelectedPath(path_id="P1", segments=[
            _seg("company", "has_acquired", "company"),
        ])
        result = build_relationships_from_paths(
            [path], ontology, anchor="company",
        )
        col_names = [
            c["name"]
            for c in result["has_acquired[company*2]"]["columns"]
        ]
        assert col_names == ["has_acquired[company*2].organization_name"]

    def test_transitivity_one_emits_plain_target(self):
        """The common case: transitivity=1 means no *N marker."""
        ontology = self._organization_universe()
        path = SelectedPath(path_id="P1", segments=[
            _seg("organization", "made_investment", "funding_round"),
        ])
        result = build_relationships_from_paths(
            [path], ontology, anchor="organization",
        )
        col_names = [
            c["name"]
            for c in result["made_investment[funding_round]"]["columns"]
        ]
        # No `*N` markers in any emitted chain.
        for n in col_names:
            assert "*" not in n

    def test_two_multi_segment_paths_emit_their_own_chains(self):
        """Multi-segment paths from the same anchor produce their respective
        nested chains; the normalization pre-pass is a no-op here because
        both paths already start at the anchor."""
        customer = _FakeConcept(
            name="customer", properties={"customer_segment": _FakeProp("customer_segment")},
        )
        order = _FakeConcept(
            name="order", properties={"order_id": _FakeProp("order_id")},
        )
        product = _FakeConcept(
            name="product", properties={"product_name": _FakeProp("product_name")},
            relationships={"contains": _FakeRel("contains", "material")},
        )
        material = _FakeConcept(
            name="material", properties={"material_name": _FakeProp("material_name")},
        )
        ontology = _FakeOntology({
            "customer": customer, "order": order,
            "product": product, "material": material,
        })

        # P2 explicitly traverses order → product → material in one path
        # → the cross-segment chain shows up under includes_product.
        paths = [
            SelectedPath(path_id="P1", segments=[
                _seg("order", "of_customer", "customer"),
            ]),
            SelectedPath(path_id="P2", segments=[
                _seg("order", "includes_product", "product"),
                _seg("product", "contains", "material"),
            ]),
        ]
        result = build_relationships_from_paths(paths, ontology, anchor="order")

        # Per-hop keys: P1's one segment → 1 bucket; P2's two segments → 2 buckets.
        assert set(result.keys()) == {
            "of_customer[customer]",
            "includes_product[product]",
            "includes_product[product].contains[material]",
        }
        # Hop 1 of P2 holds product's direct cols.
        ip_names = [
            c["name"] for c in result["includes_product[product]"]["columns"]
        ]
        assert "includes_product[product].product_name" in ip_names
        # Cross-segment chain emitted because P2 walks both hops — material's
        # direct cols now sit in their OWN per-hop bucket.
        deeper = [
            c["name"]
            for c in result["includes_product[product].contains[material]"]["columns"]
        ]
        assert (
            "includes_product[product].contains[material].material_name"
            in deeper
        )

    def test_multiple_paths_with_same_first_hop_share_bucket(self):
        """Two paths starting with the same first hop must collapse under
        a single top-level key — _build_rel_columns_str iterates keys once."""
        a = _FakeConcept(
            name="a", properties={"x": _FakeProp("x")},
            relationships={"r1": _FakeRel("r1", "b")},
        )
        b = _FakeConcept(
            name="b", properties={"y": _FakeProp("y")},
            relationships={
                "r2": _FakeRel("r2", "c"),
                "r3": _FakeRel("r3", "d"),
            },
        )
        c = _FakeConcept(name="c", properties={"z": _FakeProp("z")})
        d = _FakeConcept(name="d", properties={"w": _FakeProp("w")})
        ontology = _FakeOntology({"a": a, "b": b, "c": c, "d": d})
        paths = [
            SelectedPath(path_id="P1", segments=[
                _seg("a", "r1", "b"),
                _seg("b", "r2", "c"),
            ]),
            SelectedPath(path_id="P2", segments=[
                _seg("a", "r1", "b"),
                _seg("b", "r3", "d"),
            ]),
        ]
        result = build_relationships_from_paths(paths, ontology, anchor="a")
        # Per-hop keys: P1 → r1[b], r1[b].r2[c]; P2 → r1[b] (dedup'd via
        # seen_prefixes), r1[b].r3[d].
        assert set(result.keys()) == {
            "r1[b]", "r1[b].r2[c]", "r1[b].r3[d]",
        }
        # Pool all bucket columns to assert presence + the b-side dedup.
        names = [
            c["name"]
            for bucket in result.values()
            for c in bucket["columns"]
        ]
        # b's `y` shows up exactly once thanks to the seen-prefix dedup
        # (P2's first hop r1[b] repeats P1's first hop).
        assert "r1[b].y" in names
        assert names.count("r1[b].y") == 1
        assert "r1[b].r2[c].z" in names
        assert "r1[b].r3[d].w" in names

    def test_missing_concept_metadata_is_tolerated(self):
        """If ontology.get_concept_metadata raises for a concept, that
        segment contributes no properties but the rest of the path still
        emits. The rebuild must NOT crash on a partial ontology."""
        a = _FakeConcept(
            name="a", properties={"x": _FakeProp("x")},
            relationships={"r1": _FakeRel("r1", "missing")},
        )
        # 'missing' concept is absent from the fake ontology → KeyError.
        ontology = _FakeOntology({"a": a})
        path = SelectedPath(path_id="P1", segments=[
            _seg("a", "r1", "missing"),
        ])
        result = build_relationships_from_paths([path], ontology, anchor="a")
        # Bucket exists (created before the metadata lookup) but is empty.
        assert "r1[missing]" in result
        assert result["r1[missing]"]["columns"] == []
        assert result["r1[missing]"]["measures"] == []

    def test_empty_paths_returns_empty_dict(self):
        ontology = self._organization_universe()
        assert build_relationships_from_paths(
            [], ontology, anchor="organization",
        ) == {}

    def test_path_with_no_segments_is_skipped(self):
        ontology = self._organization_universe()
        path = SelectedPath(path_id="P1", segments=[])
        assert build_relationships_from_paths(
            [path], ontology, anchor="organization",
        ) == {}


# ---- Anchor-rooted prefix bug: Test 1, Test 2, Test 3 --------------------


def _supply_chain_universe():
    """Customer → order → product → material mini-ontology with extra
    off-path rels/concepts so we can assert they never leak."""
    customer = _FakeConcept(
        name="customer",
        properties={
            "customer_id": _FakeProp("customer_id"),
            "customer_segment": _FakeProp("customer_segment"),
        },
        relationships={
            "made_order": _FakeRel("made_order", "order"),
            # Off-path
            "received_shipment": _FakeRel("received_shipment", "shipment"),
        },
    )
    order = _FakeConcept(
        name="order",
        properties={
            "order_id": _FakeProp("order_id"),
            "order_date": _FakeProp("order_date"),
        },
        relationships={
            "includes_product": _FakeRel("includes_product", "product"),
            # Off-path
            "of_inventory": _FakeRel("of_inventory", "inventory"),
        },
    )
    product = _FakeConcept(
        name="product",
        properties={
            "product_id": _FakeProp("product_id"),
            "product_name": _FakeProp("product_name"),
        },
        relationships={
            "contains": _FakeRel("contains", "material"),
            # Off-path
            "has_bill_of_material": _FakeRel("has_bill_of_material", "bom"),
        },
    )
    material = _FakeConcept(
        name="material",
        properties={
            "material_id": _FakeProp("material_id"),
            "material_name": _FakeProp("material_name"),
        },
    )
    # Off-path concepts.
    shipment = _FakeConcept(name="shipment", properties={"x": _FakeProp("x")})
    inventory = _FakeConcept(name="inventory", properties={"y": _FakeProp("y")})
    bom = _FakeConcept(name="bom", properties={"z": _FakeProp("z")})
    return _FakeOntology({
        "customer": customer, "order": order,
        "product": product, "material": material,
        "shipment": shipment, "inventory": inventory, "bom": bom,
    })


def _sensobi_universe():
    """Organization → funding_round → company →* company mini-ontology.

    ``acquired_by`` is configured with ontology transitivity=4 so the
    constructor emits ``acquired_by[company*4]`` directly; this matches the
    post-override depth the LLM picks in the real Sensobi case (see
    apply_transitivity_overrides for the rewrite path)."""
    organization = _FakeConcept(
        name="organization",
        properties={"organization_id": _FakeProp("organization_id")},
        relationships={
            "made_investment": _FakeRel("made_investment", "funding_round"),
            # Off-path
            "has_employee": _FakeRel("has_employee", "person"),
            "has_office": _FakeRel("has_office", "office"),
            "wrote_funding_round": _FakeRel("wrote_funding_round", "funding_round"),
        },
    )
    funding_round = _FakeConcept(
        name="funding_round",
        properties={"funding_round_id": _FakeProp("funding_round_id")},
        relationships={
            "funding_of": _FakeRel("funding_of", "company"),
            # Off-path
            "invested_from": _FakeRel("invested_from", "organization"),
            "invested_by": _FakeRel("invested_by", "person"),
        },
    )
    company = _FakeConcept(
        name="company",
        properties={"organization_name": _FakeProp("organization_name")},
        relationships={
            "acquired_by": _FakeRel(
                "acquired_by", "company", transitivity=4,
            ),
            # Off-path
            "reached_milestone": _FakeRel("reached_milestone", "milestone"),
            "basis_for_ipo": _FakeRel("basis_for_ipo", "ipo"),
        },
    )
    person = _FakeConcept(
        name="person",
        properties={"name": _FakeProp("name")},
        relationships={"works_at2": _FakeRel("works_at2", "organization")},
    )
    office = _FakeConcept(name="office", properties={"city": _FakeProp("city")})
    milestone = _FakeConcept(name="milestone", properties={"name": _FakeProp("name")})
    ipo = _FakeConcept(name="ipo", properties={"date": _FakeProp("date")})
    fund = _FakeConcept(name="fund", properties={"name": _FakeProp("name")})
    degree = _FakeConcept(name="degree", properties={"name": _FakeProp("name")})
    profile = _FakeConcept(name="profile", properties={"name": _FakeProp("name")})
    financial_organization = _FakeConcept(
        name="financial_organization", properties={"name": _FakeProp("name")},
    )
    product = _FakeConcept(name="product", properties={"name": _FakeProp("name")})
    return _FakeOntology({
        "organization": organization, "funding_round": funding_round,
        "company": company, "person": person, "office": office,
        "milestone": milestone, "ipo": ipo, "fund": fund, "degree": degree,
        "profile": profile, "financial_organization": financial_organization,
        "product": product,
    })


class TestAnchorRootedPrefixBug:
    """Test 1, Test 2, Test 3 + cross-test equivalence (Test 1 == Test 3).

    Every emitted property prefix MUST start at the anchor. Full-chain and
    single-hop path styles must produce identical output."""

    def test_test1_full_chain_single_multi_segment_path(self):
        """Test 1: anchor=customer, one 3-segment path. Pre-existing
        behavior — must keep passing."""
        ontology = _supply_chain_universe()
        paths = [
            SelectedPath(path_id="P1", segments=[
                _seg("customer", "made_order", "order"),
                _seg("order", "includes_product", "product"),
                _seg("product", "contains", "material"),
            ]),
        ]
        result = build_relationships_from_paths(
            paths, ontology, anchor="customer",
        )

        # Per-hop keys: one bucket per segment, keyed by accumulated prefix.
        assert set(result.keys()) == {
            "made_order[order]",
            "made_order[order].includes_product[product]",
            "made_order[order].includes_product[product].contains[material]",
        }
        # Pool every bucket's columns for the presence + anti-leak checks.
        names = [
            c["name"]
            for bucket in result.values()
            for c in bucket["columns"]
        ]

        # MUST EXIST.
        assert "made_order[order].order_id" in names
        assert (
            "made_order[order].includes_product[product].product_name"
            in names
        )
        assert (
            "made_order[order].includes_product[product]"
            ".contains[material].material_name" in names
        )

        # MUST NOT EXIST: prefixes not rooted at the anchor.
        for n in names:
            assert not n.startswith("includes_product[product]"), n
            assert not n.startswith("contains[material]"), n

        # MUST NOT EXIST: off-path rels.
        for forbidden in [
            "has_employee", "received_shipment",
            "of_inventory", "has_bill_of_material",
        ]:
            for n in names:
                assert forbidden not in n, (
                    f"off-path rel {forbidden!r} leaked into {n!r}"
                )

        # Maximum nesting depth ≤ 3.
        for n in names:
            assert n.count("[") <= 3, f"over-depth chain: {n}"

    def test_test2_sensobi_recursive_segment_full_chain(self):
        """Test 2: anchor=organization, 3-segment path with a recursive
        final segment (acquired_by, transitivity=4 in ontology). The
        recursive segment must carry the ``*4`` marker and the full
        anchor-rooted prefix."""
        ontology = _sensobi_universe()
        paths = [
            SelectedPath(path_id="P1", segments=[
                _seg("organization", "made_investment", "funding_round"),
                _seg("funding_round", "funding_of", "company"),
                _seg("company", "acquired_by", "company"),
            ], is_recursive=True),
        ]
        result = build_relationships_from_paths(
            paths, ontology, anchor="organization",
        )

        # Per-hop keys: one bucket per segment (with the *4 transitivity
        # baked into the last bucket's key).
        assert set(result.keys()) == {
            "made_investment[funding_round]",
            "made_investment[funding_round].funding_of[company]",
            (
                "made_investment[funding_round].funding_of[company]"
                ".acquired_by[company*4]"
            ),
        }
        names = [
            c["name"]
            for bucket in result.values()
            for c in bucket["columns"]
        ]

        # MUST EXIST.
        assert (
            "made_investment[funding_round].funding_round_id" in names
        )
        assert (
            "made_investment[funding_round].funding_of[company]"
            ".organization_name" in names
        )
        assert (
            "made_investment[funding_round].funding_of[company]"
            ".acquired_by[company*4].organization_name" in names
        ), (
            f"recursive segment prefix missing or wrong; got {names}"
        )

        # MUST NOT EXIST: prefixes not rooted at the anchor.
        for n in names:
            assert not n.startswith("funding_of[company]"), n
            assert not n.startswith("acquired_by[company"), n

        # MUST NOT EXIST: off-path rels.
        for forbidden in [
            "has_employee", "invested_from", "invested_by", "works_at2",
            "reached_milestone", "basis_for_ipo", "has_office",
            "wrote_funding_round",
        ]:
            for n in names:
                assert forbidden not in n, (
                    f"off-path rel {forbidden!r} leaked into {n!r}"
                )

        # MUST NOT EXIST: off-path concepts. Match the concept name only
        # when followed by `]` or `*` so e.g. `fund` doesn't match
        # `funding_round`.
        import re as _re
        for forbidden_concept in [
            "person", "office", "milestone", "ipo", "fund", "degree",
            "profile", "financial_organization", "product",
        ]:
            pat = _re.compile(rf"\[{_re.escape(forbidden_concept)}[\]\*]")
            for n in names:
                assert not pat.search(n), (
                    f"off-path concept {forbidden_concept!r} in {n!r}"
                )

        # Maximum nesting depth ≤ 3 (the *4 marker is on transitivity, not
        # an extra segment).
        for n in names:
            assert n.count("[") <= 3, f"over-depth chain: {n}"

    def test_test3_single_hop_style_chained_one_segment_paths(self):
        """Test 3: anchor=customer, three 1-segment paths chained
        end-to-start. Output MUST be identical to Test 1 — the normalization
        pre-pass rewires each path to start at the anchor by prepending the
        previous path's segments."""
        ontology = _supply_chain_universe()
        paths = [
            SelectedPath(path_id="P1", segments=[
                _seg("customer", "made_order", "order"),
            ]),
            SelectedPath(path_id="P2", segments=[
                _seg("order", "includes_product", "product"),
            ]),
            SelectedPath(path_id="P3", segments=[
                _seg("product", "contains", "material"),
            ]),
        ]
        result = build_relationships_from_paths(
            paths, ontology, anchor="customer",
        )

        # Per-hop keys match Test 1's keys (single-hop / full-chain styles
        # are semantically equivalent after normalization).
        assert set(result.keys()) == {
            "made_order[order]",
            "made_order[order].includes_product[product]",
            "made_order[order].includes_product[product].contains[material]",
        }
        names = [
            c["name"]
            for bucket in result.values()
            for c in bucket["columns"]
        ]

        # MUST EXIST.
        assert "made_order[order].order_id" in names
        assert (
            "made_order[order].includes_product[product].product_name"
            in names
        )
        assert (
            "made_order[order].includes_product[product]"
            ".contains[material].material_name" in names
        )

        # MUST NOT EXIST: standalone prefixes that don't start at the anchor.
        for n in names:
            assert not n.startswith("includes_product[product]"), n
            assert not n.startswith("contains[material]"), n

        # MUST NOT EXIST: off-path rels.
        for forbidden in [
            "has_employee", "received_shipment",
            "of_inventory", "has_bill_of_material",
        ]:
            for n in names:
                assert forbidden not in n

        # Maximum nesting depth ≤ 3.
        for n in names:
            assert n.count("[") <= 3, f"over-depth chain: {n}"

    def test_test1_and_test3_emit_identical_column_name_sets(self):
        """Cross-test equivalence: the SET of column names from Test 1
        (full-chain) and Test 3 (single-hop) must be identical. Both
        express the same logical traversal — the prompt accepts both
        styles, the constructor must honor that."""
        ontology = _supply_chain_universe()
        full_chain = [
            SelectedPath(path_id="P1", segments=[
                _seg("customer", "made_order", "order"),
                _seg("order", "includes_product", "product"),
                _seg("product", "contains", "material"),
            ]),
        ]
        single_hop = [
            SelectedPath(path_id="P1", segments=[
                _seg("customer", "made_order", "order"),
            ]),
            SelectedPath(path_id="P2", segments=[
                _seg("order", "includes_product", "product"),
            ]),
            SelectedPath(path_id="P3", segments=[
                _seg("product", "contains", "material"),
            ]),
        ]
        r1 = build_relationships_from_paths(
            full_chain, ontology, anchor="customer",
        )
        r3 = build_relationships_from_paths(
            single_hop, ontology, anchor="customer",
        )
        names1 = {
            c["name"]
            for bucket in r1.values()
            for c in bucket["columns"]
        }
        names3 = {
            c["name"]
            for bucket in r3.values()
            for c in bucket["columns"]
        }
        assert names1 == names3, (
            f"full-chain and single-hop produced different outputs:\n"
            f"  only in full-chain: {names1 - names3}\n"
            f"  only in single-hop: {names3 - names1}"
        )

    def test_unanchored_path_with_no_predecessor_is_dropped(self):
        """A path that doesn't start at the anchor AND can't be connected
        to any earlier path's endpoint is dropped (logged), not silently
        accepted as if it were anchor-rooted."""
        ontology = _supply_chain_universe()
        paths = [
            # P1 has nothing to do with customer's reachable graph and is
            # not anchor-rooted — drop.
            SelectedPath(path_id="orphan", segments=[
                _seg("vendor", "ships_to", "warehouse"),
            ]),
        ]
        result = build_relationships_from_paths(
            paths, ontology, anchor="customer",
        )
        assert result == {}


# ---- Measure-emission bug: cross-concept leakage + duplicates ------------


def _supply_chain_with_realistic_measures():
    """Same customer → order → product → material shape, but each concept's
    ``measures`` dict mimics the real parser's behavior: it contains BOTH
    direct measures (``scoped_to_relationship=None``) AND rel-scoped entries
    (``scoped_to_relationship=<rel>``) reachable via 1-hop describe.

    The constructor must emit ONLY direct measures — rel-scoped entries
    belong to other concepts reached via further traversal, not to the
    endpoint we're standing on. Iterating them is the bug."""
    customer = _FakeConcept(
        name="customer",
        properties={"customer_id": _FakeProp("customer_id")},
        measures={
            "count_of_customer": _FakeMeasure("count_of_customer"),
        },
        relationships={"made_order": _FakeRel("made_order", "order")},
    )
    order = _FakeConcept(
        name="order",
        properties={
            "order_id": _FakeProp("order_id"),
            "order_date": _FakeProp("order_date"),
        },
        measures={
            # Direct on order — must be emitted.
            "count_of_order": _FakeMeasure("count_of_order"),
            # Rel-scoped from order's relationships — must NOT be emitted
            # under made_order[order]. These would leak as cross-concept
            # measures (customer is upstream of order, shipment is reached
            # via received_shipment, material is downstream through product).
            "of_customer.count_of_customer": _FakeMeasure(
                "count_of_customer",
                scoped_to_relationship="of_customer",
            ),
            "of_customer.count_of_customer_2": _FakeMeasure(
                "count_of_customer",  # duplicate name — triggers 3× repeat
                scoped_to_relationship="of_customer",
            ),
            "of_customer.count_of_customer_3": _FakeMeasure(
                "count_of_customer",
                scoped_to_relationship="of_customer",
            ),
            "received_shipment.count_of_shipment": _FakeMeasure(
                "count_of_shipment",
                scoped_to_relationship="received_shipment",
            ),
            "received_shipment.late_shipment_ratio": _FakeMeasure(
                "late_shipment_ratio",
                scoped_to_relationship="received_shipment",
            ),
            "received_shipment.count_of_late_shipment": _FakeMeasure(
                "count_of_late_shipment",
                scoped_to_relationship="received_shipment",
            ),
            # Material's measure leaking up through includes_product →
            # contains 2-hop describe (still parsed at graph_depth=1 from
            # order via some alias):
            "includes_product.total_price_per_1_kg": _FakeMeasure(
                "total_price_per_1_kg",
                scoped_to_relationship="includes_product",
            ),
            # Duplicate count_of_order via a self-reflective rel alias:
            "some_alias.count_of_order": _FakeMeasure(
                "count_of_order",
                scoped_to_relationship="some_alias",
            ),
        },
        relationships={
            "includes_product": _FakeRel("includes_product", "product"),
            "received_shipment": _FakeRel("received_shipment", "shipment"),
        },
    )
    product = _FakeConcept(
        name="product",
        properties={"product_name": _FakeProp("product_name")},
        measures={
            # Direct on product — must be emitted ONCE under the product prefix.
            "count_of_product": _FakeMeasure("count_of_product"),
            # Three rel-scoped duplicates → without the fix, count_of_product
            # appears 4× under the product prefix.
            "rel_a.count_of_product": _FakeMeasure(
                "count_of_product", scoped_to_relationship="rel_a",
            ),
            "rel_b.count_of_product": _FakeMeasure(
                "count_of_product", scoped_to_relationship="rel_b",
            ),
            "rel_c.count_of_product": _FakeMeasure(
                "count_of_product", scoped_to_relationship="rel_c",
            ),
            # Material's measure reachable from product via contains —
            # rel-scoped, must NOT leak under the product prefix.
            "contains.total_price_per_1_kg": _FakeMeasure(
                "total_price_per_1_kg",
                scoped_to_relationship="contains",
            ),
        },
        relationships={"contains": _FakeRel("contains", "material")},
    )
    material = _FakeConcept(
        name="material",
        properties={"material_name": _FakeProp("material_name")},
        measures={
            # Direct on material — must be emitted under material prefix.
            "total_price_per_1_kg": _FakeMeasure("total_price_per_1_kg"),
            # Product's measure leaks here via the inverse of contains —
            # rel-scoped, must NOT appear under material's prefix.
            "~contains.count_of_product": _FakeMeasure(
                "count_of_product",
                scoped_to_relationship="~contains",
            ),
        },
    )
    # Off-path concepts that exist but should never be reached.
    shipment = _FakeConcept(
        name="shipment", properties={"x": _FakeProp("x")},
        measures={
            "count_of_shipment": _FakeMeasure("count_of_shipment"),
            "late_shipment_ratio": _FakeMeasure("late_shipment_ratio"),
            "count_of_late_shipment": _FakeMeasure("count_of_late_shipment"),
        },
    )
    inventory = _FakeConcept(name="inventory", properties={"y": _FakeProp("y")})
    return _FakeOntology({
        "customer": customer, "order": order,
        "product": product, "material": material,
        "shipment": shipment, "inventory": inventory,
    })


class TestMeasureEmissionCorrectness:
    """Test A and Test B for the measure-emission bug.

    Measures defined on concepts NOT in the path (reached only via further
    traversal) MUST NOT leak under path prefixes, and no (prefix, measure)
    pair may repeat in the output."""

    def _full_chain_result(self):
        ontology = _supply_chain_with_realistic_measures()
        paths = [
            SelectedPath(path_id="P1", segments=[
                _seg("customer", "made_order", "order"),
                _seg("order", "includes_product", "product"),
                _seg("product", "contains", "material"),
            ]),
        ]
        return build_relationships_from_paths(
            paths, ontology, anchor="customer",
        )

    @staticmethod
    def _measure_names_by_prefix(result):
        """Return ``dict[prefix_str -> list[measure_name]]`` from the rebuild
        output. Each emitted measure name has shape
        ``measure.<prefix>.<measure_name>`` — split on the final ``.`` to
        recover the (prefix, name) pair."""
        out: dict[str, list[str]] = {}
        for bucket in result.values():
            for m in bucket.get("measures", []):
                full = m["name"]
                assert full.startswith("measure."), (
                    f"unexpected measure name shape: {full!r}"
                )
                stripped = full[len("measure."):]
                prefix, _, m_name = stripped.rpartition(".")
                out.setdefault(prefix, []).append(m_name)
        return out

    def test_A_no_cross_concept_measure_leakage(self):
        """Test A: measures defined on off-path concepts (or on concepts
        reached only via further traversal) MUST NOT appear under path
        prefixes."""
        by_prefix = self._measure_names_by_prefix(self._full_chain_result())

        order_pref = "made_order[order]"
        product_pref = "made_order[order].includes_product[product]"
        material_pref = (
            "made_order[order].includes_product[product].contains[material]"
        )

        # Under made_order[order]: no shipment/material/customer measures.
        order_measures = set(by_prefix.get(order_pref, []))
        forbidden_at_order = {
            "count_of_shipment", "late_shipment_ratio", "count_of_late_shipment",
            "total_price_per_1_kg",  # material's
            "count_of_customer",     # customer's (anchor)
        }
        leaked = forbidden_at_order & order_measures
        assert not leaked, (
            f"cross-concept measure leakage under {order_pref!r}: {sorted(leaked)} "
            f"(all emitted: {sorted(order_measures)})"
        )

        # Under includes_product[product]: no shipment/material/customer/order.
        product_measures = set(by_prefix.get(product_pref, []))
        forbidden_at_product = {
            "count_of_shipment", "late_shipment_ratio", "count_of_late_shipment",
            "total_price_per_1_kg",  # material's — reached via contains
            "count_of_customer",
            "count_of_order",
        }
        leaked = forbidden_at_product & product_measures
        assert not leaked, (
            f"cross-concept measure leakage under {product_pref!r}: {sorted(leaked)} "
            f"(all emitted: {sorted(product_measures)})"
        )

        # Under contains[material]: no product measures.
        material_measures = set(by_prefix.get(material_pref, []))
        forbidden_at_material = {"count_of_product"}
        leaked = forbidden_at_material & material_measures
        assert not leaked, (
            f"cross-concept measure leakage under {material_pref!r}: {sorted(leaked)} "
            f"(all emitted: {sorted(material_measures)})"
        )

    def test_B_no_measure_duplication(self):
        """Test B: every (prefix, measure_name) pair appears exactly once
        in the output. Specifically the named cases the user reported
        (count_of_product 4×, count_of_customer 3×, count_of_order 2×)
        must all collapse to ≤ 1."""
        by_prefix = self._measure_names_by_prefix(self._full_chain_result())

        product_pref = "made_order[order].includes_product[product]"
        order_pref = "made_order[order]"

        # Named cases from the bug report.
        assert by_prefix.get(product_pref, []).count("count_of_product") <= 1, (
            f"count_of_product appears {by_prefix[product_pref].count('count_of_product')}× "
            f"under {product_pref!r}; all measures: {by_prefix.get(product_pref)}"
        )
        assert by_prefix.get(order_pref, []).count("count_of_customer") <= 1, (
            f"count_of_customer repeats under {order_pref!r}: "
            f"{by_prefix.get(order_pref)}"
        )
        assert by_prefix.get(order_pref, []).count("count_of_order") <= 1
        assert by_prefix.get(order_pref, []).count("count_of_product") <= 1

        # Global: no (prefix, measure_name) pair repeats anywhere.
        for prefix, names in by_prefix.items():
            from collections import Counter
            counts = Counter(names)
            dups = {n: c for n, c in counts.items() if c > 1}
            assert not dups, (
                f"duplicate measures under prefix {prefix!r}: {dups}"
            )


class TestApplyTransitivityOverrides:
    """The regex rewriter that bakes LLM-chosen depth into context strings."""

    def test_rewrites_simple_transitive_column(self):
        text = "`has_acquired[company*2].name`"
        ov = [TransitivityOverride(rel="has_acquired", target="company", level=3)]
        out = apply_transitivity_overrides(text, ov)
        assert out == "`has_acquired[company*3].name`"

    def test_rewrites_all_occurrences(self):
        text = (
            "Columns: `has_acquired[company*2].name`, "
            "`has_acquired[company*2].id`, `has_acquired[company*2].founded`"
        )
        ov = [TransitivityOverride(rel="has_acquired", target="company", level=5)]
        out = apply_transitivity_overrides(text, ov)
        assert "*5" in out
        assert "*2" not in out

    def test_tilde_prefixed_rel_is_literal_name(self):
        text = "~has_acquired[company*2].name"
        ov = [TransitivityOverride(rel="has_acquired", target="company", level=3)]
        out = apply_transitivity_overrides(text, ov)
        assert out == text
        ov_literal = [TransitivityOverride(rel="~has_acquired", target="company", level=3)]
        assert apply_transitivity_overrides(text, ov_literal) == "~has_acquired[company*3].name"

    def test_only_matching_rel_target_rewritten(self):
        text = (
            "`has_acquired[company*2].name` and `owns_office[location*3].city`"
        )
        ov = [TransitivityOverride(rel="has_acquired", target="company", level=5)]
        out = apply_transitivity_overrides(text, ov)
        assert "has_acquired[company*5]" in out
        assert "owns_office[location*3]" in out

    def test_handles_nested_chain(self):
        text = "`acquired_by[org*2].own_company[company*2].name`"
        ov = [TransitivityOverride(rel="own_company", target="company", level=4)]
        out = apply_transitivity_overrides(text, ov)
        assert out == "`acquired_by[org*2].own_company[company*4].name`"

    def test_no_match_is_noop(self):
        text = "made_order[order].sales"
        ov = [TransitivityOverride(rel="made_order", target="order", level=3)]
        out = apply_transitivity_overrides(text, ov)
        assert out == text

    def test_empty_overrides_returns_text_unchanged(self):
        text = "has_acquired[company*2].name"
        assert apply_transitivity_overrides(text, []) == text
        assert apply_transitivity_overrides(text, None) == text

    def test_empty_text_returns_empty(self):
        ov = [TransitivityOverride(rel="x", target="y", level=2)]
        assert apply_transitivity_overrides("", ov) == ""
        assert apply_transitivity_overrides(None, ov) == ""

    def test_multiple_overrides_applied_independently(self):
        text = (
            "`has_acquired[company*2].x`, `has_parent[work_item*1].y`"
        )
        ov = [
            TransitivityOverride(rel="has_acquired", target="company", level=3),
            TransitivityOverride(rel="has_parent", target="work_item", level=5),
        ]
        out = apply_transitivity_overrides(text, ov)
        assert "has_acquired[company*3]" in out
        assert "has_parent[work_item*5]" in out

    def test_special_regex_chars_in_rel_name_escaped(self):
        text = "`foo.bar[baz*2].x`"
        ov = [TransitivityOverride(rel="foo.bar", target="baz", level=5)]
        out = apply_transitivity_overrides(text, ov)
        assert out == "`foo.bar[baz*5].x`"


# ---------------------------------------------------------------------------
# Cardinality injection into the rel description (dynamic-rebuild branch)
# ---------------------------------------------------------------------------


from langchain_timbr.ontology_context.context_builder.rebuild import (
    compose_rel_description_with_cardinality,
)


class _FakeOntologyWithCardinality(_FakeOntology):
    """Extends _FakeOntology with a cardinality_of mapping keyed by
    ``(from_concept, rel_name)``. Used by the cardinality tests below."""

    def __init__(self, concepts, cardinalities):
        super().__init__(concepts)
        self._cardinalities = cardinalities

    def cardinality_of(self, concept: str, rel_name: str) -> str:
        key = (concept, rel_name)
        if key not in self._cardinalities:
            raise KeyError(key)
        return self._cardinalities[key]


class TestComposeRelDescriptionWithCardinality:
    """Unit-level coverage for the format helper."""

    def test_raw_and_card_present_appends_in_parens(self):
        assert (
            compose_rel_description_with_cardinality("links A to B", "N:M")
            == "links A to B (cardinality: N:M)"
        )

    def test_no_raw_card_becomes_description(self):
        assert (
            compose_rel_description_with_cardinality("", "N:1")
            == "cardinality: N:1"
        )
        assert (
            compose_rel_description_with_cardinality(None, "1:N")
            == "cardinality: 1:N"
        )

    def test_no_card_returns_raw_unchanged(self):
        assert (
            compose_rel_description_with_cardinality("links A to B", None)
            == "links A to B"
        )
        assert (
            compose_rel_description_with_cardinality("links A to B", "")
            == "links A to B"
        )

    def test_both_empty_returns_empty_string(self):
        # Defensive — no stray "cardinality: " ever introduced.
        assert compose_rel_description_with_cardinality(None, None) == ""
        assert compose_rel_description_with_cardinality("", "") == ""
        assert compose_rel_description_with_cardinality("   ", "  ") == ""


class TestBuildRelationshipsFromPathsCardinality:
    """End-to-end: cardinality flows into the bucket description via
    build_relationships_from_paths so the dynamic path surfaces it the
    same way the static path does."""

    def _company_employees_universe(self, cardinalities):
        company = _FakeConcept(
            name="company",
            properties={"company_id": _FakeProp("company_id")},
            relationships={
                "has_employee": _FakeRel(
                    "has_employee", "person", description="company employs",
                ),
                "has_product": _FakeRel(
                    "has_product", "product",
                    # No description on purpose — exercises the "card-only"
                    # branch of the composer.
                    description=None,
                ),
            },
        )
        person = _FakeConcept(
            name="person",
            properties={"first_name": _FakeProp("first_name")},
        )
        product = _FakeConcept(
            name="product",
            properties={"product_name": _FakeProp("product_name")},
        )
        return _FakeOntologyWithCardinality(
            {"company": company, "person": person, "product": product},
            cardinalities=cardinalities,
        )

    def test_description_appended_with_cardinality_when_both_present(self):
        ontology = self._company_employees_universe({
            ("company", "has_employee"): "N:M",
        })
        path = SelectedPath(path_id="P1", segments=[
            _seg("company", "has_employee", "person"),
        ])
        result = build_relationships_from_paths(
            [path], ontology, anchor="company",
        )
        assert (
            result["has_employee[person]"]["description"]
            == "company employs (cardinality: N:M)"
        )

    def test_cardinality_becomes_description_when_raw_is_none(self):
        ontology = self._company_employees_universe({
            ("company", "has_product"): "1:N",
        })
        path = SelectedPath(path_id="P1", segments=[
            _seg("company", "has_product", "product"),
        ])
        result = build_relationships_from_paths(
            [path], ontology, anchor="company",
        )
        assert (
            result["has_product[product]"]["description"]
            == "cardinality: 1:N"
        )

    def test_missing_cardinality_does_not_break_rebuild(self):
        """When ``cardinality_of`` raises KeyError for the rel, the safe
        wrapper returns None and the bucket description falls back to the
        raw description unchanged."""
        ontology = self._company_employees_universe(cardinalities={})
        path = SelectedPath(path_id="P1", segments=[
            _seg("company", "has_employee", "person"),
        ])
        result = build_relationships_from_paths(
            [path], ontology, anchor="company",
        )
        # Cardinality lookup failed → description is the raw description
        # only (no stray "cardinality: " marker).
        assert (
            result["has_employee[person]"]["description"]
            == "company employs"
        )


# ---------------------------------------------------------------------------
# Multi-hop chains — one bucket per prefix, each with its OWN description +
# cardinality + only that hop's direct columns / measures.
# ---------------------------------------------------------------------------


class TestBuildRelationshipsFromPathsMultiHop:
    """Per-hop blocks in the rendered SQL-gen prompt. Locks the user's
    requested shape: ``of_customer[customer]`` then
    ``of_customer[customer].received_shipment[shipment]`` etc., each
    block carrying ITS hop's description + cardinality.
    """

    def _abcd_universe(
        self, *, with_measures: bool = False, cardinalities: dict | None = None,
    ):
        """Linear chain A → relAB → B → relBC → C → relCD → D."""
        def _concept(name: str, *, with_measure: bool) -> _FakeConcept:
            props = {f"{name}_id": _FakeProp(f"{name}_id")}
            measures = (
                {f"count_of_{name}": _FakeMeasure(f"count_of_{name}")}
                if with_measure else {}
            )
            return _FakeConcept(name=name, properties=props, measures=measures)

        # A is the anchor — its direct props are NOT emitted by the rebuild
        # (handled by ``filter_columns_for_concepts``); only its rels matter.
        a = _FakeConcept(
            name="A",
            properties={"A_id": _FakeProp("A_id")},
            relationships={"relAB": _FakeRel("relAB", "B", description="A→B")},
        )
        b = _concept("B", with_measure=with_measures)
        b.relationships["relBC"] = _FakeRel("relBC", "C", description="B→C")
        c = _concept("C", with_measure=with_measures)
        c.relationships["relCD"] = _FakeRel("relCD", "D", description="C→D")
        d = _concept("D", with_measure=with_measures)

        return _FakeOntologyWithCardinality(
            {"A": a, "B": b, "C": c, "D": d},
            cardinalities=cardinalities or {},
        )

    def test_three_hop_chain_produces_three_independent_buckets(self):
        """A→B→C→D path: 3 segments → 3 dict entries keyed by accumulated
        prefix. Each bucket's description carries ITS OWN cardinality
        marker (not the top hop's), and each bucket's columns contain
        only that hop's target concept's direct props."""
        ontology = self._abcd_universe(cardinalities={
            ("A", "relAB"): "N:M",
            ("B", "relBC"): "1:N",
            ("C", "relCD"): "N:1",
        })
        path = SelectedPath(path_id="P1", segments=[
            _seg("A", "relAB", "B"),
            _seg("B", "relBC", "C"),
            _seg("C", "relCD", "D"),
        ])
        result = build_relationships_from_paths(
            [path], ontology, anchor="A",
        )

        # Per-hop keys in walk order.
        assert list(result.keys()) == [
            "relAB[B]",
            "relAB[B].relBC[C]",
            "relAB[B].relBC[C].relCD[D]",
        ]

        # Each bucket's description ends with its OWN cardinality, NOT the
        # top hop's.
        assert (
            result["relAB[B]"]["description"]
            == "A→B (cardinality: N:M)"
        )
        assert (
            result["relAB[B].relBC[C]"]["description"]
            == "B→C (cardinality: 1:N)"
        )
        assert (
            result["relAB[B].relBC[C].relCD[D]"]["description"]
            == "C→D (cardinality: N:1)"
        )

        # Each bucket's columns are ONLY the target concept's direct props.
        b_cols = [c["col_name"] for c in result["relAB[B]"]["columns"]]
        c_cols = [c["col_name"] for c in result["relAB[B].relBC[C]"]["columns"]]
        d_cols = [c["col_name"] for c in result["relAB[B].relBC[C].relCD[D]"]["columns"]]
        assert b_cols == ["B_id"]
        assert c_cols == ["C_id"]
        assert d_cols == ["D_id"]

        # No cross-bleed: B's column name starts with "relAB[B]." (not
        # "relAB[B].relBC[C]."), C's starts with "relAB[B].relBC[C].",
        # D's starts with the full 3-hop prefix.
        assert result["relAB[B]"]["columns"][0]["name"].startswith("relAB[B].")
        assert not result["relAB[B]"]["columns"][0]["name"].startswith("relAB[B].relBC")
        assert result["relAB[B].relBC[C]"]["columns"][0]["name"].startswith(
            "relAB[B].relBC[C]."
        )
        assert result["relAB[B].relBC[C].relCD[D]"]["columns"][0]["name"].startswith(
            "relAB[B].relBC[C].relCD[D]."
        )

        # Render and confirm 3 separate "- The following columns are part of"
        # lines, one per hop.
        from langchain_timbr.utils.timbr_llm_utils import _build_rel_columns_str
        rendered = _build_rel_columns_str(result)
        col_blocks = [
            line for line in rendered.split('\n')
            if "The following columns are part of" in line
        ]
        assert len(col_blocks) == 3
        assert "relAB[B] relationship" in col_blocks[0]
        assert "relAB[B].relBC[C] relationship" in col_blocks[1]
        assert "relAB[B].relBC[C].relCD[D] relationship" in col_blocks[2]
        # Each block's description includes its OWN cardinality token.
        assert "cardinality: N:M" in col_blocks[0]
        assert "cardinality: 1:N" in col_blocks[1]
        assert "cardinality: N:1" in col_blocks[2]

    def test_three_hop_chain_splits_measures_per_prefix(self):
        """Same A→B→C→D path; each target concept has a NATIVE measure.
        Measures must split into their owning hop's bucket — no measure
        leaks under the wrong prefix."""
        ontology = self._abcd_universe(with_measures=True, cardinalities={
            ("A", "relAB"): "N:M",
            ("B", "relBC"): "1:N",
            ("C", "relCD"): "N:1",
        })
        path = SelectedPath(path_id="P1", segments=[
            _seg("A", "relAB", "B"),
            _seg("B", "relBC", "C"),
            _seg("C", "relCD", "D"),
        ])
        result = build_relationships_from_paths(
            [path], ontology, anchor="A",
        )

        # Each bucket's measures contain ONLY their hop's native measure.
        b_meas = [m["col_name"] for m in result["relAB[B]"]["measures"]]
        c_meas = [m["col_name"] for m in result["relAB[B].relBC[C]"]["measures"]]
        d_meas = [m["col_name"] for m in result["relAB[B].relBC[C].relCD[D]"]["measures"]]
        assert b_meas == ["count_of_B"]
        assert c_meas == ["count_of_C"]
        assert d_meas == ["count_of_D"]

        # The measure NAME field carries the full per-hop prefix.
        assert result["relAB[B]"]["measures"][0]["name"] == "measure.relAB[B].count_of_B"
        assert (
            result["relAB[B].relBC[C]"]["measures"][0]["name"]
            == "measure.relAB[B].relBC[C].count_of_C"
        )
        assert (
            result["relAB[B].relBC[C].relCD[D]"]["measures"][0]["name"]
            == "measure.relAB[B].relBC[C].relCD[D].count_of_D"
        )

        # Render and confirm THREE separate "calculated measures" lines —
        # one per hop, each carrying its bucket's prefix in the header.
        from langchain_timbr.utils.timbr_llm_utils import _build_rel_columns_str
        rendered = _build_rel_columns_str(result)
        measure_blocks = [
            line for line in rendered.split('\n')
            if "calculated measures" in line
        ]
        assert len(measure_blocks) == 3
        assert "relAB[B] relationship" in measure_blocks[0]
        assert "relAB[B].relBC[C] relationship" in measure_blocks[1]
        assert "relAB[B].relBC[C].relCD[D] relationship" in measure_blocks[2]
        # No measure leaks: hop-1's measure block must NOT mention C or D.
        assert "count_of_C" not in measure_blocks[0]
        assert "count_of_D" not in measure_blocks[0]
        # Hop-2's measure block must NOT mention D.
        assert "count_of_D" not in measure_blocks[1]

    def test_four_hop_mocked_chain_renders_four_blocks(self):
        """Structure scales with depth: A→B→C→D→E (4 segments) produces 4
        independent dict entries + 4 column blocks + 4 measure blocks in
        the rendered prompt."""
        # Extend the ABCD universe with E (and a relDE on D).
        ontology = self._abcd_universe(with_measures=True, cardinalities={
            ("A", "relAB"): "N:M",
            ("B", "relBC"): "1:N",
            ("C", "relCD"): "N:1",
            ("D", "relDE"): "1:1",
        })
        # Patch in E + the relDE edge on D.
        e = _FakeConcept(
            name="E",
            properties={"E_id": _FakeProp("E_id")},
            measures={"count_of_E": _FakeMeasure("count_of_E")},
        )
        ontology._concepts["E"] = e
        ontology._concepts["D"].relationships["relDE"] = _FakeRel(
            "relDE", "E", description="D→E",
        )

        path = SelectedPath(path_id="P1", segments=[
            _seg("A", "relAB", "B"),
            _seg("B", "relBC", "C"),
            _seg("C", "relCD", "D"),
            _seg("D", "relDE", "E"),
        ])
        result = build_relationships_from_paths(
            [path], ontology, anchor="A",
        )

        # 4 dict entries in walk order.
        assert list(result.keys()) == [
            "relAB[B]",
            "relAB[B].relBC[C]",
            "relAB[B].relBC[C].relCD[D]",
            "relAB[B].relBC[C].relCD[D].relDE[E]",
        ]

        # Each bucket's description has its OWN cardinality.
        descs = {k: v["description"] for k, v in result.items()}
        assert descs["relAB[B]"].endswith("(cardinality: N:M)")
        assert descs["relAB[B].relBC[C]"].endswith("(cardinality: 1:N)")
        assert descs["relAB[B].relBC[C].relCD[D]"].endswith("(cardinality: N:1)")
        assert descs["relAB[B].relBC[C].relCD[D].relDE[E]"].endswith(
            "(cardinality: 1:1)"
        )

        from langchain_timbr.utils.timbr_llm_utils import _build_rel_columns_str
        rendered = _build_rel_columns_str(result)
        col_blocks = [
            line for line in rendered.split('\n')
            if "The following columns are part of" in line
        ]
        measure_blocks = [
            line for line in rendered.split('\n')
            if "calculated measures" in line
        ]
        assert len(col_blocks) == 4
        assert len(measure_blocks) == 4
        # The 4th block's header carries the full 4-hop prefix.
        assert (
            "relAB[B].relBC[C].relCD[D].relDE[E] relationship"
            in col_blocks[3]
        )
        # The 4th block's column name field carries the full 4-hop prefix
        # too.
        assert (
            "relAB[B].relBC[C].relCD[D].relDE[E].E_id"
            in col_blocks[3]
        )
