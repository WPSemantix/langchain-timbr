"""Tests for subgraph.py — edge-count heuristic, BFS, DDL serialization."""

from __future__ import annotations

from langchain_timbr.ontology_context.context_builder.edge_index import EdgeIndex
from langchain_timbr.ontology_context.ontology.graph import Ontology
from langchain_timbr.ontology_context.context_builder.metadata_config import MetadataContextConfig
from langchain_timbr.ontology_context.context_builder.subgraph import (
    estimate_subgraph_edge_count,
    retrieve_subgraph,
    serialize_compact_ddl,
    should_skip_static_build,
)


# ---- FakeClient for synthetic ontologies -----------------------------------


def _row(col_name, **extras):
    base = {"col_name": col_name, "data_type": "varchar", "comment": "", "inheritance_marker": "", "pk_marker": ""}
    base.update(extras)
    return base


class FakeClient:
    """Minimal fake client. Concepts dict maps name -> describe rows.

    ``inheritance_rows`` (optional) follows the shape produced by
    ``TimbrOntologyClient.fetch_inheritance_meta``: a list of
    ``{"concept": <name>, "inheritance": "<csv parents>"}`` dicts.
    """

    def __init__(self, concepts, relationships=None, version="v1", inheritance_rows=None):
        self._concepts = concepts
        self._relationships = relationships or []
        self._inheritance_rows = inheritance_rows or []
        self._version = version

    def fetch_version_id(self):
        return self._version

    def describe_concept(self, name):
        return list(self._concepts.get(name, []))

    def fetch_relationships_meta(self):
        return list(self._relationships)

    def fetch_inheritance_meta(self):
        return list(self._inheritance_rows)


# ---- estimate_subgraph_edge_count + should_skip_static_build ---------------


class TestEdgeCountHeuristic:
    def _simple_chain_ontology(self):
        """A → B → C with one edge each (line graph)."""
        return FakeClient({
            "A": [_row("a1", pk_marker="PK"), _row("to_b[B].b1")],
            "B": [_row("b1", pk_marker="PK"), _row("to_c[C].c1")],
            "C": [_row("c1", pk_marker="PK")],
        })

    def test_zero_hops_returns_zero(self):
        ontology = Ontology(self._simple_chain_ontology())
        edge_index = EdgeIndex(ontology)
        assert estimate_subgraph_edge_count("A", edge_index, 0) == 0

    def test_one_hop(self):
        ontology = Ontology(self._simple_chain_ontology())
        edge_index = EdgeIndex(ontology)
        assert estimate_subgraph_edge_count("A", edge_index, 1) == 1

    def test_three_hops_reaches_all(self):
        ontology = Ontology(self._simple_chain_ontology())
        edge_index = EdgeIndex(ontology)
        assert estimate_subgraph_edge_count("A", edge_index, 3) == 2

    def test_should_skip_static_false_below_threshold(self):
        ontology = Ontology(self._simple_chain_ontology())
        edge_index = EdgeIndex(ontology)
        cfg = MetadataContextConfig(static_attempt_edge_threshold=100)
        # graph_depth=2 → only 2 edges, threshold 100 → don't skip static.
        assert not should_skip_static_build(2, "A", edge_index, cfg)

    def test_should_skip_static_false_when_depth_under_3(self):
        ontology = Ontology(self._simple_chain_ontology())
        edge_index = EdgeIndex(ontology)
        cfg = MetadataContextConfig(static_attempt_edge_threshold=0)
        # graph_depth=2 → below 3 hop threshold, never skip.
        assert not should_skip_static_build(2, "A", edge_index, cfg)

    def test_should_skip_static_true_at_3plus_above_threshold(self):
        # Hub-shaped ontology: A connects to many targets.
        hub_concepts = {
            "A": [_row("a1", pk_marker="PK")] + [
                _row(f"rel_{i}[T{i}].t1") for i in range(20)
            ],
        }
        for i in range(20):
            hub_concepts[f"T{i}"] = [_row("t1", pk_marker="PK")]
        ontology = Ontology(FakeClient(hub_concepts))
        edge_index = EdgeIndex(ontology)
        cfg = MetadataContextConfig(static_attempt_edge_threshold=5)
        assert should_skip_static_build(3, "A", edge_index, cfg)


# ---- retrieve_subgraph -----------------------------------------------------


class TestRetrieveSubgraph:
    def test_anchor_predecessor_is_none(self):
        client = FakeClient({"A": [_row("to_b[B].b1")], "B": []})
        ontology = Ontology(client)
        edge_index = EdgeIndex(ontology)
        concepts, preds, edges = retrieve_subgraph(
            "A", edge_index, MetadataContextConfig(), max_hop=2,
        )
        assert preds["A"] is None
        assert "A" in concepts

    def test_bfs_visits_in_order(self):
        client = FakeClient({
            "A": [_row("to_b[B].b1")],
            "B": [_row("to_c[C].c1")],
            "C": [],
        })
        ontology = Ontology(client)
        edge_index = EdgeIndex(ontology)
        concepts, preds, edges = retrieve_subgraph(
            "A", edge_index, MetadataContextConfig(), max_hop=3,
        )
        assert concepts == ["A", "B", "C"]
        assert preds["B"] == "A"
        assert preds["C"] == "B"

    def test_hub_concept_emits_all_outbound_edges(self):
        """With the hub cap removed, a hub-shaped concept emits ALL its
        outbound edges. DDL-size control now lives in the downstream
        concept pre-filter, not in BFS edge pruning."""
        hub_concepts = {"A": [_row(f"r{i}[T{i}].t1") for i in range(50)]}
        for i in range(50):
            hub_concepts[f"T{i}"] = [_row("t1")]
        client = FakeClient(hub_concepts)
        ontology = Ontology(client)
        edge_index = EdgeIndex(ontology)
        cfg = MetadataContextConfig()
        _concepts, _preds, edges = retrieve_subgraph(
            "A", edge_index, cfg, max_hop=2,
        )
        # All 50 edges from A must survive; no silent truncation.
        assert len(edges) == 50

    def test_self_ref_edges_kept_alongside_many_others(self):
        """Self-ref edges show up alongside all the rest — no special-casing
        needed now that the hub cap is gone."""
        client = FakeClient({
            "company": (
                [_row("has_acquired[company].name")]  # self-ref
                + [_row(f"rel_{i}[T{i}].x") for i in range(15)]
            ),
            **{f"T{i}": [_row("x")] for i in range(15)},
        })
        ontology = Ontology(client)
        edge_index = EdgeIndex(ontology)
        cfg = MetadataContextConfig()
        _concepts, _preds, edges = retrieve_subgraph(
            "company", edge_index, cfg, max_hop=1,
        )
        self_refs = [e for e in edges if e.is_self_ref]
        assert self_refs
        rel_names = {e.relationship_name for e in self_refs}
        assert "has_acquired" in rel_names
        # All 16 edges (1 self-ref + 15 others) should be present.
        assert len(edges) == 16

    def test_anchor_only_when_no_outbound(self):
        client = FakeClient({"X": [_row("name")]})
        ontology = Ontology(client)
        edge_index = EdgeIndex(ontology)
        concepts, preds, edges = retrieve_subgraph(
            "X", edge_index, MetadataContextConfig(),
        )
        assert concepts == ["X"]
        assert preds == {"X": None}
        assert edges == []


# ---- serialize_compact_ddl -------------------------------------------------


class TestSerializeCompactDDL:
    def test_emits_concepts_and_relationships(self):
        client = FakeClient({
            "A": [_row("a1", pk_marker="PK"), _row("to_b[B].b1")],
            "B": [_row("b1", pk_marker="PK")],
        })
        ontology = Ontology(client)
        edge_index = EdgeIndex(ontology)
        concepts, preds, edges = retrieve_subgraph(
            "A", edge_index, MetadataContextConfig(), max_hop=2,
        )
        ddl, stage = serialize_compact_ddl(
            concepts, edges, ontology, preds, MetadataContextConfig(),
        )
        # INHERITANCE and SELF_REF sections were removed — concepts-only layout.
        assert "## INHERITANCE" not in ddl
        assert "## SELF_REF" not in ddl
        assert "## CONCEPTS" in ddl
        assert "### A [anchor]" in ddl
        assert "### B [hop=1]" in ddl
        assert "-[to_b" in ddl  # outgoing rel from A
        assert stage == 1

    def test_cascade_reduces_under_tight_budget(self):
        # Build an ontology large enough that stage 1 overflows. Each T concept
        # has its own verbose props; A's outbound rels expose all of them.
        concepts_dict = {"A": [_row("a1", pk_marker="PK")] + [
            _row(f"to_t{i}[T{i}].t1") for i in range(20)
        ]}
        for i in range(20):
            concepts_dict[f"T{i}"] = [
                _row("t1", pk_marker="PK"),
                _row("verbose_prop_a"),
                _row("verbose_prop_b"),
                _row("verbose_prop_c"),
            ]
        client = FakeClient(concepts_dict)
        ontology = Ontology(client)
        edge_index = EdgeIndex(ontology)
        cfg = MetadataContextConfig(metadata_context_filter_max_tokens=300)
        concepts, preds, edges = retrieve_subgraph(
            "A", edge_index, cfg, max_hop=1,
        )
        ddl, stage = serialize_compact_ddl(concepts, edges, ontology, preds, cfg)
        # Should drop down at least one stage from full output.
        assert stage >= 2
        # Rels must NEVER be dropped, even at max compression.
        assert "-[to_t0" in ddl
        # Stage 4 = props content dropped (replaced with the truncation marker).
        if stage == 4:
            assert "props: [hidden by cascade" in ddl
            assert "verbose_prop_a" not in ddl

    def test_inverse_bounce_back_dropped_in_ddl(self):
        # B → A (inverse) should be filtered when serializing B because we came from A.
        client = FakeClient({
            "A": [_row("to_b[B].b1")],
            "B": [_row("~from_a[A].a1")],
        }, relationships=[
            {"concept": "B", "relationship_name": "from_a", "target_concept": "A",
             "is_inverse": 1, "is_mtm": 0, "source_properties": "", "target_properties": "", "description": ""},
        ])
        ontology = Ontology(client)
        edge_index = EdgeIndex(ontology)
        concepts, preds, edges = retrieve_subgraph(
            "A", edge_index, MetadataContextConfig(), max_hop=2,
        )
        ddl, _ = serialize_compact_ddl(
            concepts, edges, ontology, preds, MetadataContextConfig(),
        )
        # Inverse bounce-back from B back to A should NOT appear.
        assert "-[~from_a" not in ddl
        assert "from_a, " not in ddl  # not under B's rels:

    def test_cardinality_marker_present(self):
        client = FakeClient({
            "A": [_row("to_b[B].b1")],
            "B": [_row("b1", pk_marker="PK")],
        }, relationships=[
            {"concept": "A", "relationship_name": "to_b", "target_concept": "B",
             "is_inverse": 0, "is_mtm": 1, "source_properties": "", "target_properties": "", "description": ""},
        ])
        ontology = Ontology(client)
        edge_index = EdgeIndex(ontology)
        concepts, preds, edges = retrieve_subgraph(
            "A", edge_index, MetadataContextConfig(),
        )
        ddl, _ = serialize_compact_ddl(
            concepts, edges, ontology, preds, MetadataContextConfig(),
        )
        assert "N:M" in ddl
        # In the new format, cardinality appears inline on the rel edge.
        assert "-[to_b, N:M]-> B" in ddl

    def test_props_grouped_by_normalized_type(self):
        client = FakeClient({
            "A": [
                _row("name", data_type="varchar"),
                _row("description", data_type="text"),
                _row("count", data_type="integer"),
                _row("price", data_type="decimal(18,2)"),
                _row("created_at", data_type="date"),
                _row("is_active", data_type="boolean"),
            ]
        })
        ontology = Ontology(client)
        edge_index = EdgeIndex(ontology)
        concepts, preds, edges = retrieve_subgraph("A", edge_index, MetadataContextConfig())
        ddl, _ = serialize_compact_ddl(concepts, edges, ontology, preds, MetadataContextConfig())
        # Each group on its own line, lex-sorted within.
        assert "  str: description, name" in ddl
        assert "  num: count [int], price [dec]" in ddl
        assert "  date: created_at" in ddl
        assert "  bool: is_active" in ddl

    def test_type_hints_dropped_at_stage_2(self):
        """Stage 2 of the cascade drops [int]/[dec] hints (still in [num] group).

        Exercises the renderer at each stage explicitly via the internal
        _render so we don't depend on token counts to land us at a specific
        stage."""
        from langchain_timbr.ontology_context.context_builder.subgraph import _render

        client = FakeClient({"A": [_row("price", data_type="decimal")]})
        ontology = Ontology(client)
        edge_index = EdgeIndex(ontology)
        cfg = MetadataContextConfig()
        concepts, preds, edges = retrieve_subgraph("A", edge_index, cfg)
        hop_by = {"A": 0}
        # Stage 1: hint visible.
        stage1 = _render(1, concepts, edges, ontology, hop_by, cfg)
        assert "[dec]" in stage1
        assert "num: price [dec]" in stage1
        # Stage 2: hint dropped, num group kept.
        stage2 = _render(2, concepts, edges, ontology, hop_by, cfg)
        assert "[dec]" not in stage2
        assert "num: price" in stage2
        # Stage 3: measures dropped (no props change). Props still visible.
        stage3 = _render(3, concepts, edges, ontology, hop_by, cfg)
        assert "num: price" in stage3
        # Stage 4: props content dropped — replaced with a truncation marker
        # so the LLM distinguishes "no props" from "props hidden".
        stage4 = _render(4, concepts, edges, ontology, hop_by, cfg)
        assert "price" not in stage4
        assert "[hidden by cascade" in stage4
        assert "props: [hidden by cascade" in stage4

    def test_native_measures_only_no_relationship_scoped(self):
        """measures: should list ONLY direct measures, not relationship-scoped ones."""
        client = FakeClient({
            "A": [
                _row("measure.total_count"),
                _row("measure.of_b[B].some_rel_measure"),  # relationship-scoped
            ],
            "B": [_row("b1")],
        })
        ontology = Ontology(client)
        edge_index = EdgeIndex(ontology)
        concepts, preds, edges = retrieve_subgraph("A", edge_index, MetadataContextConfig())
        ddl, _ = serialize_compact_ddl(concepts, edges, ontology, preds, MetadataContextConfig())
        # measures: line under A contains only the native measure.
        assert "measures: total_count" in ddl
        # Relationship-scoped measure should NOT appear under A.
        # (It would appear under B's own block as a native measure if B had it.)
        assert "some_rel_measure" not in ddl

    def test_entity_columns_dropped(self):
        """entity_id, entity_label, entity_type are filtered out of the DDL."""
        client = FakeClient({
            "A": [
                _row("entity_id", data_type="integer"),
                _row("entity_label", data_type="varchar"),
                _row("entity_type", data_type="varchar"),
                _row("real_prop", data_type="varchar"),
            ]
        })
        ontology = Ontology(client)
        edge_index = EdgeIndex(ontology)
        concepts, preds, edges = retrieve_subgraph("A", edge_index, MetadataContextConfig())
        ddl, _ = serialize_compact_ddl(concepts, edges, ontology, preds, MetadataContextConfig())
        assert "entity_id" not in ddl
        assert "entity_label" not in ddl
        assert "entity_type" not in ddl
        assert "real_prop" in ddl  # the real column survives

    def test_self_ref_inlined_in_rels_block_with_recursive_marker(self):
        """Self-referential edges appear inline in the concept's rels: block,
        marked with `# recursive` (no separate SELF_REF section)."""
        client = FakeClient({
            "work_item": [
                _row("title"),
                _row("has_child[work_item].child_id"),
            ]
        })
        ontology = Ontology(client)
        edge_index = EdgeIndex(ontology)
        concepts, preds, edges = retrieve_subgraph(
            "work_item", edge_index, MetadataContextConfig(),
        )
        ddl, _ = serialize_compact_ddl(
            concepts, edges, ontology, preds, MetadataContextConfig(),
        )
        # No SELF_REF section anywhere.
        assert "## SELF_REF" not in ddl
        # work_item's rels block contains has_child with the recursive marker.
        wi_block = ddl.split("### work_item [anchor]")[1]
        assert "rels:" in wi_block
        assert "-[has_child" in wi_block
        assert "-> work_item" in wi_block
        assert "# recursive" in wi_block

    def test_transitivity_marker_on_rel_edge(self):
        """Transitive relationships show *N suffix on the rel name."""
        client = FakeClient({
            "A": [_row("has_x[B*3].b1")],
            "B": [_row("b1")],
        })
        ontology = Ontology(client)
        edge_index = EdgeIndex(ontology)
        concepts, preds, edges = retrieve_subgraph(
            "A", edge_index, MetadataContextConfig(),
        )
        ddl, _ = serialize_compact_ddl(
            concepts, edges, ontology, preds, MetadataContextConfig(),
        )
        assert "-[has_x*3," in ddl

    def test_inverse_marker_NOT_shown_on_rel_edge(self):
        """Inverse relationships (is_inverse=True) do NOT get a ~ prefix in
        the DDL output. The LLM doesn't need the inverse-vs-canonical
        distinction — every edge in the rels list is a valid traversal
        target. Bounce-back inverses are still dropped upstream by
        should_include_in_ddl."""
        # ~of_customer column on order → is_inverse=True on the edge. With
        # anchor=order (no previous hop), this inverse is NOT a bounce-back
        # so it's retained, but should appear without the ~ prefix.
        client = FakeClient({
            "order": [_row("~of_customer[customer].name")],
            "customer": [_row("name")],
        })
        ontology = Ontology(client)
        edge_index = EdgeIndex(ontology)
        concepts, preds, edges = retrieve_subgraph(
            "order", edge_index, MetadataContextConfig(),
        )
        ddl, _ = serialize_compact_ddl(
            concepts, edges, ontology, preds, MetadataContextConfig(),
        )
        # Edge IS in the output (it's not a bounce-back from order's perspective).
        assert "-[of_customer," in ddl
        # But the ~ prefix is NOT shown.
        assert "~of_customer" not in ddl

    def test_subconcepts_emitted_when_include_logic_concepts_enabled(self):
        """With include_logic_concepts=True, each concept block lists its
        sub-concepts under a `subconcepts:` line."""
        # Two concepts where europe_order inherits from order.
        client = FakeClient(
            {
                "order": [_row("id")],
                "europe_order": [_row("id")],
            },
            inheritance_rows=[
                {"concept": "europe_order", "inheritance": "order,thing"},
                {"concept": "order", "inheritance": "thing"},
            ],
        )
        ontology = Ontology(client)
        edge_index = EdgeIndex(ontology)
        # Force a 2-concept subgraph by feeding both in.
        concepts = ["order", "europe_order"]
        preds = {"order": None, "europe_order": "order"}
        edges = []  # no edges needed
        cfg = MetadataContextConfig(include_logic_concepts=True)
        ddl, _ = serialize_compact_ddl(concepts, edges, ontology, preds, cfg)
        assert "subconcepts: europe_order" in ddl

    def test_subconcepts_NOT_emitted_when_flag_off(self):
        """include_logic_concepts=False → no subconcepts: line."""
        client = FakeClient(
            {
                "order": [_row("id")],
                "europe_order": [_row("id")],
            },
            inheritance_rows=[
                {"concept": "europe_order", "inheritance": "order,thing"},
            ],
        )
        ontology = Ontology(client)
        concepts = ["order", "europe_order"]
        preds = {"order": None, "europe_order": "order"}
        edges = []
        cfg = MetadataContextConfig(include_logic_concepts=False)
        ddl, _ = serialize_compact_ddl(concepts, edges, ontology, preds, cfg)
        assert "subconcepts:" not in ddl

    def test_inheritance_section_not_emitted(self):
        """The ## INHERITANCE section was removed — even with inheritance rows
        present in sys_ontology, the DDL must NOT include it."""
        client = FakeClient(
            {"company": [_row("name")]},
            inheritance_rows=[
                {"concept": "company", "inheritance": "organization,thing"},
            ],
        )
        ontology = Ontology(client)
        edge_index = EdgeIndex(ontology)
        concepts, preds, edges = retrieve_subgraph(
            "company", edge_index, MetadataContextConfig(),
        )
        ddl, _ = serialize_compact_ddl(
            concepts, edges, ontology, preds, MetadataContextConfig(),
        )
        assert "## INHERITANCE" not in ddl
        assert "company :> organization" not in ddl

    def test_incoming_block_lists_subgraph_sources(self):
        """`incoming:` lists relationships from other concepts in the subgraph
        pointing TO this concept. Format: `- <source>.<rel_name>`."""
        client = FakeClient({
            "A": [_row("a1"), _row("to_b[B].b1")],
            "B": [_row("b1"), _row("to_c[C].c1")],
            "C": [_row("c1")],
        })
        ontology = Ontology(client)
        edge_index = EdgeIndex(ontology)
        concepts, preds, edges = retrieve_subgraph(
            "A", edge_index, MetadataContextConfig(), max_hop=2,
        )
        ddl, _ = serialize_compact_ddl(
            concepts, edges, ontology, preds, MetadataContextConfig(),
        )
        # C's block should list A→C? No — only direct incoming (B→C).
        c_block = ddl.split("### C [hop=2]")[1]
        assert "incoming:" in c_block
        assert "- B.to_c" in c_block
        # B's block should list A→B.
        b_block = ddl.split("### B [hop=1]")[1].split("### C")[0]
        assert "incoming:" in b_block
        assert "- A.to_b" in b_block

    def test_incoming_block_absent_for_anchor_when_no_inbound(self):
        """When the anchor has no in-subgraph inbound edges, no `incoming:`
        block is emitted under it."""
        client = FakeClient({
            "A": [_row("a1"), _row("to_b[B].b1")],
            "B": [_row("b1")],
        })
        ontology = Ontology(client)
        edge_index = EdgeIndex(ontology)
        concepts, preds, edges = retrieve_subgraph(
            "A", edge_index, MetadataContextConfig(), max_hop=1,
        )
        ddl, _ = serialize_compact_ddl(
            concepts, edges, ontology, preds, MetadataContextConfig(),
        )
        a_block = ddl.split("### A [anchor]")[1].split("### B")[0]
        assert "incoming:" not in a_block

    def test_truncation_marker_only_emitted_when_concept_has_data(self):
        """At stage 4, `props: [hidden by cascade ...]` appears ONLY for
        concepts that actually have selectable properties — concepts with
        zero props don't get a false truncation marker."""
        from langchain_timbr.ontology_context.context_builder.subgraph import _render
        from langchain_timbr.ontology_context.context_builder.subgraph import _compute_hop_distances

        client = FakeClient({
            "with_props": [_row("a"), _row("b"), _row("to_no_props[no_props].x")],
            "no_props": [],
        })
        ontology = Ontology(client)
        edge_index = EdgeIndex(ontology)
        concepts, preds, edges = retrieve_subgraph(
            "with_props", edge_index, MetadataContextConfig(), max_hop=1,
        )
        hop_by = _compute_hop_distances(concepts, preds)
        cfg = MetadataContextConfig()

        stage4 = _render(4, concepts, edges, ontology, hop_by, cfg)
        # The concept with props gets a truncation marker.
        with_props_block = stage4.split("### with_props [anchor]")[1].split("### no_props")[0]
        assert "props: [hidden by cascade" in with_props_block
        # The concept with NO props gets no marker (genuine absence).
        no_props_block = stage4.split("### no_props")[1]
        assert "props:" not in no_props_block

    def test_measures_truncation_marker_at_stage_3(self):
        """At stage 3, `measures: [hidden by cascade ...]` replaces the
        list for concepts that have measures."""
        from langchain_timbr.ontology_context.context_builder.subgraph import _render
        from langchain_timbr.ontology_context.context_builder.subgraph import _compute_hop_distances

        client = FakeClient({
            "X": [_row("name"), _row("measure.total_count")],
        })
        ontology = Ontology(client)
        edge_index = EdgeIndex(ontology)
        concepts, preds, edges = retrieve_subgraph(
            "X", edge_index, MetadataContextConfig(),
        )
        hop_by = _compute_hop_distances(concepts, preds)
        cfg = MetadataContextConfig()

        stage3 = _render(3, concepts, edges, ontology, hop_by, cfg)
        assert "measures: [hidden by cascade" in stage3
        # The actual measure name is gone.
        assert "total_count" not in stage3

    def test_incoming_block_excludes_self_refs(self):
        """Self-refs are conveyed by the outgoing `# recursive` marker.
        They must NOT also appear under `incoming:` (avoid duplication)."""
        client = FakeClient({
            "work_item": [
                _row("title"),
                _row("has_child[work_item].child_id"),
            ],
        })
        ontology = Ontology(client)
        edge_index = EdgeIndex(ontology)
        concepts, preds, edges = retrieve_subgraph(
            "work_item", edge_index, MetadataContextConfig(),
        )
        ddl, _ = serialize_compact_ddl(
            concepts, edges, ontology, preds, MetadataContextConfig(),
        )
        # `# recursive` appears in rels — self-ref is conveyed there.
        assert "# recursive" in ddl
        # `incoming:` block must NOT list the self-ref.
        # (work_item.has_child going TO work_item itself.)
        assert "- work_item.has_child" not in ddl

    def test_reachable_menu_band_preserves_caller_order(self):
        """The serializer preserves the CALLER's ordering for the menu band —
        it only dedupes. The caller (typically ``menu_builder.split_bands``)
        sorts by shortest-hop then alphabetical within hop. Tested directly
        here at the serializer level by passing a specific input order and
        asserting it's preserved verbatim."""
        client = FakeClient({
            "anchor_c": [_row("name"), _row("to_b[B].x")],
            "B": [_row("name")],
        })
        ontology = Ontology(client)
        edge_index = EdgeIndex(ontology)
        concepts, preds, edges = retrieve_subgraph(
            "anchor_c", edge_index, MetadataContextConfig(),
        )
        # Caller passes already-sorted-by-(hop, lex). Serializer must NOT re-sort.
        ddl, _ = serialize_compact_ddl(
            concepts, edges, ontology, preds, MetadataContextConfig(),
            menu_concepts=["zulu", "alpha", "mike"],
        )
        assert "## REACHABLE: zulu, alpha, mike" in ddl

    def test_reachable_menu_band_dedupes_while_preserving_order(self):
        """When the caller (defensively) passes duplicates, the serializer
        dedupes — keeping the first occurrence so caller ordering is preserved."""
        client = FakeClient({
            "anchor_c": [_row("name")],
        })
        ontology = Ontology(client)
        edge_index = EdgeIndex(ontology)
        concepts, preds, edges = retrieve_subgraph(
            "anchor_c", edge_index, MetadataContextConfig(),
        )
        ddl, _ = serialize_compact_ddl(
            concepts, edges, ontology, preds, MetadataContextConfig(),
            menu_concepts=["alpha", "beta", "alpha", "gamma", "beta"],
        )
        assert "## REACHABLE: alpha, beta, gamma" in ddl

    def test_reachable_menu_band_absent_when_no_menu_concepts(self):
        """No menu_concepts → no ## REACHABLE header (backward compat)."""
        client = FakeClient({
            "A": [_row("name")],
        })
        ontology = Ontology(client)
        edge_index = EdgeIndex(ontology)
        concepts, preds, edges = retrieve_subgraph(
            "A", edge_index, MetadataContextConfig(),
        )
        ddl, _ = serialize_compact_ddl(
            concepts, edges, ontology, preds, MetadataContextConfig(),
        )
        assert "## REACHABLE" not in ddl


# ---------------------------------------------------------------------------
# Fix 2 + Fix 4 — mandatory rels:, descriptions, minimal blocks
# ---------------------------------------------------------------------------


def _client_with_concept_descriptions():
    """FakeClient extended with concept-level descriptions on describe_concept
    rows. Used to exercise Fix 4A's universal-description rendering."""
    class ClientWithDescriptions(FakeClient):
        def __init__(self, concepts, descriptions, **kwargs):
            super().__init__(concepts, **kwargs)
            self._descriptions = descriptions

        def describe_concept(self, name):
            return list(self._concepts.get(name, []))

    return ClientWithDescriptions


class _OntologyWithDesc(Ontology):
    """Ontology subclass that overrides concept descriptions via a passed-in
    dict, so we can test the description: line without touching the parser."""

    def __init__(self, client, concept_descriptions=None):
        super().__init__(client)
        self._concept_descriptions = concept_descriptions or {}

    def get_concept_metadata(self, name):
        meta = super().get_concept_metadata(name)
        desc = self._concept_descriptions.get(name)
        if desc is None:
            return meta
        from dataclasses import replace
        return replace(meta, description=desc)


class TestMandatoryRelsBlock:
    """Fix 2: detail-band concepts MUST emit a `rels:` block — sentinel
    `rels: (none)` when no outgoing edges are in scope, so the LLM can
    distinguish a leaf concept from missing data."""

    def test_detail_concept_with_no_outbound_renders_rels_none(self):
        client = FakeClient({
            "A": [_row("a1", pk_marker="PK"), _row("to_b[B].b1")],
            "B": [_row("b1", pk_marker="PK")],   # B has no outbound edges
        })
        ontology = Ontology(client)
        edge_index = EdgeIndex(ontology)
        concepts, preds, edges = retrieve_subgraph(
            "A", edge_index, MetadataContextConfig(), max_hop=2,
        )
        ddl, _ = serialize_compact_ddl(
            concepts, edges, ontology, preds, MetadataContextConfig(),
        )
        # B is a leaf → its block must explicitly say `rels: (none)`.
        b_block = ddl.split("### B [hop=1]")[1]
        b_block_only = b_block.split("###")[0]   # cut at next concept heading
        assert "rels: (none)" in b_block_only

    def test_detail_concept_with_outbound_renders_normal_rels(self):
        """Sanity: existing behavior preserved when outbound edges exist."""
        client = FakeClient({
            "A": [_row("a1", pk_marker="PK"), _row("to_b[B].b1")],
            "B": [_row("b1", pk_marker="PK")],
        })
        ontology = Ontology(client)
        edge_index = EdgeIndex(ontology)
        concepts, preds, edges = retrieve_subgraph(
            "A", edge_index, MetadataContextConfig(), max_hop=2,
        )
        ddl, _ = serialize_compact_ddl(
            concepts, edges, ontology, preds, MetadataContextConfig(),
        )
        a_block = ddl.split("### A [anchor]")[1].split("###")[0]
        assert "rels:" in a_block
        assert "(none)" not in a_block   # A has an outbound edge


class TestUniversalDescriptions:
    """Fix 4A: every concept block emits a `description:` line when the
    ontology has one, and every `rels:` line carries an inline
    `# <description>` annotation when the relationship has one."""

    def test_concept_description_line_emitted_when_present(self):
        client = FakeClient({
            "A": [_row("a1"), _row("to_b[B].b1")],
            "B": [_row("b1")],
        })
        ontology = _OntologyWithDesc(client, concept_descriptions={
            "A": "primary entity",
            "B": "target entity",
        })
        edge_index = EdgeIndex(ontology)
        concepts, preds, edges = retrieve_subgraph(
            "A", edge_index, MetadataContextConfig(),
        )
        ddl, _ = serialize_compact_ddl(
            concepts, edges, ontology, preds, MetadataContextConfig(),
        )
        # Description appears between heading and the first body block.
        a_block = ddl.split("### A [anchor]")[1].split("###")[0]
        assert "description: primary entity" in a_block
        b_block = ddl.split("### B [hop=1]")[1].split("###")[0]
        assert "description: target entity" in b_block

    def test_concept_description_omitted_when_absent(self):
        """Concepts without a description silently skip the line — no empty
        `description:` shows up."""
        client = FakeClient({"A": [_row("a1")]})
        ontology = Ontology(client)   # parser sets description=None
        edge_index = EdgeIndex(ontology)
        concepts, preds, edges = retrieve_subgraph(
            "A", edge_index, MetadataContextConfig(),
        )
        ddl, _ = serialize_compact_ddl(
            concepts, edges, ontology, preds, MetadataContextConfig(),
        )
        a_block = ddl.split("### A [anchor]")[1].split("###")[0]
        assert "description:" not in a_block

    def test_rel_description_inline_annotation(self):
        """When EdgeMeta.description is non-empty, the rels: line includes
        `  # <description>` inline."""
        client = FakeClient({
            "A": [_row("to_b[B].b1")],
            "B": [_row("b1")],
        }, relationships=[
            {"concept": "A", "relationship_name": "to_b",
             "target_concept": "B", "is_inverse": 0, "is_mtm": 0,
             "source_properties": "", "target_properties": "",
             "description": "A links to B via foreign key"},
        ])
        ontology = Ontology(client)
        edge_index = EdgeIndex(ontology)
        concepts, preds, edges = retrieve_subgraph(
            "A", edge_index, MetadataContextConfig(),
        )
        ddl, _ = serialize_compact_ddl(
            concepts, edges, ontology, preds, MetadataContextConfig(),
        )
        assert "# A links to B via foreign key" in ddl

    def test_self_ref_combines_recursive_and_description_annotations(self):
        """When a self-ref edge ALSO has a description, the inline annotation
        combines both: `# recursive | <description>`."""
        client = FakeClient({
            "work_item": [
                _row("title"),
                _row("has_child[work_item].child_id"),
            ],
        }, relationships=[
            {"concept": "work_item", "relationship_name": "has_child",
             "target_concept": "work_item", "is_inverse": 0, "is_mtm": 0,
             "source_properties": "", "target_properties": "",
             "description": "parent of"},
        ])
        ontology = Ontology(client)
        edge_index = EdgeIndex(ontology)
        concepts, preds, edges = retrieve_subgraph(
            "work_item", edge_index, MetadataContextConfig(),
        )
        ddl, _ = serialize_compact_ddl(
            concepts, edges, ontology, preds, MetadataContextConfig(),
        )
        assert "# recursive | parent of" in ddl


class TestMinimalBlocks:
    """Fix 4B: expand_to-promoted concepts render as MINIMAL blocks —
    description + connecting rels only (no props, no measures, no
    incoming)."""

    def test_minimal_block_renders_description_and_filtered_rels_only(self):
        """Pass an expand_minimal_concepts set with a connecting rel triple
        and verify the minimal block omits props/measures/incoming but
        keeps description + the filtered rels: line."""
        # 4-concept chain: A → B → C → D.
        client = FakeClient({
            "A": [_row("a1"), _row("to_b[B].b1")],
            "B": [_row("b1"), _row("to_c[C].c1")],
            "C": [_row("c1"), _row("to_d[D].d1")],
            "D": [
                _row("d1"),   # D has a prop
                _row("measure.d_measure"),   # and a measure
            ],
        })
        ontology = _OntologyWithDesc(client, concept_descriptions={
            "D": "downstream destination",
        })
        edge_index = EdgeIndex(ontology)
        concepts, preds, edges = retrieve_subgraph(
            "A", edge_index, MetadataContextConfig(), max_hop=3,
        )
        # Render with A,B,C in detail; D as minimal-block via expand_to.
        ddl, _ = serialize_compact_ddl(
            ["A", "B", "C"], edges, ontology, preds, MetadataContextConfig(),
            expand_minimal_concepts=["D"],
        )
        # D's minimal block exists with its heading + description.
        assert "### D [hop=3]" in ddl
        d_block = ddl.split("### D [hop=3]")[1].split("###")[0]
        assert "description: downstream destination" in d_block
        # D's minimal block MUST NOT contain props/measures/incoming.
        assert "props:" not in d_block
        assert "measures:" not in d_block
        assert "incoming:" not in d_block
        assert "d_measure" not in d_block
        # D's outgoing edges (none in this ontology) → minimal block uses
        # the sentinel.
        assert "rels: (none)" in d_block

    def test_minimal_block_concept_removed_from_reachable_band(self):
        """When a concept earns a minimal block, it should be REMOVED from
        the ## REACHABLE band (no longer "behind the curtain")."""
        client = FakeClient({
            "A": [_row("a1"), _row("to_b[B].b1")],
            "B": [_row("b1"), _row("to_c[C].c1")],
            "C": [_row("c1")],
        })
        ontology = Ontology(client)
        edge_index = EdgeIndex(ontology)
        concepts, preds, edges = retrieve_subgraph(
            "A", edge_index, MetadataContextConfig(),
        )
        ddl, _ = serialize_compact_ddl(
            ["A"], edges, ontology, preds, MetadataContextConfig(),
            menu_concepts=["B", "C"],
            expand_minimal_concepts=["B"],
        )
        # B got a minimal block → not in REACHABLE band.
        reachable_lines = [
            ln for ln in ddl.splitlines() if ln.startswith("## REACHABLE")
        ]
        assert reachable_lines, "## REACHABLE band should still be present (C remains)"
        # Parse the comma-separated names so 'B' doesn't substring-match 'REACHABLE'.
        names_part = reachable_lines[0].split(":", 1)[1].strip()
        emitted_names = {n.strip() for n in names_part.split(",")}
        assert "B" not in emitted_names
        assert "C" in emitted_names
        # B has its minimal block heading.
        assert "### B [hop=1]" in ddl
        # B's minimal block carries B's COMPLETE outgoing rels (no on-path
        # filter). The edge B → C IS present even though C stays in
        # ## REACHABLE — showing an edge does NOT promote its target.
        b_block = ddl.split("### B [hop=1]")[1].split("###")[0]
        assert "-[to_c," in b_block
        assert "-> C" in b_block


class TestMinimalBlockFullOutgoingRels:
    """User update: minimal blocks emit the concept's COMPLETE outgoing
    rels, bounded only by the BFS truncation at ``max_graph_depth``. One expand
    reveals the full chain — no on-path/connecting-only filter, so the
    planner can build_path STRAIGHT THROUGH the expanded concept on the
    next round without a second expand_to."""

    def test_expanding_a_mid_chain_concept_reveals_onward_edge(self):
        """detail_depth=1, expand product (hop=2). product's outbound to
        material (hop=3) must appear in its minimal block, even though
        material itself stays in ## REACHABLE. The visible edge lets the
        planner walk customer → order → product → material in ONE
        follow-up build_path round."""
        # customer → order → product → material chain.
        client = FakeClient({
            "customer": [_row("name"), _row("made_order[order].x")],
            "order": [_row("x"), _row("includes_product[product].x")],
            "product": [_row("x"), _row("contains[material].x")],
            "material": [_row("name")],
        })
        ontology = Ontology(client)
        edge_index = EdgeIndex(ontology)
        # BFS to max_graph_depth=5 so material is in the edge set.
        concepts_full, preds, edges = retrieve_subgraph(
            "customer", edge_index, MetadataContextConfig(), max_hop=5,
        )
        # detail_depth=1 → only customer + order in detail. product
        # promoted via expand_to to a minimal block. material stays in
        # ## REACHABLE (NOT promoted by the rels: line showing it).
        ddl, _ = serialize_compact_ddl(
            ["customer", "order"], edges, ontology, preds,
            MetadataContextConfig(),
            menu_concepts=["product", "material"],
            expand_minimal_concepts=["product"],
        )
        # product's minimal block carries its FULL outgoing rels — the
        # contains → material edge is visible.
        assert "### product [hop=2]" in ddl
        product_block = ddl.split("### product [hop=2]")[1].split("###")[0]
        assert "-[contains," in product_block
        assert "-> material" in product_block
        # The minimal block STAYS minimal in the other dimensions: no
        # props, measures, or incoming.
        assert "props:" not in product_block
        assert "measures:" not in product_block
        assert "incoming:" not in product_block
        # material is NOT promoted to a minimal block — only the named
        # expand_to target (product) earns one. material stays as a name
        # in ## REACHABLE.
        assert "### material" not in ddl
        reachable_line = [
            ln for ln in ddl.splitlines() if ln.startswith("## REACHABLE")
        ][0]
        names = {n.strip() for n in reachable_line.split(":", 1)[1].split(",")}
        assert "material" in names

    def test_minimal_block_bound_by_max_graph_depth_naturally(self):
        """Because the renderer uses the BFS-truncated edge set, an
        outgoing edge whose target sits past ``max_graph_depth`` is never
        materialized into ``edges`` in the first place. Concepts at hop
        ``max_graph_depth`` therefore render ``rels: (none)`` in a minimal
        block — the bound is implicit, not enforced post-hoc."""
        # Chain a → b → c → d → e, with max_graph_depth=2 truncation.
        client = FakeClient({
            "a": [_row("to_b[b].x")],
            "b": [_row("to_c[c].x")],
            "c": [_row("to_d[d].x")],
            "d": [_row("to_e[e].x")],
            "e": [_row("name")],
        })
        ontology = Ontology(client)
        edge_index = EdgeIndex(ontology)
        # max_hop=2 → BFS stops at c (hop=2); c's outbound edges are
        # NOT enumerated.
        concepts_full, preds, edges = retrieve_subgraph(
            "a", edge_index, MetadataContextConfig(), max_hop=2,
        )
        # Render c as a minimal block. With BFS truncation, c's
        # outgoing edges aren't in the edges list → rels: (none).
        ddl, _ = serialize_compact_ddl(
            ["a", "b"], edges, ontology, preds, MetadataContextConfig(),
            menu_concepts=["c"],
            expand_minimal_concepts=["c"],
        )
        assert "### c [hop=2]" in ddl
        c_block = ddl.split("### c [hop=2]")[1].split("###")[0]
        # Bound by max_graph_depth — c's onward edges are outside the
        # retained edge set, so the minimal block emits the sentinel.
        assert "rels: (none)" in c_block
