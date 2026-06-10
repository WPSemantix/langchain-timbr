"""Tests for menu_builder.py — hop_map BFS, band split, edge materialization,
and depth-validation integration into the orchestrator.
"""

from __future__ import annotations

import json
from typing import List

import pytest

from langchain_timbr.ontology_context.context_builder import build_filtered as _bf
from langchain_timbr.ontology_context.context_builder.edge_index import EdgeIndex
from langchain_timbr.ontology_context.context_builder.menu_builder import (
    MenuEntry,
    build_hop_map,
    materialize_concept_outbound_edges,
    split_bands,
)
from langchain_timbr.ontology_context.context_builder.metadata_config import (
    MetadataContextConfig,
)
from langchain_timbr.ontology_context.ontology.graph import Ontology


def _row(col_name, **extras):
    base = {
        "col_name": col_name, "data_type": "varchar", "comment": "",
        "inheritance_marker": "", "pk_marker": "",
    }
    base.update(extras)
    return base


class FakeClient:
    def __init__(self, concepts, version="v1"):
        self._concepts = concepts
        self._version = version

    def fetch_version_id(self):
        return self._version

    def describe_concept(self, name):
        return list(self._concepts.get(name, []))

    def fetch_relationships_meta(self):
        return []

    def fetch_inheritance_meta(self):
        return []


def _line_ontology():
    """Linear: a -> b -> c -> d -> e -> f (single chain, 5 hops)."""
    return Ontology(FakeClient({
        "a": [_row("to_b[b].x")],
        "b": [_row("to_c[c].x")],
        "c": [_row("to_d[d].x")],
        "d": [_row("to_e[e].x")],
        "e": [_row("to_f[f].x")],
        "f": [],
    }))


def _branch_ontology():
    """Anchor 'A' fans out to multiple concepts at varying hops.
    A → B (hop 1), C (hop 1)
    B → D (hop 2)
    C → E (hop 2)
    D → F (hop 3)
    """
    return Ontology(FakeClient({
        "A": [_row("to_b[B].x"), _row("to_c[C].x")],
        "B": [_row("to_d[D].x")],
        "C": [_row("to_e[E].x")],
        "D": [_row("to_f[F].x")],
        "E": [],
        "F": [],
    }))


# ---------------------------------------------------------------------------
# build_hop_map
# ---------------------------------------------------------------------------


class TestBuildHopMap:
    def test_anchor_at_hop_zero(self):
        edge_index = EdgeIndex(_line_ontology())
        hop_map = build_hop_map("a", edge_index, max_graph_depth=5)
        assert hop_map["a"] == 0

    def test_linear_chain_assigns_correct_hops(self):
        edge_index = EdgeIndex(_line_ontology())
        hop_map = build_hop_map("a", edge_index, max_graph_depth=5)
        assert hop_map == {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5}

    def test_max_graph_depth_caps_bfs(self):
        edge_index = EdgeIndex(_line_ontology())
        hop_map = build_hop_map("a", edge_index, max_graph_depth=3)
        # 'd' is at hop 3, 'e' and 'f' must be absent (hop > 3).
        assert "d" in hop_map
        assert "e" not in hop_map
        assert "f" not in hop_map

    def test_branch_uses_shortest_hop(self):
        """When a concept is reachable via multiple paths, the smallest hop wins."""
        edge_index = EdgeIndex(_branch_ontology())
        hop_map = build_hop_map("A", edge_index, max_graph_depth=5)
        assert hop_map["A"] == 0
        assert hop_map["B"] == 1
        assert hop_map["C"] == 1
        assert hop_map["D"] == 2
        assert hop_map["E"] == 2
        assert hop_map["F"] == 3

    def test_zero_max_graph_depth_returns_anchor_only(self):
        edge_index = EdgeIndex(_line_ontology())
        hop_map = build_hop_map("a", edge_index, max_graph_depth=0)
        assert hop_map == {"a": 0}


# ---------------------------------------------------------------------------
# split_bands
# ---------------------------------------------------------------------------


class TestSplitBands:
    def test_detail_includes_anchor_and_within_depth(self):
        hop_map = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5}
        detail, menu = split_bands(hop_map, detail_depth=3, max_graph_depth=5)
        assert set(detail) == {"a", "b", "c", "d"}
        assert {e.concept for e in menu} == {"e", "f"}

    def test_menu_entries_tagged_depth_band(self):
        hop_map = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}
        _, menu = split_bands(hop_map, detail_depth=2, max_graph_depth=5)
        assert all(e.source == "depth_band" for e in menu)

    def test_menu_sorted_shortest_hop_then_lex(self):
        hop_map = {"a": 0, "z": 3, "y": 3, "b": 1, "x": 4, "w": 4}
        _, menu = split_bands(hop_map, detail_depth=1, max_graph_depth=5)
        # Expect: hop 3: y, z (lex within hop); hop 4: w, x.
        ordered = [(e.hop, e.concept) for e in menu]
        assert ordered == [(3, "y"), (3, "z"), (4, "w"), (4, "x")]

    def test_concepts_beyond_max_graph_depth_excluded(self):
        hop_map = {"a": 0, "b": 1, "c": 2, "d": 5, "e": 8}
        detail, menu = split_bands(hop_map, detail_depth=1, max_graph_depth=5)
        # detail_depth=1 → hops 0,1 are detail; hops 2..5 are menu;
        # hop 8 (e) is beyond max_graph_depth and excluded entirely.
        assert set(detail) == {"a", "b"}
        assert {entry.concept for entry in menu} == {"c", "d"}
        # 'e' (hop 8 > max_graph_depth 5) must NOT appear anywhere.
        assert "e" not in detail
        assert all(entry.concept != "e" for entry in menu)


# ---------------------------------------------------------------------------
# materialize_concept_outbound_edges
# ---------------------------------------------------------------------------


class TestMaterializeOutboundEdges:
    def test_adds_new_outbound_edges_to_list(self):
        edge_index = EdgeIndex(_line_ontology())
        # No edges yet — empty list + seen set.
        edges: List = []
        edge_seen = set()
        added = materialize_concept_outbound_edges("c", edge_index, edges, edge_seen)
        # 'c' has one outbound edge to 'd' via 'to_c'.
        assert len(added) == 1
        assert added[0].from_concept == "c"
        assert added[0].to_concept == "d"
        assert added[0] in edges

    def test_idempotent_via_edge_seen(self):
        edge_index = EdgeIndex(_line_ontology())
        edges: List = []
        edge_seen = set()
        materialize_concept_outbound_edges("c", edge_index, edges, edge_seen)
        # Second call must not re-add.
        added_again = materialize_concept_outbound_edges("c", edge_index, edges, edge_seen)
        assert added_again == []
        assert len(edges) == 1

    def test_no_op_for_concept_with_no_outbound_edges(self):
        edge_index = EdgeIndex(_line_ontology())
        edges: List = []
        edge_seen = set()
        added = materialize_concept_outbound_edges("f", edge_index, edges, edge_seen)
        assert added == []
        assert edges == []


# ---------------------------------------------------------------------------
# Depth-ordering validation in the orchestrator
# ---------------------------------------------------------------------------


class TestDepthClamping:
    """The orchestrator is forgiving — when ``graph_depth >= max_graph_depth``,
    it clamps to ``max_graph_depth - 1`` rather than raising. This lets chains
    configured before the menu-band split landed (e.g. ``graph_depth=5``
    against ``max_graph_depth=5``) keep working without a config change.
    The clamp is logged + recorded in stats so operators can observe it."""

    def _scripted_build_path(self):
        class ScriptedLLM:
            def __init__(self):
                self.calls = 0

            def invoke(self, _):
                self.calls += 1

                class R:
                    content = json.dumps({
                        "action": "build_path",
                        "selected_concepts": ["a", "b"],
                        "selected_paths": [{
                            "path_id": "P1",
                            "purpose": "",
                            "segments": [{"from": "a", "rel": "to_b", "to": "b"}],
                            "is_recursive": False,
                        }],
                    })
                return R()
        return ScriptedLLM()

    def test_graph_depth_equal_to_max_graph_depth_clamps(self):
        """``graph_depth == max_graph_depth`` should NOT raise — clamp to
        ``max_graph_depth - 1`` and emit a clamp warning + stat."""
        ontology = _line_ontology()
        cfg = MetadataContextConfig(
            mode="dynamic", max_graph_depth=3,
            metadata_context_dynamic_retry=0,
            max_detail_concepts=100,
        )
        result = _bf.build_filtered_metadata(
            question="any", anchor="a", ontology=ontology,
            llm=self._scripted_build_path(),
            config=cfg, graph_depth=3,
        )
        assert result.error is None
        assert result.stats["graph_depth_clamped_from"] == 3
        assert result.stats["graph_depth_clamped_to"] == 2
        assert any("graph_depth_clamped" in w for w in result.warnings)

    def test_graph_depth_greater_than_max_graph_depth_clamps(self):
        """``graph_depth > max_graph_depth`` clamps to ``max_graph_depth - 1`` as well."""
        ontology = _line_ontology()
        cfg = MetadataContextConfig(
            mode="dynamic", max_graph_depth=3,
            metadata_context_dynamic_retry=0,
            max_detail_concepts=100,
        )
        result = _bf.build_filtered_metadata(
            question="any", anchor="a", ontology=ontology,
            llm=self._scripted_build_path(),
            config=cfg, graph_depth=10,
        )
        assert result.error is None
        assert result.stats["graph_depth_clamped_from"] == 10
        assert result.stats["graph_depth_clamped_to"] == 2

    def test_graph_depth_less_than_max_graph_depth_passes_without_clamp(self):
        """No clamp when ``graph_depth < max_graph_depth`` — stats record
        ``graph_depth_clamped_from = None``."""
        ontology = _line_ontology()
        cfg = MetadataContextConfig(
            mode="dynamic", max_graph_depth=5,
            metadata_context_dynamic_retry=0,
            max_detail_concepts=100,
        )
        result = _bf.build_filtered_metadata(
            question="any", anchor="a", ontology=ontology,
            llm=self._scripted_build_path(),
            config=cfg, graph_depth=2,
        )
        assert result.error is None
        assert result.stats.get("graph_depth_clamped_from") is None
        assert not any("graph_depth_clamped" in w for w in result.warnings)

    def test_clamp_floors_at_1_when_max_graph_depth_is_1(self):
        """Degenerate config: ``max_graph_depth=1`` leaves no room for any detail
        beyond the anchor. The clamp floors at 1 (never produces 0)."""
        ontology = _line_ontology()
        cfg = MetadataContextConfig(
            mode="dynamic", max_graph_depth=1,
            metadata_context_dynamic_retry=0,
            max_detail_concepts=100,
        )
        result = _bf.build_filtered_metadata(
            question="any", anchor="a", ontology=ontology,
            llm=self._scripted_build_path(),
            config=cfg, graph_depth=5,
        )
        assert result.error is None
        # max_graph_depth - 1 = 0, but the floor keeps it at 1.
        assert result.stats["graph_depth_clamped_to"] == 1


# ---------------------------------------------------------------------------
# Dual-source menu — orchestrator integration
# ---------------------------------------------------------------------------


def _expand_to_payload(targets):
    return json.dumps({"action": "expand_to", "expand_to": targets})


def _build_path_payload(*, segments, selected_concepts=None):
    return json.dumps({
        "action": "build_path",
        "selected_concepts": selected_concepts or [],
        "selected_paths": [{
            "path_id": "P1",
            "purpose": "",
            "segments": segments,
            "is_recursive": False,
        }],
    })


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def invoke(self, messages):
        sys_msg = messages[0].content if hasattr(messages[0], "content") else messages[0].get("content", "")
        user_msg = ""
        if len(messages) > 1:
            user_msg = messages[1].content if hasattr(messages[1], "content") else messages[1].get("content", "")
        self.calls.append({"system": sys_msg, "user": user_msg})
        if not self.responses:
            raise AssertionError(f"ScriptedLLM exhausted after {len(self.calls)} calls")
        text = self.responses.pop(0)

        class R:
            def __init__(self, c):
                self.content = c

        return R(text)


class TestDepthBandMenuRendering:
    def test_concepts_beyond_detail_depth_appear_in_menu_band(self):
        """With detail_depth=2 (graph_depth) and max_graph_depth=5 on the linear
        ontology, concepts 'd' and 'e' (hop 3, 4) MUST appear in the
        ## REACHABLE band of the rendered DDL."""
        ontology = _line_ontology()
        cfg = MetadataContextConfig(
            mode="dynamic", max_graph_depth=5,
            metadata_context_dynamic_retry=0,
            max_detail_concepts=100,
        )
        llm = ScriptedLLM([
            _build_path_payload(
                selected_concepts=["a", "b", "c"],
                segments=[
                    {"from": "a", "rel": "to_b", "to": "b"},
                    {"from": "b", "rel": "to_c", "to": "c"},
                ],
            ),
        ])
        _bf.build_filtered_metadata(
            question="any",
            anchor="a",
            ontology=ontology,
            llm=llm,
            config=cfg,
            graph_depth=2,
        )
        # The system+user prompt of the single LLM call.
        user_msg = llm.calls[0]["user"]
        # Detail-band concepts appear in the ## CONCEPTS block.
        assert "### a [anchor]" in user_msg
        assert "### b [hop=1]" in user_msg
        assert "### c [hop=2]" in user_msg
        # Beyond detail_depth=2: 'd' and 'e' MUST appear in the ## REACHABLE band.
        assert "## REACHABLE" in user_msg
        assert "d" in user_msg.split("## REACHABLE", 1)[1]
        assert "e" in user_msg.split("## REACHABLE", 1)[1]

    def test_menu_band_rendered_in_hop_then_lex_order_end_to_end(self):
        """The rendered ``## REACHABLE`` band MUST list menu concepts in
        shortest-hop-first, alphabetical-within-hop order. This is the
        deterministic-ordering contract end-to-end (menu_builder split →
        serializer render).

        Graph (anchor=A):
            A → B (hop 1), A → Z (hop 1)
            B → mid (hop 2)
            Z → far (hop 2)
            mid → deeper_alpha (hop 3), mid → deeper_beta (hop 3)
            far → deeper_gamma (hop 3)
        With detail_depth=1, the menu band is {mid, Z?, far, deeper_*}.
        Wait — Z is hop 1 so it stays in detail. We expect menu = hops 2,3:
            hop 2: far, mid           (alphabetical)
            hop 3: deeper_alpha, deeper_beta, deeper_gamma
        """
        ontology = Ontology(FakeClient({
            "A": [_row("to_b[B].x"), _row("to_z[Z].x")],
            "B": [_row("to_mid[mid].x")],
            "Z": [_row("to_far[far].x")],
            "mid": [
                _row("to_alpha[deeper_alpha].x"),
                _row("to_beta[deeper_beta].x"),
            ],
            "far": [_row("to_gamma[deeper_gamma].x")],
            "deeper_alpha": [],
            "deeper_beta": [],
            "deeper_gamma": [],
        }))
        cfg = MetadataContextConfig(
            mode="dynamic", max_graph_depth=5,
            metadata_context_dynamic_retry=0,
            max_detail_concepts=100,
        )
        llm = ScriptedLLM([
            _build_path_payload(
                selected_concepts=["A", "B"],
                segments=[{"from": "A", "rel": "to_b", "to": "B"}],
            ),
        ])
        _bf.build_filtered_metadata(
            question="any",
            anchor="A",
            ontology=ontology,
            llm=llm,
            config=cfg,
            graph_depth=1,
        )
        user_msg = llm.calls[0]["user"]
        # The ## REACHABLE band must list in (hop, lex) order: hop-2 first
        # (alphabetical), then hop-3 (alphabetical). Verify the EXACT order
        # of names in the band.
        assert "## REACHABLE" in user_msg
        reachable_line = [
            line for line in user_msg.splitlines()
            if line.startswith("## REACHABLE")
        ][0]
        # Strip the header prefix to get the comma-separated name list.
        names_part = reachable_line.split(":", 1)[1].strip()
        emitted = [n.strip() for n in names_part.split(",")]
        # hop 2 (alphabetical) → far, mid
        # hop 3 (alphabetical) → deeper_alpha, deeper_beta, deeper_gamma
        assert emitted == [
            "far", "mid",
            "deeper_alpha", "deeper_beta", "deeper_gamma",
        ], f"menu band order broken: {emitted!r}"

    def test_expand_to_promotes_depth_band_and_materializes_outbound_edges(self):
        """When the planner emits expand_to on a depth-band concept, the
        orchestrator must materialize its outbound edges (so paths through
        it are visible in the next render) AND list it in the detail
        band on the re-call."""
        ontology = _line_ontology()
        cfg = MetadataContextConfig(
            mode="dynamic", max_graph_depth=5,
            metadata_context_dynamic_retry=0,
            max_detail_concepts=100,
        )
        llm = ScriptedLLM([
            # First call: planner asks to expand 'd' (currently in menu band).
            _expand_to_payload(["d"]),
            # Second call (after promote): planner builds the path.
            _build_path_payload(
                selected_concepts=["a", "b", "c", "d"],
                segments=[
                    {"from": "a", "rel": "to_b", "to": "b"},
                    {"from": "b", "rel": "to_c", "to": "c"},
                    {"from": "c", "rel": "to_d", "to": "d"},
                ],
            ),
        ])
        result = _bf.build_filtered_metadata(
            question="any",
            anchor="a",
            ontology=ontology,
            llm=llm,
            config=cfg,
            graph_depth=2,
        )
        # Action history reflects expand_to → build_path.
        assert result.stats["action_history"] == ["expand_to", "build_path"]
        # 'd' was promoted: appears in detail band of 2nd call's user prompt.
        second_user = llm.calls[1]["user"]
        assert "### d [hop=3]" in second_user
        # 'd' is a depth_band source — promotion was tagged.
        breakdown = result.stats["expand_source_breakdown"]
        assert breakdown == [{"depth_band": 1, "prefilter_overflow": 0}]
        # The build_path resolved successfully.
        assert result.stats["resolved_by"] == "llm_paths"
        assert result.validated_paths


# ---------------------------------------------------------------------------
# Fix 1 — prefilter telemetry: 'under_threshold' stamp when guard is silent
# ---------------------------------------------------------------------------


class TestPrefilterUnderThresholdTelemetry:
    """Fix 1 regression: when the candidate set is small enough that neither
    the count nor the token gate fires, the orchestrator must stamp
    ``stats['prefilter_trigger'] = 'under_threshold'`` so integration tests
    can distinguish "didn't fire" from "not checked"."""

    def test_under_threshold_stamp_on_small_subgraph(self):
        ontology = _line_ontology()
        cfg = MetadataContextConfig(
            mode="dynamic",
            max_graph_depth=5,
            metadata_context_dynamic_retry=0,
            max_detail_concepts=100,   # well above the line-ontology size
            metadata_context_filter_max_tokens=100_000,
            metadata_context_filter_max_tokens_hard_ceiling=200_000,
        )
        llm = ScriptedLLM([
            _build_path_payload(
                selected_concepts=["a", "b"],
                segments=[{"from": "a", "rel": "to_b", "to": "b"}],
            ),
        ])
        result = _bf.build_filtered_metadata(
            question="any",
            anchor="a",
            ontology=ontology,
            llm=llm,
            config=cfg,
            graph_depth=2,
        )
        # The orchestrator did not invoke run_concept_prefilter ...
        assert result.stats["prefilter_used"] is False
        # ... and the telemetry distinguishes "didn't fire" from "not checked".
        assert result.stats["prefilter_trigger"] == "under_threshold"


# ---------------------------------------------------------------------------
# Fix 2 + Fix 3 — supply-metrics regression (one build_path, no prefilter,
# no expand_to, material in detail with rels:)
# ---------------------------------------------------------------------------


def _supply_metrics_ontology():
    """A 4-concept linear chain matching the supply-metrics regression shape:
    customer →(made_order)→ order →(includes_product)→ product →(contains)→ material.
    Designed to fit fully within hops 0..3, so with graph_depth=3 every
    concept lands in the detail band (none in the menu)."""
    return Ontology(FakeClient({
        "customer": [
            _row("entity_id", data_type="integer", pk_marker="PK"),
            _row("segment_name", data_type="varchar"),
            _row("made_order[order].entity_id"),
        ],
        "order": [
            _row("entity_id", data_type="integer", pk_marker="PK"),
            _row("order_total", data_type="decimal"),
            _row("includes_product[product].entity_id"),
        ],
        "product": [
            _row("entity_id", data_type="integer", pk_marker="PK"),
            _row("product_name", data_type="varchar"),
            _row("contains[material].entity_id"),
        ],
        "material": [
            _row("entity_id", data_type="integer", pk_marker="PK"),
            _row("material_type", data_type="varchar"),
        ],
    }))


class TestSupplyMetricsRegression:
    """Regression target for the four-fix pass. The 'which customer segment
    purchased the most metal material' question shape must resolve in:

      - ONE build_path Step-1 LLM call (plus the validation-retry budget
        if any segment fails, but here we use retry=0 so just one call),
      - NO concept_prefilter call (small subgraph, under all gates),
      - NO expand_to round (material at hop 3 is in detail band),
      - REACHABLE band EMPTY (no concepts beyond detail),
      - selected_paths carries the full chain end-to-end.
    """

    def test_supply_metrics_one_build_path_no_prefilter_no_expand(self):
        ontology = _supply_metrics_ontology()
        cfg = MetadataContextConfig(
            mode="dynamic",
            max_graph_depth=5,
            metadata_context_dynamic_retry=0,
            max_detail_concepts=100,
            metadata_context_filter_max_tokens=100_000,
            metadata_context_filter_max_tokens_hard_ceiling=200_000,
        )
        llm = ScriptedLLM([
            _build_path_payload(
                selected_concepts=["customer", "order", "product", "material"],
                segments=[
                    {"from": "customer", "rel": "made_order", "to": "order"},
                    {"from": "order", "rel": "includes_product", "to": "product"},
                    {"from": "product", "rel": "contains", "to": "material"},
                ],
            ),
        ])
        result = _bf.build_filtered_metadata(
            question="Which customer segment purchased the most metal material",
            anchor="customer",
            ontology=ontology,
            llm=llm,
            config=cfg,
            graph_depth=3,
        )
        # EXACTLY one Step-1 LLM call (no prefilter, no validation retry,
        # no expand_to round).
        assert len(llm.calls) == 1
        # Telemetry confirms no auxiliary LLM calls.
        assert result.stats["prefilter_used"] is False
        assert result.stats["prefilter_trigger"] == "under_threshold"
        assert result.stats["expand_rounds"] == 0
        assert result.stats["action_history"] == ["build_path"]
        # The single rendered DDL shows material in CONCEPTS (with its rels:
        # block) and no REACHABLE band — material at hop 3 lands in detail.
        ddl = llm.calls[0]["user"]
        assert "### customer [anchor]" in ddl
        assert "### material [hop=3]" in ddl
        material_block = ddl.split("### material [hop=3]")[1].split("###")[0]
        # Fix 2 sentinel: material is a leaf in this ontology — its block
        # MUST emit `rels: (none)` so the LLM can tell leaf from missing.
        assert "rels: (none)" in material_block
        # product's full block contains the rel pointing to material.
        product_block = ddl.split("### product [hop=2]")[1].split("###")[0]
        assert "-[contains," in product_block
        assert "-> material" in product_block
        # No REACHABLE band — everything fit in detail.
        assert "## REACHABLE" not in ddl
        # The build_path resolved with the three-segment chain.
        assert result.validated_paths
        assert len(result.validated_paths[0].segments) == 3


# ---------------------------------------------------------------------------
# Fix 4B — expand_to renders MINIMAL blocks (description + connecting rels
# only), NOT full DDL
# ---------------------------------------------------------------------------


def _supply_metrics_with_supplier_ontology():
    """Supply-metrics shape extended with material → manufactured_by →
    supplier, putting supplier at hop 4 (beyond detail_depth=2 in this
    test). Used to exercise the expand_to minimal-block contract."""
    return Ontology(FakeClient({
        "customer": [
            _row("entity_id", data_type="integer", pk_marker="PK"),
            _row("made_order[order].entity_id"),
        ],
        "order": [
            _row("entity_id", data_type="integer", pk_marker="PK"),
            _row("includes_product[product].entity_id"),
        ],
        "product": [
            _row("entity_id", data_type="integer", pk_marker="PK"),
            _row("contains[material].entity_id"),
        ],
        "material": [
            _row("entity_id", data_type="integer", pk_marker="PK"),
            _row("material_kind", data_type="varchar"),
            _row("manufactured_by[supplier].entity_id"),
        ],
        "supplier": [
            _row("entity_id", data_type="integer", pk_marker="PK"),
            _row("supplier_name", data_type="varchar"),
        ],
    }))


class TestExpandToMinimalBlock:
    """Fix 4B: when the planner emits expand_to=[X], the re-rendered DDL
    must show X as a MINIMAL block — description (when present) + the
    connecting rels: chain only. The target's full props/measures/
    incoming MUST be omitted (the LLM doesn't get free access to the
    column list via expand_to)."""

    def test_expand_to_emits_minimal_block_not_full_detail(self):
        ontology = _supply_metrics_with_supplier_ontology()
        cfg = MetadataContextConfig(
            mode="dynamic",
            max_graph_depth=5,
            metadata_context_dynamic_retry=0,
            max_detail_concepts=100,
            metadata_context_filter_max_tokens=100_000,
            metadata_context_filter_max_tokens_hard_ceiling=200_000,
        )
        # detail_depth=2: customer (0), order (1), product (2) in detail;
        # material (3), supplier (4) in the menu band.
        llm = ScriptedLLM([
            _expand_to_payload(["supplier"]),
            _build_path_payload(
                selected_concepts=[
                    "customer", "order", "product", "material", "supplier",
                ],
                segments=[
                    {"from": "customer", "rel": "made_order", "to": "order"},
                    {"from": "order", "rel": "includes_product", "to": "product"},
                ],
            ),
        ])
        _bf.build_filtered_metadata(
            question="which suppliers ship the most material",
            anchor="customer",
            ontology=ontology,
            llm=llm,
            config=cfg,
            graph_depth=2,
        )
        # The SECOND prompt (after the expand_to round) is what the planner
        # sees with supplier promoted.
        second_user = llm.calls[1]["user"]
        # supplier appears as its OWN block (not just a name in REACHABLE).
        assert "### supplier [hop=4]" in second_user
        # Minimal block contract: NO props block, NO measures line, NO
        # incoming block for supplier — those would defeat the whole point
        # of expand_to-as-connecting-rels.
        supplier_block = second_user.split("### supplier [hop=4]")[1].split("###")[0]
        assert "props:" not in supplier_block
        assert "measures:" not in supplier_block
        assert "incoming:" not in supplier_block
        # The named target column `supplier_name` MUST NOT appear (it's a prop).
        assert "supplier_name" not in supplier_block
        # material is on the path from detail frontier (product) to supplier
        # — it should ALSO appear as a minimal block (intermediate concept).
        assert "### material [hop=3]" in second_user
        material_block = second_user.split("### material [hop=3]")[1].split("###")[0]
        assert "props:" not in material_block
        assert "material_kind" not in material_block   # prop hidden
        # material's minimal block carries the connecting rel to supplier.
        assert "-[manufactured_by," in material_block
        assert "-> supplier" in material_block
        # Once supplier got its own block, it must NOT appear in REACHABLE.
        if "## REACHABLE" in second_user:
            reachable_line = [
                ln for ln in second_user.splitlines()
                if ln.startswith("## REACHABLE")
            ][0]
            assert "supplier" not in reachable_line


# ---------------------------------------------------------------------------
# Stochastic-resilience regression: 5 distinct (but all valid) LLM scripts
# for the supply-metrics question. ALL must produce the canonical concept
# chain — locks correctness across LLM variance, not byte-identical output.
# ---------------------------------------------------------------------------


def _canonical_concept_chain(validated_paths):
    """Collapse a set of validated paths into the (anchor → ... → terminus)
    concept tuple. Single-hop and full-chain segment styles, emit-order
    variations, and waypoint duplication are all normalized away.

    The tuple is built by:
      1. Collecting every (from, to) edge across all paths.
      2. Topologically walking from the anchor (in-degree-0 node) along the
         unique outbound edge at each step until a terminus (out-degree-0
         in this set) is reached.
    """
    if not validated_paths:
        return ()
    edges = set()
    for path in validated_paths:
        for seg in path.segments:
            edges.add((seg.from_concept, seg.to_concept))
    from_set = {f for f, _ in edges}
    to_set = {t for _, t in edges}
    starts = list(from_set - to_set)
    if not starts:
        return ()
    chain = [starts[0]]
    current = starts[0]
    out_by_from = {}
    for f, t in edges:
        out_by_from.setdefault(f, []).append(t)
    seen = {current}
    while current in out_by_from:
        nexts = [t for t in out_by_from[current] if t not in seen]
        if not nexts:
            break
        current = nexts[0]
        chain.append(current)
        seen.add(current)
    return tuple(chain)


_SUPPLY_METRICS_CANONICAL_CHAIN = ("customer", "order", "product", "material")


def _supply_metrics_build_path_single_hop():
    """Variant A — every segment is one hop, chained implicitly by order."""
    return [_build_path_payload(
        selected_concepts=["customer", "order", "product", "material"],
        segments=[
            {"from": "customer", "rel": "made_order", "to": "order"},
            {"from": "order", "rel": "includes_product", "to": "product"},
            {"from": "product", "rel": "contains", "to": "material"},
        ],
    )]


def _supply_metrics_build_path_full_chain():
    """Variant B — single path that walks the full chain anchor→target."""
    return [json.dumps({
        "action": "build_path",
        "selected_concepts": ["customer", "order", "product", "material"],
        "selected_paths": [{
            "path_id": "P1",
            "purpose": "customer → material",
            "segments": [
                {"from": "customer", "rel": "made_order", "to": "order"},
                {"from": "order", "rel": "includes_product", "to": "product"},
                {"from": "product", "rel": "contains", "to": "material"},
            ],
            "is_recursive": False,
        }],
    })]


def _supply_metrics_build_path_mixed_segment_styles():
    """Variant E — build_path mixes a full-chain segment with a single hop;
    the canonical chain must still resolve correctly."""
    return [json.dumps({
        "action": "build_path",
        "selected_concepts": ["customer", "order", "product", "material"],
        "selected_paths": [
            {
                "path_id": "P1",
                "purpose": "customer→product",
                "segments": [
                    {"from": "customer", "rel": "made_order", "to": "order"},
                    {"from": "order", "rel": "includes_product", "to": "product"},
                ],
                "is_recursive": False,
            },
            {
                "path_id": "P2",
                "purpose": "product→material",
                "segments": [
                    {"from": "product", "rel": "contains", "to": "material"},
                ],
                "is_recursive": False,
            },
        ],
    })]


@pytest.mark.parametrize(
    "script_factory, label",
    [
        (_supply_metrics_build_path_single_hop, "single_hop"),
        (_supply_metrics_build_path_full_chain, "full_chain"),
        (
            _supply_metrics_build_path_mixed_segment_styles,
            "mixed_segment_styles",
        ),
    ],
)
class TestSupplyMetricsStochasticRegression:
    """Correct-under-variance acceptance for the supply-metrics question.

    Per FROZEN SPEC: "the planner is non-deterministic (no temperature) — do
    NOT require identical output across runs ... ALL must satisfy: exactly
    ONE build_path (NO expand_to, NO reanchor), NO standalone prefilter
    call, NO shortest-path branch, selected_paths semantically = canonical
    chain."

    The 3 scripts encode legitimate LLM variance — single-hop vs full-chain
    segment style, mixed-style emission. ALL must converge to the canonical
    concept chain.

    Note: ``expand_to`` is impossible under this regression target because
    the supply-metrics ontology fits entirely in the detail band (menu is
    empty), so Fix 2a's structural grammar narrowing drops expand_to from
    the allowed-actions list. The LLM cannot emit it. The variants here are
    therefore the only legitimate stochastic shapes for this query.
    Invalid expand_to recovery is exercised separately by
    ``TestExpandToValidationAndRetry`` in test_action_grammar.py, against a
    menu-capable ontology where the grammar permits the action.
    """

    def test_canonical_chain_under_variance(self, script_factory, label):
        ontology = _supply_metrics_ontology()
        cfg = MetadataContextConfig(
            mode="dynamic",
            max_graph_depth=5,
            metadata_context_dynamic_retry=0,
            max_detail_concepts=100,
            metadata_context_filter_max_tokens=100_000,
            metadata_context_filter_max_tokens_hard_ceiling=200_000,
        )
        llm = ScriptedLLM(script_factory())
        result = _bf.build_filtered_metadata(
            question=(
                "Which customer segment purchased the most metal material"
            ),
            anchor="customer",
            ontology=ontology,
            llm=llm,
            config=cfg,
            graph_depth=3,
        )
        # Canonical concept chain — order of segment-emission, single-hop vs
        # full-chain style, etc. all normalize to the same tuple.
        assert _canonical_concept_chain(result.validated_paths) == (
            _SUPPLY_METRICS_CANONICAL_CHAIN
        ), f"variant {label!r} did not resolve to the canonical chain"
        # Acceptance assertions: exactly ONE build_path, no expand_to, no
        # reanchor, no standalone prefilter, no BFS branch.
        assert result.stats["action_history"] == ["build_path"]
        assert "expand_to" not in result.stats["action_history"]
        assert result.stats["reanchor_rounds"] == 0
        assert result.stats["prefilter_used"] is False
        assert result.stats["prefilter_trigger"] == "under_threshold"
        assert result.stats["resolved_by"] == "llm_paths"
        assert "bfs" not in (result.stats["resolved_by"] or "")
        # Only ONE LLM call — no auxiliary planner / prefilter rounds.
        assert len(llm.calls) == 1
        # Grammar on the single call had no expand_to (menu was empty).
        assert "expand_to" not in result.stats["allowed_actions_history"][0]
