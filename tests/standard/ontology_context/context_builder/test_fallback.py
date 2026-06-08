"""Tests for fallback.py — all-simple-paths enumeration with loose interior.

Covers:
  - Basic reachability (single chain).
  - Loose-interior: bridge concepts NOT in selected_concepts are allowed on
    the path's interior, so customer→order→product works even when only
    customer and product are in selected_concepts.
  - Multiple distinct relationships between same (anchor, target) pair both
    surface (no shortest-only-one-pick truncation).
  - length_cap stops path growth at the configured hop limit.
  - Dedup on full segment identity, not just destination.
  - Safety cap (circuit breaker) trips on pathological enumeration and aborts.
  - No node revisits within a single path (cycle prevention).
"""

from __future__ import annotations

from langchain_timbr.ontology_context.context_builder.edge_index import EdgeIndex
from langchain_timbr.ontology_context.context_builder.fallback import generate_fallback_paths
from langchain_timbr.ontology_context.ontology.graph import Ontology


def _row(col_name, **extras):
    base = {"col_name": col_name, "data_type": "varchar", "comment": "", "inheritance_marker": "", "pk_marker": ""}
    base.update(extras)
    return base


class FakeClient:
    def __init__(self, concepts, relationships=None, version="v1"):
        self._concepts = concepts
        self._relationships = relationships or []
        self._version = version

    def fetch_version_id(self):
        return self._version

    def describe_concept(self, name):
        return list(self._concepts.get(name, []))

    def fetch_relationships_meta(self):
        return list(self._relationships)

    def fetch_inheritance_meta(self):
        return []


def _segments(path):
    return [(s.from_concept, s.relationship_name, s.to_concept) for s in path.segments]


class TestGenerateFallbackPaths:
    def _simple(self):
        # customer -[made_order]-> order -[contains_product]-> product
        client = FakeClient({
            "customer": [_row("made_order[order].o_id")],
            "order": [_row("contains_product[product].p_id")],
            "product": [],
        })
        return EdgeIndex(Ontology(client))

    def test_reaches_target_through_chain(self):
        idx = self._simple()
        paths = generate_fallback_paths(
            "customer", ["product"], idx, length_cap=3,
        )
        assert paths
        seg = _segments(paths[0])
        assert seg[0][0] == "customer"
        assert seg[-1][2] == "product"

    def test_loose_interior_bridge_concept_allowed(self):
        """When selected_concepts contains only the endpoints (customer, product)
        but the path requires passing through `order` as a bridge, the enumeration
        MUST find customer→order→product. This is the recall-killer scenario
        strict-interior mode would silently fail on."""
        idx = self._simple()
        paths = generate_fallback_paths(
            "customer", ["product"], idx, length_cap=3,
        )
        assert paths
        # Confirm the bridge concept appears in the path interior.
        intermediate_concepts = {s[2] for s in _segments(paths[0])[:-1]}
        assert "order" in intermediate_concepts

    def test_no_path_returns_empty(self):
        idx = self._simple()
        paths = generate_fallback_paths(
            "customer", ["unreachable_concept"], idx, length_cap=3,
        )
        assert paths == []

    def test_anchor_skipped_in_selected(self):
        idx = self._simple()
        paths = generate_fallback_paths(
            "customer", ["customer", "order"], idx, length_cap=3,
        )
        # Only target "order" produces a path; anchor is filtered out.
        assert len(paths) == 1
        assert _segments(paths[0])[-1][2] == "order"

    def test_multiple_relationships_same_target_both_kept(self):
        """Two distinct relationships customer→order MUST both surface as
        separate paths (not collapsed by destination-only dedup)."""
        client = FakeClient({
            "customer": [
                _row("made_order[order].o_id"),
                _row("cancelled_order[order].o_id"),
            ],
            "order": [],
        })
        idx = EdgeIndex(Ontology(client))
        paths = generate_fallback_paths(
            "customer", ["order"], idx, length_cap=2,
        )
        rel_names = {_segments(p)[0][1] for p in paths}
        assert rel_names == {"made_order", "cancelled_order"}, (
            f"Expected both relationships to surface; got {rel_names}"
        )

    def test_length_cap_bounds_enumeration(self):
        # A → B → C → D — length_cap=2 should only reach B, not C or D.
        client = FakeClient({
            "A": [_row("to_b[B].b_id")],
            "B": [_row("to_c[C].c_id")],
            "C": [_row("to_d[D].d_id")],
            "D": [],
        })
        idx = EdgeIndex(Ontology(client))
        paths = generate_fallback_paths("A", ["B", "C", "D"], idx, length_cap=2)
        # length_cap=2 means up to 2 segments — B and C reachable, D NOT.
        targets = {_segments(p)[-1][2] for p in paths}
        assert "B" in targets
        assert "C" in targets
        assert "D" not in targets

    def test_no_node_revisits_within_single_path(self):
        """Cycle prevention: even with a self-ref + non-self-ref out-edge,
        a single path doesn't revisit the same node."""
        client = FakeClient({
            "A": [
                _row("self_ref[A].a_id"),    # self-ref
                _row("to_b[B].b_id"),
            ],
            "B": [],
        })
        idx = EdgeIndex(Ontology(client))
        paths = generate_fallback_paths("A", ["B"], idx, length_cap=5)
        # A path A→A→B would violate no-revisit. Only A→B should survive.
        for p in paths:
            visited = ["A"] + [s[2] for s in _segments(p)]
            assert len(visited) == len(set(visited)), (
                f"Path revisits a node: {_segments(p)}"
            )

    def test_dedup_on_full_segment_identity(self):
        """Even with multiple DFS branches that could reach the same path,
        dedup by ordered segment tuple keeps the output set canonical."""
        idx = self._simple()
        paths = generate_fallback_paths(
            "customer", ["order", "product"], idx, length_cap=3,
        )
        keys = [tuple(_segments(p)) for p in paths]
        assert len(keys) == len(set(keys))

    def test_safety_cap_trips_on_pathological_enumeration(self, caplog):
        """Build a small but high-fanout graph and request a tiny safety_cap
        to force the circuit breaker. The result must be bounded, and an
        ERROR log must fire."""
        import logging as _logging
        # 5 targets all reachable in 1 hop AND in 2 hops (via each other).
        # With safety_cap=3, enumeration aborts after the first 3 paths.
        concepts = {f"T{i}": [
            _row(f"to_t{j}[T{j}].x") for j in range(5) if j != i
        ] for i in range(5)}
        concepts["A"] = [_row(f"to_t{i}[T{i}].x") for i in range(5)]
        client = FakeClient(concepts)
        idx = EdgeIndex(Ontology(client))
        targets = [f"T{i}" for i in range(5)]
        with caplog.at_level(_logging.ERROR):
            paths = generate_fallback_paths(
                "A", targets, idx, length_cap=4, safety_cap=3,
            )
        assert len(paths) <= 3
        # ERROR log emitted for the safety cap trip.
        assert any("safety cap" in r.message.lower() for r in caplog.records)

    def test_determinism(self):
        idx = self._simple()
        a = generate_fallback_paths("customer", ["product", "order"], idx, length_cap=3)
        b = generate_fallback_paths("customer", ["product", "order"], idx, length_cap=3)
        assert [_segments(p) for p in a] == [_segments(p) for p in b]
