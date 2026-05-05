"""Tests for inheritance module."""

import pytest

from langchain_timbr.technical_context.statistics_loader.inheritance import build_descendants_map
from langchain_timbr.technical_context.statistics_loader.types import OntologyConceptRow


class TestBuildDescendantsMap:
    """Test building the descendants map."""

    def test_single_parent(self):
        """Single inheritance chain: thing -> person -> customer."""
        concepts = {
            "thing": OntologyConceptRow(concept="thing", inheritance="", query=None),
            "person": OntologyConceptRow(concept="person", inheritance="thing", query=None),
            "customer": OntologyConceptRow(concept="customer", inheritance="person", query=None),
        }
        desc = build_descendants_map(concepts)
        assert desc["thing"] == {"person", "customer"}
        assert desc["person"] == {"customer"}
        assert desc["customer"] == set()

    def test_multi_parent(self):
        """Multi-parent inheritance via comma-separated field."""
        concepts = {
            "parent_a": OntologyConceptRow(concept="parent_a", inheritance="", query=None),
            "parent_b": OntologyConceptRow(concept="parent_b", inheritance="", query=None),
            "child": OntologyConceptRow(concept="child", inheritance="parent_a, parent_b", query=None),
        }
        desc = build_descendants_map(concepts)
        assert "child" in desc["parent_a"]
        assert "child" in desc["parent_b"]

    def test_deep_chain(self):
        """Deep inheritance chain — all ancestors include all descendants."""
        concepts = {}
        for i in range(5):
            parent = f"level_{i - 1}" if i > 0 else ""
            concepts[f"level_{i}"] = OntologyConceptRow(
                concept=f"level_{i}", inheritance=parent, query=None
            )
        desc = build_descendants_map(concepts)
        # level_0 should have level_1, level_2, level_3, level_4 as descendants
        assert desc["level_0"] == {"level_1", "level_2", "level_3", "level_4"}
        assert desc["level_3"] == {"level_4"}

    def test_depth_cap(self):
        """Depth cap truncates traversal with warning."""
        concepts = {}
        for i in range(20):
            parent = f"c_{i - 1}" if i > 0 else ""
            concepts[f"c_{i}"] = OntologyConceptRow(
                concept=f"c_{i}", inheritance=parent, query=None
            )
        desc = build_descendants_map(concepts, max_depth=5)
        # c_0's descendants should be truncated at depth 5
        assert "c_5" in desc["c_0"]
        assert "c_6" not in desc["c_0"]

    def test_no_inheritance(self):
        """Concept with no inheritance has no descendants."""
        concepts = {
            "standalone": OntologyConceptRow(concept="standalone", inheritance="", query=None),
        }
        desc = build_descendants_map(concepts)
        assert desc["standalone"] == set()

    def test_empty_concepts(self):
        """Empty concepts dict returns empty descendants."""
        desc = build_descendants_map({})
        assert desc == {}

    def test_diamond_inheritance(self):
        """Diamond pattern: D inherits from B and C, which both inherit from A."""
        concepts = {
            "A": OntologyConceptRow(concept="A", inheritance="", query=None),
            "B": OntologyConceptRow(concept="B", inheritance="A", query=None),
            "C": OntologyConceptRow(concept="C", inheritance="A", query=None),
            "D": OntologyConceptRow(concept="D", inheritance="B, C", query=None),
        }
        desc = build_descendants_map(concepts)
        assert desc["A"] == {"B", "C", "D"}
        assert desc["B"] == {"D"}
        assert desc["C"] == {"D"}
        assert desc["D"] == set()

    def test_whitespace_in_inheritance(self):
        """Whitespace around comma-separated parents is stripped."""
        concepts = {
            "A": OntologyConceptRow(concept="A", inheritance="", query=None),
            "B": OntologyConceptRow(concept="B", inheritance="", query=None),
            "C": OntologyConceptRow(concept="C", inheritance=" A , B ", query=None),
        }
        desc = build_descendants_map(concepts)
        assert "C" in desc["A"]
        assert "C" in desc["B"]

    def test_cycle_in_inheritance(self):
        """Cycles in inheritance graph don't cause infinite loops."""
        concepts = {
            "A": OntologyConceptRow(concept="A", inheritance="B", query=None),
            "B": OntologyConceptRow(concept="B", inheritance="A", query=None),
        }
        # Should terminate without error
        desc = build_descendants_map(concepts)
        # A's children include B, and B's children include A
        assert "B" in desc["A"]
        assert "A" in desc["B"]
