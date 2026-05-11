"""Tests for mapping_resolver module."""

import pytest

from langchain_timbr.technical_context.statistics_loader.mapping_resolver import (
    resolve_concept_mappings,
)
from langchain_timbr.technical_context.statistics_loader.types import (
    OntologyConceptRow,
    ConceptMappingRow,
    ConceptMappingSet,
)
from langchain_timbr.technical_context.statistics_loader.config import StatisticsLoaderConfig
from langchain_timbr.technical_context.statistics_loader.inheritance import build_descendants_map


class TestResolveMappings:
    """Test mapping resolution logic."""

    def _resolve(self, concept, ontology, mappings, config=None):
        """Helper to resolve with computed descendants."""
        if config is None:
            config = StatisticsLoaderConfig()
        descendants = build_descendants_map(ontology)
        return resolve_concept_mappings(
            concept=concept,
            ontology=ontology,
            mappings_by_concept=mappings,
            descendants=descendants,
            config=config,
        )

    def test_direct_only(self, sample_ontology, sample_mappings, default_config):
        """Concept with only direct mappings."""
        descendants = build_descendants_map(sample_ontology)
        result = resolve_concept_mappings(
            concept="order",
            ontology=sample_ontology,
            mappings_by_concept=sample_mappings,
            descendants=descendants,
            config=default_config,
        )
        assert result.concept == "order"
        assert len(result.mappings) == 1
        assert result.mappings[0].mapping_name == "map_order"
        assert result.mappings[0].via == "direct"
        assert result.total_rows == 5000

    def test_derived_via_descendants(self, default_config):
        """Concept picks up mappings from descendants."""
        ontology = {
            "animal": OntologyConceptRow(concept="animal", inheritance="", query=None),
            "dog": OntologyConceptRow(concept="dog", inheritance="animal", query=None),
            "cat": OntologyConceptRow(concept="cat", inheritance="animal", query=None),
        }
        mappings = {
            "dog": [ConceptMappingRow(concept="dog", mapping_name="map_dog", number_of_rows=100)],
            "cat": [ConceptMappingRow(concept="cat", mapping_name="map_cat", number_of_rows=200)],
        }
        descendants = build_descendants_map(ontology)
        result = resolve_concept_mappings(
            concept="animal",
            ontology=ontology,
            mappings_by_concept=mappings,
            descendants=descendants,
            config=default_config,
        )
        mapping_names = {m.mapping_name for m in result.mappings}
        assert "map_dog" in mapping_names
        assert "map_cat" in mapping_names
        assert all(m.via == "derived" for m in result.mappings)
        assert result.total_rows == 300

    def test_logic_recursive(self, default_config):
        """Logic-concept resolution follows query reference."""
        ontology = {
            "base": OntologyConceptRow(concept="base", inheritance="", query=None),
            "logic_a": OntologyConceptRow(
                concept="logic_a", inheritance="",
                query="SELECT * FROM dtimbr.base WHERE x = 1"
            ),
        }
        mappings = {
            "base": [ConceptMappingRow(concept="base", mapping_name="map_base", number_of_rows=1000)],
        }
        descendants = build_descendants_map(ontology)
        result = resolve_concept_mappings(
            concept="logic_a",
            ontology=ontology,
            mappings_by_concept=mappings,
            descendants=descendants,
            config=default_config,
        )
        assert len(result.mappings) == 1
        assert result.mappings[0].mapping_name == "map_base"
        assert result.mappings[0].via == "logic"

    def test_all_three_combined(self, default_config):
        """Direct + derived + logic mappings combined with dedup."""
        ontology = {
            "parent": OntologyConceptRow(concept="parent", inheritance="", query=None),
            "child": OntologyConceptRow(concept="child", inheritance="parent", query=None),
            "concept_x": OntologyConceptRow(
                concept="concept_x", inheritance="",
                query="SELECT * FROM dtimbr.parent WHERE y = 1"
            ),
        }
        mappings = {
            "concept_x": [ConceptMappingRow(concept="concept_x", mapping_name="map_direct", number_of_rows=100)],
            "parent": [ConceptMappingRow(concept="parent", mapping_name="map_parent", number_of_rows=500)],
            "child": [ConceptMappingRow(concept="child", mapping_name="map_child", number_of_rows=200)],
        }
        descendants = build_descendants_map(ontology)
        result = resolve_concept_mappings(
            concept="concept_x",
            ontology=ontology,
            mappings_by_concept=mappings,
            descendants=descendants,
            config=default_config,
        )
        mapping_names = {m.mapping_name for m in result.mappings}
        # map_direct is direct, map_parent and map_child come from logic→parent (which has child descendant)
        assert "map_direct" in mapping_names
        assert "map_parent" in mapping_names
        assert "map_child" in mapping_names

    def test_dedup_priority_direct_over_logic(self, default_config):
        """Direct mapping wins over logic when same mapping_name appears."""
        ontology = {
            "base": OntologyConceptRow(concept="base", inheritance="", query=None),
            "logic_x": OntologyConceptRow(
                concept="logic_x", inheritance="",
                query="SELECT * FROM dtimbr.base WHERE x = 1"
            ),
        }
        # Same mapping_name in both
        mappings = {
            "logic_x": [ConceptMappingRow(concept="logic_x", mapping_name="shared_map", number_of_rows=100)],
            "base": [ConceptMappingRow(concept="base", mapping_name="shared_map", number_of_rows=500)],
        }
        descendants = build_descendants_map(ontology)
        result = resolve_concept_mappings(
            concept="logic_x",
            ontology=ontology,
            mappings_by_concept=mappings,
            descendants=descendants,
            config=default_config,
        )
        # Direct should win
        assert len(result.mappings) == 1
        assert result.mappings[0].via == "direct"
        assert result.mappings[0].number_of_rows == 100

    def test_cycle_protection(self, default_config):
        """Cycles in logic queries don't cause infinite recursion."""
        ontology = {
            "A": OntologyConceptRow(concept="A", inheritance="", query="SELECT * FROM dtimbr.B WHERE x = 1"),
            "B": OntologyConceptRow(concept="B", inheritance="", query="SELECT * FROM dtimbr.A WHERE y = 1"),
        }
        mappings = {
            "A": [ConceptMappingRow(concept="A", mapping_name="map_a", number_of_rows=100)],
            "B": [ConceptMappingRow(concept="B", mapping_name="map_b", number_of_rows=200)],
        }
        descendants = build_descendants_map(ontology)
        # Should not raise, should terminate
        result = resolve_concept_mappings(
            concept="A",
            ontology=ontology,
            mappings_by_concept=mappings,
            descendants=descendants,
            config=default_config,
        )
        assert result.concept == "A"
        # A gets map_a (direct) + map_b (logic from B), but B→A is a cycle (stopped)
        mapping_names = {m.mapping_name for m in result.mappings}
        assert "map_a" in mapping_names
        assert "map_b" in mapping_names

    def test_depth_cap(self):
        """Depth cap prevents deep logic chains."""
        config = StatisticsLoaderConfig(logic_query_max_depth=2)
        # Chain: A→logic→B→logic→C→logic→D (depth 3, should stop)
        ontology = {
            "A": OntologyConceptRow(concept="A", inheritance="", query="SELECT * FROM dtimbr.B WHERE x = 1"),
            "B": OntologyConceptRow(concept="B", inheritance="", query="SELECT * FROM dtimbr.C WHERE x = 1"),
            "C": OntologyConceptRow(concept="C", inheritance="", query="SELECT * FROM dtimbr.D WHERE x = 1"),
            "D": OntologyConceptRow(concept="D", inheritance="", query=None),
        }
        mappings = {
            "D": [ConceptMappingRow(concept="D", mapping_name="map_d", number_of_rows=100)],
        }
        descendants = build_descendants_map(ontology)
        result = resolve_concept_mappings(
            concept="A",
            ontology=ontology,
            mappings_by_concept=mappings,
            descendants=descendants,
            config=config,
        )
        # D's mapping should NOT be reachable at depth cap 2 (A→B is depth 1, B→C is depth 2, C→D exceeds)
        mapping_names = {m.mapping_name for m in result.mappings}
        assert "map_d" not in mapping_names

    def test_concept_not_in_ontology(self, default_config):
        """Concept not in ontology returns empty mappings."""
        ontology = {
            "other": OntologyConceptRow(concept="other", inheritance="", query=None),
        }
        mappings = {}
        descendants = build_descendants_map(ontology)
        result = resolve_concept_mappings(
            concept="nonexistent",
            ontology=ontology,
            mappings_by_concept=mappings,
            descendants=descendants,
            config=default_config,
        )
        assert result.mappings == []
        assert result.total_rows == 0

    def test_no_mappings(self, default_config):
        """Concept exists in ontology but has no mappings anywhere."""
        ontology = {
            "empty": OntologyConceptRow(concept="empty", inheritance="", query=None),
        }
        mappings = {}
        descendants = build_descendants_map(ontology)
        result = resolve_concept_mappings(
            concept="empty",
            ontology=ontology,
            mappings_by_concept=mappings,
            descendants=descendants,
            config=default_config,
        )
        assert result.mappings == []
        assert result.total_rows == 0
