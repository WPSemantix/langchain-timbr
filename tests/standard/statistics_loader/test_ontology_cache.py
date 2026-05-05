"""Tests for ontology_cache module."""

import pytest
from unittest.mock import patch, MagicMock

from langchain_timbr.technical_context.statistics_loader.ontology_cache import (
    load_ontology_concepts,
    load_concept_mappings,
    load_view_row_counts,
)
from langchain_timbr.technical_context.statistics_loader.types import (
    OntologyConceptRow,
    ConceptMappingRow,
)


class TestLoadOntologyConcepts:
    """Test ontology concepts loading."""

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_basic_load(self, mock_run_query, conn_params):
        """Loads concepts from sys_ontology."""
        mock_run_query.return_value = [
            {"concept": "customer", "inheritance": "person", "query": None},
            {"concept": "person", "inheritance": "thing", "query": None},
            {"concept": "logic_concept", "inheritance": "", "query": "SELECT * FROM dtimbr.customer WHERE active = 1"},
        ]

        # Need to bypass cache for testing
        result = load_ontology_concepts.__wrapped__(conn_params)

        assert "customer" in result
        assert result["customer"].inheritance == "person"
        assert result["customer"].query is None
        assert result["logic_concept"].query == "SELECT * FROM dtimbr.customer WHERE active = 1"

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_missing_concept_field(self, mock_run_query, conn_params):
        """Rows with missing concept field are skipped."""
        mock_run_query.return_value = [
            {"concept": "valid", "inheritance": "", "query": None},
            {"concept": None, "inheritance": "", "query": None},
            {"inheritance": "thing", "query": None},  # no concept key
        ]

        result = load_ontology_concepts.__wrapped__(conn_params)
        assert len(result) == 1
        assert "valid" in result

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_empty_result(self, mock_run_query, conn_params):
        """Empty query result returns empty dict."""
        mock_run_query.return_value = []
        result = load_ontology_concepts.__wrapped__(conn_params)
        assert result == {}


class TestLoadConceptMappings:
    """Test concept mappings loading."""

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_basic_load(self, mock_run_query, conn_params):
        """Loads mappings grouped by concept."""
        mock_run_query.return_value = [
            {"concept": "customer", "mapping_name": "map_a", "number_of_rows": 1000},
            {"concept": "customer", "mapping_name": "map_b", "number_of_rows": 500},
            {"concept": "order", "mapping_name": "map_c", "number_of_rows": 5000},
        ]

        result = load_concept_mappings.__wrapped__(conn_params)

        assert len(result["customer"]) == 2
        assert result["customer"][0].mapping_name == "map_a"
        assert result["customer"][0].number_of_rows == 1000
        assert len(result["order"]) == 1

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_null_number_of_rows(self, mock_run_query, conn_params):
        """NULL number_of_rows defaults to -1."""
        mock_run_query.return_value = [
            {"concept": "customer", "mapping_name": "map_a", "number_of_rows": None},
        ]

        result = load_concept_mappings.__wrapped__(conn_params)
        assert result["customer"][0].number_of_rows == -1

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_empty_result(self, mock_run_query, conn_params):
        """Empty query result returns empty dict."""
        mock_run_query.return_value = []
        result = load_concept_mappings.__wrapped__(conn_params)
        assert result == {}


class TestLoadViewRowCounts:
    """Test view row counts loading."""

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_basic_load(self, mock_run_query, conn_params):
        """Loads view row counts."""
        mock_run_query.return_value = [
            {"view_name": "v_customers", "number_of_rows": 1000},
            {"view_name": "v_orders", "number_of_rows": 5000},
        ]

        result = load_view_row_counts.__wrapped__(conn_params)

        assert result["v_customers"] == 1000
        assert result["v_orders"] == 5000

    @patch("langchain_timbr.utils.timbr_utils.run_query")
    def test_null_rows(self, mock_run_query, conn_params):
        """NULL number_of_rows defaults to -1."""
        mock_run_query.return_value = [
            {"view_name": "v_test", "number_of_rows": None},
        ]

        result = load_view_row_counts.__wrapped__(conn_params)
        assert result["v_test"] == -1
