"""Shared fixtures for statistics_loader unit tests."""

import pytest
from unittest.mock import patch, MagicMock

from langchain_timbr.technical_context.statistics_loader.config import StatisticsLoaderConfig
from langchain_timbr.technical_context.statistics_loader.types import (
    OntologyConceptRow,
    ConceptMappingRow,
    ConceptMappingSet,
    MappingRef,
    RawStatsRow,
    TopKEntry,
    ColumnStatistics,
)


@pytest.fixture
def default_config():
    """Default StatisticsLoaderConfig."""
    return StatisticsLoaderConfig()


@pytest.fixture
def strict_config():
    """Config that raises on missing stats and logic parse failures."""
    return StatisticsLoaderConfig(
        on_missing_stats="raise",
        on_logic_parse_failure="raise",
    )


@pytest.fixture
def small_chunk_config():
    """Config with small chunk size for testing chunking behavior."""
    return StatisticsLoaderConfig(in_clause_chunk_size=3)


@pytest.fixture
def conn_params():
    """Minimal connection parameters for tests."""
    return {
        "url": "http://localhost:11000",
        "token": "test_token",
        "ontology": "test_ontology",
    }


@pytest.fixture
def sample_ontology():
    """Sample ontology concepts for testing."""
    return {
        "customer": OntologyConceptRow(
            concept="customer", inheritance="person", query=None
        ),
        "person": OntologyConceptRow(
            concept="person", inheritance="thing", query=None
        ),
        "thing": OntologyConceptRow(
            concept="thing", inheritance="", query=None
        ),
        "order": OntologyConceptRow(
            concept="order", inheritance="thing", query=None
        ),
        "order_item": OntologyConceptRow(
            concept="order_item", inheritance="thing", query=None
        ),
        "active_customer": OntologyConceptRow(
            concept="active_customer",
            inheritance="customer",
            query="SELECT * FROM dtimbr.customer WHERE status = 'active'",
        ),
        "vip_customer": OntologyConceptRow(
            concept="vip_customer",
            inheritance="customer",
            query="SELECT * FROM dtimbr.active_customer WHERE tier = 'vip'",
        ),
    }


@pytest.fixture
def sample_mappings():
    """Sample concept mappings for testing."""
    return {
        "customer": [
            ConceptMappingRow(concept="customer", mapping_name="map_customer_main", number_of_rows=1000),
            ConceptMappingRow(concept="customer", mapping_name="map_customer_archive", number_of_rows=500),
        ],
        "person": [
            ConceptMappingRow(concept="person", mapping_name="map_person", number_of_rows=2000),
        ],
        "order": [
            ConceptMappingRow(concept="order", mapping_name="map_order", number_of_rows=5000),
        ],
        "order_item": [
            ConceptMappingRow(concept="order_item", mapping_name="map_order_item", number_of_rows=20000),
        ],
    }
