"""Shared fixtures for statistics_loader integration tests."""

import pytest
from unittest.mock import patch

from langchain_timbr.technical_context.statistics_loader.config import StatisticsLoaderConfig
from langchain_timbr.utils.timbr_utils import clear_cache


@pytest.fixture(autouse=True)
def _clear_timbr_cache():
    """Clear timbr cache before each test to avoid cross-test contamination."""
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def conn_params():
    """Minimal connection parameters for integration tests."""
    return {
        "url": "http://localhost:11000",
        "token": "test_token",
        "ontology": "test_ontology",
    }


@pytest.fixture
def default_config():
    return StatisticsLoaderConfig()
