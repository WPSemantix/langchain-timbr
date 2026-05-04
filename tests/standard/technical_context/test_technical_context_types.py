"""Unit tests for types module."""

import pytest
from langchain_timbr.technical_context.types import (
    ColumnRef, MatchResult, SemanticType, TechnicalContextResult,
)


class TestSemanticType:
    """Test SemanticType enum."""

    def test_all_values(self):
        assert SemanticType.BOOLEAN.value == "boolean"
        assert SemanticType.ID.value == "id"
        assert SemanticType.NUMERIC.value == "numeric"
        assert SemanticType.DATE.value == "date"
        assert SemanticType.FREE_TEXT.value == "free_text"
        assert SemanticType.CODE_LIKE.value == "code_like"
        assert SemanticType.BUSINESS_KEY_LIKE.value == "business_key_like"
        assert SemanticType.CATEGORICAL_TEXT.value == "categorical_text"


class TestColumnRef:
    """Test ColumnRef dataclass."""

    def test_creation(self):
        ref = ColumnRef(name="status", sql_type="varchar(50)", ontology_distance=0, priority_band=2)
        assert ref.name == "status"
        assert ref.sql_type == "varchar(50)"
        assert ref.ontology_distance == 0
        assert ref.priority_band == 2
        assert ref.semantic_type is None

    def test_with_semantic_type(self):
        ref = ColumnRef(name="x", sql_type="int", ontology_distance=1, priority_band=4, semantic_type=SemanticType.NUMERIC)
        assert ref.semantic_type == SemanticType.NUMERIC


class TestMatchResult:
    """Test MatchResult dataclass."""

    def test_creation(self):
        m = MatchResult(column_name="country", matched_value="USA", score=100, match_type="exact", candidate="usa")
        assert m.column_name == "country"
        assert m.matched_value == "USA"
        assert m.score == 100
        assert m.match_type == "exact"
        assert m.candidate == "usa"


class TestTechnicalContextResult:
    """Test TechnicalContextResult dataclass."""

    def test_empty(self):
        r = TechnicalContextResult(column_annotations={})
        assert r.is_empty is True

    def test_not_empty(self):
        r = TechnicalContextResult(column_annotations={"col": "annotation"})
        assert r.is_empty is False

    def test_metadata_default(self):
        r = TechnicalContextResult(column_annotations={})
        assert r.metadata == {}
