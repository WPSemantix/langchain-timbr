"""Unit tests for TechnicalContextConfig."""

import pytest
from langchain_timbr.technical_context.config import TechnicalContextConfig


class TestTechnicalContextConfigDefaults:
    """Test default configuration values."""

    def test_defaults(self):
        cfg = TechnicalContextConfig()
        assert cfg.enable_technical_context is True
        assert cfg.mode == "auto"
        assert cfg.max_tokens == 3000
        assert cfg.safety_ceiling == 10000
        assert cfg.max_values_per_column == 20
        assert cfg.show_all_under == 50

        assert cfg.free_text_distinct_threshold == 10000
        assert cfg.id_unique_ratio_threshold == 0.95
        assert cfg.fuzzy_threshold_default == 88
        assert cfg.fuzzy_threshold_strict == 95
        assert cfg.trim_sequence == (200, 100, 50, 20, 10, 5)

    def test_custom_values(self):
        cfg = TechnicalContextConfig(max_tokens=1000, mode="filter_matched", fuzzy_threshold_default=90)
        assert cfg.max_tokens == 1000
        assert cfg.mode == "filter_matched"
        assert cfg.fuzzy_threshold_default == 90


class TestTechnicalContextConfigValidation:
    """Test validation in __post_init__."""

    def test_max_tokens_zero(self):
        with pytest.raises(ValueError, match="max_tokens must be > 0"):
            TechnicalContextConfig(max_tokens=0)

    def test_max_tokens_negative(self):
        with pytest.raises(ValueError, match="max_tokens must be > 0"):
            TechnicalContextConfig(max_tokens=-1)

    def test_safety_ceiling_zero(self):
        with pytest.raises(ValueError, match="safety_ceiling must be > 0"):
            TechnicalContextConfig(safety_ceiling=0)

    def test_max_tokens_exceeds_safety_ceiling(self):
        with pytest.raises(ValueError, match="max_tokens must be < safety_ceiling"):
            TechnicalContextConfig(max_tokens=5000, safety_ceiling=5000)

    def test_max_values_per_column_zero(self):
        with pytest.raises(ValueError, match="max_values_per_column must be > 0"):
            TechnicalContextConfig(max_values_per_column=0)

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode must be"):
            TechnicalContextConfig(mode="bad_mode")

    def test_invalid_id_unique_ratio(self):
        with pytest.raises(ValueError, match="id_unique_ratio_threshold"):
            TechnicalContextConfig(id_unique_ratio_threshold=0)

    def test_invalid_fuzzy_threshold(self):
        with pytest.raises(ValueError, match="fuzzy_threshold_default"):
            TechnicalContextConfig(fuzzy_threshold_default=0)

    def test_empty_trim_sequence(self):
        with pytest.raises(ValueError, match="trim_sequence must not be empty"):
            TechnicalContextConfig(trim_sequence=())
