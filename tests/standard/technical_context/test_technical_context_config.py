"""Unit tests for TechnicalContextConfig."""

import logging

import pytest
from langchain_timbr.technical_context.config import TechnicalContextConfig


class TestTechnicalContextConfigDefaults:
    """Test default configuration values."""

    def test_defaults(self):
        cfg = TechnicalContextConfig()
        assert cfg.mode == "auto"
        assert cfg.max_tokens == 3000
        assert cfg.safety_ceiling == 20000
        assert cfg.max_values_per_column == 20
        assert cfg.show_all_under == 50

        assert cfg.free_text_distinct_threshold == 10000
        assert cfg.id_unique_ratio_threshold == 0.95
        assert cfg.fuzzy_threshold_default == 88
        assert cfg.fuzzy_threshold_strict == 95

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

    def test_max_tokens_at_the_ceiling_is_accepted(self):
        cfg = TechnicalContextConfig(max_tokens=5000, safety_ceiling=5000)
        assert cfg.max_tokens == 5000

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


class TestMaxTokensClamping:
    """A budget larger than the hard cap must be clamped, never rejected: rejecting it
    leaves the caller with no technical context at all."""

    def test_above_ceiling_is_clamped_not_rejected(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = TechnicalContextConfig(max_tokens=TechnicalContextConfig.safety_ceiling + 1)
        assert cfg.max_tokens == TechnicalContextConfig.safety_ceiling
        assert "clamping" in caplog.text

    def test_far_above_ceiling_is_clamped(self):
        cfg = TechnicalContextConfig(max_tokens=10 ** 6)
        assert cfg.max_tokens == TechnicalContextConfig.safety_ceiling

    def test_custom_ceiling_clamps_too(self):
        cfg = TechnicalContextConfig(max_tokens=5000, safety_ceiling=4000)
        assert cfg.max_tokens == 4000

    def test_budget_below_ceiling_is_untouched(self):
        cfg = TechnicalContextConfig(max_tokens=9000)
        assert cfg.max_tokens == 9000
