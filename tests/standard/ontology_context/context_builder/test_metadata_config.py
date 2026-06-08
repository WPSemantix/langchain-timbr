"""Tests for MetadataContextConfig + config_from_module."""

from __future__ import annotations

import pytest

from langchain_timbr.ontology_context.context_builder.metadata_config import (
    MetadataContextConfig,
    config_from_module,
)


class TestMetadataContextConfig:
    def test_defaults_are_valid(self):
        cfg = MetadataContextConfig()
        assert cfg.mode == "static"
        # SQL-gen metadata budget is now a SINGLE soft cap (the old hard-
        # ceiling-with-static-revert knob was removed; oversizing logs but
        # never reverts to static).
        assert cfg.metadata_context_max_tokens == 12_000
        # Retry budget is an int (Plan 2 update). Default 2 = up to 3 Step 1
        # LLM calls per chain.invoke() in the worst case.
        assert cfg.metadata_context_dynamic_retry == 2

    def test_negative_retry_budget_raises(self):
        with pytest.raises(ValueError):
            MetadataContextConfig(metadata_context_dynamic_retry=-1)

    def test_retry_budget_zero_is_valid(self):
        cfg = MetadataContextConfig(metadata_context_dynamic_retry=0)
        assert cfg.metadata_context_dynamic_retry == 0

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            MetadataContextConfig(mode="invalid")  # type: ignore[arg-type]

    def test_ddl_hard_under_soft_raises(self):
        with pytest.raises(ValueError):
            MetadataContextConfig(
                metadata_context_filter_max_tokens=6_000,
                metadata_context_filter_max_tokens_hard_ceiling=3_000,
            )

    def test_negative_caps_raise(self):
        with pytest.raises(ValueError):
            MetadataContextConfig(max_concept_prefilter_token=0)

    def test_max_concept_prefilter_token_default(self):
        cfg = MetadataContextConfig()
        assert cfg.max_concept_prefilter_token == 2_000


class TestConfigFromModule:
    def test_module_defaults_yield_static_mode(self):
        cfg = config_from_module()
        # Default in config.py is 'static' for the initial release.
        assert cfg.mode == "static"

    def test_overrides_applied(self):
        cfg = config_from_module(mode="dynamic", metadata_context_max_tokens=8_000)
        assert cfg.mode == "dynamic"
        assert cfg.metadata_context_max_tokens == 8_000

    def test_none_overrides_ignored(self):
        cfg = config_from_module(mode="auto", metadata_context_max_tokens=None)
        assert cfg.mode == "auto"
        # None overrides fall back to module/config default.
        assert cfg.metadata_context_max_tokens == 12_000

    def test_unknown_keys_silently_ignored(self):
        # Defensive: orchestrators pass mixed kwargs; unknown keys must not raise.
        cfg = config_from_module(mode="auto", banana="yellow")  # type: ignore[arg-type]
        assert cfg.mode == "auto"
