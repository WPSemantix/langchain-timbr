"""Regression tests for the budget-knob consolidation refactor.

Locks two behaviors that are easy to silently regress:

1. ``max_graph_depth`` resolution on chain constructors — agent_options
   override wins; kwarg falls back to the constructor default; both branches
   reach ``self._max_graph_depth``.

2. The dynamic metadata-context pipeline NEVER reverts to the static strings
   purely because the rebuilt output exceeded the soft cap. The old
   ``metadata_context_max_tokens`` hard-ceiling-with-static-revert branch
   was removed; oversizing must log a warning and emit the rebuilt strings
   unchanged. See ``timbr_llm_utils.py`` line ~1247.
"""

from __future__ import annotations

from unittest.mock import patch

from langchain_timbr import config
from langchain_timbr.langchain.generate_timbr_sql_chain import GenerateTimbrSqlChain
from langchain_timbr.utils import timbr_llm_utils as _utils


MOCK_URL = "http://test.timbr.ai"
MOCK_TOKEN = "tk_test"


class TestMaxGraphDepthChainResolution:
    """Agent-options override wins over the kwarg; kwarg falls back to default."""

    @patch("langchain_timbr.langchain.generate_timbr_sql_chain.get_timbr_agent_options")
    def test_agent_options_override_wins(self, mock_get_options):
        mock_get_options.return_value = {"max_graph_depth": "7"}
        chain = GenerateTimbrSqlChain(
            llm=object(),  # bypass LlmWrapper env-var fetch
            url=MOCK_URL,
            token=MOCK_TOKEN,
            agent="some-agent",
            max_graph_depth=2,  # ignored — agent_options has the key
        )
        assert chain._max_graph_depth == 7

    @patch("langchain_timbr.langchain.generate_timbr_sql_chain.get_timbr_agent_options")
    def test_kwarg_used_when_agent_options_missing_key(self, mock_get_options):
        mock_get_options.return_value = {}  # no max_graph_depth key
        chain = GenerateTimbrSqlChain(
            llm=object(),
            url=MOCK_URL,
            token=MOCK_TOKEN,
            agent="some-agent",
            max_graph_depth=4,
        )
        assert chain._max_graph_depth == 4

    def test_no_agent_kwarg_branch_uses_default(self):
        chain = GenerateTimbrSqlChain(
            llm=object(),
            url=MOCK_URL,
            token=MOCK_TOKEN,
            # No agent → kwarg-only branch → kwarg default == config.max_graph_depth
        )
        assert chain._max_graph_depth == config.max_graph_depth


class TestNoStaticRevertWhenOverSoftCap:
    """The old hard-revert-to-static branch is gone (line ~1247 of
    timbr_llm_utils.py). Rebuilt strings over the soft cap must be emitted
    as-is, with a warning log — never replaced by the static triple."""

    def test_warning_log_message_present_in_source(self):
        """Locks the warning message wording so the no-revert branch can't
        be silently re-introduced. Inspecting the source is a much lighter
        lock than a full pipeline mock — it catches anyone who replaces the
        warning with a ``return STATIC`` block."""
        import inspect
        src = inspect.getsource(_utils._apply_dynamic_metadata_context)
        # Warning must be present
        assert "Dynamic metadata-context still over soft cap" in src
        assert "no revert to static" in src
        # And NO ``ontology.set_filtered_cache(cache_key, entry)`` SHORTLY AFTER
        # a static-strings rebuild assignment. The marker we lock: the comment
        # explaining the policy must remain.
        assert "dynamic-over-budget is preferred over static-but-much-larger" in src

    def test_no_set_filtered_cache_in_old_revert_position(self):
        """The OLD branch wrote the STATIC strings to the filtered cache when
        rebuilt > cap. After the refactor, the only ``set_filtered_cache``
        call below the soft-cap check writes the REBUILT strings — confirm
        the static-write path is gone by checking the rebuild-variable names
        flow through to the cache write."""
        import inspect
        src = inspect.getsource(_utils._apply_dynamic_metadata_context)
        # The cache write that follows the warning must use rebuilt strings,
        # not ``static_columns_str``/``static_measures_str``/``static_rel_prop_str``.
        cap_check = src.index("Dynamic metadata-context still over soft cap")
        tail = src[cap_check:]
        # Locate the first set_filtered_cache call after the warning
        cache_call = tail.index("set_filtered_cache")
        # In a 200-char window around the cache call, rebuilt names appear;
        # static names must not.
        window = tail[max(0, cache_call - 200) : cache_call + 200]
        assert "new_columns_str" in window
        assert "static_columns_str" not in window
