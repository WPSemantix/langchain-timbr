"""The LLM timeout must release the caller, not wait out the model.

Before the shared executor, both call sites built a per-call ThreadPoolExecutor
inside a ``with`` block. Leaving the block called ``shutdown(wait=True)``, which
joined the worker -- so the caller waited for the full model call regardless of
the timeout it asked for.
"""

import time

import pytest

from langchain_timbr.llm_wrapper.llm_wrapper import _apply_default_llm_timeout
from langchain_timbr.technical_context.extraction.llm import _call_llm
from langchain_timbr.utils import llm_executor
from langchain_timbr.utils.llm_executor import get_llm_executor
from langchain_timbr.utils.timbr_llm_utils import _call_llm_with_timeout


class SlowLLM:
    """Stub whose invoke outlives the caller's timeout."""

    def __init__(self, delay=5.0):
        self.delay = delay

    def invoke(self, prompt):
        time.sleep(self.delay)
        return "too late"


class FastLLM:
    def invoke(self, prompt):
        return "on time"


class TestTimeoutReleasesCaller:
    def test_generation_call_released_at_timeout(self):
        start = time.monotonic()
        with pytest.raises(TimeoutError, match="timed out after 1 seconds"):
            _call_llm_with_timeout(SlowLLM(delay=5.0), "prompt", timeout=1)
        elapsed = time.monotonic() - start
        assert elapsed < 2.0, f"caller held for {elapsed:.2f}s, expected release at ~1s"

    def test_extraction_call_released_at_timeout(self):
        start = time.monotonic()
        with pytest.raises(TimeoutError, match="extraction timed out after 1s"):
            _call_llm(SlowLLM(delay=5.0), "prompt", timeout=1)
        elapsed = time.monotonic() - start
        assert elapsed < 2.0, f"caller held for {elapsed:.2f}s, expected release at ~1s"

    def test_pool_still_usable_after_a_timeout(self):
        with pytest.raises(TimeoutError):
            _call_llm_with_timeout(SlowLLM(delay=3.0), "prompt", timeout=1)
        assert _call_llm_with_timeout(FastLLM(), "prompt", timeout=5) == "on time"


class TestHappyPath:
    def test_generation_returns_response(self):
        assert _call_llm_with_timeout(FastLLM(), "prompt", timeout=5) == "on time"

    def test_extraction_returns_response(self):
        assert _call_llm(FastLLM(), "prompt", timeout=5) == "on time"


class TestSharedExecutor:
    def test_executor_is_reused_across_calls(self):
        """One pool for the process -- no per-call construction and teardown."""
        first = get_llm_executor()
        _call_llm_with_timeout(FastLLM(), "prompt", timeout=5)
        _call_llm(FastLLM(), "prompt", timeout=5)
        assert get_llm_executor() is first

    def test_both_call_sites_share_one_pool(self):
        from langchain_timbr.technical_context.extraction import llm as extraction_llm
        from langchain_timbr.utils import timbr_llm_utils

        assert extraction_llm.get_llm_executor is timbr_llm_utils.get_llm_executor

    def test_fork_reset_rebuilds_the_pool(self):
        """A pool inherited across fork() has no live threads; drop and rebuild it."""
        original = get_llm_executor()
        try:
            llm_executor._reset_after_fork()
            assert llm_executor._executor is None
            assert get_llm_executor() is not original
        finally:
            llm_executor._reset_after_fork()

    def test_pool_is_not_built_at_import(self):
        """Import-time construction is what breaks under `gunicorn --preload`."""
        import importlib

        fresh = importlib.reload(llm_executor)
        try:
            assert fresh._executor is None
        finally:
            importlib.reload(llm_executor)


class TestClientTimeoutDefault:
    def test_default_applied_when_caller_passed_none(self):
        assert _apply_default_llm_timeout({})["timeout"] == 120

    @pytest.mark.parametrize(
        "alias", ["timeout", "request_timeout", "default_request_timeout"]
    )
    def test_caller_timeout_is_respected(self, alias):
        assert _apply_default_llm_timeout({alias: 45}) == {alias: 45}

    def test_reaches_the_openai_client(self):
        """The param must actually land on the client, not just in the dict."""
        from langchain_openai import ChatOpenAI

        params = _apply_default_llm_timeout({})
        client = ChatOpenAI(openai_api_key="test-key", model_name="gpt-4o", **params)
        assert client.request_timeout == 120

class TestBedrockReadTimeout:
    """Bedrock has no timeout kwarg; the bound goes on the botocore Config."""

    @staticmethod
    def _build(**llm_params):
        try:
            import langchain_aws
            from unittest.mock import patch

            from langchain_timbr.llm_wrapper.llm_wrapper import LlmWrapper

            with patch.object(langchain_aws, "ChatBedrockConverse") as fake:
                LlmWrapper.__new__(LlmWrapper)._connect_to_llm(
                    "amazon_bedrock_converse_chat", api_key="k", model="m", **llm_params
                )
            return fake.call_args.kwargs
        except ImportError:
            pytest.skip("langchain_aws is not installed")

    def test_default_read_timeout_applied(self):
        assert self._build()["config"].read_timeout == 120

    def test_caller_supplied_config_is_left_alone(self):
        try:
            from botocore.config import Config

            caller_config = Config(read_timeout=30)
            assert self._build(config=caller_config)["config"] is caller_config
        except ImportError:
            pytest.skip("botocore is not installed")
