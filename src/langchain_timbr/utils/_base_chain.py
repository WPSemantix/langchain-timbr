import threading
from typing import Any, Dict, List, Optional
from langchain_core.runnables import Runnable

try:
    from langsmith import trace as ls_trace
    _LANGSMITH_AVAILABLE = True
except ImportError:
    _LANGSMITH_AVAILABLE = False


# Re-entrancy guard for the once-per-question version refresh. An agent runs
# several chains for one question (generate -> validate -> execute), and each
# goes through ``invoke``; without this, one question would force a probe per
# chain instead of one. Thread-local because a thread serves one question at a
# time under ``gunicorn --threads N``.
_question_scope = threading.local()


def _init_chain_context(ctx: Optional[dict]) -> dict:
    """Initialize or ensure a chain_context dict has the required sub-dicts."""
    if ctx is None:
        ctx = {}
    ctx.setdefault("duration", {})
    ctx.setdefault("reasoning", {})
    ctx.setdefault("tokens", {})
    ctx.setdefault("memory", None)
    return ctx


class Chain(Runnable):
    """
    Compatibility base class that mimics the legacy langchain.chains.base.Chain
    interface (removed in langchain 1.x).

    Subclasses should implement:
      - ``_call(self, inputs, run_manager=None) -> dict``
      - ``input_keys`` property
      - ``output_keys`` property
    """

    def __init__(self, **kwargs):
        self._received_log_ctx = None
        self._received_chain_context = None

    @property
    def input_keys(self) -> List[str]:
        return []

    @property
    def output_keys(self) -> List[str]:
        return []

    def _call(self, inputs: Dict[str, Any], run_manager=None) -> Dict[str, Any]:
        raise NotImplementedError

    # ---- once-per-question ontology-version refresh ------------------------

    def _refresh_ontology_version(self) -> None:
        """Hook: refresh the ontology version once, at the start of a question.

        A no-op here, because the base class has no connection of its own.
        Chains that hold connection parameters override it (one line, calling
        ``_refresh_version_for``), so an ontology edit is visible to the *next
        question* rather than up to ``CACHE_TIMEOUT`` later.

        A chain that does not override degrades safely: it falls back to the
        shared probe's throttle and loses the promptness, never correctness.
        """
        return None

    @staticmethod
    def _refresh_version_for(conn_params) -> None:
        """Force one shared-probe refresh for ``conn_params``.

        Failures are swallowed on purpose. This runs before the question starts,
        and a probe that cannot reach the server must not be the thing that
        fails the request — the throttled path will pick the version up within
        ``CACHE_TIMEOUT``, and any real connectivity problem surfaces on the very
        next query with a better message.
        """
        try:
            # Read the switch at call time, not import time, so it can be
            # toggled (tests turn it off — a unit test should not do network
            # I/O just to start a chain).
            from .. import config

            if not config.version_refresh_per_question:
                return
            from .timbr_utils import get_ontology_version

            get_ontology_version(conn_params, force=True)
        except Exception:
            pass

    def _enter_question_scope(self) -> bool:
        """Refresh the version if this is the outermost invoke on this thread."""
        if getattr(_question_scope, "active", False):
            return False
        _question_scope.active = True
        try:
            self._refresh_ontology_version()
        except Exception:
            pass
        return True

    @staticmethod
    def _exit_question_scope(owns_scope: bool) -> None:
        if owns_scope:
            _question_scope.active = False

    def invoke(self, input: Dict[str, Any], config=None, log_ctx=None, **kwargs) -> Dict[str, Any]:
        self._received_log_ctx = log_ctx
        self._received_chain_context = _init_chain_context(input.get("chain_context"))
        owns_scope = self._enter_question_scope()
        try:
            if _LANGSMITH_AVAILABLE:
                with ls_trace(name=self.__class__.__name__, run_type="chain", inputs={"input": input}) as rt:
                    result = self._call(input)
                    result["chain_context"] = self._received_chain_context
                    rt.end(outputs=result)
                    return result
            result = self._call(input)
            result["chain_context"] = self._received_chain_context
            return result
        finally:
            self._exit_question_scope(owns_scope)

    async def ainvoke(self, input: Dict[str, Any], config=None, log_ctx=None, **kwargs) -> Dict[str, Any]:
        self._received_log_ctx = log_ctx
        self._received_chain_context = _init_chain_context(input.get("chain_context"))
        owns_scope = self._enter_question_scope()
        try:
            if _LANGSMITH_AVAILABLE:
                with ls_trace(name=self.__class__.__name__, run_type="chain", inputs={"input": input}) as rt:
                    result = self._call(input)
                    result["chain_context"] = self._received_chain_context
                    rt.end(outputs=result)
                    return result
            result = self._call(input)
            result["chain_context"] = self._received_chain_context
            return result
        finally:
            self._exit_question_scope(owns_scope)
