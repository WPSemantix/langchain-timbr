"""A failed execution must close its running-log row.

The running row (sys_agents_running) is only removed by a history post. A
failure that skipped the post — an exception out of a chain, or an error
returned by a chain that never writes history — left the execution shown as
running forever.
"""
from unittest.mock import Mock, patch

import pytest

from langchain_timbr import (
    GenerateAnswerChain,
    IdentifyTimbrConceptChain,
    create_timbr_sql_agent,
)
from langchain_timbr.utils import chain_logger
from langchain_timbr.utils.chain_logger import AgentLogContext, _now, log_agent_failure
from langchain_timbr.utils.memory import MEMORY_DISABLED

RUNNING = "/timbr-server/log_agent/running"
HISTORY = "/timbr-server/log_agent/history"
CONN = dict(url="http://test", token="test", ontology="test")


@pytest.fixture
def posts():
    """Every log post as (endpoint, payload); nothing reaches the network."""
    captured = []

    def _capture(url, token, endpoint_path, payload, verify_ssl=True):
        captured.append((endpoint_path, payload))

    with patch.object(chain_logger, "_dispatch", side_effect=_capture), \
            patch("langchain_timbr.utils.memory.resolve_memory", return_value=MEMORY_DISABLED), \
            patch("langchain_timbr.kbclient.fetch_rules", return_value=None):
        yield captured


def _of(posts, endpoint):
    return [payload for path, payload in posts if path == endpoint]


def _ctx(**overrides):
    fields = dict(
        query_id="q1", agent_name="", url="http://test", token="t", chain_type="Test",
        start_time=_now(), prompt="a question", enable_trace=True,
    )
    fields.update(overrides)
    return AgentLogContext(**fields)


class TestLogAgentFailure:
    def test_posts_failed_history_with_the_step(self, posts):
        ctx = _ctx(current_step="generating_sql", concept="customer", ontology="ont")
        log_agent_failure(ctx, "LLM call failed: boom")

        (history,) = _of(posts, HISTORY)
        assert history["query_id"] == "q1"
        assert history["status"] == "failed"
        assert history["error"] == "LLM call failed: boom"
        assert history["failed_at_step"] == "generating_sql"
        assert history["concept"] == "customer"

    def test_timeout_status(self, posts):
        log_agent_failure(_ctx(), "LLM call timed out after 120 seconds")
        assert _of(posts, HISTORY)[0]["status"] == "timeout"

    def test_history_is_posted_once(self, posts):
        ctx = _ctx()
        log_agent_failure(ctx, "boom")
        log_agent_failure(ctx, "boom again")
        assert len(_of(posts, HISTORY)) == 1

    def test_no_context_is_a_noop(self, posts):
        log_agent_failure(None, "boom")
        assert posts == []


class TestStandaloneChain:
    @patch("langchain_timbr.langchain.identify_concept_chain.determine_concept")
    def test_returned_error_closes_the_running_row(self, mock_determine, posts, mock_llm):
        mock_determine.side_effect = Exception("Failed to determine concept: not valid JSON")
        chain = IdentifyTimbrConceptChain(llm=mock_llm, enable_trace=True, **CONN)

        result = chain.invoke({"prompt": "a question"})

        (running,) = _of(posts, RUNNING)
        (history,) = _of(posts, HISTORY)
        assert history["query_id"] == running["query_id"]
        assert history["status"] == "failed"
        assert history["failed_at_step"] == "identifying_concept"
        assert history["error"] == result["error"]

    @patch("langchain_timbr.langchain.identify_concept_chain.determine_concept")
    def test_success_posts_no_failure(self, mock_determine, posts, mock_llm):
        mock_determine.return_value = {
            "concept": "customer", "schema": "dtimbr", "concept_metadata": {},
            "identify_concept_reason": "r", "ontology": "test", "usage_metadata": {}, "duration_ms": 1,
        }
        chain = IdentifyTimbrConceptChain(llm=mock_llm, enable_trace=True, **CONN)

        chain.invoke({"prompt": "a question"})

        assert _of(posts, HISTORY) == []

    @patch("langchain_timbr.langchain.identify_concept_chain.determine_concept")
    def test_logging_disabled_posts_nothing(self, mock_determine, posts, mock_llm):
        mock_determine.side_effect = Exception("boom")
        chain = IdentifyTimbrConceptChain(llm=mock_llm, enable_trace=False, **CONN)

        chain.invoke({"prompt": "a question"})

        assert posts == []

    def test_raised_exception_closes_the_running_row(self, posts, mock_llm):
        chain = GenerateAnswerChain(llm=mock_llm, enable_trace=True, enable_history=True, **CONN)
        chain._execute_chain.invoke = Mock(side_effect=RuntimeError("Error executing the chain: boom"))

        with pytest.raises(RuntimeError):
            chain.invoke({"prompt": "a question"})

        (running,) = _of(posts, RUNNING)
        (history,) = _of(posts, HISTORY)
        assert history["query_id"] == running["query_id"]
        assert history["status"] == "failed"
        assert "boom" in history["error"]

    @patch("langchain_timbr.langchain.generate_answer_chain.answer_question")
    def test_nested_execute_chain_shares_the_log_context(self, mock_answer, posts, mock_llm):
        mock_answer.return_value = {"answer": "an answer", "usage_metadata": {}}
        chain = GenerateAnswerChain(llm=mock_llm, enable_trace=True, enable_history=True, **CONN)
        chain._execute_chain.invoke = Mock(return_value={"rows": [{"n": 1}], "sql": "SELECT 1"})

        chain.invoke({"prompt": "a question"})

        passed_ctx = chain._execute_chain.invoke.call_args.kwargs["log_ctx"]
        (running,) = _of(posts, RUNNING)
        assert passed_ctx is not None and passed_ctx.query_id == running["query_id"]
        # The answer chain posted its own (successful) history; no failure row on top.
        (history,) = _of(posts, HISTORY)
        assert history["status"] == "completed"


class TestAgent:
    def _agent(self, mock_llm, **kwargs):
        return create_timbr_sql_agent(llm=mock_llm, enable_trace=True, **CONN, **kwargs)

    def test_chain_exception_closes_the_running_row(self, posts, mock_llm):
        agent = self._agent(mock_llm, generate_answer=True)
        agent._chain.invoke = Mock(side_effect=RuntimeError("Error executing the chain: Failed to determine concept"))

        result = agent.invoke({"input": "a question"})

        assert "Failed to determine concept" in result["error"]
        (running,) = _of(posts, RUNNING)
        (history,) = _of(posts, HISTORY)
        assert history["query_id"] == running["query_id"]
        assert history["status"] == "failed"
        assert "Failed to determine concept" in history["error"]

    def test_chain_timeout_is_logged_as_timeout(self, posts, mock_llm):
        agent = self._agent(mock_llm, generate_answer=True)
        agent._chain.invoke = Mock(side_effect=TimeoutError("LLM call timed out while answering question"))

        agent.invoke({"input": "a question"})

        assert _of(posts, HISTORY)[0]["status"] == "timeout"

    def test_returned_error_without_history_closes_the_running_row(self, posts, mock_llm):
        agent = self._agent(mock_llm, generate_answer=False)
        agent._chain.invoke = Mock(return_value={"rows": [], "sql": "SELECT x", "error": "invalid SQL"})

        agent.invoke({"input": "a question"})

        (history,) = _of(posts, HISTORY)
        assert history["status"] == "failed"
        assert history["error"] == "invalid SQL"

    def test_no_second_history_when_the_chain_already_posted_it(self, posts, mock_llm):
        agent = self._agent(mock_llm, generate_answer=True)

        def _chain_posts_history(inputs, log_ctx=None):
            log_agent_failure(log_ctx, "invalid SQL")  # stands in for the chain's own history post
            return {"rows": [], "sql": "SELECT x", "error": "invalid SQL"}

        agent._chain.invoke = Mock(side_effect=_chain_posts_history)

        agent.invoke({"input": "a question"})

        assert len(_of(posts, HISTORY)) == 1
