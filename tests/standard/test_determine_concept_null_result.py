"""Regression tests for null/None concept-result handling.

Covers the bug where the concept-identifying LLM returns ``{"result": null}`` (or a
provider response shape that yields no text), which previously crashed with
``AttributeError: 'NoneType' object has no attribute 'strip'`` instead of a clear
"could not identify a concept" error. Also covers the standalone
``IdentifyTimbrConceptChain`` surfacing the error instead of raising.
"""
import pytest
from unittest.mock import Mock, patch

from langchain_core.messages import HumanMessage, AIMessage

from langchain_timbr import IdentifyTimbrConceptChain
from langchain_timbr.utils.timbr_llm_utils import determine_concept, _get_response_text

MOCK_URL = "http://test-timbr-url"
MOCK_TOKEN = "test-token"


class _Resp:
    """Minimal stand-in for an LLM response object with a ``.content`` attribute."""

    def __init__(self, content):
        self.content = content


def _two_concepts(conn_params=None, **_):
    # Two concepts so determine_concept takes the LLM path instead of
    # short-circuiting on a single candidate.
    return {
        "customer": {"concept": "customer", "description": "a customer", "is_view": "false"},
        "product": {"concept": "product", "description": "a product", "is_view": "false"},
    }


def _prompt_template():
    template = Mock()
    template.format_messages.return_value = [
        HumanMessage(content="system text"),
        HumanMessage(content="user text with no quotes"),
    ]
    return template


class TestGetResponseTextNoneSafe:
    """_get_response_text must never crash / return None on empty content."""

    def test_none_content_returns_empty_string(self):
        assert _get_response_text(_Resp(None)) == ""

    def test_list_without_text_part_returns_empty_string(self):
        assert _get_response_text(_Resp([{"type": "reasoning", "summary": "thinking"}])) == ""


class TestDetermineConceptNullResult:
    """determine_concept must raise a clear error (not AttributeError) on a null result."""

    @patch("langchain_timbr.utils.timbr_llm_utils.get_determine_concept_prompt_template")
    @patch("langchain_timbr.utils.timbr_llm_utils.get_tags", return_value={"concept_tags": {}, "view_tags": {}})
    @patch("langchain_timbr.utils.timbr_llm_utils.get_ontology_description", return_value=("", ""))
    @patch("langchain_timbr.utils.timbr_llm_utils.get_concepts", side_effect=_two_concepts)
    @patch("langchain_timbr.utils.timbr_llm_utils._call_llm_with_timeout")
    def test_null_json_result_raises_clear_error(
        self, mock_call, mock_concepts, mock_desc, mock_tags, mock_prompt
    ):
        mock_prompt.return_value = _prompt_template()
        mock_call.return_value = AIMessage(
            content='```json\n{"reason": "no match", "result": null}\n```'
        )

        llm = Mock()
        llm._llm_type = "openai"
        conn_params = {"url": MOCK_URL, "token": MOCK_TOKEN, "ontology": "test_ont"}

        with pytest.raises(Exception) as exc_info:
            determine_concept("Anything about sales?", llm, conn_params, retries=1)

        msg = str(exc_info.value)
        assert "Failed to determine concept" in msg
        assert "Reason: no match" in msg
        assert "strip" not in msg  # not the old AttributeError

    @patch("langchain_timbr.utils.timbr_llm_utils.get_determine_concept_prompt_template")
    @patch("langchain_timbr.utils.timbr_llm_utils.get_tags", return_value={"concept_tags": {}, "view_tags": {}})
    @patch("langchain_timbr.utils.timbr_llm_utils.get_ontology_description", return_value=("", ""))
    @patch("langchain_timbr.utils.timbr_llm_utils.get_concepts", side_effect=_two_concepts)
    @patch("langchain_timbr.utils.timbr_llm_utils._call_llm_with_timeout")
    def test_no_text_response_raises_clear_error(
        self, mock_call, mock_concepts, mock_desc, mock_tags, mock_prompt
    ):
        mock_prompt.return_value = _prompt_template()
        # List content with no `type == 'text'` part -> _get_response_text yields "".
        mock_call.return_value = _Resp([{"type": "reasoning", "summary": "thinking"}])

        llm = Mock()
        llm._llm_type = "openai"
        conn_params = {"url": MOCK_URL, "token": MOCK_TOKEN, "ontology": "test_ont"}

        with pytest.raises(Exception) as exc_info:
            determine_concept("Anything about sales?", llm, conn_params, retries=1)

        msg = str(exc_info.value)
        assert "Failed to determine concept" in msg
        assert "strip" not in msg


class TestIdentifyConceptChainErrorSurface:
    """IdentifyTimbrConceptChain must surface the error instead of raising."""

    @patch("langchain_timbr.langchain.identify_concept_chain.determine_concept")
    def test_chain_returns_error_field_on_failure(self, mock_determine, mock_llm):
        mock_determine.side_effect = Exception(
            "Failed to determine concept: The model could not identify a relevant concept for the question."
        )
        chain = IdentifyTimbrConceptChain(
            llm=mock_llm, url="http://test", token="test", ontology="test"
        )

        result = chain.invoke({"prompt": "an irrelevant question"})

        assert result.get("concept") is None
        assert "error" in result
        assert "Failed to determine concept" in result["error"]

    @patch("langchain_timbr.utils.chain_logger._safe_post")
    @patch("langchain_timbr.utils.chain_logger.log_chain_trace")
    @patch("langchain_timbr.langchain.identify_concept_chain.determine_concept")
    def test_chain_logs_failed_trace(self, mock_determine, mock_log_trace, _mock_post, mock_llm):
        mock_determine.side_effect = Exception("boom")
        chain = IdentifyTimbrConceptChain(
            llm=mock_llm, url="http://test", token="test", ontology="test",
            enable_trace=True,
        )

        chain.invoke({"prompt": "an irrelevant question"})

        statuses = [c.kwargs.get("status") for c in mock_log_trace.call_args_list]
        assert "failed" in statuses
