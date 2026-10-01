"""LLM answers that cannot be used: tolerant parsing, and re-asking with feedback.

Pins the behaviour behind three production failures:

  - Claude Sonnet 4.6 answering identify-concept with ``"result": customer_cube``
    (no quotes around the value), sometimes followed by a self-corrected block;
  - models with extended thinking (Claude Sonnet 5) returning ``content`` as a
    list of ``thinking`` + ``text`` parts instead of a string;
  - an unusable answer being retried with a hint that gave the model nothing to
    correct, instead of its own output plus the error.
"""
import json
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_timbr.llm_wrapper.llm_wrapper import LlmWrapper
from langchain_timbr.technical_context.extraction.llm import (
    _extraction_cache_clear,
    extract_candidates_with_llm,
)
from langchain_timbr.utils import memory
from langchain_timbr.utils.timbr_llm_utils import (
    LLMOutputError,
    _call_llm_with_output_retry,
    _extract_json,
    _parse_sql_and_reason_from_llm_response,
    answer_question,
    determine_concept,
)

_THINKING = {"type": "thinking", "thinking": "", "signature": "sig"}


def _thinking_response(text: str) -> AIMessage:
    return AIMessage(content=[_THINKING, {"type": "text", "text": text}])


class ScriptedLLM:
    """Returns queued answers in order and records the prompt of every call."""

    _llm_type = "test"

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class TestExtractJson:
    def test_plain_and_fenced(self):
        assert _extract_json('{"result": "customer"}') == {"result": "customer"}
        assert _extract_json('```json\n{"result": "customer"}\n```') == {"result": "customer"}

    def test_unquoted_value_is_repaired(self):
        raw = '```json\n{\n  "reason": "best match",\n  "result": customer_cube\n}\n```'
        assert _extract_json(raw) == {"reason": "best match", "result": "customer_cube"}

    def test_json_literals_are_not_quoted(self):
        raw = '{"a": true, "b": null, "result": order_cube}'
        assert _extract_json(raw) == {"a": True, "b": None, "result": "order_cube"}

    def test_self_corrected_answer_uses_the_last_block(self):
        raw = (
            '```json\n{"reason": "r", "result": order_cube}\n```\n\n'
            'Wait, I need to return valid JSON. Let me correct:\n\n'
            '```json\n{"reason": "r", "result": "shipment"}\n```'
        )
        assert _extract_json(raw)["result"] == "shipment"

    def test_raw_newline_inside_a_string(self):
        assert _extract_json('{"reason": "r", "result": "customer\n"}')["result"] == "customer\n"

    def test_prose_around_the_object(self):
        assert _extract_json('Here you go: {"result": "customer"} Hope it helps.') == {"result": "customer"}

    def test_not_json_raises(self):
        with pytest.raises(LLMOutputError):
            _extract_json("SELECT 1")


class TestCallLlmWithOutputRetry:
    @staticmethod
    def _parse(response):
        return _extract_json(response.content)

    def test_valid_answer_costs_one_call(self):
        llm = ScriptedLLM([AIMessage(content='{"ok": 1}')])
        parsed, _, _ = _call_llm_with_output_retry(llm, [HumanMessage(content="q")], self._parse)
        assert parsed == {"ok": 1}
        assert len(llm.prompts) == 1

    def test_retry_carries_request_output_and_error(self):
        llm = ScriptedLLM([AIMessage(content="not json"), AIMessage(content='{"ok": 1}')])
        prompt = [SystemMessage(content="rules"), HumanMessage(content="the question")]

        parsed, _, _ = _call_llm_with_output_retry(llm, prompt, self._parse)

        assert parsed == {"ok": 1}
        retry = llm.prompts[1]
        assert [m.content for m in retry[:2]] == ["rules", "the question"]
        assert isinstance(retry[2], AIMessage) and retry[2].content == "not json"
        assert "not valid JSON" in retry[3].content

    def test_string_prompt_is_kept_as_the_original_request(self):
        llm = ScriptedLLM([AIMessage(content="nope"), AIMessage(content='{"ok": 1}')])
        _call_llm_with_output_retry(llm, "the request", self._parse)
        assert llm.prompts[1][0].content == "the request"

    def test_gives_up_after_two_retries(self):
        llm = ScriptedLLM([AIMessage(content="bad")] * 3)
        with pytest.raises(LLMOutputError):
            _call_llm_with_output_retry(llm, [HumanMessage(content="q")], self._parse)
        assert len(llm.prompts) == 3

    def test_timeout_is_not_retried(self):
        with patch(
            "langchain_timbr.utils.timbr_llm_utils._call_llm_with_timeout",
            side_effect=TimeoutError("LLM call timed out after 1 seconds"),
        ) as mock_call:
            with pytest.raises(TimeoutError):
                _call_llm_with_output_retry(Mock(), "q", self._parse, retry_call_errors=True)
        assert mock_call.call_count == 1

    def test_call_errors_are_not_retried_by_default(self):
        llm = ScriptedLLM([RuntimeError("boom"), AIMessage(content='{"ok": 1}')])
        with pytest.raises(RuntimeError):
            _call_llm_with_output_retry(llm, "q", self._parse)
        assert len(llm.prompts) == 1

    def test_usage_is_summed_over_attempts(self):
        def _resp(text):
            return AIMessage(content=text, response_metadata={"usage": {"input_tokens": 10, "output_tokens": 2}})

        llm = ScriptedLLM([_resp("bad"), _resp('{"ok": 1}')])
        _, _, usage = _call_llm_with_output_retry(llm, "q", self._parse)
        assert usage["input_tokens"] == 20
        assert usage["output_tokens"] == 4


def _concepts(conn_params=None, **_):
    return {
        "customer": {"concept": "customer", "description": "a customer", "is_view": "false"},
        "product": {"concept": "product", "description": "a product", "is_view": "false"},
    }


def _identify_template():
    template = Mock()
    template.format_messages.side_effect = lambda **_: [
        SystemMessage(content="'result': the exact table name, no quotes"),
        HumanMessage(content="BUSINESS QUESTION: q"),
    ]
    return template


# The catalog builder reaches for the server; off, the legacy lines are rendered.
@patch("langchain_timbr.utils.timbr_llm_utils.config.enable_identify_concept_context", False)
@patch("langchain_timbr.utils.timbr_llm_utils.get_tags", return_value={"concept_tags": {}, "view_tags": {}})
@patch("langchain_timbr.utils.timbr_llm_utils.get_ontology_description", return_value=("", ""))
@patch("langchain_timbr.utils.timbr_llm_utils.get_concepts", side_effect=_concepts)
@patch("langchain_timbr.utils.timbr_llm_utils.get_determine_concept_prompt_template")
class TestDetermineConcept:
    CONN = {"url": "http://test", "token": "t", "ontology": "test_ont"}

    def _run(self, mock_prompt, answers, **kwargs):
        mock_prompt.return_value = _identify_template()
        llm = ScriptedLLM(answers)
        return determine_concept("Which customers?", llm, self.CONN, **kwargs), llm

    def test_unquoted_result_needs_no_retry(self, mock_prompt, *_):
        res, llm = self._run(mock_prompt, [AIMessage(content='```json\n{"reason": "r", "result": customer\n}\n```')])
        assert res["concept"] == "customer"
        assert len(llm.prompts) == 1

    def test_trailing_newline_in_result(self, mock_prompt, *_):
        res, _ = self._run(mock_prompt, [AIMessage(content='{"reason": "r", "result": "customer\n"}')])
        assert res["concept"] == "customer"

    def test_thinking_content_parts(self, mock_prompt, *_):
        res, _ = self._run(mock_prompt, [_thinking_response('{"reason": "r", "result": "product"}')])
        assert res["concept"] == "product"
        assert res["identify_concept_reason"] == "r"

    def test_prompt_no_longer_says_no_quotes(self, mock_prompt, *_):
        _, llm = self._run(mock_prompt, [AIMessage(content='{"result": "customer"}')])
        system = llm.prompts[0][0].content
        assert "no quotes" not in system
        assert "JSON string" in system

    def test_unknown_concept_is_sent_back_with_its_output(self, mock_prompt, *_):
        res, llm = self._run(mock_prompt, [
            AIMessage(content='{"reason": "r", "result": "clients"}'),
            AIMessage(content='{"reason": "r", "result": "customer"}'),
        ])
        assert res["concept"] == "customer"
        retry = llm.prompts[1]
        assert '"result": "clients"' in retry[-2].content
        assert "Concept 'clients' not found" in retry[-1].content

    def test_invalid_json_is_not_used_as_the_concept_name(self, mock_prompt, *_):
        mock_prompt.return_value = _identify_template()
        llm = ScriptedLLM([AIMessage(content='{"reason": "r" "result": }')] * 3)
        with pytest.raises(Exception) as exc_info:
            determine_concept("Which customers?", llm, self.CONN)
        assert "not valid JSON" in str(exc_info.value)
        assert len(llm.prompts) == 3

    def test_retries_counts_attempts(self, mock_prompt, *_):
        mock_prompt.return_value = _identify_template()
        llm = ScriptedLLM([AIMessage(content='{"result": "nope"}')] * 3)
        with pytest.raises(Exception):
            determine_concept("Which customers?", llm, self.CONN, retries=1)
        assert len(llm.prompts) == 1


class TestSqlResponseParsing:
    def test_broken_json_is_not_returned_as_sql(self):
        with pytest.raises(LLMOutputError):
            _parse_sql_and_reason_from_llm_response(AIMessage(content='{"reason": "r", "result": "SELECT 1'))

    def test_null_result_is_an_output_error(self):
        with pytest.raises(LLMOutputError):
            _parse_sql_and_reason_from_llm_response(AIMessage(content='{"reason": "r", "result": null}'))

    def test_plain_sql_is_still_accepted(self):
        parsed = _parse_sql_and_reason_from_llm_response(AIMessage(content="SELECT 1"))
        assert parsed["sql"] == "SELECT 1"

    def test_thinking_content_parts(self):
        parsed = _parse_sql_and_reason_from_llm_response(
            _thinking_response('{"reason": "r", "result": "SELECT 1"}')
        )
        assert parsed == {"sql": "SELECT 1", "reason": "r", "decisions": None}


class TestThinkingContentParts:
    """``content`` as a list of parts must reach every consumer as text."""

    def test_llm_wrapper_returns_text(self):
        wrapper = LlmWrapper.__new__(LlmWrapper)
        client = Mock()
        client.invoke.return_value = _thinking_response("the answer")
        object.__setattr__(wrapper, "client", client)
        assert wrapper._call("prompt") == "the answer"

    @patch("langchain_timbr.utils.timbr_llm_utils.get_qa_prompt_template")
    def test_answer_is_a_string(self, mock_template):
        mock_template.return_value.format_messages.return_value = [
            SystemMessage(content="s"), HumanMessage(content="h"),
        ]
        llm = ScriptedLLM([_thinking_response("42 orders.")])
        res = answer_question("How many?", llm, {"url": "u", "token": "t", "ontology": "o"}, results=[{"n": 42}])
        assert res["answer"] == "42 orders."

    @patch("langchain_timbr.utils.memory.get_memory_kb_classifier_prompt_template")
    def test_memory_classifier_reads_text_part(self, mock_template):
        mock_template.return_value.format_messages.return_value = [HumanMessage(content="classify")]
        llm = ScriptedLLM([_thinking_response(json.dumps({"is_follow_up": False, "summary": "s"}))])
        out = memory.classify_follow_up(llm, {}, "a question", messages=[], id_map={})
        assert out is not None and out["is_follow_up"] is False

    @patch("langchain_timbr.utils.memory.get_memory_kb_classifier_prompt_template")
    def test_memory_classifier_invalid_json_is_retried(self, mock_template):
        mock_template.return_value.format_messages.return_value = [HumanMessage(content="classify")]
        llm = ScriptedLLM([
            AIMessage(content="Sure! It is not a follow up."),
            AIMessage(content=json.dumps({"is_follow_up": False, "summary": "s"})),
        ])
        out = memory.classify_follow_up(llm, {}, "a question", messages=[], id_map={})
        assert out is not None
        assert len(llm.prompts) == 2


class TestCandidateExtractionRetry:
    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        _extraction_cache_clear()
        yield
        _extraction_cache_clear()

    def test_thinking_content_parts(self):
        llm = ScriptedLLM([_thinking_response('{"candidates": [{"literal": "Active", "synonyms": []}]}')])
        assert extract_candidates_with_llm("active orders", llm=llm) == ["Active"]

    def test_invalid_json_is_sent_back_with_the_output(self):
        llm = ScriptedLLM(["Active, US", '{"candidates": [{"literal": "Active", "synonyms": []}]}'])
        assert extract_candidates_with_llm("active orders", llm=llm) == ["Active"]
        assert "Active, US" in llm.prompts[1]
        assert llm.prompts[1].startswith(llm.prompts[0])

    def test_unusable_answer_is_not_cached(self):
        llm = ScriptedLLM(["bad"] * 3 + ['{"candidates": [{"literal": "Active", "synonyms": []}]}'])
        assert extract_candidates_with_llm("active orders", llm=llm) == []
        assert extract_candidates_with_llm("active orders", llm=llm) == ["Active"]
        assert len(llm.prompts) == 4
