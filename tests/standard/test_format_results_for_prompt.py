"""Tests for the token-optimized rendering of query results sent to the LLM."""
import csv
import io
import json
from unittest.mock import Mock, patch

from langchain_timbr.utils.general import format_results_for_prompt


def _dict_rows(row_count, column_count=3):
    return [
        {f"col_{c}": f"r{r}c{c}" for c in range(column_count)}
        for r in range(row_count)
    ]


def _parse_csv(payload):
    header, _, body = payload.partition('\n')
    return header, list(csv.reader(io.StringIO(body)))


class TestPassThrough:
    """Results that are small, non-tabular or unparsable are returned untouched."""

    def test_small_narrow_result_unchanged(self):
        rows = _dict_rows(50, 10)
        assert format_results_for_prompt(rows) is rows

    def test_empty_and_none_unchanged(self):
        assert format_results_for_prompt([]) == []
        assert format_results_for_prompt(None) is None

    def test_plain_string_unchanged(self):
        text = "no rows returned"
        assert format_results_for_prompt(text) is text

    def test_malformed_json_string_unchanged(self):
        text = "[{'broken': ,}]"
        assert format_results_for_prompt(text) is text

    def test_list_of_scalars_unchanged(self):
        rows = [f"value_{i}" for i in range(100)]
        assert format_results_for_prompt(rows) is rows

    def test_rows_without_columns_unchanged(self):
        rows = [{} for _ in range(100)]
        assert format_results_for_prompt(rows) is rows


class TestThresholds:

    def test_many_rows_converted(self):
        result = format_results_for_prompt(_dict_rows(51, 3))
        assert isinstance(result, str)
        assert result.startswith("Query results in CSV format (51 rows, 3 columns")

    def test_few_rows_many_columns_converted(self):
        result = format_results_for_prompt(_dict_rows(5, 11))
        assert isinstance(result, str)
        assert result.startswith("Query results in CSV format (5 rows, 11 columns")

    def test_csv_is_smaller_than_json(self):
        rows = _dict_rows(200, 8)
        assert len(format_results_for_prompt(rows)) < len(json.dumps(rows))


class TestConfigurableThresholds:
    """ANSWER_RESULTS_MAX_ROWS / ANSWER_RESULTS_MAX_COLUMNS override the defaults."""

    def test_env_defaults_match_documented_values(self):
        from langchain_timbr import config

        assert config.answer_results_max_rows == 50
        assert config.answer_results_max_columns == 10

    def test_env_variables_are_read(self, monkeypatch):
        import importlib
        from langchain_timbr import config

        monkeypatch.setenv("ANSWER_RESULTS_MAX_ROWS", "500")
        monkeypatch.setenv("ANSWER_RESULTS_MAX_COLUMNS", "40")
        try:
            reloaded = importlib.reload(config)
            assert reloaded.answer_results_max_rows == 500
            assert reloaded.answer_results_max_columns == 40
        finally:
            monkeypatch.undo()
            importlib.reload(config)

    def test_raised_row_threshold_keeps_json(self, monkeypatch):
        from langchain_timbr import config

        monkeypatch.setattr(config, "answer_results_max_rows", 500)
        rows = _dict_rows(100, 3)
        assert format_results_for_prompt(rows) is rows

    def test_lowered_row_threshold_converts_small_result(self, monkeypatch):
        from langchain_timbr import config

        monkeypatch.setattr(config, "answer_results_max_rows", 2)
        result = format_results_for_prompt(_dict_rows(3, 3))
        assert result.startswith("Query results in CSV format (3 rows, 3 columns")

    def test_raised_column_threshold_keeps_json(self, monkeypatch):
        from langchain_timbr import config

        monkeypatch.setattr(config, "answer_results_max_columns", 40)
        rows = _dict_rows(5, 20)
        assert format_results_for_prompt(rows) is rows

    def test_lowered_column_threshold_converts_narrow_result(self, monkeypatch):
        from langchain_timbr import config

        monkeypatch.setattr(config, "answer_results_max_columns", 2)
        result = format_results_for_prompt(_dict_rows(5, 3))
        assert result.startswith("Query results in CSV format (5 rows, 3 columns")

    def test_unreadable_config_falls_back_to_defaults(self, monkeypatch):
        from langchain_timbr import config

        monkeypatch.delattr(config, "answer_results_max_rows")
        rows = _dict_rows(51, 3)
        assert format_results_for_prompt(rows).startswith("Query results in CSV format")


class TestCsvContent:

    def test_header_and_rows(self):
        rows = _dict_rows(60, 3)
        _, records = _parse_csv(format_results_for_prompt(rows))
        assert records[0] == ["col_0", "col_1", "col_2"]
        assert records[1] == ["r0c0", "r0c1", "r0c2"]
        assert records[60] == ["r59c0", "r59c1", "r59c2"]
        assert len(records) == 61

    def test_union_of_keys_with_missing_values(self):
        rows = [{"a": i} for i in range(60)]
        rows[10]["b"] = "extra"
        _, records = _parse_csv(format_results_for_prompt(rows))
        assert records[0] == ["a", "b"]
        assert records[1] == ["0", ""]
        assert records[11] == ["10", "extra"]

    def test_none_becomes_empty_cell(self):
        rows = [{"a": None, "b": 1} for _ in range(60)]
        _, records = _parse_csv(format_results_for_prompt(rows))
        assert records[1] == ["", "1"]

    def test_special_characters_round_trip(self):
        rows = [{"a": 'has, comma and "quotes"', "b": "line\nbreak"} for _ in range(60)]
        _, records = _parse_csv(format_results_for_prompt(rows))
        assert records[1] == ['has, comma and "quotes"', "line\nbreak"]

    def test_nested_values_become_compact_json(self):
        rows = [{"a": {"x": [1, 2]}} for _ in range(60)]
        _, records = _parse_csv(format_results_for_prompt(rows))
        assert records[1] == ['{"x":[1,2]}']

    def test_list_rows_have_no_header(self):
        rows = [[i, f"name_{i}"] for i in range(60)]
        payload = format_results_for_prompt(rows)
        assert "first line holds the column names" not in payload
        _, records = _parse_csv(payload)
        assert records[0] == ["0", "name_0"]
        assert len(records) == 60


class TestSerializedInput:

    def test_json_string_converted(self):
        rows = _dict_rows(60, 3)
        _, records = _parse_csv(format_results_for_prompt(json.dumps(rows)))
        assert records[0] == ["col_0", "col_1", "col_2"]
        assert len(records) == 61

    def test_python_repr_string_converted(self):
        rows = _dict_rows(60, 3)
        _, records = _parse_csv(format_results_for_prompt(str(rows)))
        assert records[0] == ["col_0", "col_1", "col_2"]
        assert len(records) == 61

    def test_small_json_string_unchanged(self):
        payload = json.dumps(_dict_rows(3, 3))
        assert format_results_for_prompt(payload) is payload


class _Response:
    """Minimal LLM response: only the attributes answer_question reads."""

    def __init__(self, content):
        self.content = content


class TestAnswerQuestionIntegration:

    def test_answer_question_sends_csv(self):
        from langchain_timbr.utils import timbr_llm_utils

        rows = _dict_rows(60, 3)
        prompt_template = Mock()
        prompt_template.format_messages.return_value = ["prompt"]
        llm = Mock()
        llm._llm_type = "openai"
        response = _Response("the answer")

        with patch.object(timbr_llm_utils, 'get_qa_prompt_template', return_value=prompt_template), \
             patch.object(timbr_llm_utils, '_calculate_token_count', return_value=1), \
             patch.object(timbr_llm_utils, '_call_llm_with_timeout', return_value=response):
            result = timbr_llm_utils.answer_question(
                question="how many?",
                llm=llm,
                conn_params={"url": "http://test", "token": "test"},
                results=rows,
            )

        assert result["answer"] == "the answer"
        formatted_rows = prompt_template.format_messages.call_args.kwargs["formatted_rows"]
        assert formatted_rows.startswith("Query results in CSV format (60 rows, 3 columns")

    def test_answer_question_keeps_small_results_as_is(self):
        from langchain_timbr.utils import timbr_llm_utils

        rows = _dict_rows(5, 3)
        prompt_template = Mock()
        prompt_template.format_messages.return_value = ["prompt"]
        llm = Mock()
        llm._llm_type = "openai"
        response = _Response("the answer")

        with patch.object(timbr_llm_utils, 'get_qa_prompt_template', return_value=prompt_template), \
             patch.object(timbr_llm_utils, '_calculate_token_count', return_value=1), \
             patch.object(timbr_llm_utils, '_call_llm_with_timeout', return_value=response):
            timbr_llm_utils.answer_question(
                question="how many?",
                llm=llm,
                conn_params={"url": "http://test", "token": "test"},
                results=rows,
            )

        assert prompt_template.format_messages.call_args.kwargs["formatted_rows"] is rows
