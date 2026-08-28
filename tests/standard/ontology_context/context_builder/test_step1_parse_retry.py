"""Step 1 planner: recovery when the LLM answers with something other than JSON.

The planner is asked for a concept-selection object and sometimes returns the
finished SQL query instead. Before the retry existed, that discarded the call
and dropped every relationship from the emitted context — costing both a
round-trip and output quality. These tests pin the recovery behaviour:

  - a non-JSON first answer is re-prompted once out of the existing retry budget
  - the correction actually reaches the planner
  - a second non-JSON answer still fails, and a zero retry budget does not retry
"""

from __future__ import annotations

from langchain_timbr.ontology_context.context_builder import build_filtered as _bf
from langchain_timbr.ontology_context.context_builder.metadata_config import (
    MetadataContextConfig,
)

from .test_action_grammar import (
    ScriptedLLM,
    _build_path_payload,
    _simple_ontology,
)

_SQL_INSTEAD_OF_JSON = (
    "SELECT\n  o.o_id,\n  c.name\nFROM `dtimbr`.`customer` c\nJOIN `dtimbr`.`order` o"
)

_GOOD = _build_path_payload(
    selected_concepts=["customer", "order"],
    segments=[{"from": "customer", "rel": "made_order", "to": "order"}],
)


def _config(retry: int):
    return MetadataContextConfig(
        mode="dynamic", max_detail_concepts=100, metadata_context_dynamic_retry=retry,
    )


def _run(llm, cfg):
    return _bf.build_filtered_metadata(
        question="customers and orders",
        anchor="customer",
        ontology=_simple_ontology(),
        llm=llm,
        config=cfg,
        graph_depth=2,
    )


class TestStep1ParseRetry:
    def test_sql_instead_of_json_is_reprompted_and_recovers(self):
        llm = ScriptedLLM([_SQL_INSTEAD_OF_JSON, _GOOD])
        result = _run(llm, _config(retry=2))

        assert len(llm.calls) == 2
        assert result.stats.get("parse_retry_used") is True
        assert result.stats["resolved_by"] == "llm_paths"
        assert result.validated_paths

    def test_correction_reaches_the_planner(self):
        llm = ScriptedLLM([_SQL_INSTEAD_OF_JSON, _GOOD])
        _run(llm, _config(retry=2))

        retry_prompt = llm.calls[1]["system"] + llm.calls[1]["user"]
        assert "could not be parsed" in retry_prompt

    def test_second_failure_still_falls_back(self):
        llm = ScriptedLLM([_SQL_INSTEAD_OF_JSON, _SQL_INSTEAD_OF_JSON])
        result = _run(llm, _config(retry=2))

        assert len(llm.calls) == 2
        assert result.stats["resolved_by"] != "llm_paths"

    def test_zero_retry_budget_does_not_reprompt(self):
        llm = ScriptedLLM([_SQL_INSTEAD_OF_JSON])
        result = _run(llm, _config(retry=0))

        assert len(llm.calls) == 1
        assert result.stats.get("parse_retry_used") is None

    def test_valid_json_first_time_costs_one_call(self):
        llm = ScriptedLLM([_GOOD])
        result = _run(llm, _config(retry=2))

        assert len(llm.calls) == 1
        assert result.stats.get("parse_retry_used") is None
        assert result.stats["resolved_by"] == "llm_paths"
