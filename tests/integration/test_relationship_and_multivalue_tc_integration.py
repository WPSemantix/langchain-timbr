"""Technical context for many-to-many relationship properties and multi-value
properties on timbr_crunchbase_llm_tests.

- ``has_employee[person]_title``: a property of the relationship itself, profiled
  under the relationship's mapping (``map_person_works_at_organization``).
- ``company.t1``: a multi-value property, profiled under its own mapping
  (``map_company_t1``).

Neither mapping is listed in sys_concept_mappings, so both need their own lookup.
"""

from unittest.mock import patch

import pytest

import langchain_timbr.technical_context as tc_module
from langchain_timbr import ExecuteTimbrQueryChain, GenerateTimbrSqlChain
from langchain_timbr.ontology_context.ontology.shared import reset_shared_ontologies
from langchain_timbr.technical_context.statistics_loader import load_column_statistics

ONTOLOGY = "timbr_crunchbase_llm_tests"


def _conn_params(config):
    return {
        "url": config["timbr_url"],
        "token": config["timbr_token"],
        "ontology": ONTOLOGY,
        "verify_ssl": config["verify_ssl"],
    }


def _top_values(stats):
    return [str(e.value) for e in stats.top_k or []]


class TestStatistics:
    """The loader reads both columns' statistics from their own mappings."""

    @pytest.mark.parametrize("concept", ["organization", "company"])
    def test_relationship_property_title(self, config, concept):
        # has_employee is declared on organization; company inherits it.
        col = "has_employee[person]_title"
        stats = load_column_statistics(
            schema="dtimbr",
            table_name=concept,
            columns=[{"name": col, "type": "varchar"}],
            conn_params=_conn_params(config),
        )[col]
        assert stats.contributing_mappings == ["map_person_works_at_organization"]
        assert "CEO" in _top_values(stats)

    def test_multi_value_property_t1(self, config):
        stats = load_column_statistics(
            schema="dtimbr",
            table_name="company",
            columns=[{"name": "t1", "type": "varchar"}],
            conn_params=_conn_params(config),
        )["t1"]
        assert "map_company_t1" in stats.contributing_mappings
        assert "2013" in _top_values(stats)


def _run_chain(llm, config, prompt, mode, chain_cls=ExecuteTimbrQueryChain, **kwargs):
    """Invoke the chain and return (result, every technical-context annotation built)."""
    annotations: dict = {}
    real = tc_module.build_technical_context
    # The dynamic pipeline memoizes its rendered prompt per question, and the
    # technical-context mode is not part of that key: without a reset, a repeat
    # question would reuse an earlier test's annotations and never build its own.
    reset_shared_ontologies()

    def spy(*args, **kw):
        out = real(*args, **kw)
        print(f"[mode={mode}] technical context:", out.metadata)
        annotations.update(out.column_annotations or {})
        return out

    chain = chain_cls(
        llm=llm,
        url=config["timbr_url"],
        token=config["timbr_token"],
        ontology=ONTOLOGY,
        views_list="None",
        verify_ssl=config["verify_ssl"],
        enable_technical_context=True,
        technical_context_mode=mode,
        technical_context_max_tokens=3000,
        **kwargs,
    )
    with patch.object(tc_module, "build_technical_context", spy):
        result = chain.invoke({"prompt": prompt})
    print(f"[mode={mode}] sql:", result.get("sql"))
    print(f"[mode={mode}] annotated:", sorted(annotations))
    return result, annotations


class TestQueryGeneration:
    """End to end: the column reaches the prompt with its statistics, and the SQL uses it."""

    @pytest.mark.parametrize("metadata_context_mode", ["dynamic", "static"])
    @pytest.mark.parametrize("mode", ["include_all", "filter_matched", "auto"])
    def test_title_ceo(self, llm, config, mode, metadata_context_mode):
        result, annotations = _run_chain(
            llm, config,
            "find persons that works at software company and have title ceo",
            mode, metadata_context_mode=metadata_context_mode,
        )
        sql = result["sql"] or ""
        assert "]_title" in sql, f"SQL should use the relationship's title, got: {sql}"
        assert "ceo" in sql.lower(), f"SQL should filter on CEO, got: {sql}"
        title_keys = [k for k in annotations if k.endswith("]_title")]
        assert title_keys, f"No technical context for a ]_title column, annotated: {sorted(annotations)}"
        assert any("CEO" in annotations[k] for k in title_keys), {k: annotations[k] for k in title_keys}

    @pytest.mark.parametrize("mode", ["include_all", "filter_matched", "auto"])
    def test_t1_in_2021(self, llm, config, mode):
        # Generate only: the t1 mapping fans out to tens of millions of rows, and a
        # value that does not exist (2021) makes the query scan all of them.
        result, annotations = _run_chain(
            llm, config, "Find companies with t1 in 2021", mode,
            chain_cls=GenerateTimbrSqlChain, concepts_list="company",
        )
        sql = result["sql"] or ""
        assert "t1" in sql, f"SQL should filter on t1, got: {sql}"
        assert "t1" in annotations, f"No technical context for t1, annotated: {sorted(annotations)}"
