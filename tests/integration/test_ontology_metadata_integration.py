"""Integration tests for the `ontology_metadata` NL2SQL concept (Appendix C).

These are behavioral, read-only, on-ontology tests that exercise the metadata
path end to end against a live Timbr backend (connection params sourced from the
environment / launch.json). They verify:

  * metadata questions route to the `ontology_metadata` concept (T1)
  * data questions do NOT over-trigger the metadata concept (T2)
  * generate-SQL emits the real sentinel query, and execute expands it into a
    flat, section-tagged rows[] (T3)
  * the generic answer chain consumes the assembled ontology definition (T4)
  * with a Data Agent, metadata routes to the correct single ontology (T5)

The feature is gated by `enable_ontology_questions` (default False), so every
chain here is constructed with `enable_ontology_questions=True`.
"""

import pytest

from langchain_timbr import (
    IdentifyTimbrConceptChain,
    GenerateTimbrSqlChain,
    ExecuteTimbrQueryChain,
    GenerateAnswerChain,
)
from langchain_timbr.llm_wrapper.llm_wrapper import LlmWrapper


# --------------------------------------------------------------------------- #
# constants / anchors (see Appendix C.1)
# --------------------------------------------------------------------------- #
META = "ontology_metadata"
SENTINEL = "SELECT * FROM timbr.sys_ontology"
AGENT = "llm_test_agent"


def _rows_blob(rows) -> str:
    """Serialize all rows into one lowercase string for tolerant token presence checks."""
    return " ".join(str(r) for r in (rows or [])).lower()


# --------------------------------------------------------------------------- #
# T1 — metadata questions route to `ontology_metadata`
# --------------------------------------------------------------------------- #
class TestOntologyMetadataRouting:
    """Explicit metadata questions must select the `ontology_metadata` concept alone."""

    @pytest.mark.parametrize(
        "prompt",
        [
            "What is the ontology definition for total revenue measure?",  # measure-level
            "Explain the ontology",                # whole-ontology
        ],
    )
    def test_metadata_question_routes_to_ontology_metadata(self, llm, config, prompt):
        chain = IdentifyTimbrConceptChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            verify_ssl=config["verify_ssl"],
            enable_ontology_questions=True,
        )
        result = chain.invoke({"prompt": prompt})
        print("T1 IdentifyTimbrConceptChain result:", result)

        assert result.get("concept") == META, \
            f"Metadata question should route to '{META}', got '{result.get('concept')}'"
        assert result.get("identify_concept_reason"), "Reason should not be empty"
        usage = result.get(chain.usage_metadata_key)
        assert usage and set(usage.keys()) == {"determine_concept"}, \
            "Usage metadata should contain only 'determine_concept'"

    # ----------------------------------------------------------------------- #
    # T2 — negative control: a data question must NOT route to metadata
    # ----------------------------------------------------------------------- #
    def test_data_question_does_not_route_to_metadata(self, llm, config):
        chain = IdentifyTimbrConceptChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            verify_ssl=config["verify_ssl"],
            enable_ontology_questions=True,
        )
        result = chain.invoke({"prompt": "How many customers do we have?"})
        print("T2 IdentifyTimbrConceptChain result:", result)

        concept = result.get("concept")
        assert concept, "A data question should still resolve to a concept"
        assert concept != META, \
            f"Data question must NOT route to '{META}' (over-trigger), got '{concept}'"


# --------------------------------------------------------------------------- #
# T3 / T4 — sentinel + expansion, and answer consumes the definition
# --------------------------------------------------------------------------- #
class TestOntologyMetadataExecution:
    """The generate→execute→answer path for the metadata concept."""

    def test_sentinel_and_expansion(self, llm, config):
        # generate-SQL short-circuits to the real sentinel query for the metadata concept
        gen_chain = GenerateTimbrSqlChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            concept=META,
            verify_ssl=config["verify_ssl"],
            enable_ontology_questions=True,
        )
        gen_result = gen_chain.invoke({"prompt": "Explain the ontology"})
        print("T3 GenerateTimbrSqlChain result:", gen_result)
        assert gen_result.get("sql") == SENTINEL, \
            f"Metadata concept should emit the sentinel query, got '{gen_result.get('sql')}'"

        # execute recognizes the metadata concept and expands into concept-centric rows[]
        exec_chain = ExecuteTimbrQueryChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            concept=META,
            verify_ssl=config["verify_ssl"],
            enable_ontology_questions=True,
        )
        exec_result = exec_chain.invoke({"prompt": "Explain the ontology"})
        rows = exec_result.get("rows")
        print("T3 ExecuteTimbrQueryChain rows count:", len(rows or []))

        assert rows, "Execute should return a non-empty metadata rows[]"
        assert any("concept" in r for r in rows), "Metadata rows should include pivoted concept objects"

        blob = _rows_blob(rows)
        assert "customer" in blob, "Assembled definition should include the 'customer' concept"
        assert "total_revenue" in blob, "Assembled definition should include the 'total_revenue' measure"

    def test_answer_consumes_ontology_definition(self, llm, config):
        exec_chain = ExecuteTimbrQueryChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            concept=META,
            verify_ssl=config["verify_ssl"],
            enable_ontology_questions=True,
        )
        exec_result = exec_chain.invoke({"prompt": "What concepts are available?"})
        rows = exec_result.get("rows")

        blob = _rows_blob(rows)
        assert "customer" in blob, "Definition rows should contain the 'customer' concept"
        assert "shipment" in blob, "Definition rows should contain the 'shipment' concept"

        answer_chain = GenerateAnswerChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
        )
        answer_result = answer_chain.invoke({
            "prompt": "What concepts are available?",
            "rows": rows,
            "sql": exec_result.get("sql"),
        })
        answer = answer_result.get("answer")
        print("T4 GenerateAnswerChain answer:", answer)

        assert answer, "Answer should not be empty"
        answer_lc = answer.lower()
        assert any(tok in answer_lc for tok in ("customer", "order", "shipment", "product")), \
            "Answer should mention at least one of the ontology's concepts"


# --------------------------------------------------------------------------- #
# T5 — agent routes metadata to the correct single ontology
# --------------------------------------------------------------------------- #
class TestOntologyMetadataAgentRouting:
    """With a Data Agent, metadata questions return `<ontology>.ontology_metadata`."""

    @pytest.mark.parametrize(
        "prompt,expected_concept",
        [
            ("Explain the shipment concept metadata", "supply_metrics.ontology_metadata"),
            ("What measures exists in supply chain?", "supply_metrics.ontology_metadata"),
            ("What are the patient concept relationships?", "timbr_patient_journey.ontology_metadata"),
            ("Describe the google analytics ontology", "google_analytics_campaigns.ontology_metadata"),
        ],
    )
    def test_agent_routes_metadata_to_correct_ontology(self, llm, config, prompt, expected_concept):
        
        chain = IdentifyTimbrConceptChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            agent=AGENT,
            verify_ssl=config["verify_ssl"],
            enable_ontology_questions=True,
        )
        result = chain.invoke({"prompt": prompt})
        print("T5 IdentifyTimbrConceptChain result:", result)

        assert result.get("ontology") + "." + result.get("concept") == expected_concept, \
            f"Expected '{expected_concept}', got '{result.get('ontology')}.{result.get('concept')}'"
        assert result.get("identify_concept_reason"), "Reason should not be empty"
