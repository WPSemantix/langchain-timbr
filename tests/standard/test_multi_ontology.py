"""Unit tests for the multiple ontologies bug fix.

These tests cover the case where conn_params['ontology'] is a comma-separated
string (e.g. "ontology_a,ontology_b"), which previously caused a DatabaseError:
  Unknown database 'ontology_a,ontology_b'
"""
import pytest
from unittest.mock import Mock, patch

from langchain_timbr.utils.timbr_utils import run_query, get_timbr_agent_options
from langchain_timbr.utils.timbr_llm_utils import determine_concept

MOCK_URL = "http://test-timbr-url"
MOCK_TOKEN = "test-token"


class TestMultiOntologyFix:
    """Tests for the bug fix that handles comma-separated ontology strings."""

    # ------------------------------------------------------------------ #
    # run_query                                                            #
    # ------------------------------------------------------------------ #

    @patch("langchain_timbr.utils.timbr_utils.timbr_http_connector")
    def test_run_query_multi_ontology_uses_first_ontology(self, mock_connector):
        """run_query must strip to the first ontology when a comma-separated list is given."""
        mock_connector.run_query.return_value = []
        conn_params = {"url": MOCK_URL, "token": MOCK_TOKEN, "ontology": "ontology_a,ontology_b"}

        run_query("SELECT 1", conn_params)

        kwargs = mock_connector.run_query.call_args.kwargs
        assert kwargs["ontology"] == "ontology_a"

    @patch("langchain_timbr.utils.timbr_utils.timbr_http_connector")
    def test_run_query_does_not_mutate_conn_params(self, mock_connector):
        """run_query must not mutate the caller's conn_params dict."""
        mock_connector.run_query.return_value = []
        conn_params = {"url": MOCK_URL, "token": MOCK_TOKEN, "ontology": "ontology_a,ontology_b"}

        run_query("SELECT 1", conn_params)

        assert conn_params["ontology"] == "ontology_a,ontology_b"

    @patch("langchain_timbr.utils.timbr_utils.timbr_http_connector")
    def test_run_query_multi_ontology_strips_whitespace(self, mock_connector):
        """run_query must strip surrounding whitespace from the first ontology."""
        mock_connector.run_query.return_value = []
        conn_params = {"url": MOCK_URL, "token": MOCK_TOKEN, "ontology": " ontology_a , ontology_b "}

        run_query("SELECT 1", conn_params)

        kwargs = mock_connector.run_query.call_args.kwargs
        assert kwargs["ontology"] == "ontology_a"

    @patch("langchain_timbr.utils.timbr_utils.timbr_http_connector")
    def test_run_query_single_ontology_unchanged(self, mock_connector):
        """run_query must leave a single (non-comma) ontology value unchanged."""
        mock_connector.run_query.return_value = []
        conn_params = {"url": MOCK_URL, "token": MOCK_TOKEN, "ontology": "ontology_a"}

        run_query("SELECT 1", conn_params)

        kwargs = mock_connector.run_query.call_args.kwargs
        assert kwargs["ontology"] == "ontology_a"

    # ------------------------------------------------------------------ #
    # get_timbr_agent_options                                              #
    # ------------------------------------------------------------------ #

    @patch("langchain_timbr.utils.timbr_utils.timbr_http_connector")
    def test_get_timbr_agent_options_uses_system_db_for_multi_ontology(self, mock_connector):
        """get_timbr_agent_options must query system_db when multiple ontologies are present."""
        mock_connector.run_query.return_value = [
            {"option_name": "ontology", "option_value": "ontology_a,ontology_b"}
        ]
        conn_params = {"url": MOCK_URL, "token": MOCK_TOKEN, "ontology": "ontology_a,ontology_b"}

        get_timbr_agent_options("my_agent", conn_params)

        kwargs = mock_connector.run_query.call_args.kwargs
        assert kwargs["ontology"] == "system_db"

    @patch("langchain_timbr.utils.timbr_utils.timbr_http_connector")
    def test_get_timbr_agent_options_single_ontology_unchanged(self, mock_connector):
        """get_timbr_agent_options must pass the single ontology through unchanged."""
        mock_connector.run_query.return_value = [
            {"option_name": "ontology", "option_value": "ontology_a"}
        ]
        conn_params = {"url": MOCK_URL, "token": MOCK_TOKEN, "ontology": "ontology_a"}

        get_timbr_agent_options("my_agent", conn_params)

        kwargs = mock_connector.run_query.call_args.kwargs
        assert kwargs["ontology"] == "ontology_a"

    # ------------------------------------------------------------------ #
    # determine_concept                                                    #
    # ------------------------------------------------------------------ #

    @patch("langchain_timbr.utils.timbr_llm_utils.get_concepts")
    @patch("langchain_timbr.utils.timbr_llm_utils.get_ontology_description")
    @patch("langchain_timbr.utils.timbr_llm_utils.get_tags")
    @patch("langchain_timbr.utils.timbr_llm_utils.get_determine_concept_prompt_template")
    def test_determine_concept_prompt_template_uses_single_ontology(
        self, mock_get_prompt, mock_get_tags, mock_get_ontology_desc, mock_get_concepts
    ):
        """get_determine_concept_prompt_template must be called with a single-ontology
        conn_param, not the raw comma-separated string."""
        mock_get_prompt.return_value = Mock()
        mock_get_tags.return_value = {"concept_tags": {}, "view_tags": {}}
        mock_get_ontology_desc.return_value = ("", "")
        # Only ontology_a returns a concept; ontology_b is empty
        mock_get_concepts.side_effect = lambda conn_params=None, **_: (
            {"city": {"concept": "city", "description": "a city", "is_view": "false"}}
            if conn_params and conn_params.get("ontology") == "ontology_a"
            else {}
        )

        conn_params = {"url": MOCK_URL, "token": MOCK_TOKEN, "ontology": "ontology_a,ontology_b"}
        determine_concept("What are the cities?", Mock(), conn_params)

        # Prompt template must be built exactly once (not once per ontology)
        mock_get_prompt.assert_called_once()
        # The call must use a single-ontology conn_param (first one in the list)
        called_conn_params = mock_get_prompt.call_args[0][0]
        assert called_conn_params["ontology"] == "ontology_a"
        assert "," not in called_conn_params["ontology"]
