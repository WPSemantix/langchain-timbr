"""Regression test for the agent/param note de-duplication.

When an agent carries a ``note`` and the identical string is also passed as the
``note`` kwarg, the two must not be concatenated — otherwise the note appears
twice in the LLM prompt. See the ``if note and note != self._note`` guard in the
chain constructors.
"""

from __future__ import annotations

from unittest.mock import patch

from langchain_timbr.langchain.generate_timbr_sql_chain import GenerateTimbrSqlChain


MOCK_URL = "http://test.timbr.ai"
MOCK_TOKEN = "tk_test"


@patch("langchain_timbr.langchain.generate_timbr_sql_chain.get_timbr_agent_options")
def test_identical_agent_and_param_note_not_duplicated(mock_get_options):
    """Agent note == param note → kept once, not concatenated into a duplicate."""
    mock_get_options.return_value = {"note": "Only count active rows."}
    chain = GenerateTimbrSqlChain(
        llm=object(),  # bypass LlmWrapper env-var fetch
        url=MOCK_URL,
        token=MOCK_TOKEN,
        agent="some-agent",
        note="Only count active rows.",  # same string the agent already carries
    )
    assert chain._note == "Only count active rows."
