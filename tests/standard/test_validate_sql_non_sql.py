"""``validate_sql`` must report non-SQL text as invalid, not raise.

When the LLM answered in prose ("I cannot answer this..."), the text reached
``validate_sql`` as the SQL. The prefix-stripping fallback did
``sql.index("SELECT")``, which raised ``ValueError: substring not found``. That
skipped the validation retry and failed the whole chain with
"Error executing the chain: LLM call failed: substring not found".

No DB or LLM: ``_validate`` (the EXPLAIN round-trip) and the SQL generator are stubbed.
"""

from __future__ import annotations

import pytest

from langchain_timbr.utils import timbr_llm_utils as TLU
from langchain_timbr.utils import timbr_utils as TU


PROSE = "I cannot answer this question with the available columns"
GOOD_SQL = "SELECT COUNT(*) FROM dtimbr.`shipment`"


def _fake_validate(sql, conn_params):
    """EXPLAIN succeeds only for text that starts with SELECT."""
    if not sql.upper().startswith("SELECT"):
        raise Exception("syntax error at position 1")
    return True


@pytest.fixture
def fake_validate(monkeypatch):
    monkeypatch.setattr(TU, "_validate", _fake_validate)


@pytest.mark.parametrize("text", [PROSE, "N/A"])
def test_non_sql_text_is_invalid_not_raised(fake_validate, text):
    is_valid, error, sql = TU.validate_sql(text, {})
    assert is_valid is False
    assert error == "syntax error at position 1"
    assert sql == text


def test_leading_text_before_select_is_still_stripped(fake_validate):
    is_valid, error, sql = TU.validate_sql(f"Here is the query: {GOOD_SQL}", {})
    assert (is_valid, error, sql) == (True, None, GOOD_SQL)


def test_validation_retry_regenerates_after_non_sql_answer(fake_validate, monkeypatch):
    """The first answer is prose; the retry loop must run and feed the error back."""
    notes: list[str] = []

    def _fake_generate(**kwargs):
        notes.append(kwargs["note"])
        return {"sql": GOOD_SQL, "error": None, "apx_token_count": 0, "usage_metadata": {}}

    monkeypatch.setattr(TLU, "_generate_sql_with_llm", _fake_generate)
    monkeypatch.setattr(TLU, "get_generate_sql_prompt_template", lambda *a, **k: object())

    class _LLM:
        _llm_type = "fake"

    validation_errors: list = []
    is_valid, error, sql = TLU.handle_validate_generate_sql(
        sql_query=PROSE,
        question="how many shipments?",
        llm=_LLM(),
        conn_params={},
        current_context={},
        retries=2,
        timeout=10,
        debug=False,
        usage_metadata={},
        validation_errors=validation_errors,
    )

    assert (is_valid, error, sql) == (True, None, GOOD_SQL)
    assert len(notes) == 1
    assert PROSE in notes[0] and "syntax error at position 1" in notes[0]
    assert validation_errors == ["syntax error at position 1"]
