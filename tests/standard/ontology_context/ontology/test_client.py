"""Tests for TimbrOntologyClient — the SQL it issues and the headers it sends.

The bulk fetches read whole ``sys_*`` tables. The caller's ``results-limit``
header is ``max_limit``, a cap on how many DATA rows a generated SQL answer may
return (default 100) — applied to a metadata read it does not return less
metadata, it returns wrong metadata: relationships lose their is_mtm flag and
their cardinality, concepts lose their primary keys.
"""

from __future__ import annotations

import pytest

from langchain_timbr.ontology_context.ontology import client as client_mod
from langchain_timbr.ontology_context.ontology.client import TimbrOntologyClient


@pytest.fixture
def captured(monkeypatch):
    """Capture (sql, conn_params) of every run_query the client issues."""
    calls = []

    def fake_run_query(sql, conn_params, *args, **kwargs):
        calls.append({"sql": sql, "conn": conn_params})
        return []

    monkeypatch.setattr(client_mod, "run_query", fake_run_query)
    return calls


def _limit_of(call):
    return (call["conn"].get("additional_headers") or {}).get("results-limit")


class TestMetadataRowLimit:
    def test_relationships_fetch_raises_the_row_cap(self, captured):
        conn = {"url": "u", "token": "t",
                "additional_headers": {"results-limit": "100"}}
        TimbrOntologyClient(conn).fetch_relationships_meta()
        assert _limit_of(captured[0]) == str(client_mod._METADATA_ROW_LIMIT)

    def test_inheritance_fetch_raises_the_row_cap(self, captured):
        conn = {"url": "u", "token": "t",
                "additional_headers": {"results-limit": "100"}}
        TimbrOntologyClient(conn).fetch_inheritance_meta()
        assert _limit_of(captured[0]) == str(client_mod._METADATA_ROW_LIMIT)

    def test_cap_is_set_even_when_the_caller_sends_no_headers(self, captured):
        TimbrOntologyClient({"url": "u", "token": "t"}).fetch_relationships_meta()
        assert _limit_of(captured[0]) == str(client_mod._METADATA_ROW_LIMIT)

    def test_the_callers_conn_params_are_not_mutated(self, captured):
        headers = {"results-limit": "100"}
        conn = {"url": "u", "token": "t", "additional_headers": headers}
        TimbrOntologyClient(conn).fetch_relationships_meta()
        assert headers == {"results-limit": "100"}, "caller's dict was mutated"
        assert conn["additional_headers"] is headers

    def test_other_headers_are_preserved(self, captured):
        conn = {"url": "u", "token": "t",
                "additional_headers": {"results-limit": "100", "x-trace": "abc"}}
        TimbrOntologyClient(conn).fetch_relationships_meta()
        assert captured[0]["conn"]["additional_headers"]["x-trace"] == "abc"

    def test_describe_is_left_alone(self, captured):
        """`describe concept ...` starts with DESC, which run_query already
        exempts — no need to override the caller's headers."""
        conn = {"url": "u", "token": "t",
                "additional_headers": {"results-limit": "100"}}
        TimbrOntologyClient(conn).describe_concept("customer")
        assert _limit_of(captured[0]) == "100"


class TestMetadataSql:
    def test_relationships_select_carries_the_edge_columns(self, captured):
        TimbrOntologyClient({"url": "u"}).fetch_relationships_meta()
        sql = captured[0]["sql"]
        for column in ("target_concept", "transitivity", "is_mtm", "is_inverse",
                       "source_properties", "target_properties", "description"):
            assert column in sql, f"{column} missing — the edge set needs it"

    def test_inheritance_select_carries_primary_keys(self, captured):
        TimbrOntologyClient({"url": "u"}).fetch_inheritance_meta()
        sql = captured[0]["sql"]
        assert "primary_keys" in sql
        assert "inheritance" in sql
