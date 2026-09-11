"""The results-limit header must never truncate the library's metadata reads.

``results-limit`` is ``max_limit`` — a cap on how many DATA rows a generated SQL
answer may return. Applied to a ``sys_*`` read it does not return less metadata,
it returns wrong metadata: a missing sys_concept_relationships row costs a
relationship its is_mtm flag and therefore its cardinality.

run_query already exempted metadata reads, but decided via ``'.SYS' in query``,
which the backtick-quoted form ``timbr`.`sys_x`` defeats.
"""

from __future__ import annotations

import pytest

from langchain_timbr.utils import timbr_utils


@pytest.fixture
def sent(monkeypatch):
    """Capture the conn params run_query actually sends downstream."""
    calls = []

    def fake_send(**kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(timbr_utils, "_send_query", fake_send)
    return calls


def _limit(call):
    return (call.get("additional_headers") or {}).get("results-limit")


CONN = {"url": "u", "token": "t", "additional_headers": {"results-limit": "100"}}

BACKTICKED = "SELECT relationship_name FROM `timbr`.`SYS_CONCEPT_RELATIONSHIPS`"
UNQUOTED = "SELECT relationship_name FROM timbr.SYS_CONCEPT_RELATIONSHIPS"


class TestMetadataReadsAreNotCapped:
    @pytest.mark.parametrize("sql", [
        BACKTICKED,
        UNQUOTED,
        'SELECT * FROM "timbr"."sys_ontology"',
        "SELECT * FROM [timbr].[sys_ontology]",
        "SELECT concept, inheritance, primary_keys FROM `timbr`.`sys_ontology`",
        "SELECT property_name, description FROM `timbr`.`SYS_PROPERTIES` "
        "WHERE description is not null",
    ])
    def test_sys_reads_drop_the_limit_however_the_schema_is_quoted(self, sent, sql):
        timbr_utils.run_query(sql, dict(CONN))
        assert _limit(sent[0]) is None, f"results-limit survived on: {sql}"

    def test_show_and_describe_still_exempt(self, sent):
        timbr_utils.run_query("SHOW VERSION", dict(CONN))
        timbr_utils.run_query("describe concept `dtimbr`.`customer`", dict(CONN))
        assert _limit(sent[0]) is None
        assert _limit(sent[1]) is None

    def test_the_callers_conn_params_are_not_mutated(self, sent):
        headers = {"results-limit": "100"}
        conn = {"url": "u", "token": "t", "additional_headers": headers}
        timbr_utils.run_query(BACKTICKED, conn)
        assert headers == {"results-limit": "100"}


class TestDataQueriesStayCapped:
    def test_ordinary_query_keeps_the_limit(self, sent):
        timbr_utils.run_query("SELECT * FROM `dtimbr`.`deliveries`", dict(CONN))
        assert _limit(sent[0]) == "100"

    def test_use_query_limit_keeps_the_cap_even_on_a_sys_table(self, sent):
        """Generated SQL runs with use_query_limit=True. It must stay capped
        whatever tables it names — the exemption is for the library's own reads."""
        timbr_utils.run_query(BACKTICKED, dict(CONN), use_query_limit=True)
        assert _limit(sent[0]) == "100"

    def test_a_column_named_like_sys_does_not_exempt_a_data_query(self, sent):
        timbr_utils.run_query(
            "SELECT t.system_id FROM `dtimbr`.`deliveries` t", dict(CONN))
        assert _limit(sent[0]) == "100"
