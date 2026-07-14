"""Tests for ontology_metadata.py — the single support module for the
ontology_metadata NL2SQL concept."""

from langchain_timbr import ontology_metadata as om

# --------------------------------------------------------------------------- #
# concept detection / ontology parsing
# --------------------------------------------------------------------------- #
def test_is_metadata_concept():
    assert om.is_metadata_concept("ontology_metadata")
    assert om.is_metadata_concept("supply_metrics.ontology_metadata")
    assert not om.is_metadata_concept("order")
    assert not om.is_metadata_concept("")
    assert not om.is_metadata_concept(None)
    # must not match a mere substring / suffix without the dot separator
    assert not om.is_metadata_concept("my_ontology_metadata_extra")

def test_parse_ontology():
    assert om.parse_ontology("supply_metrics.ontology_metadata", "def") == "supply_metrics"
    assert om.parse_ontology("ontology_metadata", "def") == "def"
    # multi-dot ontology names survive
    assert om.parse_ontology("a.b.ontology_metadata", "def") == "a.b"

# --------------------------------------------------------------------------- #
# config gate (default OFF for backward compatibility)
# --------------------------------------------------------------------------- #
def test_is_enabled_default_false():
    assert om.is_enabled(None) is False
    assert om.is_enabled({}) is False
    assert om.is_enabled({"enable_ontology_questions": False}) is False

def test_is_enabled_true():
    assert om.is_enabled({"enable_ontology_questions": True}) is True

def test_prompt_lines_gated():
    assert om.prompt_lines(None) == ""
    assert om.prompt_lines({}) == ""
    out = om.prompt_lines({"enable_ontology_questions": True})
    assert "ontology_metadata" in out
    assert "metadata model-introspection questions" in out

# --------------------------------------------------------------------------- #
# scope detection
# --------------------------------------------------------------------------- #
def test_scope_semantic_always_on():
    assert om.scope_of("what measures exist")["semantic"] is True

def test_scope_sources_gated():
    assert om.scope_of("what properties does order have")["sources"] is False
    assert om.scope_of("which table backs shipment")["sources"] is True
    assert om.scope_of("where does the order concept come from")["sources"] is True
    assert om.scope_of("show me the lineage of product")["sources"] is True
    assert om.scope_of("")["sources"] is False

# --------------------------------------------------------------------------- #
# tables parsing (1/2/3-level, comma separated)
# --------------------------------------------------------------------------- #
def test_parse_tables():
    assert om.parse_tables("scdata_demo.customer") == ["scdata_demo.customer"]
    assert om.parse_tables("timbr.customer, timbr.order") == ["timbr.customer", "timbr.order"]
    assert om.parse_tables("cat.sch.tbl, tbl2") == ["cat.sch.tbl", "tbl2"]
    assert om.parse_tables("") == []
    assert om.parse_tables(None) == []

# --------------------------------------------------------------------------- #
# generate_sql short-circuit
# --------------------------------------------------------------------------- #
def test_build_metadata_sql_is_real_query():
    r = om.build_metadata_sql("supply_metrics.ontology_metadata", "list measures", "def")
    assert r["kind"] == "ONTOLOGY_METADATA"
    assert r["ontology"] == "supply_metrics"
    # the emitted SQL must be a valid best-effort query for non-execute callers
    assert r["sql"] == "SELECT * FROM timbr.sys_ontology"
    assert r["scope"]["semantic"] is True

def test_build_metadata_sql_default_ontology():
    r = om.build_metadata_sql("ontology_metadata", "which table backs shipment", "supply_metrics")
    assert r["ontology"] == "supply_metrics"
    assert r["scope"]["sources"] is True

# --------------------------------------------------------------------------- #
# execute branch — fake run_sql / get_version / cache
# --------------------------------------------------------------------------- #
def _fake_run_sql(sql):
    s = sql.upper()
    if "SYS_ONTOLOGY" in s:
        return [
            {"concept": "thing", "inheritance": "", "description": "root", "concept_logic": ""},
            {"concept": "customer", "inheritance": "thing",
             "description": "customer related info", "concept_logic": ""},
            {"concept": "consumer_customer", "inheritance": "customer",
             "description": "",
             "concept_logic": "SELECT * FROM dtimbr.`customer` WHERE customer_segment = 'Consumer'"},
            {"concept": "order", "inheritance": "thing", "description": "", "concept_logic": ""},
        ]
    if "SYS_PROPERTIES" in s:
        return [
            {"property_name": "customer_id", "is_measure": "false", "logic_query": ""},
            {"property_name": "customer_email", "is_measure": "false", "logic_query": ""},
            {"property_name": "customer_name", "is_measure": "false", "logic_query": ""},
            {"property_name": "count_of_customer", "is_measure": "true",
             "logic_query": "SELECT COUNT(DISTINCT customer_id)"},
        ]
    if "SYS_CONCEPT_PROPERTIES" in s and "IS NOT NULL" in s:
        return []  # multivalue: none in this fixture
    if "SYS_CONCEPT_PROPERTIES" in s:
        return [
            {"concept": "customer", "property_name": "customer_id"},
            {"concept": "customer", "property_name": "customer_email"},
            {"concept": "customer", "property_name": "customer_name"},
            {"concept": "customer", "property_name": "count_of_customer"},
        ]
    if "SYS_CONCEPT_RELATIONSHIPS" in s and "IS NOT NULL" in s:
        return []
    if "SYS_CONCEPT_RELATIONSHIPS" in s:
        return [
            {"concept": "customer", "relationship_name": "made_order", "target_concept": "order"},
            {"concept": "customer", "relationship_name": "received_shipment",
             "target_concept": "shipment"},
        ]
    if "SYS_VIEWS" in s:
        return [
            {"view_name": "customer_360", "is_cube": "false",
             "tables": "timbr.customer", "description": "360 view"},
            {"view_name": "sales_cube", "is_cube": "true", "tables": "", "description": ""},
        ]
    if "SYS_CONCEPT_MAPPINGS" in s:
        return [
            {"concept": "customer", "tables": "scdata_demo.customer", "datasource_id": "mysql"},
            {"concept": "consumer_customer", "tables": "", "datasource_id": "mysql"},
        ]
    return []

class _Cache:
    def __init__(self):
        self.store = {}
        self.sets = 0

    def get(self, k):
        return self.store.get(k)

    def set(self, k, v):
        self.store[k] = v
        self.sets += 1

def test_fetch_pivots_concepts():
    rows = om.fetch_metadata_rows(
        "supply_metrics", {"semantic": True, "sources": False},
        run_sql=_fake_run_sql, get_version=lambda o: "v1",
    )
    by_name = {r["concept"]: r for r in rows if "concept" in r}
    # root `thing` is dropped
    assert "thing" not in by_name
    cust = by_name["customer"]
    assert cust["description"] == "customer related info"
    assert cust["tables"] == "scdata_demo.customer"
    # non-measure props sorted + comma-joined; measures excluded from properties
    assert cust["properties"] == "customer_email, customer_id, customer_name"
    assert cust["measures"] == "count_of_customer = COUNT(DISTINCT customer_id)"
    assert cust["relationships"] == "made_order -> order, received_shipment -> shipment"
    # subtype: inherits + stripped logic; empty fields are omitted
    sub = by_name["consumer_customer"]
    assert sub["inherits"] == "customer"
    assert sub["logic"] == "customer WHERE customer_segment = 'Consumer'"
    assert "tables" not in sub       # empty mapping omitted
    assert "properties" not in sub
    assert "description" not in sub
    # no empty/None values leak into any object
    for r in rows:
        assert all(v not in (None, "") for v in r.values())

def test_fetch_appends_views():
    rows = om.fetch_metadata_rows(
        "supply_metrics", {"semantic": True, "sources": False},
        run_sql=_fake_run_sql, get_version=lambda o: "v1",
    )
    by_view = {r["view"]: r for r in rows if "view" in r}
    assert by_view["customer_360"]["is_cube"] is False
    assert by_view["customer_360"]["tables"] == "timbr.customer"
    assert by_view["sales_cube"]["is_cube"] is True
    assert "tables" not in by_view["sales_cube"]  # empty tables omitted

def test_fetch_scope_independent():
    semantic = om.fetch_metadata_rows(
        "supply_metrics", {"semantic": True, "sources": False},
        _fake_run_sql, lambda o: "v1",
    )
    with_sources = om.fetch_metadata_rows(
        "supply_metrics", {"semantic": True, "sources": True},
        _fake_run_sql, lambda o: "v1",
    )
    # output is scope-independent; underlying tables always attached
    assert semantic == with_sources
    cust = {r["concept"]: r for r in semantic if "concept" in r}["customer"]
    assert cust["tables"] == "scdata_demo.customer"

def test_fetch_uses_cache_keyed_by_version():
    cache = _Cache()
    scope = {"semantic": True, "sources": False}
    versions = {"n": 0}

    def get_version(o):
        return f"v{versions['n']}"

    r1 = om.fetch_metadata_rows("supply_metrics", scope, _fake_run_sql, get_version, cache)
    r2 = om.fetch_metadata_rows("supply_metrics", scope, _fake_run_sql, get_version, cache)
    assert r1 == r2
    assert cache.sets == 1  # second call served from cache

    # version bump invalidates
    versions["n"] = 1
    om.fetch_metadata_rows("supply_metrics", scope, _fake_run_sql, get_version, cache)
    assert cache.sets == 2

def test_scope_does_not_change_cache_key():
    cache = _Cache()
    om.fetch_metadata_rows("supply_metrics", {"semantic": True, "sources": False},
                           _fake_run_sql, lambda o: "v1", cache)
    om.fetch_metadata_rows("supply_metrics", {"semantic": True, "sources": True},
                           _fake_run_sql, lambda o: "v1", cache)
    assert cache.sets == 1  # scope is no longer part of the cache key
