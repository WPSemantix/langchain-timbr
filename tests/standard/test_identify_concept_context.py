"""Unit tests for the identify-concept context builder and its metadata sourcing.

Everything here runs against pure in-memory fakes — no DB, no LLM. The builder's
DB loader (`_load_catalog`) is monkeypatched to return a catalog assembled from a
fixture row set, so we exercise the real assembly + render + token-ladder logic.
"""

import pytest

from langchain_timbr import ontology_metadata as om
from langchain_timbr import identify_concept_context as icc
from langchain_timbr import config


# --------------------------------------------------------------------------- #
# ontology_metadata catalog-sourcing helpers
# --------------------------------------------------------------------------- #
def test_parse_view_concepts_keeps_only_two_level_semantic_refs():
    tables = "dtimbr.customer, timbr.order, scdata.raw_customers, cat.sch.tbl, timbr.sys_views, dtimbr.customer"
    assert om.parse_view_concepts(tables) == ["customer", "order"]


def test_parse_view_concepts_excludes_thing():
    assert om.parse_view_concepts("timbr.thing") == []
    assert om.parse_view_concepts("timbr.company, timbr.thing") == ["company"]


def test_parse_view_concepts_empty():
    assert om.parse_view_concepts("") == []
    assert om.parse_view_concepts(None) == []


def test_parse_parents_excludes_thing():
    assert om.parse_parents("thing") == []
    assert om.parse_parents("person, thing") == ["person"]
    assert om.parse_parents("") == []


def test_build_inheritance_map():
    rows = [
        {"concept": "thing"},
        {"concept": "person", "inheritance": "thing"},
        {"concept": "customer", "inheritance": "person"},
        {"concept": "vip", "inheritance": "customer"},
        {"concept": "order", "inheritance": "thing"},
    ]
    m = om.build_inheritance_map(rows)
    assert m == {"person": ["customer"], "customer": ["vip"]}


def test_fetch_catalog_rows_isolates_failing_section():
    def run_sql(sql):
        if "SYS_VIEW_PROPERTIES" in sql.upper():
            raise RuntimeError("table missing on this backend")
        if "SYS_ONTOLOGY" in sql.upper():
            return [{"concept": "order"}]
        return []

    out = om.fetch_catalog_rows("ont", run_sql, get_version=lambda o: "v1")
    assert out["concepts"] == [{"concept": "order"}]
    assert out["view_properties"] == []          # failing section -> empty, not fatal
    assert set(out) >= {"concepts", "properties", "concept_properties",
                        "relationships", "views", "view_properties"}


def test_fetch_catalog_rows_uses_cache():
    calls = {"n": 0}

    def run_sql(sql):
        calls["n"] += 1
        return []

    class Cache:
        def __init__(self):
            self.store = {}

        def get(self, k):
            return self.store.get(k)

        def set(self, k, v):
            self.store[k] = v

    cache = Cache()
    om.fetch_catalog_rows("ont", run_sql, lambda o: "v1", cache)
    first = calls["n"]
    om.fetch_catalog_rows("ont", run_sql, lambda o: "v1", cache)
    assert calls["n"] == first  # second call fully served from cache


# --------------------------------------------------------------------------- #
# catalog fixture
# --------------------------------------------------------------------------- #
def _rows():
    return {
        "concepts": [
            {"concept": "thing"},
            {"concept": "person", "inheritance": "thing", "inheritance_level": 1},
            {"concept": "customer", "inheritance": "person", "inheritance_level": 2},
            {"concept": "vip_customer", "inheritance": "customer", "inheritance_level": 3},
            {"concept": "order", "inheritance": "thing", "inheritance_level": 1},
        ],
        "properties": [
            {"property_name": "age_years", "is_measure": "true"},
            {"property_name": "total_spent", "is_measure": "true"},
            {"property_name": "order_amount", "is_measure": "true"},
            {"property_name": "name", "is_measure": "false"},
            {"property_name": "region", "is_measure": "false"},
        ],
        "concept_properties": [
            {"concept": "person", "property_name": "name"},
            {"concept": "person", "property_name": "age_years"},
            {"concept": "customer", "property_name": "total_spent"},
            {"concept": "order", "property_name": "order_amount"},
        ],
        "relationships": [
            {"concept": "customer", "target_concept": "order",
             "relationship_name": "placed_order", "inverse_name": "ordered_by"},
        ],
        "views": [
            {"view_name": "customer_360", "is_cube": "false",
             "tables": "dtimbr.customer, dtimbr.order, scdata.raw_customers, timbr.sys_x"},
        ],
        "view_properties": [
            {"view_name": "customer_360", "property_name": "total_spent"},
            {"view_name": "customer_360", "property_name": "region"},
        ],
    }


@pytest.fixture
def patched_catalog(monkeypatch):
    catalog = icc._build_catalog(_rows())
    monkeypatch.setattr(icc, "_load_catalog", lambda conn_params: catalog)
    return catalog


def _cv(*names):
    """Fake get_concepts output for the given candidate names."""
    view_names = {"customer_360"}
    return {
        n: {"concept": n, "description": f"{n} desc",
            "is_view": "true" if n in view_names else "false"}
        for n in names
    }


def _render(question, cv, **kw):
    return "\n".join(icc.build_catalog_lines(question, {"ontology": "ont"}, cv, **kw))


# --------------------------------------------------------------------------- #
# catalog assembly
# --------------------------------------------------------------------------- #
def test_build_catalog_measures_and_relationships():
    cat = icc._build_catalog(_rows())
    assert cat.nodes["customer"].measures == ["total_spent"]
    assert ("placed_order", "order") in cat.nodes["customer"].relationships
    # inverse relationship is attached to the target automatically
    assert ("ordered_by", "customer") in cat.nodes["order"].relationships
    assert cat.nodes["customer_360"].is_view is True
    assert cat.nodes["customer_360"].connected == ["customer", "order"]
    assert cat.nodes["customer_360"].measures == ["total_spent"]
    assert [p[0] for p in cat.nodes["customer_360"].properties] == ["region"]


# --------------------------------------------------------------------------- #
# rendering: nesting, inlining, never-drop, hints, views
# --------------------------------------------------------------------------- #
def test_parent_present_nests_child_with_inherits_marker(patched_catalog):
    out = _render("list customers", _cv("person", "customer", "order"))
    assert "`customer`" in out
    assert "inherits `person`, independently selectable" in out
    # child is indented under its parent
    assert "\n  - `customer`" in out
    # with parent present, child shows only its DIRECT measure (not inherited age_years)
    cust_line = [l for l in out.splitlines() if "measures:" in l and "total_spent" in l]
    assert cust_line and "age_years" not in cust_line[0]


def test_parent_absent_inlines_inherited_measures(patched_catalog):
    out = _render("list customers", _cv("customer"))
    # person is NOT in the filtered set, so customer inlines the inherited measure
    measure_line = [l for l in out.splitlines() if "measures:" in l][0]
    assert "total_spent" in measure_line
    assert "age_years" in measure_line
    # parent absent: no "independently selectable" nesting marker, but the concept
    # still declares the ancestor it inherits from
    assert "independently selectable" not in out
    assert "inherits `person`" in out


def test_concept_plain_property_inlined_when_parent_absent(patched_catalog):
    # Properties are trigram-filtered to the question, so name it explicitly.
    out = _render("customer name", _cv("customer"))
    # person absent -> customer inlines the inherited plain property `name`
    prop_lines = [l for l in out.splitlines() if "properties" in l]
    assert any("`name`" in l for l in prop_lines)


def test_concept_plain_property_not_repeated_when_parent_present(patched_catalog):
    out = _render("person name and customer", _cv("person", "customer"))
    # `name` is owned by person; with person present, the nested customer must not
    # repeat it (parent-subtraction), so it appears exactly once.
    prop_lines = [l for l in out.splitlines() if "properties" in l and "`name`" in l]
    assert len(prop_lines) == 1


def test_never_drops_candidate_without_metadata(patched_catalog):
    out = _render("anything", _cv("customer", "mystery_concept"))
    assert "`mystery_concept`" in out


def test_inline_subtype_hint_under_parent(patched_catalog):
    out = _render("show me vip customers", _cv("customer"))
    # vip_customer is an out-of-list descendant that trigram-matches the question
    assert "narrow to sub-types:" in out
    assert "`vip_customer`" in out
    hint_line = [l for l in out.splitlines() if "narrow to sub-types" in l][0]
    # hint sits under customer (indented)
    assert hint_line.startswith("    ")


def test_view_renders_connected_concepts_and_props(patched_catalog):
    out = _render("customer region overview", _cv("customer_360"))
    assert "`customer_360` (view)" in out
    assert "connected concepts: `customer`, `order`" in out
    assert "properties (1 of 1 match): `region`" in out
    assert "measures: `total_spent`" in out


def test_cube_measures_come_from_measure_prefix():
    # Cubes flag their measure axis with a `measure.` column prefix (is_measure is
    # not server-filterable for cubes). The prefix is stripped in the render and the
    # non-prefixed columns fall through to plain properties.
    rows = _rows()
    rows["views"].append({"view_name": "order_cube", "is_cube": "true",
                          "tables": "timbr.order, timbr.customer"})
    rows["view_properties"] += [
        {"view_name": "order_cube", "property_name": "measure.total_revenue"},
        {"view_name": "order_cube", "property_name": "measure.count_of_order"},
        {"view_name": "order_cube", "property_name": "order_date"},
    ]
    cat = icc._build_catalog(rows)
    cube = cat.nodes["order_cube"]
    assert cube.is_cube is True
    assert cube.measures == ["total_revenue", "count_of_order"]
    assert [p[0] for p in cube.properties] == ["order_date"]
    # bare (unprefixed) measure names are rendered, not the `measure.` form
    assert "measure." not in ", ".join(cube.measures)


def test_multi_ontology_prefix(patched_catalog):
    out = _render("list customers", _cv("customer"), prefix="`sales`.")
    assert "`sales`.`customer`" in out


# --------------------------------------------------------------------------- #
# token ladder
# --------------------------------------------------------------------------- #
def test_ladder_truncates_then_drops_descriptions(monkeypatch, patched_catalog):
    cv = {
        "customer": {"concept": "customer", "is_view": "false",
                     "description": "x" * 4000},
        "order": {"concept": "order", "is_view": "false",
                  "description": "y" * 4000},
    }
    # Force the desc-trim ceiling to bite so the ladder must truncate/drop.
    monkeypatch.setattr(config, "identify_concept_context_desc_trim_tokens", 50)
    monkeypatch.setattr(config, "identify_concept_context_rel_trim_tokens", 80)
    monkeypatch.setattr(config, "identify_concept_context_hard_limit_tokens", 100000)
    out = _render("customers and orders", cv)
    # long descriptions must have been compressed away from full length
    assert "x" * 4000 not in out
    assert "y" * 4000 not in out


def test_ladder_hard_limit_never_fails_and_keeps_all_names(monkeypatch, patched_catalog):
    # The names-only floor is the trimming terminus, not an error gate: even with
    # every ceiling forced to 1 token the builder returns (no raise) and every
    # candidate name stays present.
    cv = _cv("customer", "order", "person")
    for attr in ("identify_concept_context_desc_trim_tokens",
                 "identify_concept_context_rel_trim_tokens",
                 "identify_concept_context_hard_limit_tokens"):
        monkeypatch.setattr(config, attr, 1)
    out = _render("anything", cv)
    for name in ("customer", "order", "person"):
        assert f"`{name}`" in out


# --------------------------------------------------------------------------- #
# _type_of_ discriminator columns are dropped from both axes (§1)
# --------------------------------------------------------------------------- #
def test_type_of_columns_excluded_from_concept_and_view():
    rows = _rows()
    rows["concept_properties"].append({"concept": "person", "property_name": "_type_of_person"})
    rows["view_properties"].append({"view_name": "customer_360", "property_name": "_type_of_customer"})
    cat = icc._build_catalog(rows)
    person_props = [p[0] for p in cat.nodes["person"].properties]
    view_props = [p[0] for p in cat.nodes["customer_360"].properties]
    assert "_type_of_person" not in person_props
    assert "_type_of_person" not in cat.nodes["person"].measures
    assert "_type_of_customer" not in view_props


def test_type_of_cube_measure_prefix_excluded():
    rows = _rows()
    rows["views"].append({"view_name": "c", "is_cube": "true", "tables": "timbr.order"})
    rows["view_properties"].append({"view_name": "c", "property_name": "measure._type_of_order"})
    cat = icc._build_catalog(rows)
    assert cat.nodes["c"].measures == []


# --------------------------------------------------------------------------- #
# property rendering: filtered count, omission, survives every ladder stage (§1)
# --------------------------------------------------------------------------- #
def _station_catalog(monkeypatch):
    rows = {
        "concepts": [
            {"concept": "thing"},
            {"concept": "station", "inheritance": "thing", "inheritance_level": 1},
        ],
        "properties": [
            {"property_name": "lrt", "is_measure": "false"},
            {"property_name": "platform_count", "is_measure": "false"},
        ],
        "concept_properties": [
            {"concept": "station", "property_name": "lrt"},
            {"concept": "station", "property_name": "platform_count"},
        ],
        "relationships": [],
        "views": [],
        "view_properties": [],
    }
    catalog = icc._build_catalog(rows)
    monkeypatch.setattr(icc, "_load_catalog", lambda conn_params: catalog)
    return catalog


def test_property_line_reports_filtered_count(monkeypatch):
    _station_catalog(monkeypatch)
    out = _render("average lrt per station", _cv("station"))
    prop_line = [l for l in out.splitlines() if "properties" in l][0]
    assert "properties (1 of 2 match): `lrt`" in prop_line
    assert "platform_count" not in prop_line


def test_property_line_omitted_when_nothing_matches(monkeypatch):
    _station_catalog(monkeypatch)
    out = _render("filtration systems overview", _cv("station"))
    assert "properties" not in out


def test_short_name_property_matches_word_boundary(monkeypatch):
    _station_catalog(monkeypatch)
    # `lrt` (one trigram) is matched via the whole-word fallback, not substring.
    assert "`lrt`" in _render("the LRT-line schedule", _cv("station"))
    assert "`lrt`" not in _render("filtration plant", _cv("station"))


def test_property_survives_collapse_stage(monkeypatch):
    _station_catalog(monkeypatch)
    for attr in ("identify_concept_context_desc_trim_tokens",
                 "identify_concept_context_rel_trim_tokens"):
        monkeypatch.setattr(config, attr, 1)
    monkeypatch.setattr(config, "identify_concept_context_hard_limit_tokens", 100000)
    out = _render("average lrt per station", _cv("station"))
    assert "properties (1 of 2 match): `lrt`" in out


# --------------------------------------------------------------------------- #
# multiple inheritance across the ancestor DAG (§2)
# --------------------------------------------------------------------------- #
def _diamond_catalog(monkeypatch):
    rows = {
        "concepts": [
            {"concept": "thing"},
            {"concept": "person", "inheritance": "thing", "inheritance_level": 1},
            {"concept": "account", "inheritance": "thing", "inheritance_level": 1},
            {"concept": "premium", "inheritance": "person, account", "inheritance_level": 2},
        ],
        "properties": [
            {"property_name": "email", "is_measure": "false"},
            {"property_name": "balance", "is_measure": "false"},
        ],
        "concept_properties": [
            {"concept": "person", "property_name": "email"},
            {"concept": "account", "property_name": "balance"},
        ],
        "relationships": [],
        "views": [],
        "view_properties": [],
    }
    catalog = icc._build_catalog(rows)
    monkeypatch.setattr(icc, "_load_catalog", lambda conn_params: catalog)
    return catalog


def test_multi_parent_renders_also_inherits_marker(monkeypatch):
    _diamond_catalog(monkeypatch)
    out = _render("premium email and balance", _cv("person", "account", "premium"))
    # premium nests under one parent (account, first by sort_key) and flags the
    # divergent-line ancestor.
    assert "(also inherits `person`)" in out
    # rendered exactly once, no double placement under both parents
    assert out.count("`premium`") == 1


def test_multi_parent_pulls_member_from_second_line(monkeypatch):
    _diamond_catalog(monkeypatch)
    out = _render("premium email and balance", _cv("account", "premium"))
    # account is the nesting parent (owns balance); premium still inlines `email`
    # inherited from its OTHER parent `person` (which is absent from the set).
    premium_block = out.split("`premium`", 1)[1]
    assert "`email`" in premium_block


def test_inheritance_cycle_is_safe(monkeypatch):
    rows = {
        "concepts": [
            {"concept": "a", "inheritance": "b", "inheritance_level": 1},
            {"concept": "b", "inheritance": "a", "inheritance_level": 1},
        ],
        "properties": [], "concept_properties": [],
        "relationships": [], "views": [], "view_properties": [],
    }
    catalog = icc._build_catalog(rows)
    monkeypatch.setattr(icc, "_load_catalog", lambda conn_params: catalog)
    out = _render("a and b", _cv("a", "b"))  # must terminate, not hang
    assert "`a`" in out and "`b`" in out


# --------------------------------------------------------------------------- #
# relationship / measure axis trim: three-axis match + "+N more" (§4)
# --------------------------------------------------------------------------- #
def _force_rel_trim_stage(monkeypatch):
    monkeypatch.setattr(config, "identify_concept_context_desc_trim_tokens", 1)
    monkeypatch.setattr(config, "identify_concept_context_rel_trim_tokens", 1)
    monkeypatch.setattr(config, "identify_concept_context_hard_limit_tokens", 100000)


def test_relationship_trim_keeps_matching_and_counts_rest(monkeypatch):
    rows = {
        "concepts": [
            {"concept": "thing"},
            {"concept": "hub", "inheritance": "thing", "inheritance_level": 1},
            {"concept": "alpha", "inheritance": "thing", "inheritance_level": 1},
            {"concept": "beta", "inheritance": "thing", "inheritance_level": 1},
        ],
        "properties": [], "concept_properties": [],
        "relationships": [
            {"concept": "hub", "target_concept": "alpha",
             "relationship_name": "to_alpha", "inverse_name": ""},
            {"concept": "hub", "target_concept": "beta",
             "relationship_name": "to_beta", "inverse_name": ""},
        ],
        "views": [], "view_properties": [],
    }
    catalog = icc._build_catalog(rows)
    monkeypatch.setattr(icc, "_load_catalog", lambda conn_params: catalog)
    _force_rel_trim_stage(monkeypatch)
    out = _render("alpha connections", _cv("hub"))
    rel_line = [l for l in out.splitlines() if "relationships:" in l][0]
    assert "alpha" in rel_line
    assert "to_beta" not in rel_line
    assert "+1 more" in rel_line


def test_measure_trim_keeps_matching_and_counts_rest(monkeypatch):
    rows = {
        "concepts": [
            {"concept": "thing"},
            {"concept": "metrics", "inheritance": "thing", "inheritance_level": 1},
        ],
        "properties": [
            {"property_name": "revenue", "is_measure": "true"},
            {"property_name": "latency", "is_measure": "true"},
        ],
        "concept_properties": [
            {"concept": "metrics", "property_name": "revenue"},
            {"concept": "metrics", "property_name": "latency"},
        ],
        "relationships": [], "views": [], "view_properties": [],
    }
    catalog = icc._build_catalog(rows)
    monkeypatch.setattr(icc, "_load_catalog", lambda conn_params: catalog)
    _force_rel_trim_stage(monkeypatch)
    out = _render("revenue report", _cv("metrics"))
    measure_line = [l for l in out.splitlines() if "measures:" in l][0]
    assert "`revenue`" in measure_line
    assert "latency" not in measure_line
    assert "+1 more" in measure_line


# --------------------------------------------------------------------------- #
# sub-type hint cap (§3)
# --------------------------------------------------------------------------- #
def test_subtype_hints_capped_with_overflow_count(monkeypatch):
    concepts = [
        {"concept": "thing"},
        {"concept": "item", "inheritance": "thing", "inheritance_level": 1},
    ]
    for color in ("red", "blue", "green", "yellow"):
        concepts.append({"concept": f"{color}_item", "inheritance": "item",
                         "inheritance_level": 2})
    rows = {
        "concepts": concepts, "properties": [], "concept_properties": [],
        "relationships": [], "views": [], "view_properties": [],
    }
    catalog = icc._build_catalog(rows)
    monkeypatch.setattr(icc, "_load_catalog", lambda conn_params: catalog)
    monkeypatch.setattr(config, "identify_concept_context_hint_cap", 2)
    out = _render("red blue green yellow items", _cv("item"))
    hint_line = [l for l in out.splitlines() if "narrow to sub-types" in l][0]
    assert "+2 more" in hint_line


# --------------------------------------------------------------------------- #
# description truncation hygiene (§4)
# --------------------------------------------------------------------------- #
def test_truncate_collapses_whitespace():
    assert icc._truncate("a\n\n  b   c", 100) == "a b c"


def test_truncate_cuts_on_word_boundary():
    out = icc._truncate("alpha beta gamma delta epsilon", 14)
    assert "\n" not in out
    assert out.endswith("...")
    assert " gamma" not in out  # cut fell back to the previous whole word


def test_descriptions_truncated_never_dropped(monkeypatch, patched_catalog):
    cv = {
        "customer": {"concept": "customer", "is_view": "false",
                     "description": "sentence. " * 400},
    }
    monkeypatch.setattr(config, "identify_concept_context_desc_trim_tokens", 1)
    monkeypatch.setattr(config, "identify_concept_context_rel_trim_tokens", 1)
    monkeypatch.setattr(config, "identify_concept_context_hard_limit_tokens", 100000)
    out = _render("customers", cv)
    # a truncated fragment of the description survives at the deepest stage
    assert "sentence" in out
    assert ("sentence. " * 400) not in out

