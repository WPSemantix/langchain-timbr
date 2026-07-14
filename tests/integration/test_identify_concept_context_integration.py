"""Integration tests for the identify-concept context builder (v2 Test Design Guide).

These exercise the catalog builder against a live Timbr backend. Every structural
test (A–G) captures the *built catalog prompt* directly via `build_catalog_lines`
and asserts on it — no LLM is invoked. Only §I runs the real chain (one LLM call).

Ontologies used (both reachable with the same TIMBR_TOKEN):
  * supply_metrics_llm_tests   — completeness, cubes/views, trim path, the LLM call
  * timbr_crunchbase_llm_tests — hierarchy, subtraction, sub-type hints, connected concepts

Per the guide's instruction, every catalog is built with include_logic_concepts=True
so sub-concepts participate in all relevant tests.
"""

import contextlib
import re

import pytest
from rapidfuzz import fuzz

from langchain_timbr import IdentifyTimbrConceptChain
from langchain_timbr import config as timbr_config
from langchain_timbr import trigram
from langchain_timbr.identify_concept_context import build_catalog_lines
from langchain_timbr.utils.timbr_utils import get_concepts, get_tags


SUPPLY = "supply_metrics_llm_tests"
CRUNCH = "timbr_crunchbase_llm_tests"


# --------------------------------------------------------------------------- #
# fuzzy presence helpers (guide §ground rules — ≥95% token similarity)
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def present(prompt, name, thr=95):
    toks = _TOKEN_RE.findall(prompt)
    return max((fuzz.ratio(name, t) for t in toks), default=0) >= thr


def absent(prompt, name, thr=95):
    return not present(prompt, name, thr)


def connected_lines(prompt):
    """Join every `connected concepts:` line so connected-axis asserts are scoped
    to the actual axis (not the whole prompt, where the view name itself appears)."""
    return "\n".join(l for l in prompt.splitlines() if "connected concepts:" in l)


def region_between(prompt, start_name, end_name=None):
    """The prompt block starting at the header line that owns `start_name` up to the
    next top-level concept header (a line matching `- \\`...\\``) or `end_name`.

    Used for §C adjacency/subtraction — a property must appear in one concept's
    block but not a sibling's.
    """
    lines = prompt.splitlines()
    start = None
    for i, l in enumerate(lines):
        if l.lstrip().startswith("- ") and f"`{start_name}`" in l:
            start = i
            break
    if start is None:
        return ""
    out = [lines[start]]
    header_re = re.compile(r"^\s*- `")
    for l in lines[start + 1:]:
        if end_name and f"`{end_name}`" in l and header_re.match(l):
            break
        if header_re.match(l) and l.startswith("- "):  # next top-level header
            break
        out.append(l)
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# prompt-capture seam — build the catalog with NO LLM
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _config_overrides(**overrides):
    saved = {k: getattr(timbr_config, k) for k in overrides}
    try:
        for k, v in overrides.items():
            setattr(timbr_config, k, v)
        yield
    finally:
        for k, v in saved.items():
            setattr(timbr_config, k, v)


def _conn(config, ontology):
    return {
        "url": config["timbr_url"],
        "token": config["timbr_token"],
        "ontology": ontology,
        "verify_ssl": config["verify_ssl"],
    }


def make_render(config):
    def render(question, concepts_list=None, views_list=None, ontology=SUPPLY, **overrides):
        conn = _conn(config, ontology)
        with _config_overrides(**overrides):
            cv = get_concepts(
                conn_params=conn,
                concepts_list=concepts_list,
                views_list=views_list,
                include_logic_concepts=True,
            )
            tags = get_tags(conn_params=conn)
            lines = build_catalog_lines(
                question=question,
                conn_params=conn,
                concepts_and_views=cv,
                tags=tags,
                prefix="",
            )
        return "\n".join(lines)

    return render


@pytest.fixture
def render(config):
    return make_render(config)


# Force the rel/measure trim stage: descriptions + full-rel ceilings tiny, hard
# ceiling generous so the ladder lands on the rel_trim stage (not the collapse floor).
FORCE_REL_TRIM = dict(
    identify_concept_context_desc_trim_tokens=1,
    identify_concept_context_rel_trim_tokens=1,
    identify_concept_context_hard_limit_tokens=100000,
)


# =========================================================================== #
# A. Completeness — never drop a candidate (supply_metrics_llm_tests)
# =========================================================================== #
class TestCompleteness:
    BASE = ["bill_of_material", "customer", "inventory", "material",
            "order", "plant", "product", "shipment"]
    VIEWS = ["customer_360", "order_metrics"]
    CUBES = ["customer_cube", "inventory_cube", "order_cube", "product_cube"]

    def test_everything_present(self, render):
        out = render("show me everything about the supply chain")
        for name in self.BASE + self.VIEWS + self.CUBES:
            assert present(out, name), f"{name} missing from catalog"

    def test_lexically_narrow_still_keeps_all_base(self, render):
        out = render("what is the total revenue")
        for name in ["plant", "material", "bill_of_material", "inventory"]:
            assert present(out, name), f"{name} dropped on a narrow question"


# =========================================================================== #
# B. Hierarchy: nesting + [level: N] + inherits (timbr_crunchbase_llm_tests)
# =========================================================================== #
class TestHierarchy:
    def test_nesting_levels_and_inherits(self, render):
        out = render(
            "companies",
            concepts_list=["organization", "company", "advertising_company"],
            ontology=CRUNCH,
        )
        for name in ["organization", "company", "advertising_company"]:
            assert present(out, name)
        assert "[level: 0]" not in out
        assert "[level: 1]" not in out
        assert "[level: 2]" in out
        assert "[level: 3]" in out
        assert "inherits `company`" in out
        assert "inherits `organization`" in out


# =========================================================================== #
# C. Direct-only vs inherited subtraction (timbr_crunchbase_llm_tests)
# =========================================================================== #
class TestSubtraction:
    def test_parent_present_direct_only(self, render):
        # Properties are trigram-filtered, so the question names both the owned
        # (`category_code`) and inherited (`organization_name`) properties under
        # test; the subtraction behavior is what's asserted below.
        out = render(
            "company category code and organization name funding",
            concepts_list=["organization", "company"],
            ontology=CRUNCH,
        )
        company_block = region_between(out, "company")
        org_block = region_between(out, "organization", end_name="company")
        assert present(company_block, "category_code")
        assert present(org_block, "organization_name")
        # organization_name is inherited, not owned by company -> absent from its block
        assert absent(company_block, "organization_name")

    def test_parent_absent_inlines_inherited(self, render):
        # Question names both properties so the trigram filter surfaces the owned
        # and the inlined-inherited one.
        out = render(
            "company category code and organization name funding",
            concepts_list=["company"],
            ontology=CRUNCH,
        )
        assert present(out, "category_code")
        assert present(out, "organization_name")
        assert "inherits `organization`" in out


# =========================================================================== #
# D. Out-of-list sub-type hints (timbr_crunchbase_llm_tests)
# =========================================================================== #
class TestSubtypeHints:
    OTHER_COMPANIES = ["advertising_company", "analytics_company",
                       "ecommerce_company", "web_company"]

    def test_single_word_hint(self, render):
        out = render("biotech companies that went public",
                     concepts_list=["company"], ontology=CRUNCH)
        assert present(out, "biotech_company")
        for other in self.OTHER_COMPANIES:
            assert absent(out, other), f"{other} should not be hinted"

    def test_multi_word_hint(self, render):
        out = render("games and video companies",
                     concepts_list=["company"], ontology=CRUNCH)
        assert present(out, "games_video_company")

    def test_parent_not_in_list_no_hint(self, render):
        out = render("biotech", concepts_list=["organization"], ontology=CRUNCH)
        assert absent(out, "biotech_company")


# =========================================================================== #
# E. Views & cubes (§6)
# =========================================================================== #
class TestConnectedConceptsCrunch:
    def test_bio_tech_investments(self, render):
        out = render("bio tech investments",
                     views_list=["bio_tech_investments"], ontology=CRUNCH)
        conn = connected_lines(out)
        for name in ["financial_organization", "funding_round", "biotech_company", "ipo"]:
            assert present(conn, name), f"{name} not connected"

    def test_companies_acquired_by_ebay(self, render):
        out = render("companies acquired by ebay",
                     views_list=["companies_acquired_by_ebay"], ontology=CRUNCH)
        conn = connected_lines(out)
        assert present(conn, "ecommerce_company")
        assert present(conn, "company")

    def test_bio_company_ipo_no_db_table_leak(self, render):
        # Properties are trigram-filtered to the question, so name the ones we
        # assert on ("organization name" / "investments"). The DB-table-leak
        # guard below is independent of the question.
        out = render("bio company ipo organization name and num of investments",
                     views_list=["bio_company_ipo"], ontology=CRUNCH)
        assert present(out, "bio_company_ipo")  # still a candidate
        # property hints from sys_view_properties
        assert present(out, "organization_name")
        assert present(out, "num_of_investments")
        # raw DB tables must never leak as connected concepts
        conn = connected_lines(out)
        for tbl in ["cb_ipos", "cb_investments", "cb_objects", "cb_funding_rounds"]:
            assert absent(conn, tbl), f"{tbl} leaked as connected concept"

    def test_vtimbr_ref_not_injected(self, render):
        out = render("number of companies acquired by ebay",
                     views_list=["number_of_companies_acquired_by_ebay"], ontology=CRUNCH)
        assert present(out, "number_of_companies_acquired_by_ebay")  # candidate
        # a vtimbr.<view> ref is a view, not a concept axis — not injected as connected
        assert absent(connected_lines(out), "companies_acquired_by_ebay")

    def test_null_tables_no_crash(self, render):
        out = render("count bio company",
                     views_list=["count_bio_company"], ontology=CRUNCH)
        assert present(out, "count_bio_company")  # candidate, no crash

    def test_thing_excluded_from_connected(self, render):
        out = render("raised amount",
                     views_list=["raised_amount_view"], ontology=CRUNCH)
        assert present(out, "raised_amount_view")  # candidate
        assert absent(connected_lines(out), "thing")


class TestCubeMeasuresSupply:
    def test_order_cube(self, render):
        # Plain (non-measure) cube properties are trigram-filtered to the
        # question, so name them; measures render in full regardless.
        out = render(
            "order cube by order date, order region, market, customer segment, "
            "department and product name",
            views_list=["order_cube"], ontology=SUPPLY)
        conn = connected_lines(out)
        for name in ["order", "product", "customer", "shipment"]:
            assert present(conn, name), f"{name} not connected"
        measure_hints = ["total_revenue", "total_sales", "count_of_order",
                         "count_of_customer", "count_of_shipment",
                         "count_of_late_shipment", "late_shipment_ratio",
                         "average_product_price", "maximum_product_price"]
        for m in measure_hints:
            assert present(out, m), f"measure hint {m} missing"
        plain = ["order_date", "order_region", "market", "customer_segment",
                 "department", "product_name"]
        for p in plain:
            assert present(out, p), f"plain property {p} missing"


class TestCubeConnectingToSubconceptsCrunch:
    def test_biotech_cube(self, render):
        out = render("biotech cube", views_list=["biotech_cube"], ontology=CRUNCH)
        conn = connected_lines(out)
        for name in ["person", "biotech_company", "company"]:
            assert present(conn, name), f"{name} not connected"


class TestNonCubeViewCrossRefSupply:
    def test_order_metrics_measures_exact(self, render):
        # Non-measure properties are trigram-filtered to the question, so the
        # question names the ones asserted present below. Measures render in full
        # (they are only trimmed under token pressure).
        out = render(
            "order metrics: count of shipments and late shipments, total revenue "
            "europe and us canada, order year, customer segment",
            views_list=["order_metrics"], ontology=SUPPLY)
        # measure hints (rendered on a `measures:` line)
        measure_line = "\n".join(l for l in out.splitlines() if "measures:" in l)
        for m in ["count_of_order", "late_shipment_ratio", "total_revenue"]:
            assert present(measure_line, m), f"{m} not flagged as measure"
        # present but NOT measures (plural aliases / view-local computed)
        for p in ["count_of_shipments", "count_of_late_shipments",
                  "total_revenue_europe", "total_revenue_us_canada",
                  "order_year", "customer_segment"]:
            assert present(out, p), f"{p} missing"
            assert absent(measure_line, p), f"{p} wrongly flagged as a measure"


# =========================================================================== #
# F. Trim path — forced threshold (supply_metrics_llm_tests)
# =========================================================================== #
class TestTrimPath:
    def test_source_axis_self_protection(self, render):
        out = render("customer information and orders", **FORCE_REL_TRIM)
        cust = region_between(out, "customer")
        assert present(cust, "made_order")
        assert present(cust, "received_shipment")

    def test_target_axis_keep(self, render):
        out = render("orders and their shipment", **FORCE_REL_TRIM)
        order = region_between(out, "order")
        assert present(order, "in_shipment")

    def test_off_topic_trim_keeps_all_concepts(self, render):
        out = render("customer orders", **FORCE_REL_TRIM)
        # off-topic inventory/plant relationship axis is trimmable
        inv = region_between(out, "inventory")
        assert absent(inv, "made_in_plant")
        # but every base concept name is still present
        for name in ["customer", "order", "plant", "inventory", "shipment",
                     "product", "material", "bill_of_material"]:
            assert present(out, name)

    def test_hard_limit_never_fails(self, render):
        # every ceiling forced to 1 -> names-only floor, NO exception, all names present
        out = render(
            "anything at all",
            identify_concept_context_desc_trim_tokens=1,
            identify_concept_context_rel_trim_tokens=1,
            identify_concept_context_hard_limit_tokens=1,
        )
        for name in ["customer", "order", "plant", "inventory", "shipment",
                     "product", "material", "bill_of_material"]:
            assert present(out, name)


# =========================================================================== #
# G. Trigram matcher — direct unit tests (no backend, no LLM)
# =========================================================================== #
class TestTrigramMatcher:
    THR = 0.65
    FLOOR = 3

    def _match(self, name, question):
        return trigram.contains(
            trigram.to_trigram_set(name),
            trigram.to_trigram_set(question),
            self.THR, self.FLOOR,
        )

    def test_single_word_morphology(self):
        assert self._match("category", "which categories are trending")
        assert self._match("company", "list the companies that ipod")

    def test_multiword_ratio_keeps_near_drops_far(self):
        assert self._match("games_video_company", "games and video companies")
        assert not self._match("financial_organization", "list all products")

    def test_trigram_floor_blocks_coincidence(self):
        # 'shipment' shares only two coincidental trigrams (shi, hip) with the
        # question — below the >=3 floor — so it does not false-match. (A single
        # 1-2 gram name like 'id' still matches by subset; the floor guards the
        # multi-gram case the guide targets.)
        assert not self._match("shipment", "the ship sailed the sea")

    def test_name_as_denominator_invariant_to_question_length(self):
        short_q = "biotech companies"
        long_q = ("biotech companies " + "unrelated filler words " * 40)
        assert self._match("biotech_company", short_q)
        assert self._match("biotech_company", long_q)

    def test_diverges_from_token_set_ratio(self):
        # Morphology case: the question says "companies", the concept is "company".
        # The name-as-denominator trigram score matches on the shared `compan-` stem,
        # but a symmetric token_set_ratio drops below threshold — assert the divergence.
        name, question = "company", "these companies grew"
        assert self._match(name, question)
        tsr = fuzz.token_set_ratio(name, question) / 100.0
        assert tsr < self.THR


# =========================================================================== #
# I. Integration — the one real LLM call (supply_metrics_llm_tests)
# =========================================================================== #
class TestLlmIdentify:
    CASES = [
        ("total revenue by year", {"order", "order_metrics", "order_cube"}),
        ("how many late deliveries", {"shipment", "late_delivery_shipment", "order_metrics"}),
        ("materials used in each product", {"material", "product", "bill_of_material"}),
        ("which plant produces each product",
         {"plant", "product", "inventory", "bill_of_material"}),
    ]

    @pytest.mark.parametrize("question, acceptable", CASES)
    def test_identify(self, llm, config, question, acceptable):
        chain = IdentifyTimbrConceptChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=SUPPLY,
            include_logic_concepts=True,
            verify_ssl=config["verify_ssl"],
        )
        result = chain.invoke({"prompt": question})

        concept = (result.get("concept") or "").split(".")[-1].lower()
        assert concept, "chain returned an empty concept"
        assert concept in acceptable, f"{concept!r} not in {acceptable} for {question!r}"

        assert result.get("identify_concept_reason"), "empty identify_concept_reason"
        usage = result[chain.usage_metadata_key]
        assert len(usage) == 1 and "determine_concept" in usage, \
            "usage_metadata should contain exactly one key: determine_concept"
