"""
Integration tests for knowledge-base example retrieval inside the chains.

Verifies against a live Timbr backend that KB examples configured on the test
ontology (``supply_metrics_kb``) and on a Data Agent (``supply_metrics_kb2`` via
agent ``llm_test_agent_kb``) are retrieved, matched by the classifier against a
near-duplicate question (~90% similar), and injected as an
``[Approved reference examples]`` block into the generate-SQL / execute-query
chain paths.

Injection is observed through the ``MemoryContext`` the chain resolves and
stores on the instance (``chain._received_chain_context["memory"]``), so it is
available even if downstream SQL execution errors out.
"""
import os
import re

import pytest

from langchain_timbr import config
from langchain_timbr import kbclient
from langchain_timbr import decrypt_prompt, generate_key
from langchain_timbr.langchain.generate_timbr_sql_chain import GenerateTimbrSqlChain
from langchain_timbr.langchain.execute_timbr_query_chain import ExecuteTimbrQueryChain
from langchain_timbr.langchain.identify_concept_chain import IdentifyTimbrConceptChain
from langchain_timbr.utils.memory import (
    MemoryContext,
    format_memory_note_for_sql,
    _resolve_available_kbs,
)
from langchain_timbr.utils.timbr_llm_utils import _prompt_to_string

# Data Agent that exposes only ``supply_metrics_kb2`` (one example: order_rca_loss).
AGENT_KB = os.environ.get("TIMBR_AGENT_KB", "llm_test_agent_kb")

# ~90%-similar rewordings of the real KB example questions.
PROMPT_ORDER_LOSS = (
    "Which regions and product departments are driving losses on completed "
    "orders, and how much late delivery and discounting is involved?"
)
PROMPT_INVENTORY_STOCKOUT = (
    "Which warehouses are at risk of stock-out for a given department, "
    "comparing on-hand quantity against order demand?"
)
PROMPT_AGENT_ORDER_RCA = "Which region and product department are driving the losses?"


def _build_conn_params(config):
    return {
        "url": config["timbr_url"],
        "token": config["timbr_token"],
        "ontology": config["timbr_ontology"],
    }


@pytest.fixture
def kb_enabled(monkeypatch):
    """Enable KB retrieval. Fallback is OFF so tests assert real KB matches."""
    monkeypatch.setattr(config, "enable_knowledge_base", True)
    monkeypatch.setattr(config, "kb_fallback_example", False)
    yield


def _capture_memory(chain, prompt):
    """Invoke a chain and return the ``MemoryContext`` it resolved.

    Memory is resolved before any SQL execution and stored on the chain
    instance, so it is captured even when execute-query later raises on the
    generated SQL.
    """
    try:
        chain.invoke({"prompt": prompt})
    except Exception:
        pass
    ctx = chain._received_chain_context or {}
    return ctx.get("memory")


def _example_names(memory_ctx):
    return {ex.get("example_name") for ex in (memory_ctx.kb_examples or [])}


class TestOntologyKnowledgeBaseInjection:
    """KB examples on the ontology are matched and injected into the chains."""

    def test_generate_sql_chain_injects_ontology_kb_example(self, config, llm, kb_enabled):
        chain = GenerateTimbrSqlChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
        )
        memory_ctx = _capture_memory(chain, PROMPT_ORDER_LOSS)

        assert isinstance(memory_ctx, MemoryContext), (
            "generate-sql chain did not resolve a MemoryContext (KB gate off?)"
        )
        assert "order_loss_root_cause" in _example_names(memory_ctx), (
            f"classifier did not select the expected ontology KB example; "
            f"got {_example_names(memory_ctx)}"
        )
        note = format_memory_note_for_sql(memory_ctx)
        assert "[Approved reference examples]" in note
        assert "order_loss_root_cause" in note

    def test_execute_query_chain_injects_ontology_kb_example(self, config, llm, kb_enabled):
        chain = ExecuteTimbrQueryChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
        )
        memory_ctx = _capture_memory(chain, PROMPT_INVENTORY_STOCKOUT)

        assert isinstance(memory_ctx, MemoryContext), (
            "execute-query chain did not resolve a MemoryContext (KB gate off?)"
        )
        assert "inventory_stockout_risk" in _example_names(memory_ctx), (
            f"classifier did not select the expected ontology KB example; "
            f"got {_example_names(memory_ctx)}"
        )
        note = format_memory_note_for_sql(memory_ctx)
        assert "[Approved reference examples]" in note
        assert "inventory_stockout_risk" in note


class TestAgentKnowledgeBaseInjection:
    """The agent's KB is used exclusively (ontology KB is NOT consulted)."""

    def test_agent_resolves_only_its_own_kb(self, config, kb_enabled):
        conn_params = _build_conn_params(config)
        # Agent-first: ontology is ignored when an agent is supplied.
        kb_names = _resolve_available_kbs(conn_params, AGENT_KB, "supply_metrics")
        assert kb_names == ["supply_metrics_kb2"], (
            f"expected only the agent KB, got {kb_names}"
        )

    def test_agent_kb_example_injected_and_isolated(self, config, llm, kb_enabled):
        chain = GenerateTimbrSqlChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            agent=AGENT_KB,
        )
        memory_ctx = _capture_memory(chain, PROMPT_AGENT_ORDER_RCA)

        assert isinstance(memory_ctx, MemoryContext), (
            "agent chain did not resolve a MemoryContext (KB gate off?)"
        )
        assert "order_rca_loss" in _example_names(memory_ctx), (
            f"classifier did not select the agent KB example; "
            f"got {_example_names(memory_ctx)}"
        )
        # Isolation: every selected example must come from the agent KB only.
        kbs = {ex.get("knowledge_base") for ex in memory_ctx.kb_examples}
        assert kbs == {"supply_metrics_kb2"}, (
            f"agent run leaked non-agent knowledge bases: {kbs}"
        )
        note = format_memory_note_for_sql(memory_ctx)
        assert "[Approved reference examples]" in note
        assert "order_rca_loss" in note


# ---------------------------------------------------------------------------
# Knowledge-base RULES injection (timbr.sys_knowledgebase_rules)
# ---------------------------------------------------------------------------
@pytest.fixture
def rules_enabled(kb_enabled):
    """Rule injection rides on the shared ``enable_knowledge_base`` gate (turned on
    by the ``kb_enabled`` fixture); start each test from a clean session rules cache."""
    kbclient.clear_rules_cache()
    yield


@pytest.fixture
def capture_prompts(monkeypatch):
    """Record every prompt sent to the LLM.

    All pipeline stages (identify / prefilter / filter / generate_sql / reasoning)
    route their LLM call through ``timbr_llm_utils._call_llm_with_timeout`` — the
    module-level callers and the lazy imports in filter/prefilter both resolve the
    name at call time — so patching it here is a single universal capture point.
    Delegates to the original so the live pipeline runs unchanged.
    """
    from langchain_timbr.utils import timbr_llm_utils as tu

    prompts: list = []
    original = tu._call_llm_with_timeout

    def spy(llm, prompt, timeout=120, *args, **kwargs):
        try:
            prompts.append(_prompt_to_string(prompt))
        except Exception:
            prompts.append(str(prompt))
        return original(llm, prompt, timeout=timeout, *args, **kwargs)

    monkeypatch.setattr(tu, "_call_llm_with_timeout", spy)
    return prompts


class TestKnowledgeBaseRulesInjection:
    """Real KB rules (``supply_metrics_kb``) reach the correct per-stage prompt.

    End-to-end: live ``sys_knowledgebase_rules`` -> ``fetch_rules`` -> pipeline ->
    prompt. Each test pins the anchor so the ruled object is deterministically in
    context, then captures the prompt(s) actually sent to the LLM.
    """

    # ---- helpers -----------------------------------------------------------
    def _generate_chain(self, config, llm, **kw):
        return GenerateTimbrSqlChain(
            llm=llm,
            url=config["timbr_url"],
            token=config["timbr_token"],
            ontology=config["timbr_ontology"],
            **kw,
        )

    @staticmethod
    def _run(chain, prompt):
        """Invoke a chain, returning its result dict (or None). Prompts are
        captured by the fixture even when downstream SQL execution raises."""
        try:
            return chain.invoke({"prompt": prompt})
        except Exception:
            return None

    @staticmethod
    def _joined(prompts):
        return "\n\n".join(prompts)

    @staticmethod
    def _stage_prompt(prompts, marker):
        return next((p for p in prompts if marker in p), None)

    @staticmethod
    def _generate_sql_prompt(result):
        """Exact generate_sql prompt, recovered from the debug result's p_hash."""
        assert result, "chain returned no result (generation failed)"
        p_hash = (
            (result.get("generate_sql_usage_metadata") or {})
            .get("generate_sql", {})
            .get("p_hash")
        )
        assert p_hash, f"no generate_sql p_hash in result (debug=True?): {result.get('generate_sql_usage_metadata')}"
        return decrypt_prompt(p_hash, generate_key())

    def _sanity_rule(self, config, name, target_types, kinds, needle):
        """Fail with a clear diagnostic if the rule isn't fetched from the live KB
        (distinct from an injection failure)."""
        conn = _build_conn_params(config)
        rs = kbclient.fetch_rules(conn, ontology=config["timbr_ontology"])
        got = rs.rules_for(name, target_types, kinds)
        text = " ".join(t for texts in got.values() for t in texts)
        assert needle.lower() in text.lower(), (
            f"rule for {name!r} ({target_types}/{kinds}) not found in live KB "
            f"(resolved kbs={rs.kb_names}); is supply_metrics_kb attached to "
            f"{config['timbr_ontology']!r}? got={got}"
        )

    # ---- identify ----------------------------------------------------------
    def test_concept_selection_reaches_identify_prompt(self, config, llm, rules_enabled, capture_prompts):
        # delivery_anchor_concept: SELECTION on concept `shipment`
        self._sanity_rule(config, "shipment", ("concept",), {"selection"},
                          "anchor the query on the shipment concept")
        chain = IdentifyTimbrConceptChain(
            llm=llm, url=config["timbr_url"], token=config["timbr_token"],
            ontology=config["timbr_ontology"],
        )
        self._run(chain, "Which shipments were delivered late last year?")
        assert "anchor the query on the shipment concept" in self._joined(capture_prompts), (
            f"shipment SELECTION rule not in identify prompt; captured {len(capture_prompts)} prompt(s)"
        )

    # ---- filter (relationship of order, pulled via order+product) ----------
    def test_relationship_selection_reaches_filter_prompt(self, config, llm, rules_enabled, capture_prompts):
        # product_selection: SELECTION on relationship `includes_product`
        self._sanity_rule(config, "includes_product", ("relationship",), {"selection"},
                          "default to product name")
        chain = self._generate_chain(config, llm, concept="order", metadata_context_mode="dynamic")
        self._run(chain, "List the product names included in each order")
        filter_prompt = self._stage_prompt(capture_prompts, "ONTOLOGY SUBGRAPH")
        assert filter_prompt is not None, "dynamic filter prompt was not produced"
        assert "default to product name" in filter_prompt, (
            "includes_product SELECTION rule not injected into the filter prompt"
        )

    # ---- generate_sql: concept INSTRUCTION ---------------------------------
    def test_concept_instruction_reaches_generate_sql_prompt(self, config, llm, rules_enabled):
        # order_grain_semantics: INSTRUCTION on concept `order`
        self._sanity_rule(config, "order", ("concept",), {"instruction"}, "ORDER-ITEM")
        chain = self._generate_chain(config, llm, concept="order", metadata_context_mode="static", debug=True)
        result = self._run(chain, "How many orders were placed per market?")
        assert "ORDER-ITEM (line) grain" in self._generate_sql_prompt(result)

    # ---- generate_sql: measure INSTRUCTION ---------------------------------
    def test_measure_instruction_reaches_generate_sql_prompt(self, config, llm, rules_enabled):
        # late_shipment_ratio_semantics: INSTRUCTION on measure `late_shipment_ratio`
        self._sanity_rule(config, "late_shipment_ratio", ("measure",), {"instruction"}, "ON-TIME fraction")
        chain = self._generate_chain(config, llm, concept="shipment", metadata_context_mode="static", debug=True)
        result = self._run(chain, "What is the late shipment ratio by market?")
        assert "computes the ON-TIME fraction" in self._generate_sql_prompt(result)

    # ---- generate_sql: property SELECTION ----------------------------------
    def test_property_selection_market_reaches_generate_sql_prompt(self, config, llm, rules_enabled):
        # default_geography_grain: SELECTION on property `market`
        self._sanity_rule(config, "market", ("property",), {"selection"}, "continent-level")
        chain = self._generate_chain(config, llm, concept="order", metadata_context_mode="static", debug=True)
        result = self._run(chain, "Show total sales by market")
        assert "default to market (continent-level)" in self._generate_sql_prompt(result)

    def test_property_selection_order_date_reaches_generate_sql_prompt(self, config, llm, rules_enabled):
        # default_time_property: SELECTION on property `order_date`
        self._sanity_rule(config, "order_date", ("property",), {"selection"}, "order_date")
        chain = self._generate_chain(config, llm, concept="order", metadata_context_mode="static", debug=True)
        result = self._run(chain, "Show orders over time")
        assert "Use order_date" in self._generate_sql_prompt(result)

    # ---- generate_sql: measure SELECTION -----------------------------------
    def test_measure_selection_total_revenue_reaches_generate_sql_prompt(self, config, llm, rules_enabled):
        # revenue_measure_choice: SELECTION on measure `total_revenue`
        self._sanity_rule(config, "total_revenue", ("measure",), {"selection"}, "total_revenue")
        chain = self._generate_chain(config, llm, concept="order", metadata_context_mode="static", debug=True)
        result = self._run(chain, "What is the revenue by market?")
        assert "map it to total_revenue" in self._generate_sql_prompt(result)

    # ---- generate_sql: property VALIDATION ---------------------------------
    def test_property_validation_market_enum_reaches_generate_sql_prompt(self, config, llm, rules_enabled):
        # market_enum_guard: VALIDATION on property `market`
        self._sanity_rule(config, "market", ("property",), {"validation"}, "closed enumeration")
        chain = self._generate_chain(config, llm, concept="order", metadata_context_mode="static", debug=True)
        result = self._run(chain, "Show revenue for the USCA market")
        assert "market is a closed enumeration" in self._generate_sql_prompt(result)

    # ---- reasoning (cube VALIDATION) — prompt AND generated SQL ------------
    def test_cube_validation_reaches_reasoning_prompt_and_sql(self, config, llm, rules_enabled, capture_prompts):
        # order_cube_status_rule: VALIDATION on cube `order_cube`
        self._sanity_rule(config, "order_cube", ("cube", "view", "concept"), {"validation"}, "order_status")
        chain = self._generate_chain(
            config, llm, concept="order_cube", schema="vtimbr", enable_reasoning=True,
        )
        # Question omits any status so the default-status validation applies.
        result = self._run(chain, "Show total revenue by market")

        # 1) Injected into the reasoning prompt.
        reasoning_prompt = self._stage_prompt(capture_prompts, "**Knowledge Base Validation Rules:**")
        assert reasoning_prompt is not None, "reasoning prompt was not produced (enable_reasoning?)"
        assert "always filter by order_status" in reasoning_prompt

        # 2) Actually applied in the generated SQL (not just present in the prompt).
        assert result and result.get("sql"), f"no SQL generated; result={result}"
        assert re.search(r"order_status.{0,20}['\"]complete", result["sql"], re.IGNORECASE | re.DOTALL), (
            f"order_status='COMPLETE' filter not applied to generated SQL:\n{result['sql']}"
        )
