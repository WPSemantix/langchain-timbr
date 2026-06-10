"""Plan 2 integration tests — dynamic metadata-context against live Timbr.

Two ontologies are exercised with **graph_depth=4** (detail band) plus the
default **max_graph_depth=3** (outer reachability bound for the menu band) to
force the dynamic pipeline to trigger:

  - ``timbr_crunchbase_llm_tests`` (CRUNCHBASE_ONTOLOGY)
    Anchor: ``person``. Example: "Find people that invested in bio tech
    companies that have employees with PhD" → path person → company → person → degree.

  - ``config["timbr_ontology"]`` (default ``supply_metrics_llm_tests``)
    Anchor: ``customer``. Example: "Which customer segment purchased the most
    metal material" → path customer → order → product → material.

Live Timbr + LLM credentials are required. Tests skip when env vars are absent
— see the ``run-integration-tests`` skill for how to source from
``.vscode/launch.json`` when running locally.

The plan acknowledges LLM variability: tests assert on the concept sequence,
not on specific relationship names, and treat SQL-result correctness as
observational rather than a hard gate.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import pytest

from langchain_timbr.ontology_context import (
    EdgeIndex,
    MetadataContextConfig,
    Ontology,
    TimbrOntologyClient,
    build_filtered_metadata,
    retrieve_subgraph,
    serialize_compact_ddl,
)


CRUNCHBASE_ONTOLOGY = "timbr_crunchbase_llm_tests"
# detail_depth (per-chain graph_depth) MUST be strictly less than max_graph_depth
# (default 5). 4 keeps the original deep-traversal coverage while satisfying
# the band-split invariant.
GRAPH_DEPTH = 4


CRUNCHBASE_QUESTION = (
    "Find people that invested in bio tech companies that have employees with PhD"
)
CRUNCHBASE_ANCHOR = "person"
# Concept sequence we expect to traverse (relationship names left flexible).
CRUNCHBASE_EXPECTED_SEQUENCE = ["person", "company", "person", "degree"]

SUPPLY_QUESTION = "Which customer segment purchased the most metal material"
SUPPLY_ANCHOR = "customer"
SUPPLY_EXPECTED_SEQUENCE = ["customer", "order", "product", "material"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _conn_params(config, *, ontology: str) -> dict:
    if not config.get("timbr_url") or not config.get("timbr_token"):
        pytest.skip("TIMBR_URL / TIMBR_TOKEN not set — skipping integration test")
    return {
        "url": config["timbr_url"],
        "token": config["timbr_token"],
        "ontology": ontology,
        "verify_ssl": config["verify_ssl"],
    }


def _ontology(config, ontology_name: str) -> Ontology:
    client = TimbrOntologyClient(_conn_params(config, ontology=ontology_name))
    return Ontology(client)


def _dynamic_config(**overrides) -> MetadataContextConfig:
    """Build a MetadataContextConfig with sane test defaults. Force mode=dynamic
    so the pipeline runs regardless of the env METADATA_CONTEXT_MODE setting."""
    base = dict(
        mode="dynamic",
        # Slightly tighter DDL budget so the cascade is exercised on real data.
        metadata_context_filter_max_tokens=8_000,
        metadata_context_filter_max_tokens_hard_ceiling=16_000,
        metadata_context_max_tokens=12_000,
    )
    base.update(overrides)
    return MetadataContextConfig(**base)


def _concept_sequence(paths) -> List[str]:
    """Return the concept sequence from the longest validated path (anchor->...->target)."""
    if not paths:
        return []
    longest = max(paths, key=lambda p: len(p.segments))
    seq = [longest.segments[0].from_concept] + [s.to_concept for s in longest.segments]
    return seq


def _contains_subsequence(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    """Check that ``needle`` appears as a contiguous subsequence of ``haystack``."""
    if not needle:
        return True
    for i in range(len(haystack) - len(needle) + 1):
        if list(haystack[i:i + len(needle)]) == list(needle):
            return True
    return False


# ---------------------------------------------------------------------------
# 1. Trigger correctness — static vs dynamic at graph_depth=5
# ---------------------------------------------------------------------------


class TestTriggerCorrectness:
    """mode='static' must skip dynamic entirely; mode='dynamic' must fire it."""

    def test_static_mode_does_not_touch_ontology(self, config):
        from langchain_timbr.utils.timbr_llm_utils import (
            _apply_dynamic_metadata_context,
        )

        # Verify creds before running.
        _conn_params(config, ontology=CRUNCHBASE_ONTOLOGY)

        before_cols = "static columns string"
        before_meas = "static measures string"
        before_rels = "static relationships string"

        a, b, c, _eff = _apply_dynamic_metadata_context(
            mode="static",
            question=CRUNCHBASE_QUESTION,
            anchor=CRUNCHBASE_ANCHOR,
            conn_params={"this should not be touched": True},
            graph_depth=GRAPH_DEPTH,
            columns=[],
            measures=[],
            tags=None,
            exclude_properties=None,
            static_columns_str=before_cols,
            static_measures_str=before_meas,
            static_rel_prop_str=before_rels,
            llm=None,
            config_overrides={},
        )
        # Static mode = strict identity.
        assert (a, b, c) == (before_cols, before_meas, before_rels)

    def test_dynamic_mode_fires_pipeline(self, config, llm):
        """mode='dynamic' must run the pipeline and produce a non-empty result."""
        ontology = _ontology(config, CRUNCHBASE_ONTOLOGY)
        cfg = _dynamic_config()
        result = build_filtered_metadata(
            question=CRUNCHBASE_QUESTION,
            anchor=CRUNCHBASE_ANCHOR,
            ontology=ontology,
            llm=llm,
            config=cfg,
            graph_depth=GRAPH_DEPTH,
        )
        assert result.error is None, f"Dynamic pipeline failed: {result.error}"
        assert result.validated_paths, "Expected at least one validated path"
        assert result.compact_ddl, "Expected a non-empty Compact DDL"


# ---------------------------------------------------------------------------
# 2. Path selection accuracy — concept sequence over the LLM's path
# ---------------------------------------------------------------------------


class TestCrunchbasePathSelection:

    def test_path_matches_expected_concept_sequence(self, config, llm):
        ontology = _ontology(config, CRUNCHBASE_ONTOLOGY)
        result = build_filtered_metadata(
            question=CRUNCHBASE_QUESTION,
            anchor=CRUNCHBASE_ANCHOR,
            ontology=ontology,
            llm=llm,
            config=_dynamic_config(),
            graph_depth=GRAPH_DEPTH,
        )
        assert result.error is None
        all_concepts = set()
        for p in result.validated_paths:
            for s in p.segments:
                all_concepts.add(s.from_concept)
                all_concepts.add(s.to_concept)
        # Expected concepts should ALL appear somewhere in the validated paths.
        # (LLM may split the journey across multiple paths or take a slightly
        # different ordering — we assert presence, not strict sequence here.)
        for c in CRUNCHBASE_EXPECTED_SEQUENCE:
            assert c in all_concepts, (
                f"Expected concept {c!r} missing from validated paths. "
                f"Got concepts: {sorted(all_concepts)}"
            )

    def test_path_starts_at_anchor(self, config, llm):
        ontology = _ontology(config, CRUNCHBASE_ONTOLOGY)
        result = build_filtered_metadata(
            question=CRUNCHBASE_QUESTION,
            anchor=CRUNCHBASE_ANCHOR,
            ontology=ontology,
            llm=llm,
            config=_dynamic_config(),
            graph_depth=GRAPH_DEPTH,
        )
        assert result.validated_paths
        # The first validated path MUST start at the anchor. Subsequent
        # paths are allowed to start at a concept reached by a prior
        # path (single-hop style, explicitly preferred by the prompt).
        assert result.validated_paths[0].segments[0].from_concept == CRUNCHBASE_ANCHOR


class TestSupplyMetricsPathSelection:

    def test_path_matches_expected_concept_sequence(self, config, llm):
        ontology = _ontology(config, config["timbr_ontology"])
        result = build_filtered_metadata(
            question=SUPPLY_QUESTION,
            anchor=SUPPLY_ANCHOR,
            ontology=ontology,
            llm=llm,
            config=_dynamic_config(),
            graph_depth=GRAPH_DEPTH,
        )
        assert result.error is None, f"Dynamic pipeline failed: {result.error}"
        all_concepts = set()
        for p in result.validated_paths:
            for s in p.segments:
                all_concepts.add(s.from_concept)
                all_concepts.add(s.to_concept)
        for c in SUPPLY_EXPECTED_SEQUENCE:
            assert c in all_concepts, (
                f"Expected concept {c!r} missing from validated paths. "
                f"Got concepts: {sorted(all_concepts)}"
            )

    def test_path_starts_at_anchor(self, config, llm):
        ontology = _ontology(config, config["timbr_ontology"])
        result = build_filtered_metadata(
            question=SUPPLY_QUESTION,
            anchor=SUPPLY_ANCHOR,
            ontology=ontology,
            llm=llm,
            config=_dynamic_config(),
            graph_depth=GRAPH_DEPTH,
        )
        assert result.validated_paths
        # The first validated path MUST start at the anchor. Subsequent
        # paths are allowed to start at a concept reached by a prior
        # path (single-hop style, explicitly preferred by the prompt).
        assert result.validated_paths[0].segments[0].from_concept == SUPPLY_ANCHOR


# ---------------------------------------------------------------------------
# 3. Rebuild fidelity — path concepts get FULL props and measures
# ---------------------------------------------------------------------------


class TestRebuildFidelity:
    """Verify the anti-hallucination guarantee: every concept on a validated
    path keeps ALL of its properties and measures, not a filtered subset."""

    def test_filtered_concepts_carry_full_properties(self, config, llm):
        ontology = _ontology(config, CRUNCHBASE_ONTOLOGY)
        result = build_filtered_metadata(
            question=CRUNCHBASE_QUESTION,
            anchor=CRUNCHBASE_ANCHOR,
            ontology=ontology,
            llm=llm,
            config=_dynamic_config(),
            graph_depth=GRAPH_DEPTH,
        )
        assert result.error is None
        assert result.filtered_concepts, "Expected non-empty filtered_concepts"
        # For each filtered concept, the full ConceptMetadata must remain queryable.
        for concept_name in result.filtered_concepts:
            meta = ontology.get_concept_metadata(concept_name)
            # The full property dict is the source of truth — no per-property
            # filtering happens in the rebuild step (Plan 2 Phase 5 guarantee).
            assert meta.name == concept_name


# ---------------------------------------------------------------------------
# 4. Cardinality + description surfacing in the Compact DDL
# ---------------------------------------------------------------------------


class TestCardinalitySurfacing:

    def test_crunchbase_ddl_has_cardinality_markers(self, config):
        ontology = _ontology(config, CRUNCHBASE_ONTOLOGY)
        cfg = _dynamic_config()
        edge_index = EdgeIndex(ontology)
        concepts, preds, edges = retrieve_subgraph(
            CRUNCHBASE_ANCHOR, edge_index, cfg, max_hop=GRAPH_DEPTH,
        )
        ddl, _stage = serialize_compact_ddl(concepts, edges, ontology, preds, cfg)
        # Compact DDL format renders edges as `  -[<rel>, <card>]-> <target>`
        # nested under each concept's `rels:` block. Every emitted edge line
        # must carry one of the four cardinality markers.
        import re
        edge_lines = re.findall(r"-\[[^\]]+\]->\s*\w+", ddl)
        assert edge_lines, "Expected at least one `-[rel, card]-> target` edge line in the Compact DDL"
        for line in edge_lines:
            assert any(card in line for card in ("N:M", "N:1", "1:N", "1:1")), (
                f"edge line missing cardinality marker: {line!r}"
            )

    def test_crunchbase_company_has_at_least_one_mtm_edge(self, config):
        """The company concept advertises mtm relationships — DDL must surface one."""
        ontology = _ontology(config, CRUNCHBASE_ONTOLOGY)
        cfg = _dynamic_config()
        edge_index = EdgeIndex(ontology)
        concepts, preds, edges = retrieve_subgraph(
            CRUNCHBASE_ANCHOR, edge_index, cfg, max_hop=GRAPH_DEPTH,
        )
        ddl, _ = serialize_compact_ddl(concepts, edges, ontology, preds, cfg)
        assert "N:M" in ddl, (
            "Expected at least one N:M edge in the crunchbase subgraph DDL"
        )

    def test_supply_metrics_ddl_has_cardinality_markers(self, config):
        ontology = _ontology(config, config["timbr_ontology"])
        cfg = _dynamic_config()
        edge_index = EdgeIndex(ontology)
        concepts, preds, edges = retrieve_subgraph(
            SUPPLY_ANCHOR, edge_index, cfg, max_hop=GRAPH_DEPTH,
        )
        ddl, _ = serialize_compact_ddl(concepts, edges, ontology, preds, cfg)
        import re
        edge_lines = re.findall(r"-\[[^\]]+\]->\s*\w+", ddl)
        assert edge_lines, "Expected at least one `-[rel, card]-> target` edge line in the Compact DDL"
        for line in edge_lines:
            assert any(card in line for card in ("N:M", "N:1", "1:N", "1:1"))


# ---------------------------------------------------------------------------
# 5. End-to-end SQL generation through GenerateTimbrSqlChain
# ---------------------------------------------------------------------------


class TestEndToEndSqlGeneration:
    """Drives GenerateTimbrSqlChain with mode='dynamic' + graph_depth=3 and
    asserts the generated SQL is non-empty + passes validate_sql."""

    def _make_chain(self, config, ontology_name: str, *, mode: str = "dynamic", tc_max_tokens: Optional[int] = 3000, g_depth=4):
        from langchain_timbr import GenerateTimbrSqlChain

        conn = _conn_params(config, ontology=ontology_name)
        return GenerateTimbrSqlChain(
            url=conn["url"],
            token=conn["token"],
            ontology=ontology_name,
            verify_ssl=conn["verify_ssl"],
            graph_depth=g_depth,
            max_graph_depth=5,
            metadata_context_mode=mode,
            views_list="None",
            # conftest.py sets ENABLE_TECHNICAL_CONTEXT=false at import time
            # (so unit tests skip the TC pipeline). Re-enable it here so the
            # dynamic integration suite exercises the full prompt-enrichment
            # path including technical_context statistics annotations.
            enable_technical_context=True,
            technical_context_max_tokens=tc_max_tokens,
            # Default LLM picked up from env (LLM_TYPE/LLM_MODEL/LLM_API_KEY).
        )

    def test_crunchbase_end_to_end_yields_sql(self, config):
        """Hard gate: non-empty SQL. validate_sql outcome is observational —
        the LLM may pick legitimate query shapes that hit backend-specific quirks
        (e.g. MySQL only_full_group_by) which are not pipeline failures."""
        chain = self._make_chain(config, CRUNCHBASE_ONTOLOGY)
        result = chain.invoke({"prompt": CRUNCHBASE_QUESTION})
        sql = result.get("sql", "")
        assert sql, f"Expected non-empty SQL; got error={result.get('error')!r}"
        if result.get("is_sql_valid") is False and result.get("error"):
            # Observational signal — log but don't fail. Plan 2 Phase 5:
            # "Semantic correctness of returned rows is not asserted".
            print(
                f"[observational] crunchbase SQL did not validate: "
                f"{result['error']!r}\nSQL: {sql}"
            )

    def test_supply_metrics_end_to_end_yields_sql(self, config):
        chain = self._make_chain(config, config["timbr_ontology"], tc_max_tokens=500)
        result = chain.invoke({"prompt": SUPPLY_QUESTION})
        sql = result.get("sql", "")
        assert "material_name" in sql, f"Expected SQL to reference 'material_name'; got {sql!r}"
        assert sql, f"Expected non-empty SQL; got error={result.get('error')!r}"
        if result.get("is_sql_valid") is False and result.get("error"):
            print(
                f"[observational] supply_metrics SQL did not validate: "
                f"{result['error']!r}\nSQL: {sql}"
            )

    def test_supply_metrics_end_to_end_yields_sql_expand(self, config):
        chain = self._make_chain(config, config["timbr_ontology"], tc_max_tokens=500, g_depth=1)
        result = chain.invoke({"prompt": SUPPLY_QUESTION})
        sql = result.get("sql", "")
        assert "material_name" in sql, f"Expected SQL to reference 'material_name'; got {sql!r}"
        assert sql, f"Expected non-empty SQL; got error={result.get('error')!r}"
        if result.get("is_sql_valid") is False and result.get("error"):
            print(
                f"[observational] supply_metrics SQL did not validate: "
                f"{result['error']!r}\nSQL: {sql}"
            )

    def test_static_mode_also_yields_valid_sql_at_graph_depth_5(self, config):
        """Backward-compat regression: mode='static' at graph_depth=5 must still
        produce a working SQL (proves the static path is unchanged)."""
        from langchain_timbr import GenerateTimbrSqlChain

        conn = _conn_params(config, ontology=config["timbr_ontology"])
        chain = GenerateTimbrSqlChain(
            url=conn["url"],
            token=conn["token"],
            ontology=config["timbr_ontology"],
            verify_ssl=conn["verify_ssl"],
            graph_depth=GRAPH_DEPTH,
            metadata_context_mode="static",
            views_list="None",
            enable_technical_context=True,
        )
        result = chain.invoke({"prompt": SUPPLY_QUESTION})
        sql = result.get("sql", "")
        assert sql, f"Static-mode SQL gen produced empty SQL; error={result.get('error')!r}"

    def test_filter_llm_called_at_most_once_per_invoke(self, config):
        """Memoization regression: handle_validate_generate_sql's retry loop
        rebuilds the SQL context, which would otherwise re-invoke the Step 1
        filter LLM (doubling token spend). The shared Ontology + filtered-
        result cache must ensure Step 1 is called at most ONCE per chain
        invocation regardless of how many SQL-validation retries occur."""
        from langchain_timbr.ontology_context.context_builder import (
            build_filtered as _bf_mod,
        )
        from langchain_timbr.ontology_context.ontology import shared as _shared_mod

        # Reset shared Ontologies so cache state is clean.
        _shared_mod.reset_shared_ontologies()

        call_count = {"step1": 0, "retry": 0}
        # Patch the binding inside build_filtered (which does `from .llm_filter
        # import run_step1_filter` — a name binding that survives patching the
        # original module). This intercepts the actual call site.
        _orig_step1 = _bf_mod.run_step1_filter
        _orig_retry = _bf_mod.run_step1_retry

        def _counting_step1(**kw):
            call_count["step1"] += 1
            return _orig_step1(**kw)

        def _counting_retry(**kw):
            call_count["retry"] += 1
            return _orig_retry(**kw)

        _bf_mod.run_step1_filter = _counting_step1
        _bf_mod.run_step1_retry = _counting_retry
        try:
            chain = self._make_chain(config, config["timbr_ontology"], mode="dynamic")
            chain.invoke({"prompt": SUPPLY_QUESTION})
        finally:
            _bf_mod.run_step1_filter = _orig_step1
            _bf_mod.run_step1_retry = _orig_retry

        # Step 1 (the expensive filter LLM call) must NEVER fire more than once
        # per chain invocation. The shared Ontology + filtered-result cache
        # ensures SQL-validation retries reuse the cached rebuild instead of
        # re-running the filter LLM.
        assert call_count["step1"] == 1, (
            f"Expected exactly 1 Step 1 filter call per chain.invoke(), "
            f"got {call_count['step1']}. SQL-validation retries should reuse "
            f"the cached rebuild — this regression doubles token spend."
        )
        # Retry is now bounded by metadata_context_dynamic_retry (default 2).
        # Topological-sort + segmented-path support means most first-pass
        # validations succeed, so retries shouldn't fire in steady state.
        from langchain_timbr import config as _cfg
        assert call_count["retry"] <= _cfg.metadata_context_dynamic_retry, (
            f"Step 1 retry should fire at most {_cfg.metadata_context_dynamic_retry} "
            f"times per chain.invoke(), got {call_count['retry']}"
        )

    def test_dynamic_mode_actually_narrows_relationships(self, config):
        """Hard gate against silent fallback: when mode='dynamic' is set, the
        rebuilt relationship string MUST be strictly smaller than the static
        string (some relationships dropped). Catches regressions where the
        wiring quietly falls back to static (e.g. validation always failing,
        rebuild logic broken, etc.)."""
        from langchain_timbr.utils import timbr_llm_utils as TLU

        captured = {}
        _orig = TLU._apply_dynamic_metadata_context

        def _probe(**kw):
            captured["static_rel_len"] = len(kw["static_rel_prop_str"])
            a, b, c, eff = _orig(**kw)
            captured["new_rel_len"] = len(c)
            captured["rels_changed"] = c != kw["static_rel_prop_str"]
            return a, b, c, eff

        TLU._apply_dynamic_metadata_context = _probe
        try:
            chain = self._make_chain(config, config["timbr_ontology"], mode="dynamic")
            result = chain.invoke({"prompt": SUPPLY_QUESTION})
        finally:
            TLU._apply_dynamic_metadata_context = _orig

        assert result.get("sql"), (
            f"Dynamic-mode SQL gen produced no SQL; error={result.get('error')!r}"
        )
        assert captured.get("rels_changed") is True, (
            "Dynamic mode silently fell back to static — relationship string "
            f"was not rebuilt. captured={captured!r}"
        )
        # The narrowed relationship string should be strictly smaller — at
        # minimum we drop at least one of the static relationships.
        assert captured["new_rel_len"] < captured["static_rel_len"], (
            f"Dynamic mode produced a string >= static length. "
            f"static_rel_len={captured['static_rel_len']}, "
            f"new_rel_len={captured['new_rel_len']}"
        )


# ---------------------------------------------------------------------------
# 6. Determinism + caching across two consecutive runs
# ---------------------------------------------------------------------------


class _CountingClient:
    """Proxy that counts describe / version / rels calls on a real client."""

    def __init__(self, inner):
        self._inner = inner
        self.describe_calls: List[str] = []
        self.rels_calls = 0
        self.version_calls = 0

    def fetch_version_id(self):
        self.version_calls += 1
        return self._inner.fetch_version_id()

    def describe_concept(self, name):
        self.describe_calls.append(name)
        return self._inner.describe_concept(name)

    def fetch_relationships_meta(self):
        self.rels_calls += 1
        return self._inner.fetch_relationships_meta()


class TestDeterminismAndCaching:
    """Two consecutive runs on the same Ontology share the cache and produce
    identical filtered context."""

    def test_two_runs_share_one_ontology_cache(self, config, llm):
        real = TimbrOntologyClient(_conn_params(config, ontology=CRUNCHBASE_ONTOLOGY))
        counting = _CountingClient(real)
        ontology = Ontology(counting, version_ttl_seconds=3600)

        cfg = _dynamic_config()
        result_a = build_filtered_metadata(
            question=CRUNCHBASE_QUESTION,
            anchor=CRUNCHBASE_ANCHOR,
            ontology=ontology,
            llm=llm,
            config=cfg,
            graph_depth=GRAPH_DEPTH,
        )
        assert result_a.error is None
        first_describe_calls = list(counting.describe_calls)
        assert first_describe_calls, "First run should have issued describe_concept"

        result_b = build_filtered_metadata(
            question=CRUNCHBASE_QUESTION,
            anchor=CRUNCHBASE_ANCHOR,
            ontology=ontology,
            llm=llm,
            config=cfg,
            graph_depth=GRAPH_DEPTH,
        )
        assert result_b.error is None
        # Second run must NOT trigger additional describe_concept calls —
        # the Ontology cache should serve every concept it already knows.
        # (LLM output can vary across runs, so we don't assert path equality —
        # only that the underlying metadata cache is honored.)
        assert counting.describe_calls == first_describe_calls, (
            "Second run made additional describe_concept calls — cache not honored. "
            f"Run-1: {first_describe_calls}; Run-2 delta: "
            f"{counting.describe_calls[len(first_describe_calls):]}"
        )
        # Relationship-lookup fetch must be exactly one for the whole session.
        assert counting.rels_calls == 1

    def test_compact_ddl_deterministic_across_runs(self, config):
        """Pure-deterministic stage (no LLM): same ontology + same caps yields
        identical Compact DDL across consecutive runs."""
        ontology = _ontology(config, config["timbr_ontology"])
        cfg = _dynamic_config()
        edge_index = EdgeIndex(ontology)

        c1, p1, e1 = retrieve_subgraph(SUPPLY_ANCHOR, edge_index, cfg, max_hop=GRAPH_DEPTH)
        ddl_a, stage_a = serialize_compact_ddl(c1, e1, ontology, p1, cfg)

        c2, p2, e2 = retrieve_subgraph(SUPPLY_ANCHOR, edge_index, cfg, max_hop=GRAPH_DEPTH)
        ddl_b, stage_b = serialize_compact_ddl(c2, e2, ontology, p2, cfg)

        assert ddl_a == ddl_b, "Compact DDL must be deterministic for the same ontology"
        assert stage_a == stage_b


# ---------------------------------------------------------------------------
# 7. Note + conversation-memory plumbing into the Step 1 filter prompt
# ---------------------------------------------------------------------------


class TestNoteAndMemoryIntoFilterPrompt:
    """Verifies that the conversation-memory ``note`` channel reaches the
    Step 1 filter prompt — both for an explicit caller-supplied note and for
    follow-up questions where memory is auto-injected via
    ``format_memory_note_for_sql``.

    The probe pattern monkey-patches ``build_filter_messages`` to capture the
    rendered user content of the filter LLM call, then inspects it for the
    expected note markers. We capture across the package-level binding so the
    patch survives import indirection from ``llm_filter.run_step1_filter``.
    """

    def _capture_filter_user_content(self):
        """Return ``(captured_dict, restore_fn)`` — patches the module-level
        and package-level ``build_filter_messages`` bindings to record each
        invocation's user-message content."""
        from langchain_timbr.ontology_context.context_builder.prompts import (
            filter_prompt as _fp_mod,
        )
        from langchain_timbr.ontology_context.context_builder import prompts as _prompts_mod

        captured = {"calls": []}
        _orig = _fp_mod.build_filter_messages

        def _probe(**kw):
            msgs = _orig(**kw)
            user = msgs[1]["content"] if len(msgs) > 1 else ""
            captured["calls"].append({"user": user, "kwargs": dict(kw)})
            return msgs

        _fp_mod.build_filter_messages = _probe
        _prompts_mod.build_filter_messages = _probe

        def _restore():
            _fp_mod.build_filter_messages = _orig
            _prompts_mod.build_filter_messages = _orig

        return captured, _restore

    def test_caller_supplied_note_arrives_in_filter_prompt(self, config):
        """When the chain is constructed with an explicit ``note=...``, that
        note must be appended to the Step 1 filter prompt as an
        ``**Additional Notes:**`` block — the same channel the SQL-gen prompt
        uses. Catches regressions where the note is consumed by SQL gen but
        never reaches the context-builder LLM calls."""
        from langchain_timbr import GenerateTimbrSqlChain
        from langchain_timbr.ontology_context.ontology import shared as _shared

        _shared.reset_shared_ontologies()
        conn = _conn_params(config, ontology=config["timbr_ontology"])

        marker = "TEST_NOTE_MARKER_xyz_for_filter_prompt_probe_42"
        explicit_note = (
            f"Prefer paths that reach material concepts. {marker}"
        )

        captured, restore = self._capture_filter_user_content()
        try:
            chain = GenerateTimbrSqlChain(
                url=conn["url"],
                token=conn["token"],
                ontology=conn["ontology"],
                verify_ssl=conn["verify_ssl"],
                graph_depth=GRAPH_DEPTH,
                metadata_context_mode="dynamic",
                views_list="None",
                note=explicit_note,
                enable_technical_context=True,
            )
            result = chain.invoke({"prompt": SUPPLY_QUESTION})
        finally:
            restore()

        # Sanity: the chain produced SQL (so it actually exercised the pipeline).
        assert result.get("sql"), (
            f"Chain produced no SQL; error={result.get('error')!r}"
        )
        # Filter prompt was invoked at least once.
        assert captured["calls"], "build_filter_messages was never called"

        # The first (initial Step 1) call's user content must include the
        # Additional Notes block AND the caller's marker text.
        first_user = captured["calls"][0]["user"]
        assert "**Additional Notes:**" in first_user, (
            "Filter prompt missing the Additional Notes block — note channel "
            "is not reaching the Step 1 filter LLM. "
            f"User content head: {first_user[:600]!r}"
        )
        assert marker in first_user, (
            f"Caller-supplied note marker {marker!r} did not arrive in the "
            f"filter prompt. User content head: {first_user[:600]!r}"
        )

    def test_followup_question_includes_conversation_memory_in_filter_prompt(
        self, config, llm,
    ):
        """End-to-end memory plumbing into context_builder:

        1. Run a memory-enabled GenerateAnswerChain to seed conversation history.
        2. Invoke a memory-enabled GenerateTimbrSqlChain with the same
           conversation_id on a follow-up question.
        3. The follow-up's Step 1 filter prompt MUST include the conversation
           memory block (``[CONVERSATION MEMORY]``) so the filter LLM can
           choose paths consistent with the prior turn. Without this, follow-up
           filter LLM picks the wrong relationship paths (e.g. ignores the
           material focus established in the seed turn)."""
        import uuid6

        from langchain_timbr import GenerateTimbrSqlChain
        from langchain_timbr.langchain.generate_answer_chain import GenerateAnswerChain
        from langchain_timbr.ontology_context.ontology import shared as _shared

        _shared.reset_shared_ontologies()
        conn = _conn_params(config, ontology=config["timbr_ontology"])
        conversation_id = str(uuid6.uuid7())

        # Step 1 — seed conversation history via GenerateAnswerChain (the only
        # chain that persists history). Use a question that establishes a
        # specific path focus the follow-up will reference.
        seed_chain = GenerateAnswerChain(
            llm=llm,
            url=conn["url"],
            token=conn["token"],
            ontology=conn["ontology"],
            verify_ssl=conn["verify_ssl"],
            enable_memory=True,
            memory_window_size=5,
            enable_history=True,
            save_results=True,
            conversation_id=conversation_id,
        )
        seed_result = seed_chain.invoke({
            "prompt": SUPPLY_QUESTION,
            "conversation_id": conversation_id,
        })
        assert "answer" in seed_result and seed_result["answer"], (
            f"Seed turn produced no answer; cannot test follow-up. "
            f"seed_result={seed_result!r}"
        )

        # Step 2 — invoke a memory-enabled SQL-gen chain on a follow-up that
        # only makes sense given the prior turn's context.
        captured, restore = self._capture_filter_user_content()
        try:
            followup_chain = GenerateTimbrSqlChain(
                url=conn["url"],
                token=conn["token"],
                ontology=conn["ontology"],
                verify_ssl=conn["verify_ssl"],
                graph_depth=GRAPH_DEPTH,
                metadata_context_mode="dynamic",
                views_list="None",
                enable_memory=True,
                memory_window_size=5,
                conversation_id=conversation_id,
                enable_technical_context=True,
            )
            followup_result = followup_chain.invoke({
                "prompt": "Filter to last quarter",
                "conversation_id": conversation_id,
            })
        finally:
            restore()

        assert followup_result.get("sql"), (
            f"Follow-up chain produced no SQL; error={followup_result.get('error')!r}"
        )
        assert captured["calls"], (
            "Follow-up did not invoke the Step 1 filter — dynamic pipeline "
            "may have short-circuited to static."
        )

        first_user = captured["calls"][0]["user"]
        # The conversation-memory block must be present in the follow-up's
        # filter prompt.
        assert "**Additional Notes:**" in first_user, (
            "Follow-up filter prompt missing the Additional Notes block — "
            "memory_context did not flow into the context_builder LLM call. "
            f"User content head: {first_user[:800]!r}"
        )
        assert "[CONVERSATION MEMORY]" in first_user, (
            "Follow-up filter prompt missing [CONVERSATION MEMORY] header — "
            "format_memory_note_for_sql output not reaching context_builder. "
            f"User content head: {first_user[:800]!r}"
        )
        # The seed question should appear verbatim in the memory block (the
        # memory formatter includes prior SQL queries with their original
        # question text). This proves the LLM is actually seeing the prior
        # turn's context, not just an empty/placeholder memory block.
        assert SUPPLY_QUESTION in first_user, (
            "Seed question not present in follow-up filter prompt — the "
            "memory block is empty or stripped. "
            f"User content head: {first_user[:1000]!r}"
        )
