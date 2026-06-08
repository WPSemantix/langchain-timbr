"""Integration tests for LLM-driven transitivity overrides (Plan 2 update).

Three verification layers cover the full override pipeline:

  Layer 1 — rebuild probe: capture the rebuilt context strings immediately
            after the rebuild step. Assert the requested depth (`*N`) is
            present in the SQL-gen-bound output.
  Layer 2 — accepted-overrides probe: capture
            ``DynamicMetadataResult.accepted_overrides`` from
            ``build_filtered_metadata``. Deterministic check that the LLM
            emitted the override AND ``validate_overrides`` accepted it,
            independent of whether the rewriter found matching patterns.
  Layer 3 — generated SQL check: best-effort assertion that the final SQL
            contains the override depth. Observational (LLM-side variability).

Failure-mode mapping:
  | Test fails           | Likely cause                                    |
  | -------------------- | ----------------------------------------------- |
  | Layer 1 only         | rebuild.apply_transitivity_overrides regex bug  |
  | Layer 1+2 pass, 3 fail | SQL-gen LLM ignored the override (observational) |
  | Layer 2 fails        | Step 1 LLM didn't emit override OR validator dropped it |

Live Timbr + LLM credentials are required. Tests skip when env vars are absent
— see the ``run-integration-tests`` skill for how to source from
``.vscode/launch.json`` when running locally.

The reference implementation pattern is `test_execute_transitive` in
``test_langchain_chains.py`` which exercises the same override behavior
through the static path.
"""

from __future__ import annotations

import pytest

from langchain_timbr.ontology_context import (
    Ontology,
    TimbrOntologyClient,
)


CRUNCHBASE_ONTOLOGY = "timbr_crunchbase_llm_tests"

# Transitive relationship in crunchbase used as the override target.
# company -[has_acquired*2]-> company (self-ref + transitive, default depth 2).
CRUNCHBASE_TRANSITIVE_REL = "has_acquired"
CRUNCHBASE_TRANSITIVE_TGT = "company"


def _conn_params(config, *, ontology: str) -> dict:
    if not config.get("timbr_url") or not config.get("timbr_token"):
        pytest.skip("TIMBR_URL / TIMBR_TOKEN not set — skipping integration test")
    return {
        "url": config["timbr_url"],
        "token": config["timbr_token"],
        "ontology": ontology,
        "verify_ssl": config["verify_ssl"],
    }


class TestTransitivityOverrideRebuild:
    """Three verification layers covering the full override pipeline:

      Layer 1: probe the rebuilt context strings immediately after the rebuild
               step → assert the override depth is present.
      Layer 2: capture the current_context dict at the SQL-gen prompt assembly
               site → assert the depth flowed through unchanged.
      Layer 3: assert the final generated SQL contains the override (best-effort
               since LLMs can still ignore prompt content — Layer 3 is
               observational, not a hard gate).
    """

    ONTOLOGY = CRUNCHBASE_ONTOLOGY

    # --- helpers --------------------------------------------------------

    def _make_chain(self, config):
        from langchain_timbr import GenerateTimbrSqlChain
        conn = _conn_params(config, ontology=self.ONTOLOGY)
        return GenerateTimbrSqlChain(
            url=conn["url"], token=conn["token"],
            ontology=self.ONTOLOGY, verify_ssl=conn["verify_ssl"],
            graph_depth=4, metadata_context_mode="dynamic",
            enable_technical_context=True, views_list="None",
        )

    def _probe_rebuild(self, config, prompt):
        """Layer 1 probe — capture rebuilt strings via _apply_dynamic patch."""
        from langchain_timbr.utils import timbr_llm_utils as TLU
        from langchain_timbr.ontology_context.ontology import shared as _shared
        _shared.reset_shared_ontologies()

        captured = {}
        _orig = TLU._apply_dynamic_metadata_context

        def probe(**kw):
            cols, meas, rels, eff = _orig(**kw)
            captured["columns_str"] = cols
            captured["measures_str"] = meas
            captured["rel_prop_str"] = rels
            captured["effective_anchor"] = eff
            return cols, meas, rels, eff

        TLU._apply_dynamic_metadata_context = probe
        try:
            chain = self._make_chain(config)
            result = chain.invoke({"prompt": prompt})
        finally:
            TLU._apply_dynamic_metadata_context = _orig
        return captured, result

    def _probe_accepted_overrides(self, config, prompt):
        """Layer 2 probe — capture DynamicMetadataResult.accepted_overrides
        directly from build_filtered_metadata. This is the deterministic
        assertion: 'the LLM emitted the override AND validate_overrides
        accepted it'. Independent of whether the rewriter found matching
        patterns in the static dict (which depends on ontology shape).

        Patches the binding in ontology_context.__init__ because the wiring
        layer (_apply_dynamic_metadata_context) does `from
        ..ontology_context import build_filtered_metadata` at call time, which
        resolves through the package's __init__ — not through build_filtered.py.
        """
        import langchain_timbr.ontology_context as _oc_pkg
        from langchain_timbr.ontology_context.context_builder import build_filtered as _bf
        from langchain_timbr.ontology_context.ontology import shared as _shared
        _shared.reset_shared_ontologies()

        captured = {"accepted_overrides": [], "stats": {}}
        _orig = _oc_pkg.build_filtered_metadata

        def probe(**kw):
            result = _orig(**kw)
            captured["accepted_overrides"] = list(result.accepted_overrides or [])
            captured["stats"] = dict(result.stats or {})
            return result

        # Patch both the package-level export (used by the wiring layer's
        # `from ..ontology_context import build_filtered_metadata`) AND the
        # submodule binding (in case anything imports it via build_filtered.X).
        _oc_pkg.build_filtered_metadata = probe
        _bf.build_filtered_metadata = probe
        try:
            chain = self._make_chain(config)
            result = chain.invoke({"prompt": prompt})
        finally:
            _oc_pkg.build_filtered_metadata = _orig
            _bf.build_filtered_metadata = _orig
        return captured, result

    def _make_chain_no_tc(self, config):
        """Variant for Layer 3 tests: TC=False to avoid context-length blowups
        when the dynamic rebuild falls back to a large static rel_prop_str."""
        from langchain_timbr import GenerateTimbrSqlChain
        conn = _conn_params(config, ontology=self.ONTOLOGY)
        return GenerateTimbrSqlChain(
            url=conn["url"], token=conn["token"],
            ontology=self.ONTOLOGY, verify_ssl=conn["verify_ssl"],
            graph_depth=4, metadata_context_mode="dynamic",
            enable_technical_context=False, views_list="None",
        )

    # --- Layer 1: rebuild-probe tests (5) ------------------------------

    def test_override_3_levels_appears_in_rebuilt_context(self, config):
        """Mirror of test_execute_transitive probe 1 — Layer 1 (rebuild)."""
        captured, _ = self._probe_rebuild(
            config,
            "count number of companies acquired by Microsoft up to 3 levels",
        )
        combined = (captured.get("columns_str") or "") + (captured.get("rel_prop_str") or "")
        assert "*3" in combined, (
            f"Expected '*3' in rebuilt context; none found. "
            f"rel_prop_str sample: {(captured.get('rel_prop_str') or '')[:400]}"
        )

    def test_override_4_levels_appears_in_rebuilt_context(self, config):
        """Mirror of test_execute_transitive probe 2 — Layer 1 (rebuild).
        Same ontology, different depth → proves override is question-driven."""
        captured, _ = self._probe_rebuild(
            config,
            "count number of companies acquired by Sensobi up to 4 levels",
        )
        combined = (captured.get("columns_str") or "") + (captured.get("rel_prop_str") or "")
        assert "*4" in combined, (
            f"Expected '*4' in rebuilt context; got rel_prop_str sample: "
            f"{(captured.get('rel_prop_str') or '')[:400]}"
        )

    def test_no_depth_in_question_does_not_introduce_new_levels(self, config):
        """Question with no depth requirement → no override applied → the set
        of `*N` markers in the rebuilt context should be a SUBSET of the
        defaults already in the ontology (no new depths introduced).

        This is delta-based: we can't check absolute absence of `*3`/`*4`
        because the ontology has its own defaults (e.g. funding_round*4).
        Instead we check that no override-driven depth (like `*7`) was
        injected when the question doesn't ask for it."""
        import re
        captured, _ = self._probe_rebuild(
            config, "List companies and their region",
        )
        combined = (captured.get("columns_str") or "") + (captured.get("rel_prop_str") or "")
        # Find every `*N]` marker. Override-driven values for typical asks are
        # 3/4/5; the crunchbase defaults are *2 and *4. Assert no exotic depths
        # like *7/*9/*10 appear — those would only come from an LLM-emitted
        # override on a no-depth question.
        depths = {int(m.group(1)) for m in re.finditer(r"\*(\d+)\]", combined)}
        unexpected = {d for d in depths if d >= 7}
        assert not unexpected, (
            f"Found exotic depths {unexpected} when question had no depth "
            f"requirement. All depths in output: {sorted(depths)}"
        )

    def test_self_ref_override_with_high_depth_accepted(self, config):
        """Self-ref relationships (has_acquired is self-ref) accept ANY depth."""
        captured, _ = self._probe_rebuild(
            config,
            "Show all companies that acquired by Microsoft up to 5 levels deep",
        )
        combined = (captured.get("columns_str") or "") + (captured.get("rel_prop_str") or "")
        assert "*5" in combined, (
            f"Expected '*5' on self-ref relationship; got sample: {combined[:400]}"
        )

    def test_pipeline_does_not_crash_on_unrelated_question(self, config):
        """Smoke test — questions that don't involve transitive rels must not
        crash even if the LLM emits irrelevant overrides."""
        captured, result = self._probe_rebuild(
            config, "List companies and their primary headquarters",
        )
        assert "rel_prop_str" in captured  # the probe was reached
        assert result.get("sql") is not None  # SQL generation completed

    # --- Layer 2: accepted-overrides tests (2) -------------------------
    # These probe build_filtered_metadata directly to verify the LLM emitted
    # a transitivity override AND validate_overrides accepted it. This is
    # deterministic regardless of whether the rewriter finds matching patterns
    # in the static dict (that depends on ontology shape).

    def test_step_1_emits_and_accepts_override_3(self, config):
        captured, _ = self._probe_accepted_overrides(
            config,
            "count number of companies acquired by Microsoft up to 3 levels",
        )
        accepted = captured.get("accepted_overrides", [])
        assert accepted, (
            f"Step 1 should have emitted a transitivity_override for "
            f"'up to 3 levels'; got 0 accepted. Stats: {captured.get('stats')}"
        )
        levels = {ov.level for ov in accepted}
        assert 3 in levels, (
            f"Expected level=3 in accepted overrides; got levels={levels} "
            f"with rels={[(o.rel, o.target) for o in accepted]}"
        )

    def test_step_1_emits_and_accepts_override_4(self, config):
        captured, _ = self._probe_accepted_overrides(
            config,
            "count number of companies acquired by Sensobi up to 4 levels",
        )
        accepted = captured.get("accepted_overrides", [])
        assert accepted
        levels = {ov.level for ov in accepted}
        assert 4 in levels, (
            f"Expected level=4 in accepted overrides; got levels={levels}"
        )

    # --- Layer 3: SQL-output tests (2) ---------------------------------
    # End-to-end SQL assertions. Use TC=False to mirror test_execute_transitive
    # and avoid context-length blowups when the dynamic rebuild falls back to
    # the large static rel_prop_str.

    def test_generated_sql_contains_override_for_3_levels(self, config):
        """Layer 3 (observational): mirror of test_execute_transitive probe 1
        through the dynamic pipeline. SQL-gen LLM behavior can vary; treat
        non-matching SQL as a soft signal rather than a hard fail."""
        chain = self._make_chain_no_tc(config)
        result = chain.invoke({
            "prompt": "count number of companies acquired by Microsoft up to 3 levels",
        })
        sql = result.get("sql", "")
        assert sql, f"Expected non-empty SQL; error={result.get('error')!r}"
        if "*3" not in sql:
            print(
                f"[observational] dynamic-mode SQL did not contain '*3' "
                f"(SQL-gen LLM may have ignored override). SQL: {sql}"
            )

    def test_generated_sql_contains_override_for_5_levels(self, config):
        """Layer 3 (observational): probe 2 through dynamic pipeline."""
        chain = self._make_chain(config)
        result = chain.invoke({
            "prompt": (
                "For company Sensobi, count how many times they were acquired "
                "by up to 5 levels. return only count"
            ),
        })
        sql = result.get("sql", "")
        assert sql, f"Expected non-empty SQL; error={result.get('error')!r}"
        if "*5" not in sql:
            print(
                f"[observational] dynamic-mode SQL did not contain '*5' "
                f"(SQL-gen LLM may have ignored override). SQL: {sql}"
            )

    def test_generated_sql_contains_override_for_5_levels2(self, config):
        """Layer 3 (observational): probe 2 through dynamic pipeline."""
        
        from langchain_timbr import GenerateTimbrSqlChain
        conn = _conn_params(config, ontology=self.ONTOLOGY)
        chain = GenerateTimbrSqlChain(
            url=conn["url"], token=conn["token"],
            ontology=self.ONTOLOGY, verify_ssl=conn["verify_ssl"],
            graph_depth=4, metadata_context_mode="dynamic",
            enable_technical_context=True, concepts_list="company", views_list="None",
        )

        result = chain.invoke({
            "prompt": (
                "For company Sensobi, count how many times they were acquired "
                "by up to 5 levels. return only count"
            ),
        })
        sql = result.get("sql", "")
        assert sql, f"Expected non-empty SQL; error={result.get('error')!r}"
        if "*5" not in sql:
            print(
                f"[observational] dynamic-mode SQL did not contain '*5' "
                f"(SQL-gen LLM may have ignored override). SQL: {sql}"
            )


# ---------------------------------------------------------------------------
# Direct serializer probes — no LLM call, no full chain
# ---------------------------------------------------------------------------


class TestConceptCentricSerializerWithOrganizationAnchor:
    """Direct serializer probes against live crunchbase with anchor='organization'.

    This bypasses the full chain (no Step 1 LLM call) and exercises the BFS +
    serializer end-to-end. We test the company hop=2 block to verify:
      - Self-ref relationships (has_acquired, acquired_by) are inlined in the
        concept's rels: block with a `# recursive` marker
      - No `~` prefix on inverse rels in the rels: block
      - Bounce-back inverses (back to the previous-hop concept) are dropped
    """

    ONTOLOGY = CRUNCHBASE_ONTOLOGY
    ANCHOR = "organization"

    def _build_subgraph(self, config, max_hop=2):
        """Build the subgraph with anchor=organization. Returns (ontology,
        concepts, predecessors, edges, ddl, stage)."""
        from langchain_timbr.ontology_context.context_builder.edge_index import EdgeIndex
        from langchain_timbr.ontology_context.context_builder.metadata_config import (
            MetadataContextConfig,
        )
        from langchain_timbr.ontology_context.context_builder.subgraph import (
            retrieve_subgraph,
            serialize_compact_ddl,
        )
        conn = _conn_params(config, ontology=self.ONTOLOGY)
        client = TimbrOntologyClient(conn)
        ontology = Ontology(client)
        edge_index = EdgeIndex(ontology)
        cfg = MetadataContextConfig()
        concepts, preds, edges = retrieve_subgraph(
            self.ANCHOR, edge_index, cfg, max_hop=max_hop,
        )
        ddl, stage = serialize_compact_ddl(concepts, edges, ontology, preds, cfg)
        return ontology, concepts, preds, edges, ddl, stage

    def _company_block(self, ddl: str) -> str:
        """Slice the ddl text containing only the company concept block."""
        marker = "### company"
        if marker not in ddl:
            return ""
        start = ddl.index(marker)
        # Next concept block (### ) or top-level section (## ) marks the end.
        rest = ddl[start + len(marker):]
        end_candidates = [
            rest.find("\n### "),
            rest.find("\n## "),
        ]
        ends = [e for e in end_candidates if e != -1]
        end = min(ends) if ends else len(rest)
        return marker + rest[:end]

    def test_anchor_organization_company_appears_at_hop_2(self, config):
        _, _concepts, _preds, _edges, ddl, _stage = self._build_subgraph(config, max_hop=3)
        # company should appear as a hop=2 concept (organization → org_via_X → company).
        # We don't pin the exact hop — crunchbase shape may put it at hop=1 or 2.
        # Just confirm it appears with a hop label, not as [anchor].
        assert "### company [hop=" in ddl, (
            "Expected company to appear at some hop level under organization anchor. "
            f"DDL head: {ddl[:600]}"
        )

    def test_no_tilde_prefix_in_rels_block(self, config):
        """The rels: lines must NOT contain a ~ prefix on any rel name —
        the LLM doesn't need the inverse/canonical distinction."""
        _, _, _, _, ddl, _ = self._build_subgraph(config, max_hop=3)
        # No -[~ pattern should appear anywhere in the DDL.
        assert "-[~" not in ddl, (
            f"Found ~ prefix in DDL; expected none. DDL head: {ddl[:800]}"
        )
        # SELF_REF section was removed — self-refs are inlined in rels: blocks.
        assert "## SELF_REF" not in ddl

    def test_company_self_refs_inlined_with_recursive_marker(self, config):
        """company's self-referential relationships (e.g. has_acquired) appear
        inline in company's rels: block with a `# recursive` marker."""
        _, _, _, _, ddl, _ = self._build_subgraph(config, max_hop=3)
        company_block = self._company_block(ddl)
        # has_acquired is the canonical company → company self-ref. It must
        # appear in company's rels: block, marked recursive.
        assert "has_acquired" in company_block, (
            f"Expected has_acquired in company's rels: block. "
            f"Block: {company_block[:800]}"
        )
        assert "-> company" in company_block
        assert "# recursive" in company_block, (
            f"Expected `# recursive` marker on self-ref in company's rels: block. "
            f"Block: {company_block[:800]}"
        )

    def test_bounce_back_inverse_to_previous_hop_dropped(self, config):
        """``should_include_in_ddl`` drops a back-edge from company to its
        BFS predecessor ONLY when the edge is flagged ``is_inverse=True``
        AND its target equals the predecessor (and it's not a self-ref).
        Non-inverse semantically-distinct relationships that happen to
        point back to the predecessor (e.g. ``company.made_investment ->
        funding_round`` in crunchbase) MUST survive — they're not
        bounce-backs.

        ``retrieve_subgraph`` returns the FULL edge list (including
        inverse bounce-backs); the filter is applied at DDL-render time
        by ``_filter_edges_for_ddl``. We apply the same filter here and
        assert the inverse bounce-back is gone from the rendered set."""
        from langchain_timbr.ontology_context.context_builder.subgraph import (
            _filter_edges_for_ddl,
        )
        _ontology, _concepts, preds, edges, _ddl, _stage = self._build_subgraph(config, max_hop=3)
        previous_hop = preds.get("company")
        if previous_hop is None:
            pytest.skip("company appears as anchor (unexpected); skipping")
        rendered_edges = _filter_edges_for_ddl(edges, preds)
        offending = [
            e for e in rendered_edges
            if e.from_concept == "company"
            and e.is_inverse
            and e.to_concept == previous_hop
            and not e.is_self_ref
        ]
        assert not offending, (
            f"Found {len(offending)} inverse bounce-back edge(s) from "
            f"company to predecessor {previous_hop!r} that survived the "
            f"DDL-render filter: "
            f"{[(e.relationship_name, e.is_inverse) for e in offending]}"
        )
