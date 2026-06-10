"""Tests for the LLM concept pre-filter — token estimation, adaptive
truncation strategy, output validation, defensive fallbacks."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List

import pytest

from langchain_timbr.ontology_context.context_builder.concept_prefilter import (
    _Candidate,
    apply_truncation_and_render,
    build_prefilter_prompt,
    estimate_full_ddl_tokens,
    render_names_only,
    render_with_descriptions,
    run_concept_prefilter,
    truncate_to_tokens,
)
from langchain_timbr.ontology_context.context_builder.metadata_config import (
    MetadataContextConfig,
)


# ---------------------------------------------------------------------------
# FakeOntology — minimal stand-in returning ConceptMetadata-shaped objects
# ---------------------------------------------------------------------------

@dataclass
class _FakeMeta:
    name: str
    description: str = ""
    properties: dict = field(default_factory=dict)
    measures: dict = field(default_factory=dict)
    relationships: dict = field(default_factory=dict)


class FakeOntology:
    def __init__(self, concept_descriptions: Dict[str, str]):
        self._descs = concept_descriptions

    def get_concept_metadata(self, name: str) -> _FakeMeta:
        if name not in self._descs:
            raise KeyError(name)
        return _FakeMeta(name=name, description=self._descs[name])


class FakeLLM:
    """Returns a pre-canned JSON string for every invoke call."""

    def __init__(self, payload):
        if isinstance(payload, (dict, list)):
            payload = json.dumps(payload)
        self.payload = payload
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)

        class _R:
            def __init__(self, c):
                self.content = c

        return _R(self.payload)


# ---------------------------------------------------------------------------
# truncate_to_tokens — word-boundary respect
# ---------------------------------------------------------------------------


class TestTruncateToTokens:
    def test_short_text_passthrough(self):
        text = "hello world"
        assert truncate_to_tokens(text, 50) == text

    def test_truncation_respects_word_boundaries(self):
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        out = truncate_to_tokens(text, 4)
        # Whatever the boundary is, must end at a word boundary — no partial word.
        assert all(out.startswith(prefix) is False or out.endswith(prefix.split()[-1])
                   for prefix in [out])
        # No trailing mid-word artifact: split on whitespace and confirm the
        # last token in the output is also a whole token from the input.
        if out:
            assert out.split()[-1] in text.split()

    def test_target_zero_returns_empty(self):
        assert truncate_to_tokens("anything goes here", 0) == ""

    def test_empty_input_returns_empty(self):
        assert truncate_to_tokens("", 10) == ""


# ---------------------------------------------------------------------------
# estimate_full_ddl_tokens
# ---------------------------------------------------------------------------


class TestEstimateFullDdlTokens:
    def test_estimate_grows_with_concept_count(self):
        ontology = FakeOntology({f"C{i}": "" for i in range(10)})
        small = estimate_full_ddl_tokens(["C1", "C2"], ontology)
        large = estimate_full_ddl_tokens([f"C{i}" for i in range(10)], ontology)
        assert large > small

    def test_estimate_grows_with_attributes(self):
        # Two concepts: one with rich attributes, one bare.
        rich_meta = _FakeMeta(
            name="A",
            properties={f"p{i}": object() for i in range(20)},
            relationships={f"r{i}": object() for i in range(20)},
        )
        bare_meta = _FakeMeta(name="B")

        class Ont:
            def get_concept_metadata(self, n):
                return rich_meta if n == "A" else bare_meta

        ont = Ont()
        assert (
            estimate_full_ddl_tokens(["A"], ont)
            > estimate_full_ddl_tokens(["B"], ont)
        )


# ---------------------------------------------------------------------------
# build_prefilter_prompt — adaptive description strategy
# ---------------------------------------------------------------------------


def _make_candidates(specs):
    """specs: list of (name, description)."""
    return [_Candidate(name=n, description=d) for n, d in specs]


class TestBuildPrefilterPrompt:
    def test_small_candidate_set_full_descriptions(self):
        # 50 candidates with very short descriptions; everything fits.
        candidates = _make_candidates([
            (f"C{i}", f"desc {i}") for i in range(50)
        ])
        block, with_desc = build_prefilter_prompt(candidates, max_prompt_tokens=4_000)
        assert with_desc is True
        # All names appear with their descriptions.
        for c in candidates:
            assert c.name in block
            assert "desc" in block

    def test_medium_mixed_descriptions_soft_cap(self):
        # Mix: 90 short + 10 very long. Long ones should be truncated under
        # the soft-cap-with-redistribution strategy.
        short_specs = [(f"S{i}", "short note") for i in range(90)]
        long_text = " ".join([f"word{i}" for i in range(200)])
        long_specs = [(f"L{i}", long_text) for i in range(10)]
        candidates = _make_candidates(short_specs + long_specs)
        # Tight budget to force truncation but not names-only.
        block, with_desc = build_prefilter_prompt(candidates, max_prompt_tokens=3_000)
        assert with_desc is True
        # Long descriptions must not appear in full.
        full_long_count = sum(1 for line in block.splitlines() if long_text in line)
        assert full_long_count == 0
        # Short descriptions should still be visible.
        assert "short note" in block

    def test_large_verbose_descriptions_aggressive_truncation(self):
        verbose = " ".join([f"w{i}" for i in range(500)])
        candidates = _make_candidates([
            (f"V{i}", verbose) for i in range(200)
        ])
        block, with_desc = build_prefilter_prompt(candidates, max_prompt_tokens=2_000)
        # Either with-descriptions+heavy-truncation or names-only — both acceptable.
        if with_desc:
            assert verbose not in block

    def test_very_large_names_only_mode(self):
        # 500 candidates with a tight budget such that bare names alone
        # consume the available budget — descriptions must be dropped entirely.
        candidates = _make_candidates([(f"N{i}", "verbose description here") for i in range(500)])
        block, with_desc = build_prefilter_prompt(candidates, max_prompt_tokens=1_300)
        assert with_desc is False
        # Names-only: no description text leaks through.
        assert "verbose description" not in block

    def test_pathological_thousand_plus_candidates_warns(self, caplog):
        candidates = _make_candidates([(f"P{i}", "") for i in range(1_200)])
        with caplog.at_level(logging.WARNING):
            block, with_desc = build_prefilter_prompt(
                candidates, max_prompt_tokens=2_000,
            )
        assert with_desc is False
        # Warning emitted about elevated latency.
        assert any("elevated" in r.message.lower() or "pre-filter" in r.message.lower()
                   for r in caplog.records)


# ---------------------------------------------------------------------------
# run_concept_prefilter — output validation, anchor + hallucination handling
# ---------------------------------------------------------------------------


class TestRunConceptPrefilter:
    def _ontology(self, names):
        return FakeOntology({n: "" for n in names})

    def test_anchor_missing_from_output_auto_added(self, caplog):
        ontology = self._ontology(["customer", "order", "product"])
        llm = FakeLLM({"relevant_concepts": ["order", "product"]})
        with caplog.at_level(logging.WARNING):
            result = run_concept_prefilter(
                llm=llm,
                question="What products did customers order?",
                anchor="customer",
                candidate_concepts=["customer", "order", "product"],
                ontology=ontology,
                config=MetadataContextConfig(),
            )
        assert "customer" in result.filtered_concepts
        assert any("anchor" in r.message.lower() for r in caplog.records)

    def test_unknown_concept_in_output_dropped(self, caplog):
        ontology = self._ontology(["customer", "order"])
        llm = FakeLLM({"relevant_concepts": ["customer", "phantom_concept", "order"]})
        with caplog.at_level(logging.WARNING):
            result = run_concept_prefilter(
                llm=llm,
                question="any",
                anchor="customer",
                candidate_concepts=["customer", "order"],
                ontology=ontology,
                config=MetadataContextConfig(),
            )
        assert "phantom_concept" not in result.filtered_concepts
        assert set(result.filtered_concepts) == {"customer", "order"}
        assert any("hallucinated" in r.message.lower() for r in caplog.records)

    def test_empty_output_falls_back_to_full_candidates(self):
        ontology = self._ontology(["a", "b", "c"])
        llm = FakeLLM({"relevant_concepts": []})
        result = run_concept_prefilter(
            llm=llm,
            question="any",
            anchor="a",
            candidate_concepts=["a", "b", "c"],
            ontology=ontology,
            config=MetadataContextConfig(),
        )
        assert set(result.filtered_concepts) == {"a", "b", "c"}
        assert result.fallback_used is True

    def test_llm_exception_falls_back_to_full_candidates(self):
        ontology = self._ontology(["a", "b"])

        class BoomLLM:
            def invoke(self, _):
                raise RuntimeError("boom")

        result = run_concept_prefilter(
            llm=BoomLLM(),
            question="any",
            anchor="a",
            candidate_concepts=["a", "b"],
            ontology=ontology,
            config=MetadataContextConfig(),
        )
        assert result.fallback_used is True
        assert set(result.filtered_concepts) == {"a", "b"}

    def test_result_reports_input_and_output_counts(self):
        ontology = self._ontology(["a", "b", "c", "d"])
        llm = FakeLLM({"relevant_concepts": ["a", "b"]})
        result = run_concept_prefilter(
            llm=llm,
            question="any",
            anchor="a",
            candidate_concepts=["a", "b", "c", "d"],
            ontology=ontology,
            config=MetadataContextConfig(),
        )
        assert result.input_count == 4
        assert result.output_count == 2
        assert result.latency_ms >= 0


# ---------------------------------------------------------------------------
# Demote semantics — every candidate ends up in detail OR menu
# ---------------------------------------------------------------------------


class TestDemoteSemantics:
    def _ontology(self, names):
        return FakeOntology({n: "" for n in names})

    def test_overflow_concepts_demoted_to_menu_not_dropped(self):
        """The pre-filter must split candidates into (detail, menu) such that
        together they equal the input set. No concept silently disappears."""
        ontology = self._ontology(["a", "b", "c", "d", "e"])
        llm = FakeLLM({"relevant_concepts": ["a", "b"]})
        result = run_concept_prefilter(
            llm=llm, question="x", anchor="a",
            candidate_concepts=["a", "b", "c", "d", "e"],
            ontology=ontology, config=MetadataContextConfig(),
        )
        assert set(result.detail_concepts) == {"a", "b"}
        assert set(result.menu_concepts) == {"c", "d", "e"}
        # Union equals input — no silent disappearance.
        assert set(result.detail_concepts) | set(result.menu_concepts) == {
            "a", "b", "c", "d", "e",
        }

    def test_filtered_concepts_property_alias_works(self):
        """``filtered_concepts`` is a legacy alias for ``detail_concepts`` and
        must continue to return the same list."""
        ontology = self._ontology(["a", "b", "c"])
        llm = FakeLLM({"relevant_concepts": ["a"]})
        result = run_concept_prefilter(
            llm=llm, question="x", anchor="a",
            candidate_concepts=["a", "b", "c"],
            ontology=ontology, config=MetadataContextConfig(),
        )
        assert result.filtered_concepts == result.detail_concepts

    def test_anchor_auto_promoted_into_detail_band(self):
        """Even if the LLM omits the anchor, it lands in detail (NOT menu)."""
        ontology = self._ontology(["customer", "order"])
        llm = FakeLLM({"relevant_concepts": ["order"]})
        result = run_concept_prefilter(
            llm=llm, question="x", anchor="customer",
            candidate_concepts=["customer", "order"],
            ontology=ontology, config=MetadataContextConfig(),
        )
        assert "customer" in result.detail_concepts
        assert "customer" not in result.menu_concepts


# ---------------------------------------------------------------------------
# Count trigger — fires on concept count, independent of token size
# ---------------------------------------------------------------------------


class TestCountTrigger:
    def _ontology(self, n):
        return FakeOntology({f"C{i}": "" for i in range(n)})

    def test_count_trigger_fires_at_threshold(self):
        from langchain_timbr.ontology_context.context_builder.concept_prefilter import (
            should_trigger_concept_prefilter,
        )
        cfg = MetadataContextConfig(max_detail_concepts=10)
        concepts = [f"C{i}" for i in range(15)]
        ontology = self._ontology(15)
        fire, reason = should_trigger_concept_prefilter(
            candidate_concepts=concepts, ontology=ontology, config=cfg,
        )
        assert fire is True
        assert reason == "count_overflow"

    def test_count_trigger_silent_under_threshold(self):
        from langchain_timbr.ontology_context.context_builder.concept_prefilter import (
            should_trigger_concept_prefilter,
        )
        cfg = MetadataContextConfig(
            max_detail_concepts=10,
            metadata_context_filter_max_tokens=100_000,
            metadata_context_filter_max_tokens_hard_ceiling=200_000,
        )
        concepts = [f"C{i}" for i in range(5)]
        ontology = self._ontology(5)
        fire, reason = should_trigger_concept_prefilter(
            candidate_concepts=concepts, ontology=ontology, config=cfg,
        )
        assert fire is False
        assert reason == "under_threshold"

    def test_token_trigger_still_fires_at_low_count(self):
        from langchain_timbr.ontology_context.context_builder.concept_prefilter import (
            should_trigger_concept_prefilter,
        )
        # Token budget so tight even 3 concepts overflow.
        cfg = MetadataContextConfig(max_detail_concepts=10, metadata_context_filter_max_tokens=10, metadata_context_filter_max_tokens_hard_ceiling=10)

        concepts = ["a", "b", "c"]
        ontology = FakeOntology({"a": "", "b": "", "c": ""})
        fire, reason = should_trigger_concept_prefilter(
            candidate_concepts=concepts, ontology=ontology, config=cfg,
        )
        assert fire is True
        assert reason == "token_overflow"


# ---------------------------------------------------------------------------
# render_names_only / render_with_descriptions — formatting
# ---------------------------------------------------------------------------


class TestRendering:
    def test_render_names_only_omits_descriptions(self):
        cands = _make_candidates([("a", "desc-a"), ("b", "desc-b")])
        out = render_names_only(cands)
        assert "desc-a" not in out
        assert "- a" in out and "- b" in out

    def test_render_with_descriptions_handles_blank(self):
        cands = _make_candidates([("a", ""), ("b", "hello")])
        out = render_with_descriptions(cands)
        # Blank desc renders as bare line.
        assert "- a\n" in out + "\n"
        assert "- b: hello" in out


# ---------------------------------------------------------------------------
# Fix 4 — prefilter is NEVER called outside the should_fire guard
# ---------------------------------------------------------------------------


class TestPrefilterNeverStandalone:
    """Fix 4 of the action-loop hardening pass: the orchestrator must NEVER
    call ``run_concept_prefilter`` outside the ``should_trigger_concept_prefilter``
    gate. The previous standalone safety-net call (Fix 3 deletion) was the
    last violation; this class hooks the function with a call-count spy and
    asserts AT MOST ONE call per request, AND only when the conditional
    guard returned True."""

    def _spy_prefilter(self, monkeypatch):
        """Install a spy on ``run_concept_prefilter`` exported from
        ``concept_prefilter`` AND re-exported into ``build_filtered``'s
        namespace. Returns the spy's call list."""
        import langchain_timbr.ontology_context.context_builder.concept_prefilter as cp_mod
        import langchain_timbr.ontology_context.context_builder.build_filtered as bf_mod

        calls = []
        real = cp_mod.run_concept_prefilter

        def spy(*args, **kwargs):
            calls.append({"args": args, "kwargs": dict(kwargs)})
            return real(*args, **kwargs)

        monkeypatch.setattr(cp_mod, "run_concept_prefilter", spy)
        monkeypatch.setattr(bf_mod, "run_concept_prefilter", spy)
        return calls

    def _build_ontology(self, n_concepts: int):
        """Build a flat ontology with N concepts radiating from 'anchor'."""
        from langchain_timbr.ontology_context.ontology.graph import Ontology

        class _FakeClient:
            def __init__(self, concepts):
                self._concepts = concepts

            def fetch_version_id(self):
                return "v1"

            def describe_concept(self, name):
                return list(self._concepts.get(name, []))

            def fetch_relationships_meta(self):
                return []

            def fetch_inheritance_meta(self):
                return []

        def _row(col):
            return {
                "col_name": col, "data_type": "varchar", "comment": "",
                "inheritance_marker": "", "pk_marker": "",
            }

        concepts = {
            "anchor": [_row(f"to_t{i}[t{i}].x") for i in range(n_concepts)],
        }
        for i in range(n_concepts):
            concepts[f"t{i}"] = [_row("name")]
        return Ontology(_FakeClient(concepts))

    def _scripted_build_path(self, segment):
        return json.dumps({
            "action": "build_path",
            "selected_concepts": [segment["from"], segment["to"]],
            "selected_paths": [{
                "path_id": "P1",
                "purpose": "",
                "segments": [segment],
                "is_recursive": False,
            }],
        })

    def _scripted_llm(self, responses):
        class _ScriptedLLM:
            def __init__(self, responses):
                self.responses = list(responses)
                self.calls = []

            def invoke(self, messages):
                self.calls.append(messages)
                if not self.responses:
                    raise AssertionError("ScriptedLLM exhausted")
                text = self.responses.pop(0)

                class _R:
                    def __init__(self, c):
                        self.content = c

                return _R(text)

        return _ScriptedLLM(responses)

    def test_small_subgraph_never_calls_prefilter(self, monkeypatch):
        """8 concepts, under both thresholds → prefilter must NEVER fire."""
        from langchain_timbr.ontology_context.context_builder import (
            build_filtered as _bf,
        )

        calls = self._spy_prefilter(monkeypatch)
        ontology = self._build_ontology(n_concepts=7)
        cfg = MetadataContextConfig(
            mode="dynamic",
            max_graph_depth=5,
            metadata_context_dynamic_retry=0,
            max_detail_concepts=20,
            metadata_context_filter_max_tokens=100_000,
            metadata_context_filter_max_tokens_hard_ceiling=200_000,
        )
        llm = self._scripted_llm([
            self._scripted_build_path(
                {"from": "anchor", "rel": "to_t0", "to": "t0"},
            ),
        ])
        result = _bf.build_filtered_metadata(
            question="any", anchor="anchor", ontology=ontology, llm=llm,
            config=cfg, graph_depth=2,
        )
        # Spy proves: zero calls to run_concept_prefilter under any branch.
        assert calls == []
        # Orchestrator's own telemetry agrees.
        assert result.stats["prefilter_used"] is False
        assert result.stats["prefilter_trigger"] == "under_threshold"

    def test_failed_build_path_does_not_trigger_safety_net_prefilter(
        self, monkeypatch,
    ):
        """When the planner emits a build_path that fails validation, the
        retry-fallback-redesign cascade now routes to the BFS rescue (see
        Branch 3 of build_filtered_metadata) — NOT to a standalone prefilter
        call. This test still locks: prefilter is never called as a safety
        net on planner failure for a small subgraph."""
        from langchain_timbr.ontology_context.context_builder import (
            build_filtered as _bf,
        )

        calls = self._spy_prefilter(monkeypatch)
        ontology = self._build_ontology(n_concepts=5)
        cfg = MetadataContextConfig(
            mode="dynamic",
            max_graph_depth=5,
            metadata_context_dynamic_retry=0,
            max_detail_concepts=20,
            metadata_context_filter_max_tokens=100_000,
            metadata_context_filter_max_tokens_hard_ceiling=200_000,
        )
        # Planner emits a build_path whose segment uses a nonexistent rel
        # — validation will reject it. selected_concepts=["anchor", "t0"]
        # is reachable via the legitimate to_t0 edge.
        llm = self._scripted_llm([
            self._scripted_build_path(
                {"from": "anchor", "rel": "nonexistent", "to": "t0"},
            ),
        ])
        result = _bf.build_filtered_metadata(
            question="any", anchor="anchor", ontology=ontology, llm=llm,
            config=cfg, graph_depth=2,
        )
        # Spy proves: NO standalone prefilter rescue call (still true under
        # the redesign — only the BFS rescue / depth-capped helpers fire).
        assert calls == []
        # New cascade: BFS rescue resolves the anchor->t0 path.
        assert result.stats["resolved_by"] == "bfs_selected_concepts"

    def test_large_subgraph_calls_prefilter_exactly_once(self, monkeypatch):
        """Genuine overflow → prefilter fires exactly ONCE (the conditional
        Step-0 gate). It must NOT be called a second time later for any
        rescue."""
        from langchain_timbr.ontology_context.context_builder import (
            build_filtered as _bf,
        )

        calls = self._spy_prefilter(monkeypatch)
        # 25 concepts vs max_detail_concepts=20 → count_overflow fires.
        ontology = self._build_ontology(n_concepts=24)
        cfg = MetadataContextConfig(
            mode="dynamic",
            max_graph_depth=5,
            metadata_context_dynamic_retry=0,
            max_detail_concepts=20,
            metadata_context_filter_max_tokens=100_000,
            metadata_context_filter_max_tokens_hard_ceiling=200_000,
        )
        # Prefilter LLM response, then planner build_path response.
        # (The orchestrator's prefilter call uses the same `llm` instance.)
        llm = self._scripted_llm([
            json.dumps({"relevant_concepts": ["anchor", "t0", "t1"]}),
            self._scripted_build_path(
                {"from": "anchor", "rel": "to_t0", "to": "t0"},
            ),
        ])
        result = _bf.build_filtered_metadata(
            question="any", anchor="anchor", ontology=ontology, llm=llm,
            config=cfg, graph_depth=2,
        )
        # Prefilter ran EXACTLY ONCE — the conditional Step-0 gate fired,
        # no subsequent rescue call.
        assert len(calls) == 1
        assert result.stats["prefilter_used"] is True
        assert result.stats["prefilter_trigger"] in (
            "count_overflow", "token_overflow",
        )
