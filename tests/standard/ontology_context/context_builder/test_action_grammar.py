"""Tests for the planner action grammar: build_path | expand_to | reanchor.

Covers:
  - Step1Output parses each action variant
  - Backward-compat: missing ``action`` field defaults to ``build_path``
  - Filter prompt narrowing: when ``allowed_actions`` is restricted, the
    rendered prompt omits the disallowed variants
  - ``_allowed_actions`` helper enforces caps correctly
  - Orchestrator (smoke-test via mocked LLM):
      * build_path terminal path resolves in one round
      * expand_to round-trip promotes + re-renders + re-calls
      * reanchor swaps anchor and restarts; counter is monotonic
      * expand cap narrows the next prompt's allowed actions
      * reanchor cap narrows the next prompt's allowed actions
      * reanchor resets expand_count
"""

from __future__ import annotations

import json
from typing import List, Optional

import pytest

from langchain_timbr.ontology_context.context_builder import build_filtered as _bf
from langchain_timbr.ontology_context.context_builder.llm_filter import _parse_step1
from langchain_timbr.ontology_context.context_builder.prompts.filter_prompt import (
    _build_system_prompt,
    build_filter_messages,
)
from langchain_timbr.ontology_context.ontology.graph import Ontology


# ---------------------------------------------------------------------------
# Step1Output parsing — each action variant
# ---------------------------------------------------------------------------


class TestActionParsing:
    def test_build_path_parses(self):
        raw = json.dumps({
            "action": "build_path",
            "selected_concepts": ["A", "B"],
            "selected_paths": [{
                "path_id": "P1",
                "purpose": "demo",
                "segments": [{"from": "A", "rel": "r", "to": "B"}],
                "is_recursive": False,
            }],
        })
        out = _parse_step1(raw)
        assert out.action == "build_path"
        assert out.selected_concepts == ["A", "B"]
        assert len(out.selected_paths) == 1
        assert out.expand_to == []
        assert out.reanchor_to is None

    def test_expand_to_parses(self):
        raw = json.dumps({
            "action": "expand_to",
            "expand_to": ["fund", "investor"],
        })
        out = _parse_step1(raw)
        assert out.action == "expand_to"
        assert out.expand_to == ["fund", "investor"]
        assert out.selected_paths == []
        assert out.reanchor_to is None

    def test_reanchor_parses(self):
        raw = json.dumps({"action": "reanchor", "reanchor_to": "company"})
        out = _parse_step1(raw)
        assert out.action == "reanchor"
        assert out.reanchor_to == "company"
        assert out.expand_to == []
        assert out.selected_paths == []

    def test_missing_action_defaults_to_build_path(self):
        """Legacy responses without an ``action`` field must default to
        ``build_path`` so the planner is backward-compatible."""
        raw = json.dumps({
            "selected_concepts": ["X"],
            "selected_paths": [],
        })
        out = _parse_step1(raw)
        assert out.action == "build_path"


# ---------------------------------------------------------------------------
# Filter prompt — grammar narrowing
# ---------------------------------------------------------------------------


class TestGrammarNarrowing:
    def test_full_union_documented_by_default(self):
        sys_prompt = _build_system_prompt(["build_path", "expand_to", "reanchor"])
        assert "build_path" in sys_prompt
        assert "expand_to" in sys_prompt
        assert "reanchor" in sys_prompt
        assert "Allowed actions this round" in sys_prompt

    def test_expand_to_dropped_when_capped(self):
        sys_prompt = _build_system_prompt(["build_path", "reanchor"])
        assert "build_path" in sys_prompt
        assert "reanchor" in sys_prompt
        # The action variant header must NOT appear when capped.
        assert "### Action: `expand_to`" not in sys_prompt
        # The allowed-actions line must list only what's allowed.
        assert "Allowed actions this round: `build_path`, `reanchor`." in sys_prompt

    def test_reanchor_dropped_when_capped(self):
        sys_prompt = _build_system_prompt(["build_path", "expand_to"])
        assert "build_path" in sys_prompt
        assert "expand_to" in sys_prompt
        assert "### Action: `reanchor`" not in sys_prompt
        assert "`reanchor`" not in sys_prompt

    def test_both_capped_leaves_only_build_path(self):
        sys_prompt = _build_system_prompt(["build_path"])
        assert "build_path" in sys_prompt
        assert "### Action: `expand_to`" not in sys_prompt
        assert "### Action: `reanchor`" not in sys_prompt

    def test_build_filter_messages_passes_allowed_actions(self):
        msgs = build_filter_messages(
            question="q", anchor="A", compact_ddl="## CONCEPTS\n",
            allowed_actions=["build_path"],
        )
        sys_msg = msgs[0]["content"]
        assert "Allowed actions this round: `build_path`." in sys_msg
        assert "### Action: `expand_to`" not in sys_msg


# ---------------------------------------------------------------------------
# _allowed_actions cap enforcement
# ---------------------------------------------------------------------------


class TestAllowedActionsCaps:
    def test_full_budget_returns_all_three(self):
        assert set(_bf._allowed_actions(0, 0)) == {
            "build_path", "expand_to", "reanchor",
        }

    def test_expand_cap_hit_drops_expand_to(self):
        actions = _bf._allowed_actions(_bf._EXPAND_CAP, 0)
        assert "expand_to" not in actions
        assert "build_path" in actions
        assert "reanchor" in actions

    def test_reanchor_cap_hit_drops_reanchor(self):
        actions = _bf._allowed_actions(0, _bf._REANCHOR_CAP)
        assert "reanchor" not in actions
        assert "build_path" in actions
        assert "expand_to" in actions

    def test_both_caps_hit_leaves_only_build_path(self):
        actions = _bf._allowed_actions(_bf._EXPAND_CAP, _bf._REANCHOR_CAP)
        assert actions == ["build_path"]


# ---------------------------------------------------------------------------
# Orchestrator smoke tests with mocked LLM
# ---------------------------------------------------------------------------


def _row(col_name, **extras):
    base = {
        "col_name": col_name, "data_type": "varchar", "comment": "",
        "inheritance_marker": "", "pk_marker": "",
    }
    base.update(extras)
    return base


class FakeClient:
    def __init__(self, concepts, version="v1"):
        self._concepts = concepts
        self._version = version

    def fetch_version_id(self):
        return self._version

    def describe_concept(self, name):
        return list(self._concepts.get(name, []))

    def fetch_relationships_meta(self):
        return []

    def fetch_inheritance_meta(self):
        return []


def _simple_ontology():
    from langchain_timbr.ontology_context.ontology.graph import Ontology
    return Ontology(FakeClient({
        "customer": [_row("name"), _row("made_order[order].o_id")],
        "order": [_row("o_id"), _row("contains_product[product].p_id")],
        "product": [_row("p_id")],
    }))


class ScriptedLLM:
    """LLM that returns a queue of pre-canned JSON responses. Records every
    call so tests can introspect the system prompts (and verify grammar
    narrowing in particular)."""

    def __init__(self, responses: List[str]):
        self.responses = list(responses)
        self.calls: List[dict] = []

    def invoke(self, messages):
        # Find system + user content for inspection (works with both langchain
        # message objects and plain dicts).
        sys_msg = getattr(messages[0], "content", None)
        if sys_msg is None and isinstance(messages[0], dict):
            sys_msg = messages[0].get("content", "")
        user_msg = ""
        if len(messages) > 1:
            user_msg = getattr(messages[1], "content", None)
            if user_msg is None and isinstance(messages[1], dict):
                user_msg = messages[1].get("content", "")
        self.calls.append({"system": sys_msg, "user": user_msg})
        if not self.responses:
            raise AssertionError(
                f"ScriptedLLM exhausted after {len(self.calls)} calls"
            )
        text = self.responses.pop(0)

        class _R:
            def __init__(self, c):
                self.content = c

        return _R(text)


def _build_path_payload(*, segments, selected_concepts=None):
    return json.dumps({
        "action": "build_path",
        "selected_concepts": selected_concepts or [],
        "selected_paths": [{
            "path_id": "P1",
            "purpose": "",
            "segments": segments,
            "is_recursive": False,
        }],
    })


def _expand_to_payload(targets):
    return json.dumps({"action": "expand_to", "expand_to": targets})


def _reanchor_payload(target):
    return json.dumps({"action": "reanchor", "reanchor_to": target})


class TestOrchestratorActionLoop:
    def _config_no_count_trigger(self):
        from langchain_timbr.ontology_context.context_builder.metadata_config import (
            MetadataContextConfig,
        )
        # Tight max_detail_concepts to keep test ontology fully in detail
        # (no menu band) for build_path / reanchor smoke tests.
        return MetadataContextConfig(
            mode="dynamic",
            max_detail_concepts=100,
            metadata_context_dynamic_retry=0,
        )

    def test_build_path_terminal_resolves_in_one_round(self):
        ontology = _simple_ontology()
        cfg = self._config_no_count_trigger()
        llm = ScriptedLLM([
            _build_path_payload(
                selected_concepts=["customer", "order"],
                segments=[{"from": "customer", "rel": "made_order", "to": "order"}],
            ),
        ])
        result = _bf.build_filtered_metadata(
            question="customers and orders",
            anchor="customer",
            ontology=ontology,
            llm=llm,
            config=cfg,
            graph_depth=2,
        )
        assert result.stats["action_history"] == ["build_path"]
        assert result.stats["resolved_by"] == "llm_paths"
        assert result.validated_paths
        # No reanchor — effective_anchor stays None.
        assert result.effective_anchor is None
        assert len(llm.calls) == 1

    def test_reanchor_swaps_anchor_and_restarts(self):
        """Reanchor: planner emits reanchor → orchestrator swaps and re-renders
        → planner emits build_path on the new anchor → terminates."""
        ontology = _simple_ontology()
        cfg = self._config_no_count_trigger()
        llm = ScriptedLLM([
            _reanchor_payload("order"),
            _build_path_payload(
                selected_concepts=["order", "product"],
                segments=[{"from": "order", "rel": "contains_product", "to": "product"}],
            ),
        ])
        result = _bf.build_filtered_metadata(
            question="orders and products",
            anchor="customer",
            ontology=ontology,
            llm=llm,
            config=cfg,
            graph_depth=2,
        )
        assert result.stats["action_history"] == ["reanchor", "build_path"]
        assert result.stats["reanchor_rounds"] == 1
        assert result.effective_anchor == "order"
        assert result.stats["reanchor_history"] == [
            {"from": "customer", "to": "order"},
        ]

    def test_reanchor_cap_narrows_allowed_actions_on_next_call(self):
        """After 1 reanchor (== cap), the next prompt MUST omit reanchor
        from the allowed-action list (grammar narrowing).

        With the menu-size narrowing (Fix 2a of action-loop hardening), the
        small ``_simple_ontology`` fits entirely in detail so the menu is
        empty too — meaning expand_to is also dropped from BOTH rounds. This
        test specifically locks the reanchor cap narrowing on round 2."""
        ontology = _simple_ontology()
        cfg = self._config_no_count_trigger()
        llm = ScriptedLLM([
            _reanchor_payload("order"),
            _build_path_payload(
                selected_concepts=["order", "product"],
                segments=[{"from": "order", "rel": "contains_product", "to": "product"}],
            ),
        ])
        result = _bf.build_filtered_metadata(
            question="any",
            anchor="customer",
            ontology=ontology,
            llm=llm,
            config=cfg,
            graph_depth=2,
        )
        # The second LLM call's system prompt MUST NOT include reanchor.
        second_sys = llm.calls[1]["system"]
        # Menu is empty AND reanchor cap is hit → only build_path remains.
        assert "Allowed actions this round: `build_path`." in second_sys
        assert "### Action: `reanchor`" not in second_sys
        assert "### Action: `expand_to`" not in second_sys
        # Per-round trace: menu is empty in both rounds (so no expand_to);
        # reanchor is in round 1 only.
        assert result.stats["allowed_actions_history"][0] == [
            "build_path", "reanchor",
        ]
        assert result.stats["allowed_actions_history"][1] == [
            "build_path",
        ]

    def test_invalid_reanchor_target_falls_back_to_build_path(self):
        """If the planner names a reanchor target NOT visible in the subgraph,
        the action loop emits a warning and terminates the action loop
        (treating the malformed action as build_path)."""
        ontology = _simple_ontology()
        cfg = self._config_no_count_trigger()
        # Provide enough responses for: 1 reanchor attempt + the deterministic
        # floor's pre-filter safety net (which may invoke LLM).
        llm = ScriptedLLM([
            _reanchor_payload("does_not_exist"),
            # Pre-filter safety-net response (may or may not be consumed).
            json.dumps({"relevant_concepts": ["customer"]}),
        ])
        result = _bf.build_filtered_metadata(
            question="any",
            anchor="customer",
            ontology=ontology,
            llm=llm,
            config=cfg,
            graph_depth=2,
        )
        # The invalid-target warning must appear (proving the action loop
        # detected and rejected the malformed reanchor).
        assert any(
            "reanchor: invalid target" in w for w in result.warnings
        )
        # The action loop produced exactly one action; it did NOT spin
        # forever consuming reanchor attempts.
        assert result.stats["action_history"] == ["reanchor"]
        # And reanchor_count stayed at 0 (the invalid target was rejected,
        # not counted).
        assert result.stats["reanchor_rounds"] == 0


# ---------------------------------------------------------------------------
# Fix 1 + Fix 2 — HARD-rule expand_to: structured validation & re-prompt
# ---------------------------------------------------------------------------


def _menu_capable_ontology():
    """Linear chain a→b→c→d→e→f. With graph_depth=2 and max_graph_depth=5, the
    menu band is non-empty (d, e, f), so expand_to is offered by grammar."""
    return Ontology(FakeClient({
        "a": [_row("to_b[b].x")],
        "b": [_row("to_c[c].x")],
        "c": [_row("to_d[d].x")],
        "d": [_row("to_e[e].x")],
        "e": [_row("to_f[f].x")],
        "f": [],
    }))


class TestExpandToValidationAndRetry:
    """Fix 2 of the action-loop hardening pass: invalid expand_to is a
    structured re-prompt within the expand budget, NEVER a fall-through to
    pathfinder / standalone prefilter."""

    def _cfg(self, **overrides):
        from langchain_timbr.ontology_context.context_builder.metadata_config import (
            MetadataContextConfig,
        )
        base = dict(
            mode="dynamic",
            max_graph_depth=5,
            metadata_context_dynamic_retry=0,
            max_detail_concepts=100,
            metadata_context_filter_max_tokens=100_000,
            metadata_context_filter_max_tokens_hard_ceiling=200_000,
        )
        base.update(overrides)
        return MetadataContextConfig(**base)

    def test_already_in_concepts_triggers_re_prompt(self):
        """Round 1 emits expand_to for a concept already in ## CONCEPTS;
        round 2 emits build_path. The invalid attempt consumes one expand
        budget unit, records the invalid history, and the retry prompt
        carries the structured "already in detail" message."""
        ontology = _menu_capable_ontology()
        cfg = self._cfg()
        # detail_depth=2 → {a, b, c} in CONCEPTS; menu = {d, e, f}.
        # The LLM (buggy) requests expand_to=["b"] — already in CONCEPTS.
        llm = ScriptedLLM([
            _expand_to_payload(["b"]),
            _build_path_payload(
                selected_concepts=["a", "b"],
                segments=[{"from": "a", "rel": "to_b", "to": "b"}],
            ),
        ])
        result = _bf.build_filtered_metadata(
            question="any",
            anchor="a",
            ontology=ontology,
            llm=llm,
            config=cfg,
            graph_depth=2,
        )
        assert result.stats["action_history"] == ["expand_to", "build_path"]
        assert result.stats["expand_rounds"] == 1   # invalid attempt counts
        history = result.stats["invalid_expand_to_history"]
        assert len(history) == 1
        assert history[0]["already_in_detail"] == ["b"]
        assert history[0]["hallucinated"] == []
        # Round 2's user prompt is the RETRY message — must include the
        # structured "already in ## CONCEPTS" error.
        round2_user = llm.calls[1]["user"]
        assert "already in" in round2_user and "CONCEPTS" in round2_user
        assert "['b']" in round2_user
        # No standalone prefilter LLM call — only the two planner calls.
        assert len(llm.calls) == 2
        assert result.stats["prefilter_used"] is False
        assert result.stats["prefilter_trigger"] == "under_threshold"
        assert result.stats["resolved_by"] == "llm_paths"

    def test_hallucinated_target_triggers_re_prompt(self):
        """Round 1 emits expand_to for a name that's in NEITHER band;
        round 2 emits build_path. Retry prompt carries the "not in
        REACHABLE" message."""
        ontology = _menu_capable_ontology()
        cfg = self._cfg()
        llm = ScriptedLLM([
            _expand_to_payload(["dragon"]),
            _build_path_payload(
                selected_concepts=["a", "b"],
                segments=[{"from": "a", "rel": "to_b", "to": "b"}],
            ),
        ])
        result = _bf.build_filtered_metadata(
            question="any",
            anchor="a",
            ontology=ontology,
            llm=llm,
            config=cfg,
            graph_depth=2,
        )
        assert result.stats["action_history"] == ["expand_to", "build_path"]
        history = result.stats["invalid_expand_to_history"]
        assert len(history) == 1
        assert history[0]["hallucinated"] == ["dragon"]
        assert history[0]["already_in_detail"] == []
        round2_user = llm.calls[1]["user"]
        assert "not in" in round2_user and "REACHABLE" in round2_user
        assert "dragon" in round2_user
        # Still no standalone prefilter LLM call.
        assert len(llm.calls) == 2
        assert result.stats["prefilter_used"] is False
        assert result.stats["resolved_by"] == "llm_paths"

    def test_cap_exhausted_after_invalid_expand_to_attempts_drops_action(self):
        """Two invalid expand_to in a row (== _EXPAND_CAP) exhausts the
        budget. The third planner call MUST have expand_to dropped from
        the allowed-actions grammar."""
        ontology = _menu_capable_ontology()
        cfg = self._cfg()
        llm = ScriptedLLM([
            _expand_to_payload(["b"]),       # invalid: already in CONCEPTS
            _expand_to_payload(["dragon"]),  # invalid: hallucinated
            _build_path_payload(
                selected_concepts=["a", "b"],
                segments=[{"from": "a", "rel": "to_b", "to": "b"}],
            ),
        ])
        result = _bf.build_filtered_metadata(
            question="any",
            anchor="a",
            ontology=ontology,
            llm=llm,
            config=cfg,
            graph_depth=2,
        )
        # Cap consumed across 2 invalid attempts.
        assert result.stats["expand_rounds"] == _bf._EXPAND_CAP
        # Round 3's grammar excludes expand_to.
        assert result.stats["allowed_actions_history"][2] == [
            "build_path", "reanchor",
        ]
        third_sys = llm.calls[2]["system"]
        assert "Allowed actions this round: `build_path`, `reanchor`." in third_sys
        assert "### Action: `expand_to`" not in third_sys
        assert result.stats["resolved_by"] == "llm_paths"

    def test_empty_menu_drops_expand_to_from_grammar(self):
        """When the menu band is empty (every relevant concept is already
        in CONCEPTS), expand_to MUST be dropped from the grammar — the LLM
        literally cannot request it. This is the structural counterpart
        of the prompt's HARD rule."""
        # _simple_ontology has 3 concepts at hops 0-2; with graph_depth=2
        # and max_graph_depth=5 everything fits in CONCEPTS → menu empty.
        ontology = _simple_ontology()
        cfg = self._cfg()
        llm = ScriptedLLM([
            _build_path_payload(
                selected_concepts=["customer", "order"],
                segments=[{"from": "customer", "rel": "made_order", "to": "order"}],
            ),
        ])
        result = _bf.build_filtered_metadata(
            question="any",
            anchor="customer",
            ontology=ontology,
            llm=llm,
            config=cfg,
            graph_depth=2,
        )
        # The first call's grammar must NOT include expand_to.
        assert "expand_to" not in result.stats["allowed_actions_history"][0]
        first_sys = llm.calls[0]["system"]
        assert "### Action: `expand_to`" not in first_sys
        # The allowed-actions line in the prompt reflects this.
        assert "`expand_to`" not in first_sys.split(
            "Allowed actions this round:", 1
        )[1].split("\n", 1)[0]
        assert result.stats["resolved_by"] == "llm_paths"


# ---------------------------------------------------------------------------
# Fix 3 — no deterministic-pathfinding fallback ANYWHERE
# ---------------------------------------------------------------------------


class TestRetryFallbackCascade:
    """The 4-way resolved_by decision tree from
    .claude/plans/retry-fallback-redesign.md:
      llm_paths              — planner emitted validated paths.
      llm_paths_anchor_only  — planner said no joins needed; lean rebuild.
      bfs_selected_concepts  — retry exhausted; DFS rescue produced paths.
      depth_capped_static    — no concepts to rescue OR rescue raised.
      empty                  — anchor has no neighbors at the cap.
    """

    def _cfg(self):
        from langchain_timbr.ontology_context.context_builder.metadata_config import (
            MetadataContextConfig,
        )
        return MetadataContextConfig(
            mode="dynamic",
            max_graph_depth=5,
            metadata_context_dynamic_retry=0,
            max_detail_concepts=100,
            metadata_context_filter_max_tokens=100_000,
            metadata_context_filter_max_tokens_hard_ceiling=200_000,
        )

    def test_failed_build_path_routes_to_bfs_rescue(self):
        """Retry exhausted with selected_concepts present and reachable from
        anchor → DFS rescue fires, produces paths, resolved_by becomes
        'bfs_selected_concepts'. (Inverts the previous no-rescue policy
        landed in Fix 3 of Action-Loop Hardening — see retry-fallback-
        redesign.md for the why.)"""
        ontology = _simple_ontology()
        cfg = self._cfg()
        llm = ScriptedLLM([
            _build_path_payload(
                selected_concepts=["customer", "order"],
                segments=[
                    {"from": "customer", "rel": "nonexistent_rel", "to": "order"},
                ],
            ),
        ])
        result = _bf.build_filtered_metadata(
            question="any",
            anchor="customer",
            ontology=ontology,
            llm=llm,
            config=cfg,
            graph_depth=2,
        )
        assert len(llm.calls) == 1
        assert result.stats["resolved_by"] == "bfs_selected_concepts"
        assert result.validated_paths, "rescue must produce at least one path"
        assert result.stats["rescue_path_count"] >= 1
        assert result.stats["prefilter_used"] is False

    def test_anchor_only_does_not_fall_back_to_static(self):
        """Planner returns selected_paths=[] (anchor-only, no joins needed).
        Validation finds no errors (empty list). Resolve as
        'llm_paths_anchor_only' with filtered_concepts={anchor}, NOT static."""
        ontology = _simple_ontology()
        cfg = self._cfg()
        llm = ScriptedLLM([
            json.dumps({
                "action": "build_path",
                "selected_concepts": ["customer"],
                "selected_paths": [],
            }),
        ])
        result = _bf.build_filtered_metadata(
            question="list all customers",
            anchor="customer",
            ontology=ontology,
            llm=llm,
            config=cfg,
            graph_depth=2,
        )
        assert len(llm.calls) == 1
        assert result.stats["resolved_by"] == "llm_paths_anchor_only"
        assert result.validated_paths == []
        assert result.filtered_concepts == {"customer"}
        assert result.stats["first_pass_valid"] is True

    def test_retry_exhausted_no_concepts_not_degraded_routes_to_anchor_only(self):
        """Retry exhausted AND selected_concepts empty, with a NON-degraded
        prompt → lean anchor-only (no relationships). The planner saw the full
        DDL and produced nothing usable, so we trust it and emit no joins."""
        ontology = _simple_ontology()
        cfg = self._cfg()
        llm = ScriptedLLM([
            _build_path_payload(
                selected_concepts=[],
                segments=[
                    {"from": "customer", "rel": "nonexistent_rel", "to": "order"},
                ],
            ),
        ])
        result = _bf.build_filtered_metadata(
            question="any",
            anchor="customer",
            ontology=ontology,
            llm=llm,
            config=cfg,
            graph_depth=2,
        )
        assert result.stats["resolved_by"] == "anchor_only"
        assert result.validated_paths == []
        assert result.filtered_concepts == {"customer"}

    def test_retry_exhausted_no_concepts_degraded_routes_to_depth_capped(self, monkeypatch):
        """Same setup but with a DEGRADED (cascade-trimmed) prompt → a 1-hop
        safety net (depth-capped at 1, regardless of graph_depth)."""
        monkeypatch.setattr(_bf, "is_path_prompt_degraded", lambda _ddl: True)
        ontology = _simple_ontology()
        cfg = self._cfg()
        llm = ScriptedLLM([
            _build_path_payload(
                selected_concepts=[],
                segments=[
                    {"from": "customer", "rel": "nonexistent_rel", "to": "order"},
                ],
            ),
        ])
        result = _bf.build_filtered_metadata(
            question="any",
            anchor="customer",
            ontology=ontology,
            llm=llm,
            config=cfg,
            graph_depth=2,
        )
        assert result.stats["resolved_by"] == "depth_capped_static"
        # Depth is uniformly capped at 1 now ("1 graph depth from root").
        assert result.stats["depth_capped_at"] == 1
        assert "customer" in result.filtered_concepts
        assert "order" in result.filtered_concepts
        assert result.stats["depth_capped_edge_count"] >= 1

    def test_bfs_rescue_exception_routes_to_no_paths_fallback(self, monkeypatch):
        """If the rescue helper raises, flow through to the no-paths fallback
        (anchor-only on a non-degraded prompt) with bfs_rescue_failed flagged."""
        ontology = _simple_ontology()
        cfg = self._cfg()
        llm = ScriptedLLM([
            _build_path_payload(
                selected_concepts=["customer", "order"],
                segments=[
                    {"from": "customer", "rel": "nonexistent_rel", "to": "order"},
                ],
            ),
        ])

        def _boom(**_kw):
            raise RuntimeError("rescue exploded")

        monkeypatch.setattr(_bf, "_bfs_paths_for_concepts", _boom)
        result = _bf.build_filtered_metadata(
            question="any",
            anchor="customer",
            ontology=ontology,
            llm=llm,
            config=cfg,
            graph_depth=2,
        )
        assert result.stats["resolved_by"] == "anchor_only"
        assert result.stats["bfs_rescue_failed"] is True
        assert any("bfs_rescue_error" in w for w in result.warnings)

    def test_degraded_depth_cap_is_1_excludes_two_hop_concepts(self, monkeypatch):
        """The degraded safety net is capped at 1 hop, so a 2-hop concept
        (product) is excluded while the 1-hop neighbor (order) is included."""
        monkeypatch.setattr(_bf, "is_path_prompt_degraded", lambda _ddl: True)
        ontology = _simple_ontology()
        cfg = self._cfg()
        llm = ScriptedLLM([
            _build_path_payload(
                selected_concepts=[],
                segments=[
                    {"from": "customer", "rel": "nonexistent_rel", "to": "order"},
                ],
            ),
        ])
        result = _bf.build_filtered_metadata(
            question="any",
            anchor="customer",
            ontology=ontology,
            llm=llm,
            config=cfg,
            graph_depth=2,
        )
        assert result.stats["resolved_by"] == "depth_capped_static"
        assert result.stats["depth_capped_at"] == 1
        # customer + order reachable in 1 hop; product is 2 hops away → excluded.
        assert "customer" in result.filtered_concepts
        assert "order" in result.filtered_concepts
        assert "product" not in result.filtered_concepts

    def test_bfs_rescue_zero_paths_routes_to_anchor_only(self):
        """Rescue ran but found zero paths within graph_depth → flow through
        to anchor-only (lean rebuild) with bfs_rescue_empty stat flagged.
        Triggered by naming an unreachable concept while keeping graph_depth
        too tight to reach it through the available edges."""
        ontology = _simple_ontology()
        cfg = self._cfg()
        llm = ScriptedLLM([
            _build_path_payload(
                # `product` is 2 hops from `customer`. With graph_depth=1
                # the rescue can't reach it → returns [].
                selected_concepts=["product"],
                segments=[
                    {"from": "customer", "rel": "nonexistent_rel", "to": "product"},
                ],
            ),
        ])
        result = _bf.build_filtered_metadata(
            question="any",
            anchor="customer",
            ontology=ontology,
            llm=llm,
            config=cfg,
            graph_depth=1,
        )
        assert result.stats["resolved_by"] == "llm_paths_anchor_only"
        assert result.stats["bfs_rescue_empty"] is True
        assert result.validated_paths == []
        assert result.filtered_concepts == {"customer"}
