"""Tests for validator.py — 6-rule path validator."""

from __future__ import annotations

from langchain_timbr.ontology_context.context_builder.edge_index import EdgeIndex
from langchain_timbr.ontology_context.ontology.graph import Ontology
from langchain_timbr.ontology_context.context_builder.metadata_types import (
    PathSegment,
    SelectedPath,
    TransitivityOverride,
)
from langchain_timbr.ontology_context.context_builder.validator import (
    REASON_BROKEN_CHAIN,
    REASON_HOP_BUDGET_EXCEEDED,
    REASON_INVALID_RECURSION,
    REASON_INVALID_START_CONCEPT,
    REASON_UNKNOWN_CONCEPT,
    REASON_UNKNOWN_RELATIONSHIP,
    validate_overrides,
    validate_paths,
)


def _row(col_name, **extras):
    base = {"col_name": col_name, "data_type": "varchar", "comment": "", "inheritance_marker": "", "pk_marker": ""}
    base.update(extras)
    return base


class FakeClient:
    def __init__(self, concepts, version="v1"):
        self._concepts = concepts
        self._version = version

    def fetch_version_id(self):
        return self._version

    def describe_concept(self, name):
        if name not in self._concepts:
            raise KeyError(name)
        return list(self._concepts[name])

    def fetch_relationships_meta(self):
        return []


def _build_index():
    """customer -> made_order[order]; order -> contains_product[product]; product has no outbound."""
    client = FakeClient({
        "customer": [_row("c_id", pk_marker="PK"), _row("made_order[order].o_id")],
        "order": [_row("o_id", pk_marker="PK"), _row("contains_product[product].p_id")],
        "product": [_row("p_id", pk_marker="PK")],
    })
    return EdgeIndex(Ontology(client))


def _seg(a, r, b):
    return PathSegment(**{"from": a, "rel": r, "to": b})


def _path(pid, segs, *, is_recursive=False):
    return SelectedPath(path_id=pid, purpose="", segments=segs, is_recursive=is_recursive)


class TestValidator:
    def test_valid_path_returns_empty(self):
        idx = _build_index()
        path = _path("P1", [
            _seg("customer", "made_order", "order"),
            _seg("order", "contains_product", "product"),
        ])
        errors = validate_paths([path], anchor="customer", edge_index=idx, max_hop=3)
        assert errors == []

    def test_invalid_start_concept(self):
        """A path whose start concept is neither anchor nor any other path's
        endpoint must produce INVALID_START_CONCEPT. After the Plan 2 update,
        the error message lists the reachable set so the LLM can see options."""
        idx = _build_index()
        # `product` is not anchor and no other path reaches it → unreachable start.
        path = _path("P1", [_seg("product", "junk", "x")])
        errors = validate_paths([path], anchor="customer", edge_index=idx, max_hop=3)
        invalid_start_errors = [e for e in errors if e.reason_code == REASON_INVALID_START_CONCEPT]
        assert invalid_start_errors, "Expected INVALID_START_CONCEPT for unreachable start"
        # New behavior: detail lists the reachable set.
        assert "customer" in invalid_start_errors[0].detail
        assert "product" in invalid_start_errors[0].detail  # the bad start

    def test_segmented_paths_accepted_in_order(self):
        """Plan 2 update: each path can be a single hop, chained implicitly by
        ordering. P1 ends at order; P2 starts at order; etc. No retry needed."""
        idx = _build_index()
        paths = [
            _path("P1", [_seg("customer", "made_order", "order")]),
            _path("P2", [_seg("order", "contains_product", "product")]),
        ]
        errors = validate_paths(paths, anchor="customer", edge_index=idx, max_hop=3)
        assert errors == [], f"Segmented paths should validate cleanly; got {errors}"

    def test_segmented_paths_accepted_out_of_order(self):
        """Topological sort should reorder out-of-order segmented paths before
        validation. P2 emitted first; sort moves P1 to the front."""
        idx = _build_index()
        paths = [
            _path("P2", [_seg("order", "contains_product", "product")]),
            _path("P1", [_seg("customer", "made_order", "order")]),
        ]
        errors = validate_paths(paths, anchor="customer", edge_index=idx, max_hop=3)
        assert errors == [], f"Out-of-order segmented paths should validate via topological sort; got {errors}"

    def test_segmented_path_with_unreachable_start_rejected(self):
        """A segmented path whose start is never reachable (no anchor link AND
        no earlier path reaches it) must still produce INVALID_START_CONCEPT."""
        idx = _build_index()
        paths = [
            _path("P1", [_seg("customer", "made_order", "order")]),
            # `product` is reachable only from `order`, but we have no path that
            # explicitly reaches product yet — and "junk" makes this look like a
            # misordering rather than a true unreachable. Use a fully-unrelated
            # start instead.
            _path("P2", [_seg("vendor", "junk", "thing")]),
        ]
        errors = validate_paths(paths, anchor="customer", edge_index=idx, max_hop=3)
        bad = [e for e in errors if e.reason_code == REASON_INVALID_START_CONCEPT and e.path_id == "P2"]
        assert bad, "P2 with unreachable start should produce INVALID_START_CONCEPT"

    def test_full_chain_path_still_accepted(self):
        """Backward-compat: anchored multi-hop paths (the original style)
        must still validate cleanly."""
        idx = _build_index()
        path = _path("P1", [
            _seg("customer", "made_order", "order"),
            _seg("order", "contains_product", "product"),
        ])
        errors = validate_paths([path], anchor="customer", edge_index=idx, max_hop=3)
        assert errors == []

    def test_topological_sort_unorderable_paths_appended(self):
        """Paths whose start is unreachable in any ordering get appended at the
        end of the sorted list; the validator then rejects them by Rule 1."""
        from langchain_timbr.ontology_context.context_builder.validator import _topological_sort_paths

        p_valid = _path("P1", [_seg("customer", "made_order", "order")])
        p_unreachable = _path("P2", [_seg("vendor", "junk", "thing")])
        ordered, reachable = _topological_sort_paths(
            [p_unreachable, p_valid], anchor="customer"
        )
        assert ordered[0].path_id == "P1"  # reachable path moved to front
        assert ordered[-1].path_id == "P2"  # unreachable appended
        assert "customer" in reachable and "order" in reachable

    def test_unknown_relationship(self):
        idx = _build_index()
        path = _path("P1", [_seg("customer", "phantom_rel", "order")])
        errors = validate_paths([path], anchor="customer", edge_index=idx, max_hop=3)
        assert any(e.reason_code == REASON_UNKNOWN_RELATIONSHIP for e in errors)

    def test_broken_chain(self):
        idx = _build_index()
        path = _path("P1", [
            _seg("customer", "made_order", "order"),
            _seg("product", "contains_product", "product"),  # gap: starts at product, prev ended at order
        ])
        errors = validate_paths([path], anchor="customer", edge_index=idx, max_hop=3)
        assert any(e.reason_code == REASON_BROKEN_CHAIN for e in errors)

    def test_hop_budget_exceeded(self):
        idx = _build_index()
        long_path = _path("P1", [
            _seg("customer", "made_order", "order"),
            _seg("order", "contains_product", "product"),
            _seg("product", "junk", "x"),
            _seg("x", "junk", "y"),
            _seg("y", "junk", "z"),
            _seg("z", "junk", "w"),
            _seg("w", "junk", "v"),
        ])
        errors = validate_paths([long_path], anchor="customer", edge_index=idx, max_hop=3)
        assert any(e.reason_code == REASON_HOP_BUDGET_EXCEEDED for e in errors)

    def test_is_recursive_no_longer_validated(self):
        """Rule 5 was removed in the transitivity-overrides update —
        is_recursive is informational only. A path with is_recursive=True
        and no SELF_REF segment must NOT produce INVALID_RECURSION."""
        idx = _build_index()
        path = _path("P1", [
            _seg("customer", "made_order", "order"),
        ], is_recursive=True)
        errors = validate_paths([path], anchor="customer", edge_index=idx, max_hop=3)
        assert not any(e.reason_code == REASON_INVALID_RECURSION for e in errors), (
            f"Rule 5 (INVALID_RECURSION) should be removed; got {errors}"
        )

    def test_unknown_concept(self):
        idx = _build_index()
        path = _path("P1", [_seg("customer", "made_order", "ghost_concept")])
        errors = validate_paths([path], anchor="customer", edge_index=idx, max_hop=3)
        # ghost_concept doesn't exist → UNKNOWN_CONCEPT or UNKNOWN_RELATIONSHIP
        reasons = {e.reason_code for e in errors}
        assert reasons & {REASON_UNKNOWN_CONCEPT, REASON_UNKNOWN_RELATIONSHIP}

    def test_empty_segments_path_is_invalid(self):
        idx = _build_index()
        path = _path("P1", [])
        errors = validate_paths([path], anchor="customer", edge_index=idx, max_hop=3)
        assert errors


def _build_index_with_transitive_and_self_ref():
    """Three relationships covering each acceptance branch:
    - has_acquired[company*2]  → transitive self-ref (default *2)
    - has_parent[work_item*1]  → flat self-ref (default *1) — overridable
    - made_order[order]        → flat non-self-ref — NOT overridable
    """
    client = FakeClient({
        "company": [
            _row("c_id", pk_marker="PK"),
            _row("has_acquired[company*2].name"),
        ],
        "work_item": [
            _row("w_id", pk_marker="PK"),
            _row("has_parent[work_item].title"),
        ],
        "customer": [
            _row("cust_id", pk_marker="PK"),
            _row("made_order[order].o_id"),
        ],
        "order": [_row("o_id", pk_marker="PK")],
    })
    idx = EdgeIndex(Ontology(client))
    # Pre-materialize so edge_map is populated for validate_overrides lookups.
    idx.outbound_edges("company")
    idx.outbound_edges("work_item")
    idx.outbound_edges("customer")
    return idx


class TestValidateOverrides:
    """Acceptance rules:
      - level < 2 → drop
      - is_self_ref → accept (any depth, even when default is *1)
      - transitivity > 1 → accept
      - flat non-self-ref → drop
    """

    def test_accept_override_on_transitive_self_ref(self):
        idx = _build_index_with_transitive_and_self_ref()
        ovs = [TransitivityOverride(rel="has_acquired", target="company", level=3)]
        accepted = validate_overrides(ovs, idx)
        assert len(accepted) == 1
        assert accepted[0].level == 3

    def test_accept_override_on_flat_self_ref(self):
        """Self-ref edges with default *1 CAN be overridden (timbr supports this)."""
        idx = _build_index_with_transitive_and_self_ref()
        ovs = [TransitivityOverride(rel="has_parent", target="work_item", level=3)]
        accepted = validate_overrides(ovs, idx)
        assert len(accepted) == 1
        assert accepted[0].rel == "has_parent"

    def test_drop_override_on_flat_non_self_ref(self):
        """made_order is flat (*1) and not self-ref → silently drop."""
        idx = _build_index_with_transitive_and_self_ref()
        ovs = [TransitivityOverride(rel="made_order", target="order", level=3)]
        accepted = validate_overrides(ovs, idx)
        assert accepted == []

    def test_drop_override_below_two(self):
        idx = _build_index_with_transitive_and_self_ref()
        ovs = [
            TransitivityOverride(rel="has_acquired", target="company", level=1),
            TransitivityOverride(rel="has_acquired", target="company", level=0),
        ]
        accepted = validate_overrides(ovs, idx)
        assert accepted == []

    def test_drop_override_on_unknown_relationship(self):
        idx = _build_index_with_transitive_and_self_ref()
        ovs = [TransitivityOverride(rel="phantom_rel", target="company", level=3)]
        accepted = validate_overrides(ovs, idx)
        assert accepted == []

    def test_dedup_keeps_last_entry_per_pair(self):
        idx = _build_index_with_transitive_and_self_ref()
        ovs = [
            TransitivityOverride(rel="has_acquired", target="company", level=3),
            TransitivityOverride(rel="has_acquired", target="company", level=5),
        ]
        accepted = validate_overrides(ovs, idx)
        assert len(accepted) == 1
        assert accepted[0].level == 5  # last entry wins

    def test_empty_overrides_returns_empty(self):
        idx = _build_index_with_transitive_and_self_ref()
        assert validate_overrides([], idx) == []

    def test_mixed_overrides_only_valid_ones_kept(self):
        idx = _build_index_with_transitive_and_self_ref()
        ovs = [
            TransitivityOverride(rel="has_acquired", target="company", level=3),   # ✓
            TransitivityOverride(rel="made_order", target="order", level=5),        # ✗ flat non-self-ref
            TransitivityOverride(rel="has_parent", target="work_item", level=4),    # ✓ self-ref
            TransitivityOverride(rel="phantom", target="thing", level=2),           # ✗ unknown
        ]
        accepted = validate_overrides(ovs, idx)
        rels = {o.rel for o in accepted}
        assert rels == {"has_acquired", "has_parent"}
