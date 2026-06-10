"""Tests for the waypoint filter — threshold-gated, precondition-protected.

Covers the 7 scenarios spec'd in the feature plan:

  1. Under threshold, flag ignored
  2. Over threshold + full visibility → intermediates stripped
  3. Over threshold + DEGRADED prompt → filter skipped (log)
  4. Over threshold + no flags → all kept (default false, log)
  5. Conflicting signals → keep wins
  6. Terminal override → kept anyway (log)
  7. Anchor override → kept anyway

The classification helpers are tested directly here. The orchestrator
wiring (degraded-prompt detection + threshold gate) is tested in
test_dynamic_wiring_safety.py / integration suites.
"""

from __future__ import annotations

from langchain_timbr.ontology_context.context_builder.metadata_types import (
    PathSegment,
    SelectedPath,
)
from langchain_timbr.ontology_context.context_builder.rebuild import (
    compute_waypoint_strip_set,
    is_path_prompt_degraded,
    strip_waypoint_columns,
)


def _seg(a, r, b, *, is_intermediate=False):
    return PathSegment(
        **{"from": a, "rel": r, "to": b, "is_intermediate": is_intermediate}
    )


# ---------------------------------------------------------------------------
# is_path_prompt_degraded
# ---------------------------------------------------------------------------


class TestPromptDegradedDetection:
    def test_full_visibility_is_not_degraded(self):
        ddl = (
            "## CONCEPTS\n\n"
            "### customer [anchor]\nprops:\n  str: name\nrels:\n  -[made_order, 1:N]-> order\n"
        )
        assert is_path_prompt_degraded(ddl) is False

    def test_cascade_marker_signals_degraded(self):
        ddl = (
            "## CONCEPTS\n\n"
            "### customer [anchor]\n"
            "props: [hidden by cascade — assume present, do not treat as absent]\n"
            "rels:\n  -[made_order, 1:N]-> order\n"
        )
        assert is_path_prompt_degraded(ddl) is True

    def test_empty_string_is_not_degraded(self):
        assert is_path_prompt_degraded("") is False
        assert is_path_prompt_degraded(None) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# compute_waypoint_strip_set — concept classification
# ---------------------------------------------------------------------------


class TestComputeWaypointStripSet:
    def test_terminal_concept_is_kept_even_when_marked_intermediate(self):
        """SCENARIO 6 — Terminal override: LLM sets is_intermediate=True on
        the path's terminal. The classifier MUST keep it; caller logs."""
        path = SelectedPath(path_id="P1", segments=[
            _seg("customer", "made_order", "order", is_intermediate=True),
            _seg("order", "contains", "product", is_intermediate=True),  # terminal
        ])
        keep, strip = compute_waypoint_strip_set([path], anchor="customer")
        assert "product" in keep  # terminal always kept
        assert "product" not in strip
        # order is a non-terminal intermediate → stripped
        assert "order" in strip
        assert "order" not in keep
        # Anchor is always kept
        assert "customer" in keep

    def test_anchor_is_kept_unconditionally(self):
        """SCENARIO 7 — Anchor override (defensive): even if for some reason
        the anchor showed up in any segment classification, the anchor MUST
        remain in keep_set. The anchor is the SQL FROM concept and stripping
        it would break the query."""
        # Construct a degenerate case where anchor appears as a 'to' too
        # (e.g. via self-ref). It must STILL be in keep_set.
        path = SelectedPath(path_id="P1", segments=[
            _seg("customer", "self_loop", "customer", is_intermediate=True),
        ])
        keep, strip = compute_waypoint_strip_set([path], anchor="customer")
        assert "customer" in keep
        assert "customer" not in strip

    def test_conflicting_signals_keep_wins(self):
        """SCENARIO 5 — Conflicting signals: concept C appears as `to` in
        two segments; one marks is_intermediate=True, the other False.
        Keep wins."""
        path1 = SelectedPath(path_id="P1", segments=[
            _seg("customer", "made_order", "order", is_intermediate=True),
        ])
        path2 = SelectedPath(path_id="P2", segments=[
            _seg("agent", "managed_order", "order", is_intermediate=False),
            _seg("order", "contains", "product", is_intermediate=False),
        ])
        keep, strip = compute_waypoint_strip_set(
            [path1, path2], anchor="customer",
        )
        assert "order" in keep, (
            "Keep wins for conflicting signals: order had one True + one False"
        )
        assert "order" not in strip

    def test_intermediate_only_concept_is_stripped(self):
        """SCENARIO 2 (classification half) — a non-terminal, non-anchor
        concept appearing ONLY in is_intermediate=True segments is stripped."""
        path = SelectedPath(path_id="P1", segments=[
            _seg("customer", "made_order", "order", is_intermediate=True),
            _seg("order", "contains", "product", is_intermediate=True),
            _seg("product", "of_material", "material", is_intermediate=False),
        ])
        keep, strip = compute_waypoint_strip_set([path], anchor="customer")
        assert "order" in strip
        assert "product" in strip
        # Material is the path terminal → kept
        assert "material" in keep
        # Anchor kept
        assert "customer" in keep

    def test_default_false_keeps_everything(self):
        """SCENARIO 4 — When is_intermediate defaults to False on every
        segment, no concept is stripped (backward-compat)."""
        path = SelectedPath(path_id="P1", segments=[
            _seg("customer", "made_order", "order"),  # default False
            _seg("order", "contains", "product"),
        ])
        keep, strip = compute_waypoint_strip_set([path], anchor="customer")
        assert strip == set()
        assert "order" in keep
        assert "product" in keep


# ---------------------------------------------------------------------------
# strip_waypoint_columns — column-level filtering
# ---------------------------------------------------------------------------


class TestStripWaypointColumns:
    def test_columns_terminating_at_stripped_concept_dropped(self):
        rels = {
            "made_order": {
                "description": "",
                "columns": [
                    # Terminates at 'order' — DROP if 'order' in strip_set.
                    {"name": "made_order[order].order_id",
                     "col_name": "order_id"},
                    # Terminates at 'product' — KEEP.
                    {"name": "made_order[order].contains[product].product_name",
                     "col_name": "product_name"},
                    # Terminates at 'material' — KEEP.
                    {"name": "made_order[order].contains[product]"
                             ".of_material[material].material_name",
                     "col_name": "material_name"},
                ],
                "measures": [
                    # Measure terminating at 'order'.
                    {"name": "measure.made_order[order].count_of_order",
                     "col_name": "count_of_order"},
                ],
            },
        }
        # Strip 'order' only (order is the waypoint).
        result = strip_waypoint_columns(rels, {"order"})
        kept_cols = [c["name"] for c in result["made_order"]["columns"]]
        kept_meas = [m["name"] for m in result["made_order"]["measures"]]
        # Order-terminating column dropped.
        assert "made_order[order].order_id" not in kept_cols
        # Order-terminating measure dropped.
        assert "measure.made_order[order].count_of_order" not in kept_meas
        # Chains passing THROUGH order to a deeper kept concept survive.
        assert "made_order[order].contains[product].product_name" in kept_cols
        assert (
            "made_order[order].contains[product]"
            ".of_material[material].material_name" in kept_cols
        )

    def test_empty_strip_set_is_noop(self):
        rels = {"made_order": {"columns": [{"name": "made_order[order].id"}]}}
        result = strip_waypoint_columns(rels, set())
        assert result["made_order"]["columns"] == rels["made_order"]["columns"]

    def test_strip_multiple_intermediates(self):
        """SCENARIO 2 (column half) — multiple intermediates get stripped."""
        rels = {
            "made_order": {
                "description": "",
                "columns": [
                    {"name": "made_order[order].order_id",       # → order
                     "col_name": "order_id"},
                    {"name": "made_order[order].contains[product].product_name",  # → product
                     "col_name": "product_name"},
                    {"name": "made_order[order].contains[product]"
                             ".of_material[material].material_name",  # → material
                     "col_name": "material_name"},
                ],
                "measures": [],
            },
        }
        result = strip_waypoint_columns(rels, {"order", "product"})
        kept = [c["name"] for c in result["made_order"]["columns"]]
        # Only the material-terminating chain survives.
        assert kept == [
            "made_order[order].contains[product]"
            ".of_material[material].material_name"
        ]

    def test_anchor_direct_columns_never_stripped(self):
        """SCENARIO 1 / general — direct attributes (no nested chain) belong
        to the anchor and are NEVER stripped. The function only looks at
        columns inside the relationships dict, but defensive: if a column
        has no chain, it survives regardless of strip_set."""
        rels = {
            "synthetic": {
                "description": "",
                "columns": [
                    # No nested chain — direct attribute. Always kept.
                    {"name": "name", "col_name": "name"},
                ],
                "measures": [],
            },
        }
        result = strip_waypoint_columns(rels, {"order", "product", "anything"})
        assert len(result["synthetic"]["columns"]) == 1
