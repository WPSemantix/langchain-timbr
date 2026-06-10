"""Tests for split_branching_paths — forks mis-packed into one path_id are
split into separate linear paths before validation/rebuild."""

from __future__ import annotations

from langchain_timbr.ontology_context.context_builder.edge_index import EdgeIndex
from langchain_timbr.ontology_context.ontology.graph import Ontology
from langchain_timbr.ontology_context.context_builder.metadata_types import (
    PathSegment,
    SelectedPath,
)
from langchain_timbr.ontology_context.context_builder.validator import (
    REASON_BROKEN_CHAIN,
    split_branching_paths,
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
    """anchor `order` forks: order -> customer (of_customer) AND
    order -> product (includes_product); product -> material (contains) and
    product -> bom (has_bom)."""
    client = FakeClient({
        "order": [
            _row("o_id", pk_marker="PK"),
            _row("of_customer[customer].cust_id"),
            _row("includes_product[product].p_id"),
        ],
        "customer": [_row("cust_id", pk_marker="PK")],
        "product": [
            _row("p_id", pk_marker="PK"),
            _row("contains[material].m_id"),
            _row("has_bom[bom].b_id"),
        ],
        "material": [_row("m_id", pk_marker="PK")],
        "bom": [_row("b_id", pk_marker="PK")],
    })
    return EdgeIndex(Ontology(client))


def _seg(a, r, b, *, is_intermediate=False):
    return PathSegment(**{"from": a, "rel": r, "to": b, "is_intermediate": is_intermediate})


def _path(pid, segs, *, purpose="", is_recursive=False):
    return SelectedPath(path_id=pid, purpose=purpose, segments=segs, is_recursive=is_recursive)


class TestSplitBranchingPaths:
    def test_fork_at_anchor_splits_into_linear_paths(self):
        """The user's bug: order->customer AND order->product->material packed
        into one path_id. Split into P1.1 and P1.2; both validate cleanly."""
        path = _path("P1", [
            _seg("order", "of_customer", "customer"),
            _seg("order", "includes_product", "product"),
            _seg("product", "contains", "material"),
        ])
        out = split_branching_paths([path], anchor="order")
        assert [p.path_id for p in out] == ["P1.1", "P1.2"]
        assert [(s.from_concept, s.to_concept) for s in out[0].segments] == [("order", "customer")]
        assert [(s.from_concept, s.to_concept) for s in out[1].segments] == [
            ("order", "product"), ("product", "material"),
        ]
        # The split output validates with zero errors.
        errors = validate_paths(out, anchor="order", edge_index=_build_index(), max_hop=3)
        assert errors == [], f"split fork should validate cleanly; got {errors}"

    def test_waypoint_full_chain_not_split(self):
        """A legitimate full chain (no continuity break) is returned unchanged,
        preserving the path_id and is_intermediate flags."""
        path = _path("P1", [
            _seg("order", "includes_product", "product", is_intermediate=True),
            _seg("product", "contains", "material"),
        ])
        out = split_branching_paths([path], anchor="order")
        assert len(out) == 1
        assert out[0].path_id == "P1"
        assert out[0].segments[0].is_intermediate is True
        assert out[0].segments[1].is_intermediate is False

    def test_single_segment_unchanged(self):
        path = _path("P1", [_seg("order", "of_customer", "customer")])
        out = split_branching_paths([path], anchor="order")
        assert len(out) == 1
        assert out[0].path_id == "P1"

    def test_empty_path_unchanged(self):
        path = _path("P1", [])
        out = split_branching_paths([path], anchor="order")
        assert len(out) == 1
        assert out[0].path_id == "P1"
        assert out[0].segments == []

    def test_branch_from_intermediate_left_intact(self):
        """A branch off a sibling run's INTERMEDIATE (product, not a terminal)
        is not a reachable start, so the path is left intact — and validation
        still flags BROKEN_CHAIN for the caller's retry path."""
        path = _path("P1", [
            _seg("order", "includes_product", "product"),
            _seg("product", "contains", "material"),
            _seg("product", "has_bom", "bom"),  # branches off product (an intermediate)
        ])
        out = split_branching_paths([path], anchor="order")
        assert len(out) == 1
        assert out[0].path_id == "P1"  # not split
        errors = validate_paths(out, anchor="order", edge_index=_build_index(), max_hop=3)
        assert any(e.reason_code == REASON_BROKEN_CHAIN for e in errors)

    def test_split_preserves_purpose_and_recursive(self):
        path = _path("P1", [
            _seg("order", "of_customer", "customer"),
            _seg("order", "includes_product", "product"),
        ], purpose="fork purpose", is_recursive=True)
        out = split_branching_paths([path], anchor="order")
        assert len(out) == 2
        assert all(p.purpose == "fork purpose" for p in out)
        assert all(p.is_recursive is True for p in out)

    def test_second_path_branch_uses_prior_terminal(self):
        """A later path that forks but whose run starts at an EARLIER path's
        terminal is splittable (terminals are reachable)."""
        paths = [
            _path("P1", [_seg("order", "includes_product", "product")]),
            # P2 packs two branches from `product` (P1's terminal): one to
            # material, one back via has_bom. Both start at `product`.
            _path("P2", [
                _seg("product", "contains", "material"),
                _seg("product", "has_bom", "bom"),
            ]),
        ]
        out = split_branching_paths(paths, anchor="order")
        assert [p.path_id for p in out] == ["P1", "P2.1", "P2.2"]
        errors = validate_paths(out, anchor="order", edge_index=_build_index(), max_hop=3)
        assert errors == [], f"expected clean split; got {errors}"
