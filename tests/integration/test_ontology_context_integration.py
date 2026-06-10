"""Integration tests for ontology_context against live Timbr ontologies.

Two ontologies are exercised:

  - ``timbr_crunchbase_llm_tests`` (hardcoded as ``CRUNCHBASE_ONTOLOGY``) — the
    ``company`` concept has many-to-many relationships, a transitive
    ``has_acquired[company*N]`` self-ref, and relationship additional
    properties (e.g. ``has_employee[person]_title``). This is the rich fixture.

  - ``config["timbr_ontology"]`` (default ``supply_metrics_llm_tests``) — the
    ``order`` concept has one-to-many relationships. Used as a smaller, simpler
    sanity check.

These tests require a live Timbr connection (TIMBR_URL, TIMBR_TOKEN env vars).
They are skipped if credentials are not provided.
"""

from __future__ import annotations

import pytest

from langchain_timbr.ontology_context import (
    Ontology,
    TimbrOntologyClient,
)


CRUNCHBASE_ONTOLOGY = "timbr_crunchbase_llm_tests"


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


# ---------------------------------------------------------------------------
# Crunchbase — company: mtm + relationship additional property + transitive
# ---------------------------------------------------------------------------


class TestCrunchbaseCompany:
    """Exercises the rich-shape concept (company) on timbr_crunchbase_llm_tests."""

    def test_describe_returns_metadata(self, config):
        ontology = _ontology(config, CRUNCHBASE_ONTOLOGY)
        meta = ontology.get_concept_metadata("company")
        assert meta.name == "company"
        assert meta.properties, "company must have direct properties"
        assert meta.relationships, "company must expose at least one relationship"

    def test_company_has_at_least_one_mtm_relationship(self, config):
        ontology = _ontology(config, CRUNCHBASE_ONTOLOGY)
        meta = ontology.get_concept_metadata("company")
        mtm_rels = [r for r in meta.relationships.values() if r.is_mtm]
        assert mtm_rels, (
            f"Expected at least one many-to-many relationship on company, "
            f"got relationships: {sorted(meta.relationships)}"
        )

    def test_company_has_transitive_relationship(self, config):
        """company.has_acquired traverses other companies — transitive self-ref."""
        ontology = _ontology(config, CRUNCHBASE_ONTOLOGY)
        meta = ontology.get_concept_metadata("company")
        transitive = [r for r in meta.relationships.values() if r.transitivity > 1]
        assert transitive, (
            f"Expected at least one transitive relationship on company "
            f"(e.g. has_acquired[company*N]); got: "
            f"{[(r.name, r.transitivity) for r in meta.relationships.values()]}"
        )

    def test_company_has_relationship_additional_properties(self, config):
        """has_employee[person]_title etc. should surface as additional_properties."""
        ontology = _ontology(config, CRUNCHBASE_ONTOLOGY)
        meta = ontology.get_concept_metadata("company")
        rels_with_additional = [
            r for r in meta.relationships.values() if r.additional_properties
        ]
        assert rels_with_additional, (
            "Expected at least one relationship with additional_properties on company"
        )

    def test_cardinality_lookup_for_each_relationship(self, config):
        """Cardinality must resolve for every relationship company has."""
        ontology = _ontology(config, CRUNCHBASE_ONTOLOGY)
        meta = ontology.get_concept_metadata("company")
        for rel_name in meta.relationships:
            card = ontology.cardinality_of("company", rel_name)
            assert card in {"N:M", "N:1", "1:N", "1:1"}, (
                f"Unexpected cardinality {card!r} for company.{rel_name}"
            )

    def test_company_cardinality_values_are_correct(self, config):
        """Concrete cardinality assertions on three company relationships
        that exercise distinct derivation paths:

        - ``has_employee`` → ``person`` should be ``N:M``. The relationship
          is inherited from ``organization`` (parent of company) which is
          many-to-many: a company has many employees AND a person can work
          at many organizations.
        - ``has_acquired`` → ``company`` should be ``N:M``. A direct (non-
          inherited) self-ref many-to-many: a company can have acquired many
          companies AND a company can be acquired by many (acquisition
          history).
        - ``created_fund`` → ``fund`` should be ``N:1``. The join key on the
          company side is a foreign key (``fund_id``) pointing at fund's
          primary key, so MANY companies can reference ONE fund (an
          inherited property — organization carries ``fund_id`` too).

        This test catches both (a) cardinality regressions from the
        derivation logic and (b) inheritance-lookup regressions where
        an inherited-from-parent relationship loses its source-side PK
        signal."""
        ontology = _ontology(config, CRUNCHBASE_ONTOLOGY)
        meta = ontology.get_concept_metadata("company")

        # Sanity — all three relationships exist on company.
        for rel_name in ("has_employee", "has_acquired", "created_fund"):
            assert rel_name in meta.relationships, (
                f"Expected company.{rel_name!r} relationship to exist; "
                f"got: {sorted(meta.relationships)}"
            )

        expected = {
            "has_employee": "N:M",   # inherited m2m from organization
            "has_acquired": "N:M",   # self-ref m2m, direct on company
            "created_fund": "N:1",   # FK on company side → unique fund
        }
        actual = {
            rel: ontology.cardinality_of("company", rel)
            for rel in expected
        }
        # Single assertion with the full mapping so failures show every
        # mismatch in one go (instead of stopping at the first).
        assert actual == expected, (
            f"company cardinality mismatch.\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}\n"
            f"Note: has_employee is inherited from organization (m2m); "
            f"has_acquired is a direct self-ref m2m; created_fund's "
            f"join key fund_id on the company side is a FK (inherited "
            f"from organization) referencing fund's PK → N:1."
        )


# ---------------------------------------------------------------------------
# SQL-gen prompt surfaces cardinality on each top-level rel
# ---------------------------------------------------------------------------


class TestCrunchbaseSqlGenCardinality:
    """Both static and dynamic metadata-context modes must inject the
    cardinality marker into each relationship's description so the SQL-gen
    LLM sees join multiplicity inline.

    Locks the behavior on the crunchbase ``company`` concept, where:
      - ``has_employee`` is inherited from ``organization`` (m2m) → N:M
      - ``has_product`` is a direct 1:N relationship to product
    """

    CONCEPT = "company"
    QUESTION = "count employees and products of Microsoft"
    EXPECTED_CARDS = {
        "has_employee": "N:M",
        "has_product": "1:N",
    }

    def _gen_context(self, config, *, mode: str, llm=None) -> dict:
        """Call ``_build_sql_generation_context`` directly. No SQL-gen LLM
        invocation needed — the returned dict already contains the
        rendered relationship strings the test inspects."""
        from langchain_timbr.utils.timbr_llm_utils import _build_sql_generation_context
        from langchain_timbr.ontology_context.ontology.shared import (
            reset_shared_ontologies,
        )

        reset_shared_ontologies()
        conn = _conn_params(config, ontology=CRUNCHBASE_ONTOLOGY)
        return _build_sql_generation_context(
            question=self.QUESTION,
            conn_params=conn,
            schema="dtimbr",
            concept=self.CONCEPT,
            concept_metadata={},
            graph_depth=2,
            include_tags=None,
            exclude_properties=None,
            db_is_case_sensitive=False,
            max_limit=100,
            llm=llm,
            enable_technical_context=False,   # smaller prompt, faster test
            metadata_context_mode=mode,
        )

    def _assert_rel_card_in_context(self, ctx: dict, rel_name: str, card: str):
        """The rel-description prompt strings are embedded in
        ``measures_context`` (per timbr_llm_utils.py:1338 —
        ``measures_str += f"\\n{rel_prop_str}"``). Look for the rel name and
        confirm the matching cardinality token appears in its block."""
        text = ctx.get("measures_context") or ""
        # Locate the rel's first occurrence; the description follows
        # `which described as "..."` within a few hundred chars.
        idx = text.find(rel_name)
        assert idx >= 0, (
            f"relationship {rel_name!r} missing from SQL-gen "
            f"measures_context. Excerpt: {text[:400]!r}"
        )
        # Search a generous window after the rel-name match for the
        # cardinality marker. 600 chars is enough to span the rel
        # description line emitted by ``_build_rel_columns_str``.
        window = text[idx: idx + 600]
        assert f"cardinality: {card}" in window, (
            f"expected 'cardinality: {card}' near rel {rel_name!r}. "
            f"Window: {window!r}"
        )

    def test_static_mode_injects_cardinality_into_rel_description(self, config):
        ctx = self._gen_context(config, mode="static")
        for rel, card in self.EXPECTED_CARDS.items():
            self._assert_rel_card_in_context(ctx, rel, card)

    def test_dynamic_mode_injects_cardinality_into_rel_description(
        self, config, llm,
    ):
        ctx = self._gen_context(config, mode="dynamic", llm=llm)
        for rel, card in self.EXPECTED_CARDS.items():
            self._assert_rel_card_in_context(ctx, rel, card)

    def test_dynamic_two_hop_chain_renders_one_block_per_hop(self, config):
        """A 2-hop chain (company → has_employee → person → has_degree
        → degree) must produce TWO independent blocks in the rendered
        relationships section, each carrying its OWN description +
        cardinality + only that hop's direct columns. Hand-built path
        sidesteps LLM nondeterminism — this is a structural test of the
        rebuild + render pipeline."""
        from langchain_timbr.ontology_context.context_builder.rebuild import (
            build_relationships_from_paths,
        )
        from langchain_timbr.ontology_context.context_builder.metadata_types import (
            PathSegment, SelectedPath,
        )
        from langchain_timbr.utils.timbr_llm_utils import _build_rel_columns_str
        from langchain_timbr.ontology_context.ontology.shared import (
            reset_shared_ontologies,
        )

        reset_shared_ontologies()
        ontology = _ontology(config, CRUNCHBASE_ONTOLOGY)

        def _seg(a, r, b):
            return PathSegment(**{"from": a, "rel": r, "to": b})

        path = SelectedPath(path_id="P1", segments=[
            _seg("company", "has_employee", "person"),
            _seg("person", "has_degree", "degree"),
        ])
        result = build_relationships_from_paths(
            [path], ontology, anchor="company",
        )

        # Two per-hop buckets, keyed by accumulated prefix.
        assert set(result.keys()) == {
            "has_employee[person]",
            "has_employee[person].has_degree[degree]",
        }

        # Each bucket carries its OWN cardinality:
        #   has_employee (inherited m2m from organization) → N:M
        #   has_degree (direct rel on person) → 1:N
        assert "cardinality: N:M" in result["has_employee[person]"]["description"]
        assert (
            "cardinality: 1:N"
            in result["has_employee[person].has_degree[degree]"]["description"]
        )

        # Render via the production helper and confirm TWO separate
        # "- The following columns are part of" blocks.
        rendered = _build_rel_columns_str(result)
        col_blocks = [
            line for line in rendered.split("\n")
            if "The following columns are part of" in line
        ]
        assert len(col_blocks) == 2
        # First block headers cite the hop-1 prefix; second cites the
        # 2-hop prefix.
        assert "has_employee[person] relationship" in col_blocks[0]
        assert (
            "has_employee[person].has_degree[degree] relationship"
            in col_blocks[1]
        )
        # Anti-cross-bleed: hop-1 block contains columns of person (not
        # degree); hop-2 block contains columns of degree (not person).
        # Use the column NAME prefix to verify cleanly.
        hop1_cols = [
            c["name"] for c in result["has_employee[person]"]["columns"]
        ]
        hop2_cols = [
            c["name"]
            for c in result["has_employee[person].has_degree[degree]"]["columns"]
        ]
        assert hop1_cols, "hop-1 must have person's direct cols"
        assert hop2_cols, "hop-2 must have degree's direct cols"
        for n in hop1_cols:
            assert n.startswith("has_employee[person]."), n
            assert "has_degree[degree]" not in n, n
        for n in hop2_cols:
            assert n.startswith(
                "has_employee[person].has_degree[degree]."
            ), n


# ---------------------------------------------------------------------------
# config['timbr_ontology'] — order: one-to-many relationships
# ---------------------------------------------------------------------------


class TestDefaultOntologyOrder:
    """Exercises the simpler shape (order has otm relationships)."""

    def test_describe_returns_metadata(self, config):
        ontology = _ontology(config, config["timbr_ontology"])
        meta = ontology.get_concept_metadata("order")
        assert meta.name == "order"
        assert meta.properties

    def test_order_has_otm_relationship(self, config):
        ontology = _ontology(config, config["timbr_ontology"])
        meta = ontology.get_concept_metadata("order")
        assert meta.relationships, "order should expose at least one relationship"
        cardinalities = {
            rel_name: ontology.cardinality_of("order", rel_name)
            for rel_name in meta.relationships
        }
        # At least one OTM (i.e. 1:N or N:M) relationship is expected on order
        otm_kinds = {"1:N", "N:M"}
        assert any(c in otm_kinds for c in cardinalities.values()), (
            f"Expected at least one OTM relationship on order, got: {cardinalities}"
        )


# ---------------------------------------------------------------------------
# Cache behavior against the live backend
# ---------------------------------------------------------------------------


class TestLiveCacheBehavior:
    """Sanity-checks that the lazy cache and TTL throttling work end-to-end."""

    def test_repeat_calls_share_one_describe_call(self, config):
        # Wrap a real client in a counting proxy so we can observe call counts.
        real_client = TimbrOntologyClient(
            _conn_params(config, ontology=CRUNCHBASE_ONTOLOGY)
        )

        class CountingClient:
            def __init__(self, inner):
                self.inner = inner
                self.describe_calls = 0
                self.rels_calls = 0
                self.version_calls = 0

            def fetch_version_id(self):
                self.version_calls += 1
                return self.inner.fetch_version_id()

            def describe_concept(self, name):
                self.describe_calls += 1
                return self.inner.describe_concept(name)

            def fetch_relationships_meta(self):
                self.rels_calls += 1
                return self.inner.fetch_relationships_meta()

        counting = CountingClient(real_client)
        ontology = Ontology(counting, version_ttl_seconds=3600)
        ontology.get_concept_metadata("company")
        ontology.get_concept_metadata("company")
        ontology.get_concept_metadata("company")
        assert counting.describe_calls == 1, (
            "Cache hit on the same concept must not re-call describe_concept"
        )
        assert counting.rels_calls == 1, (
            "Relationship lookup must be built once per ontology version"
        )
        assert counting.version_calls == 1, (
            "Version check must be throttled within the TTL window"
        )

    def test_invalidate_forces_refetch(self, config):
        real_client = TimbrOntologyClient(
            _conn_params(config, ontology=CRUNCHBASE_ONTOLOGY)
        )

        class CountingClient:
            def __init__(self, inner):
                self.inner = inner
                self.describe_calls = 0

            def fetch_version_id(self):
                return self.inner.fetch_version_id()

            def describe_concept(self, name):
                self.describe_calls += 1
                return self.inner.describe_concept(name)

            def fetch_relationships_meta(self):
                return self.inner.fetch_relationships_meta()

        counting = CountingClient(real_client)
        ontology = Ontology(counting)
        ontology.get_concept_metadata("company")
        ontology.invalidate()
        ontology.get_concept_metadata("company")
        assert counting.describe_calls == 2

