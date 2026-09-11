"""Timbr ontology client — thin wrapper over run_query.

Concrete implementation of the three methods consumed by the Ontology graph:
- fetch_version_id()       -> SHOW VERSION
- describe_concept(name)   -> describe concept dtimbr.`<name>` options (graph_depth='1')
- fetch_relationships_meta -> SELECT ... FROM timbr.sys_concept_relationships

All SQL goes through the existing run_query helper from utils.timbr_utils so
caching, error handling, and JWT/SSL plumbing remain centralized.
"""

from __future__ import annotations

from ...utils.timbr_utils import get_ontology_version, run_query


# Ceiling for the bulk metadata fetches below. These read whole `sys_*` tables:
# a partial answer is not a smaller answer, it is a wrong one — a missing
# relationship row silently costs an edge its is_mtm flag and its cardinality,
# and a missing sys_ontology row costs a concept its primary keys.
#
# The caller's ``results-limit`` header is ``max_limit`` — a cap on how many
# DATA rows a generated SQL answer may return (default 100). It has no business
# bounding metadata, so these queries send their own, far above any real
# ontology. An explicit value rather than deleting the header: with no header
# the server applies its own default (5000), which is also a cap we would not
# have chosen.
_METADATA_ROW_LIMIT = 100_000


class TimbrOntologyClient:
    """Stateless client that issues ontology-metadata SQL against a Timbr backend."""

    def __init__(self, conn_params: dict):
        self._conn_params = conn_params

    def _metadata_conn_params(self) -> dict:
        """``conn_params`` with the row cap raised for whole-table metadata reads."""
        params = dict(self._conn_params)
        headers = dict(params.get("additional_headers") or {})
        headers["results-limit"] = str(_METADATA_ROW_LIMIT)
        params["additional_headers"] = headers
        return params

    @property
    def conn_params(self) -> dict:
        return self._conn_params

    def fetch_version_id(self) -> str | None:
        """Return the current ontology version id, via the shared probe.

        Delegates to ``timbr_utils.get_ontology_version`` rather than issuing its
        own ``SHOW VERSION``: this consumer and the module-level query cache used
        to probe independently, so a cold run paid the round-trip twice. Only the
        *fetch* is shared — the Ontology keeps its own recorded version and its
        own decision about what to invalidate.

        ``None`` means the probe holds no value and could not obtain one right
        now (another thread is mid-fetch). Callers must read it as "no
        information", not as a version change.
        """
        return get_ontology_version(self._conn_params)

    def describe_concept(self, name: str) -> list[dict]:
        """Return rows from `describe concept dtimbr.<name>` (graph_depth=1)."""
        if not name:
            raise ValueError("describe_concept: name must be a non-empty string")
        # Match the existing pattern in get_concept_properties (timbr_utils.py:579):
        # backtick-quoted schema and concept names.
        sql = f"describe concept `dtimbr`.`{name}` options (graph_depth='1')"
        return run_query(sql, self._conn_params)

    def fetch_relationships_meta(self) -> list[dict]:
        """Return all rows from sys_concept_relationships (canonical AND inverse).

        Inverse filtering is the DDL layer's job (see inverse.should_include_in_ddl);
        the SQL fetch keeps every row.
        """
        sql = (
            "SELECT concept, relationship_name, target_concept, is_inverse, is_mtm, "
            "source_properties, target_properties, description, transitivity "
            "FROM `timbr`.`sys_concept_relationships`"
        )
        return run_query(sql, self._metadata_conn_params())

    def fetch_inheritance_meta(self) -> list[dict]:
        """Return per-concept metadata from `sys_ontology`.

        Each row has at least:
          - ``concept``: concept name
          - ``inheritance``: comma-separated parent chain
            (e.g. ``"organization,thing"`` for ``company``)
          - ``primary_keys``: comma-separated PK property names. Blank for a
            concept that inherits its PK rather than declaring one, so callers
            resolve it through ``inheritance`` (see ``Ontology.pks_of``).

        Used by the concept-centric serializer to emit the INHERITANCE section,
        and by cardinality derivation — which is why ``primary_keys`` rides
        along here instead of costing a describe per concept.
        """
        sql = (
            "SELECT concept, inheritance, primary_keys "
            "FROM `timbr`.`sys_ontology`"
        )
        return run_query(sql, self._metadata_conn_params())
