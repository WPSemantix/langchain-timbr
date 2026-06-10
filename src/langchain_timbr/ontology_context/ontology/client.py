"""Timbr ontology client — thin wrapper over run_query.

Concrete implementation of the three methods consumed by the Ontology graph:
- fetch_version_id()       -> SHOW VERSION
- describe_concept(name)   -> describe concept dtimbr.`<name>` options (graph_depth='1')
- fetch_relationships_meta -> SELECT ... FROM timbr.sys_concept_relationships

All SQL goes through the existing run_query helper from utils.timbr_utils so
caching, error handling, and JWT/SSL plumbing remain centralized.
"""

from __future__ import annotations

from ...utils.timbr_utils import run_query


class TimbrOntologyClient:
    """Stateless client that issues ontology-metadata SQL against a Timbr backend."""

    def __init__(self, conn_params: dict):
        self._conn_params = conn_params

    @property
    def conn_params(self) -> dict:
        return self._conn_params

    def fetch_version_id(self) -> str:
        """Return the current ontology version id, or 'unknown' if unavailable.

        Mirrors _get_ontology_version() at timbr_utils.py:39.
        """
        res = run_query("SHOW VERSION", self._conn_params)
        if not res:
            return "unknown"
        first = res[0]
        return first.get("id") or first.get("ID") or "unknown"

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
            "source_properties, target_properties, description "
            "FROM `timbr`.`sys_concept_relationships`"
        )
        return run_query(sql, self._conn_params)

    def fetch_inheritance_meta(self) -> list[dict]:
        """Return per-concept inheritance metadata from `sys_ontology`.

        Each row has at least:
          - ``concept``: concept name
          - ``inheritance``: comma-separated parent chain
            (e.g. ``"organization,thing"`` for ``company``)

        Used by the concept-centric serializer to emit the INHERITANCE section.
        """
        sql = "SELECT concept, inheritance FROM `timbr`.`sys_ontology`"
        return run_query(sql, self._conn_params)
