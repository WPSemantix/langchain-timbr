"""Cached ontology data loaders using @cache_with_version_check.

Loads full tables from:
  - timbr.sys_ontology (concepts with inheritance and query fields)
  - timbr.sys_concept_mappings (concept-to-mapping relationships)
  - timbr.sys_views (view row counts)
"""

from __future__ import annotations

import logging

from ...utils.timbr_utils import cache_with_version_check
from ...utils import timbr_utils as _timbr_utils
from .types import OntologyConceptRow, ConceptMappingRow

logger = logging.getLogger(__name__)


@cache_with_version_check
def load_ontology_concepts(conn_params: dict) -> dict[str, OntologyConceptRow]:
    """Load all concepts from timbr.sys_ontology.

    Returns:
        Dict of concept_name -> OntologyConceptRow.
    """
    query = "SELECT concept, inheritance, `query` FROM timbr.sys_ontology"
    rows = _timbr_utils.run_query(query, conn_params)

    result: dict[str, OntologyConceptRow] = {}
    for row in rows:
        concept = row.get("concept")
        if not concept:
            logger.warning("Ontology row missing 'concept' field, skipping: %s", row)
            continue
        result[concept] = OntologyConceptRow(
            concept=concept,
            inheritance=row.get("inheritance") or "",
            query=row.get("query") or None,
        )

    return result


@cache_with_version_check
def load_concept_mappings(conn_params: dict) -> dict[str, list[ConceptMappingRow]]:
    """Load all concept mappings from timbr.sys_concept_mappings.

    Returns:
        Dict of concept_name -> list[ConceptMappingRow].
    """
    query = "SELECT concept, mapping_name, number_of_rows FROM timbr.sys_concept_mappings"
    rows = _timbr_utils.run_query(query, conn_params)

    result: dict[str, list[ConceptMappingRow]] = {}
    for row in rows:
        concept = row.get("concept")
        mapping_name = row.get("mapping_name")
        if not concept or not mapping_name:
            continue

        num_rows = row.get("number_of_rows")
        if num_rows is None:
            num_rows = -1
        else:
            try:
                num_rows = int(num_rows)
            except (ValueError, TypeError):
                num_rows = -1

        mapping_row = ConceptMappingRow(
            concept=concept,
            mapping_name=mapping_name,
            number_of_rows=num_rows,
        )

        if concept not in result:
            result[concept] = []
        result[concept].append(mapping_row)

    return result


@cache_with_version_check
def load_view_row_counts(conn_params: dict) -> dict[str, int]:
    """Load view row counts from timbr.sys_views.

    Returns:
        Dict of view_name -> number_of_rows.
    """
    query = "SELECT view_name, number_of_rows FROM timbr.sys_views"
    rows = _timbr_utils.run_query(query, conn_params)

    result: dict[str, int] = {}
    for row in rows:
        view_name = row.get("view_name")
        if not view_name:
            continue

        num_rows = row.get("number_of_rows")
        if num_rows is None:
            num_rows = -1
        else:
            try:
                num_rows = int(num_rows)
            except (ValueError, TypeError):
                num_rows = -1

        result[view_name] = num_rows

    return result
