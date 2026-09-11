"""Cached ontology data loaders using @cache_with_version_check.

Loads full tables from:
  - timbr.sys_ontology (concepts with inheritance and query fields)
  - timbr.sys_concept_mappings (concept-to-mapping relationships)
  - timbr.sys_views (view row counts)

Only sys_ontology is shared across callers; the mapping list, the view list and
the statistics index derived from them are permission-filtered and use the
per-user cache tier.
"""

from __future__ import annotations

import logging

import time

from ...config import props_index_probe_seconds
from ...utils.timbr_utils import cache_with_version_check, get_views_only
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


@cache_with_version_check(per_user=True)
def load_concept_mappings(conn_params: dict) -> dict[str, list[ConceptMappingRow]]:
    """Load all concept mappings from timbr.sys_concept_mappings.

    Per-user tier: the server filters this table by permission. Sharing it would
    hand one caller mapping names another cannot see, and those names are then
    used to read column statistics.

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


@cache_with_version_check(per_user=True)
def load_view_row_counts(conn_params: dict) -> dict[str, int]:
    """Load view row counts from timbr.sys_views.

    Per-user tier: ``sys_views`` is permission-filtered. Projected from the
    shared ``get_views_only`` fetch rather than querying the table again.

    Returns:
        Dict of view_name -> number_of_rows.
    """
    rows = get_views_only(conn_params=conn_params)

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


@cache_with_version_check(per_user=True, version_gated=False)
def _load_mapping_properties_index(conn_params: dict) -> dict[str, set[str]]:
    """Load the mapping→properties index from timbr.sys_properties_statistics.

    Fetches all (target_name, property_name) pairs where target_type = 'mapping'.
    Per-user tier: the rows returned are those of the mappings the caller can
    see, so this is permission-derived data.

    Not version-gated: this reads the statistics table, which is recomputed on a
    schedule of its own, so an ontology DDL change is no reason to discard it.
    The per-user TTL remains the backstop.

    Returns:
        Dict of mapping_name -> set of property_names available in the stats table.
    """
    query = (
        "SELECT target_name, property_name "
        "FROM timbr.sys_properties_statistics "
        "WHERE target_type = 'mapping'"
    )
    rows = _timbr_utils.run_query(query, conn_params)

    result: dict[str, set[str]] = {}
    for row in rows:
        target_name = row.get("target_name")
        property_name = row.get("property_name")
        if not target_name or not property_name:
            continue
        if target_name not in result:
            result[target_name] = set()
        result[target_name].add(property_name)

    return result


# (last probe time, row count, watermark) per caller. Keyed by the same cache key
# the per-user tier uses, so the probe state and the entry it guards share an
# identity by construction and cannot drift apart.
_index_probe_state: dict = {}


def _reset_index_probe_state() -> None:
    """Test seam: forget every recorded probe."""
    _index_probe_state.clear()


def _probe_properties_index(conn_params: dict) -> tuple[int, object] | None:
    """Ask how many mapping-statistics rows this caller sees, and the newest.

    Two columns, one round-trip. ``MAX`` alone would miss deletions — removing
    rows does not move the newest timestamp — and ``COUNT`` alone would miss an
    in-place recompute that kept the row count. Together they are a cheap
    fingerprint of "has this list changed for me?".

    Both aggregate over the rows *this caller can see*, which is why the probe is
    per user: a caller who cannot see the most recently updated mapping reports
    an older watermark, and one shared value would under-invalidate for everyone
    who can see more.

    Returns None if the query fails — no information, so change nothing.
    """
    query = (
        "SELECT COUNT(*) AS n, MAX(updated_at) AS watermark "
        "FROM timbr.sys_properties_statistics "
        "WHERE target_type = 'mapping'"
    )
    try:
        rows = _timbr_utils.run_query(query, conn_params)
    except Exception:
        logger.warning("Properties-index freshness probe failed; keeping the cached index")
        return None
    if not rows:
        return None
    row = rows[0]
    count = row.get("n")
    try:
        count = int(count) if count is not None else 0
    except (TypeError, ValueError):
        count = 0
    # The watermark is compared for equality, never ordered, so it is kept
    # exactly as the server returned it. NULL is a real answer here — "this
    # ontology has no mapping statistics" — and has to stay distinguishable from
    # statistics having appeared.
    return count, row.get("watermark")


def load_mapping_properties_index(conn_params: dict) -> dict[str, set[str]]:
    """The mapping→properties index, refreshed within ``props_index_probe_seconds``.

    The index is a gate rather than data: a column absent from it is never asked
    about, so a column that gained statistics contributes no value hints until
    the index catches up. Its only other invalidation is the per-user TTL, which
    leaves an hour-wide window; this narrows it to the probe interval.

    Nothing else is touched. The statistics themselves live in ``StatsCache``,
    keep their own per-mapping watermarks, and are still replaced incrementally —
    refreshing this list makes the gate accurate, it does not refetch statistics.
    """
    state_key = _load_mapping_properties_index.cache_key(conn_params=conn_params)
    now = time.monotonic()
    last = _index_probe_state.get(state_key)

    if last is None or now - last[0] > props_index_probe_seconds:
        probed = _probe_properties_index(conn_params)
        if probed is not None:
            if last is not None and (last[1], last[2]) != probed:
                # Count or watermark moved: new mappings, new statistics on an
                # existing mapping, or this caller's permissions changed. Any of
                # those makes the list wrong, and the list is invalidated whole —
                # there is no such thing as partially rebuilding it.
                _load_mapping_properties_index.invalidate(conn_params=conn_params)
            _index_probe_state[state_key] = (now, probed[0], probed[1])
        elif last is not None:
            # Probe failed: keep the recorded fingerprint, retry next interval.
            _index_probe_state[state_key] = (now, last[1], last[2])

    return _load_mapping_properties_index(conn_params=conn_params)
