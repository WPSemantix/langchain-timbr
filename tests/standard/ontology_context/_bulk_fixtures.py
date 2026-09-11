"""Derive bulk-table rows from the describe rows a fixture already declares.

Concept metadata now comes from two places, not one: ``describe concept`` still
supplies properties and measures, but primary keys come from ``sys_ontology``
and relationship metadata from ``sys_concept_relationships``. Fixtures that
declare only describe rows would otherwise build an ontology with no primary
keys and no relationship metadata.

Rather than restate every fixture's shape twice, derive the bulk rows from the
describe rows using the production ``classify`` — so a fixture keeps meaning
exactly what it meant before.

**Not for equivalence tests.** Anything asserting that the bulk path agrees with
the describe path must declare its bulk rows by hand; deriving them here would
make the assertion circular.
"""

from __future__ import annotations

from langchain_timbr.ontology_context.ontology.parser import classify

# Relationship-bearing column shapes. ``measure_rel`` is deliberately absent:
# the parser records those as measures scoped to a relationship and never
# builds a relationship from them, so deriving one would invent an edge.
_REL_KINDS = ("rel_no_suffix", "rel_target_prop", "rel_additional")


def bulk_rows_from_describe(concepts):
    """Return ``(sys_ontology_rows, sys_concept_relationships_rows)``.

    ``concepts`` maps concept name -> list of describe rows.

    Primary keys come from the ``key`` marker on direct property columns.
    Relationship rows carry the target, transitivity and inverse flag encoded
    in the column name; ``is_mtm`` and the join keys are left empty because
    describe output does not carry them — which matches the defaults the parser
    applies when the relationship lookup has no entry.
    """
    onto_rows = []
    rel_rows = []
    for name, rows in (concepts or {}).items():
        pks = []
        rels = {}
        for row in rows or []:
            col = str(row.get("col_name") or "").strip()
            if not col:
                continue
            try:
                cls = classify(col)
            except ValueError:
                # Fixtures may carry deliberately malformed columns; the parser
                # raises on those too, and this helper is not the place to care.
                continue
            kind = cls[0]
            if kind == "direct":
                if str(row.get("key") or "").strip().upper() == "PK":
                    pks.append(cls[1])
            elif kind in _REL_KINDS:
                # ("rel_*", rel_name, target, transitivity, ..., is_inverse)
                rels.setdefault(cls[1], (cls[2], cls[3], cls[-1]))
        onto_rows.append({
            "concept": name,
            "inheritance": "",
            "primary_keys": ",".join(pks),
        })
        for rel_name, (target, transitivity, is_inverse) in rels.items():
            rel_rows.append({
                "concept": name,
                "relationship_name": rel_name,
                "target_concept": target,
                "transitivity": transitivity,
                "is_inverse": 1 if is_inverse else 0,
                "is_mtm": 0,
                "source_properties": "",
                "target_properties": "",
                "description": None,
            })
    return onto_rows, rel_rows


def merge_rel_rows(explicit, derived):
    """Overlay hand-written relationship rows on the derived ones.

    A fixture that passes ``relationships=`` usually does so to set one flag —
    ``is_mtm``, a join key, a description — on a single relationship, not to
    declare the concept's entire edge set. Now that the edge set is read from
    these rows rather than from describe output, replacing them wholesale would
    silently delete every relationship the fixture did not restate.

    Keyed on ``(concept, relationship_name)``; explicit rows win.
    """
    merged = {(r.get("concept"), r.get("relationship_name")): r for r in derived}
    for row in explicit or []:
        merged[(row.get("concept"), row.get("relationship_name"))] = row
    return list(merged.values())
