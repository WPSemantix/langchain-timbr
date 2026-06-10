"""Parser for `describe concept dtimbr.<name> options (graph_depth='1')` output.

Two public functions:
- classify(col): pure column-name classification (no I/O, no dependencies).
- parse_describe_output(name, rows, *, relationship_meta_lookup): builds a
  ConceptMetadata from a list of describe-output rows plus a relationship lookup
  built once per ontology version by the Ontology graph.
"""

from __future__ import annotations

from .models import (
    ConceptMetadata,
    MeasureMeta,
    PropertyMeta,
    RelationshipAdditionalProperty,
    RelationshipLookupEntry,
    RelationshipMeta,
)


def classify(col: str) -> tuple:
    """Classify a `describe concept` column name into a discriminator tuple.

    Reserved characters that cannot appear in user-defined names: ``. [ ] *``.
    All `rel_*` tuple shapes carry a trailing ``is_inverse: bool``.
    """
    if not col or not col.strip():
        raise ValueError("classify: empty column name")

    # measure.* → strip prefix, classify inner, wrap as measure_*
    if col.startswith("measure."):
        inner = col[len("measure."):]
        inner_cls = classify(inner)
        kind = inner_cls[0]
        if kind == "direct":
            return ("measure_direct", inner_cls[1])
        if kind == "rel_target_prop":
            # ("rel_target_prop", rel, target, transitivity, prop, is_inverse)
            return ("measure_rel",) + inner_cls[1:]
        # Other measure encodings are unexpected; surface as an error.
        raise ValueError(f"Unexpected measure column shape: {col!r}")

    # ~rel_name[concept]... → inverse relationship column
    if col.startswith("~") and "[" in col:
        result = _split_rel(col[1:])
        # Replace trailing is_inverse=False with True
        return result[:-1] + (True,)

    if "[" in col:
        return _split_rel(col)

    if col.startswith("_type_of_"):
        return ("type_discriminator", col)

    return ("direct", col)


def _split_rel(col: str) -> tuple:
    """Split a relationship column name. Returns one of the rel_* discriminators.

    The caller is responsible for the `~` prefix (already stripped if present).
    Sets is_inverse=False; the caller overrides for inverse columns.
    """
    rel_name, sep, rest = col.partition("[")
    if not sep:
        raise ValueError(f"Unparseable column (missing '['): {col!r}")
    target_part, sep, suffix = rest.partition("]")
    if not sep:
        raise ValueError(f"Unparseable column (missing ']'): {col!r}")

    if "*" in target_part:
        target, _, level = target_part.partition("*")
        try:
            transitivity = int(level)
        except ValueError as exc:
            raise ValueError(f"Unparseable transitivity in {col!r}") from exc
    else:
        target, transitivity = target_part, 1

    if suffix.startswith("."):
        return ("rel_target_prop", rel_name, target, transitivity, suffix[1:], False)
    if suffix.startswith("_"):
        tail = suffix[1:]
        if tail == "transitivity_level":
            return ("rel_transitivity_marker", rel_name, target, transitivity, False)
        return ("rel_additional", rel_name, target, transitivity, tail, False)
    if suffix == "":
        # Bare relationship column with no suffix — unusual but treat as the
        # relationship itself with no additional/target prop. Encode as a
        # rel_target_prop with an empty prop name and let the parser drop it.
        return ("rel_no_suffix", rel_name, target, transitivity, False)
    raise ValueError(f"Unparseable suffix in {col!r}: {suffix!r}")


# ---- describe-output row helpers ------------------------------------------

# These fields are confirmed against existing code at timbr_utils.py:601.
_FIELD_COL_NAME = "col_name"
_FIELD_DATA_TYPE = "data_type"
_FIELD_COMMENT = "comment"

# Field names below are still subject to live confirmation (Build Order Step 0).
# Plan 1 spec assumes these names; the parser degrades gracefully if absent.
_FIELD_INHERITANCE_MARKER = "inheritance_marker"
# ``describe concept`` returns the PK/FK signal in a column called ``key``
# (verified against the live demo-env ontology). The legacy name
# ``pk_marker`` does NOT exist in the response — reading it produced
# ``is_pk=False`` universally, which broke every non-mtm cardinality
# derivation downstream by zeroing out the PK sets in derive_cardinality.
_FIELD_PK_MARKER = "key"
_VALUE_INHERITED = "inherited"
_VALUE_PK = "PK"
_VALUE_FK = "FK"


def _row_str(row: dict, key: str) -> str:
    value = row.get(key)
    if value is None:
        return ""
    return str(value)


def _row_description(row: dict) -> str | None:
    text = _row_str(row, _FIELD_COMMENT).strip()
    return text or None


def _row_is_inherited(row: dict) -> bool:
    return _row_str(row, _FIELD_INHERITANCE_MARKER).strip().lower() == _VALUE_INHERITED.lower()


def _row_pk_fk(row: dict) -> tuple[bool, bool]:
    marker = _row_str(row, _FIELD_PK_MARKER).strip().upper()
    return (marker == _VALUE_PK, marker == _VALUE_FK)


def parse_describe_output(
    name: str,
    rows: list[dict],
    *,
    relationship_meta_lookup: dict[tuple[str, str], RelationshipLookupEntry] | None,
    inheritance_chain: tuple[str, ...] = (),
) -> ConceptMetadata:
    """Build a ConceptMetadata from describe-concept rows.

    Args:
        name: The concept name being described.
        rows: Output of `describe concept dtimbr.<name> options (graph_depth='1')`.
        relationship_meta_lookup: Per-(concept, relationship_name) lookup populated
            once per ontology version from sys_concept_relationships. Pass None
            (defaults applied) when the lookup is unavailable.
        inheritance_chain: Ordered tuple of parent concepts (immediate parent
            first, ``thing`` last) used as a fallback when the direct
            ``(name, rel_name)`` lookup misses. Mirrors the behavior of
            inherited relationships in Timbr: a relationship declared on
            ``organization`` is also available on its child ``company``,
            and the parent's ``is_mtm`` / join-key signal must propagate
            so cardinality derivation can produce N:M for inherited m2m
            relationships. Default ``()`` preserves legacy direct-only
            behavior for callers that don't have the chain available.
    """
    if relationship_meta_lookup is None:
        relationship_meta_lookup = {}

    properties: dict[str, PropertyMeta] = {}
    measures: dict[str, MeasureMeta] = {}
    # rel_name -> builder dict {target, transitivity, target_props: list,
    #   additional_props: list[RelationshipAdditionalProperty], any_inverse_seen: bool}
    rel_builders: dict[str, dict] = {}

    for row in rows:
        col = _row_str(row, _FIELD_COL_NAME).strip()
        if not col:
            # Skip blank rows rather than raise — defensive against odd backend output.
            continue
        data_type = _row_str(row, _FIELD_DATA_TYPE).strip().lower()
        description = _row_description(row)
        is_inherited = _row_is_inherited(row)
        is_pk, is_fk = _row_pk_fk(row)

        cls = classify(col)
        kind = cls[0]

        if kind == "type_discriminator":
            continue
        if kind == "rel_transitivity_marker":
            continue
        if kind == "rel_no_suffix":
            # Bare relationship row without a property/additional suffix — record
            # the relationship so it exists in the output even if no prop attached.
            _, rel_name, target, transitivity, is_inverse = cls
            _get_or_init_builder(rel_builders, rel_name, target, transitivity, is_inverse)
            continue

        if kind == "direct":
            prop_name = cls[1]
            properties[prop_name] = PropertyMeta(
                name=prop_name,
                data_type=data_type,
                description=description,
                is_inherited=is_inherited,
                is_pk=is_pk,
                is_fk=is_fk,
            )
            continue

        if kind == "measure_direct":
            m_name = cls[1]
            measures[m_name] = MeasureMeta(
                name=m_name,
                data_type=data_type,
                description=description,
                is_inherited=is_inherited,
                scoped_to_relationship=None,
            )
            continue

        if kind == "measure_rel":
            _, rel_name, _target, _trans, m_name, _is_inv = cls
            key = f"{rel_name}.{m_name}"
            measures[key] = MeasureMeta(
                name=m_name,
                data_type=data_type,
                description=description,
                is_inherited=is_inherited,
                scoped_to_relationship=rel_name,
            )
            continue

        if kind == "rel_target_prop":
            _, rel_name, target, transitivity, prop_name, is_inverse = cls
            builder = _get_or_init_builder(rel_builders, rel_name, target, transitivity, is_inverse)
            if prop_name and prop_name not in builder["target_props"]:
                builder["target_props"].append(prop_name)
            if is_inverse:
                builder["any_inverse_seen"] = True
            continue

        if kind == "rel_additional":
            _, rel_name, target, transitivity, additional_name, is_inverse = cls
            builder = _get_or_init_builder(rel_builders, rel_name, target, transitivity, is_inverse)
            ap = RelationshipAdditionalProperty(name=additional_name, data_type=data_type)
            if ap not in builder["additional_props"]:
                builder["additional_props"].append(ap)
            if is_inverse:
                builder["any_inverse_seen"] = True
            continue

        # Defensive: unknown classification — surface so unit tests fail loudly.
        raise ValueError(f"Unhandled classification {kind!r} for column {col!r}")

    relationships: dict[str, RelationshipMeta] = {}
    for rel_name, builder in rel_builders.items():
        # Direct lookup first; fall back through the inheritance chain so a
        # relationship declared on a parent concept (e.g. has_employee on
        # organization) propagates its is_mtm / join-key signal down to the
        # child (company). Without this, inherited m2m rels silently report
        # is_mtm=False and degrade to '1:N' in derive_cardinality.
        entry = relationship_meta_lookup.get((name, rel_name))
        if entry is None:
            for parent in inheritance_chain:
                entry = relationship_meta_lookup.get((parent, rel_name))
                if entry is not None:
                    break
        if entry is not None:
            is_mtm = entry.is_mtm
            description = entry.description
            source_join_keys = entry.source_join_keys
            target_join_keys = entry.target_join_keys
            lookup_inverse = entry.is_inverse
        else:
            is_mtm = False
            description = None
            source_join_keys = ()
            target_join_keys = ()
            lookup_inverse = False
        relationships[rel_name] = RelationshipMeta(
            name=rel_name,
            target_concept=builder["target"],
            transitivity=builder["transitivity"],
            is_mtm=is_mtm,
            is_inverse=bool(builder["any_inverse_seen"] or lookup_inverse),
            description=description,
            source_join_keys=source_join_keys,
            target_join_keys=target_join_keys,
            additional_properties=tuple(builder["additional_props"]),
            target_properties=tuple(builder["target_props"]),
        )

    return ConceptMetadata(
        name=name,
        description=None,
        properties=properties,
        measures=measures,
        relationships=relationships,
    )


def _get_or_init_builder(
    rel_builders: dict[str, dict],
    rel_name: str,
    target: str,
    transitivity: int,
    is_inverse: bool,
) -> dict:
    builder = rel_builders.get(rel_name)
    if builder is None:
        builder = {
            "target": target,
            "transitivity": transitivity,
            "target_props": [],
            "additional_props": [],
            "any_inverse_seen": bool(is_inverse),
        }
        rel_builders[rel_name] = builder
        return builder
    # Consistency checks
    if builder["target"] != target:
        raise ValueError(
            f"Relationship {rel_name!r} seen with inconsistent target_concept: "
            f"{builder['target']!r} vs {target!r}"
        )
    if builder["transitivity"] != transitivity:
        raise ValueError(
            f"Relationship {rel_name!r} seen with inconsistent transitivity: "
            f"{builder['transitivity']} vs {transitivity}"
        )
    return builder
