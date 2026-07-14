"""
ontology_metadata.py

Single support module for answering *semantic + source-table* questions about the
ontology itself (concepts / properties / measures + their logic, relationships,
views/cubes, and the physical tables that back them). Served from a cached,
deterministic set of sys_* queries.

Integration is intentionally minimal — existing nodes each add ONE guard line and
import from here. No other logic lives outside this module.

    identify_concept:  prompt += ontology_metadata.prompt_lines(agent_options)
    generate_sql:      if ontology_metadata.is_metadata_concept(target):
                           return ontology_metadata.build_metadata_sql(target, q, ont)
    execute:           if ontology_metadata.is_metadata_concept(target):
                           return ontology_metadata.fetch_metadata_rows(...)
    answer:            NO CHANGE (generic; consumes rows[])

Design notes
------------
* generate_sql emits a REAL, runnable query: ``SELECT * FROM timbr.sys_ontology``.
  - execute recognizes the metadata target concept and expands it into a
    concept-centric rows[]: one pivoted object per concept, plus view/cube objects.
  - any other API path that just runs the returned .sql still gets a valid,
    best-effort ontology snapshot (concept-level) instead of an error.
* execute returns a rows[] of pivoted concept objects (each key present only when
  it has a value), so the generic answer chain consumes it with no changes.
* cache key is the EXISTING ontology-version helper — no extra version query.
"""

import re

META = "ontology_metadata"

# Real, runnable best-effort query emitted by generate_sql. Non-execute callers get
# a valid concept-level ontology snapshot; execute expands to the full set below.
BASE_SQL = "SELECT * FROM timbr.sys_ontology"

# --------------------------------------------------------------------------- #
# concept helpers
# --------------------------------------------------------------------------- #
def is_metadata_concept(concept: str) -> bool:
    c = concept or ""
    return c == META or c.endswith(f".{META}")

def parse_ontology(concept: str, default: str) -> str:
    """`.ontology_metadata` -> ``; bare `ontology_metadata` -> default."""
    c = concept or ""
    return c[: -len(META) - 1] if c.endswith(f".{META}") else default

# --------------------------------------------------------------------------- #
# identify_concept extension (config-gated, default OFF for backward compat)
# --------------------------------------------------------------------------- #
_PROMPT_LINES = (
    " (selection_rule: when user asks for metadata model-introspection questions on the ontology definitions: lineage, concepts, views, cubes, properties, relationships, and measures. not for data values or analysis results)\n"
)

def is_enabled(agent_options: dict | None) -> bool:
    """Read `enable_ontology_questions` from Data Agent options. Default False."""
    if not agent_options:
        return False
    return bool(agent_options.get("enable_ontology_questions", True))

def prompt_lines(agent_options: dict | None = None, ontology: str | None = None) -> str:
    """Inject the metadata concept lines only when enabled; else empty string."""
    
    if ontology and ontology != "":
        return "- `" + ontology + "`.`ontology_metadata`" + _PROMPT_LINES.replace("[ontology]", ontology) if is_enabled(agent_options) else ""
    
    return "- `ontology_metadata`" + _PROMPT_LINES if is_enabled(agent_options) else ""

# --------------------------------------------------------------------------- #
# scope + parsing
# --------------------------------------------------------------------------- #
_SOURCES_RE = re.compile(
    r"\b(source table|which table|underlying table|physical table|comes? from|"
    r"feeds?|backed by|dataset|origin|lineage)\b",
    re.I,
)

def scope_of(question: str) -> dict:
    """Semantic is always on; source-tables sub-scope is keyword-gated."""
    return {"semantic": True, "sources": bool(_SOURCES_RE.search(question or ""))}

def parse_tables(s: str) -> list[str]:
    """`tables` columns are comma-separated, 1/2/3-level qualified depending on DB
    (table | schema.table | catalog.schema.table); views may use timbr.."""
    return [t.strip() for t in (s or "").split(",") if t.strip()]

# --------------------------------------------------------------------------- #
# generate_sql short-circuit
# --------------------------------------------------------------------------- #
def build_metadata_sql(concept: str, question: str, ontology: str) -> dict:
    return {
        "kind": "ONTOLOGY_METADATA",
        "ontology": parse_ontology(concept, default=ontology),
        "scope": scope_of(question),
        "sql": BASE_SQL,  # valid best-effort query for callers that skip execute()
    }

# --------------------------------------------------------------------------- #
# query set
# --------------------------------------------------------------------------- #
_CONCEPTS_SQL = (
    "SELECT concept, inheritance, inheritance_level, primary_keys, label_keys, "
    "description, query AS concept_logic "
    "FROM timbr.SYS_ONTOLOGY ORDER BY inheritance_level, concept"
)
_PROP_DICT_SQL = (
    "SELECT property_name, property_type, is_measure, is_multi, description, logic_query "
    "FROM timbr.SYS_PROPERTIES"
)
_CONCEPT_PROP_SQL = (
    "SELECT concept, property_name, property_type FROM timbr.SYS_CONCEPT_PROPERTIES"
)
_RELS_SQL = (
    "SELECT concept, target_concept, relationship_name, inverse_name, "
    "source_properties, target_properties, is_mtm, transitivity, description "
    "FROM timbr.SYS_CONCEPT_RELATIONSHIPS WHERE is_inverse = 0"
)
_VIEWS_SQL = "SELECT view_name, description, is_cube, tables FROM timbr.SYS_VIEWS"

# source tables (only when scope.sources) — replaces SYS_LINEAGE
_SRC_CONCEPT_SQL = "SELECT concept, tables, datasource_id FROM timbr.SYS_CONCEPT_MAPPINGS"
_SRC_M2M_SQL = (
    "SELECT DISTINCT concept, target_concept, relationship_name, tables, datasource_id "
    "FROM timbr.SYS_CONCEPT_RELATIONSHIPS WHERE tables IS NOT NULL AND is_inverse = 0"
)
_SRC_MULTIVAL_SQL = (
    "SELECT DISTINCT concept, property_name, tables, datasource_id "
    "FROM timbr.SYS_CONCEPT_PROPERTIES WHERE tables IS NOT NULL"
)

_SEMANTIC = [
    ("concepts", _CONCEPTS_SQL),
    ("properties", _PROP_DICT_SQL),
    ("concept_properties", _CONCEPT_PROP_SQL),
    ("relationships", _RELS_SQL),
    ("views", _VIEWS_SQL),
]
_SOURCES = [
    ("concept_tables", _SRC_CONCEPT_SQL),
    ("m2m_tables", _SRC_M2M_SQL),
    ("multivalue_tables", _SRC_MULTIVAL_SQL),
]

# --------------------------------------------------------------------------- #
# execute branch — concept-centric serialization
# --------------------------------------------------------------------------- #
def _truthy(value) -> bool:
    """Normalize sys_* boolean-ish columns (`'true'`, `1`, `True`, ...)."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes", "t")


def _measure_formula(logic_query: str) -> str:
    """Measure formula = `logic_query` with a leading ``SELECT `` stripped."""
    return re.sub(r"^\s*select\s+", "", (logic_query or "").strip(), flags=re.I).strip()


def _strip_concept_logic(logic: str) -> str:
    """Subtype logic -> `<parent_concept> WHERE ...`.

    Drops the `SELECT ... FROM ` head and the schema qualifier / backticks from the
    first (parent) table reference, keeping the rest (WHERE clause). e.g.
    ``SELECT * FROM dtimbr.`customer` WHERE x = 1`` -> ``customer WHERE x = 1``.
    """
    s = (logic or "").strip()
    if not s:
        return ""
    m = re.search(r"\bfrom\s+", s, re.I)
    if not m:
        return re.sub(r"^\s*select\s+", "", s, flags=re.I).strip()
    rest = s[m.end():].lstrip()
    m2 = re.match(r"((?:`?\w+`?\.)*`?\w+`?)(.*)", rest, re.S)
    if not m2:
        return rest.strip()
    table_ref, tail = m2.group(1), m2.group(2)
    concept = table_ref.split(".")[-1].replace("`", "")
    return (concept + tail).strip()


def _pivot_concepts(concepts, properties, concept_properties, relationships, mappings) -> list:
    """Pivot the semantic sections into one object per concept (drops ``thing``)."""
    # property_name -> {is_measure, logic_query}
    prop_meta = {}
    for p in properties:
        name = p.get("property_name")
        if name:
            prop_meta[name] = {
                "is_measure": _truthy(p.get("is_measure")),
                "logic_query": p.get("logic_query"),
            }

    # concept -> [property_name, ...]
    concept_props = {}
    for r in concept_properties:
        c, n = r.get("concept"), r.get("property_name")
        if c and n:
            concept_props.setdefault(c, []).append(n)

    # concept -> [(relationship_name, target_concept), ...]
    concept_rels = {}
    for r in relationships:
        c = r.get("concept")
        if c:
            concept_rels.setdefault(c, []).append(
                (r.get("relationship_name"), r.get("target_concept"))
            )

    # concept -> underlying tables (first non-empty mapping wins; subtypes stay lean)
    concept_tables = {}
    for mrow in mappings:
        c = mrow.get("concept")
        t = (mrow.get("tables") or "").strip()
        if c and t and c not in concept_tables:
            concept_tables[c] = t

    out = []
    for c in concepts:
        name = c.get("concept")
        if not name or name == "thing":
            continue

        obj = {"concept": name}

        parents = parse_parents(c.get("inheritance"))
        if parents:
            obj["inherits"] = ", ".join(parents)

        description = (c.get("description") or "").strip()
        if description:
            obj["description"] = description

        if name in concept_tables:
            obj["tables"] = concept_tables[name]

        prop_names = concept_props.get(name, [])
        plain = sorted(n for n in prop_names if not prop_meta.get(n, {}).get("is_measure"))
        measures = sorted(n for n in prop_names if prop_meta.get(n, {}).get("is_measure"))
        if plain:
            obj["properties"] = ", ".join(plain)
        if measures:
            parts = []
            for mn in measures:
                formula = _measure_formula(prop_meta.get(mn, {}).get("logic_query"))
                parts.append(f"{mn} = {formula}" if formula else mn)
            obj["measures"] = ", ".join(parts)

        rels = [f"{rn} -> {tg}" for rn, tg in concept_rels.get(name, []) if rn and tg]
        if rels:
            obj["relationships"] = ", ".join(rels)

        logic = _strip_concept_logic(c.get("concept_logic"))
        if logic:
            obj["logic"] = logic

        out.append(obj)
    return out


def _pivot_views(views) -> list:
    """One object per view/cube: name + is_cube flag + tables/description when present."""
    out = []
    for v in views:
        name = v.get("view_name")
        if not name:
            continue
        obj = {"view": name, "is_cube": _truthy(v.get("is_cube"))}
        tables = (v.get("tables") or "").strip()
        if tables:
            obj["tables"] = tables
        description = (v.get("description") or "").strip()
        if description:
            obj["description"] = description
        out.append(obj)
    return out


def fetch_metadata_rows(ontology, scope, run_sql, get_version, cache=None):
    """
    Build a concept-centric rows[] for the metadata answer path.

    Instead of a flat, section-tagged dump, everything is pivoted under the object
    it belongs to: one dict per concept (dropping ``thing``) carrying its inherited
    parent, description, underlying tables, properties, measures (as ``name =
    formula``), relationships (``relationship -> target``) and subtype logic — each
    key present only when it has a value. View/cube objects are appended after the
    concepts. The generic answer chain consumes this rows[] unchanged.

    Parameters
    ----------
    scope       : retained for call-site compatibility; the concept-centric output
                  is scope-independent (underlying tables are always attached).
    run_sql     : callable(sql: str) -> list[dict]   (reuse the existing executor)
    get_version : callable(ontology: str) -> str     (reuse the existing helper)
    cache       : optional {get(key), set(key, val)}; keyed by ontology version.
    """
    version = get_version(ontology)
    key = f"meta:{ontology}:{version}"

    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return hit

    concepts = run_sql(_CONCEPTS_SQL) or []
    properties = run_sql(_PROP_DICT_SQL) or []
    concept_properties = run_sql(_CONCEPT_PROP_SQL) or []
    relationships = run_sql(_RELS_SQL) or []
    views = run_sql(_VIEWS_SQL) or []
    mappings = run_sql(_SRC_CONCEPT_SQL) or []

    rows = _pivot_concepts(concepts, properties, concept_properties, relationships, mappings)
    rows.extend(_pivot_views(views))

    if cache is not None:
        cache.set(key, rows)
    return rows


# --------------------------------------------------------------------------- #
# identify-concept catalog sourcing
# --------------------------------------------------------------------------- #
# The identify-concept context builder needs one extra query the metadata answer
# path does not: the property list of each view/cube. Concepts get their direct
# properties from SYS_CONCEPT_PROPERTIES; views expose theirs via this table.
_VIEW_PROP_SQL = (
    "SELECT view_name, property_name, property_type FROM timbr.SYS_VIEW_PROPERTIES"
)

# The catalog builder consumes the semantic sections plus view properties. Source
# tables are irrelevant to anchor-concept selection and are deliberately excluded.
_CATALOG_QUERIES = _SEMANTIC + [("view_properties", _VIEW_PROP_SQL)]


def parse_view_concepts(tables: str) -> list[str]:
    """Extract the connected concept/view axis of a view from its `tables` column.

    A view's `tables` is comma-separated and may mix physical refs and semantic
    refs. Only two-level semantic refs (`dtimbr.<concept>` / `timbr.<concept>`)
    name a concept the view is built on; one-level and three-level refs are
    physical tables, and `timbr.sys_*` are system tables. The root `thing` is
    never a connected concept (it is never a candidate either). Order is preserved
    and duplicates are dropped.
    """
    out: list[str] = []
    seen: set[str] = set()
    for raw in (tables or "").split(","):
        parts = [p.strip() for p in raw.strip().split(".")]
        if len(parts) != 2:
            continue
        prefix, name = parts[0].lower(), parts[1]
        if prefix not in ("dtimbr", "timbr") or not name:
            continue
        if prefix == "timbr" and name.lower().startswith("sys_"):
            continue
        if name.lower() == "thing":
            continue
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def build_inheritance_map(concept_rows) -> dict:
    """`{parent -> [child, ...]}` from concept rows' `inheritance` column.

    `inheritance` is a comma-separated list of direct parent concepts; the root
    marker `thing` (and blanks) are ignored. Child order follows row order, which
    `_CONCEPTS_SQL` already sorts by `(inheritance_level, concept)`.
    """
    children: dict[str, list[str]] = {}
    for row in concept_rows:
        child = row.get("concept")
        if not child or child == "thing":
            continue
        for parent in parse_parents(row.get("inheritance")):
            children.setdefault(parent, []).append(child)
    return children


def parse_parents(inheritance: str) -> list[str]:
    """Direct parent concepts from an `inheritance` column, excluding `thing`."""
    out: list[str] = []
    for p in (inheritance or "").split(","):
        p = p.strip()
        if p and p != "thing":
            out.append(p)
    return out


def fetch_catalog_rows(ontology, run_sql, get_version, cache=None) -> dict:
    """Fetch the raw sys_* rows the identify-concept catalog is built from.

    Returns `{section -> rows}` for concepts, properties, concept_properties,
    relationships, views and view_properties. Each section query is isolated —
    a failing/absent table (e.g. SYS_VIEW_PROPERTIES on an older backend) yields
    an empty section rather than aborting the whole catalog. Cached per ontology
    version so trigram/inheritance derivation happens at most once per version.
    """
    version = get_version(ontology)
    key = f"catalog:{ontology}:{version}"

    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return hit

    out: dict = {}
    for section, sql in _CATALOG_QUERIES:
        try:
            out[section] = run_sql(sql) or []
        except Exception:
            out[section] = []

    if cache is not None:
        cache.set(key, out)
    return out
