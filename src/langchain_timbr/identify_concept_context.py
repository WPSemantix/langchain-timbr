"""
identify_concept_context.py

First step of the NL2SQL pipeline: given a natural-language question and an
ontology, produce the concept catalog the LLM reads to select the *anchor*
concept the question is about.

This runs **after** the existing concept/view filter — the candidate set handed
in here is already narrowed. The one hard invariant is **never drop a candidate**:
every concept/view in the filtered set stays visible to the LLM. Under token
pressure the builder only trims the *cheapest signal first* (descriptions are
truncated — never dropped — then relationships/measures are filtered to the
question, then sub-type detail collapses to name-only pointers) — it never
removes a selectable name from consideration.

The catalog is hierarchical:

* Parent concepts are rendered first (ordered by inheritance level then
  relationship-degree centrality), sub-concepts nested beneath them and marked as
  independently selectable.
* When a candidate's parent is **absent** from the filtered set, that parent's
  measures/relationships/properties are inlined onto the child so no signal is
  lost. Concepts with multiple parents inherit across the full ancestor DAG.
* Views/cubes list their properties, measures and the concepts they connect.
* Out-of-list sub-types that trigram-match the question are surfaced as inline
  hints under their parent (hint-only — never added to the selectable set).

Properties and sub-type hints are **always** filtered to the question via the
trigram matcher, so the rendered catalog is question-conditioned at every stage.
The in-memory `_Catalog` model itself is query-independent and cached per
ontology version (`_load_catalog`); only the rendering depends on the question.

Candidate registration and the ontology header stay in `determine_concept`; this
module only produces the descriptive lines, and only when
`config.enable_identify_concept_context` is on (falls back to the legacy flat
render on any error).
"""

from dataclasses import dataclass, field
from typing import Optional
import re

from . import config
from . import ontology_metadata as om
from . import trigram
from .kbclient import render_object_rules
from .utils.timbr_utils import run_query, cache_with_version_check



# --------------------------------------------------------------------------- #
# token counting (cl100k_base, char/4 fallback)
# --------------------------------------------------------------------------- #
def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = _count_tokens._enc
        if enc is None:
            enc = _count_tokens._enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


_count_tokens._enc = None


def _is_truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes", "t")


def _to_level(v) -> int:
    try:
        return int(v) - 1
    except (TypeError, ValueError):
        return 0


# --------------------------------------------------------------------------- #
# in-memory model
# --------------------------------------------------------------------------- #
@dataclass
class _Node:
    name: str
    description: str = ""
    level: int = 0
    parents: list = field(default_factory=list)
    measures: list = field(default_factory=list)        # concepts + views
    relationships: list = field(default_factory=list)   # (rel_name, target)
    properties: list = field(default_factory=list)      # (name, trigram_set)
    connected: list = field(default_factory=list)       # views only
    is_view: bool = False
    is_cube: bool = False
    tri: frozenset = frozenset()
    degree: int = 0



@dataclass
class _Catalog:
    nodes: dict                 # name -> _Node (every concept + view in ontology)
    children: dict              # parent -> [child, ...]


def _build_catalog(rows: dict) -> _Catalog:
    """Assemble the query-independent in-memory model from raw sys_* rows."""
    measure_props = {
        r.get("property_name")
        for r in rows.get("properties", [])
        if _is_truthy(r.get("is_measure"))
    }

    nodes: dict = {}
    for r in rows.get("concepts", []):
        name = r.get("concept")
        if not name or name == "thing":
            continue
        nodes[name] = _Node(
            name=name,
            description=(r.get("description") or "").strip(),
            level=_to_level(r.get("inheritance_level")),
            parents=om.parse_parents(r.get("inheritance")),
            tri=trigram.to_trigram_set(name),
        )

    for r in rows.get("concept_properties", []):
        node = nodes.get(r.get("concept"))
        prop = r.get("property_name")
        if node is None or not prop:
            continue
        # `_type_of_*` discriminator columns are inheritance machinery, not query
        # signal — they never anchor a question, so they go to neither the
        # property nor the measure axis (a matching sub-type hint conveys the
        # useful part). Excluded from both concept and view ingestion.
        if prop.startswith("_type_of_"):
            continue
        if prop in measure_props:
            if prop not in node.measures:
                node.measures.append(prop)
        elif prop not in [p[0] for p in node.properties]:
            node.properties.append((prop, trigram.to_trigram_set(prop)))

    for r in rows.get("relationships", []):
        src, tgt = r.get("concept"), r.get("target_concept")
        rel, inv = r.get("relationship_name"), r.get("inverse_name")
        if src in nodes and rel and tgt:
            pair = (rel, tgt)
            if pair not in nodes[src].relationships:
                nodes[src].relationships.append(pair)
        if tgt in nodes and inv and src:
            pair = (inv, src)
            if pair not in nodes[tgt].relationships:
                nodes[tgt].relationships.append(pair)

    for r in rows.get("views", []):
        name = r.get("view_name")
        if not name:
            continue
        nodes[name] = _Node(
            name=name,
            description=(r.get("description") or "").strip(),
            is_view=True,
            is_cube=_is_truthy(r.get("is_cube")),
            connected=om.parse_view_concepts(r.get("tables")),
            tri=trigram.to_trigram_set(name),
        )

    for r in rows.get("view_properties", []):
        node = nodes.get(r.get("view_name"))
        prop = r.get("property_name")
        if node is None or not node.is_view or not prop:
            continue
        if prop.startswith("_type_of_"):
            continue
        if node.is_cube:
            # `is_measure` is not server-filterable for cubes; the measure axis is
            # instead flagged by a `measure.` column-name prefix. Strip it so the
            # rendered name is bare (`total_revenue`, not `measure.total_revenue`).
            if prop.startswith("measure."):
                bare = prop[len("measure."):]
                if bare and not bare.startswith("_type_of_") and bare not in node.measures:
                    node.measures.append(bare)
            elif prop not in [p[0] for p in node.properties]:
                node.properties.append((prop, trigram.to_trigram_set(prop)))
        elif prop in measure_props:
            if prop not in node.measures:
                node.measures.append(prop)
        elif prop not in [p[0] for p in node.properties]:
            node.properties.append((prop, trigram.to_trigram_set(prop)))

    for node in nodes.values():
        node.degree = len(node.relationships)

    return _Catalog(nodes=nodes, children=om.build_inheritance_map(rows.get("concepts", [])))


@cache_with_version_check
def _load_catalog(conn_params: dict) -> _Catalog:
    """Fetch + assemble the catalog for one ontology, cached per ontology version."""
    def run_sql(sql):
        return run_query(sql, conn_params)

    # Version + caching are handled by the decorator, so the inner fetch runs the
    # queries directly (get_version returns None -> no extra SHOW VERSION call).
    rows = om.fetch_catalog_rows(
        conn_params.get("ontology", ""),
        run_sql,
        get_version=lambda _o: None,
        cache=None,
    )
    return _build_catalog(rows)


# --------------------------------------------------------------------------- #
# render options (token ladder)
# --------------------------------------------------------------------------- #
@dataclass
class _Opts:
    desc_trunc: bool = False    # truncate descriptions (never drop them)
    rel_trim: bool = False      # filter relationships/measures to the question
    collapse: bool = False      # collapse nested sub-types to name-only pointers


_WS_RE = re.compile(r"\s+")


def _truncate(text: str, limit: int) -> str:
    # Collapse internal whitespace (newlines, runs) first so the char budget is
    # spent on content, not formatting, and the result is a single tidy line.
    text = _WS_RE.sub(" ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rstrip()
    space = cut.rfind(" ")
    if space > 0:
        cut = cut[:space]
    return cut + "..."



def _tags_str(name: str, is_view: bool, tags: Optional[dict]) -> str:
    if not tags:
        return ""
    bucket = tags.get("view_tags" if is_view else "concept_tags") or {}
    val = bucket.get(name)
    if not val:
        return ""
    return str(val).replace("{", "").replace("}", "").replace("'", "")


# --------------------------------------------------------------------------- #
# builder entry point
# --------------------------------------------------------------------------- #
def build_catalog_lines(question, conn_params, concepts_and_views, tags=None, prefix="", rules=None):
    """Return the hierarchical catalog lines for one ontology's filtered candidates.

    `concepts_and_views` is the already-filtered `{name -> {concept, description,
    is_view, ...}}` mapping from `get_concepts`. `prefix` is the backtick-qualified
    ontology prefix used in multi-ontology mode (e.g. `` `sales`. ``). Candidate
    registration happens in the caller — this function only renders descriptions,
    so it can never drop a candidate from the selectable set.

    `rules` is an optional `kbclient.RuleSet`; when present, concept/view/cube
    SELECTION_RULE text is appended as an indented sub-block under each node
    (outside the token-budget cascade — rules are never trimmed).
    """
    def _rules_lines(name, indent):
        if rules is None:
            return []
        txt = render_object_rules(rules.rules_for(name, ("concept", "view", "cube"), {"selection"}))
        if not txt:
            return []
        return [f"{indent}    {line}" for line in txt.split("\n")]
    catalog = _load_catalog(conn_params)
    filtered = set(concepts_and_views.keys())
    q_tri = trigram.to_trigram_set(question)
    q_padded = trigram.pad_question(question)
    thr = config.identify_concept_context_trigram_threshold
    floor = config.identify_concept_context_trigram_floor

    # ---- question-match memos (name meta computed once per name per query) --- #
    name_meta_memo: dict = {}

    def _name_meta(nm):
        meta = name_meta_memo.get(nm)
        if meta is None:
            meta = (trigram.to_trigram_set(nm), " ".join(trigram.to_tokens(nm)))
            name_meta_memo[nm] = meta
        return meta

    def name_matches(nm):
        tri, norm = _name_meta(nm)
        return trigram.matches(norm, tri, q_padded, q_tri, thr, floor)

    def prop_matches(pname, ptri):
        _, norm = _name_meta(pname)
        return trigram.matches(norm, ptri, q_padded, q_tri, thr, floor)

    def sort_key(name):
        node = catalog.nodes.get(name)
        if node is None:
            return (False, 0, 0, name)
        return (node.is_view, node.level, -node.degree, name)

    # ---- tree over the filtered set: attach each node to its nearest filtered
    #      ancestor across ALL parents (multiple inheritance); nodes with no
    #      filtered ancestor are top-level ------------------------------------ #
    def nearest_filtered_ancestor(name):
        node = catalog.nodes.get(name)
        frontier = list(node.parents) if node else []
        seen = set(frontier)
        while frontier:
            hits = sorted((p for p in frontier if p in filtered), key=sort_key)
            if hits:
                return hits[0]
            nxt = []
            for p in frontier:
                pn = catalog.nodes.get(p)
                if not pn:
                    continue
                for gp in pn.parents:
                    if gp not in seen:
                        seen.add(gp)
                        nxt.append(gp)
            frontier = nxt
        return None

    child_map: dict = {}
    top_level = []
    for name in filtered:
        anc = nearest_filtered_ancestor(name)
        if anc is None:
            top_level.append(name)
        else:
            child_map.setdefault(anc, []).append(name)

    # ---- ancestor DAG helpers (full multi-inheritance closure) -------------- #
    def ancestor_closure(name):
        """`name` plus every transitive ancestor present in the catalog."""
        out, frontier, seen = [], [name], {name}
        while frontier:
            cur = frontier.pop(0)
            out.append(cur)
            node = catalog.nodes.get(cur)
            if not node:
                continue
            for p in node.parents:
                if p in seen or p not in catalog.nodes:
                    continue
                seen.add(p)
                frontier.append(p)
        return out

    def strict_ancestors(name):
        return set(ancestor_closure(name)) - {name}

    def _member_key(m):
        return m[0] if isinstance(m, tuple) else m

    def _collect(order, attr):
        """Union `attr` members over `order`, preserving first-seen order."""
        out, seen = [], set()
        for cur in order:
            node = catalog.nodes.get(cur)
            if not node:
                continue
            for m in getattr(node, attr):
                key = _member_key(m)
                if key not in seen:
                    seen.add(key)
                    out.append(m)
        return out

    def inherited(name, stop_at, attr):
        """Members of `name` across its full ancestor DAG.

        When nested under a filtered ancestor `stop_at`, subtract that ancestor's
        own inherited members (member-level, keyed by name) so the child only
        shows what it adds — the parent already rendered the shared set.
        """
        members = _collect(ancestor_closure(name), attr)
        if stop_at is None:
            return members
        stop_keys = {_member_key(m) for m in _collect(ancestor_closure(stop_at), attr)}
        return [m for m in members if _member_key(m) not in stop_keys]


    # ---- out-of-list sub-type hints (question-matched, hint-only) ----------- #
    def descendants(name):
        out, stack, seen = [], list(catalog.children.get(name, [])), set()
        while stack:
            cur = stack.pop(0)
            if cur in seen:
                continue
            seen.add(cur)
            out.append(cur)
            stack.extend(catalog.children.get(cur, []))
        return out

    def subtype_hints(name):
        owner = catalog.nodes.get(name)
        owner_tri = owner.tri if owner else frozenset()
        owner_tokens = set(trigram.to_tokens(name))
        hits = []
        for d in descendants(name):
            if d in filtered:
                continue
            node = catalog.nodes.get(d)
            if not node:
                continue
            # Match on the descendant's *distinguishing* signal only: subtract the
            # block owner's grams/tokens so shared stems (e.g. `company` in every
            # `*_company` sub-type) can't make an unrelated question hit them. The
            # matcher's short-name fallback handles sub-types whose distinguishing
            # part is too short to trigram.
            distinguishing_tri = node.tri - owner_tri
            distinguishing_norm = " ".join(
                t for t in trigram.to_tokens(node.name) if t not in owner_tokens
            )
            if trigram.matches(distinguishing_norm, distinguishing_tri, q_padded, q_tri, thr, floor):
                hits.append(d)
        return hits

    def filtered_descendants(name):
        return [d for d in descendants(name) if d in filtered]

    def rel_matches(src, rel, tgt):
        # Three-axis: keep a relationship whose source, name, or target the
        # question references (source self-protection is applied by the caller).
        return name_matches(src) or name_matches(rel) or name_matches(tgt)

    desc_max = config.identify_concept_context_desc_max_chars
    hint_cap = config.identify_concept_context_hint_cap


    # ---- rendering ---------------------------------------------------------- #
    def render_concept(name, indent, stop_at, opts, out, seen):
        if name in seen:
            return
        seen.add(name)
        node = catalog.nodes.get(name)
        item = concepts_and_views.get(name, {})
        header = f"{indent}- {prefix}`{name}`"
        bits = []
        level = node.level if node else 0
        if level > 1:
            bits.append(f"[level: {level}]")
        if stop_at is not None:
            bits.append(f"(inherits `{stop_at}`, independently selectable)")
            # Multiple inheritance: when this child also descends from other
            # filtered ancestors on divergent lines (not themselves ancestors of
            # `stop_at`), flag them so the reader knows the nested placement is
            # only one of its lineages.
            others = sorted(
                (a for a in filtered
                 if a != stop_at
                 and a in strict_ancestors(name)
                 and a not in strict_ancestors(stop_at)),
                key=sort_key,
            )
            if others:
                bits.append("(also inherits " + ", ".join(f"`{a}`" for a in others) + ")")
        elif node and node.parents:
            bits.append(f"(inherits `{node.parents[0]}`)")
        desc = (item.get("description") or (node.description if node else "") or "").strip()
        if desc:
            if opts.desc_trunc:
                desc = _truncate(desc, desc_max)
            bits.append(f" (description: {desc})")
        tag_str = _tags_str(name, False, tags)
        if tag_str:
            bits.append(f"[tags: {tag_str}]")
        if bits:
            header += " " + " ".join(bits)
        out.append(header)
        out.extend(_rules_lines(name, indent))

        measures = inherited(name, stop_at, "measures")
        if measures:
            if opts.rel_trim and not name_matches(name):
                shown = [m for m in measures if name_matches(m)]
                extra = len(measures) - len(shown)
            else:
                shown, extra = measures, 0
            rendered = ", ".join(f"`{m}`" for m in shown)
            if extra:
                rendered = (rendered + ", " if rendered else "") + f"+{extra} more"
            out.append(f"{indent}    measures: {rendered}")

        # Properties are always filtered to the question and never trimmed by the
        # ladder — the count shows how much was withheld.
        props = inherited(name, stop_at, "properties")
        if props:
            matched = [p for p in props if prop_matches(p[0], p[1])]
            if matched:
                rendered = ", ".join(f"`{p[0]}`" for p in matched)
                out.append(
                    f"{indent}    properties ({len(matched)} of {len(props)} match): {rendered}"
                )

        rels = inherited(name, stop_at, "relationships")
        if opts.rel_trim and not name_matches(name):
            kept = [pair for pair in rels if rel_matches(name, pair[0], pair[1])]
            extra = len(rels) - len(kept)
        else:
            kept, extra = rels, 0
        if kept or extra:
            rendered = ", ".join(f"`{rel}` -> `{tgt}`" for rel, tgt in kept)
            if extra:
                rendered = (rendered + ", " if rendered else "") + f"+{extra} more"
            out.append(f"{indent}    relationships: {rendered}")

        hints = subtype_hints(name)
        if hints:
            shown, extra = hints[:hint_cap], max(0, len(hints) - hint_cap)
            rendered = ", ".join(f"`{h}`" for h in shown)
            if extra:
                rendered += f", +{extra} more"
            out.append(f"{indent}    narrow to sub-types: {rendered}")

        if opts.collapse:
            subs = filtered_descendants(name)
            if subs:
                rendered = ", ".join(f"`{prefix}{s}`" if prefix else f"`{s}`" for s in subs)
                out.append(f"{indent}    sub-types (selectable): {rendered}")
        else:
            for child in sorted(child_map.get(name, []), key=sort_key):
                render_concept(child, indent + "  ", name, opts, out, seen)

    def render_view(name, opts, out):
        node = catalog.nodes.get(name)
        item = concepts_and_views.get(name, {})
        kind = "cube" if (node and node.is_cube) else "view"
        header = f"- {prefix}`{name}` ({kind})"
        bits = []
        desc = (item.get("description") or (node.description if node else "") or "").strip()
        if desc:
            if opts.desc_trunc:
                desc = _truncate(desc, desc_max)
            bits.append(f" (description: {desc})")
        tag_str = _tags_str(name, True, tags)
        if tag_str:
            bits.append(f"[tags: {tag_str}]")
        if bits:
            header += " " + " ".join(bits)
        out.append(header)
        out.extend(_rules_lines(name, ""))

        props = node.properties if node else []
        if props:
            matched = [p for p in props if prop_matches(p[0], p[1])]
            if matched:
                rendered = ", ".join(f"`{p[0]}`" for p in matched)
                out.append(f"    properties ({len(matched)} of {len(props)} match): {rendered}")
        if node and node.measures:
            measures = node.measures
            if opts.rel_trim and not name_matches(name):
                shown = [m for m in measures if name_matches(m)]
                extra = len(measures) - len(shown)
            else:
                shown, extra = measures, 0
            rendered = ", ".join(f"`{m}`" for m in shown)
            if extra:
                rendered = (rendered + ", " if rendered else "") + f"+{extra} more"
            out.append(f"    measures: {rendered}")
        if node and node.connected:
            out.append("    connected concepts: " + ", ".join(f"`{c}`" for c in node.connected))

    def render(opts) -> str:
        out = []
        seen = set()
        for name in sorted(top_level, key=sort_key):
            node = catalog.nodes.get(name)
            if node and node.is_view:
                seen.add(name)
                render_view(name, opts, out)
            else:
                render_concept(name, "", None, opts, out, seen)
        # Never-drop safety net: a filtered concept unreachable from any root
        # (e.g. an inheritance cycle where every node has a filtered ancestor, so
        # nothing is top-level) is emitted as its own root rather than lost.
        for name in sorted(filtered - seen, key=sort_key):
            node = catalog.nodes.get(name)
            if node and node.is_view:
                seen.add(name)
                render_view(name, opts, out)
            else:
                render_concept(name, "", None, opts, out, seen)
        return "\n".join(out)

    # ---- token ladder: cut the cheapest signal first, never a candidate ----- #
    desc_ceiling = config.identify_concept_context_desc_trim_tokens
    rel_ceiling = config.identify_concept_context_rel_trim_tokens
    hard_ceiling = config.identify_concept_context_hard_limit_tokens

    stages = [
        (_Opts(), desc_ceiling),                                        # full descriptions
        (_Opts(desc_trunc=True), rel_ceiling),                          # truncate descriptions
        (_Opts(desc_trunc=True, rel_trim=True), hard_ceiling),          # filter rels/measures
        (_Opts(desc_trunc=True, rel_trim=True, collapse=True), hard_ceiling),  # collapse sub-types
    ]

    # The final stage (collapsed floor) is the trimming terminus, not a failure
    # gate: even if it still exceeds the hard ceiling it is returned as-is, so a
    # candidate name is never dropped. Only the last stage is emitted unconditionally.
    text = ""
    last = len(stages) - 1
    for i, (opts, ceiling) in enumerate(stages):
        text = render(opts)
        if i == last or _count_tokens(text) <= ceiling:
            return [text]
    return [text]  # unreachable — the loop always returns at i == last
