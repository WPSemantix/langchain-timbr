"""Step 1 filter-and-path-inference prompt for the dynamic metadata pipeline.

Renders system + user messages as plain dicts so the LLM wrapper layer can
adapt them to whichever backend is in use (langchain ChatPromptTemplate works
on dicts via convert_to_messages, OpenAI raw, etc.).

Action grammar: the planner emits EXACTLY ONE action per response
(``build_path | expand_to | reanchor``). ``allowed_actions`` narrows the
prompt's documented action set when the orchestrator has exhausted a cap —
the LLM is shown only what it's still allowed to request, so capped actions
become un-emittable rather than being rejected after the fact.

``clarify`` is NOT supported by this pipeline — chat callers handle ambiguity
via re-ask, not in-band.
"""

from __future__ import annotations

from typing import List, Optional


_PROMPT_INTRO = """You are filtering an ontology subgraph and selecting the traversal paths needed to answer a user's question via SQL.

You will receive:
- USER QUESTION: the natural-language question to answer.
- ANCHOR CONCEPT: the entity the SQL FROM clause will reference.
- ONTOLOGY SUBGRAPH: a Compact DDL listing of concepts, properties, measures, and relationships reachable from the anchor. The subgraph has TWO bands:
    * `## CONCEPTS` — concept blocks. Most are FULL-detail (description, props, measures, rels, incoming). Concepts promoted via a prior `expand_to` appear as MINIMAL blocks: heading + `description:` + a `rels:` block showing the concept's full outgoing relationships (including edges to `## REACHABLE` names). They omit `props:`, `measures:`, and `incoming:`. A MINIMAL block's rels are fully usable for `build_path` — an edge it shows to a `## REACHABLE` concept lets you END a path there with no further expand. **ANY concept with a `### <name>` heading in `## CONCEPTS` — FULL or MINIMAL, at any hop — is fully usable for `build_path` segments. Do NOT request expansion for these names.**
    * `## REACHABLE: name1, name2, ...` — names only. You can use one as the end of a path as soon as you see an edge pointing to it in a rendered `rels:` block — no expand needed to end there. Only `expand_to` a concept when you need to go THROUGH it to reach something further.

Edge direction in the subgraph (IMPORTANT — do NOT reverse):
- `rels:` lines under a concept are the ONLY traversable edges. Format `-[<rel>, <card>]-> <target>` means the edge goes FROM this concept TO <target>. Path segments MUST use this direction: {from: <this concept>, rel: <rel>, to: <target>}.
- `incoming:` lines (format `- <source>.<rel>`) are a REFERENCE LIST showing which other concepts in the subgraph point AT this concept. They are NOT traversable in this direction — the actual edge lives in `<source>`'s `rels:` block. If you need to use such an edge, write the segment in its native direction: {from: <source>, rel: <rel>, to: <this concept>}, not the reverse.
- Never invert a `rels:` edge. If the question implies a "reverse" traversal (e.g. starting from an `incoming:` source), START the path from that source concept; do not flip the from/to fields."""


_DECISION_PROCEDURE = """## DECISION PROCEDURE (follow in order, stop at first match)
1. Identify the TARGET: the deepest concept named or implied by the question.
2. Scan every visible `rels:` block for an edge pointing to TARGET.
3. If such an edge is visible → emit `build_path`. Do NOT expand.
4. Else if TARGET already has a `###` heading in `## CONCEPTS` → emit `build_path`. Do NOT expand.
5. Else if TARGET is in `## REACHABLE` → emit `expand_to: [TARGET]` (the destination itself, never an intermediate hop).
6. After any expansion, repeat from step 2.

**Generic guidance:**
- Always prefer `build_path` the moment any required edge becomes visible or the target is already in `## CONCEPTS`.
- Never expand a concept just because it is "on the way" — only expand the actual TARGET named/implied by the question.
- One `expand_to` on the destination reveals the full connecting chain (intermediates become usable without naming them yourself)."""


_ACTION_PREAMBLE = """Your task is to return a JSON object with EXACTLY ONE action per response. The shape of the object is one of the variants below — set the `action` field to indicate which variant you are emitting. Do NOT mix payloads from different actions in a single response."""


_BUILD_PATH_VARIANT = """### Action: `build_path`
Use this when you can construct the SQL traversal from concepts already in the `## CONCEPTS` (detail) band. This is the normal terminal action.

```json
{
  "action": "build_path",
  "selected_concepts": ["<concept>", ...],
  "selected_properties": ["<concept>.<property>", ...],
  "selected_measures": ["<concept>.<measure>", ...],
  "selected_paths": [
    {
      "path_id": "P1",
      "purpose": "<why this path is needed>",
      "segments": [
        { "from": "<concept>", "rel": "<rel_name>", "to": "<target>", "is_intermediate": false }
      ],
      "is_recursive": false
    }
  ],
  "transitivity_overrides": [
    { "rel": "<rel_name>", "target": "<target>", "level": 3 }
  ]
}
```

Field guidance for build_path:
- `selected_paths.segments[].is_intermediate` (default false): mark true ONLY when the segment's `to` concept exists solely to enable traversal to a deeper concept and the user would NOT expect that concept's data in the result. Anchor is never intermediate; terminal of any path is never intermediate.
- `selected_paths.is_recursive`: true only when the path includes a recursive relationship AND recursion is intended.
- `transitivity_overrides`: depth on a transitive/recursive relationship. One entry per (rel, target).

Path-construction rules:
- Each path's first segment must start from the anchor OR a concept reached by a prior path in this response.
- Single-hop style (chained by order) and full-chain style both accepted. Prefer single-hop.
- `to` of one segment must equal `from` of the next.
- Use exact names from the subgraph. Prefer shorter paths when both reach the same target."""


_EXPAND_TO_VARIANT = """### Action: `expand_to`
Use ONLY after following the DECISION PROCEDURE. Expand the destination the question names, never a nearest hop.

```json
{
  "action": "expand_to",
  "expand_to": ["<menu_concept_name>"]
}
```

Field guidance for expand_to:
- Targets must come **exclusively** from `## REACHABLE`. Anything already in `## CONCEPTS` (FULL or MINIMAL) is invalid and will be rejected.
- Reveals the target's outgoing `rels:` plus any connecting intermediates from the current frontier in one step.
- If an edge to TARGET is already visible (per step 2 of the procedure) → do NOT expand; emit `build_path`.
- One expand reveals the full necessary chain — do not expand hop-by-hop.
- Consumes one round from the per-request expand budget."""


_REANCHOR_VARIANT = """### Action: `reanchor`
Use when the current ANCHOR is the wrong SQL FROM root (the natural answer subject is a different concept and forcing the join would distort results). Only one reanchor allowed per request.

```json
{
  "action": "reanchor",
  "reanchor_to": "<concept_name>"
}
```

Field guidance for reanchor:
- Target must be visible in `## CONCEPTS` or `## REACHABLE`.
- Choose the entity whose rows the query should primarily return or aggregate over."""


_RETURN_INSTRUCTION = """Return ONLY the JSON object — no surrounding prose."""


_RETRY_PREAMBLE = """PREVIOUS OUTPUT HAD VALIDATION ERRORS:

{errors}

Please correct your response. The subgraph, allowed actions, and rules above are unchanged."""


_USER_TEMPLATE = """USER QUESTION:
{question}

ANCHOR CONCEPT:
{anchor}

ONTOLOGY SUBGRAPH:
{compact_ddl}
"""


def _render_note_block(note: str) -> str:
    """Render an Additional Notes block (incl. conversation memory) for the
    user message. Returns an empty string when ``note`` is falsy.

    Same heading convention as the SQL-gen note channel so any follow-up
    conversation memory injected via ``format_memory_note_for_sql`` flows
    through unchanged.
    """
    if not note or not note.strip():
        return ""
    return "\n\n**Additional Notes:**\n" + note.strip()


def _build_system_prompt(allowed_actions: List[str]) -> str:
    """Assemble the system prompt with the variants for the allowed actions only.

    When a cap has been exhausted (e.g. ``expand_to`` budget spent), the
    caller omits that action from ``allowed_actions`` and the LLM is shown
    a narrower union. This is the "grammar narrowing" enforcement —
    capped actions become un-emittable rather than emitted-and-rejected.

    The ``## DECISION PROCEDURE`` block (see ``_DECISION_PROCEDURE``) sits
    between the two-band intro and the action specs. Empirically it cuts the
    cold-start "crawl" failure mode by ~90 percentage points on the
    supply-metrics regression — see ``.claude/eval_planner_prompt_variants.py``
    for the A/B/C/D harness that landed on this design (variant D).
    """
    allowed_set = set(allowed_actions)
    parts: List[str] = [_PROMPT_INTRO, _DECISION_PROCEDURE, _ACTION_PREAMBLE]
    parts.append(
        "Allowed actions this round: " + ", ".join(
            f"`{a}`" for a in allowed_actions
        ) + "."
    )
    if "build_path" in allowed_set:
        parts.append(_BUILD_PATH_VARIANT)
    if "expand_to" in allowed_set:
        parts.append(_EXPAND_TO_VARIANT)
    if "reanchor" in allowed_set:
        parts.append(_REANCHOR_VARIANT)
    parts.append(_RETURN_INSTRUCTION)
    return "\n\n".join(parts)


def build_filter_messages(
    *,
    question: str,
    anchor: str,
    compact_ddl: str,
    note: str = "",
    allowed_actions: Optional[List[str]] = None,
) -> List[dict]:
    """Return [{role,content}, {role,content}] for the filter call.

    ``allowed_actions`` narrows the documented action union for this round
    (default: all three of ``build_path``, ``expand_to``, ``reanchor``).
    When the orchestrator has hit a cap (e.g. expand_count == EXPAND_CAP),
    it passes a shorter list and the LLM is shown only the remaining
    variants — so the disallowed action is structurally unemittable.

    ``note`` carries conversation memory (when this is a follow-up question)
    and any caller-supplied notes — appended verbatim as the Additional Notes
    block so the filter LLM sees prior context.
    """
    if allowed_actions is None:
        allowed_actions = ["build_path", "expand_to", "reanchor"]
    system_prompt = _build_system_prompt(allowed_actions)
    user_content = _USER_TEMPLATE.format(
        question=question.strip(),
        anchor=anchor.strip(),
        compact_ddl=compact_ddl.strip(),
    ) + _render_note_block(note)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def build_retry_messages(
    *,
    question: str,
    anchor: str,
    compact_ddl: str,
    error_lines: List[str],
    note: str = "",
    allowed_actions: Optional[List[str]] = None,
) -> List[dict]:
    """Return messages for the retry call, with reason codes injected.

    Same ``allowed_actions`` semantics as ``build_filter_messages``.
    """
    if allowed_actions is None:
        allowed_actions = ["build_path", "expand_to", "reanchor"]
    system_prompt = _build_system_prompt(allowed_actions)
    errors_block = "\n".join(f"- {line}" for line in error_lines)
    retry_user = _USER_TEMPLATE.format(
        question=question.strip(),
        anchor=anchor.strip(),
        compact_ddl=compact_ddl.strip(),
    ) + _render_note_block(note) + "\n\n" + _RETRY_PREAMBLE.format(errors=errors_block)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": retry_user},
    ]
