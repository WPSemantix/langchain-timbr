"""Concept pre-filter prompt — narrows a candidate concept set to those
relevant to the user's question before the Compact DDL is serialized.

Engages only when the estimated DDL would exceed ``metadata_context_filter_max_tokens``.
The pre-filter trims the candidate set; the downstream Step 1 filter LLM
then makes the final path-selection decision on the narrowed set.
"""

from __future__ import annotations

from typing import List


_SYSTEM_PROMPT = """You are narrowing a large ontology to a subset of concepts that may be relevant to a user's natural-language question.

You will receive:
- USER QUESTION: the natural-language question to answer.
- ANCHOR CONCEPT: the entity the SQL FROM clause will reference. ALWAYS include this in your output.
- CANDIDATE CONCEPTS: a list of concept names (optionally with short descriptions).

Your task is to return a JSON object with exactly one field:
- relevant_concepts: list of concept names from the CANDIDATE CONCEPTS list that may be needed to answer the question.

Rules:
- Always include the anchor concept.
- Prefer recall over precision — include any concept that *might* be needed. The downstream filter trims further.
- Do not invent names. Use only names from the CANDIDATE CONCEPTS list.
- Aim for a focused subset; do not return the full candidate list unless the question genuinely needs all of it.

Return ONLY the JSON object — no surrounding prose."""


_USER_TEMPLATE_WITH_DESCRIPTIONS = """USER QUESTION:
{question}

ANCHOR CONCEPT:
{anchor}

CANDIDATE CONCEPTS (with brief descriptions):
{candidates_block}
"""


_USER_TEMPLATE_NAMES_ONLY = """USER QUESTION:
{question}

ANCHOR CONCEPT:
{anchor}

CANDIDATE CONCEPTS:
{candidates_block}
"""


def _render_note_block(note: str) -> str:
    if not note or not note.strip():
        return ""
    return "\n\n**Additional Notes:**\n" + note.strip()


def build_prefilter_messages(
    *,
    question: str,
    anchor: str,
    candidates_block: str,
    with_descriptions: bool,
    note: str = "",
) -> List[dict]:
    """Return [{role,content}, {role,content}] for the pre-filter call.

    ``note`` carries conversation memory (follow-up context) and any
    caller-supplied notes — appended so the pre-filter LLM can keep
    previously-relevant concepts in scope.
    """
    template = (
        _USER_TEMPLATE_WITH_DESCRIPTIONS if with_descriptions
        else _USER_TEMPLATE_NAMES_ONLY
    )
    user_content = template.format(
        question=question.strip(),
        anchor=anchor.strip(),
        candidates_block=candidates_block.strip(),
    ) + _render_note_block(note)
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
