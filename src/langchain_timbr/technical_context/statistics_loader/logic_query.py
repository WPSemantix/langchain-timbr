"""Regex parser for logic-concept query strings from timbr.sys_ontology.

Extracts the referenced concept from queries of the form:
    SELECT * FROM dtimbr.inventory WHERE ...
    SELECT * FROM `dtimbr`.`inventory` WHERE ...
    SELECT * FROM "dtimbr"."inventory" WHERE ...
    (and mixed quoting variants)

Returns the concept name or None if no match is found.
"""

from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)


class LogicQueryParseError(Exception):
    """Raised when a logic query cannot be parsed (when config demands it)."""


# Matches: <optional_quote>dtimbr|timbr<optional_quote>.<optional_quote>concept_name<optional_quote> WHERE
_LOGIC_QUERY_RE = re.compile(
    r"""
        (?:["`])?            # optional opening quote on schema
        (?:`)?               # optional backtick on schema
        (d?timbr)            # schema: dtimbr or timbr
        (?:`)?               # optional closing backtick on schema
        (?:["`])?            # optional closing quote on schema
        \.                   # dot separator
        (?:["`])?            # optional opening quote on concept
        (?:`)?               # optional backtick on concept
        ([^\s"``.]+)         # concept name (capture group 2)
        (?:`)?               # optional closing backtick on concept
        (?:["`])?            # optional closing quote on concept
        \s+WHERE\b           # followed by WHERE keyword
    """,
    re.IGNORECASE | re.VERBOSE,
)


def parse_logic_query(query: str | None) -> str | None:
    """Parse a logic-concept query string to extract the referenced concept.

    Args:
        query: The query string from sys_ontology.query column.

    Returns:
        The concept name referenced in the query, or None if not parseable.

    Examples:
        >>> parse_logic_query("SELECT * FROM dtimbr.inventory WHERE status = 'active'")
        'inventory'
        >>> parse_logic_query("SELECT * FROM `dtimbr`.`my_concept` WHERE x > 0")
        'my_concept'
        >>> parse_logic_query(None)
        None
    """
    if not query:
        return None

    m = _LOGIC_QUERY_RE.search(query)
    if m:
        return m.group(2)

    return None
