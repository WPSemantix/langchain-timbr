"""Parse column names into structured ColumnPath objects.

Handles three shapes:
  - Direct:    "customer_id"           → hops=[], final_property="customer_id"
  - Single-hop: "orders[order].total"  → hops=[("orders","order")], final_property="total"
  - Multi-hop:  "orders[order].items[order_item].quantity"
                → hops=[("orders","order"),("items","order_item")], final_property="quantity"
"""

from __future__ import annotations

import re
import logging
from typing import Optional

from .types import ColumnPath

logger = logging.getLogger(__name__)

# Matches relationship[concept] segments
_HOP_RE = re.compile(r"(\w+)\[([^\]]+)\]")


class ColumnPathParseError(Exception):
    """Raised when a column name cannot be parsed into a valid ColumnPath."""


def parse_column_path(column_name: str, selected_table: str) -> ColumnPath:
    """Parse a column name into a ColumnPath.

    Args:
        column_name: Raw column name (e.g. "orders[order].items[order_item].quantity")
        selected_table: The root concept/table name, used as owning_concept when no hops.

    Returns:
        ColumnPath with parsed hops, final_property, and owning_concept.

    Raises:
        ColumnPathParseError: If the column name has malformed brackets.
    """
    if not column_name or not column_name.strip():
        raise ColumnPathParseError(f"Empty column name")

    # Check for malformed brackets
    _validate_brackets(column_name)

    # Extract all hop segments: relationship[concept]
    hops: list[tuple[str, str]] = _HOP_RE.findall(column_name)

    if not hops:
        # Direct column — no relationship hops
        # Strip any leading "measure." prefix for the final_property
        final_property = column_name
        return ColumnPath(
            raw=column_name,
            hops=[],
            final_property=final_property,
            owning_concept=selected_table,
        )

    # Remove all hop segments from the string to find the final property
    remaining = _HOP_RE.sub("", column_name)
    # remaining is like ".total" or ".items..quantity" → split on '.', take last non-empty
    parts = [p for p in remaining.split(".") if p]

    if not parts:
        raise ColumnPathParseError(
            f"Column name '{column_name}' has relationship hops but no final property"
        )

    final_property = parts[-1]
    owning_concept = hops[-1][1]

    return ColumnPath(
        raw=column_name,
        hops=hops,
        final_property=final_property,
        owning_concept=owning_concept,
    )


def _validate_brackets(column_name: str) -> None:
    """Validate bracket structure in a column name."""
    depth = 0
    i = 0
    while i < len(column_name):
        ch = column_name[i]
        if ch == "[":
            depth += 1
            if depth > 1:
                raise ColumnPathParseError(
                    f"Nested brackets in column name '{column_name}'"
                )
            # Check for empty brackets
            if i + 1 < len(column_name) and column_name[i + 1] == "]":
                raise ColumnPathParseError(
                    f"Empty brackets in column name '{column_name}'"
                )
        elif ch == "]":
            if depth == 0:
                raise ColumnPathParseError(
                    f"Unbalanced closing bracket in column name '{column_name}'"
                )
            depth -= 1
        i += 1

    if depth != 0:
        raise ColumnPathParseError(
            f"Unbalanced opening bracket in column name '{column_name}'"
        )
