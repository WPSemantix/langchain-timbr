"""Parse the `stats` JSON column from sys_properties_statistics.

Each row has exactly one of:
  {"top_k": [{"value": "COMPLETE", "count": 29754}, ...]}
  {"min_value": "-2.750000000000000", "max_value": "0.500000000000000"}

Min/max values are strings in the JSON and must be converted to proper types
based on the column's SQL type to enable correct numeric/temporal comparison.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from ..matching.normalize import normalize, normalize_keep_spaces
from .types import TopKEntry, ValueKind

logger = logging.getLogger(__name__)

# SQL types mapped to parse functions
_INTEGER_TYPES = frozenset({"int", "bigint", "smallint", "tinyint", "integer"})
_DECIMAL_TYPES = frozenset({"decimal", "numeric", "float", "double", "real"})
_DATE_TYPES = frozenset({"date"})
_DATETIME_TYPES = frozenset({"timestamp", "datetime"})


_DIGITS = frozenset("0123456789")


def _is_numeric(value: str) -> bool:
    """``float()`` rather than a hand-rolled scan: one C call beats an
    interpreted loop, measured 14 ms against 39 ms over 873k values."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def _is_date(value: str) -> bool:
    """``YYYY-MM-DD…`` shape, checked positionally — no regex, no allocation."""
    return (
        len(value) >= 8 and value[4] in "-/"
        and value[0] in _DIGITS and value[1] in _DIGITS
        and value[2] in _DIGITS and value[3] in _DIGITS
    )


def classify_values(values: list[str]) -> ValueKind:
    """Classify a column's values, short-circuiting as soon as both hypotheses die.

    Three quarters of columns are text and exit on their first value, so this
    measures 30.8 ms over 872,953 values — 0.03 ms per column, and once per
    fetch rather than once per request.
    """
    if not values:
        return "text"
    numeric = date = True
    for v in values:
        if numeric and not _is_numeric(v):
            numeric = False
        if date and not _is_date(v):
            date = False
        if not numeric and not date:
            return "text"
    if numeric:
        return "numeric"
    if date:
        return "date"
    return "text"


def index_date_years(values: list[str]) -> dict[str, tuple[frozenset[str], int]]:
    """year -> (months present, count). 12 entries instead of 1,408 values."""
    acc: dict[str, list] = {}
    for v in values:
        year, month = v[:4], v[5:7]
        slot = acc.get(year)
        if slot is None:
            acc[year] = [{month}, 1]
        else:
            slot[0].add(month)
            slot[1] += 1
    return {y: (frozenset(m), n) for y, (m, n) in acc.items()}


def parse_stats_json(
    stats_str: str | None,
    sql_type: str | None = None,
    norm_memo: dict[str, tuple[str, str]] | None = None,
) -> tuple[list[TopKEntry] | None, Any | None, Any | None]:
    """Parse the stats JSON column into structured data.

    Args:
        stats_str: Raw JSON string from the `stats` column. May be None.
        sql_type: The column's SQL type (e.g. "decimal(18,2)", "bigint", "date").
                  Used for type-aware min/max parsing.
        norm_memo: Optional value -> (norm, norm_space) memo, shared across the
                  rows of one fetch. The same value appears in many mappings'
                  top-K lists (measured 6.5x on the captured fixture), so without
                  it each occurrence would hold its own copy of the normalized
                  strings. Scoped to the caller's fetch, so nothing accumulates.

    Returns:
        Tuple of (top_k, min_value, max_value).
        Exactly one of (top_k) or (min_value/max_value) will be non-None on healthy data.
        All-None on parse failure.
    """
    if not stats_str:
        return None, None, None

    try:
        data = json.loads(stats_str)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("Failed to parse stats JSON: %s — %s", e, stats_str[:100] if stats_str else "")
        return None, None, None

    if not isinstance(data, dict):
        logger.warning("Stats JSON is not a dict: %s", type(data))
        return None, None, None

    # Top-K path
    if "top_k" in data:
        raw_top_k = data["top_k"]
        if isinstance(raw_top_k, list):
            top_k = []
            for entry in raw_top_k:
                if isinstance(entry, dict) and "value" in entry and "count" in entry:
                    try:
                        value = str(entry["value"])
                        if norm_memo is None:
                            norm, norm_space = normalize(value), normalize_keep_spaces(value)
                        else:
                            pair = norm_memo.get(value)
                            if pair is None:
                                pair = norm_memo[value] = (
                                    normalize(value), normalize_keep_spaces(value),
                                )
                            norm, norm_space = pair
                        top_k.append(TopKEntry(
                            value=value,
                            count=int(entry["count"]),
                            norm=norm,
                            norm_space=norm_space,
                        ))
                    except (ValueError, TypeError):
                        continue
            return top_k if top_k else None, None, None
        return None, None, None

    # Min/Max path
    if "min_value" in data or "max_value" in data:
        raw_min = data.get("min_value")
        raw_max = data.get("max_value")

        min_value = _parse_typed_value(raw_min, sql_type) if raw_min is not None else None
        max_value = _parse_typed_value(raw_max, sql_type) if raw_max is not None else None

        return None, min_value, max_value

    # Neither top_k nor min/max
    logger.warning("Stats JSON has neither top_k nor min/max: %s", list(data.keys()))
    return None, None, None


def _parse_typed_value(raw_value: Any, sql_type: str | None) -> Any:
    """Parse a raw min/max value string to the appropriate Python type.

    Falls back to the raw string (with warning) if parsing fails.
    """
    if raw_value is None:
        return None

    value_str = str(raw_value)
    base_type = _extract_base_type(sql_type) if sql_type else None

    if base_type in _INTEGER_TYPES:
        try:
            return int(Decimal(value_str))
        except (InvalidOperation, ValueError, OverflowError):
            logger.warning("Cannot parse '%s' as integer (type=%s), keeping as string", value_str, sql_type)
            return value_str

    if base_type in _DECIMAL_TYPES:
        try:
            return Decimal(value_str)
        except InvalidOperation:
            logger.warning("Cannot parse '%s' as Decimal (type=%s), keeping as string", value_str, sql_type)
            return value_str

    if base_type in _DATE_TYPES:
        try:
            return date.fromisoformat(value_str)
        except ValueError:
            logger.warning("Cannot parse '%s' as date (type=%s), keeping as string", value_str, sql_type)
            return value_str

    if base_type in _DATETIME_TYPES:
        try:
            return datetime.fromisoformat(value_str)
        except ValueError:
            logger.warning("Cannot parse '%s' as datetime (type=%s), keeping as string", value_str, sql_type)
            return value_str

    # Unknown type — keep as string (no warning for unknown types, this is expected)
    return value_str


def _extract_base_type(sql_type: str) -> str | None:
    """Extract the base type name from a SQL type string (strip precision/scale).

    "decimal(18,2)" → "decimal"
    "varchar(255)" → "varchar"
    "bigint" → "bigint"
    """
    if not sql_type:
        return None
    # Take everything before the first '(' and lowercase
    base = sql_type.split("(")[0].strip().lower()
    return base if base else None


def annotate_stripped_forms(top_k: list[TopKEntry]) -> None:
    """Fill in the zero-stripped form, in place. Numeric columns only.

    The caller classifies; this is the per-entry half. Only numeric columns get
    it because stripping zeros from a code like ``007`` would invent a match
    on ``7``.
    """
    for e in top_k:
        # Integers only. `0.5` would strip to `.5` and `-0010` to `-0010`, and
        # neither can ever match: lookups come in normalized, and normalization
        # keeps only alphanumerics. Restricting to digits keeps the rule exactly
        # what it says — leading zeros on a whole number.
        if not e.value.isdigit():
            continue
        stripped = e.value.lstrip("0")
        # keep at least one character: "0000" strips to "0", not ""
        e.norm_stripped = stripped if stripped else "0"
