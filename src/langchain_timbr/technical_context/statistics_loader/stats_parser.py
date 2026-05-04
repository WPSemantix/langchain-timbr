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

from .types import TopKEntry

logger = logging.getLogger(__name__)

# SQL types mapped to parse functions
_INTEGER_TYPES = frozenset({"int", "bigint", "smallint", "tinyint", "integer"})
_DECIMAL_TYPES = frozenset({"decimal", "numeric", "float", "double", "real"})
_DATE_TYPES = frozenset({"date"})
_DATETIME_TYPES = frozenset({"timestamp", "datetime"})


def parse_stats_json(
    stats_str: str | None,
    sql_type: str | None = None,
) -> tuple[list[TopKEntry] | None, Any | None, Any | None]:
    """Parse the stats JSON column into structured data.

    Args:
        stats_str: Raw JSON string from the `stats` column. May be None.
        sql_type: The column's SQL type (e.g. "decimal(18,2)", "bigint", "date").
                  Used for type-aware min/max parsing.

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
                        top_k.append(TopKEntry(
                            value=str(entry["value"]),
                            count=int(entry["count"]),
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
