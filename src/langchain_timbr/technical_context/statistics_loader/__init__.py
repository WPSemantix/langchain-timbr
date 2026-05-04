"""Statistics Loader — loads per-column statistics from timbr.sys_properties_statistics.

Public API:
    load_column_statistics(schema, table_name, columns, conn_params, config=None)
        -> dict[str, ColumnStatistics]

    StatisticsLoaderConfig — configuration dataclass
"""

from .config import StatisticsLoaderConfig
from .types import ColumnStatistics, TopKEntry
from .loader import load_column_statistics
from .stats_cache import StatsCache

__all__ = [
    "load_column_statistics",
    "StatisticsLoaderConfig",
    "ColumnStatistics",
    "TopKEntry",
    "StatsCache",
]
