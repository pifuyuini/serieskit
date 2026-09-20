"""kimi-space: 轻量数据分析工具集。"""

from kimi_space.outliers import (
    iqr_mask,
    iqr_outliers,
    zscore_mask,
    zscore_outliers,
)
from kimi_space.stats import SummaryStats, moving_average, summarize, zscores
from kimi_space.transform import log_returns, minmax_normalize, pct_change

__version__ = "0.1.0"

__all__ = [
    "SummaryStats",
    "iqr_mask",
    "iqr_outliers",
    "log_returns",
    "minmax_normalize",
    "moving_average",
    "pct_change",
    "summarize",
    "zscore_mask",
    "zscore_outliers",
    "zscores",
]
