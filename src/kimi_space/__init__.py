"""kimi-space: 轻量数据分析工具集。"""

from kimi_space.outliers import (
    iqr_mask,
    iqr_outliers,
    zscore_mask,
    zscore_outliers,
)
from kimi_space.stats import (
    SummaryStats,
    geometric_mean,
    harmonic_mean,
    moving_average,
    summarize,
    weighted_mean,
    zscores,
)
from kimi_space.transform import log_returns, minmax_normalize, pct_change

__version__ = "0.2.0"

__all__ = [
    "SummaryStats",
    "geometric_mean",
    "harmonic_mean",
    "iqr_mask",
    "iqr_outliers",
    "log_returns",
    "minmax_normalize",
    "moving_average",
    "pct_change",
    "summarize",
    "weighted_mean",
    "zscore_mask",
    "zscore_outliers",
    "zscores",
]
