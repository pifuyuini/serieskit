"""kimi-space: 轻量数据分析工具集。"""

from kimi_space.outliers import (
    iqr_mask,
    iqr_outliers,
    zscore_mask,
    zscore_outliers,
)
from kimi_space.stats import (
    SummaryStats,
    autocorr,
    geometric_mean,
    harmonic_mean,
    median_abs_deviation,
    moving_average,
    moving_std,
    summarize,
    weighted_mean,
    zscores,
)
from kimi_space.transform import log_returns, minmax_normalize, pct_change

__version__ = "0.7.0"

__all__ = [
    "SummaryStats",
    "autocorr",
    "geometric_mean",
    "harmonic_mean",
    "iqr_mask",
    "iqr_outliers",
    "log_returns",
    "median_abs_deviation",
    "minmax_normalize",
    "moving_average",
    "moving_std",
    "pct_change",
    "summarize",
    "weighted_mean",
    "zscore_mask",
    "zscore_outliers",
    "zscores",
]
