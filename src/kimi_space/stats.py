"""描述性统计与滑动窗口计算。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats as scipy_stats

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True)
class SummaryStats:
    """一组数值的描述性统计摘要。

    std 为样本标准差（ddof=1）；kurtosis 为超额峰度（正态分布为 0）。
    """

    count: int
    mean: float
    std: float
    min: float
    q25: float
    median: float
    q75: float
    max: float
    skew: float
    kurtosis: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "q25": self.q25,
            "median": self.median,
            "q75": self.q75,
            "max": self.max,
            "skew": self.skew,
            "kurtosis": self.kurtosis,
        }


def _as_clean_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float).ravel()
    return arr[~np.isnan(arr)]


def summarize(values: Iterable[float]) -> SummaryStats:
    """计算描述性统计摘要；NaN 会被剔除，空输入抛出 ValueError。"""
    arr = _as_clean_array(values)
    if arr.size == 0:
        msg = "values 为空（或全部为 NaN）"
        raise ValueError(msg)
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return SummaryStats(
        count=int(arr.size),
        mean=float(np.mean(arr)),
        std=std,
        min=float(np.min(arr)),
        q25=float(np.percentile(arr, 25)),
        median=float(np.median(arr)),
        q75=float(np.percentile(arr, 75)),
        max=float(np.max(arr)),
        skew=float(scipy_stats.skew(arr)),
        kurtosis=float(scipy_stats.kurtosis(arr)),
    )


def moving_average(values: Iterable[float], window: int) -> np.ndarray:
    """等权滑动平均，返回长度为 len(values) - window + 1 的数组。"""
    if window < 1:
        msg = f"window 必须 >= 1，收到 {window}"
        raise ValueError(msg)
    arr = np.asarray(list(values), dtype=float).ravel()
    if window > arr.size:
        msg = f"window ({window}) 超过数据长度 ({arr.size})"
        raise ValueError(msg)
    kernel = np.full(window, 1.0 / window)
    return np.convolve(arr, kernel, mode="valid")


def zscores(values: Iterable[float]) -> np.ndarray:
    """基于样本标准差的标准化；常数序列返回全零。"""
    arr = np.asarray(list(values), dtype=float).ravel()
    if arr.size == 0:
        msg = "values 为空"
        raise ValueError(msg)
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    if std == 0.0:
        return np.zeros_like(arr)
    return (arr - float(np.mean(arr))) / std
