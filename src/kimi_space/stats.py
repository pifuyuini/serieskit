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
    """计算描述性统计摘要；NaN 会被剔除，空输入抛出 ValueError。

    >>> s = summarize([1, 2, 3, 4, 5])
    >>> s.mean, s.median, s.count
    (3.0, 3.0, 5)
    >>> summarize([1.0, float("nan"), 3.0]).count
    2
    """
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
    """等权滑动平均，返回长度为 len(values) - window + 1 的数组。

    >>> moving_average([1, 2, 3, 4], 2)
    array([1.5, 2.5, 3.5])
    """
    if window < 1:
        msg = f"window 必须 >= 1，收到 {window}"
        raise ValueError(msg)
    arr = np.asarray(list(values), dtype=float).ravel()
    if window > arr.size:
        msg = f"window ({window}) 超过数据长度 ({arr.size})"
        raise ValueError(msg)
    kernel = np.full(window, 1.0 / window)
    return np.convolve(arr, kernel, mode="valid")


def moving_std(values: Iterable[float], window: int) -> np.ndarray:
    """滑动窗口样本标准差（ddof=1），返回长度为 len(values) - window + 1 的数组。

    window 必须 >= 2 且不超过数据长度。

    >>> moving_std([1, 2, 3, 4], 2)
    array([0.70710678, 0.70710678, 0.70710678])
    """
    if window < 2:
        msg = f"window 必须 >= 2，收到 {window}"
        raise ValueError(msg)
    arr = np.asarray(list(values), dtype=float).ravel()
    if window > arr.size:
        msg = f"window ({window}) 超过数据长度 ({arr.size})"
        raise ValueError(msg)
    windows = np.lib.stride_tricks.sliding_window_view(arr, window)
    return windows.std(axis=-1, ddof=1)


def zscores(values: Iterable[float]) -> np.ndarray:
    """基于样本标准差的标准化；常数序列返回全零。

    >>> zscores([7, 7, 7])
    array([0., 0., 0.])
    """
    arr = np.asarray(list(values), dtype=float).ravel()
    if arr.size == 0:
        msg = "values 为空"
        raise ValueError(msg)
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    if std == 0.0:
        return np.zeros_like(arr)
    return (arr - float(np.mean(arr))) / std


def weighted_mean(values: Iterable[float], weights: Iterable[float]) -> float:
    """加权平均；长度不一致、权重为负或权重和为 0 时抛出 ValueError。"""
    v = np.asarray(list(values), dtype=float).ravel()
    w = np.asarray(list(weights), dtype=float).ravel()
    if v.size == 0:
        msg = "values 为空"
        raise ValueError(msg)
    if v.size != w.size:
        msg = f"长度不一致: values {v.size} vs weights {w.size}"
        raise ValueError(msg)
    if (w < 0).any():
        msg = "权重不能为负"
        raise ValueError(msg)
    total = float(np.sum(w))
    if total == 0.0:
        msg = "权重和为 0"
        raise ValueError(msg)
    return float(np.dot(v, w) / total)


def geometric_mean(values: Iterable[float]) -> float:
    """几何平均；空输入或非正值抛出 ValueError。"""
    arr = _as_clean_array(values)
    if arr.size == 0:
        msg = "values 为空"
        raise ValueError(msg)
    if (arr <= 0).any():
        msg = "几何平均要求全部为正数"
        raise ValueError(msg)
    return float(scipy_stats.gmean(arr))


def harmonic_mean(values: Iterable[float]) -> float:
    """调和平均；空输入或非正值抛出 ValueError。"""
    arr = _as_clean_array(values)
    if arr.size == 0:
        msg = "values 为空"
        raise ValueError(msg)
    if (arr <= 0).any():
        msg = "调和平均要求全部为正数"
        raise ValueError(msg)
    return float(scipy_stats.hmean(arr))


def autocorr(values: Iterable[float], lag: int = 1) -> float:
    """lag-k 自相关系数：x[:-k] 与 x[k:] 的 Pearson 相关。

    lag 必须满足 1 <= lag < len(values)；切片方差为 0 时返回 NaN。

    >>> round(autocorr([1, 2, 3, 4, 5]), 6)
    1.0
    """
    arr = np.asarray(list(values), dtype=float).ravel()
    if not 1 <= lag < arr.size:
        msg = f"lag 需满足 1 <= lag < {arr.size}，收到 {lag}"
        raise ValueError(msg)
    a, b = arr[:-lag], arr[lag:]
    sa, sb = float(np.std(a)), float(np.std(b))
    if sa == 0.0 or sb == 0.0:
        return float("nan")
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))
