"""常用序列变换：归一化、变化率、对数收益。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable


def minmax_normalize(
    values: Iterable[float],
    *,
    feature_range: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """最小-最大归一化；常数序列映射到区间中点。

    >>> minmax_normalize([10, 20, 30])
    array([0. , 0.5, 1. ])
    """
    lo, hi = feature_range
    if not lo < hi:
        msg = f"feature_range 下界必须小于上界，收到 {feature_range}"
        raise ValueError(msg)
    arr = np.asarray(list(values), dtype=float).ravel()
    if arr.size == 0:
        msg = "values 为空"
        raise ValueError(msg)
    span = float(np.max(arr) - np.min(arr))
    if span == 0.0:
        return np.full_like(arr, (lo + hi) / 2)
    return (arr - float(np.min(arr))) / span * (hi - lo) + lo


def pct_change(values: Iterable[float]) -> np.ndarray:
    """相邻变化率，长度与输入一致；首元素与前值为 0 的位置为 NaN。"""
    arr = np.asarray(list(values), dtype=float).ravel()
    if arr.size == 0:
        msg = "values 为空"
        raise ValueError(msg)
    out = np.full_like(arr, np.nan)
    np.divide(np.diff(arr), arr[:-1], out=out[1:], where=arr[:-1] != 0)
    return out


def log_returns(values: Iterable[float]) -> np.ndarray:
    """对数收益 log(v_t / v_{t-1})，首元素 NaN；比值非正的位置为 NaN。"""
    arr = np.asarray(list(values), dtype=float).ravel()
    if arr.size == 0:
        msg = "values 为空"
        raise ValueError(msg)
    out = np.full_like(arr, np.nan)
    ratio = np.divide(
        arr[1:], arr[:-1], out=np.full(arr.size - 1, np.nan), where=arr[:-1] != 0
    )
    np.log(ratio, out=out[1:], where=ratio > 0)
    return out
