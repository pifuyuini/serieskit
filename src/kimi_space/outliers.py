"""异常值检测：IQR 栅栏法与 z-score 法。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from kimi_space.stats import zscores

if TYPE_CHECKING:
    from collections.abc import Iterable


def iqr_mask(values: Iterable[float], *, k: float = 1.5) -> npt.NDArray[np.bool_]:
    """IQR 栅栏法：返回布尔掩码，True 表示落在 [Q1-k*IQR, Q3+k*IQR] 内的正常值。"""
    arr = np.asarray(list(values), dtype=float).ravel()
    if arr.size == 0:
        msg = "values 为空"
        raise ValueError(msg)
    q1, q3 = (float(v) for v in np.percentile(arr, [25, 75]))
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return (arr >= lo) & (arr <= hi)


def zscore_mask(
    values: Iterable[float], *, threshold: float = 3.0
) -> npt.NDArray[np.bool_]:
    """z-score 法：返回布尔掩码，True 表示 |z| <= threshold 的正常值。"""
    arr = np.asarray(list(values), dtype=float).ravel()
    if arr.size == 0:
        msg = "values 为空"
        raise ValueError(msg)
    return np.abs(zscores(arr)) <= threshold


def iqr_outliers(values: Iterable[float], *, k: float = 1.5) -> npt.NDArray[np.float64]:
    """返回 IQR 栅栏法判定为异常的值。

    >>> iqr_outliers([1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 100.0])
    array([100.])
    """
    arr = np.asarray(list(values), dtype=float).ravel()
    return arr[~iqr_mask(arr, k=k)]


def zscore_outliers(
    values: Iterable[float], *, threshold: float = 3.0
) -> npt.NDArray[np.float64]:
    """返回 z-score 法判定为异常的值。"""
    arr = np.asarray(list(values), dtype=float).ravel()
    return arr[~zscore_mask(arr, threshold=threshold)]
