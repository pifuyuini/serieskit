"""matplotlib 统一风格绘图辅助。

所有函数都返回 ``matplotlib.axes.Axes``，便于继续定制或保存。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kimi_space.outliers import iqr_mask
from kimi_space.stats import moving_average

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from matplotlib.axes import Axes


def apply_style() -> None:
    """应用统一的图表风格（简洁网格、隐藏上/右边框）。"""
    plt.rcParams.update(
        {
            "figure.figsize": (8, 4.5),
            "figure.dpi": 100,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.frameon": False,
        }
    )


def _get_ax(ax: Axes | None) -> Axes:
    if ax is not None:
        return ax
    _, ax = plt.subplots()
    return ax


def plot_series(
    values: Iterable[float],
    *,
    ax: Axes | None = None,
    title: str | None = None,
    label: str | None = None,
    smooth_window: int | None = None,
) -> Axes:
    """绘制序列折线；smooth_window 给定时叠加等权滑动平均曲线。"""
    ax = _get_ax(ax)
    arr = np.asarray(list(values), dtype=float)
    ax.plot(arr, label=label or "series")
    if smooth_window is not None:
        ma = moving_average(arr, smooth_window)
        offset = smooth_window - 1
        ax.plot(np.arange(offset, arr.size), ma, label=f"MA{smooth_window}")
    if title:
        ax.set_title(title)
    ax.legend()
    return ax


def plot_histogram(
    values: Iterable[float],
    *,
    bins: int = 20,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """绘制直方图并标注均值与中位数参考线。"""
    ax = _get_ax(ax)
    arr = np.asarray(list(values), dtype=float)
    ax.hist(arr, bins=bins, alpha=0.75)
    ax.axvline(np.mean(arr), color="C1", linestyle="--", label="mean")
    ax.axvline(np.median(arr), color="C2", linestyle=":", label="median")
    if title:
        ax.set_title(title)
    ax.legend()
    return ax


def plot_outliers(
    values: Iterable[float],
    *,
    k: float = 1.5,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """绘制序列折线并以红色散点标出 IQR 栅栏法判定的异常值。"""
    ax = _get_ax(ax)
    arr = np.asarray(list(values), dtype=float)
    ax.plot(arr, label="series")
    mask = iqr_mask(arr, k=k)
    if not mask.all():
        idx = np.flatnonzero(~mask)
        ax.scatter(idx, arr[idx], color="C3", zorder=3, label=f"outliers ({idx.size})")
    if title:
        ax.set_title(title)
    ax.legend()
    return ax


def plot_box(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """按列绘制箱线图，便于比较多列数值的分布。"""
    data = (
        df[list(columns)] if columns is not None else df.select_dtypes(include="number")
    )
    ax = _get_ax(ax)
    ax.boxplot([data[c].dropna() for c in data.columns], tick_labels=list(data.columns))
    if title:
        ax.set_title(title)
    return ax


def plot_correlation_heatmap(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """绘制数值列 Pearson 相关系数热力图，并在格内标注数值。"""
    data = (
        df[list(columns)] if columns is not None else df.select_dtypes(include="number")
    )
    corr = data.corr()
    ax = _get_ax(ax)
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.figure.colorbar(image, ax=ax, fraction=0.046)
    ticks = np.arange(len(corr.columns))
    ax.set_xticks(ticks, corr.columns, rotation=45, ha="right")
    ax.set_yticks(ticks, corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(
                j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8
            )
    return ax
