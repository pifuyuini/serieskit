"""matplotlib 统一风格绘图辅助。

所有函数都返回 ``matplotlib.axes.Axes``，便于继续定制或保存。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from kimi_space.outliers import iqr_mask
from kimi_space.stats import moving_average, moving_std

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from matplotlib.axes import Axes


def apply_style() -> None:
    """应用统一的图表风格（简洁网格、隐藏上/右边框、CJK 字体回退）。"""
    plt.rcParams.update(
        {
            "figure.figsize": (8, 4.5),
            "figure.dpi": 100,
            "font.sans-serif": [
                "PingFang SC",
                "Hiragino Sans GB",
                "Noto Sans CJK SC",
                "Microsoft YaHei",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
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
    x: Iterable[float] | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    label: str | None = None,
    smooth_window: int | None = None,
) -> Axes:
    """绘制序列折线；x 可指定横轴，smooth_window 给定时叠加等权滑动平均曲线。"""
    ax = _get_ax(ax)
    arr = np.asarray(list(values), dtype=float)
    xs = np.arange(arr.size) if x is None else np.asarray(list(x))
    if xs.size != arr.size:
        msg = f"x 与 values 长度不一致: {xs.size} vs {arr.size}"
        raise ValueError(msg)
    ax.plot(xs, arr, label=label or "series")
    if smooth_window is not None:
        ma = moving_average(arr, smooth_window)
        offset = smooth_window - 1
        ax.plot(xs[offset:], ma, label=f"MA{smooth_window}")
    if title:
        ax.set_title(title)
    ax.legend()
    return ax


def plot_histogram(
    values: Iterable[float],
    *,
    bins: int = 20,
    kde: bool = False,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """绘制直方图并标注均值与中位数参考线；kde=True 时叠加核密度估计曲线。

    kde 模式下直方图切换为密度刻度，与 KDE 曲线同尺度。
    """
    ax = _get_ax(ax)
    arr = np.asarray(list(values), dtype=float)
    ax.hist(arr, bins=bins, alpha=0.75, density=kde)
    ax.axvline(np.mean(arr), color="C1", linestyle="--", label="mean")
    ax.axvline(np.median(arr), color="C2", linestyle=":", label="median")
    if kde:
        density = scipy_stats.gaussian_kde(arr)
        xs = np.linspace(np.min(arr), np.max(arr), 200)
        ax.plot(xs, density(xs), color="C0", label="kde")
    if title:
        ax.set_title(title)
    ax.legend()
    return ax


def plot_outliers(
    values: Iterable[float],
    *,
    k: float = 1.5,
    mask: Iterable[bool] | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """绘制序列折线并以红色散点标出异常值。

    默认用 IQR 栅栏法判定；mask 给定时直接使用（True 表示正常值）。
    """
    ax = _get_ax(ax)
    arr = np.asarray(list(values), dtype=float)
    ax.plot(arr, label="series")
    m = iqr_mask(arr, k=k) if mask is None else np.asarray(list(mask), dtype=bool)
    if m.size != arr.size:
        msg = f"mask 与 values 长度不一致: {m.size} vs {arr.size}"
        raise ValueError(msg)
    if not m.all():
        idx = np.flatnonzero(~m)
        ax.scatter(idx, arr[idx], color="C3", zorder=3, label=f"outliers ({idx.size})")
    if title:
        ax.set_title(title)
    ax.legend()
    return ax


def plot_bands(
    values: Iterable[float],
    *,
    window: int = 20,
    num_std: float = 2.0,
    x: Iterable[float] | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """绘制序列及其滑动均值 ± num_std 倍滑动标准差的波动带。"""
    ax = _get_ax(ax)
    arr = np.asarray(list(values), dtype=float)
    ma = moving_average(arr, window)
    sd = moving_std(arr, window)
    xs = np.arange(arr.size) if x is None else np.asarray(list(x))
    if xs.size != arr.size:
        msg = f"x 与 values 长度不一致: {xs.size} vs {arr.size}"
        raise ValueError(msg)
    offset = window - 1
    ax.plot(xs, arr, alpha=0.4, label="series")
    ax.plot(xs[offset:], ma, label=f"MA{window}")
    ax.fill_between(
        xs[offset:],
        ma - num_std * sd,
        ma + num_std * sd,
        alpha=0.2,
        label=f"±{num_std}σ",
    )
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
