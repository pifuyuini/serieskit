"""matplotlib 统一风格绘图辅助。

所有函数都返回 ``matplotlib.axes.Axes``，便于继续定制或保存。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from kimi_space.outliers import iqr_mask, mad_mask, zscore_mask
from kimi_space.stats import (
    median_abs_deviation,
    moving_average,
    moving_median,
    moving_percentile,
    moving_std,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

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
    """绘制序列折线。

    Args:
        values: 纵轴序列。
        x: 横轴取值；默认行号 0..n-1。长度必须与 values 一致。
        ax: 目标 Axes；默认新建。
        title: 标题。
        label: 图例标签，默认 "series"。
        smooth_window: 给定窗口时叠加等权滑动平均曲线（横轴对齐窗口右端）。

    Returns:
        绘制完成的 Axes。

    Raises:
        ValueError: x 与 values 长度不一致，或 smooth_window 非法。
    """
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
    """绘制直方图并标注均值与中位数参考线。

    Args:
        values: 数据序列。
        bins: 分箱数，默认 20。
        kde: True 时叠加核密度估计曲线，直方图同步切换为密度刻度。
        ax: 目标 Axes；默认新建。
        title: 标题。

    Returns:
        绘制完成的 Axes。
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
    method: str = "iqr",
    k: float = 1.5,
    threshold: float | None = None,
    mask: Iterable[bool] | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """绘制序列折线并以红色散点标出异常值。

    Args:
        values: 数据序列。
        method: 内置判定方法，"iqr"（默认）、"zscore" 或 "mad"；mask 给定时忽略。
        k: IQR 栅栏系数，默认 1.5（method="iqr" 时生效）。
        threshold: zscore/mad 阈值（默认 zscore=3.0，mad=3.5）。
        mask: 外部判定的正常值掩码（True 为正常）；给定时优先于内置判定。
        ax: 目标 Axes；默认新建。
        title: 标题。

    Returns:
        绘制完成的 Axes。

    Raises:
        ValueError: mask 与 values 长度不一致，或 method 未知。
    """
    ax = _get_ax(ax)
    arr = np.asarray(list(values), dtype=float)
    ax.plot(arr, label="series")
    if mask is not None:
        m = np.asarray(list(mask), dtype=bool)
    elif method == "iqr":
        m = iqr_mask(arr, k=k)
    elif method == "zscore":
        m = zscore_mask(arr, threshold=threshold or 3.0)
    elif method == "mad":
        m = mad_mask(arr, threshold=threshold or 3.5)
    else:
        msg = f"未知 method: {method!r}"
        raise ValueError(msg)
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


def plot_robust_bands(
    values: Iterable[float],
    *,
    window: int = 20,
    num_mad: float = 3.5,
    x: Iterable[float] | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """绘制序列及其滑动中位数 ± num_mad 倍全局 MAD 的稳健波动带。

    与 plot_bands（均值 ± kσ）对照：中心为滑动中位数，带宽为全局
    中位数绝对偏差的倍数，对离群点不敏感。

    Args:
        values: 数据序列。
        window: 中位数滑动窗口，默认 20（须 >= 1）。
        num_mad: 带宽的 MAD 倍数，默认 3.5。
        x: 横轴取值；默认行号。长度必须与 values 一致。
        ax: 目标 Axes；默认新建。
        title: 标题。

    Returns:
        绘制完成的 Axes。

    Raises:
        ValueError: x 长度不一致，或 window 非法。
    """
    ax = _get_ax(ax)
    arr = np.asarray(list(values), dtype=float)
    med = moving_median(arr, window)
    mad = median_abs_deviation(arr)
    xs = np.arange(arr.size) if x is None else np.asarray(list(x))
    if xs.size != arr.size:
        msg = f"x 与 values 长度不一致: {xs.size} vs {arr.size}"
        raise ValueError(msg)
    offset = window - 1
    ax.plot(xs, arr, alpha=0.4, label="series")
    ax.plot(xs[offset:], med, label=f"median{window}")
    ax.fill_between(
        xs[offset:],
        med - num_mad * mad,
        med + num_mad * mad,
        alpha=0.2,
        label=f"±{num_mad}·MAD",
    )
    if title:
        ax.set_title(title)
    ax.legend()
    return ax


def plot_quantile_bands(
    values: Iterable[float],
    *,
    window: int = 20,
    q_lo: float = 10.0,
    q_hi: float = 90.0,
    x: Iterable[float] | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """绘制序列及滑动分位数带（默认 P10–P90），中心线为滑动中位数。

    与 plot_bands / plot_robust_bands 对照：带由滑动分位数直接给出，
    不假设任何分布形态。

    Args:
        values: 数据序列。
        window: 滑动窗口，默认 20（须 >= 1）。
        q_lo: 下分位点，默认 10。
        q_hi: 上分位点，默认 90（须满足 q_lo < q_hi <= 100）。
        x: 横轴取值；默认行号。长度必须与 values 一致。
        ax: 目标 Axes；默认新建。
        title: 标题。

    Returns:
        绘制完成的 Axes。

    Raises:
        ValueError: 分位点或横轴非法。
    """
    if not 0.0 <= q_lo < q_hi <= 100.0:
        msg = f"须满足 0 <= q_lo < q_hi <= 100，收到 ({q_lo}, {q_hi})"
        raise ValueError(msg)
    ax = _get_ax(ax)
    arr = np.asarray(list(values), dtype=float)
    med = moving_median(arr, window)
    lo = moving_percentile(arr, window, q_lo)
    hi = moving_percentile(arr, window, q_hi)
    xs = np.arange(arr.size) if x is None else np.asarray(list(x))
    if xs.size != arr.size:
        msg = f"x 与 values 长度不一致: {xs.size} vs {arr.size}"
        raise ValueError(msg)
    offset = window - 1
    ax.plot(xs, arr, alpha=0.4, label="series")
    ax.plot(xs[offset:], med, label=f"median{window}")
    ax.fill_between(
        xs[offset:],
        lo,
        hi,
        alpha=0.2,
        label=f"P{q_lo:g}–P{q_hi:g}",
    )
    if title:
        ax.set_title(title)
    ax.legend()
    return ax


def plot_multi_series(
    series: Mapping[str, Iterable[float]],
    *,
    x: Iterable[float] | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """在同一坐标系绘制多条命名序列，便于对比。

    Args:
        series: {名称: 序列} 映射，绘制顺序即映射迭代顺序。
        x: 公共横轴；默认行号。长度必须与每条序列一致。
        ax: 目标 Axes；默认新建。
        title: 标题。

    Returns:
        绘制完成的 Axes。

    Raises:
        ValueError: series 为空，或任一序列与 x 长度不一致。
    """
    if not series:
        msg = "series 为空"
        raise ValueError(msg)
    ax = _get_ax(ax)
    arrays = {name: np.asarray(list(v), dtype=float) for name, v in series.items()}
    n = next(iter(arrays.values())).size
    xs = np.arange(n) if x is None else np.asarray(list(x))
    if xs.size != n:
        msg = f"x 与序列长度不一致: {xs.size} vs {n}"
        raise ValueError(msg)
    for name, arr in arrays.items():
        if arr.size != n:
            msg = f"序列长度不一致: {name} 为 {arr.size}，预期 {n}"
            raise ValueError(msg)
        ax.plot(xs, arr, label=name)
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
    """绘制序列及其滑动均值 ± num_std 倍滑动标准差的波动带。

    Args:
        values: 数据序列。
        window: 滑动窗口，默认 20（须 >= 2）。
        num_std: 波动带标准差倍数，默认 2.0。
        x: 横轴取值；默认行号。长度必须与 values 一致。
        ax: 目标 Axes；默认新建。
        title: 标题。

    Returns:
        绘制完成的 Axes。

    Raises:
        ValueError: x 长度不一致，或 window 非法。
    """
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
    """按列绘制箱线图，便于比较多列数值的分布。

    Args:
        df: 数据表；默认取全部数值列。
        columns: 列名子集；默认 None 表示全部数值列。
        ax: 目标 Axes；默认新建。
        title: 标题。

    Returns:
        绘制完成的 Axes。
    """
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
    """绘制数值列 Pearson 相关系数热力图，并在格内标注数值。

    Args:
        df: 数据表；默认取全部数值列。
        columns: 列名子集。
        ax: 目标 Axes；默认新建。

    Returns:
        绘制完成的 Axes（含 colorbar）。
    """
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
