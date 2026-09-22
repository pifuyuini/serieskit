"""serieskit 命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from serieskit import __version__
from serieskit.io import load_csv, numeric_summary, numeric_summary_by
from serieskit.outliers import iqr_mask, mad_mask, zscore_mask
from serieskit.plotting import (
    apply_style,
    plot_bands,
    plot_box,
    plot_correlation_heatmap,
    plot_histogram,
    plot_multi_series,
    plot_outliers,
    plot_robust_bands,
    plot_series,
)
from serieskit.stats import (
    autocorr,
    median_abs_deviation,
    moving_average,
    moving_median,
    moving_std,
    zscores,
)
from serieskit.transform import (
    diff,
    log_returns,
    minmax_normalize,
    pct_change,
    robust_zscores,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes


def _save(ax: Axes, output: str) -> None:
    cast("Figure", ax.figure).savefig(output, bbox_inches="tight")
    plt.close(cast("Figure", ax.figure))
    print(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="serieskit",
        description="轻量数据分析工具集：CSV 摘要与快速可视化",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_summary = sub.add_parser("summary", help="输出 CSV 数值列的描述性统计")
    p_summary.add_argument("csv", help="CSV 文件路径")
    p_summary.add_argument("--json", action="store_true", help="以 JSON 格式输出")
    p_summary.add_argument("--output", help="将统计表保存为 CSV 文件")
    p_summary.add_argument("--by", help="按指定列分组后逐组统计（多列用逗号分隔）")
    p_summary.set_defaults(func=_cmd_summary)

    p_hist = sub.add_parser("hist", help="为指定数值列绘制直方图并保存")
    p_hist.add_argument("csv", help="CSV 文件路径")
    p_hist.add_argument("column", help="数值列名")
    p_hist.add_argument("--bins", type=int, default=20, help="直方图分箱数（默认 20）")
    p_hist.add_argument("--kde", action="store_true", help="叠加核密度估计曲线")
    p_hist.add_argument("--output", help="输出图片路径（默认 <列名>_hist.png）")
    p_hist.set_defaults(func=_cmd_hist)

    p_series = sub.add_parser("series", help="绘制数值列折线图并保存")
    p_series.add_argument("csv", help="CSV 文件路径")
    p_series.add_argument("column", help="数值列名（多列用逗号分隔）")
    p_series.add_argument("--x", help="作为横轴的列名（默认行号）")
    g_series = p_series.add_mutually_exclusive_group()
    g_series.add_argument("--smooth", type=int, help="滑动平均窗口，如 10")
    g_series.add_argument("--bands", type=int, help="波动带窗口（MA±2σ），如 20")
    g_series.add_argument(
        "--robust-bands", type=int, help="稳健波动带窗口（中位数±3.5·MAD）"
    )
    p_series.add_argument("--output", help="输出图片路径（默认 <列名>_series.png）")
    p_series.set_defaults(func=_cmd_series)

    p_heat = sub.add_parser("heatmap", help="绘制数值列相关系数热力图并保存")
    p_heat.add_argument("csv", help="CSV 文件路径")
    p_heat.add_argument("--columns", help="逗号分隔的列名子集")
    p_heat.add_argument("--output", default="heatmap.png", help="输出图片路径")
    p_heat.set_defaults(func=_cmd_heatmap)

    p_box = sub.add_parser("box", help="绘制数值列箱线图并保存")
    p_box.add_argument("csv", help="CSV 文件路径")
    p_box.add_argument("--columns", help="逗号分隔的列名子集")
    p_box.add_argument("--output", default="box.png", help="输出图片路径")
    p_box.set_defaults(func=_cmd_box)

    p_corr = sub.add_parser("corr", help="输出数值列相关系数矩阵（文本表格）")
    p_corr.add_argument("csv", help="CSV 文件路径")
    p_corr.add_argument("--columns", help="逗号分隔的列名子集")
    p_corr.add_argument(
        "--method",
        choices=["pearson", "spearman", "kendall"],
        default="pearson",
        help="相关系数方法（默认 pearson）",
    )
    p_corr.add_argument("--output", help="将相关矩阵保存为 CSV 文件")
    p_corr.set_defaults(func=_cmd_corr)

    p_out = sub.add_parser("outliers", help="检测指定数值列的异常值")
    p_out.add_argument("csv", help="CSV 文件路径")
    p_out.add_argument("column", help="数值列名")
    p_out.add_argument(
        "--method",
        choices=["iqr", "zscore", "mad"],
        default="iqr",
        help="检测方法（默认 iqr）",
    )
    p_out.add_argument("--k", type=float, default=1.5, help="IQR 栅栏系数（默认 1.5）")
    p_out.add_argument(
        "--threshold",
        type=float,
        help="z-score/mad 阈值（默认 zscore=3.0，mad=3.5）",
    )
    p_out.add_argument("--plot", help="同时保存异常值标注图到指定路径")
    p_out.set_defaults(func=_cmd_outliers)

    p_trans = sub.add_parser("transform", help="对指定数值列做变换并逐行输出")
    p_trans.add_argument("csv", help="CSV 文件路径")
    p_trans.add_argument("column", help="数值列名")
    p_trans.add_argument(
        "--method",
        choices=[
            "minmax",
            "pct_change",
            "log_returns",
            "zscore",
            "robust_zscore",
            "diff",
            "moving_average",
            "moving_std",
            "moving_median",
        ],
        default="zscore",
        help="变换方法（默认 zscore）",
    )
    p_trans.add_argument(
        "--window", type=int, default=10, help="滑动窗口（moving_* 方法，默认 10）"
    )
    p_trans.set_defaults(func=_cmd_transform)

    p_ac = sub.add_parser("autocorr", help="输出指定数值列的 lag-k 自相关系数")
    p_ac.add_argument("csv", help="CSV 文件路径")
    p_ac.add_argument("column", help="数值列名")
    p_ac.add_argument("--lag", type=int, default=1, help="滞后阶数（默认 1）")
    p_ac.set_defaults(func=_cmd_autocorr)

    p_pct = sub.add_parser("percentile", help="输出指定数值列的分位数")
    p_pct.add_argument("csv", help="CSV 文件路径")
    p_pct.add_argument("column", help="数值列名")
    p_pct.add_argument("--q", type=float, default=50.0, help="分位点 0-100（默认 50）")
    p_pct.set_defaults(func=_cmd_percentile)

    p_mad = sub.add_parser("mad", help="输出指定数值列的中位数绝对偏差（MAD）")
    p_mad.add_argument("csv", help="CSV 文件路径")
    p_mad.add_argument("column", help="数值列名")
    p_mad.set_defaults(func=_cmd_mad)

    p_rep = sub.add_parser("report", help="一键生成报告目录（统计表 + 全部图表）")
    p_rep.add_argument("csv", help="CSV 文件路径")
    p_rep.add_argument("--outdir", default="report", help="输出目录（默认 report/）")
    p_rep.add_argument("--by", help="按指定列分组统计（多列用逗号分隔）")
    p_rep.set_defaults(func=_cmd_report)

    return parser


def _cmd_summary(args: argparse.Namespace) -> int:
    df = load_csv(args.csv)
    summary = (
        numeric_summary_by(df, args.by.split(",")) if args.by else numeric_summary(df)
    )
    if args.output:
        summary.to_csv(args.output)
        print(args.output)
        return 0
    if args.json:

        def key_of(idx: object) -> str:
            return " | ".join(map(str, idx)) if isinstance(idx, tuple) else str(idx)

        payload = {
            key_of(idx): {
                key: value if np.isfinite(value) else None
                for key, value in row.to_dict().items()
            }
            for idx, row in summary.iterrows()
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
            print(summary.to_string())
    return 0


def _cmd_hist(args: argparse.Namespace) -> int:
    df = load_csv(args.csv, numeric_columns=[args.column])
    apply_style()
    output = args.output or f"{args.column}_hist.png"
    ax = plot_histogram(
        df[args.column], bins=args.bins, kde=args.kde, title=args.column
    )
    _save(ax, output)
    return 0


def _cmd_series(args: argparse.Namespace) -> int:
    columns = args.column.split(",")
    df = load_csv(args.csv, numeric_columns=columns)
    apply_style()
    output = args.output or f"{args.column.replace(',', '_')}_series.png"
    x = df[args.x] if args.x else None
    if len(columns) > 1:
        if any(
            value is not None for value in (args.smooth, args.bands, args.robust_bands)
        ):
            msg = "--smooth/--bands/--robust-bands 仅支持单列"
            raise ValueError(msg)
        ax = plot_multi_series(
            {col: df[col] for col in columns}, x=x, title=args.column
        )
    elif args.robust_bands is not None:
        ax = plot_robust_bands(
            df[args.column], window=args.robust_bands, x=x, title=args.column
        )
    elif args.bands is not None:
        ax = plot_bands(df[args.column], window=args.bands, x=x, title=args.column)
    else:
        ax = plot_series(
            df[args.column],
            x=x,
            smooth_window=args.smooth,
            title=args.column,
        )
    _save(ax, output)
    return 0


def _cmd_heatmap(args: argparse.Namespace) -> int:
    df = load_csv(args.csv)
    columns = args.columns.split(",") if args.columns else None
    apply_style()
    ax = plot_correlation_heatmap(df, columns=columns)
    _save(ax, args.output)
    return 0


def _cmd_box(args: argparse.Namespace) -> int:
    df = load_csv(args.csv)
    columns = args.columns.split(",") if args.columns else None
    apply_style()
    ax = plot_box(df, columns=columns)
    _save(ax, args.output)
    return 0


def _cmd_corr(args: argparse.Namespace) -> int:
    df = load_csv(args.csv)
    data = (
        df[args.columns.split(",")]
        if args.columns
        else df.select_dtypes(include="number")
    )
    corr = data.corr(method=args.method)
    if args.output:
        corr.to_csv(args.output)
        print(args.output)
        return 0
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(corr.to_string())
    return 0


def _cmd_outliers(args: argparse.Namespace) -> int:
    df = load_csv(args.csv, numeric_columns=[args.column])
    if args.method == "iqr":
        mask = iqr_mask(df[args.column], k=args.k)
    elif args.method == "mad":
        mask = mad_mask(
            df[args.column], threshold=3.5 if args.threshold is None else args.threshold
        )
    else:
        mask = zscore_mask(
            df[args.column], threshold=3.0 if args.threshold is None else args.threshold
        )
    values = df[args.column].to_numpy(dtype=float)
    result = values[~mask]
    for value in result:
        print(value)
    print(f"# {len(result)} 个异常值", file=sys.stderr)
    if args.plot:
        apply_style()
        ax = plot_outliers(values, mask=mask, title=args.column)
        cast("Figure", ax.figure).savefig(args.plot, bbox_inches="tight")
        plt.close(cast("Figure", ax.figure))
    return 0


def _cmd_transform(args: argparse.Namespace) -> int:
    df = load_csv(args.csv, numeric_columns=[args.column])
    series = df[args.column]
    if args.method == "minmax":
        result = minmax_normalize(series)
    elif args.method == "pct_change":
        result = pct_change(series)
    elif args.method == "log_returns":
        result = log_returns(series)
    elif args.method == "robust_zscore":
        result = robust_zscores(series)
    elif args.method == "diff":
        result = diff(series)
    elif args.method == "moving_average":
        result = moving_average(series, args.window)
    elif args.method == "moving_std":
        result = moving_std(series, args.window)
    elif args.method == "moving_median":
        result = moving_median(series, args.window)
    else:
        result = zscores(series)
    for value in result:
        print(value)
    return 0


def _cmd_autocorr(args: argparse.Namespace) -> int:
    df = load_csv(args.csv, numeric_columns=[args.column])
    print(autocorr(df[args.column], lag=args.lag))
    return 0


def _cmd_percentile(args: argparse.Namespace) -> int:
    df = load_csv(args.csv, numeric_columns=[args.column])
    if not 0.0 <= args.q <= 100.0:
        msg = f"q 必须在 [0, 100]，收到 {args.q}"
        raise ValueError(msg)
    print(float(np.percentile(df[args.column].to_numpy(dtype=float), args.q)))
    return 0


def _cmd_mad(args: argparse.Namespace) -> int:
    df = load_csv(args.csv, numeric_columns=[args.column])
    print(median_abs_deviation(df[args.column]))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = load_csv(args.csv)
    if args.by:
        summary = numeric_summary_by(df, args.by.split(","))
        summary.to_csv(outdir / "summary_by.csv")
    else:
        numeric_summary(df).to_csv(outdir / "summary.csv")
    numeric = df.select_dtypes(include="number")
    apply_style()
    for col in numeric.columns:
        _save(
            plot_histogram(numeric[col], title=str(col)),
            str(outdir / f"{col}_hist.png"),
        )
    if len(numeric.columns) > 1:
        _save(plot_correlation_heatmap(df), str(outdir / "heatmap.png"))
        _save(plot_box(df), str(outdir / "box.png"))
    print(f"报告已生成: {outdir}/")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
