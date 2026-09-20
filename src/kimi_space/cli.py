"""kimi-space 命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from typing import TYPE_CHECKING, cast

import pandas as pd
from matplotlib.figure import Figure

from kimi_space import __version__
from kimi_space.io import load_csv, numeric_summary
from kimi_space.outliers import iqr_outliers, zscore_outliers
from kimi_space.plotting import (
    apply_style,
    plot_box,
    plot_correlation_heatmap,
    plot_histogram,
    plot_series,
)
from kimi_space.stats import zscores
from kimi_space.transform import log_returns, minmax_normalize, pct_change

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes


def _save(ax: Axes, output: str) -> None:
    cast("Figure", ax.figure).savefig(output, bbox_inches="tight")
    print(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kimi-space",
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
    p_summary.set_defaults(func=_cmd_summary)

    p_hist = sub.add_parser("hist", help="为指定数值列绘制直方图并保存")
    p_hist.add_argument("csv", help="CSV 文件路径")
    p_hist.add_argument("column", help="数值列名")
    p_hist.add_argument("--bins", type=int, default=20, help="直方图分箱数（默认 20）")
    p_hist.add_argument("--output", help="输出图片路径（默认 <列名>_hist.png）")
    p_hist.set_defaults(func=_cmd_hist)

    p_series = sub.add_parser("series", help="绘制数值列折线图（可选滑动平均）并保存")
    p_series.add_argument("csv", help="CSV 文件路径")
    p_series.add_argument("column", help="数值列名")
    p_series.add_argument("--smooth", type=int, help="滑动平均窗口，如 10")
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

    p_out = sub.add_parser("outliers", help="检测指定数值列的异常值")
    p_out.add_argument("csv", help="CSV 文件路径")
    p_out.add_argument("column", help="数值列名")
    p_out.add_argument(
        "--method",
        choices=["iqr", "zscore"],
        default="iqr",
        help="检测方法（默认 iqr）",
    )
    p_out.add_argument("--k", type=float, default=1.5, help="IQR 栅栏系数（默认 1.5）")
    p_out.add_argument(
        "--threshold", type=float, default=3.0, help="z-score 阈值（默认 3.0）"
    )
    p_out.set_defaults(func=_cmd_outliers)

    p_trans = sub.add_parser("transform", help="对指定数值列做变换并逐行输出")
    p_trans.add_argument("csv", help="CSV 文件路径")
    p_trans.add_argument("column", help="数值列名")
    p_trans.add_argument(
        "--method",
        choices=["minmax", "pct_change", "log_returns", "zscore"],
        default="zscore",
        help="变换方法（默认 zscore）",
    )
    p_trans.set_defaults(func=_cmd_transform)

    return parser


def _cmd_summary(args: argparse.Namespace) -> int:
    df = load_csv(args.csv)
    summary = numeric_summary(df)
    if args.output:
        summary.to_csv(args.output)
        print(args.output)
        return 0
    if args.json:
        payload = {col: row.to_dict() for col, row in summary.iterrows()}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
            print(summary.to_string())
    return 0


def _cmd_hist(args: argparse.Namespace) -> int:
    df = load_csv(args.csv, numeric_columns=[args.column])
    apply_style()
    output = args.output or f"{args.column}_hist.png"
    ax = plot_histogram(df[args.column], bins=args.bins, title=args.column)
    _save(ax, output)
    return 0


def _cmd_series(args: argparse.Namespace) -> int:
    df = load_csv(args.csv, numeric_columns=[args.column])
    apply_style()
    output = args.output or f"{args.column}_series.png"
    ax = plot_series(
        df[args.column],
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


def _cmd_outliers(args: argparse.Namespace) -> int:
    df = load_csv(args.csv, numeric_columns=[args.column])
    if args.method == "iqr":
        result = iqr_outliers(df[args.column], k=args.k)
    else:
        result = zscore_outliers(df[args.column], threshold=args.threshold)
    for value in result:
        print(value)
    print(f"# {len(result)} 个异常值", file=sys.stderr)
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
    else:
        result = zscores(series)
    for value in result:
        print(value)
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
