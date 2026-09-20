"""kimi-space 命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from typing import TYPE_CHECKING

import pandas as pd

from kimi_space.io import load_csv, numeric_summary
from kimi_space.plotting import (
    apply_style,
    plot_correlation_heatmap,
    plot_histogram,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kimi-space",
        description="轻量数据分析工具集：CSV 摘要与快速可视化",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_summary = sub.add_parser("summary", help="输出 CSV 数值列的描述性统计")
    p_summary.add_argument("csv", help="CSV 文件路径")
    p_summary.add_argument("--json", action="store_true", help="以 JSON 格式输出")
    p_summary.set_defaults(func=_cmd_summary)

    p_hist = sub.add_parser("hist", help="为指定数值列绘制直方图并保存")
    p_hist.add_argument("csv", help="CSV 文件路径")
    p_hist.add_argument("column", help="数值列名")
    p_hist.add_argument("--bins", type=int, default=20, help="直方图分箱数（默认 20）")
    p_hist.add_argument("--output", help="输出图片路径（默认 <列名>_hist.png）")
    p_hist.set_defaults(func=_cmd_hist)

    p_heat = sub.add_parser("heatmap", help="绘制数值列相关系数热力图并保存")
    p_heat.add_argument("csv", help="CSV 文件路径")
    p_heat.add_argument("--columns", help="逗号分隔的列名子集")
    p_heat.add_argument("--output", default="heatmap.png", help="输出图片路径")
    p_heat.set_defaults(func=_cmd_heatmap)

    return parser


def _cmd_summary(args: argparse.Namespace) -> int:
    df = load_csv(args.csv)
    summary = numeric_summary(df)
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
    ax.figure.savefig(output, bbox_inches="tight")
    print(output)
    return 0


def _cmd_heatmap(args: argparse.Namespace) -> int:
    df = load_csv(args.csv)
    columns = args.columns.split(",") if args.columns else None
    apply_style()
    ax = plot_correlation_heatmap(df, columns=columns)
    ax.figure.savefig(args.output, bbox_inches="tight")
    print(args.output)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
