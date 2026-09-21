"""CSV 数据装载与轻量校验。"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pandas as pd

from kimi_space.stats import summarize

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


def load_csv(
    path: str | Path,
    *,
    required_columns: Iterable[str] | None = None,
    numeric_columns: Iterable[str] | None = None,
    parse_dates: Iterable[str] | None = None,
) -> pd.DataFrame:
    """读取 CSV 并可校验必需列、将指定列强制转为数值类型。

    path 为 "-" 时从标准输入读取。numeric_columns 中无法解析为数值的单元格
    会抛出 ValueError。parse_dates 指定按日期解析的列。
    """
    df = pd.read_csv(
        sys.stdin if str(path) == "-" else path,
        parse_dates=list(parse_dates) if parse_dates else None,
    )
    if required_columns is not None:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            msg = f"CSV 缺少必需列: {missing}"
            raise ValueError(msg)
    if numeric_columns is not None:
        for col in numeric_columns:
            if col not in df.columns:
                msg = f"CSV 缺少数值列: {col}"
                raise ValueError(msg)
            df[col] = pd.to_numeric(df[col], errors="raise")
    return df


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """对 DataFrame 的全部数值列逐列计算描述性统计。"""
    numeric = df.select_dtypes(include="number")
    if len(numeric.columns) == 0:
        msg = "DataFrame 不含数值列"
        raise ValueError(msg)
    rows = {col: summarize(numeric[col]).to_dict() for col in numeric.columns}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("column")


def numeric_summary_by(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """按分类列分组后逐组计算数值列描述性统计（MultiIndex: 组, 列）。

    分组列本身不参与统计。
    """
    if by not in df.columns:
        msg = f"分组列不存在: {by}"
        raise ValueError(msg)
    parts = {
        str(name): numeric_summary(g.drop(columns=[by], errors="ignore"))
        for name, g in df.groupby(by)
    }
    return pd.concat(parts, names=[by, "column"])
