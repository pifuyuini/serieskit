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

    Args:
        path: CSV 文件路径；"-" 表示从标准输入读取。
        required_columns: 必须存在的列；缺失时抛 ValueError。
        numeric_columns: 强制转为数值类型的列；列缺失或含无法解析的
            单元格时抛 ValueError。
        parse_dates: 按日期解析的列（透传 pandas）。

    Returns:
        装载完成的 DataFrame。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 列校验或数值强转失败。
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
    """对 DataFrame 的全部数值列逐列计算描述性统计。

    Args:
        df: 数据表。

    Returns:
        行索引为列名、列为统计量（count/mean/std/…/mad）的 DataFrame。

    Raises:
        ValueError: df 不含数值列。
    """
    numeric = df.select_dtypes(include="number")
    if len(numeric.columns) == 0:
        msg = "DataFrame 不含数值列"
        raise ValueError(msg)
    rows = {col: summarize(numeric[col]).to_dict() for col in numeric.columns}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("column")


def numeric_summary_by(df: pd.DataFrame, by: str | Iterable[str]) -> pd.DataFrame:
    """按一列或多列分组后逐组计算数值列描述性统计。

    Args:
        df: 数据表。
        by: 分组列名或列名序列；分组列本身不参与统计。

    Returns:
        MultiIndex（分组列… × column）的统计表。

    Raises:
        ValueError: 分组列不存在，或某组不含数值列。
    """
    keys = [by] if isinstance(by, str) else list(by)
    missing = [k for k in keys if k not in df.columns]
    if missing:
        msg = f"分组列不存在: {missing}"
        raise ValueError(msg)
    parts = {}
    for name, g in df.groupby(keys):
        label = name if isinstance(name, tuple) else (name,)
        parts[tuple(str(v) for v in label)] = numeric_summary(
            g.drop(columns=keys, errors="ignore")
        )
    return pd.concat(parts, names=[*keys, "column"])
