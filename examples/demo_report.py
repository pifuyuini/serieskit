"""生成一份演示报告：合成数据 -> 统计摘要 -> 图表 PNG。

用法：.venv/bin/python examples/demo_report.py [输出目录]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kimi_space.io import numeric_summary
from kimi_space.plotting import (
    apply_style,
    plot_correlation_heatmap,
    plot_histogram,
    plot_series,
)


def make_dataset(n: int = 200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = np.arange(n)
    trend = 100 + 0.1 * days
    noise = rng.normal(0, 2, n)
    price = trend + noise
    volume = 1000 + 200 * np.sin(days / 10) + rng.normal(0, 50, n)
    return pd.DataFrame({"day": days, "price": price, "volume": volume})


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("examples/output")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = make_dataset()
    csv_path = out_dir / "demo.csv"
    df.to_csv(csv_path, index=False)

    summary = numeric_summary(df)
    print(summary.to_string(float_format=lambda v: f"{v:.3f}"))

    apply_style()
    plot_series(df["price"], title="price with MA10", smooth_window=10).figure.savefig(
        out_dir / "series.png", bbox_inches="tight"
    )
    plot_histogram(df["price"], title="price distribution").figure.savefig(
        out_dir / "hist.png", bbox_inches="tight"
    )
    plot_correlation_heatmap(df).figure.savefig(
        out_dir / "heatmap.png", bbox_inches="tight"
    )
    print(f"报告产物已写入 {out_dir}/")


if __name__ == "__main__":
    main()
