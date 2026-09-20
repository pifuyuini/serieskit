"""生成一份演示报告：合成数据 -> 统计摘要 -> 异常值 -> 变换 -> 图表 PNG。

用法：.venv/bin/python examples/demo_report.py [输出目录]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kimi_space.io import numeric_summary
from kimi_space.plotting import (
    apply_style,
    plot_box,
    plot_correlation_heatmap,
    plot_histogram,
    plot_outliers,
    plot_series,
)
from kimi_space.stats import geometric_mean, harmonic_mean
from kimi_space.transform import log_returns


def make_dataset(n: int = 200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = np.arange(n)
    trend = 100 + 0.1 * days
    noise = rng.normal(0, 2, n)
    price = trend + noise
    price[50] *= 1.3  # 人为注入一个异常值
    price[120] *= 0.7
    volume = 1000 + 200 * np.sin(days / 10) + rng.normal(0, 50, n)
    return pd.DataFrame({"day": days, "price": price, "volume": volume})


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("examples/output")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = make_dataset()
    df.to_csv(out_dir / "demo.csv", index=False)

    summary = numeric_summary(df)
    print(summary.to_string(float_format=lambda v: f"{v:.3f}"))
    print(
        f"volume 几何均值 {geometric_mean(df['volume']):.2f}"
        f"，调和均值 {harmonic_mean(df['volume']):.2f}"
    )

    df["log_ret"] = log_returns(df["price"])

    apply_style()
    plot_series(
        df["price"], x=df["day"], title="price with MA10", smooth_window=10
    ).figure.savefig(out_dir / "series.png", bbox_inches="tight")
    plot_histogram(df["price"], title="price distribution").figure.savefig(
        out_dir / "hist.png", bbox_inches="tight"
    )
    plot_correlation_heatmap(df[["price", "volume", "log_ret"]]).figure.savefig(
        out_dir / "heatmap.png", bbox_inches="tight"
    )
    plot_outliers(df["price"], title="price outliers").figure.savefig(
        out_dir / "outliers.png", bbox_inches="tight"
    )
    plot_box(df[["price", "volume"]], title="distribution compare").figure.savefig(
        out_dir / "box.png", bbox_inches="tight"
    )
    print(f"报告产物已写入 {out_dir}/")


if __name__ == "__main__":
    main()
