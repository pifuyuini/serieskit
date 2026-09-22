"""对比 Python(numpy) 与 Rust 内核的滑动平均吞吐。

用法：.venv/bin/python examples/benchmark.py
工作负载：200_000 个随机浮点，窗口 50 的等权滑动平均 + 全量描述性统计。
"""

import subprocess
import time
from pathlib import Path

import numpy as np

from serieskit.stats import moving_average, moving_std, summarize

ROOT = Path(__file__).resolve().parent.parent
N = 200_000
WINDOW = 50


def main() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=N)

    t0 = time.perf_counter()
    ma = moving_average(data, WINDOW)
    t1 = time.perf_counter()
    print(
        f"python moving_average: n={N} window={WINDOW} -> {ma.size} points "
        f"in {(t1 - t0) * 1e3:.3f} ms (checksum {float(np.sum(ma)):.6f})"
    )

    t2 = time.perf_counter()
    s = summarize(data)
    t3 = time.perf_counter()
    print(f"python summarize: n={N} in {(t3 - t2) * 1e3:.3f} ms (mean {s.mean:.6f})")

    t4 = time.perf_counter()
    ms = moving_std(data, WINDOW)
    t5 = time.perf_counter()
    print(
        f"python moving_std: n={N} window={WINDOW} -> {ms.size} points "
        f"in {(t5 - t4) * 1e3:.3f} ms (checksum {float(np.sum(ms)):.6f})"
    )

    proc = subprocess.run(
        ["cargo", "run", "--release", "--quiet", "--example", "bench"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    print(proc.stdout, end="")


if __name__ == "__main__":
    main()
