"""Python 统计模块与 Rust 数值内核的数值一致性校验。

需要本机安装 cargo；没有 cargo 的环境（如纯 Python CI）自动跳过。
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target" / "debug" / "kimi-space-rs"

pytestmark = pytest.mark.skipif(shutil.which("cargo") is None, reason="需要 cargo")


@pytest.fixture(scope="module")
def rust_bin():
    subprocess.run(
        ["cargo", "build", "--quiet", "--bin", "kimi-space-rs"],
        cwd=ROOT,
        check=True,
    )
    return BIN


def rust_summary(rust_bin, values) -> dict[str, float]:
    proc = subprocess.run(
        [str(rust_bin)],
        input=" ".join(map(repr, values)),
        capture_output=True,
        text=True,
        check=True,
    )
    return dict(line.split("=", 1) for line in proc.stdout.strip().splitlines())


class TestRustParity:
    def test_normal_sample(self, rust_bin):
        rng = np.random.default_rng(42)
        data = rng.normal(loc=3.0, scale=2.0, size=1000)
        rust = rust_summary(rust_bin, data.tolist())
        assert int(rust["count"]) == 1000
        assert float(rust["mean"]) == pytest.approx(np.mean(data), rel=1e-12)
        assert float(rust["std"]) == pytest.approx(np.std(data, ddof=1), rel=1e-12)
        assert float(rust["min"]) == pytest.approx(np.min(data), rel=1e-12)
        assert float(rust["max"]) == pytest.approx(np.max(data), rel=1e-12)
        assert float(rust["q25"]) == pytest.approx(np.percentile(data, 25), rel=1e-12)
        assert float(rust["median"]) == pytest.approx(np.median(data), rel=1e-12)
        assert float(rust["q75"]) == pytest.approx(np.percentile(data, 75), rel=1e-12)
        assert float(rust["skew"]) == pytest.approx(scipy_stats.skew(data), rel=1e-9)
        assert float(rust["kurtosis"]) == pytest.approx(
            scipy_stats.kurtosis(data), rel=1e-9
        )

    def test_integer_sequence(self, rust_bin):
        data = list(range(1, 11))
        rust = rust_summary(rust_bin, data)
        assert float(rust["mean"]) == pytest.approx(5.5)
        assert float(rust["q25"]) == pytest.approx(3.25)
        assert float(rust["q75"]) == pytest.approx(7.75)
        assert float(rust["kurtosis"]) == pytest.approx(-1.2242424242424244, rel=1e-12)
