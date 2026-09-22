"""Python 统计模块与 Rust 数值内核的数值一致性校验。

需要本机安装 cargo；没有 cargo 的环境（如纯 Python CI）自动跳过。
"""

import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as scipy_stats

from serieskit.outliers import mad_outliers
from serieskit.stats import (
    autocorr,
    geometric_mean,
    harmonic_mean,
    median_abs_deviation,
    moving_average,
    moving_median,
    moving_percentile,
    moving_std,
    summarize,
    zscores,
)
from serieskit.transform import (
    diff,
    log_returns,
    minmax_normalize,
    pct_change,
    robust_zscores,
)

ROOT = Path(__file__).resolve().parent.parent
BIN = (
    ROOT
    / "target"
    / "debug"
    / ("serieskit-rs.exe" if os.name == "nt" else "serieskit-rs")
)

pytestmark = pytest.mark.skipif(shutil.which("cargo") is None, reason="需要 cargo")


@pytest.fixture(scope="module")
def rust_bin():
    subprocess.run(
        ["cargo", "build", "--quiet", "--bin", "serieskit-rs"],
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


def rust_rows(rust_bin, command: list[str], values) -> list[float]:
    proc = subprocess.run(
        [str(rust_bin), *command],
        input=" ".join(map(repr, values)),
        capture_output=True,
        text=True,
        check=True,
    )
    return [float(line) for line in proc.stdout.strip().splitlines()]


def rust_key_values(rust_bin, command: list[str], values) -> dict[str, float]:
    proc = subprocess.run(
        [str(rust_bin), *command],
        input=" ".join(map(repr, values)),
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        k: float(v)
        for k, v in (line.split("=", 1) for line in proc.stdout.strip().splitlines())
    }


def assert_rows_close(actual, expected) -> None:
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected, strict=True):
        if math.isnan(e):
            assert math.isnan(a)
        else:
            assert a == pytest.approx(e, rel=1e-12)


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
        assert float(rust["mad"]) == pytest.approx(
            median_abs_deviation(data), rel=1e-12
        )

    def test_integer_sequence(self, rust_bin):
        data = list(range(1, 11))
        rust = rust_summary(rust_bin, data)
        assert float(rust["mean"]) == pytest.approx(5.5)
        assert float(rust["q25"]) == pytest.approx(3.25)
        assert float(rust["q75"]) == pytest.approx(7.75)
        assert float(rust["kurtosis"]) == pytest.approx(-1.2242424242424244, rel=1e-12)

    def test_zscores_parity(self, rust_bin):
        rng = np.random.default_rng(7)
        data = rng.normal(size=200).tolist()
        rust = rust_rows(rust_bin, ["zscores"], data)
        assert_rows_close(rust, zscores(data).tolist())

    def test_pct_change_parity(self, rust_bin):
        rng = np.random.default_rng(8)
        data = (100 + rng.normal(size=100).cumsum()).tolist()
        rust = rust_rows(rust_bin, ["pct-change"], data)
        assert_rows_close(rust, pct_change(data).tolist())

    def test_log_returns_parity(self, rust_bin):
        rng = np.random.default_rng(9)
        data = np.exp(rng.normal(size=100)).tolist()
        rust = rust_rows(rust_bin, ["log-returns"], data)
        assert_rows_close(rust, log_returns(data).tolist())

    def test_minmax_parity(self, rust_bin):
        rng = np.random.default_rng(10)
        data = rng.normal(size=100).tolist()
        rust = rust_rows(rust_bin, ["minmax", "-1", "1"], data)
        expected = minmax_normalize(data, feature_range=(-1.0, 1.0)).tolist()
        assert_rows_close(rust, expected)

    def test_means_parity(self, rust_bin):
        rng = np.random.default_rng(11)
        data = rng.uniform(0.5, 5.0, size=300).tolist()
        rust = rust_key_values(rust_bin, ["means"], data)
        assert rust["arithmetic"] == pytest.approx(np.mean(data), rel=1e-12)
        assert rust["geometric"] == pytest.approx(geometric_mean(data), rel=1e-12)
        assert rust["harmonic"] == pytest.approx(harmonic_mean(data), rel=1e-12)

    def test_moving_average_parity(self, rust_bin):
        rng = np.random.default_rng(12)
        data = rng.normal(size=300).tolist()
        rust = rust_rows(rust_bin, ["moving-average", "10"], data)
        assert_rows_close(rust, moving_average(data, 10).tolist())

    def test_autocorr_parity(self, rust_bin):
        rng = np.random.default_rng(13)
        data = rng.normal(size=500).tolist()
        for lag in [1, 5, 20]:
            rust = rust_rows(rust_bin, ["autocorr", str(lag)], data)
            assert rust[0] == pytest.approx(autocorr(data, lag), rel=1e-9)

    def test_moving_std_parity(self, rust_bin):
        rng = np.random.default_rng(14)
        data = rng.normal(size=300).tolist()
        rust = rust_rows(rust_bin, ["moving-std", "7"], data)
        assert_rows_close(rust, moving_std(data, 7).tolist())

    def test_percentile_parity(self, rust_bin):
        rng = np.random.default_rng(15)
        data = rng.normal(size=400).tolist()
        for q in [0, 10, 25, 50, 90, 100]:
            rust = rust_rows(rust_bin, ["percentile", str(q)], data)
            assert rust[0] == pytest.approx(np.percentile(data, q), rel=1e-12)

    def test_mad_parity(self, rust_bin):
        rng = np.random.default_rng(16)
        data = rng.normal(size=400).tolist()
        rust = rust_rows(rust_bin, ["mad"], data)
        assert rust[0] == pytest.approx(median_abs_deviation(data), rel=1e-12)

    def test_mad_outliers_parity(self, rust_bin):
        rng = np.random.default_rng(17)
        data = rng.normal(size=300).tolist() + [50.0, -45.0]
        rust = rust_rows(rust_bin, ["mad-outliers"], data)
        expected = mad_outliers(data).tolist()
        assert_rows_close(sorted(rust), sorted(expected))

    def test_robust_zscores_parity(self, rust_bin):
        rng = np.random.default_rng(18)
        data = rng.normal(size=300).tolist() + [50.0]
        rust = rust_rows(rust_bin, ["robust-zscores"], data)
        assert_rows_close(rust, robust_zscores(data).tolist())

    def test_diff_parity(self, rust_bin):
        rng = np.random.default_rng(19)
        data = rng.normal(size=300).tolist()
        rust = rust_rows(rust_bin, ["diff"], data)
        assert_rows_close(rust, diff(data).tolist())

    def test_moving_median_parity(self, rust_bin):
        rng = np.random.default_rng(20)
        data = rng.normal(size=300).tolist()
        rust = rust_rows(rust_bin, ["moving-median", "6"], data)
        assert_rows_close(rust, moving_median(data, 6).tolist())

    def test_moving_percentile_parity(self, rust_bin):
        rng = np.random.default_rng(22)
        data = rng.normal(size=300).tolist()
        for q in [10, 50, 90]:
            rust = rust_rows(rust_bin, ["moving-percentile", "8", str(q)], data)
            assert_rows_close(rust, moving_percentile(data, 8, q).tolist())

    def test_summary_json_parity(self, rust_bin):
        rng = np.random.default_rng(21)
        data = rng.normal(size=200).tolist()
        proc = subprocess.run(
            [str(rust_bin), "summary", "--json"],
            input=" ".join(map(repr, data)),
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(proc.stdout)  # 严格 JSON 解析成功即格式合法
        s = summarize(data)
        for key in [
            "count",
            "mean",
            "std",
            "min",
            "q25",
            "median",
            "q75",
            "max",
            "skew",
            "kurtosis",
            "mad",
        ]:
            assert payload[key] == pytest.approx(getattr(s, key), rel=1e-9)
