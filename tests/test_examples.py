"""示例脚本的冒烟测试：保证示例随代码演进保持可运行。"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_demo_report_runs(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "demo_report.py"), str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "price" in proc.stdout
    for name in [
        "demo.csv",
        "series.png",
        "bands.png",
        "hist.png",
        "heatmap.png",
        "outliers.png",
        "box.png",
    ]:
        artifact = tmp_path / name
        assert artifact.exists(), f"缺少产物 {name}"
        assert artifact.stat().st_size > 0


@pytest.mark.skipif(shutil.which("cargo") is None, reason="需要 cargo")
def test_benchmark_runs():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "benchmark.py")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "python moving_average" in proc.stdout
    assert "rust moving_average" in proc.stdout
    assert "rust summarize" in proc.stdout
