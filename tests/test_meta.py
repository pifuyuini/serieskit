"""版本号与文档的一致性校验：pyproject、Cargo.toml、__init__.py、CHANGELOG。"""

import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

import kimi_space

ROOT = Path(__file__).resolve().parent.parent


def test_pyproject_matches_package():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["version"] == kimi_space.__version__


def test_cargo_matches_package():
    data = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    assert data["package"]["version"] == kimi_space.__version__


def test_changelog_documents_current_version():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    headings = re.findall(r"^## \[(\d+\.\d+\.\d+)\]", changelog, flags=re.M)
    assert headings, "CHANGELOG 没有版本条目"
    assert headings[0] == kimi_space.__version__


def test_license_exists():
    assert (ROOT / "LICENSE").exists()


def test_readme_python_cli_subcommands_exist():
    from kimi_space.cli import main

    text = (ROOT / "README.md").read_text(encoding="utf-8")
    cmds = set(re.findall(r"kimi-space (\w+)", text))
    assert {
        "summary",
        "hist",
        "series",
        "heatmap",
        "box",
        "corr",
        "outliers",
        "transform",
        "autocorr",
        "percentile",
        "mad",
        "report",
    } <= cmds
    for cmd in cmds:
        with pytest.raises(SystemExit) as exc_info:
            main([cmd, "--help"])
        assert exc_info.value.code == 0


def test_api_md_lists_all_cli_subcommands():
    api = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    for cmd in [
        "summary",
        "means",
        "zscores",
        "pct-change",
        "log-returns",
        "moving-average",
        "moving-std",
        "moving-median",
        "moving-percentile",
        "autocorr",
        "percentile",
        "outliers",
        "mad-outliers",
        "minmax",
        "mad",
        "robust-zscores",
        "diff",
    ]:
        assert cmd in api, f"api.md 缺少 Rust 子命令 {cmd}"


def test_changelog_top_matches_latest_tag():
    """最新 tag 必须在 CHANGELOG 中有条目（允许 CHANGELOG 领先于 tag，即发版进行中）。

    无 git 或无 tag 的环境自动跳过。
    """
    if shutil.which("git") is None:
        pytest.skip("需要 git")
        return
    proc = subprocess.run(
        ["git", "tag", "--sort=-version:refname"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    latest = proc.stdout.splitlines()[0] if proc.stdout.strip() else ""
    if not latest:
        pytest.skip("仓库没有 tag")
        return
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    headings = set(re.findall(r"^## \[(\d+\.\d+\.\d+)\]", changelog, flags=re.M))
    assert latest.lstrip("v") in headings, f"tag {latest} 在 CHANGELOG 中没有条目"
