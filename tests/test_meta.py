"""版本号与文档的一致性校验：pyproject、Cargo.toml、__init__.py、CHANGELOG。"""

import re
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
        "autocorr",
        "outliers",
        "minmax",
    ]:
        assert cmd in api, f"api.md 缺少 Rust 子命令 {cmd}"
