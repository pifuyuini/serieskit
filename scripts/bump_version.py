"""发版版本号同步脚本：更新 pyproject.toml、Cargo.toml、__init__.py 与 CHANGELOG。

用法：.venv/bin/python scripts/bump_version.py 0.7.0
"""

import re
import subprocess
import sys
import tomllib
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def bump(files_version: str) -> None:
    if not re.fullmatch(r"\d+\.\d+\.\d+", files_version):
        print(f"error: 版本号须为 X.Y.Z，收到 {files_version!r}", file=sys.stderr)
        sys.exit(1)

    pyproject = ROOT / "pyproject.toml"
    init_py = ROOT / "src" / "kimi_space" / "__init__.py"
    cargo = ROOT / "Cargo.toml"
    changelog = ROOT / "CHANGELOG.md"

    current = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]
    if files_version == current:
        print(f"error: 版本已是 {current}", file=sys.stderr)
        sys.exit(1)

    for path, pattern in [
        (pyproject, r'(?m)^(version = ")[\d.]+("$)'),
        (cargo, r'(?m)^(version = ")[\d.]+("$)'),
        (init_py, r'(?m)^(__version__ = ")[\d.]+("$)'),
    ]:
        text = path.read_text(encoding="utf-8")
        new_text, n = re.subn(pattern, rf"\g<1>{files_version}\g<2>", text, count=1)
        if n != 1:
            print(f"error: {path} 中未找到版本号", file=sys.stderr)
            sys.exit(1)
        path.write_text(new_text, encoding="utf-8")

    text = changelog.read_text(encoding="utf-8")
    heading = f"## [{files_version}] - {date.today().isoformat()}"
    if heading in text:
        print(f"error: CHANGELOG 已存在 {heading}", file=sys.stderr)
        sys.exit(1)
    marker = "\n## ["
    idx = text.index(marker)
    text = text[:idx] + f"\n{heading}\n\n### 新增\n\n- TODO\n" + text[idx:]
    changelog.write_text(text, encoding="utf-8")

    subprocess.run(["uv", "sync", "-q"], cwd=ROOT, check=True)
    subprocess.run(["cargo", "build", "-q"], cwd=ROOT, check=True)
    print(f"已升级到 {files_version}（CHANGELOG 顶部留有 TODO，请补全后提交）")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    bump(sys.argv[1])
