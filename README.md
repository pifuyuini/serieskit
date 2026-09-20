# kimi-space

一个轻量的个人数据分析工具集：基于 `numpy` / `pandas` / `scipy` / `matplotlib`，
提供描述性统计、CSV 数据装载、统一风格的绘图辅助以及命令行快速摘要能力；
仓库同时包含一个 Rust 数值内核 crate，作为高性能计算的对照实现。

## 项目结构

```text
├── src/kimi_space/     # Python 包（src 布局）
│   ├── __init__.py
│   ├── stats.py        # 描述性统计与滑动窗口计算
│   ├── io.py           # CSV 装载与轻量校验
│   ├── plotting.py     # matplotlib 统一风格绘图辅助
│   └── cli.py          # 命令行入口：kimi-space
├── tests/              # pytest 测试
├── src/ (Rust)         # Rust 数值内核（cargo crate）
├── pyproject.toml      # Python 项目与工具链配置（uv 管理）
└── Cargo.toml          # Rust crate 配置
```

## 环境要求

- Python >= 3.12，使用 [uv](https://docs.astral.sh/uv/) 管理依赖，虚拟环境位于 `.venv/`
- Rust 工具链（cargo），用于构建 Rust 数值内核

## 快速开始

```bash
uv sync                  # 安装全部依赖（含 dev 依赖）
.venv/bin/pytest         # 运行 Python 测试
.venv/bin/ruff check .   # 代码检查
cargo test               # 运行 Rust 测试
```

## 命令行用法

```bash
kimi-space summary data.csv            # 输出 CSV 各数值列的描述性统计
kimi-space summary data.csv --json     # 以 JSON 格式输出
```
