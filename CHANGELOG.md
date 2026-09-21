# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 约定。

## [0.3.0] - 2026-09-21

### 新增

- CLI
  - 全部子命令支持 `-` 从标准输入读取 CSV（Unix 管道友好）
  - `corr` 子命令：文本相关系数矩阵（pearson/spearman/kendall）
  - `outliers --plot`：检出同时保存异常值标注图
  - `series --x`：自定义横轴列
- 库
  - `plot_series` 支持自定义横轴；`plot_outliers` 支持外部掩码
  - `io.load_csv` 支持 `-` 标准输入
- Rust
  - `kimi-space-rs` 新增 `moving-average <w>` 子命令；`outliers` 计数写 stderr（与 Python 侧对齐）
- 测试与工程化
  - hypothesis 性质测试之外的补充：doctest 纳入测试（--doctest-modules）
  - 版本一致性元测试（pyproject/Cargo.toml/__init__.py/CHANGELOG）
  - Dependabot（github-actions / uv / cargo 周检）
  - 演示报告覆盖异常值、对数收益与箱线图

## [0.2.0] - 2026-09-21

### 新增

- Python
  - `stats`：加权平均、几何平均、调和平均
  - `plotting`：`plot_box` 多列箱线图
  - CLI：`box` 子命令；`summary --output` 落盘 CSV
- Rust
  - 内核：`minmax_normalize`、`pct_change`、`log_returns`、`weighted_mean`、`geometric_mean`、`harmonic_mean`
  - `kimi-space-rs` 子命令化：`summary`（默认）、`means`、`zscores`、`pct-change`、`log-returns`、`outliers [k]`、`minmax <lo> <hi>`
- 测试
  - parity 校验扩展到变换与均值子命令（随机数据 rel 1e-12）
- CI
  - Rust job 增加 cargo 缓存
  - Python 3.12/3.13 矩阵构建

## [0.1.0] - 2026-09-21

首个可用版本。

### 新增

- Python 包 `kimi_space`（src 布局，带 py.typed）
  - `stats`：描述性统计摘要（含偏度/超额峰度）、等权滑动平均、z-score 标准化
  - `io`：CSV 装载（必需列校验、数值列强转）与逐列统计汇总
  - `plotting`：统一风格、折线（可叠加滑动平均）、直方图、相关系数热力图、异常值标注图
  - `outliers`：IQR 栅栏法与 z-score 法异常值检测
  - `transform`：min-max 归一化、相邻变化率、对数收益
  - CLI `kimi-space`：`summary` / `hist` / `series` / `heatmap` / `outliers` / `transform` 六个子命令，支持 `--version`
- Rust 数值内核 crate
  - 与 Python 侧语义一致的统计函数、`Summary` 结构与 `iqr_mask`
  - `kimi-space-rs` stdin 摘要二进制
- 工程化
  - pytest 测试 99 个（含 hypothesis 性质测试与 Python/Rust 数值一致性校验）
  - 行覆盖率 100%，CI 门槛 95%
  - ruff（check + format）、mypy --strict、cargo fmt / clippy
  - pre-commit 钩子（ruff-check + ruff-format）
  - GitHub Actions 双语言 CI
  - 示例：`examples/demo_report.py`、`examples/benchmark.py`（Python/Rust 吞吐对照）
