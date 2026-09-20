# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 约定。

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
