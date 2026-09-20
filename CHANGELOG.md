# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 约定。

## [0.1.0] - 2026-09-21

首个可用版本。

### 新增

- Python 包 `kimi_space`（src 布局）
  - `stats`：描述性统计摘要（含偏度/超额峰度）、等权滑动平均、z-score 标准化
  - `io`：CSV 装载（必需列校验、数值列强转）与逐列统计汇总
  - `plotting`：统一风格、折线（可叠加滑动平均）、直方图、相关系数热力图
  - `outliers`：IQR 栅栏法与 z-score 法异常值检测
  - CLI `kimi-space`：`summary` / `hist` / `series` / `heatmap` 四个子命令
- Rust 数值内核 crate
  - 与 Python 侧语义一致的统计函数及 `Summary` 结构
  - `kimi-space-rs` stdin 摘要二进制
- 工程化
  - pytest 测试 60+ 个（含 Python/Rust 数值一致性校验）
  - ruff（check + format）、cargo fmt / clippy
  - GitHub Actions 双语言 CI
  - 演示脚本 `examples/demo_report.py`
