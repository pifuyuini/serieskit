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
│   ├── outliers.py     # 异常值检测（IQR / z-score）
│   ├── transform.py    # 归一化、变化率、对数收益
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
kimi-space summary data.csv --output summary.csv   # 统计表落盘
kimi-space hist data.csv price         # 直方图保存为 price_hist.png
kimi-space series data.csv price --smooth 10   # 折线图（叠加 10 点滑动平均）
kimi-space series data.csv price --x date      # 自定义横轴列
kimi-space heatmap data.csv            # 相关系数热力图保存为 heatmap.png
kimi-space box data.csv                # 多列箱线图保存为 box.png
kimi-space corr data.csv --method spearman     # 文本相关系数矩阵
kimi-space outliers data.csv price --method iqr   # 逐行输出异常值
kimi-space outliers data.csv price --plot out.png # 检出并保存标注图
kimi-space transform data.csv price --method pct_change  # 逐行输出变换结果
cat data.csv | kimi-space summary -    # 路径为 - 时从标准输入读取（全部子命令通用）
```

## Rust 数值内核

Rust crate 提供与 Python 统计模块语义一致的数值内核（`mean`、`sample_std`、
`weighted_mean`、`geometric_mean`、`harmonic_mean`、`moving_average`、`zscores`、
`percentile`、`skew`、`kurtosis`、`summarize`、`iqr_mask`、`minmax_normalize`、
`pct_change`、`log_returns`），以及一个 Unix 风格的 stdin 工具：

```bash
echo "1 2 3 4" | cargo run --bin kimi-space-rs                # key=value 描述性统计
echo "1 2 3 4" | cargo run --bin kimi-space-rs -- means       # 三种均值
echo "1 2 3 4" | cargo run --bin kimi-space-rs -- moving-average 2
cargo test                                                     # 运行全部 Rust 测试
```

## Python 库用法

```python
from kimi_space import summarize, moving_average, zscores
from kimi_space.io import load_csv, numeric_summary
from kimi_space.plotting import apply_style, plot_series

stats = summarize([1, 2, 3, 4, 5])  # SummaryStats(count=5, mean=3.0, ...)
df = load_csv("data.csv", required_columns=["price"])
table = numeric_summary(df)  # 每个数值列一行的统计表

apply_style()
ax = plot_series(df["price"], smooth_window=10, title="price")
ax.figure.savefig("price.png")
```

## 示例

```bash
.venv/bin/python examples/demo_report.py   # 生成合成数据、统计摘要与五张图表
.venv/bin/python examples/benchmark.py     # Python 与 Rust 内核吞吐对照
```

完整 API 清单见 [docs/api.md](docs/api.md)。

## 开发

```bash
pre-commit install                  # 安装 ruff 钩子（可选，配置见 .pre-commit-config.yaml）
.venv/bin/mypy src/kimi_space       # 静态类型检查（--strict）
.venv/bin/pytest --cov=kimi_space   # 带覆盖率的测试（CI 门槛 95%）
cargo clippy --all-targets          # Rust lint
```

提交前请遵循 AGENTS.md 的审计式提交规范（按路径精确 `git add`，禁止 `git add -A`）。
