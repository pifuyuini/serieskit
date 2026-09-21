# API 参考

## Python 包 `kimi_space`

### 统计（`kimi_space.stats`）

| 函数 | 说明 |
| --- | --- |
| `summarize(values) -> SummaryStats` | 描述性统计摘要（count/mean/std/min/q25/median/q75/max/skew/kurtosis/mad），NaN 自动剔除 |
| `moving_average(values, window) -> ndarray` | 等权滑动平均，长度 n-window+1 |
| `moving_std(values, window) -> ndarray` | 滑动窗口样本标准差（ddof=1，window>=2） |
| `zscores(values) -> ndarray` | 样本标准差标准化；常数序列返回全零 |
| `weighted_mean(values, weights) -> float` | 加权平均（权重非负且和为正） |
| `geometric_mean(values) -> float` | 几何平均（要求全正） |
| `harmonic_mean(values) -> float` | 调和平均（要求全正） |
| `autocorr(values, lag=1) -> float` | lag-k 自相关系数（切片 Pearson 语义） |
| `median_abs_deviation(values) -> float` | 中位数绝对偏差（MAD），稳健离散度 |

`SummaryStats` 为 frozen dataclass，`to_dict()` 可转字典。

### 数据装载（`kimi_space.io`）

| 函数 | 说明 |
| --- | --- |
| `load_csv(path, *, required_columns=None, numeric_columns=None) -> DataFrame` | 读 CSV；`path="-"` 读标准输入；支持必需列校验与数值列强转 |
| `numeric_summary(df) -> DataFrame` | 逐数值列统计（行索引为列名） |
| `numeric_summary_by(df, by) -> DataFrame` | 分组统计（MultiIndex：组 × 列） |

### 变换（`kimi_space.transform`）

| 函数 | 说明 |
| --- | --- |
| `minmax_normalize(values, *, feature_range=(0, 1)) -> ndarray` | 最小-最大归一化；常数序列映射区间中点 |
| `pct_change(values) -> ndarray` | 相邻变化率；首元素与前值为 0 处为 NaN |
| `log_returns(values) -> ndarray` | 对数收益；比值非正处为 NaN |

### 异常值（`kimi_space.outliers`）

| 函数 | 说明 |
| --- | --- |
| `iqr_mask(values, *, k=1.5) -> ndarray[bool]` | IQR 栅栏法正常值掩码 |
| `iqr_outliers(values, *, k=1.5) -> ndarray` | IQR 判定的异常值 |
| `zscore_mask(values, *, threshold=3.0) -> ndarray[bool]` | z-score 法正常值掩码 |
| `zscore_outliers(values, *, threshold=3.0) -> ndarray` | z-score 判定的异常值 |
| `mad_mask(values, *, threshold=3.5) -> ndarray[bool]` | MAD 修正 z-score 法正常值掩码（重尾稳健） |
| `mad_outliers(values, *, threshold=3.5) -> ndarray` | MAD 法判定的异常值 |

### 绘图（`kimi_space.plotting`，均返回 `matplotlib.axes.Axes`）

| 函数 | 说明 |
| --- | --- |
| `apply_style()` | 应用统一风格 rcParams |
| `plot_series(values, *, x=None, smooth_window=None, ...)` | 折线，可叠加滑动平均 |
| `plot_histogram(values, *, bins=20, ...)` | 直方图 + 均值/中位数参考线 |
| `plot_correlation_heatmap(df, *, columns=None, ...)` | 相关系数热力图（格内标注） |
| `plot_box(df, *, columns=None, ...)` | 多列箱线图 |
| `plot_bands(values, *, window=20, num_std=2.0, ...)` | 滑动均值 ± kσ 波动带 |
| `plot_outliers(values, *, k=1.5, mask=None, ...)` | 折线 + 异常值散点标注 |

## Rust crate `kimi-space`

内核函数（均返回 `Option`/`Vec`，不 panic）：
`mean`、`sample_std`、`weighted_mean`、`geometric_mean`、`harmonic_mean`、
`moving_average`、`moving_std`、`zscores`、`percentile`、`skew`、`kurtosis`、`summarize`、
`iqr_mask`、`minmax_normalize`、`pct_change`、`log_returns`、`autocorr`、`median_abs_deviation`。

二进制 `kimi-space-rs`（stdin 读数，stdout 输出）：

| 子命令 | 输出 |
| --- | --- |
| `summary`（默认） | key=value 描述性统计 |
| `means` | arithmetic/geometric/harmonic 三行 |
| `mad` | 中位数绝对偏差 |
| `zscores` / `pct-change` / `log-returns` | 逐行变换结果 |
| `moving-average <w>` | 逐行滑动平均 |
| `moving-std <w>` | 逐行滑动样本标准差 |
| `autocorr [lag]` | 单个自相关系数 |
| `percentile <q>` | 单个分位数（q ∈ [0, 100]） |
| `outliers [k]` | 逐行异常值（计数在 stderr） |
| `minmax <lo> <hi>` | 逐行归一化结果 |
| `-h` / `--help` | 用法说明 |
| `-V` / `--version` | 版本号 |

Python 与 Rust 的数值一致性由 `tests/test_parity.py` 保证（随机数据，rel 1e-9~1e-12）。
