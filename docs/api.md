# API 参考

## Python 包 `serieskit`

### 统计（`serieskit.stats`）

| 函数 | 说明 |
| --- | --- |
| `summarize(values) -> SummaryStats` | 描述性统计摘要（count/mean/std/min/q25/median/q75/max/skew/kurtosis/mad），NaN 自动剔除 |
| `moving_average(values, window) -> ndarray` | 等权滑动平均，长度 n-window+1 |
| `moving_std(values, window) -> ndarray` | 滑动窗口样本标准差（ddof=1，window>=2） |
| `moving_median(values, window) -> ndarray` | 滑动窗口中位数（稳健滑动统计） |
| `moving_percentile(values, window, q) -> ndarray` | 滑动窗口任意分位数（q ∈ [0,100]） |
| `zscores(values) -> ndarray` | 样本标准差标准化；常数序列返回全零 |
| `weighted_mean(values, weights) -> float` | 加权平均（权重非负且和为正） |
| `geometric_mean(values) -> float` | 几何平均（要求全正） |
| `harmonic_mean(values) -> float` | 调和平均（要求全正） |
| `autocorr(values, lag=1) -> float` | lag-k 自相关系数（切片 Pearson 语义） |
| `median_abs_deviation(values) -> float` | 中位数绝对偏差（MAD），稳健离散度 |

`SummaryStats` 为 frozen dataclass，`to_dict()` 可转字典。

### 数据装载（`serieskit.io`）

| 函数 | 说明 |
| --- | --- |
| `load_csv(path, *, required_columns=None, numeric_columns=None, parse_dates=None) -> DataFrame` | 读 CSV；`path="-"` 读标准输入；支持必需列校验与数值列强转 |
| `numeric_summary(df) -> DataFrame` | 逐数值列统计（行索引为列名） |
| `numeric_summary_by(df, by) -> DataFrame` | 分组统计（MultiIndex：组 × 列） |

### 变换（`serieskit.transform`）

| 函数 | 说明 |
| --- | --- |
| `minmax_normalize(values, *, feature_range=(0, 1)) -> ndarray` | 最小-最大归一化；常数序列映射区间中点 |
| `pct_change(values) -> ndarray` | 相邻变化率；首元素与前值为 0 处为 NaN |
| `diff(values) -> ndarray` | 一阶差分；首元素为 NaN |
| `log_returns(values) -> ndarray` | 对数收益；比值非正处为 NaN |
| `robust_zscores(values) -> ndarray` | 稳健 z-score（中位数 + MAD）；MAD=0 时全零 |

### 异常值（`serieskit.outliers`）

| 函数 | 说明 |
| --- | --- |
| `iqr_mask(values, *, k=1.5) -> ndarray[bool]` | IQR 栅栏法正常值掩码 |
| `iqr_outliers(values, *, k=1.5) -> ndarray` | IQR 判定的异常值 |
| `zscore_mask(values, *, threshold=3.0) -> ndarray[bool]` | z-score 法正常值掩码 |
| `zscore_outliers(values, *, threshold=3.0) -> ndarray` | z-score 判定的异常值 |
| `mad_mask(values, *, threshold=3.5) -> ndarray[bool]` | MAD 修正 z-score 法正常值掩码（重尾稳健） |
| `mad_outliers(values, *, threshold=3.5) -> ndarray` | MAD 法判定的异常值 |

### 绘图（`serieskit.plotting`，均返回 `matplotlib.axes.Axes`）

| 函数 | 说明 |
| --- | --- |
| `apply_style()` | 应用统一风格 rcParams |
| `plot_series(values, *, x=None, smooth_window=None, ...)` | 折线，可叠加滑动平均 |
| `plot_multi_series(series, *, x=None, ...)` | 多序列对比（Mapping[名称, 序列]） |
| `plot_histogram(values, *, bins=20, ...)` | 直方图 + 均值/中位数参考线 |
| `plot_correlation_heatmap(df, *, columns=None, ...)` | 相关系数热力图（格内标注） |
| `plot_box(df, *, columns=None, ...)` | 多列箱线图 |
| `plot_bands(values, *, window=20, num_std=2.0, ...)` | 滑动均值 ± kσ 波动带 |
| `plot_quantile_bands(values, *, window=20, q_lo=10, q_hi=90, ...)` | 滑动分位数带（无分布假设） |
| `plot_robust_bands(values, *, window=20, num_mad=3.5, ...)` | 滑动中位数 ± k·全局 MAD 稳健波动带 |
| `plot_outliers(values, *, k=1.5, mask=None, ...)` | 折线 + 异常值散点标注 |

## Rust crate `serieskit`

内核函数（均返回 `Option`/`Vec`，不 panic）：
`mean`、`sample_std`、`weighted_mean`、`geometric_mean`、`harmonic_mean`、
`moving_average`、`moving_std`、`moving_median`、`moving_percentile`、`zscores`、`percentile`、`skew`、`kurtosis`、`summarize`、
`iqr_mask`、`mad_mask`、`minmax_normalize`、`pct_change`、`log_returns`、`autocorr`、`median_abs_deviation`、
`robust_zscores`、`diff`。

二进制 `serieskit-rs`（stdin 读数，stdout 输出）：

| 子命令 | 输出 |
| --- | --- |
| `summary`（默认） | key=value 描述性统计；`--json` 输出 JSON，非有限值为 null |
| `means` | arithmetic/geometric/harmonic 三行 |
| `mad` | 中位数绝对偏差 |
| `zscores` / `robust-zscores` / `pct-change` / `log-returns` / `diff` | 逐行变换结果 |
| `moving-average <w>` | 逐行滑动平均 |
| `moving-std <w>` | 逐行滑动样本标准差 |
| `moving-median <w>` | 逐行滑动中位数 |
| `moving-percentile <w> <q>` | 逐行滑动分位数 |
| `autocorr [lag]` | 单个自相关系数 |
| `percentile <q>` | 单个分位数（q ∈ [0, 100]） |
| `outliers [k]` | 逐行异常值（计数在 stderr） |
| `mad-outliers [t]` | 逐行 MAD 法异常值（默认阈值 3.5） |
| `minmax <lo> <hi>` | 逐行归一化结果 |
| `-h` / `--help` | 用法说明 |
| `-V` / `--version` | 版本号 |

Python 与 Rust 的数值一致性由 `tests/test_parity.py` 保证（随机数据，rel 1e-9~1e-12）。

## 数值与输出约定

- Python `summarize`、`median_abs_deviation`、几何及调和平均会剔除 NaN；
  滑动统计和多数变换会保留或传播 NaN。Rust 不统一剔除 NaN。
- Python 非法输入通常抛出 ValueError；Rust 按函数返回 None 或空向量。
  parity 测试覆盖给定的数据域，并不意味着所有边界输入完全等价。
- 异常值掩码中 True 表示正常值；MAD 为零时，检测只保留中位数值，
  而稳健标准化返回全零。
- Python `summary --json` 和 Rust `summary --json` 都以 null 表示非有限值。
  Python 输出按列名（分组时为组名与列名）组织，Rust 输出单个摘要对象。
- CLI 图片保存后会关闭对应 Figure；库函数返回 Axes，由调用者管理生命周期。
