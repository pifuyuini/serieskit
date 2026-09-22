# SeriesKit

Statistics, CSV analysis, and plots from your terminal or Python code.

轻量数据分析工具集，支持描述统计、滑动窗口、异常值检测、分组统计和图表报告。

[![CI](https://github.com/pifuyuini/serieskit/actions/workflows/ci.yml/badge.svg)](https://github.com/pifuyuini/serieskit/actions/workflows/ci.yml)
[Releases](https://github.com/pifuyuini/serieskit/releases) · [API reference（中文）](docs/api.md) · [Contributing](CONTRIBUTING.md)

SeriesKit includes a Python library and CLI backed by NumPy, pandas, SciPy,
and Matplotlib, plus an independent Rust statistics library and CLI.
Python users do not need Rust. The two implementations are compared in numerical
parity tests; edge-case behavior is documented per function.

## Install

Python 3.12 or newer is required. Until a package-index release is available,
install from the versioned GitHub source:

```bash
# CLI in its own environment (requires uv and Git)
uv tool install "git+https://github.com/pifuyuini/serieskit.git@v0.21.1"

# Library in your active Python environment
python -m pip install "git+https://github.com/pifuyuini/serieskit.git@v0.21.1"

serieskit --version
serieskit --help
```

Alternatively, download the Python wheel from
[Releases](https://github.com/pifuyuini/serieskit/releases/latest) and install it
with `python -m pip install /path/to/serieskit-0.21.1-py3-none-any.whl`.
Both installation methods also resolve the Python runtime dependencies.
The Rust CLI has separate native downloads for Linux x86_64, macOS Apple Silicon,
and Windows x86_64; extract the matching archive and run `serieskit-rs --help`
(`serieskit-rs.exe` on Windows). Other platforms can build from source:

```bash
cargo install --git https://github.com/pifuyuini/serieskit.git --tag v0.21.1 --locked
```

No PyPI or crates.io publication is implied by a GitHub release.

## Try it

Download [sample.csv](examples/sample.csv) or clone this repository and run:

```bash
serieskit summary examples/sample.csv
serieskit summary examples/sample.csv --json
serieskit summary examples/sample.csv --by region --output summary.csv
serieskit report examples/sample.csv --outdir report
```

The report contains a statistics CSV, one histogram per numeric column,
and a correlation heatmap and box plot when there are multiple numeric columns.
With `--by region`, the statistics are grouped; the charts still use the whole
dataset. Output files with the same names are overwritten.

## Commands

All 12 commands accept a CSV path; use `-` to read CSV from standard input.

| Command | Example |
| --- | --- |
| Summary | `serieskit summary data.csv --by region` |
| Histogram | `serieskit hist data.csv price --kde` |
| Series | `serieskit series data.csv price --x date --smooth 3` |
| Heatmap | `serieskit heatmap data.csv` |
| Box plot | `serieskit box data.csv` |
| Correlation | `serieskit corr data.csv --method spearman` |
| Outliers | `serieskit outliers data.csv price --method mad` |
| Transform | `serieskit transform data.csv price --method diff` |
| Autocorrelation | `serieskit autocorr data.csv price --lag 2` |
| Percentile | `serieskit percentile data.csv price --q 90` |
| MAD | `serieskit mad data.csv price` |
| Report | `serieskit report data.csv --outdir report` |

Use `serieskit summary --help` for options, or `python -m serieskit --help`.
For multi-column series, pass `price,volume`. Smoothing and bands apply to a
single column. Headless environments can set `MPLBACKEND=Agg`.

## Python library

```python
from serieskit import summarize, moving_average
from serieskit.io import load_csv, numeric_summary_by
from serieskit.plotting import plot_quantile_bands

stats = summarize([1, 2, 3, 4, 5])
assert stats.mean == 3.0
print(moving_average([1, 2, 3, 4], window=2))

df = load_csv("examples/sample.csv", parse_dates=["date"])
print(numeric_summary_by(df, by="region"))
ax = plot_quantile_bands(df["price"], window=3)
ax.figure.savefig("bands.png", bbox_inches="tight")
```

Statistics use sample standard deviation (`ddof=1`) and excess kurtosis.
Outlier masks mark **normal** values with `True`. Rolling results have length
`n - window + 1`. Missing-value handling varies by function; see the
[API reference](docs/api.md). Summary JSON represents non-finite values as `null`.

## Rust CLI

The Rust CLI reads whitespace-separated numbers rather than CSV:

```bash
echo "1 2 3 4" | serieskit-rs summary --json
echo "1 2 3 4" | serieskit-rs moving-average 2
echo "1 2 3 4" | serieskit-rs percentile 90
```

It offers 17 computation commands. `--help` and `--version` are flags.

## Development and migration

See [CONTRIBUTING.md](CONTRIBUTING.md) for local setup, tests, and releases.
The API documentation is currently in Chinese; command help and errors are
also primarily Chinese.

Versions up to v0.20.0 were named **kimi-space**. Starting with v0.21.0:

- Replace `import kimi_space` with `import serieskit`.
- Replace commands `kimi-space` / `kimi-space-rs` with `serieskit` / `serieskit-rs`.
- Change the Rust dependency to `serieskit` and use `serieskit::...`.
- Old names are not provided as compatibility aliases. Earlier tags preserve
  the original project and installation names.

MIT licensed; see [LICENSE](LICENSE).
