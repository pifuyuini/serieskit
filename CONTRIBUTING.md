# Contributing to SeriesKit

## Setup

Use Python 3.12+ and uv. Rust (edition 2024) is needed for Rust development
and cross-language tests.

```bash
git clone https://github.com/pifuyuini/serieskit.git
cd serieskit
uv sync --locked
uv run serieskit summary examples/sample.csv
```

The repository uses a `.venv/` environment. Ruff, mypy, pytest, and test
support packages are development dependencies.

## Validate

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/serieskit
uv run pytest --cov=serieskit --cov-report=term-missing --cov-fail-under=95
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
```

The Python tests include doctests, property checks, example runs, CLI checks,
and numerical comparisons with Rust. Parity tests build the Rust CLI and are
skipped locally if cargo is unavailable. CI explicitly installs Rust so those
checks run. The coverage floor is 95%.

Optional hooks: `uv run pre-commit install`. Benchmark with
`cargo bench --bench kernels` or `uv run python examples/benchmark.py`.

## Layout

- `src/serieskit/`: Python statistics, CSV I/O, plotting, outliers, transforms, CLI.
- `src/lib.rs`: independent Rust numeric kernels.
- `src/bin/serieskit-rs.rs`: Rust stdin CLI.
- `tests/`: Python tests and Rust CLI integration tests.
- `docs/api.md`: API reference; `examples/`: sample data, report, benchmarks.

Keep changes focused, preserve existing user work, and follow the repository's
`AGENTS.md` commit rules. Include tests for behavior changes and update the
API documentation and CHANGELOG.

## Release

1. Run `uv run python scripts/bump_version.py X.Y.Z` and complete the new
   CHANGELOG entry. The script updates four version locations and both lockfiles.
2. Update versioned installation examples in README and run the checks above.
3. Commit the release and push main. Create an annotated tag `vX.Y.Z` and push
   that tag separately after the repository workflows are registered.
4. The Release workflow runs CI, builds the Python wheel/source distribution and
   native Rust archives, then creates a GitHub Release only after every job passes.

Only new release tags should be pushed to trigger publication; historical tags
retain their original source and workflows. GitHub Releases do not publish to
PyPI or crates.io. Those registries require a separate, explicitly configured
release process.

Python release artifacts are built from the source distribution and the wheel
entry point is smoke-tested before upload. Rust assets cover Linux x86_64, macOS
arm64, and Windows x86_64.

To retry publication manually, run the Release workflow with the release tag as
the ref. Branch refs are rejected by the version check.
