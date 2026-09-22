import json

import pytest

from serieskit.cli import main

CSV_TEXT = "name,a,b\nx,1,1.5\ny,2,2.5\nz,3,3.5\n"


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(CSV_TEXT, encoding="utf-8")
    return path


class TestSummary:
    def test_table_output(self, csv_path, capsys):
        assert main(["summary", str(csv_path)]) == 0
        out = capsys.readouterr().out
        assert "mean" in out and "a" in out and "b" in out

    def test_json_output(self, csv_path, capsys):
        assert main(["summary", str(csv_path), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["a"]["mean"] == pytest.approx(2.0)
        assert payload["b"]["max"] == pytest.approx(3.5)

    def test_missing_file_returns_1(self, tmp_path, capsys):
        assert main(["summary", str(tmp_path / "absent.csv")]) == 1
        assert "error" in capsys.readouterr().err

    def test_output_writes_csv(self, csv_path, tmp_path, capsys):
        import pandas as pd

        out = tmp_path / "summary.csv"
        assert main(["summary", str(csv_path), "--output", str(out)]) == 0
        assert str(out) in capsys.readouterr().out
        saved = pd.read_csv(out, index_col=0)
        assert saved.loc["a", "mean"] == pytest.approx(2.0)
        assert saved.loc["b", "count"] == 3

    def test_group_by(self, csv_path, capsys):
        assert main(["summary", str(csv_path), "--by", "name"]) == 0
        out = capsys.readouterr().out
        assert "x" in out and "y" in out and "z" in out

    def test_group_by_json(self, csv_path, capsys):
        assert main(["summary", str(csv_path), "--by", "name", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["x | a"]["mean"] == pytest.approx(1.0)
        assert payload["z | b"]["max"] == pytest.approx(3.5)

    def test_group_by_missing_column_returns_1(self, csv_path, capsys):
        assert main(["summary", str(csv_path), "--by", "nope"]) == 1
        assert "error" in capsys.readouterr().err

    def test_group_by_two_columns(self, tmp_path, capsys):
        csv = tmp_path / "g2.csv"
        csv.write_text("g1,g2,v\na,u,1\na,u,2\na,v,3\nb,u,4\n", encoding="utf-8")
        assert main(["summary", str(csv), "--by", "g1,g2"]) == 0
        out = capsys.readouterr().out
        assert "g1" in out and "g2" in out

    def test_group_by_two_columns_json(self, tmp_path, capsys):
        csv = tmp_path / "g2.csv"
        csv.write_text("g1,g2,v\na,u,1\na,u,2\na,v,3\nb,u,4\n", encoding="utf-8")
        assert main(["summary", str(csv), "--by", "g1,g2", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["a | u | v"]["mean"] == pytest.approx(1.5)


class TestHist:
    def test_saves_default_output(self, csv_path, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        assert main(["hist", str(csv_path), "a"]) == 0
        assert (tmp_path / "a_hist.png").exists()
        assert "a_hist.png" in capsys.readouterr().out

    def test_custom_output_and_bins(self, csv_path, tmp_path):
        out = tmp_path / "custom.png"
        assert (
            main(["hist", str(csv_path), "b", "--bins", "2", "--output", str(out)]) == 0
        )
        assert out.exists() and out.stat().st_size > 0

    def test_non_numeric_column_returns_1(self, csv_path, capsys):
        assert main(["hist", str(csv_path), "name"]) == 1
        assert "error" in capsys.readouterr().err

    def test_kde_flag(self, csv_path, tmp_path):
        out = tmp_path / "kde.png"
        assert main(["hist", str(csv_path), "a", "--kde", "--output", str(out)]) == 0
        assert out.exists() and out.stat().st_size > 0


class TestSeries:
    def test_saves_default_output(self, csv_path, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        assert main(["series", str(csv_path), "a"]) == 0
        assert (tmp_path / "a_series.png").exists()
        assert "a_series.png" in capsys.readouterr().out

    def test_smooth_and_custom_output(self, csv_path, tmp_path):
        out = tmp_path / "s.png"
        code = main(
            ["series", str(csv_path), "b", "--smooth", "2", "--output", str(out)]
        )
        assert code == 0
        assert out.exists() and out.stat().st_size > 0

    def test_non_numeric_column_returns_1(self, csv_path, capsys):
        assert main(["series", str(csv_path), "name"]) == 1
        assert "error" in capsys.readouterr().err

    def test_custom_x_column(self, csv_path, tmp_path):
        out = tmp_path / "s.png"
        code = main(["series", str(csv_path), "a", "--x", "b", "--output", str(out)])
        assert code == 0
        assert out.exists()

    def test_bands_option(self, csv_path, tmp_path):
        out = tmp_path / "band.png"
        code = main(
            ["series", str(csv_path), "a", "--bands", "2", "--output", str(out)]
        )
        assert code == 0
        assert out.exists() and out.stat().st_size > 0

    def test_bands_bad_window_returns_1(self, csv_path, capsys, tmp_path):
        out = tmp_path / "band.png"
        code = main(
            ["series", str(csv_path), "a", "--bands", "9", "--output", str(out)]
        )
        assert code == 1
        assert "error" in capsys.readouterr().err

    def test_smooth_and_bands_are_exclusive(self, csv_path, tmp_path):
        out = tmp_path / "x.png"
        with pytest.raises(SystemExit):
            main(
                [
                    "series",
                    str(csv_path),
                    "a",
                    "--smooth",
                    "2",
                    "--bands",
                    "2",
                    "--output",
                    str(out),
                ]
            )

    def test_multi_column(self, csv_path, tmp_path):
        out = tmp_path / "multi.png"
        code = main(["series", str(csv_path), "a,b", "--output", str(out)])
        assert code == 0
        assert out.exists() and out.stat().st_size > 0

    def test_multi_column_rejects_smooth(self, csv_path, tmp_path, capsys):
        out = tmp_path / "multi.png"
        code = main(
            ["series", str(csv_path), "a,b", "--smooth", "2", "--output", str(out)]
        )
        assert code == 1
        assert "仅支持单列" in capsys.readouterr().err

    def test_robust_bands_option(self, csv_path, tmp_path):
        out = tmp_path / "rb.png"
        code = main(
            ["series", str(csv_path), "a", "--robust-bands", "2", "--output", str(out)]
        )
        assert code == 0
        assert out.exists() and out.stat().st_size > 0

    def test_robust_bands_excludes_smooth(self, csv_path, tmp_path):
        out = tmp_path / "x.png"
        with pytest.raises(SystemExit):
            main(
                [
                    "series",
                    str(csv_path),
                    "a",
                    "--smooth",
                    "2",
                    "--robust-bands",
                    "2",
                    "--output",
                    str(out),
                ]
            )


class TestHeatmap:
    def test_saves_output(self, csv_path, tmp_path):
        out = tmp_path / "heat.png"
        assert main(["heatmap", str(csv_path), "--output", str(out)]) == 0
        assert out.exists() and out.stat().st_size > 0

    def test_column_subset(self, csv_path, tmp_path):
        out = tmp_path / "heat.png"
        code = main(
            ["heatmap", str(csv_path), "--columns", "a,b", "--output", str(out)]
        )
        assert code == 0
        assert out.exists()

    def test_bad_column_returns_1(self, csv_path, tmp_path, capsys):
        out = tmp_path / "heat.png"
        code = main(
            ["heatmap", str(csv_path), "--columns", "nope", "--output", str(out)]
        )
        assert code == 1
        assert "error" in capsys.readouterr().err


class TestBox:
    def test_saves_output(self, csv_path, tmp_path):
        out = tmp_path / "box.png"
        assert main(["box", str(csv_path), "--output", str(out)]) == 0
        assert out.exists() and out.stat().st_size > 0

    def test_column_subset(self, csv_path, tmp_path):
        out = tmp_path / "box.png"
        code = main(["box", str(csv_path), "--columns", "a,b", "--output", str(out)])
        assert code == 0
        assert out.exists()

    def test_bad_column_returns_1(self, csv_path, tmp_path, capsys):
        out = tmp_path / "box.png"
        code = main(["box", str(csv_path), "--columns", "nope", "--output", str(out)])
        assert code == 1
        assert "error" in capsys.readouterr().err


class TestParser:
    def test_no_command_exits(self):
        with pytest.raises(SystemExit):
            main([])

    def test_version(self, capsys):
        import serieskit

        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0
        assert f"serieskit {serieskit.__version__}" in capsys.readouterr().out


class TestOutliers:
    @pytest.fixture
    def outlier_csv(self, tmp_path):
        path = tmp_path / "out.csv"
        rows = "\n".join(f"v,{v}" for v in [*range(10, 110), 200])
        path.write_text(f"name,x\n{rows}\n", encoding="utf-8")
        return path

    def test_iqr_default(self, outlier_csv, capsys):
        assert main(["outliers", str(outlier_csv), "x"]) == 0
        captured = capsys.readouterr()
        assert "200.0" in captured.out
        assert "1 个异常值" in captured.err

    def test_zscore_method(self, outlier_csv, capsys):
        code = main(["outliers", str(outlier_csv), "x", "--method", "zscore"])
        assert code == 0
        assert "200.0" in capsys.readouterr().out

    def test_mad_method(self, outlier_csv, capsys):
        code = main(["outliers", str(outlier_csv), "x", "--method", "mad"])
        assert code == 0
        captured = capsys.readouterr()
        assert "200.0" in captured.out
        assert "1 个异常值" in captured.err

    def test_mad_custom_threshold(self, outlier_csv, capsys):
        code = main(
            ["outliers", str(outlier_csv), "x", "--method", "mad", "--threshold", "100"]
        )
        assert code == 0
        assert "0 个异常值" in capsys.readouterr().err

    def test_clean_column_reports_none(self, csv_path, capsys):
        assert main(["outliers", str(csv_path), "a"]) == 0
        captured = capsys.readouterr()
        assert captured.out.strip() == ""
        assert "0 个异常值" in captured.err

    def test_non_numeric_column_returns_1(self, outlier_csv, capsys):
        assert main(["outliers", str(outlier_csv), "name"]) == 1
        assert "error" in capsys.readouterr().err

    def test_plot_option_saves_figure(self, outlier_csv, tmp_path):
        out = tmp_path / "out.png"
        code = main(["outliers", str(outlier_csv), "x", "--plot", str(out)])
        assert code == 0
        assert out.exists() and out.stat().st_size > 0


class TestTransform:
    def test_zscore_default(self, csv_path, capsys):
        assert main(["transform", str(csv_path), "a"]) == 0
        lines = capsys.readouterr().out.strip().splitlines()
        assert len(lines) == 3
        assert float(lines[0]) == pytest.approx(-1.0)

    def test_pct_change_first_line_is_nan(self, csv_path, capsys):
        assert main(["transform", str(csv_path), "a", "--method", "pct_change"]) == 0
        lines = capsys.readouterr().out.strip().splitlines()
        assert lines[0] == "nan"
        assert float(lines[1]) == pytest.approx(1.0)

    def test_minmax_in_unit_range(self, csv_path, capsys):
        assert main(["transform", str(csv_path), "b", "--method", "minmax"]) == 0
        values = [float(x) for x in capsys.readouterr().out.strip().splitlines()]
        assert min(values) == pytest.approx(0.0)
        assert max(values) == pytest.approx(1.0)

    def test_log_returns(self, csv_path, capsys):
        assert main(["transform", str(csv_path), "a", "--method", "log_returns"]) == 0
        lines = capsys.readouterr().out.strip().splitlines()
        assert lines[0] == "nan"
        assert float(lines[1]) == pytest.approx(0.693147, rel=1e-5)

    def test_bad_method_rejected(self, csv_path):
        with pytest.raises(SystemExit):
            main(["transform", str(csv_path), "a", "--method", "nope"])

    def test_robust_zscore_method(self, csv_path, capsys):
        assert main(["transform", str(csv_path), "a", "--method", "robust_zscore"]) == 0
        lines = capsys.readouterr().out.strip().splitlines()
        assert len(lines) == 3
        assert float(lines[1]) == pytest.approx(0.0)

    def test_diff_method(self, csv_path, capsys):
        assert main(["transform", str(csv_path), "a", "--method", "diff"]) == 0
        lines = capsys.readouterr().out.strip().splitlines()
        assert lines[0] == "nan"
        assert float(lines[1]) == pytest.approx(1.0)

    def test_moving_average_method(self, csv_path, capsys):
        code = main(
            [
                "transform",
                str(csv_path),
                "a",
                "--method",
                "moving_average",
                "--window",
                "2",
            ]
        )
        assert code == 0
        lines = capsys.readouterr().out.strip().splitlines()
        assert [float(x) for x in lines] == [pytest.approx(1.5), pytest.approx(2.5)]

    def test_moving_median_method(self, csv_path, capsys):
        code = main(
            [
                "transform",
                str(csv_path),
                "a",
                "--method",
                "moving_median",
                "--window",
                "2",
            ]
        )
        assert code == 0
        lines = capsys.readouterr().out.strip().splitlines()
        assert len(lines) == 2

    def test_moving_std_bad_window_returns_1(self, csv_path, capsys):
        code = main(
            ["transform", str(csv_path), "a", "--method", "moving_std", "--window", "9"]
        )
        assert code == 1
        assert "error" in capsys.readouterr().err

    def test_non_numeric_column_returns_1(self, csv_path, capsys):
        assert main(["transform", str(csv_path), "name"]) == 1
        assert "error" in capsys.readouterr().err


class TestCorr:
    def test_default_pearson(self, csv_path, capsys):
        assert main(["corr", str(csv_path)]) == 0
        out = capsys.readouterr().out
        assert "1.0000" in out
        assert "a" in out and "b" in out
        assert "name" not in out

    def test_column_subset(self, csv_path, capsys):
        assert main(["corr", str(csv_path), "--columns", "a"]) == 0
        out = capsys.readouterr().out
        assert "b" not in out.splitlines()[0]

    def test_spearman_method(self, csv_path, capsys):
        assert main(["corr", str(csv_path), "--method", "spearman"]) == 0
        assert "1.0000" in capsys.readouterr().out

    def test_bad_column_returns_1(self, csv_path, capsys):
        assert main(["corr", str(csv_path), "--columns", "nope"]) == 1
        assert "error" in capsys.readouterr().err

    def test_output_writes_csv(self, csv_path, tmp_path, capsys):
        import pandas as pd

        out = tmp_path / "corr.csv"
        assert main(["corr", str(csv_path), "--output", str(out)]) == 0
        assert str(out) in capsys.readouterr().out
        saved = pd.read_csv(out, index_col=0)
        assert saved.loc["a", "b"] == pytest.approx(1.0)


class TestAutocorr:
    def test_default_lag(self, csv_path, capsys):
        assert main(["autocorr", str(csv_path), "a"]) == 0
        assert float(capsys.readouterr().out.strip()) == pytest.approx(1.0)

    def test_explicit_lag(self, csv_path, capsys):
        assert main(["autocorr", str(csv_path), "a", "--lag", "2"]) == 0
        float(capsys.readouterr().out.strip())

    def test_invalid_lag_returns_1(self, csv_path, capsys):
        assert main(["autocorr", str(csv_path), "a", "--lag", "0"]) == 1
        assert "error" in capsys.readouterr().err


class TestPercentile:
    def test_default_median(self, csv_path, capsys):
        assert main(["percentile", str(csv_path), "a"]) == 0
        assert float(capsys.readouterr().out.strip()) == pytest.approx(2.0)

    def test_explicit_q(self, csv_path, capsys):
        assert main(["percentile", str(csv_path), "b", "--q", "75"]) == 0
        assert float(capsys.readouterr().out.strip()) == pytest.approx(3.0)

    def test_out_of_range_returns_1(self, csv_path, capsys):
        assert main(["percentile", str(csv_path), "a", "--q", "101"]) == 1
        assert "error" in capsys.readouterr().err


class TestReport:
    def test_generates_all_artifacts(self, csv_path, tmp_path, capsys):
        outdir = tmp_path / "rep"
        assert main(["report", str(csv_path), "--outdir", str(outdir)]) == 0
        assert (outdir / "summary.csv").exists()
        assert (outdir / "a_hist.png").exists()
        assert (outdir / "b_hist.png").exists()
        assert (outdir / "heatmap.png").exists()
        assert (outdir / "box.png").exists()
        assert "报告已生成" in capsys.readouterr().out

    def test_single_numeric_column_skips_multi_plots(self, tmp_path):
        csv = tmp_path / "one.csv"
        csv.write_text("x,s\n1,a\n2,b\n", encoding="utf-8")
        outdir = tmp_path / "rep1"
        assert main(["report", str(csv), "--outdir", str(outdir)]) == 0
        assert (outdir / "x_hist.png").exists()
        assert not (outdir / "heatmap.png").exists()

    def test_no_numeric_column_returns_1(self, tmp_path, capsys):
        csv = tmp_path / "bad.csv"
        csv.write_text("s\na\nb\n", encoding="utf-8")
        assert main(["report", str(csv), "--outdir", str(tmp_path / "r")]) == 1
        assert "error" in capsys.readouterr().err

    def test_group_by_writes_summary_by(self, tmp_path, capsys):
        import pandas as pd

        csv = tmp_path / "g.csv"
        csv.write_text("g,x\na,1\na,3\nb,10\nb,20\n", encoding="utf-8")
        outdir = tmp_path / "rep"
        assert main(["report", str(csv), "--by", "g", "--outdir", str(outdir)]) == 0
        assert not (outdir / "summary.csv").exists()
        saved = pd.read_csv(outdir / "summary_by.csv", index_col=[0, 1])
        assert saved.loc[("a", "x"), "mean"] == pytest.approx(2.0)
        assert (outdir / "x_hist.png").exists()


class TestMad:
    def test_value(self, csv_path, capsys):
        assert main(["mad", str(csv_path), "a"]) == 0
        assert float(capsys.readouterr().out.strip()) == pytest.approx(1.0)

    def test_non_numeric_column_returns_1(self, csv_path, capsys):
        assert main(["mad", str(csv_path), "name"]) == 1
        assert "error" in capsys.readouterr().err


def test_summary_json_uses_null_for_non_finite_values(tmp_path, capsys):
    csv = tmp_path / "single.csv"
    csv.write_text("value\n5\n", encoding="utf-8")
    assert main(["summary", str(csv), "--json"]) == 0
    payload = json.loads(
        capsys.readouterr().out,
        parse_constant=lambda token: pytest.fail(f"Non-standard JSON: {token}"),
    )
    assert payload["value"]["skew"] is None
    assert payload["value"]["kurtosis"] is None
    assert payload["value"]["mean"] == 5.0


@pytest.mark.parametrize("method", ["zscore", "mad"])
def test_outliers_honors_zero_threshold(tmp_path, capsys, method):
    csv = tmp_path / "values.csv"
    csv.write_text("value\n1\n2\n3\n", encoding="utf-8")
    assert (
        main(["outliers", str(csv), "value", "--method", method, "--threshold", "0"])
        == 0
    )
    assert capsys.readouterr().out.splitlines() == ["1.0", "3.0"]


@pytest.mark.parametrize("option", ["--smooth", "--bands", "--robust-bands"])
def test_series_rejects_zero_window(csv_path, tmp_path, capsys, option):
    out = tmp_path / "invalid.png"
    assert main(["series", str(csv_path), "a", option, "0", "--output", str(out)]) == 1
    assert "error:" in capsys.readouterr().err
    assert not out.exists()


def test_saved_cli_figure_is_closed(csv_path, tmp_path):
    from matplotlib import pyplot as plt

    before = set(plt.get_fignums())
    assert (
        main(["hist", str(csv_path), "a", "--output", str(tmp_path / "hist.png")]) == 0
    )
    assert set(plt.get_fignums()) == before


def test_module_entrypoint():
    import subprocess
    import sys

    from serieskit import __version__

    result = subprocess.run(
        [sys.executable, "-m", "serieskit", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == f"serieskit {__version__}"
