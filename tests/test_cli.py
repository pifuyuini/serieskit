import json

import pytest

from kimi_space.cli import main

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


class TestParser:
    def test_no_command_exits(self):
        with pytest.raises(SystemExit):
            main([])

    def test_version(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0
        assert "kimi-space 0.1.0" in capsys.readouterr().out


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

    def test_clean_column_reports_none(self, csv_path, capsys):
        assert main(["outliers", str(csv_path), "a"]) == 0
        captured = capsys.readouterr()
        assert captured.out.strip() == ""
        assert "0 个异常值" in captured.err

    def test_non_numeric_column_returns_1(self, outlier_csv, capsys):
        assert main(["outliers", str(outlier_csv), "name"]) == 1
        assert "error" in capsys.readouterr().err


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

    def test_non_numeric_column_returns_1(self, csv_path, capsys):
        assert main(["transform", str(csv_path), "name"]) == 1
        assert "error" in capsys.readouterr().err
