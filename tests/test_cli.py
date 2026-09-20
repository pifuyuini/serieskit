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
