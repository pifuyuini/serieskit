import pandas as pd
import pytest

from kimi_space.io import load_csv, numeric_summary

CSV_TEXT = "name,a,b\nx,1,1.5\ny,2,2.5\nz,3,3.5\n"


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(CSV_TEXT, encoding="utf-8")
    return path


class TestLoadCsv:
    def test_plain_load(self, csv_path):
        df = load_csv(csv_path)
        assert list(df.columns) == ["name", "a", "b"]
        assert len(df) == 3
        assert df["a"].tolist() == [1, 2, 3]

    def test_required_columns_ok(self, csv_path):
        df = load_csv(csv_path, required_columns=["a"])
        assert "a" in df.columns

    def test_required_columns_missing(self, csv_path):
        with pytest.raises(ValueError, match="缺少必需列"):
            load_csv(csv_path, required_columns=["a", "nope"])

    def test_numeric_columns_coercion(self, csv_path):
        df = load_csv(csv_path, numeric_columns=["b"])
        assert df["b"].dtype == float

    def test_numeric_column_not_found(self, csv_path):
        with pytest.raises(ValueError, match="缺少数值列"):
            load_csv(csv_path, numeric_columns=["nope"])

    def test_numeric_column_unparseable(self, csv_path):
        with pytest.raises(ValueError, match="Unable to parse"):
            load_csv(csv_path, numeric_columns=["name"])

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_csv(tmp_path / "absent.csv")

    def test_stdin_dash(self, monkeypatch):
        import io

        monkeypatch.setattr("sys.stdin", io.StringIO(CSV_TEXT))
        df = load_csv("-")
        assert list(df.columns) == ["name", "a", "b"]
        assert len(df) == 3


class TestNumericSummary:
    def test_summarizes_numeric_columns(self):
        df = pd.DataFrame(
            {"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0], "s": list("xyz")}
        )
        summary = numeric_summary(df)
        assert list(summary.index) == ["a", "b"]
        assert summary.loc["a", "mean"] == pytest.approx(2.0)
        assert summary.loc["b", "max"] == pytest.approx(30.0)
        assert summary.loc["a", "count"] == 3

    def test_no_numeric_columns_raises(self):
        df = pd.DataFrame({"s": list("xyz")})
        with pytest.raises(ValueError, match="不含数值列"):
            numeric_summary(df)
