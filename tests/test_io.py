import pandas as pd
import pytest

from kimi_space.io import load_csv, numeric_summary, numeric_summary_by

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

    def test_parse_dates(self, tmp_path):
        path = tmp_path / "ts.csv"
        path.write_text("date,v\n2026-01-01,1\n2026-01-02,2\n", encoding="utf-8")
        df = load_csv(path, parse_dates=["date"])
        assert pd.api.types.is_datetime64_any_dtype(df["date"])
        assert df["date"].iloc[0].year == 2026


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


class TestNumericSummaryBy:
    def test_groups_rows(self):
        df = pd.DataFrame({"g": ["a", "a", "b", "b"], "x": [1.0, 3.0, 10.0, 20.0]})
        result = numeric_summary_by(df, "g")
        assert result.loc[("a", "x"), "mean"] == pytest.approx(2.0)
        assert result.loc[("b", "x"), "mean"] == pytest.approx(15.0)
        assert result.loc[("a", "x"), "count"] == 2

    def test_numeric_group_column_excluded(self):
        df = pd.DataFrame({"g": [0, 0, 1, 1], "x": [1.0, 2.0, 3.0, 5.0]})
        result = numeric_summary_by(df, "g")
        assert "g" not in result.index.get_level_values("column")

    def test_missing_group_column_raises(self):
        df = pd.DataFrame({"x": [1.0]})
        with pytest.raises(ValueError, match="分组列不存在"):
            numeric_summary_by(df, "nope")

    def test_multiple_keys(self):
        df = pd.DataFrame(
            {
                "g1": ["a", "a", "a", "b", "b", "b"],
                "g2": ["u", "u", "v", "v", "u", "u"],
                "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        result = numeric_summary_by(df, ["g1", "g2"])
        assert result.index.names == ["g1", "g2", "column"]
        assert result.loc[("a", "u", "x"), "mean"] == pytest.approx(1.5)
        assert result.loc[("b", "u", "x"), "count"] == 2

    def test_multiple_keys_missing_column(self):
        df = pd.DataFrame({"g": ["a"], "x": [1.0]})
        with pytest.raises(ValueError, match="分组列不存在"):
            numeric_summary_by(df, ["g", "nope"])
