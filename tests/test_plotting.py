import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from kimi_space.plotting import (
    apply_style,
    plot_box,
    plot_correlation_heatmap,
    plot_histogram,
    plot_outliers,
    plot_series,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


class TestApplyStyle:
    def test_sets_rcparams(self):
        apply_style()
        assert plt.rcParams["axes.spines.top"] is False
        assert plt.rcParams["axes.grid"] is True


class TestPlotSeries:
    def test_returns_axes_with_one_line(self):
        ax = plot_series([1, 2, 3, 4])
        assert isinstance(ax, Axes)
        assert len(ax.lines) == 1

    def test_smooth_window_adds_overlay_line(self):
        ax = plot_series([1, 2, 3, 4, 5], smooth_window=3)
        assert len(ax.lines) == 2
        ma_line = ax.lines[1]
        np.testing.assert_allclose(ma_line.get_ydata(), [2.0, 3.0, 4.0])

    def test_uses_provided_axes(self):
        fig, own_ax = plt.subplots()
        ax = plot_series([1, 2], ax=own_ax)
        assert ax is own_ax

    def test_title_is_set(self):
        ax = plot_series([1, 2], title="demo")
        assert ax.get_title() == "demo"

    def test_custom_x_axis(self):
        ax = plot_series([10, 20, 30], x=[100, 200, 300])
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [100, 200, 300])

    def test_custom_x_aligned_with_smoothing(self):
        ax = plot_series([1, 2, 3, 4], x=[10, 20, 30, 40], smooth_window=2)
        np.testing.assert_array_equal(ax.lines[1].get_xdata(), [20, 30, 40])

    def test_x_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="长度不一致"):
            plot_series([1, 2, 3], x=[1, 2])


class TestPlotHistogram:
    def test_returns_axes_with_reference_lines(self):
        rng = np.random.default_rng(0)
        ax = plot_histogram(rng.normal(size=200))
        assert isinstance(ax, Axes)
        assert len(ax.patches) == 20
        assert len(ax.lines) == 2  # mean + median

    def test_custom_bins(self):
        ax = plot_histogram(np.arange(100), bins=10)
        assert len(ax.patches) == 10


class TestPlotOutliers:
    def test_marks_outliers_with_scatter(self):
        data = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 100.0]
        ax = plot_outliers(data)
        assert len(ax.collections) == 1
        offsets = ax.collections[0].get_offsets()
        assert offsets.shape == (1, 2)
        assert offsets[0, 0] == 6  # 异常值的索引
        assert offsets[0, 1] == 100.0

    def test_clean_series_has_no_scatter(self):
        ax = plot_outliers([1.0, 2.0, 3.0, 4.0])
        assert len(ax.collections) == 0

    def test_returns_axes_with_line(self):
        ax = plot_outliers([1.0, 2.0, 100.0, 2.0, 3.0, 4.0, 5.0], k=1.5)
        assert len(ax.lines) == 1
        assert ax.get_legend() is not None

    def test_title_is_set(self):
        ax = plot_outliers([1.0, 2.0, 3.0], title="demo")
        assert ax.get_title() == "demo"

    def test_explicit_mask(self):
        ax = plot_outliers([1.0, 2.0, 3.0], mask=[True, False, True])
        assert len(ax.collections) == 1
        offsets = ax.collections[0].get_offsets()
        assert offsets[0, 0] == 1
        assert offsets[0, 1] == 2.0

    def test_mask_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="长度不一致"):
            plot_outliers([1.0, 2.0], mask=[True])


class TestPlotBox:
    def test_labels_match_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 3, 4, 5], "s": list("wxyz")})
        ax = plot_box(df)
        assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b"]

    def test_column_subset(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1], "c": [1, 1, 2]})
        ax = plot_box(df, columns=["a", "c"])
        assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "c"]

    def test_title_is_set(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        ax = plot_box(df, title="box")
        assert ax.get_title() == "box"

    def test_saves_to_file(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8]})
        out = tmp_path / "box.png"
        plot_box(df).figure.savefig(out)
        assert out.exists() and out.stat().st_size > 0


class TestPlotCorrelationHeatmap:
    def test_draws_mesh_and_text(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8], "c": [4, 3, 2, 1]})
        ax = plot_correlation_heatmap(df)
        assert len(ax.images) == 1
        assert len(ax.texts) == 9
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels == ["a", "b", "c"]

    def test_perfect_correlation_annotated(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8]})
        ax = plot_correlation_heatmap(df)
        values = {t.get_text() for t in ax.texts}
        assert "1.00" in values

    def test_column_subset(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1], "c": [1, 1, 2]})
        ax = plot_correlation_heatmap(df, columns=["a", "b"])
        assert len(ax.texts) == 4

    def test_figure_saves_to_file(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8]})
        ax = plot_correlation_heatmap(df)
        out = tmp_path / "heatmap.png"
        ax.figure.savefig(out)
        assert out.exists() and out.stat().st_size > 0
