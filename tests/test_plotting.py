import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from kimi_space.plotting import (
    apply_style,
    plot_bands,
    plot_box,
    plot_correlation_heatmap,
    plot_histogram,
    plot_multi_series,
    plot_outliers,
    plot_robust_bands,
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

    def test_cjk_font_fallback_and_minus(self):
        apply_style()
        fonts = plt.rcParams["font.sans-serif"]
        assert fonts[0] == "PingFang SC"
        assert "DejaVu Sans" in fonts
        assert plt.rcParams["axes.unicode_minus"] is False

    def test_cjk_title_renders(self):
        apply_style()
        ax = plot_series([1.0, 2.0, 3.0], title="价格走势")
        assert ax.get_title() == "价格走势"


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

    def test_kde_adds_density_curve(self):
        rng = np.random.default_rng(3)
        ax = plot_histogram(rng.normal(size=300), kde=True)
        # mean/median 参考线 + kde 曲线
        assert len(ax.lines) == 3
        kde_line = ax.lines[-1]
        assert kde_line.get_label() == "kde"
        assert len(kde_line.get_xdata()) == 200

    def test_kde_uses_density_scale(self):
        rng = np.random.default_rng(4)
        ax = plot_histogram(rng.normal(size=300), bins=10, kde=True)
        widths = np.diff([p.get_x() for p in ax.patches[:2]]).item(0)
        heights_sum = sum(p.get_height() for p in ax.patches)
        assert heights_sum * widths == pytest.approx(1.0, rel=1e-6)


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

    def test_mad_method(self):
        data = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 100.0]
        ax = plot_outliers(data, method="mad")
        offsets = ax.collections[0].get_offsets()
        assert offsets[0, 0] == 6
        assert offsets[0, 1] == 100.0

    def test_zscore_method(self):
        rng = np.random.default_rng(1)
        data = np.concatenate([rng.normal(0, 1, 500), [20.0]])
        ax = plot_outliers(data, method="zscore")
        assert len(ax.collections) == 1

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="未知 method"):
            plot_outliers([1.0, 2.0], method="nope")


class TestPlotRobustBands:
    def test_draws_lines_and_band(self):
        rng = np.random.default_rng(0)
        data = 10 + rng.normal(size=100).cumsum() * 0.1
        ax = plot_robust_bands(data, window=10)
        assert len(ax.lines) == 2  # series + median
        assert len(ax.collections) == 1

    def test_band_center_is_median(self):
        ax = plot_robust_bands([1.0, 100.0, 3.0, 4.0, 5.0], window=2)
        np.testing.assert_allclose(ax.lines[1].get_ydata(), [50.5, 51.5, 3.5, 4.5])

    def test_constant_data_band_is_flat(self):
        ax = plot_robust_bands([5.0] * 30, window=5)
        ys = ax.collections[0].get_paths()[0].vertices[:, 1]
        np.testing.assert_allclose(np.abs(ys - 5.0), 0.0, atol=1e-12)

    def test_x_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="长度不一致"):
            plot_robust_bands([1.0, 2.0], window=2, x=[1.0])

    def test_title_is_set(self):
        ax = plot_robust_bands([1.0, 2.0, 3.0], window=2, title="稳健带")
        assert ax.get_title() == "稳健带"


class TestPlotMultiSeries:
    def test_draws_all_series_with_labels(self):
        ax = plot_multi_series({"a": [1.0, 2.0], "b": [2.0, 1.0]})
        assert len(ax.lines) == 2
        assert {line.get_label() for line in ax.lines} == {"a", "b"}
        assert ax.get_legend() is not None

    def test_custom_x(self):
        ax = plot_multi_series({"a": [1.0, 2.0]}, x=[10, 20])
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [10, 20])

    def test_empty_mapping_raises(self):
        with pytest.raises(ValueError, match="为空"):
            plot_multi_series({})

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="长度不一致"):
            plot_multi_series({"a": [1.0, 2.0], "b": [1.0]})

    def test_x_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="长度不一致"):
            plot_multi_series({"a": [1.0, 2.0]}, x=[1.0])

    def test_title_is_set(self):
        ax = plot_multi_series({"a": [1.0, 2.0]}, title="对比")
        assert ax.get_title() == "对比"


class TestPlotBands:
    def test_draws_lines_and_band(self):
        rng = np.random.default_rng(0)
        data = 10 + rng.normal(size=100).cumsum() * 0.1
        ax = plot_bands(data, window=10)
        assert len(ax.lines) == 2  # series + MA
        assert len(ax.collections) == 1  # fill_between 波动带

    def test_band_geometry_on_constant_data(self):
        ax = plot_bands([5.0] * 30, window=5, num_std=2.0)
        poly = ax.collections[0]
        ys = poly.get_paths()[0].vertices[:, 1]
        np.testing.assert_allclose(np.abs(ys - 5.0), 0.0, atol=1e-12)

    def test_custom_x(self):
        ax = plot_bands([1.0, 2.0, 3.0, 4.0], window=2, x=[10, 20, 30, 40])
        np.testing.assert_array_equal(ax.lines[1].get_xdata(), [20, 30, 40])

    def test_x_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="长度不一致"):
            plot_bands([1.0, 2.0], window=2, x=[1.0])

    def test_title_is_set(self):
        ax = plot_bands([1.0, 2.0, 3.0], window=2, title="band")
        assert ax.get_title() == "band"


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
