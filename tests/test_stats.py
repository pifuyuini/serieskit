import numpy as np
import pytest

from kimi_space.stats import (
    SummaryStats,
    autocorr,
    geometric_mean,
    harmonic_mean,
    median_abs_deviation,
    moving_average,
    moving_std,
    summarize,
    weighted_mean,
    zscores,
)


class TestSummarize:
    def test_basic_sequence(self):
        stats = summarize(range(1, 11))
        assert isinstance(stats, SummaryStats)
        assert stats.count == 10
        assert stats.mean == pytest.approx(5.5)
        assert stats.median == pytest.approx(5.5)
        assert stats.min == 1.0
        assert stats.max == 10.0
        assert stats.q25 == pytest.approx(3.25)
        assert stats.q75 == pytest.approx(7.75)
        assert stats.std == pytest.approx(np.std(np.arange(1, 11), ddof=1))
        assert stats.skew == pytest.approx(0.0, abs=1e-12)

    def test_nan_values_are_dropped(self):
        stats = summarize([1.0, 2.0, float("nan"), 3.0])
        assert stats.count == 3
        assert stats.mean == pytest.approx(2.0)

    def test_single_value(self):
        stats = summarize([42.0])
        assert stats.count == 1
        assert stats.mean == 42.0
        assert stats.std == 0.0

    def test_right_skewed_data_has_positive_skew(self):
        stats = summarize([1, 1, 1, 2, 2, 3, 4, 5, 10, 100])
        assert stats.skew > 0

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="为空"):
            summarize([])

    def test_all_nan_raises(self):
        with pytest.raises(ValueError, match="为空"):
            summarize([float("nan"), float("nan")])

    def test_to_dict_keys(self):
        d = summarize([1, 2, 3]).to_dict()
        assert set(d) == {
            "count",
            "mean",
            "std",
            "min",
            "q25",
            "median",
            "q75",
            "max",
            "skew",
            "kurtosis",
            "mad",
        }


class TestMovingAverage:
    def test_window_two(self):
        result = moving_average([1, 2, 3, 4], 2)
        np.testing.assert_allclose(result, [1.5, 2.5, 3.5])

    def test_window_one_is_identity(self):
        result = moving_average([5.0, -1.0, 3.0], 1)
        np.testing.assert_allclose(result, [5.0, -1.0, 3.0])

    def test_window_equal_to_length(self):
        result = moving_average([1, 2, 3], 3)
        np.testing.assert_allclose(result, [2.0])

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError, match=">= 1"):
            moving_average([1, 2, 3], 0)

    def test_window_larger_than_data_raises(self):
        with pytest.raises(ValueError, match="超过数据长度"):
            moving_average([1, 2], 3)


class TestMovingStd:
    def test_known_values(self):
        result = moving_std([1, 2, 3, 4], 2)
        np.testing.assert_allclose(result, [np.sqrt(0.5)] * 3)

    def test_window_full_length(self):
        result = moving_std([1, 2, 3], 3)
        np.testing.assert_allclose(result, [1.0])

    def test_window_one_raises(self):
        with pytest.raises(ValueError, match=">= 2"):
            moving_std([1, 2, 3], 1)

    def test_window_larger_than_data_raises(self):
        with pytest.raises(ValueError, match="超过数据长度"):
            moving_std([1, 2], 3)


class TestZscores:
    def test_normalized_output(self):
        result = zscores([1, 2, 3, 4, 5])
        assert np.mean(result) == pytest.approx(0.0)
        assert np.std(result, ddof=1) == pytest.approx(1.0)

    def test_known_values(self):
        result = zscores([1, 2, 3])
        expected = (np.array([1, 2, 3]) - 2.0) / np.std([1, 2, 3], ddof=1)
        np.testing.assert_allclose(result, expected)

    def test_constant_series_returns_zeros(self):
        np.testing.assert_array_equal(zscores([7, 7, 7]), np.zeros(3))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="为空"):
            zscores([])


class TestMeans:
    def test_weighted_mean(self):
        assert weighted_mean([80, 90, 60], [3, 2, 5]) == pytest.approx(72.0)

    def test_weighted_mean_equal_weights_is_mean(self):
        assert weighted_mean([1, 2, 3], [1, 1, 1]) == pytest.approx(2.0)

    def test_weighted_mean_length_mismatch(self):
        with pytest.raises(ValueError, match="长度不一致"):
            weighted_mean([1, 2], [1])

    def test_weighted_mean_negative_weight(self):
        with pytest.raises(ValueError, match="权重不能为负"):
            weighted_mean([1, 2], [1, -1])

    def test_weighted_mean_zero_total(self):
        with pytest.raises(ValueError, match="权重和为 0"):
            weighted_mean([1, 2], [0, 0])

    def test_weighted_mean_empty(self):
        with pytest.raises(ValueError, match="为空"):
            weighted_mean([], [])

    def test_geometric_mean(self):
        assert geometric_mean([1, 4]) == pytest.approx(2.0)

    def test_geometric_mean_rejects_non_positive(self):
        with pytest.raises(ValueError, match="正数"):
            geometric_mean([1, 0, 2])

    def test_geometric_mean_empty(self):
        with pytest.raises(ValueError, match="为空"):
            geometric_mean([])

    def test_harmonic_mean(self):
        assert harmonic_mean([1, 2, 4]) == pytest.approx(12 / 7)

    def test_harmonic_mean_rejects_negative(self):
        with pytest.raises(ValueError, match="正数"):
            harmonic_mean([1, -1])

    def test_harmonic_mean_empty(self):
        with pytest.raises(ValueError, match="为空"):
            harmonic_mean([])

    def test_mean_ordering_inequality(self):
        data = [1.0, 2.0, 4.0, 8.0]
        h = harmonic_mean(data)
        g = geometric_mean(data)
        a = float(np.mean(data))
        assert h <= g <= a


class TestAutocorr:
    def test_linear_trend_is_one(self):
        assert autocorr([1, 2, 3, 4, 5]) == pytest.approx(1.0)

    def test_alternating_lag1_is_negative_one(self):
        data = [1.0, -1.0] * 5
        assert autocorr(data, lag=1) == pytest.approx(-1.0)

    def test_alternating_lag2_is_positive_one(self):
        data = [1.0, -1.0] * 5
        assert autocorr(data, lag=2) == pytest.approx(1.0)

    def test_noise_autocorr_is_small(self):
        rng = np.random.default_rng(0)
        r = autocorr(rng.normal(size=2000))
        assert abs(r) < 0.1

    def test_constant_series_is_nan(self):
        assert np.isnan(autocorr([7.0, 7.0, 7.0]))

    def test_invalid_lag_raises(self):
        with pytest.raises(ValueError, match="lag"):
            autocorr([1, 2, 3], lag=0)
        with pytest.raises(ValueError, match="lag"):
            autocorr([1, 2, 3], lag=3)


class TestMedianAbsDeviation:
    def test_known_value(self):
        assert median_abs_deviation([1, 2, 3, 4, 100]) == pytest.approx(1.0)

    def test_robust_to_outlier(self):
        clean = [1.0, 2.0, 3.0, 4.0, 5.0]
        dirty = [*clean, 1000.0]
        assert median_abs_deviation(dirty) < 2.5  # MAD 几乎不动
        assert np.std(dirty) > 100  # 而 std 被拉爆

    def test_constant_series_is_zero(self):
        assert median_abs_deviation([7.0, 7.0, 7.0]) == 0.0

    def test_nan_dropped(self):
        assert median_abs_deviation([1.0, 2.0, float("nan"), 3.0]) == pytest.approx(1.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="为空"):
            median_abs_deviation([])
