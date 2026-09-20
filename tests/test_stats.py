import numpy as np
import pytest

from kimi_space.stats import SummaryStats, moving_average, summarize, zscores


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
