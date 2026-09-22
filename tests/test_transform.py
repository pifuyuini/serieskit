import numpy as np
import pytest

from serieskit.transform import (
    diff,
    log_returns,
    minmax_normalize,
    pct_change,
    robust_zscores,
)


class TestMinmaxNormalize:
    def test_default_range(self):
        result = minmax_normalize([10, 20, 30])
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_custom_range(self):
        result = minmax_normalize([0, 5, 10], feature_range=(-1.0, 1.0))
        np.testing.assert_allclose(result, [-1.0, 0.0, 1.0])

    def test_constant_maps_to_midpoint(self):
        np.testing.assert_allclose(minmax_normalize([7, 7, 7]), [0.5, 0.5, 0.5])

    def test_invalid_range_raises(self):
        with pytest.raises(ValueError, match="下界必须小于上界"):
            minmax_normalize([1, 2], feature_range=(1.0, 1.0))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="为空"):
            minmax_normalize([])


class TestPctChange:
    def test_known_values(self):
        result = pct_change([100.0, 110.0, 99.0])
        assert np.isnan(result[0])
        np.testing.assert_allclose(result[1:], [0.1, -0.1])

    def test_zero_predecessor_is_nan(self):
        result = pct_change([0.0, 5.0, 10.0])
        assert np.isnan(result[1])
        assert result[2] == pytest.approx(1.0)

    def test_length_preserved(self):
        assert len(pct_change([1.0, 2.0, 3.0, 4.0])) == 4

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="为空"):
            pct_change([])


class TestLogReturns:
    def test_known_values(self):
        result = log_returns([1.0, np.e, np.e**2])
        assert np.isnan(result[0])
        np.testing.assert_allclose(result[1:], [1.0, 1.0])

    def test_negative_ratio_is_nan(self):
        result = log_returns([1.0, -2.0])
        assert np.isnan(result[1])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="为空"):
            log_returns([])


class TestDiff:
    def test_known_values(self):
        result = diff([1.0, 2.0, 4.0, 8.0])
        assert np.isnan(result[0])
        np.testing.assert_allclose(result[1:], [1.0, 2.0, 4.0])

    def test_length_preserved(self):
        assert len(diff([5.0, 1.0, 3.0])) == 3

    def test_negative_deltas(self):
        result = diff([10.0, 5.0])
        assert result[1] == pytest.approx(-5.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="为空"):
            diff([])


class TestRobustZscores:
    def test_outlier_dominates(self):
        result = robust_zscores([1.0, 2.0, 3.0, 4.0, 100.0])
        assert result[-1] > 3.5
        assert np.abs(result[:-1]).max() < 3.5

    def test_matches_mad_mask_semantics(self):
        rng = np.random.default_rng(5)
        data = np.concatenate([rng.normal(0, 1, 100), [20.0]])
        assert (np.abs(robust_zscores(data)) > 3.5).sum() == 1

    def test_constant_series_returns_zeros(self):
        np.testing.assert_array_equal(robust_zscores([7, 7, 7]), np.zeros(3))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="为空"):
            robust_zscores([])
