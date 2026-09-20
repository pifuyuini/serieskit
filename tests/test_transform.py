import numpy as np
import pytest

from kimi_space.transform import log_returns, minmax_normalize, pct_change


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
