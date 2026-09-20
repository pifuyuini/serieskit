import numpy as np
import pytest

from kimi_space.outliers import (
    iqr_mask,
    iqr_outliers,
    zscore_mask,
    zscore_outliers,
)


class TestIqrMask:
    def test_flags_extreme_value(self):
        data = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 100.0]
        mask = iqr_mask(data)
        assert mask.tolist() == [True, True, True, True, True, True, False]

    def test_clean_data_all_true(self):
        assert iqr_mask([1.0, 2.0, 3.0, 4.0]).all()

    def test_custom_k(self):
        data = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 10.0]
        assert not iqr_mask(data, k=1.0)[-1]
        assert iqr_mask(data, k=5.0)[-1]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="为空"):
            iqr_mask([])


class TestZscoreMask:
    def test_flags_extreme_value(self):
        rng = np.random.default_rng(1)
        data = np.concatenate([rng.normal(0, 1, 500), [20.0]])
        mask = zscore_mask(data, threshold=3.0)
        assert not mask[-1]
        assert mask[:-1].all()

    def test_constant_series_all_true(self):
        assert zscore_mask([7.0, 7.0, 7.0]).all()

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="为空"):
            zscore_mask([])


class TestOutlierValues:
    def test_iqr_outliers_returns_values(self):
        data = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 100.0]
        np.testing.assert_array_equal(iqr_outliers(data), [100.0])

    def test_zscore_outliers_returns_values(self):
        rng = np.random.default_rng(2)
        data = np.concatenate([rng.normal(0, 1, 500), [-15.0, 15.0]])
        result = zscore_outliers(data, threshold=3.0)
        assert sorted(result.tolist()) == [-15.0, 15.0]

    def test_no_outliers_returns_empty(self):
        assert iqr_outliers([1.0, 2.0, 3.0]).size == 0
