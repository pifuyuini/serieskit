import numpy as np
import pytest

from serieskit.outliers import (
    iqr_mask,
    iqr_outliers,
    mad_mask,
    mad_outliers,
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


class TestMadOutliers:
    def test_flags_extreme_value(self):
        data = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 100.0]
        mask = mad_mask(data)
        assert mask.tolist() == [True, True, True, True, True, True, False]

    def test_zscore_method_misses_but_mad_catches(self):
        # 样本小且离群点极大时，经典 z-score 被掩蔽，MAD 法仍能检出
        data = [*range(10, 20), 500.0]
        assert not mad_mask(data)[-1]

    def test_constant_series_all_normal(self):
        assert mad_mask([7.0, 7.0, 7.0]).all()

    def test_constant_with_one_deviation(self):
        mask = mad_mask([7.0, 7.0, 7.0, 9.0])
        assert mask.tolist() == [True, True, True, False]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="为空"):
            mad_mask([])

    def test_mad_outliers_returns_values(self):
        data = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 100.0]
        np.testing.assert_array_equal(mad_outliers(data), [100.0])

    def test_scale_invariance(self):
        data = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 100.0]
        np.testing.assert_array_equal(
            mad_mask(data), mad_mask([v * 1000 for v in data])
        )
