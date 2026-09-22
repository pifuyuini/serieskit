"""基于 hypothesis 的性质测试：对随机输入校验统计函数的不变量。"""

import math

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from serieskit.outliers import iqr_mask
from serieskit.stats import moving_average, summarize, zscores
from serieskit.transform import minmax_normalize

finite_floats = st.floats(
    allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6
)
series = st.lists(finite_floats, min_size=1, max_size=200)


class TestSummarizeProperties:
    @given(series)
    def test_quantiles_are_ordered(self, data):
        s = summarize(data)
        assert s.min <= s.q25 <= s.median <= s.q75 <= s.max

    @given(series)
    def test_count_and_bounds(self, data):
        s = summarize(data)
        assert s.count == len(data)
        assert s.min == min(data)
        assert s.max == max(data)

    @given(series)
    def test_mean_within_bounds(self, data):
        s = summarize(data)
        # 浮点求和可能有 1 ulp 的舍入，允许相对容差
        assert s.min <= s.mean or math.isclose(s.mean, s.min, rel_tol=1e-14)
        assert s.mean <= s.max or math.isclose(s.mean, s.max, rel_tol=1e-14)


class TestMovingAverageProperties:
    @given(series, st.integers(min_value=1, max_value=200))
    def test_output_length(self, data, window):
        if window > len(data):
            return
        result = moving_average(data, window)
        assert len(result) == len(data) - window + 1

    @given(series, st.integers(min_value=1, max_value=200))
    def test_output_within_input_range(self, data, window):
        if window > len(data):
            return
        result = moving_average(data, window)
        assert np.min(result) >= min(data) - 1e-9
        assert np.max(result) <= max(data) + 1e-9


class TestZscoresProperties:
    @given(st.lists(finite_floats, min_size=2, max_size=200))
    @settings(max_examples=200)
    def test_normalized_when_not_constant(self, data):
        if len(set(data)) == 1:
            return
        arr = np.asarray(data)
        # 离散度远小于量级时（如 1e6 与 1e6-1e-10），float64 无法表达该尺度差异
        assume(np.std(arr, ddof=1) > np.abs(arr).max() * 1e-9)
        z = zscores(data)
        assert abs(float(np.mean(z))) < 1e-6


class TestMinmaxNormalizeProperties:
    @given(series)
    def test_within_default_range(self, data):
        result = minmax_normalize(data)
        assert np.min(result) >= -1e-12
        assert np.max(result) <= 1.0 + 1e-12


class TestIqrMaskProperties:
    @given(series)
    def test_mask_length_matches_input(self, data):
        assert len(iqr_mask(data)) == len(data)

    @given(series)
    def test_median_is_never_outlier(self, data):
        s = summarize(data)
        mask = iqr_mask([*data, s.median])
        assert mask[-1]
