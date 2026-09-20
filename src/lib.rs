//! kimi-space 数值内核：与 Python 侧统计能力对应的 Rust 实现。

/// 算术平均值；空输入返回 None。
pub fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    Some(values.iter().sum::<f64>() / values.len() as f64)
}

/// 加权平均；空输入、长度不一致、权重为负或权重和为 0 返回 None。
pub fn weighted_mean(values: &[f64], weights: &[f64]) -> Option<f64> {
    if values.is_empty() || values.len() != weights.len() {
        return None;
    }
    if weights.iter().any(|w| *w < 0.0) {
        return None;
    }
    let total: f64 = weights.iter().sum();
    if total == 0.0 {
        return None;
    }
    Some(values.iter().zip(weights).map(|(v, w)| v * w).sum::<f64>() / total)
}

/// 几何平均；空输入或非正值返回 None。
pub fn geometric_mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() || values.iter().any(|v| *v <= 0.0) {
        return None;
    }
    let n = values.len() as f64;
    Some((values.iter().map(|v| v.ln()).sum::<f64>() / n).exp())
}

/// 调和平均；空输入或非正值返回 None。
pub fn harmonic_mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() || values.iter().any(|v| *v <= 0.0) {
        return None;
    }
    let n = values.len() as f64;
    Some(n / values.iter().map(|v| 1.0 / v).sum::<f64>())
}

/// 样本标准差（ddof=1）；元素少于 2 个时返回 None。
pub fn sample_std(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let m = mean(values)?;
    let var = values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (values.len() as f64 - 1.0);
    Some(var.sqrt())
}

/// 等权滑动平均（前缀和 O(n)）；window 为 0 或大于数据长度时返回 None。
pub fn moving_average(values: &[f64], window: usize) -> Option<Vec<f64>> {
    if window == 0 || window > values.len() {
        return None;
    }
    let mut prefix = Vec::with_capacity(values.len() + 1);
    prefix.push(0.0);
    for v in values {
        prefix.push(prefix.last()? + v);
    }
    let w = window as f64;
    Some(
        prefix
            .windows(window + 1)
            .map(|p| (p[window] - p[0]) / w)
            .collect(),
    )
}

/// 基于样本标准差的标准化；常数序列返回全零，空输入返回空向量。
pub fn zscores(values: &[f64]) -> Vec<f64> {
    let (Some(m), Some(s)) = (mean(values), sample_std(values)) else {
        return Vec::new();
    };
    if s == 0.0 {
        return vec![0.0; values.len()];
    }
    values.iter().map(|v| (v - m) / s).collect()
}

/// 线性插值分位数（与 numpy 默认方法一致）；空输入或 q 不在 [0, 100] 返回 None。
pub fn percentile(values: &[f64], q: f64) -> Option<f64> {
    if values.is_empty() || !(0.0..=100.0).contains(&q) {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    Some(percentile_sorted(&sorted, q))
}

fn percentile_sorted(sorted: &[f64], q: f64) -> f64 {
    let pos = (sorted.len() - 1) as f64 * q / 100.0;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

/// IQR 栅栏法：返回布尔向量，true 表示落在 [Q1-k*IQR, Q3+k*IQR] 内的正常值；
/// 空输入返回 None。
pub fn iqr_mask(values: &[f64], k: f64) -> Option<Vec<bool>> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let q1 = percentile_sorted(&sorted, 25.0);
    let q3 = percentile_sorted(&sorted, 75.0);
    let iqr = q3 - q1;
    let lo = q1 - k * iqr;
    let hi = q3 + k * iqr;
    Some(values.iter().map(|v| (lo..=hi).contains(v)).collect())
}

/// 总体中心矩；空输入返回 None。
fn central_moment(values: &[f64], order: u32) -> Option<f64> {
    let m = mean(values)?;
    let n = values.len() as f64;
    Some(
        values
            .iter()
            .map(|v| (v - m).powi(order as i32))
            .sum::<f64>()
            / n,
    )
}

/// 偏度（Fisher，m3 / m2^1.5，与 scipy 默认 bias=True 一致）；常数序列为 NaN。
pub fn skew(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let m2 = central_moment(values, 2)?;
    if m2 == 0.0 {
        return Some(f64::NAN);
    }
    Some(central_moment(values, 3)? / m2.powf(1.5))
}

/// 超额峰度（m4 / m2^2 - 3，正态分布为 0，与 scipy 默认一致）；常数序列为 NaN。
pub fn kurtosis(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let m2 = central_moment(values, 2)?;
    if m2 == 0.0 {
        return Some(f64::NAN);
    }
    Some(central_moment(values, 4)? / m2.powi(2) - 3.0)
}

/// 描述性统计摘要，与 Python 侧 SummaryStats 字段一致（std 为样本标准差）。
#[derive(Debug, Clone, PartialEq)]
pub struct Summary {
    pub count: usize,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub q25: f64,
    pub median: f64,
    pub q75: f64,
    pub max: f64,
    pub skew: f64,
    pub kurtosis: f64,
}

/// 计算描述性统计摘要；空输入返回 None。
pub fn summarize(values: &[f64]) -> Option<Summary> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    Some(Summary {
        count: values.len(),
        mean: mean(values)?,
        std: sample_std(values).unwrap_or(0.0),
        min: sorted[0],
        q25: percentile_sorted(&sorted, 25.0),
        median: percentile_sorted(&sorted, 50.0),
        q75: percentile_sorted(&sorted, 75.0),
        max: sorted[sorted.len() - 1],
        skew: skew(values).unwrap_or(f64::NAN),
        kurtosis: kurtosis(values).unwrap_or(f64::NAN),
    })
}

/// 最小-最大归一化到 [lo, hi]；常数序列映射到区间中点。
/// 空输入或 lo >= hi 返回 None。
pub fn minmax_normalize(values: &[f64], lo: f64, hi: f64) -> Option<Vec<f64>> {
    if values.is_empty() || lo.partial_cmp(&hi) != Some(std::cmp::Ordering::Less) {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let (min, max) = (sorted[0], sorted[sorted.len() - 1]);
    let span = max - min;
    if span == 0.0 {
        return Some(vec![(lo + hi) / 2.0; values.len()]);
    }
    Some(
        values
            .iter()
            .map(|v| (v - min) / span * (hi - lo) + lo)
            .collect(),
    )
}

/// 相邻变化率，长度与输入一致；首元素与前值为 0 处为 NaN。
pub fn pct_change(values: &[f64]) -> Vec<f64> {
    let mut out = vec![f64::NAN; values.len()];
    for i in 1..values.len() {
        if values[i - 1] != 0.0 {
            out[i] = (values[i] - values[i - 1]) / values[i - 1];
        }
    }
    out
}

/// 对数收益 ln(v_t / v_{t-1})，首元素与比值非正处为 NaN。
pub fn log_returns(values: &[f64]) -> Vec<f64> {
    let mut out = vec![f64::NAN; values.len()];
    for i in 1..values.len() {
        let ratio = values[i] / values[i - 1];
        if ratio > 0.0 {
            out[i] = ratio.ln();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-12, "actual={a}, expected={e}");
        }
    }

    #[test]
    fn mean_basic() {
        assert_eq!(mean(&[1.0, 2.0, 3.0, 4.0]), Some(2.5));
    }

    #[test]
    fn mean_empty_is_none() {
        assert_eq!(mean(&[]), None);
    }

    #[test]
    fn sample_std_known_value() {
        let s = sample_std(&[1.0, 2.0, 3.0]).unwrap();
        assert!((s - 1.0).abs() < 1e-12);
    }

    #[test]
    fn sample_std_single_is_none() {
        assert_eq!(sample_std(&[7.0]), None);
    }

    #[test]
    fn moving_average_window_two() {
        let r = moving_average(&[1.0, 2.0, 3.0, 4.0], 2).unwrap();
        assert_close(&r, &[1.5, 2.5, 3.5]);
    }

    #[test]
    fn moving_average_window_one_is_identity() {
        let r = moving_average(&[5.0, -1.0, 3.0], 1).unwrap();
        assert_close(&r, &[5.0, -1.0, 3.0]);
    }

    #[test]
    fn moving_average_invalid_windows() {
        assert_eq!(moving_average(&[1.0, 2.0], 0), None);
        assert_eq!(moving_average(&[1.0, 2.0], 3), None);
    }

    #[test]
    fn zscores_normalizes() {
        let z = zscores(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let m = mean(&z).unwrap();
        assert!(m.abs() < 1e-12);
        assert!((sample_std(&z).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn zscores_constant_series_is_zero() {
        assert_eq!(zscores(&[7.0, 7.0, 7.0]), vec![0.0; 3]);
    }

    #[test]
    fn zscores_empty_is_empty() {
        assert!(zscores(&[]).is_empty());
    }

    #[test]
    fn percentile_quartiles_match_linear_interpolation() {
        let data: Vec<f64> = (1..=10).map(f64::from).collect();
        assert_eq!(percentile(&data, 0.0), Some(1.0));
        assert_eq!(percentile(&data, 25.0), Some(3.25));
        assert_eq!(percentile(&data, 50.0), Some(5.5));
        assert_eq!(percentile(&data, 75.0), Some(7.75));
        assert_eq!(percentile(&data, 100.0), Some(10.0));
    }

    #[test]
    fn percentile_unsorted_input() {
        assert_eq!(percentile(&[3.0, 1.0, 2.0], 50.0), Some(2.0));
    }

    #[test]
    fn percentile_invalid() {
        assert_eq!(percentile(&[], 50.0), None);
        assert_eq!(percentile(&[1.0], -1.0), None);
        assert_eq!(percentile(&[1.0], 101.0), None);
    }

    #[test]
    fn skew_symmetric_is_zero() {
        let data: Vec<f64> = (1..=10).map(f64::from).collect();
        assert!(skew(&data).unwrap().abs() < 1e-12);
    }

    #[test]
    fn skew_right_tail_is_positive() {
        let data = [1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 10.0, 100.0];
        assert!(skew(&data).unwrap() > 0.0);
    }

    #[test]
    fn kurtosis_uniform_is_negative() {
        let data: Vec<f64> = (1..=10).map(f64::from).collect();
        let k = kurtosis(&data).unwrap();
        assert!((k - (-1.2242424242424244)).abs() < 1e-12, "kurtosis={k}");
    }

    #[test]
    fn constant_series_moments_are_nan() {
        assert!(skew(&[7.0, 7.0, 7.0]).unwrap().is_nan());
        assert!(kurtosis(&[7.0, 7.0, 7.0]).unwrap().is_nan());
    }

    #[test]
    fn summarize_basic() {
        let data: Vec<f64> = (1..=10).map(f64::from).collect();
        let s = summarize(&data).unwrap();
        assert_eq!(s.count, 10);
        assert!((s.mean - 5.5).abs() < 1e-12);
        assert!((s.median - 5.5).abs() < 1e-12);
        assert_eq!(s.min, 1.0);
        assert_eq!(s.max, 10.0);
        assert!(s.skew.abs() < 1e-12);
    }

    #[test]
    fn summarize_single_value() {
        let s = summarize(&[42.0]).unwrap();
        assert_eq!(s.count, 1);
        assert_eq!(s.std, 0.0);
        assert!(s.skew.is_nan());
    }

    #[test]
    fn summarize_empty_is_none() {
        assert_eq!(summarize(&[]), None);
    }

    #[test]
    fn iqr_mask_flags_extreme_value() {
        let mask = iqr_mask(&[1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 100.0], 1.5).unwrap();
        assert_eq!(mask, vec![true, true, true, true, true, true, false]);
    }

    #[test]
    fn iqr_mask_clean_data_all_true() {
        assert!(
            iqr_mask(&[1.0, 2.0, 3.0, 4.0], 1.5)
                .unwrap()
                .iter()
                .all(|m| *m)
        );
    }

    #[test]
    fn iqr_mask_custom_k() {
        let data = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 10.0];
        assert!(!iqr_mask(&data, 1.0).unwrap()[6]);
        assert!(iqr_mask(&data, 5.0).unwrap()[6]);
    }

    #[test]
    fn iqr_mask_empty_is_none() {
        assert_eq!(iqr_mask(&[], 1.5), None);
    }

    #[test]
    fn minmax_normalize_default_range() {
        let r = minmax_normalize(&[10.0, 20.0, 30.0], 0.0, 1.0).unwrap();
        assert_close(&r, &[0.0, 0.5, 1.0]);
    }

    #[test]
    fn minmax_normalize_constant_maps_to_midpoint() {
        assert_eq!(
            minmax_normalize(&[7.0, 7.0], -1.0, 1.0),
            Some(vec![0.0, 0.0])
        );
    }

    #[test]
    fn minmax_normalize_invalid() {
        assert_eq!(minmax_normalize(&[], 0.0, 1.0), None);
        assert_eq!(minmax_normalize(&[1.0], 1.0, 1.0), None);
        assert_eq!(minmax_normalize(&[1.0], 2.0, 1.0), None);
    }

    #[test]
    fn pct_change_known_values() {
        let r = pct_change(&[100.0, 110.0, 99.0]);
        assert!(r[0].is_nan());
        assert_close(&r[1..], &[0.1, -0.1]);
    }

    #[test]
    fn pct_change_zero_predecessor_is_nan() {
        let r = pct_change(&[0.0, 5.0, 10.0]);
        assert!(r[1].is_nan());
        assert_eq!(r[2], 1.0);
    }

    #[test]
    fn log_returns_known_values() {
        let r = log_returns(&[1.0, std::f64::consts::E, std::f64::consts::E.powi(2)]);
        assert!(r[0].is_nan());
        assert_close(&r[1..], &[1.0, 1.0]);
    }

    #[test]
    fn log_returns_negative_ratio_is_nan() {
        let r = log_returns(&[1.0, -2.0]);
        assert!(r[1].is_nan());
    }

    #[test]
    fn weighted_mean_basic() {
        assert_eq!(
            weighted_mean(&[80.0, 90.0, 60.0], &[3.0, 2.0, 5.0]),
            Some(72.0)
        );
    }

    #[test]
    fn weighted_mean_invalid() {
        assert_eq!(weighted_mean(&[], &[]), None);
        assert_eq!(weighted_mean(&[1.0, 2.0], &[1.0]), None);
        assert_eq!(weighted_mean(&[1.0, 2.0], &[1.0, -1.0]), None);
        assert_eq!(weighted_mean(&[1.0, 2.0], &[0.0, 0.0]), None);
    }

    #[test]
    fn geometric_mean_basic() {
        assert_eq!(geometric_mean(&[1.0, 4.0]), Some(2.0));
        assert_eq!(geometric_mean(&[1.0, 0.0]), None);
        assert_eq!(geometric_mean(&[]), None);
    }

    #[test]
    fn harmonic_mean_basic() {
        let h = harmonic_mean(&[1.0, 2.0, 4.0]).unwrap();
        assert!((h - 12.0 / 7.0).abs() < 1e-12);
        assert_eq!(harmonic_mean(&[1.0, -1.0]), None);
    }

    #[test]
    fn mean_ordering_inequality() {
        let data = [1.0, 2.0, 4.0, 8.0];
        let h = harmonic_mean(&data).unwrap();
        let g = geometric_mean(&data).unwrap();
        let a = mean(&data).unwrap();
        assert!(h <= g && g <= a);
    }
}
