//! kimi-space 数值内核：与 Python 侧统计能力对应的 Rust 实现。

/// 算术平均值；空输入返回 None。
pub fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    Some(values.iter().sum::<f64>() / values.len() as f64)
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
}
