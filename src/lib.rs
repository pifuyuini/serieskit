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
}
