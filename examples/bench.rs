//! Rust 侧吞吐基准：与 examples/benchmark.py 的工作负载对应。

use std::time::Instant;

fn xorshift_series(n: usize) -> Vec<f64> {
    let mut state = 0x9E3779B97F4A7C15u64;
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        data.push((state >> 11) as f64 / (1u64 << 53) as f64);
    }
    data
}

fn main() {
    let n = 200_000;
    let window = 50;
    let data = xorshift_series(n);

    let t0 = Instant::now();
    let ma = kimi_space::moving_average(&data, window).unwrap();
    let dt = t0.elapsed();
    println!(
        "rust moving_average: n={n} window={window} -> {} points in {:.3} ms (checksum {:.6})",
        ma.len(),
        dt.as_secs_f64() * 1e3,
        ma.iter().sum::<f64>()
    );

    let t1 = Instant::now();
    let s = kimi_space::summarize(&data).unwrap();
    let dt1 = t1.elapsed();
    println!(
        "rust summarize: n={n} in {:.3} ms (mean {:.6})",
        dt1.as_secs_f64() * 1e3,
        s.mean
    );

    let t2 = Instant::now();
    let ms = kimi_space::moving_std(&data, window).unwrap();
    let dt2 = t2.elapsed();
    println!(
        "rust moving_std: n={n} window={window} -> {} points in {:.3} ms (checksum {:.6})",
        ms.len(),
        dt2.as_secs_f64() * 1e3,
        ms.iter().sum::<f64>()
    );
}
