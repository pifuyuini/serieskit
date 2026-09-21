//! 内核函数基准：cargo bench

use criterion::{Criterion, criterion_group, criterion_main};

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

fn bench_kernels(c: &mut Criterion) {
    let data = xorshift_series(200_000);
    c.bench_function("moving_average w50", |b| {
        b.iter(|| kimi_space::moving_average(&data, 50))
    });
    c.bench_function("moving_std w50", |b| {
        b.iter(|| kimi_space::moving_std(&data, 50))
    });
    c.bench_function("summarize", |b| b.iter(|| kimi_space::summarize(&data)));
    c.bench_function("zscores", |b| b.iter(|| kimi_space::zscores(&data)));
    c.bench_function("autocorr lag1", |b| {
        b.iter(|| kimi_space::autocorr(&data, 1))
    });
    c.bench_function("mad_mask t3.5", |b| {
        b.iter(|| kimi_space::mad_mask(&data, 3.5))
    });
    c.bench_function("robust_zscores", |b| {
        b.iter(|| kimi_space::robust_zscores(&data))
    });
}

criterion_group!(benches, bench_kernels);
criterion_main!(benches);
