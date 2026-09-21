//! kimi-space-rs 二进制的集成测试：通过子进程喂 stdin、读 stdout。

use std::io::Write;
use std::process::{Command, Stdio};

fn run_bin(args: &[&str], input: &str) -> (bool, String, String) {
    let exe = env!("CARGO_BIN_EXE_kimi-space-rs");
    let mut child = Command::new(exe)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn kimi-space-rs");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    (
        out.status.success(),
        String::from_utf8(out.stdout).unwrap(),
        String::from_utf8(out.stderr).unwrap(),
    )
}

fn rows(stdout: &str) -> Vec<f64> {
    stdout.lines().map(|l| l.parse::<f64>().unwrap()).collect()
}

#[test]
fn summarizes_stdin() {
    let (ok, stdout, _) = run_bin(&[], "1 2 3 4");
    assert!(ok);
    assert!(stdout.contains("count=4"));
    assert!(stdout.contains("mean=2.5"));
    assert!(stdout.contains("median=2.5"));
    assert!(stdout.contains("min=1"));
    assert!(stdout.contains("max=4"));
}

#[test]
fn explicit_summary_subcommand() {
    let (ok, stdout, _) = run_bin(&["summary"], "1 2 3 4");
    assert!(ok);
    assert!(stdout.contains("count=4"));
}

#[test]
fn accepts_newline_separated_input() {
    let (ok, stdout, _) = run_bin(&[], "1\n2\n3\n4\n");
    assert!(ok);
    assert!(stdout.contains("count=4"));
}

#[test]
fn empty_input_fails() {
    let (ok, _, stderr) = run_bin(&[], "");
    assert!(!ok);
    assert!(stderr.contains("未读取到任何数值"));
}

#[test]
fn invalid_token_fails() {
    let (ok, _, stderr) = run_bin(&[], "1 two 3");
    assert!(!ok);
    assert!(stderr.contains("无法解析数值"));
}

#[test]
fn zscores_subcommand() {
    let (ok, stdout, _) = run_bin(&["zscores"], "1 2 3");
    assert!(ok);
    let r = rows(&stdout);
    assert_eq!(r.len(), 3);
    assert!((r[0] - -1.0).abs() < 1e-12);
    assert!(r[1].abs() < 1e-12);
    assert!((r[2] - 1.0).abs() < 1e-12);
}

#[test]
fn pct_change_subcommand() {
    let (ok, stdout, _) = run_bin(&["pct-change"], "100 110 99");
    assert!(ok);
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 3);
    assert_eq!(lines[0], "NaN");
    assert!((lines[1].parse::<f64>().unwrap() - 0.1).abs() < 1e-12);
}

#[test]
fn log_returns_subcommand() {
    let (ok, stdout, _) = run_bin(&["log-returns"], "1 2.718281828459045");
    assert!(ok);
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines[0], "NaN");
    assert!((lines[1].parse::<f64>().unwrap() - 1.0).abs() < 1e-12);
}

#[test]
fn minmax_subcommand() {
    let (ok, stdout, _) = run_bin(&["minmax", "0", "1"], "10 20 30");
    assert!(ok);
    assert_eq!(rows(&stdout), vec![0.0, 0.5, 1.0]);
}

#[test]
fn minmax_requires_bounds() {
    let (ok, _, stderr) = run_bin(&["minmax"], "1 2 3");
    assert!(!ok);
    assert!(stderr.contains("minmax 需要"));
}

#[test]
fn unknown_subcommand_fails() {
    let (ok, _, stderr) = run_bin(&["frobnicate"], "1 2 3");
    assert!(!ok);
    assert!(stderr.contains("未知子命令"));
}

#[test]
fn help_flag_prints_usage() {
    for flag in ["-h", "--help"] {
        let (ok, stdout, _) = run_bin(&[flag], "");
        assert!(ok);
        assert!(stdout.contains("用法"));
        assert!(stdout.contains("summary"));
    }
}

#[test]
fn version_flag_prints_version() {
    for flag in ["-V", "--version"] {
        let (ok, stdout, _) = run_bin(&[flag], "");
        assert!(ok);
        assert!(stdout.contains("kimi-space-rs"));
        assert!(stdout.contains(env!("CARGO_PKG_VERSION")));
    }
}

#[test]
fn outliers_subcommand() {
    let (ok, stdout, stderr) = run_bin(&["outliers"], "1 2 2.5 3 3.5 4 100");
    assert!(ok);
    assert_eq!(rows(&stdout), vec![100.0]);
    assert!(stderr.contains("1 个异常值"));
}

#[test]
fn outliers_custom_k() {
    let (ok, stdout, stderr) = run_bin(&["outliers", "5"], "1 2 2.5 3 3.5 4 10");
    assert!(ok);
    assert!(stdout.trim().is_empty());
    assert!(stderr.contains("0 个异常值"));
}

#[test]
fn outliers_bad_k_fails() {
    let (ok, _, stderr) = run_bin(&["outliers", "abc"], "1 2 3");
    assert!(!ok);
    assert!(stderr.contains("无法解析栅栏系数"));
}

#[test]
fn means_subcommand() {
    let (ok, stdout, _) = run_bin(&["means"], "1 2 4");
    assert!(ok);
    let get = |key: &str| -> f64 {
        stdout
            .lines()
            .find(|l| l.starts_with(key))
            .unwrap()
            .split('=')
            .nth(1)
            .unwrap()
            .parse()
            .unwrap()
    };
    assert!((get("arithmetic") - 7.0 / 3.0).abs() < 1e-12);
    assert!((get("geometric") - 2.0).abs() < 1e-12);
    assert!((get("harmonic") - 12.0 / 7.0).abs() < 1e-12);
}

#[test]
fn means_non_positive_gives_nan() {
    let (ok, stdout, _) = run_bin(&["means"], "1 -2 3");
    assert!(ok);
    assert!(stdout.contains("geometric=NaN"));
    assert!(stdout.contains("harmonic=NaN"));
}

#[test]
fn moving_average_subcommand() {
    let (ok, stdout, _) = run_bin(&["moving-average", "2"], "1 2 3 4");
    assert!(ok);
    assert_eq!(rows(&stdout), vec![1.5, 2.5, 3.5]);
}

#[test]
fn moving_average_requires_window() {
    let (ok, _, stderr) = run_bin(&["moving-average"], "1 2 3");
    assert!(!ok);
    assert!(stderr.contains("需要 <window>"));
}

#[test]
fn moving_average_window_too_large_fails() {
    let (ok, _, stderr) = run_bin(&["moving-average", "9"], "1 2 3");
    assert!(!ok);
    assert!(stderr.contains("window 为 0 或超过数据长度"));
}

#[test]
fn autocorr_subcommand_default_lag() {
    let (ok, stdout, _) = run_bin(&["autocorr"], "1 2 3 4 5");
    assert!(ok);
    assert!((rows(&stdout)[0] - 1.0).abs() < 1e-12);
}

#[test]
fn autocorr_subcommand_explicit_lag() {
    let (ok, stdout, _) = run_bin(&["autocorr", "2"], "1 -1 1 -1 1 -1");
    assert!(ok);
    assert!((rows(&stdout)[0] - 1.0).abs() < 1e-12);
}

#[test]
fn autocorr_invalid_lag_fails() {
    let (ok, _, stderr) = run_bin(&["autocorr", "9"], "1 2 3");
    assert!(!ok);
    assert!(stderr.contains("lag"));
}

#[test]
fn moving_std_subcommand() {
    let (ok, stdout, _) = run_bin(&["moving-std", "2"], "1 2 3 4");
    assert!(ok);
    let r = rows(&stdout);
    assert_eq!(r.len(), 3);
    for v in r {
        assert!((v - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
    }
}

#[test]
fn moving_std_window_one_fails() {
    let (ok, _, stderr) = run_bin(&["moving-std", "1"], "1 2 3");
    assert!(!ok);
    assert!(stderr.contains("window"));
}

#[test]
fn percentile_subcommand() {
    let (ok, stdout, _) = run_bin(&["percentile", "25"], "1 2 3 4 5 6 7 8 9 10");
    assert!(ok);
    assert!((rows(&stdout)[0] - 3.25).abs() < 1e-12);
}

#[test]
fn percentile_out_of_range_fails() {
    let (ok, _, stderr) = run_bin(&["percentile", "101"], "1 2 3");
    assert!(!ok);
    assert!(stderr.contains("[0, 100]"));
}

#[test]
fn percentile_requires_q() {
    let (ok, _, stderr) = run_bin(&["percentile"], "1 2 3");
    assert!(!ok);
    assert!(stderr.contains("需要 <q>"));
}

#[test]
fn mad_subcommand() {
    let (ok, stdout, _) = run_bin(&["mad"], "1 2 3 4 100");
    assert!(ok);
    assert_eq!(rows(&stdout), vec![1.0]);
}

#[test]
fn mad_empty_fails() {
    let (ok, _, stderr) = run_bin(&["mad"], "");
    assert!(!ok);
    assert!(stderr.contains("未读取到任何数值"));
}

#[test]
fn mad_outliers_subcommand() {
    let (ok, stdout, stderr) = run_bin(&["mad-outliers"], "1 2 2.5 3 3.5 4 100");
    assert!(ok);
    assert_eq!(rows(&stdout), vec![100.0]);
    assert!(stderr.contains("1 个异常值"));
}

#[test]
fn mad_outliers_custom_threshold() {
    let (ok, stdout, _) = run_bin(&["mad-outliers", "100"], "1 2 2.5 3 3.5 4 100");
    assert!(ok);
    assert!(stdout.trim().is_empty());
}

#[test]
fn mad_outliers_bad_threshold_fails() {
    let (ok, _, stderr) = run_bin(&["mad-outliers", "abc"], "1 2 3");
    assert!(!ok);
    assert!(stderr.contains("无法解析阈值"));
}
