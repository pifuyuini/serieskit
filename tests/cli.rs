//! kimi-space-rs 二进制的集成测试：通过子进程喂 stdin、读 stdout。

use std::io::Write;
use std::process::{Command, Stdio};

fn run_bin(input: &str) -> (bool, String, String) {
    let exe = env!("CARGO_BIN_EXE_kimi-space-rs");
    let mut child = Command::new(exe)
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

#[test]
fn summarizes_stdin() {
    let (ok, stdout, _) = run_bin("1 2 3 4");
    assert!(ok);
    assert!(stdout.contains("count=4"));
    assert!(stdout.contains("mean=2.5"));
    assert!(stdout.contains("median=2.5"));
    assert!(stdout.contains("min=1"));
    assert!(stdout.contains("max=4"));
}

#[test]
fn accepts_newline_separated_input() {
    let (ok, stdout, _) = run_bin("1\n2\n3\n4\n");
    assert!(ok);
    assert!(stdout.contains("count=4"));
}

#[test]
fn empty_input_fails() {
    let (ok, _, stderr) = run_bin("");
    assert!(!ok);
    assert!(stderr.contains("未读取到任何数值"));
}

#[test]
fn invalid_token_fails() {
    let (ok, _, stderr) = run_bin("1 two 3");
    assert!(!ok);
    assert!(stderr.contains("无法解析数值"));
}
