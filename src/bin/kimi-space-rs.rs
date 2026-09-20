//! kimi-space-rs：从 stdin 读取空白分隔的数值，以 key=value 形式输出描述性统计。
//!
//! 用法示例：`echo "1 2 3 4" | kimi-space-rs`

use std::io::Read;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut buf = String::new();
    if let Err(err) = std::io::stdin().read_to_string(&mut buf) {
        eprintln!("error: 读取 stdin 失败: {err}");
        return ExitCode::FAILURE;
    }
    let mut values = Vec::new();
    for token in buf.split_whitespace() {
        match token.parse::<f64>() {
            Ok(v) => values.push(v),
            Err(_) => {
                eprintln!("error: 无法解析数值: {token:?}");
                return ExitCode::FAILURE;
            }
        }
    }
    let Some(s) = kimi_space::summarize(&values) else {
        eprintln!("error: 未读取到任何数值");
        return ExitCode::FAILURE;
    };
    println!("count={}", s.count);
    println!("mean={}", s.mean);
    println!("std={}", s.std);
    println!("min={}", s.min);
    println!("q25={}", s.q25);
    println!("median={}", s.median);
    println!("q75={}", s.q75);
    println!("max={}", s.max);
    println!("skew={}", s.skew);
    println!("kurtosis={}", s.kurtosis);
    ExitCode::SUCCESS
}
