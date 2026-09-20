//! kimi-space-rs：从 stdin 读取空白分隔的数值，按子命令输出统计或变换结果。
//!
//! 用法示例：
//!   echo "1 2 3 4" | kimi-space-rs              # 等价于 summary
//!   echo "1 2 3 4" | kimi-space-rs summary      # key=value 描述性统计
//!   echo "1 2 3 4" | kimi-space-rs zscores      # 逐行输出标准化值
//!   echo "1 2 3 4" | kimi-space-rs pct-change   # 逐行输出相邻变化率
//!   echo "1 2 3 4" | kimi-space-rs log-returns  # 逐行输出对数收益
//!   echo "1 2 3 4" | kimi-space-rs minmax 0 1   # 逐行输出归一化值

use std::io::Read;
use std::process::ExitCode;

const USAGE: &str =
    "用法: kimi-space-rs [summary|zscores|pct-change|log-returns|minmax <lo> <hi>] < 数值";

fn read_values() -> Result<Vec<f64>, String> {
    let mut buf = String::new();
    std::io::stdin()
        .read_to_string(&mut buf)
        .map_err(|e| format!("读取 stdin 失败: {e}"))?;
    let mut values = Vec::new();
    for token in buf.split_whitespace() {
        match token.parse::<f64>() {
            Ok(v) => values.push(v),
            Err(_) => return Err(format!("无法解析数值: {token:?}")),
        }
    }
    Ok(values)
}

fn print_rows(values: &[f64]) {
    for v in values {
        println!("{v}");
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let command = args.first().map(String::as_str).unwrap_or("summary");

    let run = || -> Result<(), String> {
        let values = read_values()?;
        match command {
            "summary" => {
                let s = kimi_space::summarize(&values).ok_or("未读取到任何数值")?;
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
            }
            "zscores" => print_rows(&kimi_space::zscores(&values)),
            "pct-change" => print_rows(&kimi_space::pct_change(&values)),
            "log-returns" => print_rows(&kimi_space::log_returns(&values)),
            "minmax" => {
                let (Some(lo), Some(hi)) = (
                    args.get(1).and_then(|s| s.parse::<f64>().ok()),
                    args.get(2).and_then(|s| s.parse::<f64>().ok()),
                ) else {
                    return Err(format!("minmax 需要 <lo> <hi> 两个数值参数\n{USAGE}"));
                };
                let out = kimi_space::minmax_normalize(&values, lo, hi)
                    .ok_or("空输入或区间下界不小于上界")?;
                print_rows(&out);
            }
            other => return Err(format!("未知子命令: {other:?}\n{USAGE}")),
        }
        Ok(())
    };

    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("error: {msg}");
            ExitCode::FAILURE
        }
    }
}
