// SPDX-License-Identifier: Apache-2.0
//! Reproducible runner.
use graph_graphiti::{run_bench, write_results_json, BenchConfig};
use std::path::PathBuf;

fn main() {
    let out = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("results.json"));
    eprintln!("running graph-graphiti");
    let report = run_bench(&BenchConfig::default()).expect("benchmark failed");
    write_results_json(&report, &out).expect("failed to write results");
    eprintln!("wrote {}", out.display());
}
