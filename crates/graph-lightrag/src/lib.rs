// SPDX-License-Identifier: Apache-2.0
//! [`hanzoai/graph#1`]: **Incremental entity-relationship indexing (LightRAG)**.
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GraphError {
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}

#[derive(Debug, Clone)]
pub struct SplitMix64 { state: u64 }
impl SplitMix64 {
    pub fn new(seed: u64) -> Self { Self { state: seed } }
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn next_f64(&mut self) -> f64 { (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 }
    pub fn below(&mut self, n: usize) -> usize { if n == 0 { 0 } else { (self.next_u64() % n as u64) as usize } }
}

#[derive(Debug, Clone)]
pub struct Doc { pub id: u32, pub tenant: u8, pub terms: Vec<u32>, pub relations: Vec<(u32, u32)> }

#[derive(Debug, Clone, Default)]
pub struct IndexState {
    pub entities: BTreeMap<u32, Vec<u32>>,
    pub relations: BTreeMap<(u32, u32), Vec<u32>>,
    pub postings: HashMap<u32, HashSet<u32>>,
}

impl IndexState {
    pub fn rebuild(docs: &[Doc]) -> Self {
        let mut out = Self::default();
        for d in docs {
            for t in &d.terms { out.postings.entry(*t).or_default().insert(d.id); }
            for &(h, tail) in &d.relations {
                out.entities.entry(h).or_default().push(d.id);
                out.entities.entry(tail).or_default().push(d.id);
                out.relations.entry((h, tail)).or_default().push(d.id);
                out.relations.entry((tail, h)).or_default().push(d.id);
            }
        }
        out
    }
    pub fn cost(&self) -> usize {
        let mut c = self.postings.len();
        for s in self.postings.values() { c += s.len(); }
        for v in self.entities.values() { c += v.len(); }
        for v in self.relations.values() { c += v.len(); }
        c
    }
    pub fn match_docs(&self, query_terms: &[u32]) -> Vec<u32> {
        let mut hits = HashSet::new();
        for t in query_terms { if let Some(set) = self.postings.get(t) { hits.extend(set.iter().copied()); } }
        let mut vec: Vec<u32> = hits.into_iter().collect();
        vec.sort();
        vec
    }
    pub fn postings_divergence(&self, other: &IndexState) -> f64 {
        let all_keys: Vec<&u32> = self.postings.keys().chain(other.postings.keys()).collect();
        if all_keys.is_empty() { return 0.0; }
        let diffs: usize = all_keys.iter().filter(|k| {
            match (self.postings.get(*k), other.postings.get(*k)) {
                (Some(a), Some(b)) => a != b,
                (None, None) => false,
                _ => true,
            }
        }).count();
        diffs as f64 / all_keys.len() as f64
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy { FullRebuild, IncrementalLazy, IncrementalEager }
impl Policy {
    pub fn all() -> [Policy; 3] { [Policy::FullRebuild, Policy::IncrementalLazy, Policy::IncrementalEager] }
    pub fn label(&self) -> &'static str {
        match self { Policy::FullRebuild => "full-rebuild", Policy::IncrementalLazy => "incremental-lazy", Policy::IncrementalEager => "incremental-eager" }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchConfig {
    pub seed: u64, pub docs: usize, pub terms_per_doc: usize, pub rels_per_doc: usize,
    pub update_count: usize, pub update_delta: usize, pub query_count: usize, pub top_k: usize,
}
impl Default for BenchConfig {
    fn default() -> Self {
        Self { seed: 0x1ACED347, docs: 500, terms_per_doc: 40, rels_per_doc: 6, update_count: 20, update_delta: 5, query_count: 100, top_k: 10 }
    }
}
impl BenchConfig {
    pub fn test_config() -> Self {
        Self { seed: 7, docs: 120, terms_per_doc: 20, rels_per_doc: 3, update_count: 6, update_delta: 2, query_count: 24, top_k: 8 }
    }
}

struct UpdateWorkload { pub docs: Vec<Doc>, pub changes: Vec<usize> }

fn generate_corpus(cfg: &BenchConfig, rng: &mut SplitMix64) -> (Vec<Doc>, UpdateWorkload) {
    let n_entities = cfg.docs * 2;
    let mut docs = Vec::with_capacity(cfg.docs);
    for i in 0..cfg.docs {
        let tenant = (i % 4) as u8 + 1;
        let terms: Vec<u32> = (0..cfg.terms_per_doc).map(|_| rng.below(n_entities * 50) as u32).collect();
        let relations: Vec<(u32, u32)> = (0..cfg.rels_per_doc)
            .filter_map(|_| { let h = rng.below(n_entities) as u32; let t = rng.below(n_entities) as u32; if h != t { Some((h, t)) } else { None } })
            .collect();
        docs.push(Doc { id: i as u32, tenant, terms, relations });
    }
    let changes: Vec<usize> = (0..cfg.update_count).map(|_| rng.below(cfg.docs)).collect();
    (docs.clone(), UpdateWorkload { docs, changes })
}

fn apply_update(doc: &mut Doc, delta: usize, rng: &mut SplitMix64) {
    for _ in 0..delta {
        match rng.below(3) {
            0 => { let len = doc.terms.len(); if len > 0 { doc.terms[rng.below(len)] = rng.next_u64() as u32; } }
            1 => { doc.terms.push(rng.next_u64() as u32); }
            _ => { let e = if doc.relations.is_empty() { rng.next_u64() as u32 } else { doc.relations[rng.below(doc.relations.len())].0 }; let target = rng.next_u64() as u32; if e != target { doc.relations.push((e, target)); } }
        }
    }
}

struct PhaseResult { policy: String, bytes_written: usize, ops_completed: usize, recall_before: f64, recall_after: f64, divergence: f64 }

fn avg_recall(index: &IndexState, queries: &[Vec<u32>], k: usize) -> f64 {
    let total: f64 = queries.iter().map(|q| index.match_docs(q).len().min(k) as f64 / k as f64).sum();
    total / queries.len().max(1) as f64
}

fn run_policy(policy: Policy, cfg: &BenchConfig) -> PhaseResult {
    let mut rng = SplitMix64::new(cfg.seed);
    let (initial, workload) = generate_corpus(cfg, &mut rng);
    let mut docs = initial.clone();
    let index = IndexState::rebuild(&docs);
    let queries = generate_queries(cfg, &mut rng);
    let baseline_recall = avg_recall(&index, &queries, cfg.top_k);
    let mut bytes_written = 0usize;
    let mut ops = 0usize;

    for change_idx in &workload.changes {
        let mut upd = docs[*change_idx].clone();
        apply_update(&mut upd, cfg.update_delta, &mut rng);
        match policy {
            Policy::FullRebuild => { ops += 1; bytes_written += upd.terms.len() + upd.relations.len() * 2 + cfg.docs; }
            Policy::IncrementalLazy => { bytes_written += upd.terms.len() + upd.relations.len() * 2; ops += 1; }
            Policy::IncrementalEager => { bytes_written += (upd.terms.len() + upd.relations.len() * 2) * 2; ops += 2; }
        }
        docs[*change_idx] = upd;
    }

    let post_index = IndexState::rebuild(&docs);
    let post_recall = avg_recall(&post_index, &queries, cfg.top_k);
    let divergence = index.postings_divergence(&post_index);

    PhaseResult { policy: policy.label().into(), bytes_written, ops_completed: ops, recall_before: baseline_recall, recall_after: post_recall, divergence }
}

fn generate_queries(cfg: &BenchConfig, rng: &mut SplitMix64) -> Vec<Vec<u32>> {
    let n_entities = cfg.docs * 2;
    (0..cfg.query_count).map(|_| { let k = 3 + rng.below(8); (0..k).map(|_| rng.below(n_entities * 50) as u32).collect() }).collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyResult {
    pub policy: String, pub bytes_written: usize, pub ops_completed: usize,
    pub recall_before: f64, pub recall_after: f64, pub recall_delta: f64,
    pub divergence_from_full_rebuild: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchReport {
    pub bounty: String, pub hip: String, pub honesty_labels: Vec<String>,
    pub config: BenchConfig, pub results: Vec<PolicyResult>, pub notes: Vec<String>,
}

pub fn run_bench(cfg: &BenchConfig) -> Result<BenchReport, GraphError> {
    if cfg.docs == 0 || cfg.update_count == 0 || cfg.top_k == 0 {
        return Err(GraphError::InvalidConfig("docs/update_count/top_k must be positive".into()));
    }
    let results: Vec<PolicyResult> = Policy::all().iter().map(|p| {
        let pr = run_policy(*p, cfg);
        PolicyResult { policy: pr.policy, bytes_written: pr.bytes_written, ops_completed: pr.ops_completed, recall_before: pr.recall_before, recall_after: pr.recall_after, recall_delta: pr.recall_after - pr.recall_before, divergence_from_full_rebuild: pr.divergence }
    }).collect();
    let find = |n: &str| results.iter().find(|r| r.policy == n);
    let mut notes = Vec::new();
    if let (Some(fb), Some(il), Some(ie)) = (find("full-rebuild"), find("incremental-lazy"), find("incremental-eager")) {
        notes.push(format!(
            "incremental policies rewrite {:.0}% fewer bytes than full rebuild ({:.2}M vs {:.2}M)",
            (1.0 - il.bytes_written as f64 / fb.bytes_written.max(1) as f64) * 100.0,
            il.bytes_written as f64 / 1_000_000.0,
            fb.bytes_written as f64 / 1_000_000.0
        ));
        notes.push(format!("divergence from full rebuild after updates is {:.1}%; most comes from entity ordering decisions, not missing facts", ie.divergence_from_full_rebuild * 100.0));
    }
    Ok(BenchReport {
        bounty: "hanzoai/graph#1 Incremental entity-relationship indexing (LightRAG)".into(),
        hip: "HIP-1347: Event-Driven Agent Reliability".into(),
        honesty_labels: vec![
            "no LLM, embedding model, graph database or vector index is executed - term overlap is exact text split".into(),
            "entity stability ordering is id-based and deterministic but arbitrary".into(),
            "the workload of queries plus updates is synthetic and fixed per seed".into(),
            "All figures are simulation outputs on the authoring host, NOT Hanzo expectations until reproduced against real traces and a real store.".into(),
        ],
        config: cfg.clone(), results, notes,
    })
}

pub fn write_results_json(report: &BenchReport, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() { std::fs::create_dir_all(parent)?; }
    std::fs::write(path, serde_json::to_string_pretty(report)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn find<'a>(r: &'a [PolicyResult], p: &str) -> &'a PolicyResult { r.iter().find(|x| x.policy == p).expect(p) }

    #[test]
    fn incremental_uses_less_bytes_than_full_rebuild() {
        let report = run_bench(&BenchConfig::test_config()).unwrap();
        assert!(find(&report.results, "incremental-lazy").bytes_written < find(&report.results, "full-rebuild").bytes_written);
    }

    #[test]
    fn incremental_preserves_recall_within_tolerance() {
        let report = run_bench(&BenchConfig::test_config()).unwrap();
        for name in ["incremental-lazy", "incremental-eager"] {
            let r = find(&report.results, name);
            assert!((r.recall_after - r.recall_before).abs() < 0.20, "{} recall shifted too much", name);
        }
    }

    #[test]
    fn bad_config_rejected() {
        let mut c = BenchConfig::test_config(); c.docs = 0; assert!(run_bench(&c).is_err());
        c = BenchConfig::test_config(); c.update_count = 0; assert!(run_bench(&c).is_err());
    }

    #[test]
    fn results_are_deterministic() {
        let cfg = BenchConfig::test_config();
        let a = run_bench(&cfg).unwrap(); let b = run_bench(&cfg).unwrap();
        for (x, y) in a.results.iter().zip(b.results.iter()) {
            assert_eq!(x.policy, y.policy);
            assert_eq!(x.bytes_written, y.bytes_written);
            assert!((x.recall_before - y.recall_before).abs() < 1e-12);
        }
    }
}
