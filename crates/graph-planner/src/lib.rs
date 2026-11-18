// SPDX-License-Identifier: Apache-2.0
//! [`hanzoai/graph#5`]: **Native graph storage and a cost-based graph/vector query planner**.
//!
//! Clean-room simulation harness for choosing between vector-first, graph-first,
//! hybrid (vector seeds then bounded expansion) and cached-neighbourhood plans
//! from estimated cardinalities, selectivity, graph degree and a latency budget.
//!
//! # Design
//!
//! A query belongs to one of four shapes. `vector-only` answers live in documents
//! with no entity link, so only a vector scan can reach them. `direct` is an entity
//! lookup by name. `one-hop` and `two-hop` answers are related entities, so a
//! traversal is needed. Every plan has a capability set and a cost dominated by a
//! different term, so the optimal plan changes with the query shape and with
//! whether the neighbourhood is already materialised.
//!
//! The planner does **not** get the real cost model: it holds a hand-set model
//! whose constants are wrong in known ways (it underestimates the vector seed
//! probe, the frontier overlap and the cost of a cache refresh). That is the
//! point of the benchmark — a planner that is scored against its own cost model
//! proves nothing.
//!
//! - `AlwaysVector`: scan the vector index for every query.
//! - `AlwaysGraph`: traverse the entity graph for every query.
//! - `CostBased`: pick the cheapest plan its own model predicts can answer it.
//!
//! Metrics: mean and p95 cost per query, cost relative to the per-query optimum,
//! plan accuracy (chosen plan within 10% of the optimum), the planner's estimate
//! error, and tenant-isolation violations.
//!
//! # Honest boundaries
//! - No graph database, vector index, GPU or LLM is executed. Cost is a modelled
//!   function of hand-set per-vector and per-node constants.
//! - Graph degree, cache coverage, tenant mix, the planner's statistics error and
//!   the gap between the planner's cost model and the real one are hand-set, not
//!   measured on a real corpus.
//! - The workload mix is synthetic and fixed per seed, not sampled from production.
//! - All figures are simulation outputs on the authoring host, NOT Hanzo
//!   expectations until reproduced against real traces on real storage.

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PlannerError {
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}

#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Cost units per scanned vector (a 40k-vector scan costs 40 units).
pub const VEC_SCAN_UNIT: f64 = 1.0 / 1_000.0;
/// Overhead factor when a plan cannot answer the query and a fallback runs.
pub const RETRY_FACTOR: f64 = 2.5;

/// A cost model. Two instances exist: the one the planner believes, and the one
/// that actually governs the work performed.
#[derive(Debug, Clone, Copy)]
pub struct CostModel {
    /// Cost units per scanned vector.
    pub vec_unit: f64,
    /// Cost units per visited node.
    pub node_unit: f64,
    /// Cost units for a fully materialised neighbourhood lookup.
    pub cache_hit: f64,
    /// Fraction of the vector index probed to obtain top-k seeds.
    pub seed_probe: f64,
    /// How much of the next frontier is already seen.
    pub frontier_dedup: f64,
    /// Fixed per-hop cost (round trips), invisible to the planner's model.
    pub hop_tax: f64,
    /// Multiplier on the traversal degree for hybrid plans.
    pub hybrid_fanout: f64,
}

impl CostModel {
    /// What the planner believes.
    pub fn planner() -> Self {
        Self { vec_unit: VEC_SCAN_UNIT, node_unit: 2.0, cache_hit: 0.5, seed_probe: 0.25, frontier_dedup: 0.35, hop_tax: 0.0, hybrid_fanout: 0.4 }
    }

    /// What the work actually costs. The differences are deliberate and stated:
    /// the seed probe is 2.4x more expensive than the planner thinks, frontiers
    /// overlap more (0.55 vs 0.35), a cache hit still pays for a freshness check,
    /// and every hop pays a round trip the planner does not model.
    pub fn truth() -> Self {
        Self { vec_unit: VEC_SCAN_UNIT, node_unit: 2.0, cache_hit: 0.9, seed_probe: 0.6, frontier_dedup: 0.55, hop_tax: 1.5, hybrid_fanout: 0.4 }
    }

    /// Nodes visited by a bounded traversal, with de-duplication per hop.
    pub fn expansion(&self, hops: usize, degree: f64) -> f64 {
        if hops == 0 {
            return 0.0;
        }
        let mut nodes = 0.0;
        let mut frontier = degree;
        for _ in 0..hops {
            nodes += frontier;
            frontier *= degree * self.frontier_dedup;
        }
        nodes * self.node_unit + hops as f64 * self.hop_tax
    }
}

/// The shape of a query, which determines what a plan can reach.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryKind {
    /// Answer is a document with no entity link: only a vector scan reaches it.
    VectorOnly,
    /// Answer is the entity itself.
    Direct,
    /// Answer is one relationship away.
    OneHop,
    /// Answer is two relationships away.
    TwoHop,
}

impl QueryKind {
    pub fn label(&self) -> &'static str {
        match self {
            QueryKind::VectorOnly => "vector-only",
            QueryKind::Direct => "direct",
            QueryKind::OneHop => "one-hop",
            QueryKind::TwoHop => "two-hop",
        }
    }

    /// Hops a traversal must take to reach the answer.
    pub fn hops(&self) -> usize {
        match self {
            QueryKind::VectorOnly | QueryKind::Direct => 0,
            QueryKind::OneHop => 1,
            QueryKind::TwoHop => 2,
        }
    }

    fn sample(rng: &mut SplitMix64) -> Self {
        let u = rng.next_f64();
        if u < 0.20 {
            QueryKind::VectorOnly
        } else if u < 0.45 {
            QueryKind::Direct
        } else if u < 0.75 {
            QueryKind::OneHop
        } else {
            QueryKind::TwoHop
        }
    }
}

/// One query in the workload.
#[derive(Debug, Clone, Copy)]
pub struct Query {
    pub kind: QueryKind,
    /// Multiplier on the average degree: a dense query expands more.
    pub fanout: f64,
    /// Whether the neighbourhood is already materialised for this query.
    pub cached: bool,
}

impl Query {
    fn sample(rng: &mut SplitMix64, cache_coverage: f64) -> Self {
        let kind = QueryKind::sample(rng);
        let fanout = 0.8 + 0.4 * rng.next_f64();
        let cached = rng.next_f64() < cache_coverage;
        Self { kind, fanout, cached }
    }
}

/// The four plans the planner chooses between.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Plan {
    VectorFirst,
    GraphFirst,
    Hybrid,
    CachedNeighbourhood,
}

impl Plan {
    pub fn all() -> [Plan; 4] {
        [Plan::VectorFirst, Plan::GraphFirst, Plan::Hybrid, Plan::CachedNeighbourhood]
    }

    pub fn label(&self) -> &'static str {
        match self {
            Plan::VectorFirst => "vector-first",
            Plan::GraphFirst => "graph-first",
            Plan::Hybrid => "hybrid",
            Plan::CachedNeighbourhood => "cached-neighbourhood",
        }
    }

    /// Capability is a semantic property, not a statistic: the planner knows
    /// which shapes each plan can reach even when its cardinalities are wrong.
    pub fn can_answer(&self, kind: QueryKind) -> bool {
        match self {
            Plan::VectorFirst => matches!(kind, QueryKind::VectorOnly | QueryKind::Direct),
            Plan::GraphFirst | Plan::Hybrid | Plan::CachedNeighbourhood => {
                matches!(kind, QueryKind::Direct | QueryKind::OneHop | QueryKind::TwoHop)
            }
        }
    }

    /// Cost of the work this plan performs, in abstract units.
    pub fn cost(&self, q: &Query, stats: &GraphStats, model: &CostModel) -> f64 {
        let vec_scan = stats.n_vectors as f64 * model.vec_unit;
        let degree = stats.avg_degree * q.fanout;
        // A direct lookup still needs one traversal step to reach the entity.
        let hops = q.kind.hops().max(1);
        match self {
            Plan::VectorFirst => vec_scan,
            Plan::GraphFirst => model.expansion(hops, degree),
            Plan::Hybrid => model.seed_probe * vec_scan + model.expansion(hops, degree * model.hybrid_fanout),
            Plan::CachedNeighbourhood => {
                if q.cached {
                    model.cache_hit
                } else {
                    model.seed_probe * vec_scan + model.expansion(hops, degree * model.hybrid_fanout)
                }
            }
        }
    }
}

/// Statistics the planner has (possibly wrong) and the ground truth.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    pub n_vectors: usize,
    pub avg_degree: f64,
    /// Fraction of the graph that belongs to the querying tenant.
    pub tenant_fraction: f64,
}

impl GraphStats {
    /// Ground truth for the configured workload.
    pub fn from_config(cfg: &BenchConfig) -> Self {
        Self { n_vectors: cfg.n_vectors, avg_degree: cfg.avg_degree, tenant_fraction: cfg.tenant_fraction }
    }

    /// What the planner believes: each statistic is off by up to `error`.
    pub fn estimate(&self, rng: &mut SplitMix64, error: f64) -> Self {
        let jitter = |rng: &mut SplitMix64| 1.0 + (rng.next_f64() * 2.0 - 1.0) * error;
        Self {
            n_vectors: ((self.n_vectors as f64 * jitter(rng)).round() as usize).max(1),
            avg_degree: (self.avg_degree * jitter(rng)).max(1.0),
            tenant_fraction: self.tenant_fraction,
        }
    }

    /// Tenant of a node, derived from its id so the mapping is stable.
    pub fn node_tenant(&self, id: u64) -> u8 {
        let h = id.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40;
        (h % 2) as u8 + 1
    }

    fn neighbours(&self, id: u64, degree: f64) -> Vec<u64> {
        let n = degree.round().max(1.0) as usize;
        (0..n)
            .map(|j| {
                id.wrapping_mul(31)
                    .wrapping_add((j as u64).wrapping_mul(17))
                    .wrapping_add(7)
                    % 1_000_000
            })
            .collect()
    }

    /// Bounded traversal that applies the tenant filter **before** a candidate
    /// enters the frontier. A hit from another tenant is never returned.
    pub fn traverse(&self, seed: u64, tenant: u8, hops: usize, degree: f64) -> Vec<u64> {
        let mut visited = Vec::new();
        let mut frontier = vec![seed];
        for _ in 0..hops {
            let mut next = Vec::new();
            for id in frontier.iter().copied() {
                for cand in self.neighbours(id, degree) {
                    if self.node_tenant(cand) != tenant {
                        continue;
                    }
                    next.push(cand);
                }
            }
            visited.extend(next.iter().copied());
            frontier = next;
        }
        visited
    }

    /// Same walk **without** the tenant filter. Exists so the test suite can show
    /// the filter is doing real work rather than passing vacuously.
    pub fn traverse_unfiltered(&self, seed: u64, hops: usize, degree: f64) -> Vec<u64> {
        let mut visited = Vec::new();
        let mut frontier = vec![seed];
        for _ in 0..hops {
            let mut next = Vec::new();
            for id in frontier.iter().copied() {
                next.extend(self.neighbours(id, degree));
            }
            visited.extend(next.iter().copied());
            frontier = next;
        }
        visited
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    AlwaysVector,
    AlwaysGraph,
    CostBased,
}

impl Policy {
    pub fn all() -> [Policy; 3] {
        [Policy::AlwaysVector, Policy::AlwaysGraph, Policy::CostBased]
    }

    pub fn label(&self) -> &'static str {
        match self {
            Policy::AlwaysVector => "always-vector",
            Policy::AlwaysGraph => "always-graph",
            Policy::CostBased => "cost-based",
        }
    }

    /// The plan this policy issues for a query.
    pub fn choose(&self, q: &Query, estimate: &GraphStats, model: &CostModel) -> Plan {
        match self {
            Policy::AlwaysVector => Plan::VectorFirst,
            Policy::AlwaysGraph => Plan::GraphFirst,
            Policy::CostBased => {
                let mut best = Plan::VectorFirst;
                let mut best_cost = f64::INFINITY;
                for plan in Plan::all() {
                    if !plan.can_answer(q.kind) {
                        continue;
                    }
                    let c = plan.cost(q, estimate, model);
                    if c < best_cost {
                        best_cost = c;
                        best = plan;
                    }
                }
                best
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchConfig {
    pub seed: u64,
    pub queries: usize,
    pub n_vectors: usize,
    pub avg_degree: f64,
    pub cache_coverage: f64,
    /// Fractional error on every statistic the planner sees.
    pub estimate_error: f64,
    pub tenant_fraction: f64,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self { seed: 0x9E37_79B9, queries: 2_000, n_vectors: 40_000, avg_degree: 12.0, cache_coverage: 0.5, estimate_error: 0.2, tenant_fraction: 0.5 }
    }
}

impl BenchConfig {
    pub fn test_config() -> Self {
        Self { seed: 7, queries: 500, n_vectors: 40_000, avg_degree: 12.0, cache_coverage: 0.5, estimate_error: 0.2, tenant_fraction: 0.5 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyResult {
    pub policy: String,
    pub mean_cost: f64,
    pub p95_cost: f64,
    /// Mean cost divided by the mean per-query optimum.
    pub vs_optimal: f64,
    pub plan_accuracy: f64,
    pub mean_estimate_error: f64,
    pub tenant_violations: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchReport {
    pub bounty: String,
    pub hip: String,
    pub honesty_labels: Vec<String>,
    pub config: BenchConfig,
    pub results: Vec<PolicyResult>,
    pub notes: Vec<String>,
}

/// Cost of the best plan for one query, and the cost of the chosen plan.
fn run_query(policy: Policy, q: &Query, truth: &GraphStats, estimate: &GraphStats) -> (f64, f64, bool, usize) {
    let real = CostModel::truth();
    let believed = CostModel::planner();

    let optimal = Plan::all()
        .iter()
        .filter(|p| p.can_answer(q.kind))
        .map(|p| p.cost(q, truth, &real))
        .fold(f64::INFINITY, f64::min);

    let chosen = policy.choose(q, estimate, &believed);
    let answered = chosen.can_answer(q.kind);
    // A plan that cannot answer still pays for the work it did, then a fallback runs.
    let cost = if answered { chosen.cost(q, truth, &real) } else { chosen.cost(q, truth, &real) + RETRY_FACTOR * optimal };
    let accurate = answered && cost <= optimal * 1.10 + 1e-9;

    // Exercise the traversal so tenant isolation is measured, not asserted.
    let violations = if matches!(chosen, Plan::GraphFirst | Plan::Hybrid | Plan::CachedNeighbourhood) {
        let tenant = 1u8;
        truth
            .traverse(42, tenant, q.kind.hops().max(1), truth.avg_degree * q.fanout)
            .into_iter()
            .filter(|id| truth.node_tenant(*id) != tenant)
            .count()
    } else {
        0
    };

    (cost, optimal, accurate, violations)
}

pub fn evaluate(policy: Policy, cfg: &BenchConfig) -> PolicyResult {
    let mut rng = SplitMix64::new(cfg.seed);
    let truth = GraphStats::from_config(cfg);
    let estimate = truth.estimate(&mut rng, cfg.estimate_error);
    let real = CostModel::truth();
    let believed = CostModel::planner();

    let mut costs = Vec::with_capacity(cfg.queries);
    let mut optimal_sum = 0.0;
    let mut accurate = 0usize;
    let mut violations = 0usize;
    let mut estimate_err = Vec::with_capacity(cfg.queries);

    for _ in 0..cfg.queries {
        let q = Query::sample(&mut rng, cfg.cache_coverage);
        let (cost, optimal, ok, v) = run_query(policy, &q, &truth, &estimate);
        costs.push(cost);
        optimal_sum += optimal;
        if ok {
            accurate += 1;
        }
        violations += v;

        let chosen = policy.choose(&q, &estimate, &believed);
        let predicted = chosen.cost(&q, &estimate, &believed);
        let actual = chosen.cost(&q, &truth, &real).max(1e-9);
        estimate_err.push((predicted - actual).abs() / actual);
    }

    let mean_cost = costs.iter().sum::<f64>() / cfg.queries as f64;
    let optimal_mean = optimal_sum / cfg.queries as f64;
    let mut sorted = costs.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((sorted.len() as f64 - 1.0) * 0.95).round() as usize;

    PolicyResult {
        policy: policy.label().into(),
        mean_cost,
        p95_cost: sorted[idx.min(sorted.len() - 1)],
        vs_optimal: mean_cost / optimal_mean.max(1e-9),
        plan_accuracy: accurate as f64 / cfg.queries as f64,
        mean_estimate_error: estimate_err.iter().sum::<f64>() / cfg.queries as f64,
        tenant_violations: violations,
    }
}

pub fn run_bench(cfg: &BenchConfig) -> Result<BenchReport, PlannerError> {
    if cfg.queries == 0 {
        return Err(PlannerError::InvalidConfig("queries must be positive".into()));
    }
    if cfg.n_vectors == 0 {
        return Err(PlannerError::InvalidConfig("n_vectors must be positive".into()));
    }
    if !(cfg.avg_degree >= 1.0) {
        return Err(PlannerError::InvalidConfig("avg_degree must be at least 1".into()));
    }
    if !(0.0..=1.0).contains(&cfg.cache_coverage) {
        return Err(PlannerError::InvalidConfig("cache_coverage must be in 0..=1".into()));
    }

    let results = Policy::all().iter().map(|p| evaluate(*p, cfg)).collect::<Vec<_>>();
    let mut notes = Vec::new();
    if let (Some(av), Some(ag), Some(cb)) = (
        results.iter().find(|r| r.policy == "always-vector"),
        results.iter().find(|r| r.policy == "always-graph"),
        results.iter().find(|r| r.policy == "cost-based"),
    ) {
        notes.push(format!(
            "cost-based planning is {:.2}x cheaper than always-vector and {:.2}x cheaper than always-graph, and lands within 10% of the per-query optimum on {:.1}% of queries",
            av.vs_optimal / cb.vs_optimal,
            ag.vs_optimal / cb.vs_optimal,
            cb.plan_accuracy * 100.0
        ));
        notes.push(format!(
            "the planner is scored against a cost model it does not know: it sees statistics wrong by up to {:.0}%, underestimates the vector seed probe by 2.4x and does not model the per-hop round trip, so it still pays {:.1}% over the per-query optimum and misses {:.1}% of plans",
            cfg.estimate_error * 100.0,
            (cb.vs_optimal - 1.0) * 100.0,
            (1.0 - cb.plan_accuracy) * 100.0
        ));
        notes.push(format!(
            "tenant isolation is measured by walking the graph, not assumed: {} violations across all policies",
            results.iter().map(|r| r.tenant_violations).sum::<usize>()
        ));
    }

    Ok(BenchReport {
        bounty: "hanzoai/graph#5 Native graph storage and a cost-based graph/vector query planner".into(),
        hip: "HIP-1347: Event-Driven Agent Reliability".into(),
        honesty_labels: vec![
            "no graph database, vector index, GPU or LLM is executed - cost is a modelled function of hand-set constants".into(),
            "graph degree, cache coverage, tenant mix, statistics error and the gap between the planner's cost model and the real one are hand-set, not measured".into(),
            "the workload mix is synthetic and fixed per seed, not sampled from production traffic".into(),
            "All figures are simulation outputs on the authoring host, NOT Hanzo expectations until reproduced against real traces on real storage.".into(),
        ],
        config: cfg.clone(),
        results,
        notes,
    })
}

pub fn write_results_json(report: &BenchReport, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(report)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find<'a>(r: &'a [PolicyResult], p: &str) -> &'a PolicyResult {
        r.iter().find(|x| x.policy == p).expect(p)
    }

    #[test]
    fn cost_based_beats_both_fixed_policies() {
        let report = run_bench(&BenchConfig::test_config()).unwrap();
        let cb = find(&report.results, "cost-based");
        let av = find(&report.results, "always-vector");
        let ag = find(&report.results, "always-graph");
        assert!(cb.mean_cost < av.mean_cost, "cost-based {:.1} vs always-vector {:.1}", cb.mean_cost, av.mean_cost);
        assert!(cb.mean_cost < ag.mean_cost, "cost-based {:.1} vs always-graph {:.1}", cb.mean_cost, ag.mean_cost);
        assert!(cb.vs_optimal < 1.30, "cost-based should stay near the optimum, got {:.2}x", cb.vs_optimal);
    }

    /// The planner is scored against a cost model it does not know, so it must be
    /// good but not perfect. A perfect score would mean the benchmark is circular.
    #[test]
    fn planner_is_good_but_not_circularly_perfect() {
        let report = run_bench(&BenchConfig::test_config()).unwrap();
        let cb = find(&report.results, "cost-based");
        assert!(cb.plan_accuracy > 0.50, "plan accuracy too low: {:.3}", cb.plan_accuracy);
        assert!(cb.plan_accuracy < 0.99, "plan accuracy too high - is the truth model the planner's own? {:.3}", cb.plan_accuracy);
        assert!(cb.vs_optimal > 1.001, "cost-based must pay something for its wrong model, got {:.4}x", cb.vs_optimal);
        assert!(cb.mean_estimate_error > 0.01, "estimate error should be visible, got {:.4}", cb.mean_estimate_error);
    }

    #[test]
    fn tenant_filter_runs_before_expansion() {
        let stats = GraphStats::from_config(&BenchConfig::test_config());
        let tenant = 1u8;
        let filtered = stats.traverse(42, tenant, 2, 12.0);
        assert!(!filtered.is_empty(), "traversal should return in-tenant nodes");
        assert!(filtered.iter().all(|id| stats.node_tenant(*id) == tenant));

        // Without the filter the same walk crosses tenant boundaries: the filter
        // is not vacuous.
        let unfiltered = stats.traverse_unfiltered(42, 2, 12.0);
        assert!(unfiltered.iter().any(|id| stats.node_tenant(*id) != tenant));
    }

    #[test]
    fn no_policy_reports_tenant_violations() {
        let report = run_bench(&BenchConfig::test_config()).unwrap();
        for r in &report.results {
            assert_eq!(r.tenant_violations, 0, "{} leaked another tenant", r.policy);
        }
    }

    #[test]
    fn results_are_deterministic() {
        let cfg = BenchConfig::test_config();
        let a = run_bench(&cfg).unwrap();
        let b = run_bench(&cfg).unwrap();
        assert_eq!(a.results.len(), b.results.len());
        for (x, y) in a.results.iter().zip(b.results.iter()) {
            assert_eq!(x.policy, y.policy);
            assert!((x.mean_cost - y.mean_cost).abs() < 1e-12);
            assert_eq!(x.tenant_violations, y.tenant_violations);
        }
    }

    #[test]
    fn bad_config_rejected() {
        let mut c = BenchConfig::test_config();
        c.queries = 0;
        assert!(run_bench(&c).is_err());
        let mut c = BenchConfig::test_config();
        c.n_vectors = 0;
        assert!(run_bench(&c).is_err());
        let mut c = BenchConfig::test_config();
        c.cache_coverage = 1.5;
        assert!(run_bench(&c).is_err());
    }
}
