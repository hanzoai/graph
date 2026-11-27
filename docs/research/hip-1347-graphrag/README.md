# HIP-1347 Research Dossier: Community Detection, Hierarchical Summaries & Incremental Invalidation for Global Questions (GraphRAG — `hanzoai/graph#3`)

## PR Convergence Metadata (`hanzoai/.github/PR_CONVERGENCE.md`)

- **Canonical Owning Repository**: `hanzoai/graph` (`graphrag/`, `cmd/graphrag-bench/`, `docs/research/hip-1347-graphrag/`)
- **Qualification Criteria**: Pure Go standard library (`0` new dependencies in `go.mod` / `go.sum`), Apache-2.0 clean-room implementation, bit-exact `DropDerivedIndexes`/`RebuildAll` SHA-256 parity, deterministic multi-level L0 → L1 → L2 community hierarchy with Leiden connectivity guarantee, targeted leaf + strict ancestor incremental invalidation, explicit `QueryRouter` gate preventing GraphRAG on exact lookups, `0%` tenant/ACL leak rate across global summaries and local hits (including live ACL revocation and existence-oracle probes), `go vet` + `go test -race -count=1` green.
- **Verification Evidence**: Raw machine-readable receipts in [`results.json`](results.json) generated live from `go run ./cmd/graphrag-bench --repo-root . --seed 1347 --top-k 5 --iters 10` on `Go go1.26.8 (linux/amd64, 2 CPUs)` over `98` real upstream `hanzoai/graph` Go source files parsed via `go/ast`, `3` repository docs, `24` package-grounded versioned engineering regression/ADR records, and `3` multi-tenant/restricted ACL canary records (`128` canonical documents total).
- **Security Review**: Strict `(TenantID, ACLScopeKey)` partitioning of all derived entity graphs, community trees, community summaries, and BM25 chunk DF statistics (`CanaryLeaksInGlobalSummaries=0`, `CanaryLeaksInLocalHits=0`, `CrossTenantPartitionsLeaked=0`, `ExistenceOracleLeaks=0`, `PostRevocationLeaks=0` across `35` probes); deny-by-default empty ACLs; uniform `ErrNotFound` on unauthorized/cross-tenant probes; deep-copy immutability on all inputs and outputs.
- **Rollout & Rollback Plan**: Self-contained additive package (`graphrag/`) and benchmark CLI (`cmd/graphrag-bench/`) with zero edits to `go.mod`, `go.sum`, or existing `hanzoai/graph` packages; rollback is a single-commit revert with zero schema or storage migration impact.

---

## 1. Architecture, Derived Community Hierarchy & Reference License Audit

### Clean-Room Derived GraphRAG Architecture (`graphrag/`)

1. **Canonical Truth vs. Derived Projections (`graphrag/types.go`, `graphrag/engine.go`)**:
   - Canonical records (`CanonicalDocument`) carry stable `(SourceID, Version, TenantID, ACL)` identities and are stored strictly separately from all derived state.
   - All extracted entities, weighted relations, multi-level communities (`L0 → L1 → L2`), hierarchical community summaries, and local BM25+dense chunk tables are strictly **derived** projections. Calling `DropDerivedIndexes()` discards all derived state while leaving canonical documents intact; calling `RebuildAll()` reconstructs the entire derived hierarchy bit-for-bit (verified by deterministic SHA-256 digest `8c62fa92d358b5d4ef1e2f311d67571f902c3c5c07123a18c8e732b4ba865dff`).
2. **ACL-Partitioned Derived Graphs & Summaries**:
   - Within each `TenantID`, derived graphs and community trees are strictly partitioned by canonical `ACLScopeKey = SHA256(TenantID || sorted(ACL))`.
   - Restricted documents (`ACL: ["role:security-incident"]`) and foreign tenant documents (`TenantID: "tenant-beta"`) never share an entity graph, community, summary, or BM25 document-frequency map with `["*"]` or `["role:eng"]` documents.
   - When a document's ACL changes (e.g., restricted from `["*"]` to `["role:security-incident"]` or revoked to empty `nil`), `UpsertDocument` triggers `incremental_acl_move`, retracting the document from the old partition and refreshing only the affected leaf + ancestor summaries so unauthorized callers immediately see zero residual mentions.
3. **Deterministic Leiden/Louvain Hierarchical Community Detection (`graphrag/community.go`)**:
   - **Level 0 (L0 — Leaf Communities)**: Deterministic local-moving Newman-Girvan weighted modularity optimization ($\gamma_0 = 1.25$) over sorted canonical node IDs, followed by **Leiden-style connectivity refinement** splitting any disconnected subgraph so every L0 community is internally connected.
   - **Level 1 (L1 — Mid-Level Subsystem/Domain Communities)**: Exact super-graph coarsening preserving total edge/self-loop weight $2m$, followed by deterministic modularity optimization ($\gamma_1 = 0.70$) and connectivity refinement.
   - **Level 2 (L2 — Macro Corpus Communities)**: Second super-graph coarsening pass ($\gamma_2 = 0.30$) grouping L1 subsystems into compact macro themes with explicit `ParentID` and `ChildIDs` lineage.
4. **Targeted Incremental Summary Invalidation (`graphrag/engine.go`)**:
   - Per-document contribution tables (`entityContribs`, `relContribs`, `docExtractions`) allow $O(|\Delta|)$ exact retraction on document updates, ACL moves, and deletions.
   - Only leaf L0 communities touching `dirtyEntities` and their strict ancestor chain (`L0 → L1 → L2`) are invalidated and re-summarized; untouched L0/L1/L2 community summaries are reused verbatim (`Revision` unchanged).
5. **Explicit Global vs. Local Query Routing Gate (`graphrag/router.go`)**:
   - Per `hanzoai/graph#3`, GraphRAG map-reduce is **not** the default path. `QueryRouter` scores corpus-wide thematic synthesis cues (`"across our repositories"`, `"largest sources"`, `"systemic"`, `"cross-package"`, `"portfolio"`) against exact/local lookup cues (file paths `posting/mvcc.go`, incident IDs `REG-007`, exact symbols `SearchLayer`, `"where is"`, `"exact"`).
   - `ModeAuto` dispatches corpus-wide questions to `ModeGlobalL1` (or `ModeGlobalL2` for executive/portfolio prompts) and routes exact lookups to `ModeLocalHybrid` (ordinary top-K BM25 + 128-dim dense vector retrieval).

### Independent License & Dependency Audit

- **Implementation License**: Clean-room **Apache-2.0** Go standard library (`0` new dependencies; `go.mod` and `go.sum` completely untouched).
- **Study Reference Attribution**: Inspired by Edge et al., *"From Local to Global: A Graph RAG Approach to Query-Focused Summarization"* (arXiv:2404.16130, 2024). The upstream Microsoft GraphRAG repository (MIT) is explicitly noted as research software in maintenance mode with expensive LLM indexing; **zero code** is copied from the upstream reference implementation.

---

## 2. Controlled Evaluation Protocol & Reproducibility

All 5 evaluated strategies (`ordinary_hybrid_topk`, `graphrag_global_l0`, `graphrag_global_l1`, `graphrag_global_l2`, `routed_auto_gate`) execute on the **exact same canonical corpus**, **exact same caller ACL context**, **exact same TopK (`5`) and synthesis token budget (`4000`)**, **exact same deterministic extractive synthesizer**, **exact same pricing table** (`$3.00 / 1M` prompt tokens, `$15.00 / 1M` completion tokens), and **exact same judge**.

Reproduce all receipts locally:

```bash
go test -race -count=1 -v ./graphrag/...
go run ./cmd/graphrag-bench --repo-root . --seed 1347 --top-k 5 --iters 10 --out-json docs/research/hip-1347-graphrag/results.json
```

Run on an external public JSON dataset bundle:

```bash
go run ./cmd/graphrag-bench --external-dataset /path/to/workload.json --top-k 5 --token-budget 4000
```

### Corpus Composition & Honesty Disclosures

- **Real Go Codebase (`98` files)**: Parsed directly from canonical upstream `hanzoai/graph` packages (`algo`, `buildvars`, `codec`, `dql`, `lex`, `posting`, `protos/pb`, `schema`, `tok`, `tok/hnsw`, `tok/index`, `tok/options`, `tok/constraints`, `types`, `types/facets`, `x`) via `go/parser` + `go/ast`.
- **Repository Documentation (`3` files)**: `README.md`, `hanzo.yml`, `NOTICE`.
- **Package-Grounded Engineering Regressions & ADRs (`24` records)**: Deterministic versioned incident/regression reports (`REG-001`..`REG-020`) and architecture decision records (`ADR-001`..`ADR-004`) anchored to real `hanzoai/graph` files so cross-document regression themes span 20+ separate canonical records (> `TopK=5`).
- **ACL & Multi-Tenant Canary Records (`3` records)**: Two restricted `tenant-alpha` security records (`ACL: ["role:security-incident"]`) and one `tenant-beta` wildcard record (`ACL: ["*"]`) used exclusively for leak/oracle auditing.
- **Synthesizer Disclosure**: The default `ExtractiveHierarchicalSummarizer` is a zero-dependency deterministic stdlib synthesizer (no remote LLM API required) with full prompt/completion token accounting so comparisons across strategies are bit-exact and reproducible; production deployments can inject an LLM-backed `Summarizer` via `Config.Summarizer` without changing community detection, ACL partitioning, incremental invalidation, or routing.

---

## 3. Falsifiable Hypothesis & Measured Negative Regime

**Hypothesis**:
1. On **corpus-wide thematic synthesis** (`"What are the largest sources of engineering regressions across our repositories?"`) and **cross-package architecture** questions spanning many documents (> `TopK`), GraphRAG hierarchical map-reduce over derived community summaries will decisively outperform ordinary top-K hybrid RAG on cross-document synthesis accuracy, theme coverage, multi-source recall, and downstream task success.
2. **Falsifiable Negative Regime**: On **exact single-file, symbol, or incident lookups** (`"What is the exact root cause in REG-007?"`, `"Where is SearchLayer implemented in tok/hnsw/search_layer.go?"`), forcing GraphRAG map-reduce will **lose badly** to ordinary top-K hybrid RAG in latency, token cost, and exact-hit accuracy — confirming that GraphRAG must be gated behind an explicit query router and never used as the default retrieval path.
3. **Incremental Refresh**: Targeted leaf + strict ancestor summary invalidation will cut refresh latency and summary token cost by > 10x / > 50% compared to full community rebuilds.

### Measured Negative Regime: `exact_lookup_negative_regime` (`6` queries, `10` iters)

| Strategy | Acc F1 | Theme Cov | Source Recall | Task Success | Route Acc | Avg Tokens | Latency p50 / p95 (ms) | Avg Cost / Query | ACL Leaks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ordinary_hybrid_topk` | **88.9%** | 100.0% | 100.0% | **100.0%** | 100.0% | **528** | **0.37 / 0.49** | **$0.00497** | 0 |
| `graphrag_global_l0` | 5.6% | 100.0% | 100.0% | 0.0% | 0.0% | 35,326 | 22.81 / 30.10 | $0.24753 | 0 |
| `graphrag_global_l1` | 3.2% | 100.0% | 100.0% | 0.0% | 0.0% | 11,950 | 6.47 / 11.32 | $0.08684 | 0 |
| `graphrag_global_l2` | 3.3% | 100.0% | 100.0% | 0.0% | 0.0% | 6,342 | 3.16 / 6.52 | $0.04825 | 0 |
| **`routed_auto_gate`** | **88.9%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **528** | **0.39 / 0.94** | **$0.00497** | **0** |

**Negative Regime Verdict**: Forcing GraphRAG L1 on exact lookups burns **22.6x more tokens** (`11,950` vs `528`), costs **17.5x more** (`$0.08684` vs `$0.00497`), runs **17.5x slower** (`6.47 ms` vs `0.37 ms`), and collapses exact-lookup task success from **100.0% → 0.0%** due to corpus-wide citation dilution. Forcing L0 is **66.9x more expensive in tokens** (`35,326`) and **61.6x slower** (`22.81 ms`). `routed_auto_gate` accurately routes 100% of exact lookups to `ordinary_hybrid_topk`.

---

## 4. Comparative Benchmark Receipts (Accuracy, Coverage, Token Cost, Build Cost, Incremental Refresh)

### 4.1 Global Synthesis & Cross-Package Architecture Regimes

#### Regime: `global_thematic_synthesis` (`4` queries, `10` iters)

| Strategy | Acc F1 | Theme Cov | Source Recall | Task Success | Avg Tokens | Latency p50 / p95 (ms) | Avg Cost / Query | Comms Mapped | ACL Leaks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ordinary_hybrid_topk` | 69.6% | 88.1% | 11.2% | 0.0% | 599 | 0.43 / 3.18 | $0.00557 | 0.0 | 0 |
| `graphrag_global_l0` | **96.2%** | **100.0%** | **100.0%** | **100.0%** | 40,267 | 22.94 / 38.63 | $0.31463 | 68.0 | 0 |
| `graphrag_global_l1` | 92.8% | **100.0%** | **100.0%** | 75.0% | 14,852 | 6.63 / 10.71 | $0.11729 | 16.0 | 0 |
| `graphrag_global_l2` | 85.6% | **100.0%** | **100.0%** | 75.0% | 6,993 | 3.18 / 5.74 | $0.05463 | 9.0 | 0 |
| **`routed_auto_gate`** | **92.8%** | **100.0%** | **100.0%** | **75.0%** | **14,852** | **7.06 / 18.20** | **$0.11729** | **16.0** | **0** |

#### Regime: `cross_package_architecture` (`3` queries, `10` iters)

| Strategy | Acc F1 | Theme Cov | Source Recall | Task Success | Avg Tokens | Latency p50 / p95 (ms) | Avg Cost / Query | Comms Mapped | ACL Leaks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ordinary_hybrid_topk` | 79.2% | 83.3% | 41.7% | 0.0% | 650 | 0.45 / 0.69 | $0.00591 | 0.0 | 0 |
| `graphrag_global_l0` | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 41,014 | 24.86 / 29.37 | $0.31937 | 68.0 | 0 |
| `graphrag_global_l1` | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 14,970 | 6.91 / 11.33 | $0.11764 | 16.0 | 0 |
| `graphrag_global_l2` | 93.7% | **100.0%** | **100.0%** | **100.0%** | 7,063 | 3.24 / 5.43 | $0.05484 | 9.0 | 0 |
| **`routed_auto_gate`** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **12,334** | **6.58 / 16.39** | **$0.09671** | **13.7** | **0** |

#### Overall Combined Across All 13 Queries (`overall_all_regimes`)

| Strategy | Acc F1 | Theme Cov | Source Recall | Task Success | Route Acc | Avg Tokens | Latency p50 / p95 (ms) | Avg Cost / Query | ACL Leaks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ordinary_hybrid_topk` | 80.7% | 92.5% | 59.2% | 46.2% | 46.2% | 578 | 0.41 / 1.69 | $0.00537 | 0 |
| `graphrag_global_l0` | 55.3% | 100.0% | 100.0% | 53.8% | 53.8% | 38,159 | 23.27 / 32.71 | $0.28475 | 0 |
| `graphrag_global_l1` | 53.1% | 100.0% | 100.0% | 46.2% | 53.8% | 13,540 | 6.61 / 11.43 | $0.10332 | 0 |
| `graphrag_global_l2` | 49.5% | 100.0% | 100.0% | 46.2% | 53.8% | 6,709 | 3.18 / 6.14 | $0.05173 | 0 |
| **`routed_auto_gate`** | **92.7%** | **100.0%** | **100.0%** | **92.3%** | **100.0%** | **7,660** | **3.27 / 14.73** | **$0.06070** | **0** |

### 4.2 Offline Build Cost & Hierarchy Structural Quality

| Metric | Measured Value |
|---|---:|
| Canonical Documents Indexed | `128` across `3` ACL/tenant partitions |
| Derived Graph Size | `913` entities, `2,792` undirected weighted relations |
| Multi-Level Community Pyramid | `L0 = 77`, `L1 = 19`, `L2 = 12` (`108` summaries total) |
| Entity-Weighted Newman-Girvan Modularity | `Q_L0 = 0.464`, `Q_L1 = 0.550` |
| Offline Full Build Wall Time | `478.14 ms` |
| Offline Build Summary Token Cost | `74,931` prompt + `27,172` completion = `102,103` tokens (`$0.63237`) |
| Drop/Rebuild Bit-Exact Parity (`SHA-256`) | `true` (`8c62fa92d358b5d4...`) |

### 4.3 Incremental Summary Invalidation vs. Full Community Rebuild

| Workload Case | Operation | Incremental Time (ms) | Full Rebuild Time (ms) | Speedup | Communities Resummarized / Reused | Reused % | Incremental Tokens | Full Rebuild Tokens | Summary Token Savings |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `single_doc_update` | `incremental_update` | **11.73** | 498.72 | **42.52x** | 8 / 85 | **91.4%** | **10,901** | 94,549 | **88.5%** |
| `three_doc_update_batch` | `incremental_3doc_batch` | **28.13** | 498.72 | **17.73x** | 8 / 84 (avg) | **91.3%** | **43,435** | 94,549 | **54.1%** |
| `single_doc_delete` | `incremental_delete` | **5.70** | 498.72 | **87.52x** | 7 / 86 | **92.5%** | **10,915** | 94,549 | **88.5%** |

### 4.4 Security & Multi-Tenant ACL Audit

| Audit Gate | Measured Result |
|---|---:|
| Total Adversarial & Revocation Probes | `35` |
| Restricted Canary Leaks in Global Summaries (L0/L1/L2) | `0` |
| Restricted Canary Leaks in Local Hybrid Hits | `0` |
| Cross-Tenant Partitions Leaked | `0` |
| Direct Community-ID Existence-Oracle Leaks | `0` |
| Post-ACL-Revocation Residual Canary Leaks | `0` |
| Drop/Rebuild Bit-Exact SHA-256 Parity | `true` |

---

## 5. Honest Failure-Mode Analysis & Operational Caveats

1. **Upfront Offline Indexing Cost**: Even on a 128-document repository, building all 108 hierarchical summaries consumes `102,103` tokens (`$0.63` at `$3/$15` LLM rates). On a 50,000-document monorepo with an external LLM summarizer, naive full rebuilds would cost hundreds of dollars — which is why incremental leaf+ancestor invalidation (`88.5%` token savings per single-doc update) and periodic offline compaction are mandatory in production.
2. **Granularity Trade-Off (L0 vs. L1 vs. L2)**: Notice that on `global_thematic_synthesis`, `graphrag_global_l1` achieved `75.0%` task success (`92.8%` AccF1) versus `graphrag_global_l0`'s `100.0%` task success (`96.2%` AccF1). Rolling up 77 leaf communities into 19 L1 summaries truncates lower-frequency findings (e.g., niche ops keywords like `minio`/`fips` in `GLOBAL-04`), trading `~25%` rare-keyword detail for a **2.7x token reduction** (`14,852` vs `40,267`) and **3.5x latency speedup** (`6.63 ms` vs `22.94 ms`).
3. **ACL Scope Fragmentation**: Partitioning derived graphs by exact `(TenantID, ACLScopeKey)` guarantees mathematically watertight `0%` summary leakage, but if a tenant creates hundreds of bespoke per-user ACL combinations, each distinct ACL scope maintains its own community tree. Production deployments should normalize ACLs to coarse team/role principals (RBAC groups) rather than unbounded per-user lists.
4. **Incremental Structural Drift**: Targeted incremental updates attach new entities to their strongest neighboring L0 community and refresh dirty L0 → L1 → L2 summaries without global re-clustering. After large corpus shifts (> 15–20% document churn), operators should schedule an offline `RebuildAll()` to re-optimize global Newman-Girvan modularity.

---

## 6. Production Integration Recommendation

- **Disposition**: **`MERGE`** as an opt-in global synthesis subsystem behind `QueryRouter` (`ModeAuto` or explicit `ModeGlobalL1` / `ModeGlobalL2`), **never** as the default exact-lookup retriever.
- **Recommended Deployment Policy**:
  1. Run `BuildAll()` offline on release/nightly cadences per RBAC ACL partition; apply `UpsertDocument` / `DeleteDocument` incremental leaf+ancestor invalidation on continuous PR/commit/incident streams.
  2. Gate all interactive queries through `QueryRouter`: exact symbol, file, DQL, and incident lookups stay on the sub-millisecond hybrid path (`0.37 ms`, `528 tokens`); corpus-wide thematic questions (`"largest sources of regressions across our repositories"`) escalate to `ModeGlobalL1` (`6.6 ms`, `100%` theme coverage & source recall) or `ModeGlobalL2` for budget-constrained executive summaries.
  3. Keep `DropDerivedIndexes()` + `RebuildAll()` wired into operator runbooks for zero-data-loss index recovery and schema upgrades.
