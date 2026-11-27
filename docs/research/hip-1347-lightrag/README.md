# HIP-1347 Research Dossier: Incremental Entity-Relationship Indexing (LightRAG) (`hanzoai/graph#1`)

## 1. Reference Architecture & Independent License Audit

- **Primary Reference**: LightRAG (*Simple and Fast Retrieval-Augmented Generation*, Guo et al., EMNLP 2025, [arXiv:2410.05779](https://arxiv.org/abs/2410.05779)).
- **License Audit**:
  - Upstream LightRAG reference codebase (`HKUDS/LightRAG`) is licensed under the **MIT License**.
  - This implementation in `lightrag/` and `cmd/lightrag-bench/` is a **clean-room, independent Apache-2.0 Go implementation** written strictly with the Go standard library (`go/parser`, `go/ast`, `go/token`, `crypto/sha256`, `hash/fnv`, `sync`, etc.).
  - **`go.mod` and `go.sum` are completely untouched — 0 new direct or transitive dependencies.**
  - **Evaluation Corpus License**: All `97` Go source files and `5` repository documentation files evaluated by `cmd/lightrag-bench` are real, versioned **Apache-2.0** files from `hanzoai/graph` itself (`algo/`, `buildvars/`, `codec/`, `dql/`, `lex/`, `posting/`, `schema/`, `tok/`, `types/`, `x/`, `README.md`, `dql/README.md`, `tok/options/README.md`, `go.mod`, `NOTICE`), augmented with `12` Apache-2.0 operational engineering records (commits, ADRs, incident reports, and agent outputs) grounded directly in those files and symbols. Zero third-party copyrighted datasets are vendored.
- **Architectural Core (HIP-1347 Shared Invariants)**:
  1. **Canonical Source Truth vs. Derived Indexes**: Every ingested artifact has a stable `(TenantID, SourceID, Version, ACL)` identity stored in the canonical source map. Chunks, typed entity-relationship graphs, canonical alias tables, incremental Okapi BM25 inverted indexes, 128-dim L2-normalized dense vector tables, and bounded query caches are strictly **DERIVED** projections. `DropDerivedIndexes()` discards every derived projection without touching canonical records (enforcing `ErrIndexesNotBuilt` on reads), and `RebuildAll()` deterministically reconstructs bit-identical derived state.
  2. **Incremental Subgraph Delta Updates ($O(|\Delta G|)$)**: Updating or deleting a `SourceRecord` retracts only the chunks, entity contributions, and relationship contributions attributed to that `SourceID`, removes orphaned nodes/edges/aliases/postings/vectors, deterministically re-merges surviving shared nodes/edges in sorted `SourceID` order, refreshes only affected BM25/vector entries, and atomically invalidates tenant query caches under the write lock.
  3. **Canonical Entity Resolution & Pluggable Extraction**: Exposes `Extractor` (plus `FuncExtractor`) and ships `DefaultExtractor`, combining Go AST extraction (`go/parser` for packages, imports, types, functions, methods, receiver bindings, and cross-package call edges) with structured technical-text extraction for commits, ADRs, incidents, and agent outputs. Deterministic canonicalization collapses duplicate aliases (`github.com/hanzoai/graph/posting` / `posting`, `*posting.List` / `posting.List`, `BadgerDB` / `zapdb`, etc.) and tracks exact duplicate-entity telemetry.
  4. **Dual-Level Retrieval**: Implements LightRAG `ModeLocal` (low-level entity + hub-damped 1-hop/2-hop neighborhood expansion), `ModeGlobal` (high-level relationship/theme retrieval), `ModeHybrid` (Reciprocal Rank Fusion of Local + Global + direct chunk grounding), and the matched `ModeOrdinaryHybrid` chunk-only BM25 + dense vector RRF baseline.
  5. **Strict Pre-Return Tenant & ACL Isolation**: Derived graphs are physically partitioned per tenant (`tenants[tenantID]`). Within a tenant, every chunk, entity, relationship, alias lookup, and neighborhood hop enforces ACL authorization *before* scoring, expansion, or return. For entities/relations straddling an ACL boundary (referenced by both public and restricted sources), descriptions, keywords, aliases, and `SourceIDs` are materialized strictly from authorized contributions, and existence-oracle queries on unauthorized tokens are blocked.

## 2. Matched Baselines, Controlled Evaluation Protocol & Reproducibility

All `4` retrieval systems are evaluated on the **exact same canonical corpus** (`114` sources, `895,824` bytes, `786` chunks), **exact same query suite** (`26` queries across `3` workload families + `6` post-mutation freshness checks + `16` adversarial ACL isolation probes), **exact same retrieval budget** (`TopK = 5`), **exact same grounded extractive answer synthesizer** (reading strictly top-`5` retrieved chunks and grounded subgraph trails), and **exact same judge** (Recall@K, MRR@K, Answer F1, Exact Match, Downstream Task Success, uncached latency percentiles, cached p50 latency, and `$/1M useful tokens`):

1. `ordinary_hybrid` — Matched shipping baseline: chunk-level Okapi BM25 + 128-dim dense cosine RRF with pre-return ACL filtering and no graph traversal.
2. `lightrag_local` — LightRAG low-level entity retrieval + hub-damped 2-hop neighborhood expansion.
3. `lightrag_global` — LightRAG high-level relationship/theme retrieval.
4. `lightrag_hybrid` — Full LightRAG dual-level fusion (`LocalWeight = 0.45`, `GlobalWeight = 0.35`, `ChunkWeight = 0.20`).

### Exact Reproducible Commands

```bash
# 1. Run race-checked unit, property, ACL isolation, immutability, and end-to-end tests
go test -race -v ./lightrag/...

# 2. Static analysis across package and benchmark CLI
go vet ./lightrag/... ./cmd/lightrag-bench/...

# 3. Reproduce raw benchmark measurements into docs/research/hip-1347-lightrag/results.json
go run ./cmd/lightrag-bench --repo-root . --out docs/research/hip-1347-lightrag/results.json --top-k 5 --iters 10 --seed 1347
```

### Honesty & Scope Disclosures

- **Host**: `go1.26.8` (`linux/amd64`, `2` CPUs), recorded `2026-11-26T15:45:42Z` (`seed = 1347`).
- **Deterministic Embeddings**: Zero-dependency 128-dim signed FNV-1a unigram + bigram + char-4gram L2-normalized dense embeddings with a `0.15` cosine noise floor; no external GPU or hosted embedding API is called in CI.
- **Extraction Cost Accounting**: `DefaultExtractor` runs locally at `$0.00` marginal API spend; `estimated_extract_cost_usd` (`$0.129182`) and `relations_per_dollar` (`17,541.11 rels/$`) report the transparent equivalent LLM-extraction proxy cost at `$0.15 / 1M` input tokens (`225,191` tokens) and `$0.60 / 1M` output tokens (`159,006` tokens) so cost-efficiency comparisons against LLM extractors remain apples-to-apples.
- **Context Token Cost Formula**: Retrieved context tokens priced at `$3.00 / 1M` input tokens adjusted by useful evidence ratio: `Cost/1M useful tokens = $3.00 * (total_tokens / gold_evidence_tokens)`.
- **Micro-Batch Calibration**: Fast sub-millisecond retrieval queries use calibrated micro-batches (`10x` for `ordinary_hybrid`, `3x` for LightRAG modes, `50x` for warm cache hits) to eliminate coarse VM timer rounding artifacts while measuring live wall-clock latencies.

## 3. Falsifiable Hypothesis & Negative / Boundary Regimes

**Hypothesis**: On versioned multi-file engineering repositories with ongoing commits, ADRs, and incident chains, incremental subgraph delta indexing (`O(|ΔG|)`) delivers >100x faster updates than full graph rebuilds with bit-exact derived-index equivalence, while dual-level LightRAG retrieval outperforms ordinary hybrid chunk search on cross-package multi-hop causality and cross-cutting thematic synthesis — at the explicit cost of measurable graph-traversal latency overhead on simple single-symbol lookups.

### Explicitly Measured Negative & Boundary Regimes

1. **`neg_simple_single_fact_overhead` (Where Plain Hybrid Wins)**:
   - On exact single-identifier lookups (`single_fact_exact`, `8` queries such as `CountMinSketch`, `SafeMutex`, `ReservedNamespace`, `BigFloatTokenizer`), `ordinary_hybrid` achieves **100.0% Recall@K, 1.000 MRR, 100.0% EM, and 100.0% Task Success** at **0.469 ms p50** (`$13.83 / 1M useful tokens`), whereas `lightrag_hybrid` matches accuracy (**100.0% Recall@K / 1.000 MRR / 100.0% EM**) at **6.140 ms p50** — a **13.09x retrieval latency overhead** (`$15.18 / 1M useful tokens`) with `+0.0%` accuracy gain.
   - **Operational takeaway**: Route simple single-symbol or exact-identifier queries directly to `ordinary_hybrid`; reserve LightRAG dual-level retrieval for multi-hop, dependency-tracing, and thematic synthesis queries.
2. **`neg_single_channel_ablation` (Why Dual-Level Fusion Is Required)**:
   - On cross-cutting thematic queries (`global_thematic`, `8` queries), neither single graph channel suffices alone: `lightrag_global` alone reaches `58.3%` Recall@K, `0.594` MRR, `0.0%` EM, and `75.0%` Task Success; `lightrag_local` alone reaches `68.8%` Recall@K, `0.667` MRR, `50.0%` EM, and `87.5%` Task Success; fusing Local + Global + direct chunk grounding in `lightrag_hybrid` lifts ranking and completion to **0.812 MRR, 62.5% EM, and 100.0% Task Success**.
3. **`neg_acl_bridge_leakage_control` (Security Failure Mode Without Pre-Return ACL Scoping)**:
   - When multiple tenants (`tenant-alpha`, `tenant-beta`) and ACL roles (`role:eng`, `role:sec`) share entity names (`posting.List`, `x.Keys`), an unfiltered shared-entity graph leaks restricted data on **16/16 (100.0%)** adversarial probes across all `4` modes, whereas LightRAG's per-tenant partitioning and pre-return ACL scoping achieve **0/16 (0.0%)** leakage (`60` restricted candidates filtered before return).

## 4. Bounded Deliverable Scope & Canonical Integration Plan

### Deliverables in This PR

- `lightrag/doc.go` — Package documentation and HIP-1347 invariants.
- `lightrag/types.go` — Canonical `SourceRecord`, typed `Entity`/`Relation`/`Chunk`, `Extractor`/`FuncExtractor`, `UpdateTelemetry`, `Query`/`SearchResult`, `EngineConfig`, and deep-copy helpers.
- `lightrag/text.go` — Code-aware tokenization, incremental $O(|\text{doc}|)$ Okapi BM25 inverted index, 128-dim deterministic L2-normalized vector index with noise floor, line-aligned chunker, and canonical alias normalization.
- `lightrag/extractor.go` — Clean-room Go AST + structured technical-text extractor (`DefaultExtractor`) and LightRAG dual-level query decomposer (`DecomposeQuery`).
- `lightrag/engine.go` — Concurrency-safe `Engine` with $O(|\Delta G|)$ incremental retraction/merge/re-indexing, atomic cache invalidation, `DropDerivedIndexes`/`RebuildAll`, SHA-256 `IndexDigest`, strict pre-return tenant/ACL scoping, and all `4` retrieval modes.
- `lightrag/benchmark.go` — Reproducible real-corpus loader and HIP-1347 benchmark harness.
- `lightrag/lightrag_test.go` — `13` unit, property, cache-invalidation, deep-copy immutability, ACL isolation, Go AST, concurrency `-race`, and end-to-end tests.
- `cmd/lightrag-bench/main.go` — Reproducible benchmark CLI writing `docs/research/hip-1347-lightrag/results.json`.
- `docs/research/hip-1347-lightrag/{README.md,results.json}` — 6-point HIP-1347 research dossier and raw benchmark measurements.

### Canonical Hanzo Graph Integration Plan

1. **Zero Disruption to Existing Packages**: `lightrag/` is a self-contained top-level package under `github.com/hanzoai/graph/lightrag` with zero changes to `posting/`, `schema/`, `tok/`, `dql/`, `algo/`, `go.mod`, or `go.sum`.
2. **Persistence & Storage Seam**: Canonical `SourceRecord` storage can be backed directly by `github.com/luxfi/zapdb` (already in `go.mod`) via prefixed namespace keys matching `x/keys.go`, while derived entity/relation posting lists can plug into `posting/` and `tok/hnsw/` (`persistentHNSW`) without altering the `Engine` API or relaxing the derived-index rebuild contract.
3. **LLM Extractor Seam**: Production deployments can inject an LLM-backed `Extractor` (e.g., via Hanzo Gateway) using the existing `EngineConfig.Extractor` interface while keeping the exact same canonical deduplication, $O(|\Delta G|)$ subgraph update, and pre-return ACL enforcement guarantees.
4. **Adaptive Query Router Seam**: Because `ordinary_hybrid` wins by `13.09x` latency on single-symbol exact lookups while `lightrag_hybrid` wins on multi-hop and thematic synthesis, a lightweight query classifier in Hanzo Search/Agents can route exact identifier lookups to `ModeOrdinaryHybrid` and relational/multi-hop queries to `ModeLocal` or `ModeHybrid`.

## 5. Correctness, Concurrency, Rebuildability, Freshness & Security Verification

- **Race & Static Analysis**: `go test -race -v ./lightrag/...` (**13/13 tests PASS**, 0 race conditions) and `go vet ./lightrag/... ./cmd/lightrag-bench/...` (0 warnings).
- **Bit-Exact Incremental vs. Full Rebuild Equivalence**: After `21` mixed incremental updates (`15` updates, `3` inserts, `3` deletes) on the real `114`-source corpus, `IndexDigest("hanzo-graph")` yields `37496dbb202ac63c`; after `DropDerivedIndexes()` (confirming `ErrIndexesNotBuilt` enforcement) and `RebuildAll()`, `IndexDigest("hanzo-graph")` yields the **exact same digest `37496dbb202ac63c`** (`bit_exact_equivalence = true`), verified further by a 40-step randomized property test (`TestIncrementalEqualsFullRebuildBitExact`).
- **Freshness & Atomic Cache Invalidation**: Primed query caches are invalidated atomically on every `UpsertSource` and `DeleteSource` (`cache_invalidated_on_edit = true`), yielding **100.0% updated v2 recall** and **0.0% deleted ghost recall**.
- **Deep-Copy Immutability**: Non-nil empty and populated `ACL`, `Metadata`, `Keywords`, `Aliases`, and `SourceIDs` slices/maps are deep-copied on ingest and on every read (`GetSource`, `ListSources`, `GetEntity`, `GetRelation`, `Search` uncached & cached).
- **Pre-Return Tenant & ACL Isolation**: **0/16 (0.0%)** leakage across cross-tenant and shared-entity intra-tenant ACL probes (including existence-oracle probes on secret-only tokens attached to shared entities), vs. **16/16 (100.0%)** leakage in the unfiltered control.

## 6. Raw Benchmark Measurements (`results.json` Exact Parity)

### 6.1 Corpus, Entity Deduplication & Extraction Cost

| Metric | Measured Value |
|---|---:|
| Real Apache-2.0 Go source files (`hanzoai/graph`) | `97` |
| Real repository documentation files | `5` |
| Operational engineering artifacts (commits, ADRs, incidents, agent events) | `12` |
| Total canonical sources / bytes / chunks | `114` / `895,824` B / `786` chunks |
| Raw entities extracted -> Canonical unique entities | `2,193` -> `1,116` |
| Duplicate entities collapsed / Duplicate entity rate | `1,077` / **49.11%** |
| Raw relationships extracted -> Canonical unique relationships | `2,563` -> `2,266` |
| Extraction tokens (input / output) | `225,191` / `159,006` |
| Equivalent LLM proxy extraction cost (`$0.15/1M` in, `$0.60/1M` out) | `$0.129182` (`$0.00` local AST/NLP) |
| Canonical relationships extracted per proxy dollar | **17,541.11 rels/$** |

### 6.2 Incremental Delta Update vs. Full Graph Rebuild

| Strategy | Ops | p50 (ms) | p95 (ms) | p99 (ms) | Mean (ms) | Mean Churn | Max Churn | Bit-Exact Digest |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Incremental Delta (`UpsertSource` / `DeleteSource`)** | `21` (`15` upd, `3` ins, `3` del) | **4.000** | **14.032** | **28.438** | **5.408** | **1.05%** (`13.24` ent, `22.43` rel) | `3.81%` | `37496dbb202ac63c` |
| **Full Graph Rebuild (`DropDerivedIndexes` + `RebuildAll`)** | `5` runs | `843.441` | `1209.001` | `1209.001` | `949.888` | `100.00%` | `100.00%` | `37496dbb202ac63c` |
| **Incremental Speedup** | — | **210.86x** | — | — | **175.64x** | **95.2x less churn** | — | **Exact Match (`true`)** |

### 6.3 Overall Retrieval Quality, Latency & Cost (`26` Queries, `TopK = 5`)

| System | Recall@5 | MRR@5 | Answer F1 | Answer EM | Task Success | Latency p50 / p95 / p99 (ms) | Cached p50 (ms) | Mean Ctx Tok | Cost / 1M Useful Tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ordinary_hybrid` (Baseline) | 71.8% | 0.777 | 91.2% | 73.1% | 88.5% | **0.780 / 2.329 / 4.132** | 0.003 | 1398.35 | $10.40 |
| `lightrag_local` | **83.3%** | 0.821 | 95.4% | 76.9% | 96.2% | 2.273 / 6.952 / 12.141 | 0.015 | 1485.88 | **$8.88** |
| `lightrag_global` | 73.7% | 0.689 | 88.1% | 57.7% | 84.6% | 4.931 / 15.488 / 19.646 | 0.016 | 1778.73 | $11.63 |
| `lightrag_hybrid` (Dual-Level) | 81.4% | **0.885** | **96.9%** | **84.6%** | **100.0%** | 7.603 / 19.005 / 25.356 | 0.016 | 1598.77 | $10.02 |

### 6.4 Per-Workload Breakdown

#### Cross-Package Multi-Hop Causality (`cross_package_multihop`, `10` queries)

| System | Recall@5 | MRR@5 | Answer F1 | Answer EM | Task Success | Latency p50 / p99 (ms) | Cost / 1M Useful Tok |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ordinary_hybrid` | 56.7% | 0.750 | 94.0% | 70.0% | 100.0% | 0.961 / 4.132 | $12.72 |
| `lightrag_local` | **81.7%** | 0.800 | 96.0% | 80.0% | 100.0% | 2.926 / 12.510 | **$8.28** |
| `lightrag_global` | 65.0% | 0.700 | 88.0% | 70.0% | 80.0% | 4.978 / 18.095 | $11.90 |
| `lightrag_hybrid` | 78.3% | **0.850** | **98.0%** | **90.0%** | **100.0%** | 8.101 / 25.356 | $9.53 |

#### Global Cross-Cutting Thematic Synthesis (`global_thematic`, `8` queries)

| System | Recall@5 | MRR@5 | Answer F1 | Answer EM | Task Success | Latency p50 / p99 (ms) | Cost / 1M Useful Tok |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ordinary_hybrid` | 62.5% | 0.588 | 78.8% | 50.0% | 62.5% | 0.717 / 3.673 | $7.18 |
| `lightrag_local` | **68.8%** | 0.667 | 90.0% | 50.0% | 87.5% | 2.166 / 7.178 | **$6.64** |
| `lightrag_global` | 58.3% | 0.594 | 76.2% | 0.0% | 75.0% | 4.995 / 18.704 | $9.13 |
| `lightrag_hybrid` | 66.7% | **0.812** | **92.5%** | **62.5%** | **100.0%** | 7.671 / 28.915 | $7.74 |

#### Single-Fact Exact Identifier Lookup — Explicit Negative Regime (`single_fact_exact`, `8` queries)

| System | Recall@5 | MRR@5 | Answer F1 | Answer EM | Task Success | Latency p50 / p99 (ms) | Cost / 1M Useful Tok |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ordinary_hybrid` (**Winner**) | **100.0%** | **1.000** | **100.0%** | **100.0%** | **100.0%** | **0.469 / 4.310** | **$13.83** |
| `lightrag_local` | 100.0% | 1.000 | 100.0% | 100.0% | 100.0% | 2.150 / 7.521 | $14.43 |
| `lightrag_global` | 100.0% | 0.771 | 100.0% | 100.0% | 100.0% | 4.180 / 23.347 | $15.84 |
| `lightrag_hybrid` (`13.09x` slower) | 100.0% | 1.000 | 100.0% | 100.0% | 100.0% | 6.140 / 21.477 | $15.18 |

Full raw machine-readable measurements are stored in [`results.json`](results.json).
