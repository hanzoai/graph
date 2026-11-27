# HIP-1347 Research Report: Associative Memory by Bounded Personalized PageRank (`hanzoai/graph#2`)

## 1. Reference Architecture & Independent License Audit (HippoRAG / HippoRAG 2)

- **References**: HippoRAG (*Neurobiologically Inspired Long-Term Memory for Large Language Models*, arXiv:2405.14831) and HippoRAG 2 (*From RAG to Memory: Non-Parametric Continual Learning for Large Language Models*, arXiv:2502.14802).
- **License Audit**:
  - Upstream HippoRAG / HippoRAG 2 reference codebase (`OSU-NLP-Group/HippoRAG`) is released under the **MIT License**.
  - This implementation in `hipporag/` is a **clean-room, independent Apache-2.0 Go implementation** built strictly with Go's standard library (`go.mod` and `go.sum` are completely untouched — **0 new transitive dependencies**).
  - External dataset support (`--external-dataset`) accepts user-supplied HotpotQA (CC BY-SA 4.0) or MuSiQue (CC BY 4.0) JSON/JSONL files at runtime without vendoring copyrighted third-party benchmark data into the repository.
- **Architectural Core**:
  - **Derived Index Architecture**: Canonical versioned records (`SourceRecord` with `SourceID`, `Version`, `Kind`, `TenantID`, `ACL`, `Reliability`) are stored independently of all derived indexes (BM25 inverted index, 128-dim dense vector table, tenant-partitioned entity-passage-commit-event-decision graph, dense entity synonymy edges, and recognition-memory fact store). `DropDerivedIndexes()` discards every index without touching any canonical record, and `RebuildIndex()` deterministically reconstructs bit-identical indexes.
  - **HippoRAG 2 Mechanics**: Combines triple-level recognition-memory filtering (`EnableRecognitionMemory`), dual phrase + passage personalized teleportation seeding (`PassageSeedWeight`), relevance × reliability^beta × hub-degree-damped transition weights, bounded ego-subgraph expansion (`MaxHops`, `MaxExpandedNodes`), L1-converged power iteration (`MaxIterations`, `Epsilon`, `Damping`), and deterministic widest-path evidence trail reconstruction.

## 2. Matched Baselines & Controlled Evaluation Protocol

All 4 systems run against the **exact same canonical corpus** (`364` records), **exact same query suite** (`88` queries across 4 workload categories), **exact same retrieval budget** (`TopK = 5`), **exact same grounded extractive answer synthesizer** (reading strictly the top-`5` returned hits), and **exact same judge** (normalized token F1, Exact Match, and end-to-end task completion):

1. `bm25_lexical_baseline` — Okapi BM25 over tenant+ACL-authorized records.
2. `dense_vector_rag_baseline` — Dense cosine similarity over the exact same 128-dim passage vectors.
3. `unfiltered_1hop_graph_baseline` — Entity-seeded 1-hop bipartite graph expansion without multi-hop PPR, reliability weighting, hub damping, or recognition-memory filtering.
4. `hipporag2_bounded_ppr` — Full HippoRAG 2 bounded PPR with recognition-memory fact filtering, relevance+reliability+hub-damped transitions, dual phrase+passage seeding, and deterministic path reconstruction.

### Honesty & Reproducibility Disclosures

- **Host**: Go go1.26.8 (linux/amd64, 2 CPUs)
- **Embeddings**: Deterministic zero-dependency 128-dim signed FNV-1a unigram+bigram+char-4gram L2-normalized dense embeddings
- **Answer Model & Judge**: Shared deterministic grounded extractive answer synthesizer operating strictly on top-K retrieved records (identical across all 4 systems)
- **Token Cost Formula**: Measured retrieved context tokens priced at $3.00 / 1M input tokens; Cost/1M useful tokens = $3.00 * (total_tokens / gold_evidence_tokens)
- **Boundary**: No external GPU, remote LLM API, or neural bi-encoder is invoked in default CI runs; dense embeddings use deterministic stdlib feature hashing.
- **Boundary**: Standard suite workloads are deterministic synthetic reconstructions of HotpotQA (2-hop), MuSiQue (3-4 hop), and cross-repo incident chains; real HotpotQA/MuSiQue drops are supported via --external-dataset.
- **Boundary**: Latency percentiles are live wall-clock measurements on the authoring host and vary by CPU frequency and corpus size.

Reproduce all numbers locally:

```bash
go run ./cmd/hipporag-bench --write-dossier --out-dir docs/research/hip-1347-hipporag2 --iters 15 --top-k 5
```

Run on an external HotpotQA JSON or MuSiQue JSONL file:

```bash
go run ./cmd/hipporag-bench --external-dataset /path/to/hotpot_dev_distractor_v1.json --top-k 5
```

## 3. Falsifiable Hypothesis & Negative Boundary Regimes

**Hypothesis**: On 2-to-4 hop associative queries and cross-repository engineering incidents where intermediate bridge entities share zero surface overlap with the query, bounded PPR with recognition-memory filtering and relevance/reliability edge weighting substantially outperforms BM25, dense vector RAG, and unfiltered 1-hop graph retrieval on multi-hop evidence recall, path reconstruction accuracy, irrelevant-node ratio, downstream task success, and `$/1M useful tokens` — while incurring measurable latency overhead on simple single-hop lookups and collapsing if hub damping/recognition memory are removed.

### Measured Negative & Boundary Regimes

1. **`neg_unweighted_hub_collapse`**: Removing reliability weighting, hub-degree damping, and recognition-memory triple filtering causes noisy hub telemetry to flood PPR: irrelevant-node ratio spikes from 1.0% to 13.0% while multi-hop evidence recall collapses from 99.0% to 87.0% and downstream task success drops from 95.0% to 70.0%.
2. **`neg_simple_single_hop_overhead`**: On simple single-hop factual lookups, plain BM25 (100.0% EM, 0.054 ms p50) and Dense RAG (100.0% EM, 0.170 ms p50) match HippoRAG 2 (100.0% EM, 3.279 ms p50) with +0.0% accuracy gain while HippoRAG 2 adds 60.84x latency overhead. Graph PPR should be reserved for multi-hop associative queries.
3. **`neg_acl_bridge_leakage_control`**: When tenant-alpha and tenant-beta share identical entity names (plus restricted ACL documents inside tenant-alpha), an unfiltered shared-entity graph control leaks confidential neighbors on 20/20 (100.0%) queries, whereas HippoRAG 2's per-tenant partitioning and pre-PPR ACL gate achieve 0/20 (0.0%) leakage with 0 foreign nodes inspected and 0 foreign edges traversed.

## 4. Bounded Deliverable Scope

- `hipporag/` — Clean-room Apache-2.0 Go package (`doc.go`, `types.go`, `embed.go`, `engine.go`, `ppr.go`, `workloads.go`, `benchmark.go`, `hipporag_test.go`).
- `cmd/hipporag-bench/main.go` — Deterministic CLI benchmark runner with `--external-dataset` loader and dossier writer.
- `docs/research/hip-1347-hipporag2/{README.md,score_table.md,raw_results.json}` — Live-generated 6-point dossier with exact numeric parity.
- Zero modifications to `go.mod`, `go.sum`, or existing `hanzoai/graph` packages.

## 5. Correctness, Concurrency, ACL Isolation & Rebuildability Verification

- **Derived-Index Drop & Bit-Exact Rebuild**: `DropDerivedIndexes()` preserves all `364` canonical records, enforces `ErrIndexesNotBuilt` (`true`), and `RebuildIndex()` restores an identical SHA-256 index digest (`840b53f4494578d4`, `bit_exact_equivalence = true`, `query_output_identical = true`) in `58.15 ms`.
- **Strict Pre-Return Tenant & ACL Isolation**: Per-tenant derived index partitioning and pre-PPR ACL gating produce **0/20 (0.0%)** leaks on shared-entity cross-tenant and restricted-ACL bridge attacks (`0` foreign nodes inspected, `0` foreign edges traversed), versus **20/20 (100.0%)** leakage in an unfiltered shared-entity control.
- **Canonical CUDA Graph Memory Alias Chain**: Query `Which change caused the CUDA graph memory alias?` resolves culprit `pr_4821_allocator_reuse` (`match = true`) across failure event -> kernel file -> allocator ADR -> PR commit -> regression test.
- **Concurrency & Race Safety**: Verified with `go test -race ./hipporag/...` under concurrent `SearchHippoRAG2`, `SearchBM25`, `SearchDenseRAG`, `UpsertSource`, and `RebuildIndex` workloads.

## 6. Raw Measurements & HIP-1347 Summary

| System | Multi-Hop Recall | Path Recon Acc | Irrelevant Ratio | Answer F1 / EM | Multi-Hop Task Success | Latency p50 / p95 (ms) | Cost/1M Useful Tok |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bm25_lexical_baseline` | 74.2% | 0.0% | 59.9% | 70.7% / 69.3% | 60.3% | 0.063 / 0.240 | $8.42 |
| `dense_vector_rag_baseline` | 51.7% | 0.0% | 71.6% | 44.6% / 43.2% | 25.0% | 0.175 / 0.408 | $12.23 |
| `unfiltered_1hop_graph_baseline` | 49.5% | 0.0% | 42.9% | 36.9% / 35.2% | 16.2% | 0.268 / 0.545 | $7.06 |
| `hipporag2_bounded_ppr` | 98.2% | 95.5% | 44.1% | 98.9% / 98.9% | 94.1% | 5.274 / 8.324 | $5.25 |

Full per-workload tables and negative regimes are in [`score_table.md`](score_table.md), and raw machine-readable measurements are in [`raw_results.json`](raw_results.json).
