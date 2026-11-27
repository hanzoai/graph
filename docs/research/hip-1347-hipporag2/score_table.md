# HIP-1347 Comparative Benchmark Score Table (`hanzoai/graph#2`)

- **Suite**: `hip-1347-hipporag2-standard-suite` | **Canonical Records**: `364` | **Queries**: `88` | **Matched Retrieval Budget (`TopK`)**: `5` | **Timed Iterations**: `15`/query
- **Derived-Index Wipe & Bit-Exact Rebuild (`DropDerivedIndexes -> RebuildIndex`)**: `true` (SHA-256 `840b53f4494578d4`, rebuild `58.15 ms`)
- **Environment**: Go go1.26.8 (linux/amd64, 2 CPUs)

## 1. Matched 4-System Comparative Score Table (All Workloads)

| System | Multi-Hop Recall | Overall Recall | Full Chain Recall | Path Recon Acc | Irrelevant Node Ratio | Answer F1 | Answer EM | Task Success (All / Multi-Hop) | Lat p50 / p95 / p99 (ms) | Tokens/Q | Cost/1M Useful Tok | Cross-Tenant Leaks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bm25_lexical_baseline` | 74.2% | 80.1% | 59.1% | 0.0% | 59.9% | 70.7% | 69.3% | 69.3% / 60.3% | 0.063 / 0.240 / 0.283 | 148.5 | $8.42 | 0 |
| `dense_vector_rag_baseline` | 51.7% | 62.7% | 36.4% | 0.0% | 71.6% | 44.6% | 43.2% | 42.0% / 25.0% | 0.175 / 0.408 / 0.471 | 152.8 | $12.23 | 0 |
| `unfiltered_1hop_graph_baseline` | 49.5% | 61.0% | 34.1% | 0.0% | 42.9% | 36.9% | 35.2% | 35.2% / 16.2% | 0.268 / 0.545 / 0.559 | 87.6 | $7.06 | 0 |
| `hipporag2_bounded_ppr` | 98.2% | 98.6% | 95.5% | 95.5% | 44.1% | 98.9% | 98.9% | 95.5% / 94.1% | 5.274 / 8.324 / 10.934 | 131.1 | $5.25 | 0 |

## 2. Per-Workload Breakdown

### Workload: `hotpotqa_2hop` (24 queries)

| System | Evidence Recall | Full Chain Recall | Path Recon Acc | Irrelevant Ratio | Answer F1 | Answer EM | Task Success | p50 / p95 (ms) | Cost/1M Useful Tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bm25_lexical_baseline` | 93.8% | 87.5% | 0.0% | 59.4% | 87.5% | 87.5% | 87.5% | 0.052 / 0.220 | $8.83 |
| `dense_vector_rag_baseline` | 70.8% | 41.7% | 0.0% | 71.7% | 41.7% | 41.7% | 41.7% | 0.175 / 0.407 | $12.92 |
| `unfiltered_1hop_graph_baseline` | 60.4% | 20.8% | 0.0% | 46.5% | 20.8% | 20.8% | 20.8% | 0.268 / 0.553 | $6.75 |
| `hipporag2_bounded_ppr` | 100.0% | 100.0% | 100.0% | 60.0% | 100.0% | 100.0% | 100.0% | 5.124 / 5.738 | $8.23 |

### Workload: `musique_3_4hop` (24 queries)

| System | Evidence Recall | Full Chain Recall | Path Recon Acc | Irrelevant Ratio | Answer F1 | Answer EM | Task Success | p50 / p95 (ms) | Cost/1M Useful Tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bm25_lexical_baseline` | 84.0% | 45.8% | 0.0% | 42.5% | 83.3% | 83.3% | 83.3% | 0.077 / 0.257 | $5.74 |
| `dense_vector_rag_baseline` | 52.4% | 8.3% | 0.0% | 64.2% | 29.2% | 29.2% | 29.2% | 0.176 / 0.356 | $9.30 |
| `unfiltered_1hop_graph_baseline` | 56.6% | 20.8% | 0.0% | 53.9% | 27.6% | 25.0% | 25.0% | 0.284 / 0.550 | $6.93 |
| `hipporag2_bounded_ppr` | 100.0% | 100.0% | 100.0% | 30.0% | 100.0% | 100.0% | 100.0% | 5.581 / 8.324 | $4.29 |

### Workload: `cross_repo_incidents` (20 queries)

| System | Evidence Recall | Full Chain Recall | Path Recon Acc | Irrelevant Ratio | Answer F1 | Answer EM | Task Success | p50 / p95 (ms) | Cost/1M Useful Tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bm25_lexical_baseline` | 39.0% | 0.0% | 0.0% | 61.0% | 6.1% | 0.0% | 0.0% | 0.066 / 0.178 | $8.41 |
| `dense_vector_rag_baseline` | 28.0% | 0.0% | 0.0% | 72.0% | 11.3% | 5.0% | 0.0% | 0.174 / 0.408 | $11.80 |
| `unfiltered_1hop_graph_baseline` | 28.0% | 0.0% | 0.0% | 63.2% | 4.3% | 0.0% | 0.0% | 0.243 / 0.487 | $8.77 |
| `hipporag2_bounded_ppr` | 94.0% | 80.0% | 80.0% | 6.0% | 95.0% | 95.0% | 80.0% | 5.500 / 10.741 | $3.14 |

### Workload: `simple_single_hop` (20 queries)

| System | Evidence Recall | Full Chain Recall | Path Recon Acc | Irrelevant Ratio | Answer F1 | Answer EM | Task Success | p50 / p95 (ms) | Cost/1M Useful Tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bm25_lexical_baseline` | 100.0% | 100.0% | 0.0% | 80.0% | 100.0% | 100.0% | 100.0% | 0.054 / 0.146 | $22.81 |
| `dense_vector_rag_baseline` | 100.0% | 100.0% | 0.0% | 80.0% | 100.0% | 100.0% | 100.0% | 0.170 / 0.398 | $22.34 |
| `unfiltered_1hop_graph_baseline` | 100.0% | 100.0% | 0.0% | 5.0% | 100.0% | 100.0% | 100.0% | 0.256 / 0.493 | $3.30 |
| `hipporag2_bounded_ppr` | 100.0% | 100.0% | 100.0% | 80.0% | 100.0% | 100.0% | 100.0% | 3.279 / 7.326 | $18.83 |

## 3. Measured Negative & Boundary Regimes

### 3.1 `neg_unweighted_hub_collapse`

- **Setup**: 200 canonical records (20 cross-repo incident queries + 60 low-reliability hub-linked noisy telemetry logs).
- **Unweighted / No-Recognition PPR**: Recall = **87.0%**, Irrelevant Node Ratio = **13.0%**, Answer EM = **90.0%**, Task Success = **70.0%**, Cost/1M Useful Tok = **$3.60**.
- **HippoRAG 2 Bounded PPR (Full)**: Recall = **99.0%**, Irrelevant Node Ratio = **1.0%**, Answer EM = **100.0%**, Task Success = **95.0%**, Cost/1M Useful Tok = **$3.03**, Mean Facts Rejected by Recognition Memory = **67.7**/query.
- **Finding**: Removing reliability weighting, hub-degree damping, and recognition-memory triple filtering causes noisy hub telemetry to flood PPR: irrelevant-node ratio spikes from 1.0% to 13.0% while multi-hop evidence recall collapses from 99.0% to 87.0% and downstream task success drops from 95.0% to 70.0%.

### 3.2 `neg_simple_single_hop_overhead`

- **Setup**: 20 single-hop factual configuration queries where the query directly names the target record.
- **BM25 Lexical Baseline**: Recall = **100.0%**, Answer EM = **100.0%**, p50 Latency = **0.054 ms**.
- **Dense Vector RAG Baseline**: Recall = **100.0%**, Answer EM = **100.0%**, p50 Latency = **0.170 ms**.
- **HippoRAG 2 Bounded PPR**: Recall = **100.0%**, Answer EM = **100.0%**, p50 Latency = **3.279 ms** (**60.84x** vs BM25, Accuracy Gain = **+0.0%**).
- **Finding**: On simple single-hop factual lookups, plain BM25 (100.0% EM, 0.054 ms p50) and Dense RAG (100.0% EM, 0.170 ms p50) match HippoRAG 2 (100.0% EM, 3.279 ms p50) with +0.0% accuracy gain while HippoRAG 2 adds 60.84x latency overhead. Graph PPR should be reserved for multi-hop associative queries.

### 3.3 `neg_acl_bridge_leakage_control`

- **Setup**: 20 adversarial queries across 10 entity chains where `tenant-alpha` (public), `tenant-alpha` (`sec:incident-response`), and `tenant-beta` (confidential) share identical entity names.
- **Unfiltered Post-Hoc Bridge Control**: Leaked queries = **20/20 (100.0%)**.
- **HippoRAG 2 Bounded PPR (Pre-Return Isolated)**: Leaked queries = **0/20 (0.0%)**, Foreign Tenant Nodes Inspected = **0**, Foreign Tenant Edges Traversed = **0**, Unauthorized ACL Hits = **0**.
- **Finding**: When tenant-alpha and tenant-beta share identical entity names (plus restricted ACL documents inside tenant-alpha), an unfiltered shared-entity graph control leaks confidential neighbors on 20/20 (100.0%) queries, whereas HippoRAG 2's per-tenant partitioning and pre-PPR ACL gate achieve 0/20 (0.0%) leakage with 0 foreign nodes inspected and 0 foreign edges traversed.

## 4. Canonical CUDA Graph Memory Alias Incident Path Reconstruction

- **Query**: `Which change caused the CUDA graph memory alias?`
- **Extracted Answer**: `pr_4821_allocator_reuse` (Gold: `pr_4821_allocator_reuse`, Match: `true`)
- **Seed Entities**: `cuda_graph_memory_alias`, `pr_4821_allocator_reuse`, `stream_capture_kernel`, `test_cuda_graph_alias_regression`
- **Reconstructed Evidence Paths**:
  - `inc01_event (hops=1, score=0.0741): cuda_graph_memory_alias --[evidenced_in]--> Incident event: cuda graph memory alias`
  - `inc01_regtest (hops=4, score=0.0642): cuda_graph_memory_alias --[triggered_in]--> stream_capture_kernel --[assumes_invariant]--> caching_allocator_pool --[broken_by]--> pr_4821_allocator_reuse --[evidenced_in]--> Regression test: test_cuda_graph_alias_regression`
  - `inc01_commit (hops=4, score=0.0620): cuda_graph_memory_alias --[triggered_in]--> stream_capture_kernel --[assumes_invariant]--> caching_allocator_pool --[broken_by]--> pr_4821_allocator_reuse --[evidenced_in]--> PR #4821: Aggressive cross-stream segment reuse in allocator pool`
  - `inc01_kernel (hops=2, score=0.0459): cuda_graph_memory_alias --[triggered_in]--> stream_capture_kernel --[evidenced_in]--> Kernel file: stream_capture_kernel`
  - `inc01_decision (hops=3, score=0.0435): cuda_graph_memory_alias --[triggered_in]--> stream_capture_kernel --[assumes_invariant]--> caching_allocator_pool --[evidenced_in]--> ADR invariant: caching_allocator_pool`
