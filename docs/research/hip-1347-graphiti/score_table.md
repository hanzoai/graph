# HIP-1347 Comparative Benchmark Score Table (`hanzoai/graph#4`)

- **Suite**: HIP-1347 Bitemporal Knowledge Graph & Provenance Benchmark (hanzoai/graph#4)
- **Corpus**: 220 immutable `EvidenceRecord` events | **Workload**: 120 deterministic queries (20 per category × 6 categories) | **Timed Iterations**: 25/query
- **Derived-Index Wipe & Rebuild Equivalence (`RebuildFromLog`)**: `true`

## 1. Matched 4-Backend Comparative Score Table

| Backend | Overall Acc (120q) | Temporal QA Acc (100q) | Superseded Current | Historical As-Of | Audit As-Known-At | Delayed Events | Contradictions | Agent Task Success | Stale Answer Freq | Mean / p50 / p95 Lat (µs) | Tokens/Ans | Bytes/Edge (Total KB) | Cross-Tenant Leaks | Bit-Exact Rebuild |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `overwrite_mutable_graph` | 29.2% | 30.0% | 100.0% | 0.0% | 0.0% | 50.0% | 0.0% | 25.0% | 63.2% | 0.60 / 0.59 / 0.92 | 18.0 | 110.2 B (8.6 KB) | 0 | `false` |
| `append_only_no_supersession` | 25.0% | 25.0% | 0.0% | 0.0% | 100.0% | 25.0% | 0.0% | 25.0% | 68.4% | 1.77 / 0.91 / 3.83 | 39.0 | 340.7 B (73.2 KB) | 0 | `true` |
| `valid_time_only_unitemporal` | 58.3% | 60.0% | 100.0% | 100.0% | 0.0% | 100.0% | 0.0% | 50.0% | 26.3% | 0.44 / 0.42 / 0.52 | 18.0 | 308.5 B (61.8 KB) | 0 | `false` |
| `bitemporal_graphiti_provenance` | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 1.14 / 0.92 / 1.85 | 25.2 | 378.1 B (127.4 KB) | 0 | `true` |

## 2. Measured Negative & Boundary Regimes

### 2.1 `static_immutable_knowledge_overhead`

- **Setup**: 500 static non-evolving facts, 100 point queries (zero supersessions, zero delayed events, zero contradictions).
- **Accuracy**: Plain mutable graph = **100.0%**, Bitemporal Graphiti = **100.0%** (**Accuracy Gain = 0.0%**).
- **Memory Cost**: Plain = **127.0 B/edge**, Bitemporal = **364.0 B/edge** (**+237.0 B/edge**, **2.87x** footprint overhead).
- **Latency**: Plain = **0.56 µs/query**, Bitemporal = **1.13 µs/query**.
- **Finding**: On static non-evolving facts with zero supersessions or delayed events, bitemporal tracking delivers 0.0% accuracy gain while carrying extra timestamp and provenance bytes per edge.

### 2.2 `unresolvable_simultaneous_contradiction`

- **Setup**: 20 pairs of simultaneous equal-precedence sources asserting conflicting functional values over identical `[ValidFrom, ValidTo)`.
- **Measured**: Auto-resolved without trust policy = **0/20 (0.0%)**, Explicit `ContradictionConflict` surfaced = **20/20 (100.0%)**, Both evidence pointers preserved = **20/20**.
- **Finding**: Two equal-precedence sources asserting conflicting functional values for the exact same [ValidFrom, ValidTo) cannot be auto-resolved without a domain trust policy; the engine honestly surfaces ContradictionConflict with both evidence pointers.

### 2.3 `cross_tenant_acl_denial`

- **Setup**: 20 unauthorized queries (10 cross-tenant `tenant-alpha -> tenant-beta`, 10 restricted-scope `repo:public -> repo:restricted`) against valid existing facts.
- **Measured**: Returned hits = **0**, Leaked edges = **0**, Leaked conflicts = **0**, Foreign edges inspected = **0**.
- **Finding**: Pre-return tenant and ACL filtering denies 100% of cross-tenant and restricted-scope reads (0 hits returned, 0 foreign edges inspected), never relaxing isolation for speed or recall.
