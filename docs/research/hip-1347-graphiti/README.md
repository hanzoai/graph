# HIP-1347 Research Report: Temporal & Bitemporal Knowledge Graph with Provenance (`hanzoai/graph#4`)

## 1. Reference Architecture & License Audit (Graphiti)

| Artifact | License | Integration Strategy | Transitive Dependencies Added |
|---|---|---|---|
| Graphiti temporal/episodic knowledge graph model (`getzep/graphiti`) | **Apache-2.0** | Clean-room native Go implementation in `bitemporal/` with architectural attribution | **0** (100% Go standard library; `go.mod` and `go.sum` untouched) |
| Hanzo Graph (`github.com/hanzoai/graph`) | **Apache-2.0** | Native Go package `github.com/hanzoai/graph/bitemporal` + CLI `cmd/bitemporal-bench` | **0** |

Every fact is grounded in an immutable append-only source `EvidenceRecord` (`EvidenceID`, `SourceVersion`, `TenantID`, `ACLScope`, `ObservedAtTxTime`, `PayloadHash`, `FactTriple`). All bitemporal edges (`[ValidFrom, ValidTo)` × `[TxFrom, TxTo)`) and `ContradictionConflict` records are strictly **derived** indexes that can be dropped and rebuilt bit-identically from the immutable log.

## 2. Matched Baselines & Controlled Evaluation Protocol

All 4 backends were evaluated over the **exact same 220-event corpus**, **exact same 120-query deterministic workload** (20 queries × 6 categories), **exact same pre-return tenant/ACL rules**, and **exact same deterministic downstream agent action policy** (`SelectAgentAction`):

1. **`overwrite_mutable_graph`**: Mutable property graph updating `(Tenant, Scope, Subject, Predicate)` in place; smallest static memory footprint, but destroys historical valid-time state, corrupts current state on delayed retro-events, and silently drops contradictions.
2. **`append_only_no_supersession`**: Appends every edge with open-ended valid time without closing prior versions; preserves history on disk but leaks superseded values on current and point-in-time queries, inflating stale-answer frequency and downstream LLM token cost.
3. **`valid_time_only_unitemporal`**: Tracks valid-time intervals `[ValidFrom, ValidTo)` with supersession and delayed-event splicing, but mutates `ValidTo` in place without transaction time (`TxFrom`, `TxTo`), leaking future knowledge into historical `as-known-at` audits.
4. **`bitemporal_graphiti_provenance`**: Full bitemporal engine (`bitemporal.Graph`) tracking valid time, transaction time, source evidence pointers, non-destructive supersession, delayed-event splicing, explicit contradiction detection, pre-return tenant/ACL filtering, and bit-identical log rebuildability.

## 3. Falsifiable Hypothesis & Negative Boundary Regimes

**Hypothesis**: Separating an immutable append-only source evidence log from a derived bitemporal index (`valid_time` × `tx_time` + explicit contradiction tracking) will eliminate stale-answer leakage (`0.0%`), achieve `100.0%` accuracy across superseded current-state, historical `as-of`, audit `as-known-at`, delayed out-of-order convergence, contradiction detection, and downstream agent action selection, while maintaining sub-millisecond query latency and `0` cross-tenant leaks.

**Falsifiable Negative & Boundary Regimes (Measured)**:

1. **`static_immutable_knowledge_overhead`**: On 500 static facts with zero supersessions or delayed events, `bitemporal_graphiti_provenance` yields **0.0% accuracy gain** over `overwrite_mutable_graph` (both 100.0%) while incurring **+237.0 B/edge** (**2.87x** edge memory footprint: 364.0 B vs 127.0 B/edge).
2. **`unresolvable_simultaneous_contradiction`**: Across 20 simultaneous equal-precedence conflicting assertion pairs, the engine refuses to fabricate a winner (**0/20 auto-resolved**) and surfaces **20/20 (100.0%)** explicit `ContradictionConflict` records preserving both competing evidence pointers.
3. **`cross_tenant_acl_denial`**: Across 20 unauthorized cross-tenant and restricted-scope queries against valid existing facts, the engine returns **0 hits**, leaks **0 edges / 0 conflicts**, and inspects **0 foreign edges** (fail-closed pre-return isolation).

## 4. Bounded Deliverable Scope

- `bitemporal/doc.go`, `bitemporal/types.go`, `bitemporal/graph.go`: Clean-room, zero-dependency Go bitemporal knowledge graph engine (`Graph`, `EvidenceRecord`, `BitemporalEdge`, `ContradictionConflict`, `AccessContext`, `RebuildFromLog`).
- `bitemporal/benchmark.go`: Matched 4-backend benchmark harness, 120-query deterministic temporal QA & downstream agent decision suite, and 3 negative boundary regimes.
- `bitemporal/bitemporal_test.go`: Unit, property, rebuild-equivalence, multi-tenant ACL isolation, and 16-goroutine `-race` concurrency verification suite.
- `cmd/bitemporal-bench/main.go`: Reproducible stdlib CLI generating `README.md`, `score_table.md`, and `raw_results.json`.

## 5. Correctness, Concurrency, ACL Isolation & Rebuildability Verification

- **Coexistence & Zero Deletion**: `PR #47 open at 10 AM` and `PR #47 merged at 11 AM` coexist cleanly — `QueryAsOf(10:30)` returns `"open"`, `QueryAsOf(11:30)` and `QueryCurrent()` return `"merged"`, and `QueryBitemporal(11:30, tx=10:15)` returns `"open"`.
- **Out-of-Order Delayed Splicing**: Late-arriving retro-events (`09:00 draft` or mid-history `10:30 review_requested` arriving at `tx=12:00`) splice cleanly into valid time without corrupting subsequent states while remaining invisible to `tx < 12:00` audit snapshots.
- **Derived-Index Rebuildability**: Wiping the entire derived index and replaying the immutable `EvidenceLog` via `RebuildFromLog()` produces bit-identical `BitemporalEdge` slices, `ContradictionConflict` slices, and all 120 query results (`rebuild_bit_identical_verified = true`).
- **Concurrency & Multi-Tenant Isolation**: Verified race-free under `go test -race -v -count=1 ./bitemporal/...` with 16 concurrent goroutines reading, ingesting, and triggering rebuilds across 4 isolated tenants with `0` cross-tenant leaks.

## 6. Raw Measurements & HIP-1347 Status

- **`bitemporal_graphiti_provenance`**: **100.0%** overall accuracy (100.0% temporal QA, 100.0% downstream agent success), **0.0%** stale-answer frequency, **0.92 µs** p50 / **1.85 µs** p95 latency, **25.2 tokens/answer**, **378.1 B/edge**, **0** cross-tenant leaks, bit-exact rebuild = `true`.
- **`valid_time_only_unitemporal`**: **58.3%** overall accuracy — passes current (100.0%), historical as-of (100.0%), and delayed events (100.0%), but **fails** bitemporal audit `as-known-at` (**0.0%**) and contradiction handling (**0.0%**), lowering downstream agent task success to **50.0%**.
- **`append_only_no_supersession`**: **25.0%** overall accuracy with **68.4% stale-answer frequency** and **39.0 tokens/answer** (1.54x bitemporal context cost), driving downstream agent task success down to **25.0%**.
- **`overwrite_mutable_graph`**: **29.2%** overall accuracy — loses historical as-of (**0.0%**), audit as-known-at (**0.0%**), delayed events (**50.0%**), and contradictions (**0.0%**), achieving **25.0%** downstream agent task success.

Full score table: [`score_table.md`](score_table.md) | Machine-readable raw measurements: [`raw_results.json`](raw_results.json).

**HIP-1347 Status**: `READY_FOR_REVIEW` — Clean-room Apache-2.0 Go implementation, zero new dependencies, race-tested, rebuild-verified, with matched comparative benchmark and honest negative regimes.
