// Package graphrag implements offline community detection, hierarchical
// summarization, incremental summary invalidation, and explicit global-vs-local
// query routing for corpus-wide synthesis questions in Hanzo Graph (hanzoai/graph#3,
// HIP-1347).
//
// # Design & Clean-Room Attribution
//
// This package is an independent, clean-room Apache-2.0 Go standard-library
// implementation inspired by:
//   - Edge et al., "From Local to Global: A Graph RAG Approach to Query-Focused
//     Summarization" (arXiv:2404.16130, 2024).
//
// No code is copied from the upstream MIT reference implementation (which is
// explicitly noted as research software in maintenance mode with expensive
// indexing). Zero external dependencies are added to go.mod.
//
// # Shared Retrieval & Security Invariants
//
//  1. Canonical Truth vs. Derived Indexes: Source documents carry stable
//     (SourceID, Version, TenantID, ACL) identities. All extracted entities,
//     relationships, multi-level communities (L0 -> L1 -> L2), community
//     summaries, and lexical/vector retrieval structures are strictly DERIVED
//     projections that can be dropped and rebuilt bit-for-bit without losing
//     canonical records.
//  2. Strict Tenant & ACL Partitioning: Within each TenantID, derived entity
//     graphs, community trees, and community summaries are partitioned by
//     normalized ACL scope. Unauthorized callers never inspect foreign tenant
//     or unauthorized ACL partitions, guaranteeing 0% cross-tenant and
//     cross-ACL leakage into global summaries and zero existence-oracle leaks.
//  3. Incremental Summary Invalidation: Document updates, ACL changes, and
//     deletions invalidate and re-summarize only the affected leaf (L0)
//     communities and their strict ancestor chain (L1 -> L2), rather than
//     rebuilding the entire community hierarchy.
//  4. Explicit Global vs. Local Routing Gate: GraphRAG map-reduce over
//     hierarchical community summaries is designed for corpus-wide thematic
//     questions ("what are the largest sources of engineering regressions
//     across our repositories?") and is NOT the default path for exact
//     single-entity or file lookups. QueryRouter routes exact lookups to the
//     fast ordinary hybrid (BM25 + dense vector) baseline and routes
//     corpus-wide synthesis queries to GraphRAG map-reduce.
package graphrag
