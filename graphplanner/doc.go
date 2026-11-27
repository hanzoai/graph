/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Package graphplanner implements an embedded compact CSR native graph store
// and a cost-based graph/vector query planner for Hanzo Graph (HIP-1347,
// hanzoai/graph#5).
//
// Architectural invariants:
//   - Canonical versioned records (SourceID, Version, TenantID, ACL) are the
//     source of truth; CSR adjacency tables, lexical/vector indexes, and the
//     bounded LRU neighborhood cache are strictly derived representations that
//     can be dropped and rebuilt deterministically without mutating canonical
//     records.
//   - Tenant partitioning and ACL checks are enforced before any candidate is
//     expanded, scored, cached, or returned.
//   - The cost-based planner dynamically selects among five physical execution
//     plans—plain_rag_bypass, cached_neighborhood, pure_traversal, graph_first,
//     and vector_first—based on estimated candidate cardinality, predicate/ACL
//     selectivity, local graph degree, cache hit probability, and latency
//     budget.
//   - Pure Go standard library implementation with zero external dependencies.
package graphplanner
