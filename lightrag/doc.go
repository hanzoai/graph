/*
 * SPDX-FileCopyrightText: © 2026 Hanzo AI, Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package lightrag implements a clean-room, derived incremental entity-relationship
// indexing and dual-level retrieval engine inspired by LightRAG (Guo et al.,
// EMNLP 2025, arXiv:2410.05779, MIT).
//
// Architectural invariants (HIP-1347):
//
//  1. Canonical Source Truth vs. Derived Indexes: Every ingested document,
//     repository blob, Git commit, agent output, decision, or operational event
//     has a stable (TenantID, SourceID, Version, ACL) identity stored in the
//     canonical source store. All chunks, entity-relationship graphs, alias
//     tables, BM25 inverted indexes, dense vector tables, and query caches are
//     strictly derived projections that can be dropped and rebuilt bit-for-bit
//     without mutating or losing canonical source records.
//  2. Incremental Subgraph Delta Updates: Updating or deleting a source retracts
//     only the chunks, entity contributions, and relationship contributions
//     attributed to that SourceID, removes orphaned graph elements, re-merges
//     shared surviving elements deterministically, updates only touched lexical
//     and vector postings, and atomically invalidates query caches.
//  3. Strict Pre-Return Tenant & ACL Isolation: Every tenant owns an isolated
//     derived graph partition, and per-source ACL authorization is enforced before
//     scoring, neighborhood expansion, description assembly, or result return.
//  4. Dual-Level Retrieval: Supports Local (entity + bounded 1-2 hop neighborhood),
//     Global (relationship/theme index), Hybrid (Local + Global + Chunk RRF),
//     and OrdinaryHybrid (chunk-only BM25 + dense vector RRF baseline).
package lightrag
