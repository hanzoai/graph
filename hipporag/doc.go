/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Package hipporag implements a clean-room associative memory and bounded
// Personalized PageRank (PPR) retrieval engine inspired by HippoRAG
// (arXiv:2405.14831) and HippoRAG 2 (arXiv:2502.14802, MIT).
//
// Core architectural guarantees:
//
//  1. Derived Index Architecture: Canonical source records (documents,
//     repository blobs, Git commits, agent events, stored decisions) carry
//     stable SourceID, Version, TenantID, and ACL metadata. Lexical (BM25),
//     dense vector, entity-passage graph, and recognition-memory fact indexes
//     are strictly derived representations that can be dropped and rebuilt
//     deterministically from canonical records via RebuildIndex without
//     destroying or mutating any canonical record.
//
//  2. Strict Pre-Return Tenant & ACL Isolation: Source identity, tenant
//     ownership, and ACL scopes are enforced prior to recognition-memory
//     filtering, prior to PPR seed expansion, during every graph edge
//     traversal, and prior to returning any hit. Graph entity nodes are
//     strictly partitioned per tenant so shared entity names can never bridge
//     tenants.
//
//  3. Bounded Personalized PageRank with Recognition Memory: Combines
//     triple-level recognition-memory filtering, relevance and reliability edge
//     weighting, hub-degree damping, dual phrase+passage seeding, explicit hop
//     and node-expansion budgets, iteration caps, and deterministic evidence
//     path reconstruction.
package hipporag
