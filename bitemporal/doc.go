/*
 * SPDX-FileCopyrightText: © 2026 Hanzo AI, Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package bitemporal implements a clean-room temporal and bitemporal knowledge
// graph engine with source-evidence provenance, non-destructive supersession,
// out-of-order delayed-event splicing, explicit contradiction detection,
// pre-return tenant/ACL filtering, and full rebuildability from an immutable
// append-only evidence log.
//
// Architectural inspiration is drawn from Graphiti (Apache-2.0) and classical
// bitemporal relational algebra: every edge is indexed along two independent
// time axes:
//
//   - Valid Time [ValidFrom, ValidTo): when the fact was true in the domain.
//   - Transaction Time [TxFrom, TxTo): when the system learned or superseded
//     the fact version in its own belief state.
//
// Every derived edge carries provenance pointers back to the immutable
// EvidenceRecord entries in the append-only EvidenceLog. Deleting all derived
// graph indexes and replaying the EvidenceLog via RebuildFromLog reproduces
// bit-identical edges, conflicts, and query results.
//
// Access control is fail-closed: every query requires an AccessContext whose
// TenantID and AllowedScopes are enforced prior to returning or scoring any
// edge or contradiction record.
package bitemporal
