/*
 * SPDX-FileCopyrightText: © 2026 Hanzo AI, Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

package bitemporal

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"
)

// OpenEnd denotes an open-ended (unbounded future / +infinity) exclusive upper
// bound for either ValidTo or TxTo.
const OpenEnd int64 = 0

// Sentinel errors returned by ingestion and validation paths.
var (
	ErrEmptyEvidenceID    = errors.New("bitemporal: evidence_id is required")
	ErrEmptySourceVersion = errors.New("bitemporal: source_version is required")
	ErrEmptyTenantID      = errors.New("bitemporal: tenant_id is required")
	ErrEmptyACLScope      = errors.New("bitemporal: acl_scope is required")
	ErrInvalidFactTriple  = errors.New("bitemporal: subject and predicate are required")
	ErrEmptyFactObject    = errors.New("bitemporal: object is required for assertion")
	ErrInvalidValidRange  = errors.New("bitemporal: valid_from must be > 0 and valid_to must be 0 or > valid_from")
	ErrNonMonotonicTxTime = errors.New("bitemporal: observed_at_tx_time must be monotonically non-decreasing")
	ErrInvalidCardinality = errors.New("bitemporal: invalid cardinality")
)

// Cardinality controls whether a predicate permits a single functional value at
// any given valid-time instant or multiple concurrent values.
type Cardinality string

const (
	// FunctionalSingleValue predicates (for example pr.state or issue.owner)
	// hold at most one true value at any instant in valid time unless competing
	// sources produce an explicit ContradictionConflict.
	FunctionalSingleValue Cardinality = "functional_single_value"

	// MultiValue predicates (for example has_label or reviewed_by) permit
	// multiple distinct object values to hold simultaneously at the same instant.
	MultiValue Cardinality = "multi_value"
)

// Ts converts a time.Time to UTC Unix nanoseconds suitable for valid-time and
// transaction-time bounds.
func Ts(t time.Time) int64 {
	return t.UTC().UnixNano()
}

// FormatTs renders a Unix nanosecond timestamp as RFC3339Nano, or "INF" for OpenEnd.
func FormatTs(ns int64) string {
	if isOpenEnd(ns) {
		return "INF"
	}
	return time.Unix(0, ns).UTC().Format(time.RFC3339Nano)
}

func isOpenEnd(ns int64) bool {
	return ns == OpenEnd || ns == math.MaxInt64
}

func effectiveEnd(ns int64) int64 {
	if isOpenEnd(ns) {
		return math.MaxInt64
	}
	return ns
}

func containsTime(start, end, instant int64) bool {
	if instant < start {
		return false
	}
	if isOpenEnd(end) {
		return true
	}
	return instant < end
}

func intervalsOverlap(aStart, aEnd, bStart, bEnd int64) (int64, int64, bool) {
	start := aStart
	if bStart > start {
		start = bStart
	}
	ae := effectiveEnd(aEnd)
	be := effectiveEnd(bEnd)
	end := ae
	if be < end {
		end = be
	}
	if start >= end {
		return 0, 0, false
	}
	if end == math.MaxInt64 {
		return start, OpenEnd, true
	}
	return start, end, true
}

// FactTriple describes the domain assertion carried by an EvidenceRecord.
type FactTriple struct {
	Subject     string      `json:"subject"`
	Predicate   string      `json:"predicate"`
	Object      string      `json:"object"`
	Cardinality Cardinality `json:"cardinality"`
	ValidFrom   int64       `json:"valid_from"`
	ValidTo     int64       `json:"valid_to"`
}

// EvidenceRef is a provenance pointer linking a derived BitemporalEdge or
// ContradictionConflict back to an immutable EvidenceRecord.
type EvidenceRef struct {
	EvidenceID    string `json:"evidence_id"`
	SourceVersion string `json:"source_version"`
	PayloadHash   string `json:"payload_hash"`
}

// EvidenceRecord is an immutable entry in the append-only source evidence log.
// Source records are never mutated or deleted; all graph indexes are derived
// projections over this log.
type EvidenceRecord struct {
	Seq                uint64     `json:"seq"`
	EvidenceID         string     `json:"evidence_id"`
	SourceVersion      string     `json:"source_version"`
	TenantID           string     `json:"tenant_id"`
	ACLScope           string     `json:"acl_scope"`
	ObservedAtTxTime   int64      `json:"observed_at_tx_time"`
	PayloadHash        string     `json:"payload_hash"`
	Fact               FactTriple `json:"fact"`
	RetractsEvidenceID string     `json:"retracts_evidence_id,omitempty"`
}

// Ref returns the compact provenance pointer for this EvidenceRecord.
func (r EvidenceRecord) Ref() EvidenceRef {
	return EvidenceRef{
		EvidenceID:    r.EvidenceID,
		SourceVersion: r.SourceVersion,
		PayloadHash:   r.PayloadHash,
	}
}

// ComputePayloadHash returns a deterministic SHA-256 hex digest over the
// canonical identity and assertion fields of an EvidenceRecord.
func ComputePayloadHash(rec EvidenceRecord) string {
	h := sha256.New()
	fmt.Fprintf(h, "%s|%s|%s|%s|%s|%s|%s|%s|%d|%d|%s",
		rec.EvidenceID,
		rec.SourceVersion,
		rec.TenantID,
		rec.ACLScope,
		rec.Fact.Subject,
		rec.Fact.Predicate,
		rec.Fact.Object,
		rec.Fact.Cardinality,
		rec.Fact.ValidFrom,
		rec.Fact.ValidTo,
		rec.RetractsEvidenceID,
	)
	return hex.EncodeToString(h.Sum(nil))
}

// AccessContext enforces mandatory pre-return tenant and ACL filtering.
// Access is fail-closed: an empty TenantID or empty AllowedScopes denies all reads.
type AccessContext struct {
	TenantID      string   `json:"tenant_id"`
	AllowedScopes []string `json:"allowed_scopes"`
}

// Allows reports whether this AccessContext is authorized to read data belonging
// to (tenantID, scope). Cross-tenant access is unconditionally denied.
func (ctx AccessContext) Allows(tenantID, scope string) bool {
	if ctx.TenantID == "" || tenantID == "" || ctx.TenantID != tenantID {
		return false
	}
	if len(ctx.AllowedScopes) == 0 || scope == "" {
		return false
	}
	for _, allowed := range ctx.AllowedScopes {
		if allowed == "*" || allowed == scope {
			return true
		}
	}
	return false
}

// BitemporalEdge is a derived edge version indexed along valid time and
// transaction time with provenance pointers back to the source evidence log.
type BitemporalEdge struct {
	EdgeID          string        `json:"edge_id"`
	Subject         string        `json:"subject"`
	Predicate       string        `json:"predicate"`
	Object          string        `json:"object"`
	Cardinality     Cardinality   `json:"cardinality"`
	ValidFrom       int64         `json:"valid_from"`
	ValidTo         int64         `json:"valid_to"`
	TxFrom          int64         `json:"tx_from"`
	TxTo            int64         `json:"tx_to"`
	Evidence        []EvidenceRef `json:"evidence"`
	TenantID        string        `json:"tenant_id"`
	ACLScope        string        `json:"acl_scope"`
	ExplicitValidTo bool          `json:"explicit_valid_to"`
	SupersededBy    string        `json:"superseded_by,omitempty"`
	ConflictGroupID string        `json:"conflict_group_id,omitempty"`
}

// Clone returns a deep copy of the edge so callers cannot mutate internal state.
func (e BitemporalEdge) Clone() BitemporalEdge {
	out := e
	if len(e.Evidence) > 0 {
		out.Evidence = make([]EvidenceRef, len(e.Evidence))
		copy(out.Evidence, e.Evidence)
	}
	return out
}

// EstimatedBytes returns a deterministic approximate heap footprint in bytes for
// benchmarking memory overhead across backends.
func (e BitemporalEdge) EstimatedBytes() int {
	size := 128 // fixed fields, slice headers, timestamps, flags
	size += len(e.EdgeID) + len(e.Subject) + len(e.Predicate) + len(e.Object)
	size += len(e.Cardinality) + len(e.TenantID) + len(e.ACLScope)
	size += len(e.SupersededBy) + len(e.ConflictGroupID)
	for _, ref := range e.Evidence {
		size += 48 + len(ref.EvidenceID) + len(ref.SourceVersion) + len(ref.PayloadHash)
	}
	return size
}

// ConflictingClaim represents one competing assertion in a ContradictionConflict.
type ConflictingClaim struct {
	Object    string        `json:"object"`
	EdgeID    string        `json:"edge_id"`
	ValidFrom int64         `json:"valid_from"`
	ValidTo   int64         `json:"valid_to"`
	Evidence  []EvidenceRef `json:"evidence"`
}

// Clone returns a deep copy of the claim.
func (c ConflictingClaim) Clone() ConflictingClaim {
	out := c
	if len(c.Evidence) > 0 {
		out.Evidence = make([]EvidenceRef, len(c.Evidence))
		copy(out.Evidence, c.Evidence)
	}
	return out
}

// ContradictionConflict records an explicit, unresolved conflict between two or
// more functional assertions whose valid-time intervals overlap without a
// supersession ordering.
type ContradictionConflict struct {
	ConflictID       string             `json:"conflict_id"`
	TenantID         string             `json:"tenant_id"`
	ACLScope         string             `json:"acl_scope"`
	Subject          string             `json:"subject"`
	Predicate        string             `json:"predicate"`
	OverlapValidFrom int64              `json:"overlap_valid_from"`
	OverlapValidTo   int64              `json:"overlap_valid_to"`
	DetectedAtTxTime int64              `json:"detected_at_tx_time"`
	TxTo             int64              `json:"tx_to"`
	CompetingValues  []ConflictingClaim `json:"competing_values"`
}

// Clone returns a deep copy of the conflict record.
func (c ContradictionConflict) Clone() ContradictionConflict {
	out := c
	if len(c.CompetingValues) > 0 {
		out.CompetingValues = make([]ConflictingClaim, len(c.CompetingValues))
		for i, claim := range c.CompetingValues {
			out.CompetingValues[i] = claim.Clone()
		}
	}
	return out
}

// QueryResult bundles deterministically ordered edges and any overlapping
// contradiction conflicts visible under the caller's AccessContext.
type QueryResult struct {
	Edges                []BitemporalEdge        `json:"edges"`
	Conflicts            []ContradictionConflict `json:"conflicts,omitempty"`
	HasContradiction     bool                    `json:"has_contradiction"`
	InspectedTenantEdges int                     `json:"inspected_tenant_edges"`
}

// DistinctObjects returns the sorted unique object values present in Edges.
func (r QueryResult) DistinctObjects() []string {
	if len(r.Edges) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(r.Edges))
	out := make([]string, 0, len(r.Edges))
	for _, e := range r.Edges {
		if _, ok := seen[e.Object]; !ok {
			seen[e.Object] = struct{}{}
			out = append(out, e.Object)
		}
	}
	sort.Strings(out)
	return out
}

// GraphStats summarizes the immutable evidence log and derived index state.
type GraphStats struct {
	EvidenceRecords    int `json:"evidence_records"`
	TotalEdgeVersions  int `json:"total_edge_versions"`
	ActiveTxEdges      int `json:"active_tx_edges"`
	CurrentValidEdges  int `json:"current_valid_edges"`
	ActiveConflicts    int `json:"active_conflicts"`
	EstimatedIndexByte int `json:"estimated_index_bytes"`
}

func mergeEvidenceRefs(existing []EvidenceRef, incoming EvidenceRef) []EvidenceRef {
	out := make([]EvidenceRef, 0, len(existing)+1)
	found := false
	for _, ref := range existing {
		out = append(out, ref)
		if ref.EvidenceID == incoming.EvidenceID && ref.SourceVersion == incoming.SourceVersion && ref.PayloadHash == incoming.PayloadHash {
			found = true
		}
	}
	if !found {
		out = append(out, incoming)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].EvidenceID != out[j].EvidenceID {
			return out[i].EvidenceID < out[j].EvidenceID
		}
		if out[i].SourceVersion != out[j].SourceVersion {
			return out[i].SourceVersion < out[j].SourceVersion
		}
		return out[i].PayloadHash < out[j].PayloadHash
	})
	return out
}

func sortEdgesDeterministically(edges []BitemporalEdge) {
	sort.Slice(edges, func(i, j int) bool {
		if edges[i].ValidFrom != edges[j].ValidFrom {
			return edges[i].ValidFrom < edges[j].ValidFrom
		}
		if effectiveEnd(edges[i].ValidTo) != effectiveEnd(edges[j].ValidTo) {
			return effectiveEnd(edges[i].ValidTo) < effectiveEnd(edges[j].ValidTo)
		}
		if edges[i].TxFrom != edges[j].TxFrom {
			return edges[i].TxFrom < edges[j].TxFrom
		}
		if edges[i].Subject != edges[j].Subject {
			return edges[i].Subject < edges[j].Subject
		}
		if edges[i].Predicate != edges[j].Predicate {
			return edges[i].Predicate < edges[j].Predicate
		}
		if edges[i].Object != edges[j].Object {
			return edges[i].Object < edges[j].Object
		}
		return edges[i].EdgeID < edges[j].EdgeID
	})
}

func sortConflictsDeterministically(conflicts []ContradictionConflict) {
	sort.Slice(conflicts, func(i, j int) bool {
		if conflicts[i].OverlapValidFrom != conflicts[j].OverlapValidFrom {
			return conflicts[i].OverlapValidFrom < conflicts[j].OverlapValidFrom
		}
		if conflicts[i].DetectedAtTxTime != conflicts[j].DetectedAtTxTime {
			return conflicts[i].DetectedAtTxTime < conflicts[j].DetectedAtTxTime
		}
		return conflicts[i].ConflictID < conflicts[j].ConflictID
	})
}

func deterministicConflictID(tenantID, scope, subject, predicate string, ovStart, ovEnd int64, objects []string) string {
	cp := make([]string, len(objects))
	copy(cp, objects)
	sort.Strings(cp)
	h := sha256.New()
	fmt.Fprintf(h, "conflict|%s|%s|%s|%s|%d|%d|%s",
		tenantID, scope, subject, predicate, ovStart, ovEnd, strings.Join(cp, ","))
	return "cnf-" + hex.EncodeToString(h.Sum(nil))[:16]
}
