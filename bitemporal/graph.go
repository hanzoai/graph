/*
 * SPDX-FileCopyrightText: © 2026 Hanzo AI, Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

package bitemporal

import (
	"fmt"
	"sort"
	"sync"
	"time"
)

type scopeKey struct {
	tenantID  string
	aclScope  string
	subject   string
	predicate string
}

type tenantScopePair struct {
	tenantID string
	aclScope string
}

type derivedIndex struct {
	edges          []BitemporalEdge
	edgesByKey     map[scopeKey][]int
	keysByScope    map[tenantScopePair][]scopeKey
	seenKeyInScope map[scopeKey]struct{}
	conflicts      []ContradictionConflict
	conflictsByKey map[scopeKey][]int
	edgeCounter    uint64
}

func newDerivedIndex() *derivedIndex {
	return &derivedIndex{
		edges:          make([]BitemporalEdge, 0, 64),
		edgesByKey:     make(map[scopeKey][]int),
		keysByScope:    make(map[tenantScopePair][]scopeKey),
		seenKeyInScope: make(map[scopeKey]struct{}),
		conflicts:      make([]ContradictionConflict, 0, 8),
		conflictsByKey: make(map[scopeKey][]int),
	}
}

func (idx *derivedIndex) nextEdgeID(recSeq uint64) string {
	idx.edgeCounter++
	return fmt.Sprintf("e-%06d-%06d", recSeq, idx.edgeCounter)
}

func (idx *derivedIndex) addEdge(e BitemporalEdge) int {
	pos := len(idx.edges)
	idx.edges = append(idx.edges, e)
	k := scopeKey{
		tenantID:  e.TenantID,
		aclScope:  e.ACLScope,
		subject:   e.Subject,
		predicate: e.Predicate,
	}
	idx.edgesByKey[k] = append(idx.edgesByKey[k], pos)
	if _, ok := idx.seenKeyInScope[k]; !ok {
		idx.seenKeyInScope[k] = struct{}{}
		ts := tenantScopePair{tenantID: e.TenantID, aclScope: e.ACLScope}
		idx.keysByScope[ts] = append(idx.keysByScope[ts], k)
	}
	return pos
}

func (idx *derivedIndex) addConflict(c ContradictionConflict) int {
	pos := len(idx.conflicts)
	idx.conflicts = append(idx.conflicts, c)
	k := scopeKey{
		tenantID:  c.TenantID,
		aclScope:  c.ACLScope,
		subject:   c.Subject,
		predicate: c.Predicate,
	}
	idx.conflictsByKey[k] = append(idx.conflictsByKey[k], pos)
	if _, ok := idx.seenKeyInScope[k]; !ok {
		idx.seenKeyInScope[k] = struct{}{}
		ts := tenantScopePair{tenantID: c.TenantID, aclScope: c.ACLScope}
		idx.keysByScope[ts] = append(idx.keysByScope[ts], k)
	}
	return pos
}

// Graph is a thread-safe bitemporal knowledge graph backed by an immutable
// append-only EvidenceLog and a completely derived, rebuildable bitemporal
// edge and contradiction index.
type Graph struct {
	mu         sync.RWMutex
	log        []EvidenceRecord
	lastTxTime int64
	derived    *derivedIndex
}

// NewGraph constructs an empty bitemporal Graph.
func NewGraph() *Graph {
	return &Graph{
		log:     make([]EvidenceRecord, 0, 64),
		derived: newDerivedIndex(),
	}
}

// NewFromEvidenceLog constructs a fresh Graph by replaying an immutable slice
// of EvidenceRecord entries from scratch.
func NewFromEvidenceLog(records []EvidenceRecord) (*Graph, error) {
	g := NewGraph()
	for _, rec := range records {
		if _, err := g.Ingest(rec); err != nil {
			return nil, err
		}
	}
	return g, nil
}

// Ingest validates and appends an immutable EvidenceRecord to the evidence log,
// then updates the derived bitemporal graph index non-destructively.
func (g *Graph) Ingest(rec EvidenceRecord) (EvidenceRecord, error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.ingestLocked(rec)
}

// IngestBatch ingests a slice of EvidenceRecords atomically with respect to
// concurrent readers. If any record fails validation, prior records in the
// batch are not applied (rolled back cleanly via RebuildFromLog).
func (g *Graph) IngestBatch(records []EvidenceRecord) ([]EvidenceRecord, error) {
	g.mu.Lock()
	defer g.mu.Unlock()

	origLen := len(g.log)
	origLastTx := g.lastTxTime
	out := make([]EvidenceRecord, 0, len(records))
	for _, rec := range records {
		stored, err := g.ingestLocked(rec)
		if err != nil {
			g.log = g.log[:origLen]
			g.lastTxTime = origLastTx
			g.rebuildLocked()
			return nil, err
		}
		out = append(out, stored)
	}
	return out, nil
}

func (g *Graph) ingestLocked(rec EvidenceRecord) (EvidenceRecord, error) {
	if rec.EvidenceID == "" {
		return EvidenceRecord{}, ErrEmptyEvidenceID
	}
	if rec.SourceVersion == "" {
		return EvidenceRecord{}, ErrEmptySourceVersion
	}
	if rec.TenantID == "" {
		return EvidenceRecord{}, ErrEmptyTenantID
	}
	if rec.ACLScope == "" {
		return EvidenceRecord{}, ErrEmptyACLScope
	}
	if rec.RetractsEvidenceID == "" {
		if rec.Fact.Subject == "" || rec.Fact.Predicate == "" {
			return EvidenceRecord{}, ErrInvalidFactTriple
		}
		if rec.Fact.Object == "" {
			return EvidenceRecord{}, ErrEmptyFactObject
		}
		if rec.Fact.Cardinality == "" {
			rec.Fact.Cardinality = FunctionalSingleValue
		}
		if rec.Fact.Cardinality != FunctionalSingleValue && rec.Fact.Cardinality != MultiValue {
			return EvidenceRecord{}, ErrInvalidCardinality
		}
	} else {
		if rec.Fact.Subject == "" || rec.Fact.Predicate == "" {
			return EvidenceRecord{}, ErrInvalidFactTriple
		}
		if rec.Fact.Cardinality == "" {
			rec.Fact.Cardinality = FunctionalSingleValue
		}
	}
	if rec.Fact.ValidFrom <= 0 {
		return EvidenceRecord{}, ErrInvalidValidRange
	}
	if rec.Fact.ValidTo != OpenEnd && rec.Fact.ValidTo <= rec.Fact.ValidFrom {
		return EvidenceRecord{}, ErrInvalidValidRange
	}

	if rec.ObservedAtTxTime == 0 {
		now := time.Now().UTC().UnixNano()
		if now <= g.lastTxTime {
			now = g.lastTxTime + 1
		}
		rec.ObservedAtTxTime = now
	} else if rec.ObservedAtTxTime < g.lastTxTime {
		return EvidenceRecord{}, fmt.Errorf("%w: got %d < last %d", ErrNonMonotonicTxTime, rec.ObservedAtTxTime, g.lastTxTime)
	}
	g.lastTxTime = rec.ObservedAtTxTime

	if rec.PayloadHash == "" {
		rec.PayloadHash = ComputePayloadHash(rec)
	}
	rec.Seq = uint64(len(g.log) + 1)
	g.log = append(g.log, rec)
	g.applyRecordLocked(rec)
	return rec, nil
}

// RebuildFromLog drops the entire derived graph index and replays the immutable
// append-only EvidenceLog from sequence 1 to reproduce bit-identical edges,
// conflicts, and query answers.
func (g *Graph) RebuildFromLog() {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.rebuildLocked()
}

func (g *Graph) rebuildLocked() {
	g.derived = newDerivedIndex()
	for _, rec := range g.log {
		g.applyRecordLocked(rec)
	}
}

// ExportEvidenceLog returns a defensive copy of the immutable append-only
// evidence log.
func (g *Graph) ExportEvidenceLog() []EvidenceRecord {
	g.mu.RLock()
	defer g.mu.RUnlock()
	out := make([]EvidenceRecord, len(g.log))
	copy(out, g.log)
	return out
}

// ExportDerivedEdges returns a defensive copy of all derived edges (internal
// diagnostics / rebuild equivalence checks).
func (g *Graph) ExportDerivedEdges() []BitemporalEdge {
	g.mu.RLock()
	defer g.mu.RUnlock()
	out := make([]BitemporalEdge, len(g.derived.edges))
	for i, e := range g.derived.edges {
		out[i] = e.Clone()
	}
	return out
}

// ExportDerivedConflicts returns a defensive copy of all derived conflicts.
func (g *Graph) ExportDerivedConflicts() []ContradictionConflict {
	g.mu.RLock()
	defer g.mu.RUnlock()
	out := make([]ContradictionConflict, len(g.derived.conflicts))
	for i, c := range g.derived.conflicts {
		out[i] = c.Clone()
	}
	return out
}

// Stats returns aggregate statistics over the immutable log and derived index.
func (g *Graph) Stats() GraphStats {
	g.mu.RLock()
	defer g.mu.RUnlock()

	st := GraphStats{
		EvidenceRecords:   len(g.log),
		TotalEdgeVersions: len(g.derived.edges),
	}
	for _, e := range g.derived.edges {
		st.EstimatedIndexByte += e.EstimatedBytes()
		if isOpenEnd(e.TxTo) {
			st.ActiveTxEdges++
			if isOpenEnd(e.ValidTo) {
				st.CurrentValidEdges++
			}
		}
	}
	for _, c := range g.derived.conflicts {
		if isOpenEnd(c.TxTo) {
			st.ActiveConflicts++
		}
		st.EstimatedIndexByte += 160 + len(c.CompetingValues)*96
	}
	return st
}

func (g *Graph) applyRecordLocked(rec EvidenceRecord) {
	if rec.RetractsEvidenceID != "" {
		g.applyRetractionLocked(rec)
		return
	}
	if rec.Fact.Cardinality == MultiValue {
		g.applyMultiValueLocked(rec)
		return
	}
	g.applyFunctionalLocked(rec)
}

func (g *Graph) applyRetractionLocked(rec EvidenceRecord) {
	k := scopeKey{
		tenantID:  rec.TenantID,
		aclScope:  rec.ACLScope,
		subject:   rec.Fact.Subject,
		predicate: rec.Fact.Predicate,
	}
	tx := rec.ObservedAtTxTime
	retractAt := rec.Fact.ValidFrom
	indices := append([]int(nil), g.derived.edgesByKey[k]...)
	for _, idx := range indices {
		e := g.derived.edges[idx]
		if !isOpenEnd(e.TxTo) {
			continue
		}
		matched := false
		for _, ref := range e.Evidence {
			if ref.EvidenceID == rec.RetractsEvidenceID {
				matched = true
				break
			}
		}
		if !matched {
			continue
		}
		// Close this edge version in transaction time without deleting history.
		g.derived.edges[idx].TxTo = tx
		g.derived.edges[idx].SupersededBy = rec.EvidenceID
		// If retraction closes validity after ValidFrom, preserve [ValidFrom, retractAt).
		if retractAt > e.ValidFrom && (isOpenEnd(e.ValidTo) || retractAt < e.ValidTo) {
			cont := e.Clone()
			cont.EdgeID = g.derived.nextEdgeID(rec.Seq)
			cont.ValidTo = retractAt
			cont.TxFrom = tx
			cont.TxTo = OpenEnd
			cont.SupersededBy = ""
			g.derived.addEdge(cont)
		}
	}
}

func (g *Graph) applyMultiValueLocked(rec EvidenceRecord) {
	k := scopeKey{
		tenantID:  rec.TenantID,
		aclScope:  rec.ACLScope,
		subject:   rec.Fact.Subject,
		predicate: rec.Fact.Predicate,
	}
	tx := rec.ObservedAtTxTime
	vFrom := rec.Fact.ValidFrom
	vTo := rec.Fact.ValidTo
	ref := rec.Ref()

	indices := append([]int(nil), g.derived.edgesByKey[k]...)
	for _, idx := range indices {
		e := g.derived.edges[idx]
		if !isOpenEnd(e.TxTo) {
			continue
		}
		if e.Object == rec.Fact.Object && e.ValidFrom == vFrom && e.ValidTo == vTo {
			if e.TxFrom == tx {
				g.derived.edges[idx].Evidence = mergeEvidenceRefs(e.Evidence, ref)
				return
			}
			g.derived.edges[idx].TxTo = tx
			g.derived.edges[idx].SupersededBy = rec.EvidenceID
			updated := e.Clone()
			updated.EdgeID = g.derived.nextEdgeID(rec.Seq)
			updated.TxFrom = tx
			updated.TxTo = OpenEnd
			updated.SupersededBy = ""
			updated.Evidence = mergeEvidenceRefs(e.Evidence, ref)
			g.derived.addEdge(updated)
			return
		}
	}

	g.derived.addEdge(BitemporalEdge{
		EdgeID:          g.derived.nextEdgeID(rec.Seq),
		Subject:         rec.Fact.Subject,
		Predicate:       rec.Fact.Predicate,
		Object:          rec.Fact.Object,
		Cardinality:     MultiValue,
		ValidFrom:       vFrom,
		ValidTo:         vTo,
		TxFrom:          tx,
		TxTo:            OpenEnd,
		Evidence:        []EvidenceRef{ref},
		TenantID:        rec.TenantID,
		ACLScope:        rec.ACLScope,
		ExplicitValidTo: !isOpenEnd(vTo),
	})
}

func (g *Graph) applyFunctionalLocked(rec EvidenceRecord) {
	k := scopeKey{
		tenantID:  rec.TenantID,
		aclScope:  rec.ACLScope,
		subject:   rec.Fact.Subject,
		predicate: rec.Fact.Predicate,
	}
	tx := rec.ObservedAtTxTime
	vFrom := rec.Fact.ValidFrom
	vTo := rec.Fact.ValidTo
	explicitVTo := !isOpenEnd(vTo)
	obj := rec.Fact.Object
	ref := rec.Ref()

	// Snapshot active edges for this functional key in deterministic order.
	activeIndices := make([]int, 0, len(g.derived.edgesByKey[k]))
	for _, idx := range g.derived.edgesByKey[k] {
		if isOpenEnd(g.derived.edges[idx].TxTo) {
			activeIndices = append(activeIndices, idx)
		}
	}
	sort.Slice(activeIndices, func(i, j int) bool {
		ei := g.derived.edges[activeIndices[i]]
		ej := g.derived.edges[activeIndices[j]]
		if ei.ValidFrom != ej.ValidFrom {
			return ei.ValidFrom < ej.ValidFrom
		}
		if effectiveEnd(ei.ValidTo) != effectiveEnd(ej.ValidTo) {
			return effectiveEnd(ei.ValidTo) < effectiveEnd(ej.ValidTo)
		}
		return ei.EdgeID < ej.EdgeID
	})

	// 1. Check for corroboration (same object, same ValidFrom, compatible ValidTo).
	for _, idx := range activeIndices {
		e := g.derived.edges[idx]
		if e.Object == obj && e.ValidFrom == vFrom && (!explicitVTo || e.ValidTo == vTo) {
			if e.TxFrom == tx {
				g.derived.edges[idx].Evidence = mergeEvidenceRefs(e.Evidence, ref)
				return
			}
			g.derived.edges[idx].TxTo = tx
			g.derived.edges[idx].SupersededBy = rec.EvidenceID
			updated := e.Clone()
			updated.EdgeID = g.derived.nextEdgeID(rec.Seq)
			updated.TxFrom = tx
			updated.TxTo = OpenEnd
			updated.SupersededBy = ""
			updated.Evidence = mergeEvidenceRefs(e.Evidence, ref)
			g.derived.addEdge(updated)
			return
		}
	}

	// 2. Determine the effective upper bound of the incoming edge for open-ended
	// state transitions (delayed out-of-order retro-events are bounded by the
	// earliest strictly later active transition so they splice cleanly without
	// corrupting future history).
	effectiveVTo := vTo
	if !explicitVTo {
		for _, idx := range activeIndices {
			e := g.derived.edges[idx]
			if e.ValidFrom > vFrom {
				effectiveVTo = e.ValidFrom
				break
			}
		}
	}

	// 3. Detect explicit contradictions against active edges.
	// A contradiction occurs when two distinct functional values claim overlapping
	// valid intervals and cannot be ordered as a non-conflicting state transition:
	//   (a) exact same ValidFrom with different Object; OR
	//   (b) overlapping intervals where the earlier-starting edge had an explicit
	//       closed ValidTo (guaranteed validity window); OR
	//   (c) incoming has an explicit closed ValidTo that extends across an existing
	//       edge starting strictly after vFrom.
	contradictingIdx := make([]int, 0)
	for _, idx := range activeIndices {
		e := g.derived.edges[idx]
		if e.Object == obj {
			continue
		}
		_, _, overlaps := intervalsOverlap(e.ValidFrom, e.ValidTo, vFrom, effectiveVTo)
		if !overlaps {
			continue
		}
		isConflict := false
		if e.ValidFrom == vFrom {
			isConflict = true
		} else if e.ValidFrom < vFrom && e.ExplicitValidTo {
			isConflict = true
		} else if vFrom < e.ValidFrom && explicitVTo {
			isConflict = true
		}
		if isConflict {
			contradictingIdx = append(contradictingIdx, idx)
		}
	}

	if len(contradictingIdx) > 0 {
		// Record the new edge non-destructively alongside its conflicting peers
		// and emit explicit ContradictionConflict records with all evidence pointers.
		newEdge := BitemporalEdge{
			EdgeID:          g.derived.nextEdgeID(rec.Seq),
			Subject:         rec.Fact.Subject,
			Predicate:       rec.Fact.Predicate,
			Object:          obj,
			Cardinality:     FunctionalSingleValue,
			ValidFrom:       vFrom,
			ValidTo:         effectiveVTo,
			TxFrom:          tx,
			TxTo:            OpenEnd,
			Evidence:        []EvidenceRef{ref},
			TenantID:        rec.TenantID,
			ACLScope:        rec.ACLScope,
			ExplicitValidTo: explicitVTo,
		}
		newPos := g.derived.addEdge(newEdge)

		// Also truncate any earlier non-conflicting open-ended edge that overlaps vFrom.
		for _, idx := range activeIndices {
			if containsInt(contradictingIdx, idx) {
				continue
			}
			e := g.derived.edges[idx]
			if e.ValidFrom < vFrom && effectiveEnd(e.ValidTo) > vFrom && !e.ExplicitValidTo {
				g.truncateEdgeAtLocked(idx, vFrom, tx, rec.Seq, rec.EvidenceID)
			}
		}

		// Create explicit ContradictionConflict for each conflicting active peer.
		for _, cIdx := range contradictingIdx {
			peer := g.derived.edges[cIdx]
			ovStart, ovEnd, ok := intervalsOverlap(peer.ValidFrom, peer.ValidTo, newEdge.ValidFrom, newEdge.ValidTo)
			if !ok {
				continue
			}
			conflictID := deterministicConflictID(
				rec.TenantID, rec.ACLScope, rec.Fact.Subject, rec.Fact.Predicate,
				ovStart, ovEnd, []string{peer.Object, newEdge.Object},
			)

			// Tag both edges with ConflictGroupID non-destructively in tx-time.
			peerEdgeID := peer.EdgeID
			if peer.ConflictGroupID == "" {
				if peer.TxFrom == tx {
					g.derived.edges[cIdx].ConflictGroupID = conflictID
				} else {
					g.derived.edges[cIdx].TxTo = tx
					g.derived.edges[cIdx].SupersededBy = rec.EvidenceID
					tagged := peer.Clone()
					tagged.EdgeID = g.derived.nextEdgeID(rec.Seq)
					tagged.TxFrom = tx
					tagged.TxTo = OpenEnd
					tagged.SupersededBy = ""
					tagged.ConflictGroupID = conflictID
					g.derived.addEdge(tagged)
					peerEdgeID = tagged.EdgeID
				}
			}
			if g.derived.edges[newPos].ConflictGroupID == "" {
				g.derived.edges[newPos].ConflictGroupID = conflictID
			}

			claims := []ConflictingClaim{
				{
					Object:    peer.Object,
					EdgeID:    peerEdgeID,
					ValidFrom: peer.ValidFrom,
					ValidTo:   peer.ValidTo,
					Evidence:  cloneRefs(peer.Evidence),
				},
				{
					Object:    newEdge.Object,
					EdgeID:    newEdge.EdgeID,
					ValidFrom: newEdge.ValidFrom,
					ValidTo:   newEdge.ValidTo,
					Evidence:  cloneRefs(newEdge.Evidence),
				},
			}
			sort.Slice(claims, func(i, j int) bool {
				if claims[i].Object != claims[j].Object {
					return claims[i].Object < claims[j].Object
				}
				return claims[i].EdgeID < claims[j].EdgeID
			})

			g.derived.addConflict(ContradictionConflict{
				ConflictID:       conflictID,
				TenantID:         rec.TenantID,
				ACLScope:         rec.ACLScope,
				Subject:          rec.Fact.Subject,
				Predicate:        rec.Fact.Predicate,
				OverlapValidFrom: ovStart,
				OverlapValidTo:   ovEnd,
				DetectedAtTxTime: tx,
				TxTo:             OpenEnd,
				CompetingValues:  claims,
			})
		}
		return
	}

	// 4. Non-destructive forward supersession & mid-history delayed splicing:
	// Any active edge starting strictly before vFrom whose valid interval extends
	// past vFrom is closed in transaction time (preserving full auditability) and
	// replaced at tx with a truncated historical continuation [e.ValidFrom, vFrom).
	for _, idx := range activeIndices {
		e := g.derived.edges[idx]
		if e.ValidFrom < vFrom && effectiveEnd(e.ValidTo) > vFrom {
			g.truncateEdgeAtLocked(idx, vFrom, tx, rec.Seq, rec.EvidenceID)
		}
	}

	// Similarly, if an active open-ended conflict started before vFrom and extended
	// past vFrom, bound its valid-time overlap at vFrom in tx-time so future queries
	// at >= vFrom see the resolved superseding state while historical queries before
	// vFrom still surface the conflict.
	g.truncateConflictsAtLocked(k, vFrom, tx)

	// 5. Insert the new derived edge version.
	g.derived.addEdge(BitemporalEdge{
		EdgeID:          g.derived.nextEdgeID(rec.Seq),
		Subject:         rec.Fact.Subject,
		Predicate:       rec.Fact.Predicate,
		Object:          obj,
		Cardinality:     FunctionalSingleValue,
		ValidFrom:       vFrom,
		ValidTo:         effectiveVTo,
		TxFrom:          tx,
		TxTo:            OpenEnd,
		Evidence:        []EvidenceRef{ref},
		TenantID:        rec.TenantID,
		ACLScope:        rec.ACLScope,
		ExplicitValidTo: explicitVTo,
	})
}

func (g *Graph) truncateEdgeAtLocked(idx int, splitValidTime, tx int64, recSeq uint64, supersededBy string) {
	e := g.derived.edges[idx]
	if e.TxFrom == tx {
		g.derived.edges[idx].ValidTo = splitValidTime
		return
	}
	g.derived.edges[idx].TxTo = tx
	g.derived.edges[idx].SupersededBy = supersededBy

	cont := e.Clone()
	cont.EdgeID = g.derived.nextEdgeID(recSeq)
	cont.ValidTo = splitValidTime
	cont.TxFrom = tx
	cont.TxTo = OpenEnd
	cont.SupersededBy = ""
	g.derived.addEdge(cont)
}

func (g *Graph) truncateConflictsAtLocked(k scopeKey, splitValidTime, tx int64) {
	cIndices := append([]int(nil), g.derived.conflictsByKey[k]...)
	for _, idx := range cIndices {
		c := g.derived.conflicts[idx]
		if !isOpenEnd(c.TxTo) {
			continue
		}
		if c.OverlapValidFrom < splitValidTime && effectiveEnd(c.OverlapValidTo) > splitValidTime {
			if c.DetectedAtTxTime == tx {
				g.derived.conflicts[idx].OverlapValidTo = splitValidTime
				continue
			}
			g.derived.conflicts[idx].TxTo = tx
			cont := c.Clone()
			cont.OverlapValidTo = splitValidTime
			cont.DetectedAtTxTime = tx
			cont.TxTo = OpenEnd
			g.derived.addConflict(cont)
		}
	}
}

// QueryCurrent returns all currently believed (TxTo == OpenEnd) and currently
// valid (ValidTo == OpenEnd) edges matching (subject, predicate) strictly within
// the caller's AccessContext. Empty subject or predicate acts as a wildcard
// within the authorized tenant and ACL scopes.
func (g *Graph) QueryCurrent(ctx AccessContext, subject, predicate string) QueryResult {
	g.mu.RLock()
	defer g.mu.RUnlock()

	keys := g.authorizedKeysLocked(ctx, subject, predicate)
	res := QueryResult{}
	for _, k := range keys {
		for _, idx := range g.derived.edgesByKey[k] {
			e := g.derived.edges[idx]
			if !ctx.Allows(e.TenantID, e.ACLScope) {
				continue
			}
			res.InspectedTenantEdges++
			if isOpenEnd(e.TxTo) && isOpenEnd(e.ValidTo) {
				res.Edges = append(res.Edges, e.Clone())
			}
		}
		for _, cIdx := range g.derived.conflictsByKey[k] {
			c := g.derived.conflicts[cIdx]
			if !ctx.Allows(c.TenantID, c.ACLScope) {
				continue
			}
			if isOpenEnd(c.TxTo) && isOpenEnd(c.OverlapValidTo) {
				res.Conflicts = append(res.Conflicts, c.Clone())
			}
		}
	}
	sortEdgesDeterministically(res.Edges)
	sortConflictsDeterministically(res.Conflicts)
	res.HasContradiction = len(res.Conflicts) > 0
	return res
}

// QueryAsOf returns what the system currently believes (TxTo == OpenEnd) was
// true at validTime, strictly filtered by ctx before returning any record.
func (g *Graph) QueryAsOf(ctx AccessContext, subject, predicate string, validTime int64) QueryResult {
	g.mu.RLock()
	defer g.mu.RUnlock()

	keys := g.authorizedKeysLocked(ctx, subject, predicate)
	res := QueryResult{}
	for _, k := range keys {
		for _, idx := range g.derived.edgesByKey[k] {
			e := g.derived.edges[idx]
			if !ctx.Allows(e.TenantID, e.ACLScope) {
				continue
			}
			res.InspectedTenantEdges++
			if isOpenEnd(e.TxTo) && containsTime(e.ValidFrom, e.ValidTo, validTime) {
				res.Edges = append(res.Edges, e.Clone())
			}
		}
		for _, cIdx := range g.derived.conflictsByKey[k] {
			c := g.derived.conflicts[cIdx]
			if !ctx.Allows(c.TenantID, c.ACLScope) {
				continue
			}
			if isOpenEnd(c.TxTo) && containsTime(c.OverlapValidFrom, c.OverlapValidTo, validTime) {
				res.Conflicts = append(res.Conflicts, c.Clone())
			}
		}
	}
	sortEdgesDeterministically(res.Edges)
	sortConflictsDeterministically(res.Conflicts)
	res.HasContradiction = len(res.Conflicts) > 0
	return res
}

// QueryBitemporal answers the exact audit question: "As known at txTime, what
// was true at validTime?" Strictly filtered by ctx before returning any record.
func (g *Graph) QueryBitemporal(ctx AccessContext, subject, predicate string, validTime, txTime int64) QueryResult {
	g.mu.RLock()
	defer g.mu.RUnlock()

	keys := g.authorizedKeysLocked(ctx, subject, predicate)
	res := QueryResult{}
	for _, k := range keys {
		for _, idx := range g.derived.edgesByKey[k] {
			e := g.derived.edges[idx]
			if !ctx.Allows(e.TenantID, e.ACLScope) {
				continue
			}
			res.InspectedTenantEdges++
			if containsTime(e.TxFrom, e.TxTo, txTime) && containsTime(e.ValidFrom, e.ValidTo, validTime) {
				res.Edges = append(res.Edges, e.Clone())
			}
		}
		for _, cIdx := range g.derived.conflictsByKey[k] {
			c := g.derived.conflicts[cIdx]
			if !ctx.Allows(c.TenantID, c.ACLScope) {
				continue
			}
			if containsTime(c.DetectedAtTxTime, c.TxTo, txTime) && containsTime(c.OverlapValidFrom, c.OverlapValidTo, validTime) {
				res.Conflicts = append(res.Conflicts, c.Clone())
			}
		}
	}
	sortEdgesDeterministically(res.Edges)
	sortConflictsDeterministically(res.Conflicts)
	res.HasContradiction = len(res.Conflicts) > 0
	return res
}

// QueryHistory returns the complete non-destructive history of all derived edge
// versions (both active and superseded in valid time and transaction time)
// matching (subject, predicate) within ctx.
func (g *Graph) QueryHistory(ctx AccessContext, subject, predicate string) QueryResult {
	g.mu.RLock()
	defer g.mu.RUnlock()

	keys := g.authorizedKeysLocked(ctx, subject, predicate)
	res := QueryResult{}
	for _, k := range keys {
		for _, idx := range g.derived.edgesByKey[k] {
			e := g.derived.edges[idx]
			if !ctx.Allows(e.TenantID, e.ACLScope) {
				continue
			}
			res.InspectedTenantEdges++
			res.Edges = append(res.Edges, e.Clone())
		}
		for _, cIdx := range g.derived.conflictsByKey[k] {
			c := g.derived.conflicts[cIdx]
			if !ctx.Allows(c.TenantID, c.ACLScope) {
				continue
			}
			res.Conflicts = append(res.Conflicts, c.Clone())
		}
	}
	sortEdgesDeterministically(res.Edges)
	sortConflictsDeterministically(res.Conflicts)
	res.HasContradiction = len(res.Conflicts) > 0
	return res
}

// authorizedKeysLocked resolves ONLY index keys authorized by ctx BEFORE
// inspecting or scoring any edge. This enforces strict pre-return tenant and
// ACL isolation.
func (g *Graph) authorizedKeysLocked(ctx AccessContext, subject, predicate string) []scopeKey {
	if ctx.TenantID == "" || len(ctx.AllowedScopes) == 0 {
		return nil
	}

	hasWildcard := false
	for _, s := range ctx.AllowedScopes {
		if s == "*" {
			hasWildcard = true
			break
		}
	}

	var scopes []string
	if hasWildcard {
		for pair := range g.derived.keysByScope {
			if pair.tenantID == ctx.TenantID {
				scopes = append(scopes, pair.aclScope)
			}
		}
		sort.Strings(scopes)
	} else {
		seen := make(map[string]struct{}, len(ctx.AllowedScopes))
		for _, s := range ctx.AllowedScopes {
			if s == "" {
				continue
			}
			if _, ok := seen[s]; !ok {
				seen[s] = struct{}{}
				scopes = append(scopes, s)
			}
		}
		sort.Strings(scopes)
	}

	keys := make([]scopeKey, 0)
	for _, sc := range scopes {
		if subject != "" && predicate != "" {
			k := scopeKey{
				tenantID:  ctx.TenantID,
				aclScope:  sc,
				subject:   subject,
				predicate: predicate,
			}
			if _, ok := g.derived.seenKeyInScope[k]; ok {
				keys = append(keys, k)
			}
			continue
		}
		for _, k := range g.derived.keysByScope[tenantScopePair{tenantID: ctx.TenantID, aclScope: sc}] {
			if subject != "" && k.subject != subject {
				continue
			}
			if predicate != "" && k.predicate != predicate {
				continue
			}
			keys = append(keys, k)
		}
	}
	return keys
}

func containsInt(list []int, target int) bool {
	for _, v := range list {
		if v == target {
			return true
		}
	}
	return false
}

func cloneRefs(refs []EvidenceRef) []EvidenceRef {
	if len(refs) == 0 {
		return nil
	}
	out := make([]EvidenceRef, len(refs))
	copy(out, refs)
	return out
}
