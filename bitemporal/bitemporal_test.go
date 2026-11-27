/*
 * SPDX-FileCopyrightText: © 2026 Hanzo AI, Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

package bitemporal

import (
	"errors"
	"fmt"
	"reflect"
	"sync"
	"testing"
	"time"
)

func testClock(hour, min int) int64 {
	return Ts(time.Date(2026, 5, 1, hour, min, 0, 0, time.UTC))
}

func TestPR47CoexistenceAndZeroDeletion(t *testing.T) {
	g := NewGraph()
	ctx := AccessContext{TenantID: "hanzo", AllowedScopes: []string{"repo:graph"}}

	// 10:00 AM: PR #47 is open.
	ev1, err := g.Ingest(EvidenceRecord{
		EvidenceID:       "gh-webhook-pr47-open",
		SourceVersion:    "sha-1000",
		TenantID:         "hanzo",
		ACLScope:         "repo:graph",
		ObservedAtTxTime: testClock(10, 0),
		Fact: FactTriple{
			Subject:     "pr#47",
			Predicate:   "pr.state",
			Object:      "open",
			Cardinality: FunctionalSingleValue,
			ValidFrom:   testClock(10, 0),
			ValidTo:     OpenEnd,
		},
	})
	if err != nil {
		t.Fatalf("ingest open failed: %v", err)
	}

	// 11:00 AM: PR #47 is merged (supersedes open without deleting it!).
	ev2, err := g.Ingest(EvidenceRecord{
		EvidenceID:       "gh-webhook-pr47-merged",
		SourceVersion:    "sha-1100",
		TenantID:         "hanzo",
		ACLScope:         "repo:graph",
		ObservedAtTxTime: testClock(11, 0),
		Fact: FactTriple{
			Subject:     "pr#47",
			Predicate:   "pr.state",
			Object:      "merged",
			Cardinality: FunctionalSingleValue,
			ValidFrom:   testClock(11, 0),
			ValidTo:     OpenEnd,
		},
	})
	if err != nil {
		t.Fatalf("ingest merged failed: %v", err)
	}

	// 1. Current state must return ONLY "merged" with ev2 provenance.
	cur := g.QueryCurrent(ctx, "pr#47", "pr.state")
	if cur.HasContradiction {
		t.Fatalf("expected no contradiction, got %+v", cur.Conflicts)
	}
	if got := cur.DistinctObjects(); !reflect.DeepEqual(got, []string{"merged"}) {
		t.Fatalf("QueryCurrent objects = %v, want [merged]", got)
	}
	if len(cur.Edges) != 1 || cur.Edges[0].Evidence[0].EvidenceID != ev2.EvidenceID {
		t.Fatalf("QueryCurrent provenance mismatch: %+v", cur.Edges)
	}

	// 2. Historical valid-time QueryAsOf(10:30) must return "open" with [10:00, 11:00) and ev1 provenance.
	asOf1030 := g.QueryAsOf(ctx, "pr#47", "pr.state", testClock(10, 30))
	if got := asOf1030.DistinctObjects(); !reflect.DeepEqual(got, []string{"open"}) {
		t.Fatalf("QueryAsOf(10:30) objects = %v, want [open]", got)
	}
	if len(asOf1030.Edges) != 1 {
		t.Fatalf("expected 1 edge at 10:30, got %d", len(asOf1030.Edges))
	}
	if asOf1030.Edges[0].ValidFrom != testClock(10, 0) || asOf1030.Edges[0].ValidTo != testClock(11, 0) {
		t.Fatalf("expected open interval [10:00, 11:00), got [%s, %s)",
			FormatTs(asOf1030.Edges[0].ValidFrom), FormatTs(asOf1030.Edges[0].ValidTo))
	}
	if asOf1030.Edges[0].Evidence[0].EvidenceID != ev1.EvidenceID {
		t.Fatalf("expected ev1 provenance on historical open edge, got %+v", asOf1030.Edges[0].Evidence)
	}

	// 3. QueryAsOf(11:30) must return "merged".
	asOf1130 := g.QueryAsOf(ctx, "pr#47", "pr.state", testClock(11, 30))
	if got := asOf1130.DistinctObjects(); !reflect.DeepEqual(got, []string{"merged"}) {
		t.Fatalf("QueryAsOf(11:30) objects = %v, want [merged]", got)
	}

	// 4. Bitemporal audit QueryBitemporal(valid=10:30, tx=10:15) -> "open".
	bt1030 := g.QueryBitemporal(ctx, "pr#47", "pr.state", testClock(10, 30), testClock(10, 15))
	if got := bt1030.DistinctObjects(); !reflect.DeepEqual(got, []string{"open"}) {
		t.Fatalf("QueryBitemporal(10:30, tx=10:15) = %v, want [open]", got)
	}

	// 5. Bitemporal audit QueryBitemporal(valid=11:30, tx=10:15) -> "open"
	// because at tx=10:15 the system had not yet learned of the 11:00 merge.
	bt1130BeforeMerge := g.QueryBitemporal(ctx, "pr#47", "pr.state", testClock(11, 30), testClock(10, 15))
	if got := bt1130BeforeMerge.DistinctObjects(); !reflect.DeepEqual(got, []string{"open"}) {
		t.Fatalf("QueryBitemporal(11:30, tx=10:15) = %v, want [open]", got)
	}

	// 6. Zero deletion: evidence log retains both records; history retains all 3 edge versions
	// (original open [10:00, Inf) closed at tx=11:00, truncated open [10:00, 11:00), and merged [11:00, Inf)).
	if len(g.ExportEvidenceLog()) != 2 {
		t.Fatalf("expected 2 immutable evidence records, got %d", len(g.ExportEvidenceLog()))
	}
	hist := g.QueryHistory(ctx, "pr#47", "pr.state")
	if len(hist.Edges) != 3 {
		t.Fatalf("expected 3 non-destructive edge versions in history, got %d", len(hist.Edges))
	}
}

func TestDelayedOutOfOrderSplicingAndAuditability(t *testing.T) {
	g := NewGraph()
	ctx := AccessContext{TenantID: "hanzo", AllowedScopes: []string{"repo:graph"}}

	// Ingest 10:00 open, then 11:00 merged.
	_, err := g.Ingest(EvidenceRecord{
		EvidenceID: "ev-open", SourceVersion: "v1", TenantID: "hanzo", ACLScope: "repo:graph",
		ObservedAtTxTime: testClock(10, 0),
		Fact: FactTriple{
			Subject: "pr#47", Predicate: "pr.state", Object: "open",
			Cardinality: FunctionalSingleValue, ValidFrom: testClock(10, 0),
		},
	})
	if err != nil {
		t.Fatalf("ingest open: %v", err)
	}
	_, err = g.Ingest(EvidenceRecord{
		EvidenceID: "ev-merged", SourceVersion: "v2", TenantID: "hanzo", ACLScope: "repo:graph",
		ObservedAtTxTime: testClock(11, 0),
		Fact: FactTriple{
			Subject: "pr#47", Predicate: "pr.state", Object: "merged",
			Cardinality: FunctionalSingleValue, ValidFrom: testClock(11, 0),
		},
	})
	if err != nil {
		t.Fatalf("ingest merged: %v", err)
	}

	// At Tx=12:00, delayed earlier retro-fact (09:00 draft) arrives out of order.
	_, err = g.Ingest(EvidenceRecord{
		EvidenceID: "ev-draft-delayed", SourceVersion: "v0-late", TenantID: "hanzo", ACLScope: "repo:graph",
		ObservedAtTxTime: testClock(12, 0),
		Fact: FactTriple{
			Subject: "pr#47", Predicate: "pr.state", Object: "draft",
			Cardinality: FunctionalSingleValue, ValidFrom: testClock(9, 0),
		},
	})
	if err != nil {
		t.Fatalf("ingest delayed draft: %v", err)
	}

	// At Tx=12:30, delayed mid-history retro-fact (10:30 review_requested) arrives out of order.
	_, err = g.Ingest(EvidenceRecord{
		EvidenceID: "ev-review-delayed", SourceVersion: "v1b-late", TenantID: "hanzo", ACLScope: "repo:graph",
		ObservedAtTxTime: testClock(12, 30),
		Fact: FactTriple{
			Subject: "pr#47", Predicate: "pr.state", Object: "review_requested",
			Cardinality: FunctionalSingleValue, ValidFrom: testClock(10, 30),
		},
	})
	if err != nil {
		t.Fatalf("ingest delayed review: %v", err)
	}

	// Verify exact valid-time convergence across the full timeline.
	checks := []struct {
		at   int64
		want string
	}{
		{testClock(9, 30), "draft"},
		{testClock(10, 15), "open"},
		{testClock(10, 45), "review_requested"},
		{testClock(11, 30), "merged"},
	}
	for _, tc := range checks {
		r := g.QueryAsOf(ctx, "pr#47", "pr.state", tc.at)
		if got := r.DistinctObjects(); !reflect.DeepEqual(got, []string{tc.want}) {
			t.Fatalf("QueryAsOf(%s) = %v, want [%s]", FormatTs(tc.at), got, tc.want)
		}
	}

	// Current state must still be "merged" without corruption.
	if got := g.QueryCurrent(ctx, "pr#47", "pr.state").DistinctObjects(); !reflect.DeepEqual(got, []string{"merged"}) {
		t.Fatalf("QueryCurrent after delayed splices = %v, want [merged]", got)
	}

	// Auditability: at Tx=10:15, neither delayed 09:00 draft nor 10:30 review nor 11:00 merged was known!
	if got := g.QueryBitemporal(ctx, "pr#47", "pr.state", testClock(9, 30), testClock(10, 15)).DistinctObjects(); len(got) != 0 {
		t.Fatalf("QueryBitemporal(09:30, tx=10:15) = %v, want empty (delayed draft not yet known)", got)
	}
	if got := g.QueryBitemporal(ctx, "pr#47", "pr.state", testClock(10, 45), testClock(11, 30)).DistinctObjects(); !reflect.DeepEqual(got, []string{"open"}) {
		t.Fatalf("QueryBitemporal(10:45, tx=11:30) = %v, want [open] (delayed review arrived at 12:30)", got)
	}
	if got := g.QueryBitemporal(ctx, "pr#47", "pr.state", testClock(10, 45), testClock(12, 35)).DistinctObjects(); !reflect.DeepEqual(got, []string{"review_requested"}) {
		t.Fatalf("QueryBitemporal(10:45, tx=12:35) = %v, want [review_requested]", got)
	}

	// Confluence proof: ingesting the same 4 valid-time facts strictly in order must yield
	// identical QueryAsOf results at every timestamp.
	ordered := NewGraph()
	orderedFacts := []struct {
		id  string
		obj string
		at  int64
	}{
		{"ev-draft-delayed", "draft", testClock(9, 0)},
		{"ev-open", "open", testClock(10, 0)},
		{"ev-review-delayed", "review_requested", testClock(10, 30)},
		{"ev-merged", "merged", testClock(11, 0)},
	}
	for i, f := range orderedFacts {
		if _, err := ordered.Ingest(EvidenceRecord{
			EvidenceID: f.id, SourceVersion: "v1", TenantID: "hanzo", ACLScope: "repo:graph",
			ObservedAtTxTime: testClock(9+i, 0),
			Fact: FactTriple{
				Subject: "pr#47", Predicate: "pr.state", Object: f.obj,
				Cardinality: FunctionalSingleValue, ValidFrom: f.at,
			},
		}); err != nil {
			t.Fatalf("ordered ingest %s: %v", f.id, err)
		}
	}
	for _, tc := range checks {
		gotOutOfOrder := g.QueryAsOf(ctx, "pr#47", "pr.state", tc.at).DistinctObjects()
		gotOrdered := ordered.QueryAsOf(ctx, "pr#47", "pr.state", tc.at).DistinctObjects()
		if !reflect.DeepEqual(gotOutOfOrder, gotOrdered) {
			t.Fatalf("confluence mismatch at %s: out-of-order=%v vs ordered=%v", FormatTs(tc.at), gotOutOfOrder, gotOrdered)
		}
	}
}

func TestContradictionDetectionAndProvenance(t *testing.T) {
	g := NewGraph()
	ctx := AccessContext{TenantID: "hanzo", AllowedScopes: []string{"ops:prod"}}

	// Two conflicting sources assert different owners at the exact same ValidFrom = 10:00.
	_, err := g.Ingest(EvidenceRecord{
		EvidenceID: "ev-pagerduty-1", SourceVersion: "pd-1", TenantID: "hanzo", ACLScope: "ops:prod",
		ObservedAtTxTime: testClock(10, 1),
		Fact: FactTriple{
			Subject: "svc#billing", Predicate: "service.owner", Object: "alice",
			Cardinality: FunctionalSingleValue, ValidFrom: testClock(10, 0),
		},
	})
	if err != nil {
		t.Fatalf("ingest alice: %v", err)
	}
	_, err = g.Ingest(EvidenceRecord{
		EvidenceID: "ev-catalog-1", SourceVersion: "cat-1", TenantID: "hanzo", ACLScope: "ops:prod",
		ObservedAtTxTime: testClock(10, 2),
		Fact: FactTriple{
			Subject: "svc#billing", Predicate: "service.owner", Object: "bob",
			Cardinality: FunctionalSingleValue, ValidFrom: testClock(10, 0),
		},
	})
	if err != nil {
		t.Fatalf("ingest bob: %v", err)
	}

	cur := g.QueryCurrent(ctx, "svc#billing", "service.owner")
	if !cur.HasContradiction || len(cur.Conflicts) != 1 {
		t.Fatalf("expected 1 explicit ContradictionConflict, got HasContradiction=%v len=%d", cur.HasContradiction, len(cur.Conflicts))
	}
	if got := cur.DistinctObjects(); !reflect.DeepEqual(got, []string{"alice", "bob"}) {
		t.Fatalf("expected both conflicting owners preserved, got %v", got)
	}
	if countDistinctEvidence(cur) != 2 {
		t.Fatalf("expected 2 distinct evidence pointers in conflict, got %d", countDistinctEvidence(cur))
	}

	// Audit at Tx=10:01 (before bob arrived) must show ONLY alice and NO contradiction.
	beforeConflict := g.QueryBitemporal(ctx, "svc#billing", "service.owner", testClock(10, 30), testClock(10, 1))
	if beforeConflict.HasContradiction {
		t.Fatalf("expected no contradiction before second source arrived")
	}
	if got := beforeConflict.DistinctObjects(); !reflect.DeepEqual(got, []string{"alice"}) {
		t.Fatalf("expected [alice] at tx=10:01, got %v", got)
	}

	// Authoritative transition at 11:00 supersedes both conflicting claims for >= 11:00,
	// while keeping the historical contradiction intact for [10:00, 11:00).
	_, err = g.Ingest(EvidenceRecord{
		EvidenceID: "ev-override-1", SourceVersion: "override-1", TenantID: "hanzo", ACLScope: "ops:prod",
		ObservedAtTxTime: testClock(11, 0),
		Fact: FactTriple{
			Subject: "svc#billing", Predicate: "service.owner", Object: "charlie",
			Cardinality: FunctionalSingleValue, ValidFrom: testClock(11, 0),
		},
	})
	if err != nil {
		t.Fatalf("ingest charlie: %v", err)
	}

	afterResolve := g.QueryCurrent(ctx, "svc#billing", "service.owner")
	if afterResolve.HasContradiction {
		t.Fatalf("expected resolved current state at >= 11:00 to have no active conflict")
	}
	if got := afterResolve.DistinctObjects(); !reflect.DeepEqual(got, []string{"charlie"}) {
		t.Fatalf("expected [charlie] after 11:00 transition, got %v", got)
	}

	histConflict := g.QueryAsOf(ctx, "svc#billing", "service.owner", testClock(10, 30))
	if !histConflict.HasContradiction {
		t.Fatalf("expected historical contradiction at 10:30 to remain preserved")
	}
	if got := histConflict.DistinctObjects(); !reflect.DeepEqual(got, []string{"alice", "bob"}) {
		t.Fatalf("expected historical [alice bob] at 10:30, got %v", got)
	}
}

func TestCorroborationMultiValueAndRetraction(t *testing.T) {
	g := NewGraph()
	ctx := AccessContext{TenantID: "hanzo", AllowedScopes: []string{"repo:graph"}}

	// MultiValue labels coexist without superseding each other.
	for i, lbl := range []string{"p0", "bitemporal", "hip-1347"} {
		_, err := g.Ingest(EvidenceRecord{
			EvidenceID: fmt.Sprintf("ev-lbl-%d", i), SourceVersion: "v1",
			TenantID: "hanzo", ACLScope: "repo:graph",
			ObservedAtTxTime: testClock(10, i),
			Fact: FactTriple{
				Subject: "pr#47", Predicate: "has_label", Object: lbl,
				Cardinality: MultiValue, ValidFrom: testClock(10, 0),
			},
		})
		if err != nil {
			t.Fatalf("ingest label %s: %v", lbl, err)
		}
	}

	// Corroborate "p0" from a second source -> merges evidence refs without duplicating edge.
	_, err := g.Ingest(EvidenceRecord{
		EvidenceID: "ev-lbl-0-corrob", SourceVersion: "v2",
		TenantID: "hanzo", ACLScope: "repo:graph",
		ObservedAtTxTime: testClock(10, 10),
		Fact: FactTriple{
			Subject: "pr#47", Predicate: "has_label", Object: "p0",
			Cardinality: MultiValue, ValidFrom: testClock(10, 0),
		},
	})
	if err != nil {
		t.Fatalf("corroborate label: %v", err)
	}

	cur := g.QueryCurrent(ctx, "pr#47", "has_label")
	if got := cur.DistinctObjects(); !reflect.DeepEqual(got, []string{"bitemporal", "hip-1347", "p0"}) {
		t.Fatalf("multi-value labels = %v, want [bitemporal hip-1347 p0]", got)
	}
	if len(cur.Edges) != 3 {
		t.Fatalf("expected 3 deduplicated active edges, got %d", len(cur.Edges))
	}
	for _, e := range cur.Edges {
		if e.Object == "p0" && len(e.Evidence) != 2 {
			t.Fatalf("expected corroborated p0 edge to carry 2 evidence refs, got %d", len(e.Evidence))
		}
	}

	// Non-destructive retraction of "hip-1347" at 10:30 closes ValidTo=10:30 without deleting history.
	_, err = g.Ingest(EvidenceRecord{
		EvidenceID: "ev-lbl-retract", SourceVersion: "v3",
		TenantID: "hanzo", ACLScope: "repo:graph",
		ObservedAtTxTime:   testClock(10, 30),
		RetractsEvidenceID: "ev-lbl-2",
		Fact: FactTriple{
			Subject: "pr#47", Predicate: "has_label",
			Cardinality: MultiValue, ValidFrom: testClock(10, 30),
		},
	})
	if err != nil {
		t.Fatalf("retract label: %v", err)
	}
	if got := g.QueryCurrent(ctx, "pr#47", "has_label").DistinctObjects(); !reflect.DeepEqual(got, []string{"bitemporal", "p0"}) {
		t.Fatalf("after retraction QueryCurrent = %v, want [bitemporal p0]", got)
	}
	if got := g.QueryAsOf(ctx, "pr#47", "has_label", testClock(10, 15)).DistinctObjects(); !reflect.DeepEqual(got, []string{"bitemporal", "hip-1347", "p0"}) {
		t.Fatalf("historical QueryAsOf(10:15) = %v, want all 3 labels", got)
	}
}

func TestRebuildFromLogBitIdentical(t *testing.T) {
	records, queries := BuildDeterministicBenchmarkSuite()
	g, err := NewFromEvidenceLog(records)
	if err != nil {
		t.Fatalf("NewFromEvidenceLog failed: %v", err)
	}

	beforeEdges := g.ExportDerivedEdges()
	beforeConflicts := g.ExportDerivedConflicts()
	beforeStats := g.Stats()
	beforeResults := make([]QueryResult, len(queries))
	for i, q := range queries {
		beforeResults[i] = execBackendQuery(&bitemporalGraphitiBackend{graph: g}, q)
	}

	// Wipe derived index completely and rebuild from immutable log.
	g.RebuildFromLog()

	afterEdges := g.ExportDerivedEdges()
	afterConflicts := g.ExportDerivedConflicts()
	afterStats := g.Stats()
	afterResults := make([]QueryResult, len(queries))
	for i, q := range queries {
		afterResults[i] = execBackendQuery(&bitemporalGraphitiBackend{graph: g}, q)
	}

	if !reflect.DeepEqual(beforeEdges, afterEdges) {
		t.Fatalf("derived edges diverged after RebuildFromLog")
	}
	if !reflect.DeepEqual(beforeConflicts, afterConflicts) {
		t.Fatalf("derived conflicts diverged after RebuildFromLog")
	}
	if beforeStats != afterStats {
		t.Fatalf("stats diverged after RebuildFromLog: before=%+v after=%+v", beforeStats, afterStats)
	}
	if !reflect.DeepEqual(beforeResults, afterResults) {
		t.Fatalf("query results diverged after RebuildFromLog")
	}

	// Also rebuild into a completely fresh Graph instance from exported log.
	replayed, err := NewFromEvidenceLog(g.ExportEvidenceLog())
	if err != nil {
		t.Fatalf("replay NewFromEvidenceLog failed: %v", err)
	}
	if !reflect.DeepEqual(beforeEdges, replayed.ExportDerivedEdges()) {
		t.Fatalf("fresh replayed graph edges are not bit-identical")
	}
	if !reflect.DeepEqual(beforeConflicts, replayed.ExportDerivedConflicts()) {
		t.Fatalf("fresh replayed graph conflicts are not bit-identical")
	}
}

func TestPreReturnMultiTenantACLIsolation(t *testing.T) {
	g := NewGraph()

	// Seed conflicting facts in tenant-b and restricted scope in tenant-a.
	seed := []EvidenceRecord{
		{
			EvidenceID: "ev-b-1", SourceVersion: "v1", TenantID: "tenant-b", ACLScope: "repo:public",
			ObservedAtTxTime: testClock(10, 0),
			Fact: FactTriple{
				Subject: "secret#1", Predicate: "owner", Object: "bob",
				Cardinality: FunctionalSingleValue, ValidFrom: testClock(10, 0),
			},
		},
		{
			EvidenceID: "ev-b-2", SourceVersion: "v2", TenantID: "tenant-b", ACLScope: "repo:public",
			ObservedAtTxTime: testClock(10, 1),
			Fact: FactTriple{
				Subject: "secret#1", Predicate: "owner", Object: "eve",
				Cardinality: FunctionalSingleValue, ValidFrom: testClock(10, 0),
			},
		},
		{
			EvidenceID: "ev-a-sec", SourceVersion: "v1", TenantID: "tenant-a", ACLScope: "repo:secret",
			ObservedAtTxTime: testClock(10, 2),
			Fact: FactTriple{
				Subject: "internal#1", Predicate: "state", Object: "classified",
				Cardinality: FunctionalSingleValue, ValidFrom: testClock(10, 0),
			},
		},
		{
			EvidenceID: "ev-a-pub", SourceVersion: "v1", TenantID: "tenant-a", ACLScope: "repo:public",
			ObservedAtTxTime: testClock(10, 3),
			Fact: FactTriple{
				Subject: "public#1", Predicate: "state", Object: "open",
				Cardinality: FunctionalSingleValue, ValidFrom: testClock(10, 0),
			},
		},
	}
	if _, err := g.IngestBatch(seed); err != nil {
		t.Fatalf("seed batch: %v", err)
	}

	// Fail-closed: empty AccessContext sees nothing.
	if r := g.QueryCurrent(AccessContext{}, "", ""); len(r.Edges) != 0 || len(r.Conflicts) != 0 || r.InspectedTenantEdges != 0 {
		t.Fatalf("empty AccessContext leaked data: %+v", r)
	}

	// Tenant A with wildcard "*" can see tenant-a scopes ONLY, NEVER tenant-b!
	tenantAAdmin := AccessContext{TenantID: "tenant-a", AllowedScopes: []string{"*"}}
	resA := g.QueryCurrent(tenantAAdmin, "", "")
	if len(resA.Edges) != 2 || resA.HasContradiction {
		t.Fatalf("tenant-a admin expected 2 edges & 0 conflicts, got edges=%d conflicts=%d", len(resA.Edges), len(resA.Conflicts))
	}
	for _, e := range resA.Edges {
		if e.TenantID != "tenant-a" {
			t.Fatalf("cross-tenant leak into tenant-a admin: %+v", e)
		}
	}

	// Tenant A querying tenant-b subject explicitly returns 0 edges, 0 conflicts, 0 inspected.
	cross := g.QueryCurrent(tenantAAdmin, "secret#1", "owner")
	if len(cross.Edges) != 0 || len(cross.Conflicts) != 0 || cross.InspectedTenantEdges != 0 {
		t.Fatalf("cross-tenant query leaked info: %+v", cross)
	}

	// Tenant A with only repo:public cannot read tenant-a repo:secret.
	tenantAPublic := AccessContext{TenantID: "tenant-a", AllowedScopes: []string{"repo:public"}}
	denied := g.QueryCurrent(tenantAPublic, "internal#1", "state")
	if len(denied.Edges) != 0 || denied.InspectedTenantEdges != 0 {
		t.Fatalf("restricted-scope query leaked info: %+v", denied)
	}
}

func TestConcurrentMultiTenantIsolationUnderRace(t *testing.T) {
	g := NewGraph()
	const (
		numTenants    = 4
		numGoroutines = 16
		opsPerWorker  = 40
	)
	base := time.Date(2026, 5, 1, 9, 0, 0, 0, time.UTC)

	var wg sync.WaitGroup
	errCh := make(chan error, numGoroutines)

	for w := 0; w < numGoroutines; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			tenantID := fmt.Sprintf("tenant-%d", workerID%numTenants)
			scope := fmt.Sprintf("scope-%d", workerID%2)
			ctx := AccessContext{TenantID: tenantID, AllowedScopes: []string{scope}}

			for op := 0; op < opsPerWorker; op++ {
				subj := fmt.Sprintf("item-%s-%d", tenantID, op%5)
				vAt := Ts(base.Add(time.Duration(op+1) * time.Minute))
				_, err := g.Ingest(EvidenceRecord{
					EvidenceID:    fmt.Sprintf("ev-w%02d-op%03d", workerID, op),
					SourceVersion: "v1",
					TenantID:      tenantID,
					ACLScope:      scope,
					Fact: FactTriple{
						Subject:     subj,
						Predicate:   "item.state",
						Object:      fmt.Sprintf("state-%s-%d", tenantID, op),
						Cardinality: FunctionalSingleValue,
						ValidFrom:   vAt,
					},
				})
				if err != nil {
					errCh <- fmt.Errorf("worker %d op %d ingest: %w", workerID, op, err)
					return
				}

				// Interleave queries and occasional full rebuilds to stress synchronization.
				cur := g.QueryCurrent(ctx, "", "")
				for _, e := range cur.Edges {
					if e.TenantID != tenantID || e.ACLScope != scope {
						errCh <- fmt.Errorf("ACL breach in worker %d: got (%s,%s) want (%s,%s)",
							workerID, e.TenantID, e.ACLScope, tenantID, scope)
						return
					}
				}
				asOf := g.QueryAsOf(ctx, subj, "item.state", vAt)
				for _, e := range asOf.Edges {
					if e.TenantID != tenantID || e.ACLScope != scope {
						errCh <- fmt.Errorf("AsOf ACL breach in worker %d: got (%s,%s)", workerID, e.TenantID, e.ACLScope)
						return
					}
				}
				bt := g.QueryBitemporal(ctx, subj, "item.state", vAt, Ts(time.Now()))
				for _, e := range bt.Edges {
					if e.TenantID != tenantID || e.ACLScope != scope {
						errCh <- fmt.Errorf("Bitemporal ACL breach in worker %d: got (%s,%s)", workerID, e.TenantID, e.ACLScope)
						return
					}
				}

				if workerID%4 == 0 && op%15 == 0 {
					g.RebuildFromLog()
				}
			}
		}(w)
	}

	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Fatal(err)
	}

	if got := len(g.ExportEvidenceLog()); got != numGoroutines*opsPerWorker {
		t.Fatalf("expected %d evidence records after concurrent ingest, got %d", numGoroutines*opsPerWorker, got)
	}
}

func TestValidationAndAtomicBatchRollback(t *testing.T) {
	g := NewGraph()
	validRec := EvidenceRecord{
		EvidenceID: "ev-ok-1", SourceVersion: "v1", TenantID: "t1", ACLScope: "s1",
		ObservedAtTxTime: testClock(10, 0),
		Fact: FactTriple{
			Subject: "pr#1", Predicate: "pr.state", Object: "open",
			Cardinality: FunctionalSingleValue, ValidFrom: testClock(10, 0),
		},
	}
	if _, err := g.Ingest(validRec); err != nil {
		t.Fatalf("valid ingest failed: %v", err)
	}

	// Non-monotonic tx time must fail.
	badTx := validRec
	badTx.EvidenceID = "ev-bad-tx"
	badTx.ObservedAtTxTime = testClock(9, 59)
	if _, err := g.Ingest(badTx); !errors.Is(err, ErrNonMonotonicTxTime) {
		t.Fatalf("expected ErrNonMonotonicTxTime, got %v", err)
	}

	// Batch containing a valid record followed by an invalid record must roll back atomically.
	batchRec1 := validRec
	batchRec1.EvidenceID = "ev-batch-1"
	batchRec1.ObservedAtTxTime = testClock(10, 5)
	batchRec1.Fact.Object = "merged"
	batchRec1.Fact.ValidFrom = testClock(10, 5)

	batchRec2 := validRec
	batchRec2.EvidenceID = "" // invalid!
	batchRec2.ObservedAtTxTime = testClock(10, 6)

	if _, err := g.IngestBatch([]EvidenceRecord{batchRec1, batchRec2}); !errors.Is(err, ErrEmptyEvidenceID) {
		t.Fatalf("expected ErrEmptyEvidenceID from invalid batch, got %v", err)
	}
	if len(g.ExportEvidenceLog()) != 1 {
		t.Fatalf("batch rollback failed: expected 1 record in log, got %d", len(g.ExportEvidenceLog()))
	}
	cur := g.QueryCurrent(AccessContext{TenantID: "t1", AllowedScopes: []string{"s1"}}, "pr#1", "pr.state")
	if got := cur.DistinctObjects(); !reflect.DeepEqual(got, []string{"open"}) {
		t.Fatalf("batch rollback left dirty derived state: %v", got)
	}
}

func TestComparativeBenchmarkAssertions(t *testing.T) {
	report, err := RunComparativeBenchmark(5)
	if err != nil {
		t.Fatalf("RunComparativeBenchmark failed: %v", err)
	}

	if report.TotalQueries != 120 {
		t.Fatalf("expected 120 queries, got %d", report.TotalQueries)
	}
	if !report.RebuildVerified {
		t.Fatalf("expected bit-identical RebuildFromLog verification to pass")
	}

	byName := make(map[string]BackendBenchmarkResult)
	for _, b := range report.Backends {
		byName[b.BackendName] = b
		if b.CrossTenantLeaks != 0 {
			t.Fatalf("backend %s had %d cross-tenant leaks", b.BackendName, b.CrossTenantLeaks)
		}
	}

	bt := byName["bitemporal_graphiti_provenance"]
	if bt.OverallAccuracy != 1.0 {
		t.Fatalf("bitemporal overall accuracy = %.4f, want 1.0", bt.OverallAccuracy)
	}
	if bt.StaleAnswerFrequency != 0.0 {
		t.Fatalf("bitemporal stale-answer frequency = %.4f, want 0.0", bt.StaleAnswerFrequency)
	}
	if bt.DownstreamAgentTaskSuccess != 1.0 {
		t.Fatalf("bitemporal agent task success = %.4f, want 1.0", bt.DownstreamAgentTaskSuccess)
	}
	if !bt.RebuildBitIdentical {
		t.Fatalf("bitemporal must verify bit-identical rebuild")
	}

	ow := byName["overwrite_mutable_graph"]
	if ow.HistoricalAsOfAccuracy >= 0.5 || ow.BitemporalAuditAsAtAccuracy >= 0.5 {
		t.Fatalf("overwrite baseline unexpectedly passed temporal history: as-of=%.2f audit=%.2f",
			ow.HistoricalAsOfAccuracy, ow.BitemporalAuditAsAtAccuracy)
	}

	ao := byName["append_only_no_supersession"]
	if ao.StaleAnswerFrequency < 0.5 {
		t.Fatalf("append-only baseline should exhibit high stale-answer frequency, got %.4f", ao.StaleAnswerFrequency)
	}
	if ao.EstimatedTokensPerAnswer <= bt.EstimatedTokensPerAnswer {
		t.Fatalf("append-only token cost (%.2f) should exceed bitemporal (%.2f) due to stale edges",
			ao.EstimatedTokensPerAnswer, bt.EstimatedTokensPerAnswer)
	}

	vt := byName["valid_time_only_unitemporal"]
	if vt.HistoricalAsOfAccuracy != 1.0 || vt.DelayedEventConvergenceAccuracy != 1.0 {
		t.Fatalf("unitemporal baseline should pass as-of & delayed valid-time queries, got as-of=%.2f delayed=%.2f",
			vt.HistoricalAsOfAccuracy, vt.DelayedEventConvergenceAccuracy)
	}
	if vt.BitemporalAuditAsAtAccuracy != 0.0 {
		t.Fatalf("unitemporal baseline should fail bitemporal audit queries, got %.2f", vt.BitemporalAuditAsAtAccuracy)
	}

	// Verify all 3 documented negative & boundary regimes.
	st := report.NegativeRegimes.StaticImmutableKnowledgeOverhead
	if st.AccuracyGain != 0.0 || st.ExtraBytesPerEdge <= 0 {
		t.Fatalf("unexpected static overhead regime metrics: %+v", st)
	}
	sim := report.NegativeRegimes.UnresolvableSimultaneousConflict
	if sim.AutoResolvedWithoutPolicy != 0 || sim.ConflictSurfacingRate != 1.0 {
		t.Fatalf("unexpected simultaneous contradiction regime metrics: %+v", sim)
	}
	acl := report.NegativeRegimes.CrossTenantACLDenial
	if acl.ReturnedHits != 0 || acl.LeakedEdges != 0 || acl.LeakedConflicts != 0 || acl.InspectedForeignEdges != 0 {
		t.Fatalf("unexpected cross-tenant ACL denial metrics: %+v", acl)
	}
}
