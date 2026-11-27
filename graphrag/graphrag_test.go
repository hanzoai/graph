package graphrag

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"
)

func buildTestMultiClusterDocs() []CanonicalDocument {
	ts := time.Date(2026, 11, 10, 0, 0, 0, 0, time.UTC)
	return []CanonicalDocument{
		// Cluster A: MVCC & Posting
		{
			SourceID: "cluster-a/mvcc-1.md", Version: "v1", TenantID: "t1", ACL: []string{"*"},
			Kind: "regression_report", Title: "REG-101 MVCC posting rollup race",
			Content:   "MVCC posting list rollup race in posting/mvcc.go causes stale ReadTs snapshot reads.",
			Metadata:  map[string]string{"package": "posting", "root_cause": "mvcc_rollup_race", "regression_class": "mvcc_and_posting_consistency"},
			UpdatedAt: ts,
		},
		{
			SourceID: "cluster-a/mvcc-2.md", Version: "v1", TenantID: "t1", ACL: []string{"*"},
			Kind: "regression_report", Title: "REG-102 Oracle watermark stall",
			Content:   "Oracle transaction watermark stall in posting/oracle.go delays posting commit deltas.",
			Metadata:  map[string]string{"package": "posting", "root_cause": "oracle_watermark_stall", "regression_class": "mvcc_and_posting_consistency"},
			UpdatedAt: ts,
		},
		// Cluster B: DQL & Lexer
		{
			SourceID: "cluster-b/dql-1.md", Version: "v1", TenantID: "t1", ACL: []string{"*"},
			Kind: "regression_report", Title: "REG-201 DQL parser recursion",
			Content:   "DQL parser recursion in dql/parser.go overflows stack on nested filter directives.",
			Metadata:  map[string]string{"package": "dql", "root_cause": "dql_parser_recursion", "regression_class": "query_parsing_and_dql"},
			UpdatedAt: ts,
		},
		{
			SourceID: "cluster-b/dql-2.md", Version: "v1", TenantID: "t1", ACL: []string{"*"},
			Kind: "regression_report", Title: "REG-202 Lexer IRI escape drift",
			Content:   "Lexer IRI escape bug in lex/iri.go corrupts DQL predicate parsing.",
			Metadata:  map[string]string{"package": "lex", "root_cause": "lexer_iri_drift", "regression_class": "query_parsing_and_dql"},
			UpdatedAt: ts,
		},
		// Cluster C: Codec & Memory
		{
			SourceID: "cluster-c/codec-1.md", Version: "v1", TenantID: "t1", ACL: []string{"*"},
			Kind: "regression_report", Title: "REG-301 Codec buffer over-allocation",
			Content:   "Codec pack slice over-allocation in codec/codec.go causes heap OOM spikes.",
			Metadata:  map[string]string{"package": "codec", "root_cause": "codec_oom_alloc", "regression_class": "memory_allocation_and_codec"},
			UpdatedAt: ts,
		},
		{
			SourceID: "cluster-c/codec-2.md", Version: "v1", TenantID: "t1", ACL: []string{"*"},
			Kind: "regression_report", Title: "REG-302 UIDList temporary buffer churn",
			Content:   "UIDList intersection in algo/uidlist.go triggers quadratic heap allocation churn.",
			Metadata:  map[string]string{"package": "algo", "root_cause": "uidlist_buffer_churn", "regression_class": "memory_allocation_and_codec"},
			UpdatedAt: ts,
		},
		// Cluster D: HNSW & Indexing
		{
			SourceID: "cluster-d/hnsw-1.md", Version: "v1", TenantID: "t1", ACL: []string{"*"},
			Kind: "regression_report", Title: "REG-401 HNSW non-atomic flush",
			Content:   "HNSW persistent graph write in tok/hnsw/persistent_hnsw.go drops nearest-neighbor edges.",
			Metadata:  map[string]string{"package": "tok/hnsw", "root_cause": "hnsw_flush_corruption", "regression_class": "indexing_and_vector_hnsw"},
			UpdatedAt: ts,
		},
		{
			SourceID: "cluster-d/hnsw-2.md", Version: "v1", TenantID: "t1", ACL: []string{"*"},
			Kind: "regression_report", Title: "REG-402 Bleve stemmer cache drift",
			Content:   "Bleve tokenizer stemmer drift in tok/bleve.go drops fulltext index matches.",
			Metadata:  map[string]string{"package": "tok", "root_cause": "bleve_stemmer_drift", "regression_class": "indexing_and_vector_hnsw"},
			UpdatedAt: ts,
		},
	}
}

func TestDropAndRebuildBitExactParity(t *testing.T) {
	eng := NewEngine(DefaultConfig())
	docs := buildTestMultiClusterDocs()
	for _, d := range docs {
		if _, err := eng.UpsertDocument(d); err != nil {
			t.Fatalf("upsert %s: %v", d.SourceID, err)
		}
	}

	b1 := eng.BuildAll()
	if b1.DocumentsIndexed != len(docs) {
		t.Fatalf("expected %d docs indexed, got %d", len(docs), b1.DocumentsIndexed)
	}
	d1, err := eng.DerivedDigest()
	if err != nil {
		t.Fatalf("DerivedDigest failed: %v", err)
	}

	eng.DropDerivedIndexes()

	// Canonical truth MUST remain intact after dropping derived indexes.
	if got := eng.CanonicalCount("t1"); got != len(docs) {
		t.Fatalf("canonical count lost after DropDerivedIndexes: got %d want %d", got, len(docs))
	}
	caller := CallerContext{TenantID: "t1", Principals: []string{"role:eng"}}
	if _, err := eng.Query(context.Background(), caller, QueryRequest{Query: "regressions across all packages"}); !errors.Is(err, ErrDerivedIndexNotBuilt) {
		t.Fatalf("expected ErrDerivedIndexNotBuilt after drop, got %v", err)
	}
	if _, err := eng.ListCommunities(caller, -1); !errors.Is(err, ErrDerivedIndexNotBuilt) {
		t.Fatalf("expected ErrDerivedIndexNotBuilt from ListCommunities, got %v", err)
	}
	if _, err := eng.DerivedDigest(); !errors.Is(err, ErrDerivedIndexNotBuilt) {
		t.Fatalf("expected ErrDerivedIndexNotBuilt from DerivedDigest, got %v", err)
	}

	// Rebuild twice and verify bit-exact SHA-256 digest parity.
	eng.RebuildAll()
	d2, err := eng.DerivedDigest()
	if err != nil {
		t.Fatalf("DerivedDigest after rebuild 1: %v", err)
	}
	eng.DropDerivedIndexes()
	eng.RebuildAll()
	d3, err := eng.DerivedDigest()
	if err != nil {
		t.Fatalf("DerivedDigest after rebuild 2: %v", err)
	}

	if d1 != d2 || d2 != d3 {
		t.Fatalf("drop/rebuild digest mismatch: d1=%s d2=%s d3=%s", d1, d2, d3)
	}
}

func TestMultiLevelCommunityHierarchyAndConnectivity(t *testing.T) {
	eng := NewEngine(DefaultConfig())
	for _, d := range buildTestMultiClusterDocs() {
		if _, err := eng.UpsertDocument(d); err != nil {
			t.Fatalf("upsert: %v", err)
		}
	}
	stats := eng.BuildAll()
	if stats.CommunitiesByLevel[0] < 4 {
		t.Fatalf("expected at least 4 L0 communities across 4 disjoint clusters, got %d", stats.CommunitiesByLevel[0])
	}
	if stats.CommunitiesByLevel[1] == 0 || stats.CommunitiesByLevel[2] == 0 {
		t.Fatalf("expected non-empty L1 and L2 communities, got L1=%d L2=%d", stats.CommunitiesByLevel[1], stats.CommunitiesByLevel[2])
	}
	if stats.CommunitiesByLevel[0] < stats.CommunitiesByLevel[1] || stats.CommunitiesByLevel[1] < stats.CommunitiesByLevel[2] {
		t.Fatalf("hierarchy coarsening invariant violated: L0=%d L1=%d L2=%d",
			stats.CommunitiesByLevel[0], stats.CommunitiesByLevel[1], stats.CommunitiesByLevel[2])
	}
	if stats.MeanModularityL0 <= 0.3 {
		t.Fatalf("expected strong modularity Q_L0 > 0.3 on 4 disjoint clusters, got %.4f", stats.MeanModularityL0)
	}

	caller := CallerContext{TenantID: "t1", Principals: []string{"role:eng"}}
	allComms, err := eng.ListCommunities(caller, -1)
	if err != nil {
		t.Fatalf("ListCommunities: %v", err)
	}
	commByID := make(map[string]Community, len(allComms))
	for _, c := range allComms {
		commByID[c.ID] = c
	}

	for _, c := range allComms {
		if len(c.EntityIDs) == 0 {
			t.Fatalf("community %s has zero entities", c.ID)
		}
		if c.Level == 0 || c.Level == 1 {
			if c.ParentID == "" {
				t.Fatalf("L%d community %s missing ParentID", c.Level, c.ID)
			}
			parent, ok := commByID[c.ParentID]
			if !ok || parent.Level != c.Level+1 {
				t.Fatalf("L%d community %s has invalid parent %s", c.Level, c.ID, c.ParentID)
			}
		}
		if c.Level > 0 && len(c.ChildIDs) == 0 {
			t.Fatalf("L%d community %s has no children", c.Level, c.ID)
		}
		sum, err := eng.GetCommunitySummary(caller, c.ID)
		if err != nil {
			t.Fatalf("GetCommunitySummary(%s): %v", c.ID, err)
		}
		if sum.Summary == "" || len(sum.SourceRefs) == 0 {
			t.Fatalf("community %s has empty summary or missing provenance refs", c.ID)
		}
	}

	// Verify Leiden connectivity guarantee: every L0 community induces a connected subgraph.
	eng.mu.RLock()
	for _, tDerived := range eng.derived {
		for _, p := range tDerived {
			for _, l0ID := range p.byLevel[0] {
				c := p.communities[l0ID]
				if !isConnectedSubgraph(c.EntityIDs, p.relations) {
					eng.mu.RUnlock()
					t.Fatalf("Leiden connectivity violated: L0 community %s is disconnected", l0ID)
				}
			}
		}
	}
	eng.mu.RUnlock()
}

func isConnectedSubgraph(entityIDs []string, relations map[string]*Relation) bool {
	if len(entityIDs) <= 1 {
		return true
	}
	memberSet := make(map[string]struct{}, len(entityIDs))
	adj := make(map[string][]string, len(entityIDs))
	for _, id := range entityIDs {
		memberSet[id] = struct{}{}
	}
	for _, r := range relations {
		if _, ok1 := memberSet[r.SourceEntityID]; !ok1 {
			continue
		}
		if _, ok2 := memberSet[r.TargetEntityID]; !ok2 {
			continue
		}
		adj[r.SourceEntityID] = append(adj[r.SourceEntityID], r.TargetEntityID)
		adj[r.TargetEntityID] = append(adj[r.TargetEntityID], r.SourceEntityID)
	}
	visited := map[string]struct{}{entityIDs[0]: {}}
	queue := []string{entityIDs[0]}
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		for _, nb := range adj[cur] {
			if _, seen := visited[nb]; !seen {
				visited[nb] = struct{}{}
				queue = append(queue, nb)
			}
		}
	}
	return len(visited) == len(entityIDs)
}

func TestIncrementalUpdateInvalidatesOnlyAffectedAncestors(t *testing.T) {
	eng := NewEngine(DefaultConfig())
	for _, d := range buildTestMultiClusterDocs() {
		if _, err := eng.UpsertDocument(d); err != nil {
			t.Fatalf("upsert: %v", err)
		}
	}
	eng.BuildAll()

	caller := CallerContext{TenantID: "t1", Principals: []string{"role:eng"}}
	beforeComms, err := eng.ListCommunities(caller, -1)
	if err != nil {
		t.Fatalf("ListCommunities: %v", err)
	}
	beforeRev := make(map[string]uint64, len(beforeComms))
	for _, c := range beforeComms {
		sum, err := eng.GetCommunitySummary(caller, c.ID)
		if err != nil {
			t.Fatalf("GetCommunitySummary: %v", err)
		}
		beforeRev[c.ID] = sum.Revision
	}

	// Update ONLY one document in Cluster A (posting).
	updated := CanonicalDocument{
		SourceID:  "cluster-a/mvcc-1.md",
		Version:   "v2",
		TenantID:  "t1",
		ACL:       []string{"*"},
		Kind:      "regression_report",
		Title:     "REG-101 MVCC posting rollup race (FIXED_TOKEN_ALPHA_77)",
		Content:   "MVCC posting list rollup race in posting/mvcc.go mitigated via FIXED_TOKEN_ALPHA_77 barrier.",
		Metadata:  map[string]string{"package": "posting", "root_cause": "mvcc_rollup_race", "regression_class": "mvcc_and_posting_consistency"},
		UpdatedAt: time.Now().UTC(),
	}
	incStats, err := eng.UpsertDocument(updated)
	if err != nil {
		t.Fatalf("incremental UpsertDocument: %v", err)
	}

	if incStats.Operation != "incremental_update" {
		t.Fatalf("expected operation incremental_update, got %s", incStats.Operation)
	}
	if incStats.CommunitiesReused == 0 {
		t.Fatalf("expected untouched communities to be reused, got CommunitiesReused=0 (resummarized=%d)", incStats.CommunitiesResummarized)
	}
	if incStats.DirtyL0 >= incStats.TotalL0 {
		t.Fatalf("expected targeted L0 invalidation (dirtyL0 < totalL0), got %d/%d", incStats.DirtyL0, incStats.TotalL0)
	}

	// Verify untouched communities preserved exact Revision == 1 while dirty branch incremented.
	incrementedCount := 0
	unchangedCount := 0
	afterComms, _ := eng.ListCommunities(caller, -1)
	for _, c := range afterComms {
		sum, err := eng.GetCommunitySummary(caller, c.ID)
		if err != nil {
			t.Fatalf("GetCommunitySummary after update: %v", err)
		}
		if prev, ok := beforeRev[c.ID]; ok {
			if sum.Revision > prev {
				incrementedCount++
			} else {
				unchangedCount++
			}
		}
	}
	if incrementedCount == 0 || unchangedCount == 0 {
		t.Fatalf("expected partial ancestor refresh: incremented=%d unchanged=%d", incrementedCount, unchangedCount)
	}

	// Verify updated token is immediately visible in global query output.
	resp, err := eng.Query(context.Background(), caller, QueryRequest{
		Query: "Summarize MVCC posting rollup regressions across all packages",
		Mode:  ModeGlobalL1,
	})
	if err != nil {
		t.Fatalf("Query after incremental update: %v", err)
	}
	if !strings.Contains(resp.Answer, "FIXED_TOKEN_ALPHA_77") {
		t.Fatalf("expected updated token FIXED_TOKEN_ALPHA_77 in refreshed L1 summary answer, got:\n%s", resp.Answer)
	}
}

func TestIncrementalDeleteAndEmptyPruning(t *testing.T) {
	eng := NewEngine(DefaultConfig())
	for _, d := range buildTestMultiClusterDocs() {
		if _, err := eng.UpsertDocument(d); err != nil {
			t.Fatalf("upsert: %v", err)
		}
	}
	eng.BuildAll()

	// Delete both documents from Cluster D (HNSW).
	st1, err := eng.DeleteDocument("t1", "cluster-d/hnsw-1.md")
	if err != nil {
		t.Fatalf("delete hnsw-1: %v", err)
	}
	if st1.CommunitiesReused == 0 {
		t.Fatalf("expected untouched communities reused on delete")
	}
	if _, err := eng.DeleteDocument("t1", "cluster-d/hnsw-2.md"); err != nil {
		t.Fatalf("delete hnsw-2: %v", err)
	}

	caller := CallerContext{TenantID: "t1", Principals: []string{"role:eng"}}
	for _, mode := range []QueryMode{ModeGlobalL0, ModeGlobalL1, ModeGlobalL2, ModeLocalHybrid} {
		resp, err := eng.Query(context.Background(), caller, QueryRequest{
			Query: "Summarize HNSW and Bleve regressions across all packages",
			Mode:  mode,
		})
		if err != nil {
			t.Fatalf("query mode %s: %v", mode, err)
		}
		for _, cit := range resp.Citations {
			if strings.HasPrefix(cit.SourceID, "cluster-d/") {
				t.Fatalf("deleted document %s leaked in citations for mode %s", cit.SourceID, mode)
			}
		}
	}
}

func TestACLIsolationNoSummaryLeakAndZeroExistenceOracle(t *testing.T) {
	eng := NewEngine(DefaultConfig())
	docs := []CanonicalDocument{
		{
			SourceID: "t1/public.md", Version: "v1", TenantID: "tenant-a", ACL: []string{"*"},
			Kind: "regression_report", Title: "Public posting regression",
			Content:  "Public posting/mvcc.go rollup regression visible to all tenant-a callers.",
			Metadata: map[string]string{"package": "posting", "regression_class": "mvcc_and_posting_consistency"},
		},
		{
			SourceID: "t1/secret.md", Version: "v1", TenantID: "tenant-a", ACL: []string{"role:sec-admin"},
			Kind: "regression_report", Title: "Secret vault zero-day SECRET_CANARY_9001",
			Content:  "Restricted SECRET_CANARY_9001 affecting x/vault.go and posting/mvcc.go.",
			Metadata: map[string]string{"package": "x", "root_cause": "secret_canary_9001", "regression_class": "security_acl_and_encryption"},
		},
		{
			SourceID: "t1/deny-all.md", Version: "v1", TenantID: "tenant-a", ACL: nil, // deny-by-default
			Kind: "regression_report", Title: "Deny-all record DENY_CANARY_9002",
			Content: "DENY_CANARY_9002 must never be returned to any caller.",
		},
		{
			SourceID: "t2/beta.md", Version: "v1", TenantID: "tenant-b", ACL: []string{"*"},
			Kind: "regression_report", Title: "Tenant B BETA_CANARY_9003",
			Content: "Tenant B wildcard document BETA_CANARY_9003 must never leak to tenant-a.",
		},
	}
	for _, d := range docs {
		if _, err := eng.UpsertDocument(d); err != nil {
			t.Fatalf("upsert %s: %v", d.SourceID, err)
		}
	}
	eng.BuildAll()

	engCaller := CallerContext{TenantID: "tenant-a", Principals: []string{"role:eng"}}
	secCaller := CallerContext{TenantID: "tenant-a", Principals: []string{"role:eng", "role:sec-admin"}}
	betaCaller := CallerContext{TenantID: "tenant-b", Principals: []string{"role:eng"}}

	forbiddenForEng := []string{"SECRET_CANARY_9001", "DENY_CANARY_9002", "BETA_CANARY_9003"}

	// 1. No restricted or cross-tenant canary in any query mode for role:eng.
	for _, mode := range []QueryMode{ModeGlobalL0, ModeGlobalL1, ModeGlobalL2, ModeLocalHybrid, ModeAuto} {
		resp, err := eng.Query(context.Background(), engCaller, QueryRequest{
			Query: "Summarize all regressions across our repositories including SECRET_CANARY_9001 DENY_CANARY_9002 BETA_CANARY_9003",
			Mode:  mode,
		})
		if err != nil {
			t.Fatalf("query mode %s: %v", mode, err)
		}
		if containsAnyCanary(resp, forbiddenForEng) {
			t.Fatalf("ACL/tenant leak detected for role:eng in mode %s: %s", mode, resp.Answer)
		}
		for _, c := range resp.Citations {
			if c.TenantID != "tenant-a" || c.SourceID == "t1/secret.md" || c.SourceID == "t1/deny-all.md" {
				t.Fatalf("unauthorized citation %+v returned to role:eng in mode %s", c, mode)
			}
		}
	}

	// 2. Security admin DOES see SECRET_CANARY_9001, but NEVER DENY_CANARY_9002 or BETA_CANARY_9003.
	secResp, err := eng.Query(context.Background(), secCaller, QueryRequest{
		Query: "Summarize all security regressions across our repositories including SECRET_CANARY_9001",
		Mode:  ModeGlobalL1,
	})
	if err != nil {
		t.Fatalf("secCaller query: %v", err)
	}
	if !strings.Contains(secResp.Answer, "SECRET_CANARY_9001") {
		t.Fatalf("expected authorized secCaller to see SECRET_CANARY_9001")
	}
	if containsAnyCanary(secResp, []string{"DENY_CANARY_9002", "BETA_CANARY_9003"}) {
		t.Fatalf("deny-all or cross-tenant canary leaked to secCaller")
	}

	// 3. Zero existence-oracle leakage: unauthorized GetDocument and GetCommunitySummary MUST return ErrNotFound.
	if _, err := eng.GetDocument(engCaller, "t1/secret.md"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("expected ErrNotFound for unauthorized GetDocument, got %v", err)
	}
	if _, err := eng.GetDocument(engCaller, "t2/beta.md"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("expected ErrNotFound for foreign tenant GetDocument, got %v", err)
	}
	if _, err := eng.GetDocument(secCaller, "t1/deny-all.md"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("expected ErrNotFound for empty-ACL deny-by-default doc, got %v", err)
	}

	secComms, _ := eng.ListCommunities(secCaller, -1)
	engComms, _ := eng.ListCommunities(engCaller, -1)
	engSet := make(map[string]struct{}, len(engComms))
	for _, c := range engComms {
		engSet[c.ID] = struct{}{}
	}
	restrictedFound := 0
	for _, sc := range secComms {
		if _, ok := engSet[sc.ID]; ok {
			continue
		}
		restrictedFound++
		if _, err := eng.GetCommunitySummary(engCaller, sc.ID); !errors.Is(err, ErrNotFound) {
			t.Fatalf("existence oracle leak: engCaller got %v instead of ErrNotFound for %s", err, sc.ID)
		}
		if _, err := eng.GetCommunitySummary(betaCaller, sc.ID); !errors.Is(err, ErrNotFound) {
			t.Fatalf("cross-tenant existence oracle leak: betaCaller got %v for %s", err, sc.ID)
		}
	}
	if restrictedFound == 0 {
		t.Fatalf("expected at least 1 restricted community in secCaller partition")
	}
}

func TestLiveACLRevocationRemovesCanaryFromSummaries(t *testing.T) {
	eng := NewEngine(DefaultConfig())
	for _, d := range buildTestMultiClusterDocs() {
		_, _ = eng.UpsertDocument(d)
	}
	canary := "REVOCABLE_CANARY_XYZ_4242"
	doc := CanonicalDocument{
		SourceID:  "cluster-a/revocable.md",
		Version:   "v1",
		TenantID:  "t1",
		ACL:       []string{"*"},
		Kind:      "regression_report",
		Title:     "Revocable posting incident " + canary,
		Content:   "Temporary disclosure " + canary + " in posting/mvcc.go rollup.",
		Metadata:  map[string]string{"package": "posting", "regression_class": "mvcc_and_posting_consistency"},
		UpdatedAt: time.Now().UTC(),
	}
	_, _ = eng.UpsertDocument(doc)
	eng.BuildAll()

	engCaller := CallerContext{TenantID: "t1", Principals: []string{"role:eng"}}
	secCaller := CallerContext{TenantID: "t1", Principals: []string{"role:security"}}

	before, err := eng.Query(context.Background(), engCaller, QueryRequest{
		Query: "Summarize posting regressions across all packages including " + canary,
		Mode:  ModeGlobalL1,
	})
	if err != nil {
		t.Fatalf("before revocation query: %v", err)
	}
	if !strings.Contains(before.Answer, canary) {
		t.Fatalf("expected canary visible before ACL revocation")
	}

	// Revoke ACL in place: restrict from ["*"] to ["role:security"].
	doc.Version = "v2"
	doc.ACL = []string{"role:security"}
	st, err := eng.UpsertDocument(doc)
	if err != nil {
		t.Fatalf("ACL revocation UpsertDocument: %v", err)
	}
	if st.Operation != "incremental_acl_move" {
		t.Fatalf("expected incremental_acl_move, got %s", st.Operation)
	}

	// Immediately verify 0 leaks across ALL modes for role:eng!
	for _, mode := range []QueryMode{ModeGlobalL0, ModeGlobalL1, ModeGlobalL2, ModeLocalHybrid} {
		after, err := eng.Query(context.Background(), engCaller, QueryRequest{
			Query: "Summarize posting regressions across all packages including " + canary,
			Mode:  mode,
		})
		if err != nil {
			t.Fatalf("post-revocation query mode %s: %v", mode, err)
		}
		if containsAnyCanary(after, []string{canary}) {
			t.Fatalf("post-revocation leak in mode %s: canary remained in summary!", mode)
		}
	}

	// Authorized security caller still sees it!
	secAfter, err := eng.Query(context.Background(), secCaller, QueryRequest{
		Query: "Summarize posting regressions across all packages including " + canary,
		Mode:  ModeGlobalL1,
	})
	if err != nil {
		t.Fatalf("secCaller post-revocation query: %v", err)
	}
	if !strings.Contains(secAfter.Answer, canary) {
		t.Fatalf("expected authorized security caller to see revoked document")
	}
}

func TestQueryRouterGlobalVsLocalGate(t *testing.T) {
	router := NewQueryRouter(1)

	globalQueries := []string{
		"What are the largest sources of engineering regressions across our repositories?",
		"Summarize the dominant patterns and systemic bottlenecks across all packages.",
		"Provide an executive portfolio overview of cross-cutting risks across our repositories.",
	}
	for _, q := range globalQueries {
		dec := router.Route(q)
		if dec.Target != RouteGlobalGraphRAG {
			t.Errorf("expected RouteGlobalGraphRAG for %q, got %s (%s)", q, dec.Target, dec.Reason)
		}
	}

	localQueries := []string{
		"Where is NewList defined in posting/list.go?",
		"What is the exact root cause in REG-007?",
		"Which file implements Count-Min Sketch in algo/cm-sketch.go?",
		"Find the function signature of SearchLayer in tok/hnsw/search_layer.go",
	}
	for _, q := range localQueries {
		dec := router.Route(q)
		if dec.Target != RouteLocalHybrid {
			t.Errorf("expected RouteLocalHybrid for %q, got %s (%s)", q, dec.Target, dec.Reason)
		}
	}
}

func TestDeepCopyImmutability(t *testing.T) {
	eng := NewEngine(DefaultConfig())
	for _, d := range buildTestMultiClusterDocs() {
		_, _ = eng.UpsertDocument(d)
	}
	eng.BuildAll()

	caller := CallerContext{TenantID: "t1", Principals: []string{"role:eng"}}

	// Mutate returned CanonicalDocument.
	doc, err := eng.GetDocument(caller, "cluster-a/mvcc-1.md")
	if err != nil {
		t.Fatalf("GetDocument: %v", err)
	}
	doc.ACL[0] = "POISONED"
	doc.Metadata["package"] = "POISONED"
	doc2, _ := eng.GetDocument(caller, "cluster-a/mvcc-1.md")
	if doc2.ACL[0] != "*" || doc2.Metadata["package"] != "posting" {
		t.Fatalf("CanonicalDocument internal state mutated via caller slice/map")
	}

	// Mutate returned Community and CommunitySummary.
	comms, err := eng.ListCommunities(caller, 0)
	if err != nil || len(comms) == 0 {
		t.Fatalf("ListCommunities: %v", err)
	}
	cid := comms[0].ID
	origEnt := comms[0].EntityIDs[0]
	comms[0].EntityIDs[0] = "POISONED"
	comms[0].ACL[0] = "POISONED"

	comms2, _ := eng.ListCommunities(caller, 0)
	if comms2[0].EntityIDs[0] != origEnt || comms2[0].ACL[0] != "*" {
		t.Fatalf("Community internal state mutated via caller slice")
	}

	sum, err := eng.GetCommunitySummary(caller, cid)
	if err != nil {
		t.Fatalf("GetCommunitySummary: %v", err)
	}
	origTitle := sum.Title
	sum.ACL[0] = "POISONED"
	for k := range sum.ThemeCounts {
		sum.ThemeCounts[k] = 999999
	}
	sum2, _ := eng.GetCommunitySummary(caller, cid)
	if sum2.ACL[0] != "*" || sum2.Title != origTitle {
		t.Fatalf("CommunitySummary internal state mutated")
	}
	for k, v := range sum2.ThemeCounts {
		if v == 999999 {
			t.Fatalf("CommunitySummary ThemeCounts map mutated for key %s", k)
		}
	}

	// Mutate input CanonicalDocument after UpsertDocument and verify internal copy is isolated.
	inDoc := CanonicalDocument{
		SourceID: "cluster-a/input-clone-test.md",
		Version:  "v1",
		TenantID: "t1",
		ACL:      []string{"role:eng"},
		Kind:     "doc",
		Title:    "Input Clone Safety",
		Content:  "Verify caller slice/map mutations cannot poison engine canonical truth.",
		Metadata: map[string]string{"k": "orig"},
	}
	if _, err := eng.UpsertDocument(inDoc); err != nil {
		t.Fatalf("UpsertDocument input clone test: %v", err)
	}
	inDoc.ACL[0] = "POISONED"
	inDoc.Metadata["k"] = "POISONED"
	stored, err := eng.GetDocument(caller, "cluster-a/input-clone-test.md")
	if err != nil {
		t.Fatalf("GetDocument input clone test: %v", err)
	}
	if stored.ACL[0] != "role:eng" || stored.Metadata["k"] != "orig" {
		t.Fatalf("input CanonicalDocument mutated internal store: %+v", stored)
	}

	// Revoke ACL to empty (deny-by-default) and confirm caller immediately loses access.
	stored.Version = "v2"
	stored.ACL = nil
	if _, err := eng.UpsertDocument(stored); err != nil {
		t.Fatalf("revoke to deny-all: %v", err)
	}
	if _, err := eng.GetDocument(caller, "cluster-a/input-clone-test.md"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("expected ErrNotFound after revoking ACL to nil, got %v", err)
	}

	// Mutate returned QueryResponse and verify subsequent query is unaffected.
	qResp, err := eng.Query(context.Background(), caller, QueryRequest{
		Query: "Summarize all regressions across our repositories",
		Mode:  ModeGlobalL1,
	})
	if err != nil || len(qResp.Citations) == 0 {
		t.Fatalf("Query for deep-copy check: %v", err)
	}
	origCit := qResp.Citations[0].SourceID
	qResp.Citations[0].SourceID = "POISONED"
	if len(qResp.DetectedThemes) > 0 {
		qResp.DetectedThemes[0] = "POISONED"
	}
	qResp2, _ := eng.Query(context.Background(), caller, QueryRequest{
		Query: "Summarize all regressions across our repositories",
		Mode:  ModeGlobalL1,
	})
	if qResp2.Citations[0].SourceID != origCit {
		t.Fatalf("QueryResponse mutation leaked into engine state")
	}
}

func TestInputValidationAndContextCancellation(t *testing.T) {
	eng := NewEngine(DefaultConfig())
	if _, err := eng.UpsertDocument(CanonicalDocument{SourceID: "a", Version: "v1"}); !errors.Is(err, ErrInvalidInput) {
		t.Fatalf("expected ErrInvalidInput for missing TenantID, got %v", err)
	}
	if _, err := eng.UpsertDocument(CanonicalDocument{TenantID: "t1", Version: "v1"}); !errors.Is(err, ErrInvalidInput) {
		t.Fatalf("expected ErrInvalidInput for missing SourceID, got %v", err)
	}
	if _, err := eng.UpsertDocument(CanonicalDocument{TenantID: "t1", SourceID: "a"}); !errors.Is(err, ErrInvalidInput) {
		t.Fatalf("expected ErrInvalidInput for missing Version, got %v", err)
	}
	eng.BuildAll()

	if _, err := eng.Query(context.Background(), CallerContext{}, QueryRequest{Query: "test"}); !errors.Is(err, ErrUnauthorized) {
		t.Fatalf("expected ErrUnauthorized for empty caller tenant, got %v", err)
	}
	if _, err := eng.Query(context.Background(), CallerContext{TenantID: "t1"}, QueryRequest{Query: "  "}); !errors.Is(err, ErrInvalidInput) {
		t.Fatalf("expected ErrInvalidInput for empty query, got %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := eng.Query(ctx, CallerContext{TenantID: "t1"}, QueryRequest{Query: "test"}); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
}

func TestConcurrentQueriesAndIncrementalUpdatesRace(t *testing.T) {
	eng := NewEngine(DefaultConfig())
	for _, d := range buildTestMultiClusterDocs() {
		_, _ = eng.UpsertDocument(d)
	}
	eng.BuildAll()

	caller := CallerContext{TenantID: "t1", Principals: []string{"role:eng"}}
	var wg sync.WaitGroup

	// 6 concurrent reader goroutines.
	for r := 0; r < 6; r++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			modes := []QueryMode{ModeAuto, ModeGlobalL0, ModeGlobalL1, ModeGlobalL2, ModeLocalHybrid}
			for i := 0; i < 15; i++ {
				_, _ = eng.Query(context.Background(), caller, QueryRequest{
					Query: "What are the largest sources of engineering regressions across all packages?",
					Mode:  modes[(id+i)%len(modes)],
				})
				_, _ = eng.ListCommunities(caller, 1)
				_, _ = eng.DerivedDigest()
			}
		}(r)
	}

	// 2 concurrent writer goroutines performing incremental upserts & deletes.
	for w := 0; w < 2; w++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for i := 0; i < 10; i++ {
				sid := fmt.Sprintf("cluster-a/concurrent-%d-%d.md", id, i%3)
				_, _ = eng.UpsertDocument(CanonicalDocument{
					SourceID:  sid,
					Version:   fmt.Sprintf("v%d", i+1),
					TenantID:  "t1",
					ACL:       []string{"*"},
					Kind:      "regression_report",
					Title:     fmt.Sprintf("Concurrent update %s v%d", sid, i+1),
					Content:   "MVCC posting/mvcc.go concurrent mutation update.",
					Metadata:  map[string]string{"package": "posting", "regression_class": "mvcc_and_posting_consistency"},
					UpdatedAt: time.Now().UTC(),
				})
				if i%3 == 2 {
					_, _ = eng.DeleteDocument("t1", sid)
				}
			}
		}(w)
	}

	wg.Wait()
}

func TestBenchmarkEndToEndSanity(t *testing.T) {
	report, err := RunBenchmark(BenchmarkConfig{
		RepoRoot:    "..",
		Seed:        1347,
		TopK:        5,
		TokenBudget: 4000,
		Iters:       2,
	})
	if err != nil {
		t.Fatalf("RunBenchmark failed: %v", err)
	}
	if report.RealGoFilesParsed < 50 {
		t.Fatalf("expected >= 50 real Go files parsed from hanzoai/graph, got %d", report.RealGoFilesParsed)
	}
	if !report.ACLAudit.DropRebuildBitExactParity {
		t.Fatalf("DropRebuildBitExactParity failed: %s != %s", report.ACLAudit.InitialDerivedDigest, report.ACLAudit.RebuiltDerivedDigest)
	}
	if report.ACLAudit.CanaryLeaksInGlobalSummaries != 0 ||
		report.ACLAudit.CanaryLeaksInLocalHits != 0 ||
		report.ACLAudit.CrossTenantPartitionsLeaked != 0 ||
		report.ACLAudit.ExistenceOracleLeaks != 0 ||
		report.ACLAudit.PostRevocationLeaks != 0 {
		t.Fatalf("ACL audit failed with leaks: %+v", report.ACLAudit)
	}

	// Verify GraphRAG wins on global synthesis and loses on exact lookup negative regime.
	find := func(regime, strat string) StrategyRegimeMetrics {
		for _, m := range report.RegimeMetrics {
			if m.Regime == regime && m.Strategy == strat {
				return m
			}
		}
		t.Fatalf("missing metric for regime=%s strategy=%s", regime, strat)
		return StrategyRegimeMetrics{}
	}

	globalLocal := find("global_thematic_synthesis", "ordinary_hybrid_topk")
	globalL1 := find("global_thematic_synthesis", "graphrag_global_l1")
	if globalL1.TaskSuccessRate <= globalLocal.TaskSuccessRate || globalL1.SourceRecall <= globalLocal.SourceRecall {
		t.Fatalf("expected GraphRAG L1 to beat ordinary hybrid on global synthesis: L1=%+v Local=%+v", globalL1, globalLocal)
	}

	exactLocal := find("exact_lookup_negative_regime", "ordinary_hybrid_topk")
	exactL1 := find("exact_lookup_negative_regime", "graphrag_global_l1")
	if exactLocal.TaskSuccessRate <= exactL1.TaskSuccessRate || exactLocal.AvgTotalTokens >= exactL1.AvgTotalTokens {
		t.Fatalf("expected ordinary hybrid to beat forced GraphRAG L1 on exact lookup negative regime: Local=%+v L1=%+v", exactLocal, exactL1)
	}

	exactAuto := find("exact_lookup_negative_regime", "routed_auto_gate")
	if exactAuto.TaskSuccessRate != exactLocal.TaskSuccessRate {
		t.Fatalf("expected routed_auto_gate to match ordinary hybrid on exact lookups: Auto=%+v Local=%+v", exactAuto, exactLocal)
	}

	if len(report.IncrementalRefresh) == 0 || report.IncrementalRefresh[0].SpeedupVsFullRebuild < 5.0 {
		t.Fatalf("expected >= 5x incremental refresh speedup, got %+v", report.IncrementalRefresh)
	}
}
