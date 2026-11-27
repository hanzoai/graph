/*
 * SPDX-License-Identifier: Apache-2.0
 */

package graphplanner

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

func TestCompactCSRDropAndBitExactRebuild(t *testing.T) {
	store := NewGraphStore(DefaultStoreConfig())
	nodes := []CanonicalNode{
		{SourceID: "n1", Version: 1, TenantID: "t1", Kind: KindRepoBlob, Title: "posting", Text: "Posting list storage in zapdb", Labels: []string{"go_package"}},
		{SourceID: "n2", Version: 1, TenantID: "t1", Kind: KindRepoBlob, Title: "codec", Text: "Bit-packed UID encoder and decoder", Labels: []string{"go_package"}},
		{SourceID: "n3", Version: 1, TenantID: "t1", Kind: KindEntity, Title: "Encoder", Text: "Compact delta varint encoder for posting lists", Labels: []string{"go_symbol"}},
	}
	edges := []CanonicalEdge{
		{EdgeID: "e1", Version: 1, TenantID: "t1", FromSourceID: "n1", ToSourceID: "n2", Type: "imports", Weight: 1.0},
		{EdgeID: "e2", Version: 1, TenantID: "t1", FromSourceID: "n2", ToSourceID: "n3", Type: "defines", Weight: 1.0},
	}
	if err := store.UpsertBatchAndRebuild(nodes, edges); err != nil {
		t.Fatalf("UpsertBatchAndRebuild: %v", err)
	}

	stBefore := store.StorageStats()
	if !stBefore.DerivedBuilt || stBefore.IndexedNodes != 3 || stBefore.IndexedEdges != 2 {
		t.Fatalf("unexpected storage stats before drop: %+v", stBefore)
	}
	if stBefore.BytesPerEdge != 28.0 {
		t.Fatalf("expected exact 28.0 B/edge bidirectional CSR footprint, got %.2f", stBefore.BytesPerEdge)
	}

	req := QueryRequest{
		Query:                  "symbols defined by codec",
		AnchorIDs:              []string{"n2"},
		EdgeTypes:              []string{"defines"},
		MaxHops:                1,
		TopK:                   5,
		Access:                 AccessContext{TenantID: "t1"},
		RequireStructuralMatch: true,
	}
	resBefore, err := store.ExecuteQuery(req)
	if err != nil {
		t.Fatalf("ExecuteQuery before drop: %v", err)
	}
	if len(resBefore.Hits) != 1 || resBefore.Hits[0].SourceID != "n3" {
		t.Fatalf("expected hit n3 before drop, got %+v", resBefore.Hits)
	}

	// Drop derived indexes; canonical records must remain intact.
	store.DropDerivedIndexes()
	if store.HasDerivedIndexes() {
		t.Fatalf("expected HasDerivedIndexes() == false after DropDerivedIndexes")
	}
	if _, err := store.ExecuteQuery(req); err != ErrDerivedIndexNotBuilt {
		t.Fatalf("expected ErrDerivedIndexNotBuilt after drop, got %v", err)
	}
	if _, ok := store.GetNode(AccessContext{TenantID: "t1"}, "n1"); !ok {
		t.Fatalf("canonical node n1 lost after DropDerivedIndexes")
	}

	// Rebuild and assert bit-exact digest and identical query output.
	if err := store.RebuildIndex(); err != nil {
		t.Fatalf("RebuildIndex: %v", err)
	}
	stAfter := store.StorageStats()
	if stBefore.IndexDigestSHA256 != stAfter.IndexDigestSHA256 {
		t.Fatalf("rebuild digest mismatch: before=%s after=%s", stBefore.IndexDigestSHA256, stAfter.IndexDigestSHA256)
	}
	resAfter, err := store.ExecuteQuery(req)
	if err != nil {
		t.Fatalf("ExecuteQuery after rebuild: %v", err)
	}
	if len(resAfter.Hits) != len(resBefore.Hits) || resAfter.Hits[0].SourceID != resBefore.Hits[0].SourceID || resAfter.Hits[0].Score != resBefore.Hits[0].Score {
		t.Fatalf("query output drifted across drop/rebuild: before=%+v after=%+v", resBefore.Hits, resAfter.Hits)
	}
}

func TestAllFivePhysicalPlansSelectedByCostPlanner(t *testing.T) {
	bundle, err := BuildCombinedBenchmarkSuite("..", 1347)
	if err != nil {
		t.Fatalf("BuildCombinedBenchmarkSuite: %v", err)
	}
	store := NewGraphStore(DefaultStoreConfig())
	if err := store.UpsertBatchAndRebuild(bundle.Nodes, bundle.Edges); err != nil {
		t.Fatalf("UpsertBatchAndRebuild: %v", err)
	}

	seenPlans := make(map[ExecutionPlan]int)
	for _, bq := range bundle.Queries {
		store.cache.Clear()
		if bq.WarmBeforeRun {
			warm := bq.Request
			warm.ForcePlan = PlanGraphFirst
			if _, err := store.ExecuteQuery(warm); err != nil {
				t.Fatalf("warm %s: %v", bq.ID, err)
			}
		}
		resp, err := store.ExecuteQuery(bq.Request)
		if err != nil {
			t.Fatalf("ExecuteQuery %s: %v", bq.ID, err)
		}
		if len(resp.Explanation.Alternatives) != len(AllExecutionPlans) {
			t.Fatalf("query %s expected %d plan alternatives, got %d", bq.ID, len(AllExecutionPlans), len(resp.Explanation.Alternatives))
		}
		if resp.Explanation.SelectionReason == "" {
			t.Fatalf("query %s missing SelectionReason", bq.ID)
		}
		seenPlans[resp.Plan]++
	}

	for _, p := range AllExecutionPlans {
		if seenPlans[p] == 0 {
			t.Fatalf("cost-based planner never selected plan %q; distribution=%v", p, seenPlans)
		}
	}
}

func TestHubExplosionPivotsToVectorFirst(t *testing.T) {
	bundle, err := BuildCombinedBenchmarkSuite("", 1347)
	if err != nil {
		t.Fatalf("BuildCombinedBenchmarkSuite: %v", err)
	}
	store := NewGraphStore(DefaultStoreConfig())
	if err := store.UpsertBatchAndRebuild(bundle.Nodes, bundle.Edges); err != nil {
		t.Fatalf("UpsertBatchAndRebuild: %v", err)
	}

	checked := 0
	for _, bq := range bundle.Queries {
		if bq.Category != "hub_degree_explosion" {
			continue
		}
		checked++
		resp, err := store.ExecuteQuery(bq.Request)
		if err != nil {
			t.Fatalf("ExecuteQuery %s: %v", bq.ID, err)
		}
		if resp.Plan != PlanVectorFirst {
			t.Fatalf("expected hub explosion query %s to pivot to vector_first, got %s (reason: %s)", bq.ID, resp.Plan, resp.Explanation.SelectionReason)
		}
		if len(resp.Hits) == 0 || resp.Hits[0].SourceID != bq.GoldSourceIDs[0] {
			t.Fatalf("expected vector_first to rank gold hit %s first on hub query %s, got %+v", bq.GoldSourceIDs[0], bq.ID, resp.Hits)
		}
	}
	if checked == 0 {
		t.Fatalf("expected hub_degree_explosion queries in suite")
	}
}

func TestSimpleFactualNegativeRegimeBypass(t *testing.T) {
	bundle, err := BuildCombinedBenchmarkSuite("", 1347)
	if err != nil {
		t.Fatalf("BuildCombinedBenchmarkSuite: %v", err)
	}
	store := NewGraphStore(DefaultStoreConfig())
	if err := store.UpsertBatchAndRebuild(bundle.Nodes, bundle.Edges); err != nil {
		t.Fatalf("UpsertBatchAndRebuild: %v", err)
	}

	checked := 0
	for _, bq := range bundle.Queries {
		if bq.Category != "graphrag_simple_factual" {
			continue
		}
		checked++
		resp, err := store.ExecuteQuery(bq.Request)
		if err != nil {
			t.Fatalf("ExecuteQuery %s: %v", bq.ID, err)
		}
		if resp.Plan != PlanPlainRAGBypass {
			t.Fatalf("expected simple factual query %s to select plain_rag_bypass, got %s", bq.ID, resp.Plan)
		}
		if !resp.Explanation.SimpleFactualDetected {
			t.Fatalf("expected SimpleFactualDetected=true on %s", bq.ID)
		}
		if resp.Telemetry.EdgesTraversed != 0 {
			t.Fatalf("plain_rag_bypass must traverse 0 edges, got %d", resp.Telemetry.EdgesTraversed)
		}
		if len(resp.Hits) == 0 || resp.Hits[0].SourceID != bq.GoldSourceIDs[0] {
			t.Fatalf("expected gold factual doc %s as top hit on %s, got %+v", bq.GoldSourceIDs[0], bq.ID, resp.Hits)
		}
	}
	if checked == 0 {
		t.Fatalf("expected graphrag_simple_factual queries in suite")
	}
}

func TestLRUCacheEvictionScopeIsolationAndMutationInvalidation(t *testing.T) {
	cfg := DefaultStoreConfig()
	cfg.CacheCapacity = 2
	store := NewGraphStore(cfg)

	nodes := []CanonicalNode{
		{SourceID: "a1", Version: 1, TenantID: "t1", ACL: []string{"public", "admin"}, Title: "a1", Text: "anchor 1"},
		{SourceID: "a2", Version: 1, TenantID: "t1", ACL: []string{"public"}, Title: "a2", Text: "anchor 2"},
		{SourceID: "a3", Version: 1, TenantID: "t1", ACL: []string{"public"}, Title: "a3", Text: "anchor 3"},
		{SourceID: "pub1", Version: 1, TenantID: "t1", ACL: []string{"public"}, Title: "pub1", Text: "public target 1"},
		{SourceID: "sec1", Version: 1, TenantID: "t1", ACL: []string{"admin"}, Title: "sec1", Text: "secret admin target 1"},
		{SourceID: "pub2", Version: 1, TenantID: "t1", ACL: []string{"public"}, Title: "pub2", Text: "public target 2"},
		{SourceID: "pub3", Version: 1, TenantID: "t1", ACL: []string{"public"}, Title: "pub3", Text: "public target 3"},
	}
	edges := []CanonicalEdge{
		{EdgeID: "e1", Version: 1, TenantID: "t1", FromSourceID: "a1", ToSourceID: "pub1", Type: "links", Weight: 1.0},
		{EdgeID: "e2", Version: 1, TenantID: "t1", ACL: []string{"admin"}, FromSourceID: "a1", ToSourceID: "sec1", Type: "links", Weight: 1.5},
		{EdgeID: "e3", Version: 1, TenantID: "t1", FromSourceID: "a2", ToSourceID: "pub2", Type: "links", Weight: 1.0},
		{EdgeID: "e4", Version: 1, TenantID: "t1", FromSourceID: "a3", ToSourceID: "pub3", Type: "links", Weight: 1.0},
	}
	if err := store.UpsertBatchAndRebuild(nodes, edges); err != nil {
		t.Fatalf("UpsertBatchAndRebuild: %v", err)
	}

	// 1. Admin warms a1 -> sees both pub1 and sec1.
	adminReq := QueryRequest{
		Query:     "target 1",
		AnchorIDs: []string{"a1"},
		EdgeTypes: []string{"links"},
		MaxHops:   1,
		TopK:      5,
		Access:    AccessContext{TenantID: "t1", Scopes: []string{"public", "admin"}},
		ForcePlan: PlanCachedNeighborhood,
	}
	adminRes, err := store.ExecuteQuery(adminReq)
	if err != nil {
		t.Fatalf("admin query: %v", err)
	}
	if len(adminRes.Hits) != 2 {
		t.Fatalf("expected admin to see 2 neighbors, got %+v", adminRes.Hits)
	}

	// 2. Public caller queries same anchor a1 -> MUST NOT hit admin's cache entry or leak sec1!
	pubReq := adminReq
	pubReq.Access = AccessContext{TenantID: "t1", Scopes: []string{"public"}}
	pubRes, err := store.ExecuteQuery(pubReq)
	if err != nil {
		t.Fatalf("public query: %v", err)
	}
	if pubRes.Telemetry.CacheHits != 0 || pubRes.Telemetry.CacheMisses != 1 {
		t.Fatalf("expected cache miss for distinct ACL scope, got hits=%d misses=%d", pubRes.Telemetry.CacheHits, pubRes.Telemetry.CacheMisses)
	}
	if len(pubRes.Hits) != 1 || pubRes.Hits[0].SourceID != "pub1" {
		t.Fatalf("public caller leaked restricted neighbor from admin cache: %+v", pubRes.Hits)
	}

	// 3. Repeat public query -> cache HIT with 0 edges traversed.
	pubRes2, err := store.ExecuteQuery(pubReq)
	if err != nil {
		t.Fatalf("public repeat query: %v", err)
	}
	if pubRes2.Telemetry.CacheHits != 1 || pubRes2.Telemetry.EdgesTraversed != 0 {
		t.Fatalf("expected warm cache hit with 0 edges traversed, got %+v", pubRes2.Telemetry)
	}

	// 4. Insert 2 more cache entries into capacity=2 LRU -> triggers eviction.
	for _, aid := range []string{"a2", "a3"} {
		r := pubReq
		r.AnchorIDs = []string{aid}
		if _, err := store.ExecuteQuery(r); err != nil {
			t.Fatalf("query %s: %v", aid, err)
		}
	}
	st := store.StorageStats()
	if st.CacheEntries > 2 || st.CacheEvictions == 0 {
		t.Fatalf("expected bounded LRU capacity=2 with evictions, got entries=%d evictions=%d", st.CacheEntries, st.CacheEvictions)
	}

	// 5. Mutate ACL on pub3 (revoke public access) -> cache MUST be invalidated immediately and ErrDerivedIndexNotBuilt enforced.
	if err := store.UpsertNode(CanonicalNode{
		SourceID: "pub3", Version: 2, TenantID: "t1", ACL: []string{"admin"}, Title: "pub3", Text: "now restricted",
	}); err != nil {
		t.Fatalf("revoke pub3 ACL: %v", err)
	}
	stAfterMut := store.StorageStats()
	if stAfterMut.CacheEntries != 0 {
		t.Fatalf("expected tenant cache purge immediately on ACL mutation, got %d entries", stAfterMut.CacheEntries)
	}
	r3 := pubReq
	r3.AnchorIDs = []string{"a3"}
	if _, err := store.ExecuteQuery(r3); err != ErrDerivedIndexNotBuilt {
		t.Fatalf("expected ErrDerivedIndexNotBuilt after ACL mutation before rebuild, got %v", err)
	}
	if err := store.RebuildIndex(); err != nil {
		t.Fatalf("RebuildIndex after revocation: %v", err)
	}
	resAfterRev, err := store.ExecuteQuery(r3)
	if err != nil {
		t.Fatalf("query a3 after rebuild: %v", err)
	}
	if len(resAfterRev.Hits) != 0 {
		t.Fatalf("expected 0 hits after pub3 ACL revoked, got %+v", resAfterRev.Hits)
	}

	// 6. DeleteEdge invalidates cache and removes link after rebuild.
	if !store.DeleteEdge("t1", "e1") {
		t.Fatalf("expected DeleteEdge e1 to return true")
	}
	if store.HasDerivedIndexes() {
		t.Fatalf("expected HasDerivedIndexes() == false after DeleteEdge")
	}
	if err := store.RebuildIndex(); err != nil {
		t.Fatalf("RebuildIndex after DeleteEdge: %v", err)
	}
	resAfterDel, err := store.ExecuteQuery(pubReq)
	if err != nil {
		t.Fatalf("query a1 after edge delete: %v", err)
	}
	if len(resAfterDel.Hits) != 0 {
		t.Fatalf("expected 0 hits after deleting edge e1, got %+v", resAfterDel.Hits)
	}
}

func TestDeepCopyIsolationOnInputsOutputsAndCache(t *testing.T) {
	store := NewGraphStore(DefaultStoreConfig())
	callerACL := []string{"team:alpha"}
	callerLabels := []string{"doc"}
	callerMeta := map[string]string{} // empty non-nil map!
	callerEmb := ComputeEmbedding("isolated record text")

	node := CanonicalNode{
		SourceID:  "doc_iso",
		Version:   1,
		TenantID:  "t-iso",
		ACL:       callerACL,
		Kind:      KindDocument,
		Title:     "Isolated",
		Text:      "isolated record text",
		Labels:    callerLabels,
		Metadata:  callerMeta,
		Embedding: callerEmb,
	}
	if err := store.UpsertNode(node); err != nil {
		t.Fatalf("UpsertNode: %v", err)
	}
	if err := store.RebuildIndex(); err != nil {
		t.Fatalf("RebuildIndex: %v", err)
	}

	// Mutate caller inputs after UpsertNode.
	callerACL[0] = "team:hacked"
	callerLabels[0] = "corrupted"
	callerMeta["injected"] = "evil"
	callerEmb[0] = 999.0

	got, ok := store.GetNode(AccessContext{TenantID: "t-iso", Scopes: []string{"team:alpha"}}, "doc_iso")
	if !ok {
		t.Fatalf("GetNode failed after caller mutated input slices/maps")
	}
	if got.ACL[0] != "team:alpha" || got.Labels[0] != "doc" || got.Metadata == nil || len(got.Metadata) != 0 || got.Embedding[0] == 999.0 {
		t.Fatalf("input mutation leaked into canonical store: %+v", got)
	}

	// Mutate GetNode output and ExecuteQuery output; ensure store stays pristine.
	got.Metadata["mutated"] = "yes"
	got.Labels[0] = "mutated_label"

	resp, err := store.ExecuteQuery(QueryRequest{
		Query:   "isolated record text",
		MaxHops: 0,
		TopK:    3,
		Access:  AccessContext{TenantID: "t-iso", Scopes: []string{"team:alpha"}},
	})
	if err != nil || len(resp.Hits) != 1 {
		t.Fatalf("ExecuteQuery failed: err=%v hits=%+v", err, resp.Hits)
	}
	resp.Hits[0].Metadata["hit_mut"] = "yes"
	resp.Hits[0].Labels[0] = "hit_mut_label"

	fresh, ok := store.GetNode(AccessContext{TenantID: "t-iso", Scopes: []string{"team:alpha"}}, "doc_iso")
	if !ok || len(fresh.Metadata) != 0 || fresh.Labels[0] != "doc" {
		t.Fatalf("output mutation leaked into canonical store: %+v", fresh)
	}
}

func TestStrictPreReturnTenantAndACLIsolationAcrossAllPlans(t *testing.T) {
	bundle, err := BuildCombinedBenchmarkSuite("", 1347)
	if err != nil {
		t.Fatalf("BuildCombinedBenchmarkSuite: %v", err)
	}
	store := NewGraphStore(DefaultStoreConfig())
	if err := store.UpsertBatchAndRebuild(bundle.Nodes, bundle.Edges); err != nil {
		t.Fatalf("UpsertBatchAndRebuild: %v", err)
	}

	audit := runACLSecurityAudit(store, bundle.Queries)
	if audit.ProbesExecuted == 0 {
		t.Fatalf("expected non-zero ACL security probes")
	}
	if audit.LeaksDetected != 0 {
		t.Fatalf("ACL leaks detected: %+v", audit)
	}
	if audit.ForeignNodesInspected != 0 || audit.ForeignEdgesTraversed != 0 {
		t.Fatalf("foreign tenant inspected: %+v", audit)
	}
	if !audit.CrossScopeCacheIsolated || !audit.PostRevocationInvalidated {
		t.Fatalf("ACL cache isolation/revocation audit failed: %+v", audit)
	}

	// Verify passing an unauthorized node as an explicit anchor reveals zero
	// anchor count or local degree oracle info in QueryExplanation.
	expl, err := store.PlanQuery(QueryRequest{
		Query:     "classified secops audit",
		AnchorIDs: []string{"acl_secret_doc_01"},
		MaxHops:   1,
		TopK:      3,
		Access:    AccessContext{TenantID: DefaultTenantID, Scopes: []string{"team:public"}},
	})
	if err != nil {
		t.Fatalf("PlanQuery unauthorized anchor: %v", err)
	}
	if expl.AnchorCount != 0 {
		t.Fatalf("unauthorized anchor leaked existence via AnchorCount=%d", expl.AnchorCount)
	}
}

func TestRecordValidationAndVersionHistorySemantics(t *testing.T) {
	store := NewGraphStore(DefaultStoreConfig())

	// Reject invalid records.
	badNodes := []CanonicalNode{
		{SourceID: "", Version: 1, TenantID: "t1"},
		{SourceID: "n1", Version: 1, TenantID: ""},
		{SourceID: "n1", Version: 0, TenantID: "t1"},
		{SourceID: "n1", Version: 1, TenantID: "t1", Embedding: []float32{1, 2, 3}},
		{SourceID: "n1", Version: 1, TenantID: "t1", Embedding: func() []float32 { v := make([]float32, EmbeddingDim); v[0] = float32(math.NaN()); return v }()},
	}
	for i, bn := range badNodes {
		if err := store.UpsertNode(bn); err == nil {
			t.Fatalf("badNodes[%d] expected validation error, got nil", i)
		}
	}

	badEdges := []CanonicalEdge{
		{EdgeID: "", Version: 1, TenantID: "t1", FromSourceID: "a", ToSourceID: "b", Type: "links", Weight: 1},
		{EdgeID: "e1", Version: 0, TenantID: "t1", FromSourceID: "a", ToSourceID: "b", Type: "links", Weight: 1},
		{EdgeID: "e1", Version: 1, TenantID: "t1", FromSourceID: "a", ToSourceID: "b", Type: "", Weight: 1},
		{EdgeID: "e1", Version: 1, TenantID: "t1", FromSourceID: "a", ToSourceID: "b", Type: "links", Weight: float32(math.Inf(1))},
		{EdgeID: "e1", Version: 1, TenantID: "t1", FromSourceID: "a", ToSourceID: "b", Type: "links", Weight: -1},
	}
	for i, be := range badEdges {
		if err := store.UpsertEdge(be); err == nil {
			t.Fatalf("badEdges[%d] expected validation error, got nil", i)
		}
	}

	// Upsert v2, rebuild, then backfill strictly older v1: must record history WITHOUT mutating canonical or dirtying indexes!
	v2 := CanonicalNode{SourceID: "ver_doc", Version: 2, TenantID: "t1", Title: "v2", Text: "latest canonical version"}
	v1 := CanonicalNode{SourceID: "ver_doc", Version: 1, TenantID: "t1", Title: "v1", Text: "historical version"}
	if err := store.UpsertNode(v2); err != nil {
		t.Fatalf("UpsertNode v2: %v", err)
	}
	if err := store.RebuildIndex(); err != nil {
		t.Fatalf("RebuildIndex: %v", err)
	}
	if err := store.UpsertNode(v1); err != nil {
		t.Fatalf("backfill v1: %v", err)
	}
	if !store.HasDerivedIndexes() {
		t.Fatalf("older version backfill should not invalidate derived indexes")
	}
	if got := store.NodeVersionCount("t1", "ver_doc"); got != 2 {
		t.Fatalf("expected 2 historical versions, got %d", got)
	}
	cur, ok := store.GetNode(AccessContext{TenantID: "t1"}, "ver_doc")
	if !ok || cur.Version != 2 || cur.Title != "v2" {
		t.Fatalf("canonical state unexpectedly overwritten by older version: %+v", cur)
	}
}

func TestConcurrentQueriesMutationsAndRebuildsRaceSafety(t *testing.T) {
	store := NewGraphStore(DefaultStoreConfig())
	for i := 0; i < 30; i++ {
		if err := store.UpsertNode(CanonicalNode{
			SourceID: fmt.Sprintf("n_%02d", i),
			Version:  1,
			TenantID: "t-race",
			Title:    fmt.Sprintf("node_%02d", i),
			Text:     fmt.Sprintf("concurrent test document %02d for graphplanner race test", i),
			Labels:   []string{"doc"},
		}); err != nil {
			t.Fatalf("seed node: %v", err)
		}
		if i > 0 {
			if err := store.UpsertEdge(CanonicalEdge{
				EdgeID:       fmt.Sprintf("e_%02d", i),
				Version:      1,
				TenantID:     "t-race",
				FromSourceID: fmt.Sprintf("n_%02d", i-1),
				ToSourceID:   fmt.Sprintf("n_%02d", i),
				Type:         "links",
				Weight:       1.0,
			}); err != nil {
				t.Fatalf("seed edge: %v", err)
			}
		}
	}
	if err := store.RebuildIndex(); err != nil {
		t.Fatalf("RebuildIndex: %v", err)
	}

	var wg sync.WaitGroup
	for g := 0; g < 4; g++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for i := 0; i < 40; i++ {
				_, _ = store.ExecuteQuery(QueryRequest{
					Query:     "concurrent test document",
					AnchorIDs: []string{fmt.Sprintf("n_%02d", (id+i)%25)},
					MaxHops:   2,
					TopK:      3,
					Access:    AccessContext{TenantID: "t-race"},
				})
				_ = store.StorageStats()
			}
		}(g)
	}
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 15; i++ {
			_ = store.UpsertNode(CanonicalNode{
				SourceID: "n_00",
				Version:  uint64(i + 2),
				TenantID: "t-race",
				Title:    "node_00",
				Text:     fmt.Sprintf("updated concurrent doc %d", i),
				Labels:   []string{"doc"},
			})
			_ = store.RebuildIndex()
		}
	}()
	wg.Wait()
}

func TestExternalWorkloadLoader(t *testing.T) {
	tmp := t.TempDir()
	path := filepath.Join(tmp, "ext_workload.json")
	want := WorkloadBundle{
		Name: "ext_stark_sample",
		Seed: 42,
		Nodes: []CanonicalNode{
			{SourceID: "p1", Version: 1, TenantID: DefaultTenantID, Title: "Paper A", Text: "Graph neural planner paper", Labels: []string{"paper"}},
			{SourceID: "p2", Version: 1, TenantID: DefaultTenantID, Title: "Paper B", Text: "Compact CSR storage paper", Labels: []string{"paper"}},
		},
		Edges: []CanonicalEdge{
			{EdgeID: "c1", Version: 1, TenantID: DefaultTenantID, FromSourceID: "p1", ToSourceID: "p2", Type: "cites", Weight: 1.0},
		},
		Queries: []BenchmarkQuery{
			{
				ID:       "q1",
				Category: "external_stark",
				Request: QueryRequest{
					Query:     "which paper cites compact CSR storage?",
					AnchorIDs: []string{"p1"},
					EdgeTypes: []string{"cites"},
					MaxHops:   1,
					TopK:      3,
					Access:    AccessContext{TenantID: DefaultTenantID},
				},
				GoldSourceIDs: []string{"p2"},
				GoldAnswer:    "Compact CSR",
			},
		},
	}
	raw, err := json.MarshalIndent(want, "", "  ")
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	got, err := LoadExternalWorkload(path)
	if err != nil {
		t.Fatalf("LoadExternalWorkload: %v", err)
	}
	if got.Name != want.Name || len(got.Nodes) != 2 || len(got.Queries) != 1 {
		t.Fatalf("unexpected external workload: %+v", got)
	}
}

func TestEndToEndBenchmarkSuiteAndRealCodeCorpus(t *testing.T) {
	report, err := RunBenchmarkSuite(BenchmarkOptions{
		RepoRoot:   "..",
		Seed:       1347,
		TopK:       5,
		Iterations: 3,
	})
	if err != nil {
		t.Fatalf("RunBenchmarkSuite: %v", err)
	}
	if report.RealCodeFilesParsed < 20 {
		t.Fatalf("expected >= 20 real Go files parsed from hanzoai/graph, got %d", report.RealCodeFilesParsed)
	}
	if !report.Rebuild.BitExactDigest || !report.Rebuild.EnforcedNotBuiltErr || !report.Rebuild.QueryOutputIdentical {
		t.Fatalf("rebuild verification failed: %+v", report.Rebuild)
	}
	if report.Security.LeaksDetected != 0 || !report.Security.CrossScopeCacheIsolated || !report.Security.PostRevocationInvalidated {
		t.Fatalf("security audit failed: %+v", report.Security)
	}

	var adaptive *StrategyMetrics
	for i := range report.OverallStrategies {
		s := &report.OverallStrategies[i]
		if s.ACLLeaks != 0 || s.ForeignNodesInspected != 0 || s.ForeignEdgesTraversed != 0 {
			t.Fatalf("strategy %s leaked ACL/foreign tenant data: %+v", s.Strategy, s)
		}
		if s.Strategy == "cost_based_adaptive_planner" {
			adaptive = s
		}
	}
	if adaptive == nil {
		t.Fatalf("missing cost_based_adaptive_planner in overall strategies")
	}
	if adaptive.TaskSuccessRate < 0.90 {
		t.Fatalf("expected adaptive planner TaskSuccessRate >= 90%%, got %.1f%%", adaptive.TaskSuccessRate*100)
	}
	if adaptive.PlanMatchExpectedRate < 0.85 {
		t.Fatalf("expected adaptive planner PlanMatchExpectedRate >= 85%%, got %.1f%% (counts=%v)", adaptive.PlanMatchExpectedRate*100, adaptive.PlanSelectionCounts)
	}

	md := RenderMarkdownDossier(report)
	for _, heading := range []string{"## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6.", "Apache AGE", "NebulaGraph", "Cognee", "plain_rag_bypass"} {
		if !strings.Contains(md, heading) {
			t.Fatalf("generated markdown dossier missing required section/term %q", heading)
		}
	}
}
