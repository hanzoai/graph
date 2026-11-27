/*
 * SPDX-License-Identifier: Apache-2.0
 */

package hipporag

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

func TestDerivedIndexDropAndBitExactRebuild(t *testing.T) {
	bundle := BuildStandardBenchmarkSuite()
	eng := NewEngine(DefaultPPRConfig())
	if err := eng.UpsertSourcesAndRebuild(bundle.Records); err != nil {
		t.Fatalf("UpsertSourcesAndRebuild: %v", err)
	}

	countBefore := eng.SourceCount()
	snapBefore := eng.Snapshot()
	if snapBefore.DigestSHA256 == "" || snapBefore.GraphNodes == 0 || snapBefore.GraphEdges == 0 {
		t.Fatalf("expected non-empty snapshot, got %+v", snapBefore)
	}

	req := QueryRequest{
		Query:  "Which change caused the CUDA graph memory alias?",
		Access: AccessContext{TenantID: defaultTenant},
		TopK:   5,
	}
	resBefore, err := eng.SearchHippoRAG2(req)
	if err != nil {
		t.Fatalf("SearchHippoRAG2 before drop: %v", err)
	}

	// Drop all derived indexes; canonical records must remain intact.
	eng.DropDerivedIndexes()
	if eng.HasDerivedIndexes() {
		t.Fatalf("expected HasDerivedIndexes() == false after drop")
	}
	if eng.SourceCount() != countBefore {
		t.Fatalf("canonical records altered by DropDerivedIndexes: got %d want %d", eng.SourceCount(), countBefore)
	}
	if _, err := eng.SearchHippoRAG2(req); err != ErrIndexesNotBuilt {
		t.Fatalf("expected ErrIndexesNotBuilt while dropped, got %v", err)
	}
	if _, err := eng.SearchBM25(req); err != ErrIndexesNotBuilt {
		t.Fatalf("expected ErrIndexesNotBuilt on BM25 while dropped, got %v", err)
	}
	if _, err := eng.SearchDenseRAG(req); err != ErrIndexesNotBuilt {
		t.Fatalf("expected ErrIndexesNotBuilt on DenseRAG while dropped, got %v", err)
	}

	// Rebuild from canonical records and verify bit-exact SHA-256 parity.
	eng.RebuildIndex()
	if !eng.HasDerivedIndexes() {
		t.Fatalf("expected HasDerivedIndexes() == true after rebuild")
	}
	snapAfter := eng.Snapshot()
	if snapBefore.DigestSHA256 != snapAfter.DigestSHA256 {
		t.Fatalf("non-deterministic rebuild digest: before=%s after=%s", snapBefore.DigestSHA256, snapAfter.DigestSHA256)
	}

	resAfter, err := eng.SearchHippoRAG2(req)
	if err != nil {
		t.Fatalf("SearchHippoRAG2 after rebuild: %v", err)
	}
	if len(resBefore.Hits) != len(resAfter.Hits) {
		t.Fatalf("hit count mismatch after rebuild: before=%d after=%d", len(resBefore.Hits), len(resAfter.Hits))
	}
	for i := range resBefore.Hits {
		if resBefore.Hits[i].SourceID != resAfter.Hits[i].SourceID ||
			resBefore.Hits[i].Path.Trail != resAfter.Hits[i].Path.Trail {
			t.Fatalf("hit %d differed after rebuild:\n before=%+v\n after=%+v", i, resBefore.Hits[i], resAfter.Hits[i])
		}
	}
}

func TestCanonicalRecordVersioningAndValidation(t *testing.T) {
	eng := NewEngine(DefaultPPRConfig())

	if err := eng.UpsertSource(SourceRecord{SourceID: "s1", Version: 1}); err != ErrEmptyTenant {
		t.Fatalf("expected ErrEmptyTenant, got %v", err)
	}
	if err := eng.UpsertSource(SourceRecord{TenantID: "t1", Version: 1}); err != ErrEmptySourceID {
		t.Fatalf("expected ErrEmptySourceID, got %v", err)
	}
	if err := eng.UpsertSource(SourceRecord{TenantID: "t1", SourceID: "s1", Version: 0}); err != ErrInvalidVersion {
		t.Fatalf("expected ErrInvalidVersion, got %v", err)
	}

	v1 := SourceRecord{
		SourceID:    "doc1",
		Version:     1,
		Kind:        SourceDocument,
		TenantID:    "t1",
		Title:       "Allocator Spec",
		Text:        "Initial spec v1",
		Reliability: 0.90,
	}
	v2 := SourceRecord{
		SourceID:    "doc1",
		Version:     2,
		Kind:        SourceDocument,
		TenantID:    "t1",
		Title:       "Allocator Spec",
		Text:        "Updated spec v2 with stream capture pool guard",
		Reliability: 0.98,
	}
	if err := eng.UpsertSource(v1); err != nil {
		t.Fatalf("upsert v1: %v", err)
	}
	if err := eng.UpsertSource(v2); err != nil {
		t.Fatalf("upsert v2: %v", err)
	}
	eng.RebuildIndex()

	if got := eng.SourceVersionCount("t1", "doc1"); got != 2 {
		t.Fatalf("expected 2 stored versions, got %d", got)
	}
	gotRec, ok := eng.GetSource(AccessContext{TenantID: "t1"}, "doc1")
	if !ok || gotRec.Version != 2 {
		t.Fatalf("expected latest version 2, got ok=%v rec=%+v", ok, gotRec)
	}
}

func TestStrictPreReturnTenantAndACLBridgeIsolation(t *testing.T) {
	recs, queries := BuildACLBridgeNegativeWorkload()
	eng := NewEngine(DefaultPPRConfig())
	if err := eng.UpsertSourcesAndRebuild(recs); err != nil {
		t.Fatalf("UpsertSourcesAndRebuild: %v", err)
	}

	// Empty tenant must be rejected before touching any index.
	if _, err := eng.SearchHippoRAG2(QueryRequest{Query: "cuda graph memory alias"}); err != ErrEmptyTenant {
		t.Fatalf("expected ErrEmptyTenant, got %v", err)
	}

	// Verify zero leaks across every adversarial bridge query on all 4 retrievers.
	searchers := []struct {
		name string
		fn   func(QueryRequest) (QueryResult, error)
	}{
		{"hipporag2", eng.SearchHippoRAG2},
		{"bm25", eng.SearchBM25},
		{"dense", eng.SearchDenseRAG},
		{"1hop", eng.SearchUnfiltered1Hop},
	}

	for _, s := range searchers {
		for _, q := range queries {
			res, err := s.fn(QueryRequest{Query: q.Query, Access: q.Access, TopK: 5})
			if err != nil {
				t.Fatalf("%s query %s failed: %v", s.name, q.ID, err)
			}
			if res.Audit.ForeignTenantNodesInspected != 0 || res.Audit.ForeignTenantEdgesTraversed != 0 {
				t.Fatalf("%s query %s touched foreign tenant graph: %+v", s.name, q.ID, res.Audit)
			}
			for _, h := range res.Hits {
				if h.TenantID != q.Access.TenantID {
					t.Fatalf("%s query %s leaked foreign tenant %s to %s", s.name, q.ID, h.TenantID, q.Access.TenantID)
				}
				if strings.HasPrefix(h.SourceID, "beta_") || strings.HasPrefix(h.SourceID, "alpha_restricted_") {
					t.Fatalf("%s query %s leaked restricted/confidential record %s", s.name, q.ID, h.SourceID)
				}
				if _, ok := eng.GetSource(q.Access, h.SourceID); !ok {
					t.Fatalf("%s query %s returned unauthorized source %s", s.name, q.ID, h.SourceID)
				}
			}
		}
	}

	// Verify authorized caller WITH sec:incident-response CAN read alpha_restricted_00
	// while still never seeing tenant-beta records.
	privRes, err := eng.SearchHippoRAG2(QueryRequest{
		Query:  "Which change caused cuda graph memory alias in stream capture kernel?",
		Access: AccessContext{TenantID: "tenant-alpha", Scopes: []string{"team:public", "sec:incident-response"}},
		TopK:   5,
	})
	if err != nil {
		t.Fatalf("privileged alpha query: %v", err)
	}
	sawRestricted := false
	for _, h := range privRes.Hits {
		if h.TenantID != "tenant-alpha" || strings.HasPrefix(h.SourceID, "beta_") {
			t.Fatalf("privileged alpha caller saw tenant-beta record: %+v", h)
		}
		if h.SourceID == "alpha_restricted_00" {
			sawRestricted = true
		}
	}
	if !sawRestricted {
		t.Fatalf("expected privileged tenant-alpha caller to reach alpha_restricted_00, got hits=%v", privRes.Hits)
	}
}

func TestCUDAGraphMemoryAliasPathReconstruction(t *testing.T) {
	bundle := BuildStandardBenchmarkSuite()
	eng := NewEngine(DefaultPPRConfig())
	if err := eng.UpsertSourcesAndRebuild(bundle.Records); err != nil {
		t.Fatalf("UpsertSourcesAndRebuild: %v", err)
	}

	var inc01 BenchmarkQuery
	for _, q := range bundle.Queries {
		if q.ID == "inc01" {
			inc01 = q
			break
		}
	}
	if inc01.ID == "" {
		t.Fatal("inc01 CUDA graph memory alias query missing from standard suite")
	}

	res, err := eng.SearchHippoRAG2(QueryRequest{
		Query:  inc01.Query,
		Access: inc01.Access,
		TopK:   5,
	})
	if err != nil {
		t.Fatalf("SearchHippoRAG2 inc01: %v", err)
	}

	hitSet := make(map[string]RetrievedHit)
	for _, h := range res.Hits {
		hitSet[h.SourceID] = h
	}
	for _, requiredID := range inc01.GoldSourceIDs {
		h, ok := hitSet[requiredID]
		if !ok {
			t.Fatalf("missing gold source %s in top-5 hits; got %+v", requiredID, res.Hits)
		}
		if h.Path.TargetNode == "" || h.Path.Trail == "" {
			t.Fatalf("hit %s missing reconstructed path: %+v", requiredID, h.Path)
		}
	}

	pred := extractGroundedAnswer(eng, inc01, res)
	if exactMatch(pred, inc01.GoldAnswer) != 1.0 {
		t.Fatalf("expected grounded answer %q, got %q", inc01.GoldAnswer, pred)
	}
	if evaluatePathReconstruction(inc01, res) != 1.0 {
		t.Fatalf("expected full path reconstruction accuracy 1.0 for inc01; trails=%v", res.Hits)
	}
}

func TestBoundedPPRBudgetsAndHubSuppression(t *testing.T) {
	bundle := BuildHubCollapseNegativeWorkload()
	cfg := DefaultPPRConfig()
	cfg.MaxExpandedNodes = 64
	cfg.MaxIterations = 20
	cfg.MaxHops = 4
	cfg.TopK = 5

	eng := NewEngine(cfg)
	if err := eng.UpsertSourcesAndRebuild(bundle.Records); err != nil {
		t.Fatalf("UpsertSourcesAndRebuild: %v", err)
	}

	for _, q := range bundle.Queries[:5] {
		res, err := eng.SearchHippoRAG2(QueryRequest{Query: q.Query, Access: q.Access, TopK: 5})
		if err != nil {
			t.Fatalf("SearchHippoRAG2 %s: %v", q.ID, err)
		}
		if res.Audit.ExpandedNodes > cfg.MaxExpandedNodes {
			t.Fatalf("ExpandedNodes=%d exceeded MaxExpandedNodes=%d", res.Audit.ExpandedNodes, cfg.MaxExpandedNodes)
		}
		if res.Audit.PPRIterations > cfg.MaxIterations {
			t.Fatalf("PPRIterations=%d exceeded MaxIterations=%d", res.Audit.PPRIterations, cfg.MaxIterations)
		}
	}

	negHub, err := runHubCollapseRegime(BenchmarkOptions{TopK: 5, TimedIterations: 2, InputPricePer1MUSD: 3.0})
	if err != nil {
		t.Fatalf("runHubCollapseRegime: %v", err)
	}
	if negHub.HippoRAG2Recall <= negHub.UnweightedRecall {
		t.Fatalf("expected HippoRAG 2 recall (%.3f) > unweighted hub-collapse recall (%.3f)",
			negHub.HippoRAG2Recall, negHub.UnweightedRecall)
	}
	if negHub.HippoRAG2Irrelevant >= negHub.UnweightedIrrelevant {
		t.Fatalf("expected HippoRAG 2 irrelevant ratio (%.3f) < unweighted hub-collapse ratio (%.3f)",
			negHub.HippoRAG2Irrelevant, negHub.UnweightedIrrelevant)
	}
}

func TestExternalDatasetLoaderHotpotQAAndMuSiQue(t *testing.T) {
	tmpDir := t.TempDir()

	// 1. HotpotQA JSON fixture
	hpPath := filepath.Join(tmpDir, "hotpot_sample.json")
	hpJSON := `[
		{
			"_id": "hp_ext_1",
			"question": "Which city hosts the consortium operating Aurora Telescope?",
			"answer": "Geneva",
			"supporting_facts": [["Aurora Telescope", 0], ["Helvetia Astro Consortium", 0]],
			"context": [
				["Aurora Telescope", ["Aurora Telescope is operated by Helvetia Astro Consortium."]],
				["Helvetia Astro Consortium", ["Helvetia Astro Consortium is headquartered in Geneva."]],
				["Distractor Telescope", ["Unrelated notes about telescopes and cities."]]
			]
		}
	]`
	if err := os.WriteFile(hpPath, []byte(hpJSON), 0o644); err != nil {
		t.Fatalf("write hotpot fixture: %v", err)
	}
	hpBundle, err := LoadExternalDataset(hpPath)
	if err != nil {
		t.Fatalf("LoadExternalDataset HotpotQA: %v", err)
	}
	if len(hpBundle.Records) != 3 || len(hpBundle.Queries) != 1 || len(hpBundle.Queries[0].GoldSourceIDs) != 2 {
		t.Fatalf("unexpected HotpotQA bundle: records=%d queries=%d gold=%v",
			len(hpBundle.Records), len(hpBundle.Queries), hpBundle.Queries[0].GoldSourceIDs)
	}

	// 2. MuSiQue JSONL fixture
	mqPath := filepath.Join(tmpDir, "musique_sample.jsonl")
	mqJSONL := `{"id":"mq_ext_1","question":"Which algorithm backs the engine used by Zenith Queue?","answer":"BLAKE3","paragraphs":[{"idx":0,"title":"Zenith Queue","paragraph_text":"Zenith Queue stores messages in Granite Store.","is_supporting":true},{"idx":1,"title":"Granite Store","paragraph_text":"Granite Store hashes frames with BLAKE3.","is_supporting":true},{"idx":2,"title":"Noise","paragraph_text":"Unrelated logging paragraph.","is_supporting":false}]}` + "\n"
	if err := os.WriteFile(mqPath, []byte(mqJSONL), 0o644); err != nil {
		t.Fatalf("write musique fixture: %v", err)
	}
	mqBundle, err := LoadExternalDataset(mqPath)
	if err != nil {
		t.Fatalf("LoadExternalDataset MuSiQue: %v", err)
	}
	if len(mqBundle.Records) != 3 || len(mqBundle.Queries) != 1 || len(mqBundle.Queries[0].GoldSourceIDs) != 2 {
		t.Fatalf("unexpected MuSiQue bundle: records=%d queries=%d", len(mqBundle.Records), len(mqBundle.Queries))
	}

	// Run full benchmark cleanly on the external fixture.
	rep, err := RunBenchmark(BenchmarkOptions{
		TopK:                3,
		TimedIterations:     2,
		ExternalDatasetPath: mqPath,
	})
	if err != nil {
		t.Fatalf("RunBenchmark on external dataset: %v", err)
	}
	if !rep.Honesty.ExternalDatasetUsed || len(rep.Systems) != 4 {
		t.Fatalf("expected external dataset flag and 4 systems, got %+v", rep.Honesty)
	}
}

func TestConcurrentReadWriteRebuildRaceSafety(t *testing.T) {
	bundle := BuildStandardBenchmarkSuite()
	eng := NewEngine(DefaultPPRConfig())
	if err := eng.UpsertSourcesAndRebuild(bundle.Records); err != nil {
		t.Fatalf("UpsertSourcesAndRebuild: %v", err)
	}

	var wg sync.WaitGroup
	// Concurrent readers across all 4 methods.
	for g := 0; g < 6; g++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for i := 0; i < 15; i++ {
				q := bundle.Queries[(id*7+i)%len(bundle.Queries)]
				req := QueryRequest{Query: q.Query, Access: q.Access, TopK: 5}
				switch (id + i) % 4 {
				case 0:
					_, _ = eng.SearchHippoRAG2(req)
				case 1:
					_, _ = eng.SearchBM25(req)
				case 2:
					_, _ = eng.SearchDenseRAG(req)
				default:
					_, _ = eng.SearchUnfiltered1Hop(req)
				}
			}
		}(g)
	}
	// Concurrent canonical writer + index rebuilder.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 4; i++ {
			_ = eng.UpsertSource(SourceRecord{
				SourceID:    "concurrent_doc",
				Version:     uint64(i + 1),
				Kind:        SourceDocument,
				TenantID:    defaultTenant,
				Title:       "Concurrent Update",
				Text:        "Versioned concurrent update verifying RWMutex safety.",
				Reliability: 0.90,
			})
			eng.RebuildIndex()
		}
	}()
	wg.Wait()
}

func TestEndToEndBenchmarkReportAndInvariants(t *testing.T) {
	rep, err := RunBenchmark(BenchmarkOptions{
		TopK:               5,
		TimedIterations:    3,
		InputPricePer1MUSD: 3.00,
	})
	if err != nil {
		t.Fatalf("RunBenchmark: %v", err)
	}
	if !rep.RebuildVerification.BitExactEquivalence || !rep.RebuildVerification.DroppedErrorEnforced || !rep.RebuildVerification.QueryOutputIdentical {
		t.Fatalf("rebuild verification failed: %+v", rep.RebuildVerification)
	}
	if !rep.CUDAScenario.AnswerMatch {
		t.Fatalf("CUDA graph memory alias scenario failed: %+v", rep.CUDAScenario)
	}
	if rep.NegACLBridge.HippoRAG2LeakedQueries != 0 || rep.NegACLBridge.UnfilteredControlLeakedQueries == 0 {
		t.Fatalf("ACL bridge negative regime failed: %+v", rep.NegACLBridge)
	}

	var bm25, hippo SystemMetrics
	for _, s := range rep.Systems {
		if s.CrossTenantLeaks != 0 {
			t.Fatalf("system %s had %d cross-tenant leaks", s.System, s.CrossTenantLeaks)
		}
		if s.System == "bm25_lexical_baseline" {
			bm25 = s
		}
		if s.System == "hipporag2_bounded_ppr" {
			hippo = s
		}
	}
	if hippo.MultiHopRecall <= bm25.MultiHopRecall {
		t.Fatalf("expected HippoRAG 2 multi-hop recall (%.3f) > BM25 (%.3f)", hippo.MultiHopRecall, bm25.MultiHopRecall)
	}
	if hippo.MultiHopTaskSuccess <= bm25.MultiHopTaskSuccess {
		t.Fatalf("expected HippoRAG 2 multi-hop task success (%.3f) > BM25 (%.3f)", hippo.MultiHopTaskSuccess, bm25.MultiHopTaskSuccess)
	}
	if md := RenderScoreTableMarkdown(rep); !strings.Contains(md, "hipporag2_bounded_ppr") {
		t.Fatalf("RenderScoreTableMarkdown missing hipporag2_bounded_ppr")
	}
	if readme := RenderReportREADME(rep); !strings.Contains(readme, "MIT License") {
		t.Fatalf("RenderReportREADME missing MIT license audit")
	}
}

func TestUpsertInvalidatesDerivedIndexesOnACLRevocation(t *testing.T) {
	eng := NewEngine(DefaultPPRConfig())
	pubCtx := AccessContext{TenantID: "tenant-alpha", Scopes: []string{"team:public"}}
	privCtx := AccessContext{TenantID: "tenant-alpha", Scopes: []string{"sec:restricted"}}

	v1 := SourceRecord{
		SourceID:    "doc_secret_transition",
		Version:     1,
		Kind:        SourceDocument,
		TenantID:    "tenant-alpha",
		Title:       "Kernel Guard Spec",
		Text:        "Public draft describing stream_capture_guard token alpha_sentinel_v1.",
		Reliability: 0.95,
	}
	if err := eng.UpsertSourcesAndRebuild([]SourceRecord{v1}); err != nil {
		t.Fatalf("UpsertSourcesAndRebuild v1: %v", err)
	}
	if !eng.HasDerivedIndexes() {
		t.Fatalf("expected HasDerivedIndexes() == true after initial build")
	}

	reqPub := QueryRequest{
		Query:  "stream_capture_guard alpha_sentinel_v1",
		Access: pubCtx,
		TopK:   5,
	}
	if res, err := eng.SearchHippoRAG2(reqPub); err != nil || len(res.Hits) == 0 {
		t.Fatalf("expected v1 hit before revocation, got err=%v hits=%d", err, len(res.Hits))
	}

	// Revoke public access by upserting a restricted v2 WITHOUT calling RebuildIndex yet.
	v2 := SourceRecord{
		SourceID:    "doc_secret_transition",
		Version:     2,
		Kind:        SourceDocument,
		TenantID:    "tenant-alpha",
		ACL:         []string{"sec:restricted"},
		Title:       "Kernel Guard Spec",
		Text:        "Restricted revision describing stream_capture_guard token alpha_sentinel_v2.",
		Reliability: 0.98,
	}
	if err := eng.UpsertSource(v2); err != nil {
		t.Fatalf("UpsertSource v2: %v", err)
	}
	if eng.HasDerivedIndexes() {
		t.Fatalf("expected UpsertSource changing canonical record to invalidate derived indexes")
	}
	if _, ok := eng.GetSource(pubCtx, v1.SourceID); ok {
		t.Fatalf("expected GetSource to deny revoked public caller immediately")
	}
	if rec, ok := eng.GetSource(privCtx, v1.SourceID); !ok || rec.Version != 2 {
		t.Fatalf("expected privileged caller to see v2 via GetSource, got ok=%v version=%d", ok, rec.Version)
	}

	// All four search methods must reject queries until RebuildIndex runs so stale ACL/text cannot leak.
	searchers := []struct {
		name string
		fn   func(QueryRequest) (QueryResult, error)
	}{
		{"hipporag2", eng.SearchHippoRAG2},
		{"bm25", eng.SearchBM25},
		{"dense", eng.SearchDenseRAG},
		{"1hop", eng.SearchUnfiltered1Hop},
	}
	for _, s := range searchers {
		if _, err := s.fn(reqPub); err != ErrIndexesNotBuilt {
			t.Fatalf("%s: expected ErrIndexesNotBuilt after canonical ACL revocation before rebuild, got %v", s.name, err)
		}
	}

	// Rebuild and verify public caller sees zero hits while privileged caller sees v2.
	eng.RebuildIndex()
	if !eng.HasDerivedIndexes() {
		t.Fatalf("expected HasDerivedIndexes() == true after RebuildIndex")
	}
	reqPriv := QueryRequest{
		Query:  "stream_capture_guard alpha_sentinel_v2",
		Access: privCtx,
		TopK:   5,
	}
	for _, s := range searchers {
		pubRes, err := s.fn(reqPub)
		if err != nil {
			t.Fatalf("%s public query after rebuild: %v", s.name, err)
		}
		if len(pubRes.Hits) != 0 {
			t.Fatalf("%s leaked revoked source to public caller after rebuild: %+v", s.name, pubRes.Hits)
		}
		privRes, err := s.fn(reqPriv)
		if err != nil {
			t.Fatalf("%s privileged query after rebuild: %v", s.name, err)
		}
		if len(privRes.Hits) != 1 || privRes.Hits[0].Version != 2 {
			t.Fatalf("%s expected privileged v2 hit after rebuild, got %+v", s.name, privRes.Hits)
		}
	}

	// Backfilling an older version (< canonical Version 2) records history without mutating canonical or dirtying indexes.
	if err := eng.UpsertSource(v1); err != nil {
		t.Fatalf("backfill v1: %v", err)
	}
	if !eng.HasDerivedIndexes() {
		t.Fatalf("older version backfill should not invalidate derived indexes when canonical record did not change")
	}
	if got := eng.SourceVersionCount("tenant-alpha", v1.SourceID); got != 3 {
		t.Fatalf("expected 3 history versions, got %d", got)
	}
}

func TestMuSiQueDuplicateTitleAndIdxCollisionSafety(t *testing.T) {
	tmpDir := t.TempDir()
	mqPath := filepath.Join(tmpDir, "musique_collisions.jsonl")

	// Example 1 and Example 2 share the exact same article title ("Apollo Program") and local idx (0)
	// but contain different paragraph text. Example 3 repeats Example 1's exact (title, text) at idx 5.
	lines := []string{
		`{"id":"mq_dup_1","question":"Where did Apollo 11 land?","answer":"Tranquility Base","paragraphs":[{"idx":0,"title":"Apollo Program","paragraph_text":"Apollo 11 landed at Tranquility Base in July 1969.","is_supporting":true}]}`,
		`{"id":"mq_dup_2","question":"Which valley did Apollo 17 explore?","answer":"Taurus-Littrow Valley","paragraphs":[{"idx":0,"title":"Apollo Program","paragraph_text":"Apollo 17 explored Taurus-Littrow Valley in December 1972.","is_supporting":true}]}`,
		`{"id":"mq_dup_3","question":"When did Apollo 11 land at Tranquility Base?","answer":"July 1969","paragraphs":[{"idx":5,"title":"Apollo Program","paragraph_text":"Apollo 11 landed at Tranquility Base in July 1969.","is_supporting":true}]}`,
	}
	if err := os.WriteFile(mqPath, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatalf("write musique collision fixture: %v", err)
	}

	bundle, err := LoadExternalDataset(mqPath)
	if err != nil {
		t.Fatalf("LoadExternalDataset: %v", err)
	}
	if len(bundle.Records) != 2 {
		t.Fatalf("expected 2 distinct records (distinct texts kept, identical text deduplicated), got %d: %+v", len(bundle.Records), bundle.Records)
	}
	if len(bundle.Queries) != 3 {
		t.Fatalf("expected 3 queries, got %d", len(bundle.Queries))
	}

	gold1 := bundle.Queries[0].GoldSourceIDs
	gold2 := bundle.Queries[1].GoldSourceIDs
	gold3 := bundle.Queries[2].GoldSourceIDs
	if len(gold1) != 1 || len(gold2) != 1 || len(gold3) != 1 {
		t.Fatalf("unexpected gold ID counts: q1=%v q2=%v q3=%v", gold1, gold2, gold3)
	}
	if gold1[0] == gold2[0] {
		t.Fatalf("distinct MuSiQue paragraphs with same (title, idx) collided on SourceID %q", gold1[0])
	}
	if gold1[0] != gold3[0] {
		t.Fatalf("identical MuSiQue (title, text) passages should share SourceID: got %q vs %q", gold1[0], gold3[0])
	}

	recByID := make(map[string]SourceRecord, len(bundle.Records))
	for _, r := range bundle.Records {
		recByID[r.SourceID] = r
	}
	if !strings.Contains(recByID[gold1[0]].Text, "Tranquility Base") {
		t.Fatalf("q1 gold record %q has wrong text: %q", gold1[0], recByID[gold1[0]].Text)
	}
	if !strings.Contains(recByID[gold2[0]].Text, "Taurus-Littrow Valley") {
		t.Fatalf("q2 gold record %q has wrong text: %q", gold2[0], recByID[gold2[0]].Text)
	}

	// Verify the second distinct paragraph is genuinely indexed and retrieved by the engine.
	eng := NewEngine(DefaultPPRConfig())
	if err := eng.UpsertSourcesAndRebuild(bundle.Records); err != nil {
		t.Fatalf("UpsertSourcesAndRebuild: %v", err)
	}
	res, err := eng.SearchHippoRAG2(QueryRequest{
		Query:  "Taurus-Littrow Valley Apollo 17",
		Access: AccessContext{TenantID: defaultTenant},
		TopK:   2,
	})
	if err != nil {
		t.Fatalf("SearchHippoRAG2: %v", err)
	}
	if len(res.Hits) == 0 || res.Hits[0].SourceID != gold2[0] {
		t.Fatalf("expected top hit to be second Apollo paragraph %q, got %+v", gold2[0], res.Hits)
	}
}

func TestCloneDeepCopiesEmptyNonNilMetadataMap(t *testing.T) {
	eng := NewEngine(DefaultPPRConfig())
	ctx := AccessContext{TenantID: "tenant-meta"}

	callerMeta := map[string]string{}
	src := SourceRecord{
		SourceID:    "doc_empty_meta",
		Version:     1,
		Kind:        SourceDocument,
		TenantID:    "tenant-meta",
		Title:       "Isolated Source",
		Text:        "Canonical text with an initially empty non-nil metadata map.",
		Reliability: 0.90,
		ACL:         make([]string, 0, 4),
		Metadata:    callerMeta,
		Triples:     make([]ExtractedTriple, 0, 2),
	}
	target := SourceRecord{
		SourceID:    "doc_target",
		Version:     1,
		Kind:        SourceDocument,
		TenantID:    "tenant-meta",
		Title:       "Target Source",
		Text:        "Unlinked target document.",
		Reliability: 0.90,
	}
	if err := eng.UpsertSourcesAndRebuild([]SourceRecord{src, target}); err != nil {
		t.Fatalf("UpsertSourcesAndRebuild: %v", err)
	}
	snapBefore := eng.Snapshot()

	// Mutate caller's empty metadata map after UpsertSource and mutate the map returned by GetSource.
	callerMeta["links_to"] = "doc_target"
	callerMeta["caller_mut"] = "should_not_leak"

	got, ok := eng.GetSource(ctx, "doc_empty_meta")
	if !ok {
		t.Fatalf("GetSource failed for doc_empty_meta")
	}
	if got.Metadata == nil {
		t.Fatalf("expected non-nil cloned Metadata map from GetSource")
	}
	got.Metadata["links_to"] = "doc_target"
	got.Metadata["get_mut"] = "should_not_leak"

	// Concurrently mutate caller and GetSource maps while rebuilding to verify zero data races under -race.
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		for i := 0; i < 50; i++ {
			callerMeta["links_to"] = "doc_target"
			got.Metadata["links_to"] = "doc_target"
		}
	}()
	go func() {
		defer wg.Done()
		for i := 0; i < 10; i++ {
			eng.RebuildIndex()
		}
	}()
	wg.Wait()

	fresh, ok := eng.GetSource(ctx, "doc_empty_meta")
	if !ok {
		t.Fatalf("GetSource after rebuild failed")
	}
	if fresh.Metadata == nil || len(fresh.Metadata) != 0 {
		t.Fatalf("canonical Metadata was mutated externally: got %#v", fresh.Metadata)
	}
	snapAfter := eng.Snapshot()
	if snapBefore.DigestSHA256 != snapAfter.DigestSHA256 || snapBefore.GraphEdges != snapAfter.GraphEdges {
		t.Fatalf("external empty-map mutation altered derived graph: before=%+v after=%+v", snapBefore, snapAfter)
	}
}

func TestHotpotQANormalizationCollisionSafety(t *testing.T) {
	tmpDir := t.TempDir()
	hpPath := filepath.Join(tmpDir, "hotpot_collisions.json")

	// "C" and "C++" both normalize via NormalizeEntity to "c".
	// Both articles must receive distinct collision-safe SourceIDs and distinct GoldSourceIDs,
	// while a repeated identical (title, text) article deduplicates to the same SourceID.
	payload := `[
  {
    "_id": "hp_c_1",
    "question": "Who created the C programming language?",
    "answer": "Dennis Ritchie",
    "supporting_facts": [["C", 0]],
    "context": [["C", ["C was created by Dennis Ritchie at Bell Labs."]]]
  },
  {
    "_id": "hp_cpp_2",
    "question": "Who created C++?",
    "answer": "Bjarne Stroustrup",
    "supporting_facts": [["C++", 0]],
    "context": [["C++", ["C++ was created by Bjarne Stroustrup at Bell Labs."]]]
  },
  {
    "_id": "hp_c_repeat",
    "question": "Where was C created?",
    "answer": "Bell Labs",
    "supporting_facts": [["C", 0]],
    "context": [["C", ["C was created by Dennis Ritchie at Bell Labs."]]]
  }
]`
	if err := os.WriteFile(hpPath, []byte(payload), 0o644); err != nil {
		t.Fatalf("write hotpot collision fixture: %v", err)
	}

	bundle, err := LoadExternalDataset(hpPath)
	if err != nil {
		t.Fatalf("LoadExternalDataset HotpotQA: %v", err)
	}
	if len(bundle.Records) != 2 {
		t.Fatalf("expected 2 distinct HotpotQA records for C and C++, got %d: %+v", len(bundle.Records), bundle.Records)
	}
	if len(bundle.Queries) != 3 {
		t.Fatalf("expected 3 queries, got %d", len(bundle.Queries))
	}

	goldC := bundle.Queries[0].GoldSourceIDs
	goldCpp := bundle.Queries[1].GoldSourceIDs
	goldCRepeat := bundle.Queries[2].GoldSourceIDs
	if len(goldC) != 1 || len(goldCpp) != 1 || len(goldCRepeat) != 1 {
		t.Fatalf("unexpected gold ID counts: c=%v cpp=%v repeat=%v", goldC, goldCpp, goldCRepeat)
	}
	if goldC[0] == goldCpp[0] {
		t.Fatalf("distinct HotpotQA titles 'C' and 'C++' collided on SourceID %q", goldC[0])
	}
	if goldC[0] != goldCRepeat[0] {
		t.Fatalf("identical HotpotQA (title, text) articles should share SourceID: got %q vs %q", goldC[0], goldCRepeat[0])
	}

	recByID := make(map[string]SourceRecord, len(bundle.Records))
	for _, r := range bundle.Records {
		recByID[r.SourceID] = r
	}
	if recByID[goldC[0]].Title != "C" || !strings.Contains(recByID[goldC[0]].Text, "Dennis Ritchie") {
		t.Fatalf("unexpected C record: %+v", recByID[goldC[0]])
	}
	if recByID[goldCpp[0]].Title != "C++" || !strings.Contains(recByID[goldCpp[0]].Text, "Bjarne Stroustrup") {
		t.Fatalf("unexpected C++ record: %+v", recByID[goldCpp[0]])
	}
}

func TestLoadExternalDatasetNativeBundleZeroIndent(t *testing.T) {
	tmpDir := t.TempDir()
	nativePath := filepath.Join(tmpDir, "native_zero_indent.json")

	want := WorkloadBundle{
		Name: "native_zero_indent",
		Records: []SourceRecord{
			{
				SourceID:    "doc_1",
				Version:     1,
				Kind:        SourceDocument,
				TenantID:    defaultTenant,
				Title:       "Zero Indent Record",
				Text:        "Native workload bundle formatted with zero-indent JSON.",
				Reliability: 0.95,
			},
		},
		Queries: []BenchmarkQuery{
			{
				ID:            "q_1",
				Category:      "single_hop_fact",
				Query:         "What is formatted with zero-indent JSON?",
				Access:        AccessContext{TenantID: defaultTenant},
				GoldSourceIDs: []string{"doc_1"},
			},
		},
	}

	// json.MarshalIndent(..., "", "") places nested object braces on column zero ("\n{").
	raw, err := json.MarshalIndent(want, "", "")
	if err != nil {
		t.Fatalf("json.MarshalIndent: %v", err)
	}
	if !strings.Contains(string(raw), "\n{") {
		t.Fatalf("expected zero-indent JSON to contain '\\n{' for regression coverage")
	}
	if err := os.WriteFile(nativePath, raw, 0o644); err != nil {
		t.Fatalf("write native zero-indent fixture: %v", err)
	}

	got, err := LoadExternalDataset(nativePath)
	if err != nil {
		t.Fatalf("LoadExternalDataset zero-indent native bundle failed: %v", err)
	}
	if got.Name != want.Name || len(got.Records) != 1 || len(got.Queries) != 1 {
		t.Fatalf("unexpected decoded native bundle: %+v", got)
	}
	if got.Records[0].SourceID != "doc_1" || got.Queries[0].ID != "q_1" {
		t.Fatalf("mismatched decoded native bundle contents: %+v", got)
	}
}
