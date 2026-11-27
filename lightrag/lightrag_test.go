/*
 * SPDX-FileCopyrightText: © 2026 Hanzo AI, Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

package lightrag

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

func TestValidationAndErrors(t *testing.T) {
	ctx := context.Background()
	eng := NewEngine(DefaultEngineConfig())

	if _, err := eng.UpsertSource(ctx, SourceRecord{SourceID: "s1", Version: 1, Content: "hello"}); !errors.Is(err, ErrEmptyTenant) {
		t.Fatalf("expected ErrEmptyTenant, got %v", err)
	}
	if _, err := eng.UpsertSource(ctx, SourceRecord{TenantID: "t1", Version: 1, Content: "hello"}); !errors.Is(err, ErrInvalidSource) {
		t.Fatalf("expected ErrInvalidSource for empty source_id, got %v", err)
	}
	if _, err := eng.UpsertSource(ctx, SourceRecord{TenantID: "t1", SourceID: "s1", Version: 0, Content: "hello"}); !errors.Is(err, ErrInvalidSource) {
		t.Fatalf("expected ErrInvalidSource for version 0, got %v", err)
	}
	if _, err := eng.UpsertSource(ctx, SourceRecord{TenantID: "t1", SourceID: "s1", Version: 1, Content: "   "}); !errors.Is(err, ErrInvalidSource) {
		t.Fatalf("expected ErrInvalidSource for empty content, got %v", err)
	}

	if _, err := eng.UpsertSource(ctx, SourceRecord{TenantID: "t1", SourceID: "s1", Version: 2, Content: "Symbols: posting.List\nversion 2"}); err != nil {
		t.Fatalf("upsert v2 failed: %v", err)
	}
	if _, err := eng.UpsertSource(ctx, SourceRecord{TenantID: "t1", SourceID: "s1", Version: 1, Content: "stale v1"}); !errors.Is(err, ErrStaleVersion) {
		t.Fatalf("expected ErrStaleVersion, got %v", err)
	}

	if _, err := eng.DeleteSource(ctx, "t1", "nonexistent"); !errors.Is(err, ErrSourceNotFound) {
		t.Fatalf("expected ErrSourceNotFound, got %v", err)
	}
	if _, err := eng.Search(ctx, Query{Text: "posting.List"}); !errors.Is(err, ErrEmptyTenant) {
		t.Fatalf("expected ErrEmptyTenant on search, got %v", err)
	}
	if _, err := eng.Search(ctx, Query{TenantID: "t1", Text: "  "}); !errors.Is(err, ErrInvalidQuery) {
		t.Fatalf("expected ErrInvalidQuery on blank query, got %v", err)
	}
	if _, err := eng.Search(ctx, Query{TenantID: "t1", Text: "hello", Mode: "bogus"}); !errors.Is(err, ErrInvalidQuery) {
		t.Fatalf("expected ErrInvalidQuery on invalid mode, got %v", err)
	}

	// DropDerivedIndexes must preserve canonical sources while enforcing ErrIndexesNotBuilt.
	eng.DropDerivedIndexes()
	if _, err := eng.Search(ctx, Query{TenantID: "t1", Text: "posting.List", Mode: ModeHybrid}); !errors.Is(err, ErrIndexesNotBuilt) {
		t.Fatalf("expected ErrIndexesNotBuilt after DropDerivedIndexes, got %v", err)
	}
	if _, err := eng.IndexDigest("t1"); !errors.Is(err, ErrIndexesNotBuilt) {
		t.Fatalf("expected ErrIndexesNotBuilt on IndexDigest, got %v", err)
	}
	src, ok := eng.GetSource("t1", "s1")
	if !ok || src.Version != 2 {
		t.Fatalf("canonical source must survive DropDerivedIndexes, got ok=%v src=%+v", ok, src)
	}
	if _, err := eng.RebuildAll(ctx); err != nil {
		t.Fatalf("RebuildAll failed: %v", err)
	}
	if _, err := eng.Search(ctx, Query{TenantID: "t1", Text: "posting.List", Mode: ModeHybrid}); err != nil {
		t.Fatalf("Search after RebuildAll failed: %v", err)
	}
}

func TestIncrementalDeltaPreservesUnaffectedAndRetractsOrphans(t *testing.T) {
	ctx := context.Background()
	eng := NewEngine(DefaultEngineConfig())

	docs := []SourceRecord{
		{
			SourceID: "doc/s1.md",
			Version:  1,
			TenantID: "t1",
			Kind:     KindDocument,
			Title:    "S1 Alpha",
			Content:  "Symbols: posting.List, UniqueAlphaNode\nS1 links posting.List with UniqueAlphaNode.",
		},
		{
			SourceID: "doc/s2.md",
			Version:  1,
			TenantID: "t1",
			Kind:     KindDocument,
			Title:    "S2 Shared",
			Content:  "Symbols: posting.List, SharedPeerNode\nS2 links posting.List with SharedPeerNode.",
		},
		{
			SourceID: "doc/s3.md",
			Version:  1,
			TenantID: "t1",
			Kind:     KindDocument,
			Title:    "S3 Disjoint",
			Content:  "Symbols: DisjointGammaNode, DisjointDeltaNode\nS3 is completely independent.",
		},
	}
	for _, d := range docs {
		if _, err := eng.UpsertSource(ctx, d); err != nil {
			t.Fatalf("upsert %s: %v", d.SourceID, err)
		}
	}

	// Update s1 v1 -> v2: retract UniqueAlphaNode, add ReplacementBetaNode, preserve s2 & s3.
	tel, err := eng.UpsertSource(ctx, SourceRecord{
		SourceID: "doc/s1.md",
		Version:  2,
		TenantID: "t1",
		Kind:     KindDocument,
		Title:    "S1 Beta",
		Content:  "Symbols: posting.List, ReplacementBetaNode\nS1 v2 links posting.List with ReplacementBetaNode.",
	})
	if err != nil {
		t.Fatalf("update s1 v2 failed: %v", err)
	}
	if tel.Operation != "update" || tel.RemovedEntities == 0 || tel.AddedEntities == 0 {
		t.Fatalf("unexpected update telemetry: %+v", tel)
	}
	if tel.SubgraphChurnRatio >= 1.0 {
		t.Fatalf("expected partial subgraph churn < 1.0 when s3 is disjoint, got %.3f", tel.SubgraphChurnRatio)
	}

	// UniqueAlphaNode must be completely gone.
	resAlpha, err := eng.Search(ctx, Query{TenantID: "t1", Text: "UniqueAlphaNode", Mode: ModeHybrid, TopK: 5, IncludeSubgraph: true})
	if err != nil {
		t.Fatalf("search alpha: %v", err)
	}
	if resultContainsToken(resAlpha, "uniquealphanode") {
		t.Fatalf("orphan entity UniqueAlphaNode leaked after incremental update: %+v", resAlpha)
	}

	// ReplacementBetaNode and DisjointGammaNode must be present.
	resBeta, err := eng.Search(ctx, Query{TenantID: "t1", Text: "ReplacementBetaNode", Mode: ModeHybrid, TopK: 5, IncludeSubgraph: true})
	if err != nil || !resultContainsToken(resBeta, "replacementbetanode") {
		t.Fatalf("expected ReplacementBetaNode after update, err=%v", err)
	}
	resGamma, err := eng.Search(ctx, Query{TenantID: "t1", Text: "DisjointGammaNode", Mode: ModeHybrid, TopK: 5, IncludeSubgraph: true})
	if err != nil || !resultContainsToken(resGamma, "disjointgammanode") {
		t.Fatalf("expected unaffected DisjointGammaNode to survive, err=%v", err)
	}

	// Delete s1: ReplacementBetaNode must be deleted, but shared posting.List must survive with ONLY doc/s2.md.
	delTel, err := eng.DeleteSource(ctx, "t1", "doc/s1.md")
	if err != nil {
		t.Fatalf("delete s1 failed: %v", err)
	}
	if delTel.RemovedEntities == 0 {
		t.Fatalf("expected orphaned entities removed on delete: %+v", delTel)
	}

	resBetaAfter, err := eng.Search(ctx, Query{TenantID: "t1", Text: "ReplacementBetaNode", Mode: ModeHybrid, TopK: 5, IncludeSubgraph: true})
	if err != nil {
		t.Fatalf("search beta after delete: %v", err)
	}
	if resultContainsToken(resBetaAfter, "replacementbetanode") {
		t.Fatalf("deleted source entity ReplacementBetaNode still present after DeleteSource")
	}

	entID, _ := canonicalEntityID(EntityTypeDecl, "posting.List")
	ent, ok := eng.GetEntity("t1", entID, nil)
	if !ok {
		t.Fatalf("shared entity posting.List should survive deletion of s1")
	}
	if len(ent.SourceIDs) != 1 || ent.SourceIDs[0] != "doc/s2.md" {
		t.Fatalf("expected posting.List provenance [doc/s2.md], got %v", ent.SourceIDs)
	}
}

func TestIncrementalEqualsFullRebuildBitExact(t *testing.T) {
	ctx := context.Background()
	eng := NewEngine(DefaultEngineConfig())
	rng := rand.New(rand.NewSource(1347))

	symbols := []string{"posting.List", "posting.Txn", "algo.MergeSorted", "codec.Decoder", "tok.Tokenizer", "dql.GraphQuery", "schema.ParsedSchema", "x.Keys"}
	versions := make(map[string]int64)
	active := make(map[string]bool)

	for step := 0; step < 40; step++ {
		idx := rng.Intn(12)
		sid := fmt.Sprintf("repo/mod_%02d.md", idx)

		if active[sid] && rng.Intn(5) == 0 {
			if _, err := eng.DeleteSource(ctx, "t1", sid); err != nil {
				t.Fatalf("step %d delete %s: %v", step, sid, err)
			}
			delete(active, sid)
			continue
		}

		versions[sid]++
		sA := symbols[rng.Intn(len(symbols))]
		sB := symbols[rng.Intn(len(symbols))]
		rec := SourceRecord{
			SourceID: sid,
			Version:  versions[sid],
			TenantID: "t1",
			Kind:     KindCommit,
			Title:    fmt.Sprintf("Commit step %d on %s", step, sid),
			Path:     sid,
			Content: fmt.Sprintf("Commit: COMMIT-%03d\nFiles: posting/list.go, algo/uidlist.go\nSymbols: %s, %s\nStep %d modifies %s and interacts with %s.",
				step, sA, sB, step, sA, sB),
		}
		if _, err := eng.UpsertSource(ctx, rec); err != nil {
			t.Fatalf("step %d upsert %s: %v", step, sid, err)
		}
		active[sid] = true
	}

	digestInc, err := eng.IndexDigest("t1")
	if err != nil {
		t.Fatalf("IndexDigest incremental: %v", err)
	}

	modes := []RetrievalMode{ModeOrdinaryHybrid, ModeLocal, ModeGlobal, ModeHybrid}
	beforeChunks := make(map[RetrievalMode][]string)
	for _, m := range modes {
		res, err := eng.Search(ctx, Query{TenantID: "t1", Text: "posting.List codec.Decoder MVCC", Mode: m, TopK: 5, BypassCache: true})
		if err != nil {
			t.Fatalf("search before rebuild (%s): %v", m, err)
		}
		for _, rc := range res.Chunks {
			beforeChunks[m] = append(beforeChunks[m], rc.Chunk.ChunkID)
		}
	}

	eng.DropDerivedIndexes()
	if _, err := eng.RebuildAll(ctx); err != nil {
		t.Fatalf("RebuildAll: %v", err)
	}

	digestReb, err := eng.IndexDigest("t1")
	if err != nil {
		t.Fatalf("IndexDigest rebuild: %v", err)
	}
	if digestInc != digestReb {
		t.Fatalf("bit-exact parity mismatch: incremental=%s vs rebuild=%s", digestInc, digestReb)
	}

	for _, m := range modes {
		res, err := eng.Search(ctx, Query{TenantID: "t1", Text: "posting.List codec.Decoder MVCC", Mode: m, TopK: 5, BypassCache: true})
		if err != nil {
			t.Fatalf("search after rebuild (%s): %v", m, err)
		}
		var after []string
		for _, rc := range res.Chunks {
			after = append(after, rc.Chunk.ChunkID)
		}
		if strings.Join(beforeChunks[m], ",") != strings.Join(after, ",") {
			t.Fatalf("mode %s ranking diverged after rebuild: before=%v after=%v", m, beforeChunks[m], after)
		}
	}
}

func TestEntityCanonicalizationAndDeduplicationTelemetry(t *testing.T) {
	ctx := context.Background()
	eng := NewEngine(DefaultEngineConfig())

	customExt := FuncExtractor(func(_ context.Context, src SourceRecord) (ExtractionResult, error) {
		return ExtractionResult{
			Entities: []ExtractedEntity{
				{Name: "github.com/hanzoai/graph/posting", Type: EntityPackage, Description: "Full import path"},
				{Name: "posting", Type: EntityPackage, Description: "Short package alias"},
				{Name: "BadgerDB", Type: EntityDependency, Description: "Legacy storage alias"},
				{Name: "zapdb", Type: EntityDependency, Description: "Canonical storage engine"},
				{Name: "*posting.List", Type: EntityTypeDecl, Description: "Pointer syntax"},
				{Name: "posting.List", Type: EntityTypeDecl, Description: "Canonical type syntax"},
			},
			Relations: []ExtractedRelation{
				{SourceName: "posting", SourceType: EntityPackage, TargetName: "zapdb", TargetType: EntityDependency, Type: RelImports, Description: "posting stores lists in zapdb"},
			},
		}, nil
	})

	cfg := DefaultEngineConfig()
	cfg.Extractor = customExt
	eng = NewEngine(cfg)

	tel, err := eng.UpsertSource(ctx, SourceRecord{
		SourceID: "doc/aliases.md",
		Version:  1,
		TenantID: "t1",
		Kind:     KindDocument,
		Content:  "Alias deduplication test document.",
	})
	if err != nil {
		t.Fatalf("upsert aliases: %v", err)
	}
	if tel.RawEntitiesExtracted != 6 {
		t.Fatalf("expected 6 raw entities, got %d", tel.RawEntitiesExtracted)
	}
	if tel.DuplicateEntitiesCollapsed != 3 {
		t.Fatalf("expected 3 duplicates collapsed (6->3 canonical), got %d", tel.DuplicateEntitiesCollapsed)
	}
	if tel.TotalEntities != 3 {
		t.Fatalf("expected 3 unique canonical entities, got %d", tel.TotalEntities)
	}
	if tel.DuplicateEntityRate < 0.49 || tel.DuplicateEntityRate > 0.51 {
		t.Fatalf("expected 0.50 duplicate entity rate, got %.4f", tel.DuplicateEntityRate)
	}
	st := eng.Stats("t1")
	if st.TotalCollapsedDupes != 3 || st.Entities != 3 {
		t.Fatalf("unexpected stats: %+v", st)
	}
}

func TestCacheInvalidationOnUpsertAndDelete(t *testing.T) {
	ctx := context.Background()
	eng := NewEngine(DefaultEngineConfig())

	if _, err := eng.UpsertSource(ctx, SourceRecord{
		SourceID: "doc/cache.md",
		Version:  1,
		TenantID: "t1",
		ACL:      []string{"role:eng"},
		Kind:     KindDocument,
		Content:  "Symbols: CacheSentinelAlpha\nVersion 1 cache sentinel text.",
	}); err != nil {
		t.Fatalf("upsert v1: %v", err)
	}

	q := Query{TenantID: "t1", Principals: []string{"role:eng"}, Text: "CacheSentinelAlpha", Mode: ModeHybrid, TopK: 3}
	r1, err := eng.Search(ctx, q)
	if err != nil || r1.CacheHit || !resultContainsToken(r1, "cachesentinelalpha") {
		t.Fatalf("expected cold hit with token, err=%v hit=%v", err, r1.CacheHit)
	}
	r2, err := eng.Search(ctx, q)
	if err != nil || !r2.CacheHit {
		t.Fatalf("expected warm cache hit on repeat query, err=%v hit=%v", err, r2.CacheHit)
	}

	// Update content & tighten ACL -> cache MUST invalidate immediately.
	tel, err := eng.UpsertSource(ctx, SourceRecord{
		SourceID: "doc/cache.md",
		Version:  2,
		TenantID: "t1",
		ACL:      []string{"role:security"},
		Kind:     KindDocument,
		Content:  "Symbols: CacheSentinelBeta\nVersion 2 restricted text.",
	})
	if err != nil || !tel.CacheInvalidated {
		t.Fatalf("expected cache invalidation on upsert, err=%v tel=%+v", err, tel)
	}
	r3, err := eng.Search(ctx, q)
	if err != nil || r3.CacheHit || len(r3.Chunks) != 0 {
		t.Fatalf("expected cache miss and 0 hits after ACL tightening, err=%v hit=%v chunks=%d", err, r3.CacheHit, len(r3.Chunks))
	}

	// Prime cache as security role, then delete source -> cache MUST invalidate and yield 0 hits.
	secQ := Query{TenantID: "t1", Principals: []string{"role:security"}, Text: "CacheSentinelBeta", Mode: ModeHybrid, TopK: 3}
	if _, err := eng.Search(ctx, secQ); err != nil {
		t.Fatalf("prime sec query: %v", err)
	}
	secCached, err := eng.Search(ctx, secQ)
	if err != nil || !secCached.CacheHit {
		t.Fatalf("expected cached security query hit")
	}
	delTel, err := eng.DeleteSource(ctx, "t1", "doc/cache.md")
	if err != nil || !delTel.CacheInvalidated {
		t.Fatalf("expected cache invalidation on delete, err=%v tel=%+v", err, delTel)
	}
	secAfterDel, err := eng.Search(ctx, secQ)
	if err != nil || secAfterDel.CacheHit || len(secAfterDel.Chunks) != 0 {
		t.Fatalf("expected cache miss and 0 hits after delete, err=%v hit=%v chunks=%d", err, secAfterDel.CacheHit, len(secAfterDel.Chunks))
	}
}

func TestDeepCopyImmutability(t *testing.T) {
	ctx := context.Background()
	eng := NewEngine(DefaultEngineConfig())

	// Include both empty non-nil metadata/ACL and populated metadata/ACL.
	inEmptyMeta := map[string]string{}
	inEmptyACL := []string{}
	if _, err := eng.UpsertSource(ctx, SourceRecord{
		SourceID: "doc/empty.md",
		Version:  1,
		TenantID: "t1",
		ACL:      inEmptyACL,
		Metadata: inEmptyMeta,
		Content:  "Symbols: EmptyMetaNode\nEmpty non-nil metadata test.",
	}); err != nil {
		t.Fatalf("upsert empty meta: %v", err)
	}
	// Mutate caller's map after upsert.
	inEmptyMeta["poisoned"] = "yes"

	gotEmpty, ok := eng.GetSource("t1", "doc/empty.md")
	if !ok || gotEmpty.Metadata == nil || len(gotEmpty.Metadata) != 0 {
		t.Fatalf("expected isolated empty non-nil metadata, got %#v", gotEmpty.Metadata)
	}
	if gotEmpty.ACL == nil || len(gotEmpty.ACL) != 0 {
		t.Fatalf("expected isolated empty non-nil ACL, got %#v", gotEmpty.ACL)
	}
	gotEmpty.Metadata["mutated_by_reader"] = "bad"

	gotEmpty2, _ := eng.GetSource("t1", "doc/empty.md")
	if _, leaked := gotEmpty2.Metadata["mutated_by_reader"]; leaked {
		t.Fatalf("reader mutation corrupted canonical SourceRecord metadata")
	}

	// Test SearchResult and cached SearchResult immutability.
	inACL := []string{"role:eng"}
	inMeta := map[string]string{"owner": "core"}
	if _, err := eng.UpsertSource(ctx, SourceRecord{
		SourceID: "doc/pop.md",
		Version:  1,
		TenantID: "t1",
		ACL:      inACL,
		Metadata: inMeta,
		Content:  "Symbols: DeepCopyEntity\nPopulated metadata test for DeepCopyEntity.",
	}); err != nil {
		t.Fatalf("upsert pop: %v", err)
	}
	inACL[0] = "role:tampered"
	inMeta["owner"] = "tampered"

	q := Query{TenantID: "t1", Principals: []string{"role:eng"}, Text: "DeepCopyEntity", Mode: ModeHybrid, TopK: 3, IncludeSubgraph: true}
	r1, err := eng.Search(ctx, q)
	if err != nil || len(r1.Chunks) == 0 {
		t.Fatalf("expected hit for role:eng despite caller mutating input ACL, err=%v chunks=%d", err, len(r1.Chunks))
	}
	if r1.Chunks[0].Chunk.Metadata["owner"] != "core" {
		t.Fatalf("expected metadata owner=core, got %q", r1.Chunks[0].Chunk.Metadata["owner"])
	}
	// Mutate returned SearchResult slices and maps.
	r1.Chunks[0].Chunk.Metadata["owner"] = "corrupted"
	r1.Chunks[0].Chunk.ACL[0] = "corrupted"
	if len(r1.Entities) > 0 && len(r1.Entities[0].Entity.SourceIDs) > 0 {
		r1.Entities[0].Entity.SourceIDs[0] = "corrupted"
	}

	// Second call hits cache and must still see pristine values.
	r2, err := eng.Search(ctx, q)
	if err != nil || !r2.CacheHit {
		t.Fatalf("expected cached hit, err=%v hit=%v", err, r2.CacheHit)
	}
	if r2.Chunks[0].Chunk.Metadata["owner"] != "core" || r2.Chunks[0].Chunk.ACL[0] != "role:eng" {
		t.Fatalf("caller mutation corrupted cached SearchResult: meta=%v acl=%v", r2.Chunks[0].Chunk.Metadata, r2.Chunks[0].Chunk.ACL)
	}
	if len(r2.Entities) > 0 && r2.Entities[0].Entity.SourceIDs[0] == "corrupted" {
		t.Fatalf("caller mutation corrupted cached entity SourceIDs")
	}
}

func TestStrictPreReturnTenantAndACLIsolation(t *testing.T) {
	ctx := context.Background()
	eng := NewEngine(DefaultEngineConfig())

	// Shared entity posting.List across (tenant-a public, tenant-a secret, tenant-b public).
	records := []SourceRecord{
		{
			SourceID: "shared/posting.md",
			Version:  1,
			TenantID: "tenant-a",
			ACL:      []string{"role:eng"},
			Kind:     KindDocument,
			Title:    "Alpha Public Posting Guide",
			Content:  "Symbols: posting.List, algo.MergeSorted\nPublic guide for posting.List and algo.MergeSorted.",
		},
		{
			SourceID: "secret/posting-cve.md",
			Version:  1,
			TenantID: "tenant-a",
			ACL:      []string{"role:sec"},
			Kind:     KindIncident,
			Title:    "Alpha Confidential Advisory",
			Content:  "Symbols: posting.List, SecretKeyVault\nCauses: posting.List\nOMEGA_SECRET_99 vulnerability inside SecretKeyVault affecting posting.List.",
		},
		{
			SourceID: "shared/posting.md",
			Version:  1,
			TenantID: "tenant-b",
			ACL:      []string{"role:eng"},
			Kind:     KindDocument,
			Title:    "Beta Private Posting Guide",
			Content:  "Symbols: posting.List, BetaPrivateVault\nBETA_ONLY_TOKEN_42 belongs strictly to tenant-b.",
		},
	}
	for _, r := range records {
		if _, err := eng.UpsertSource(ctx, r); err != nil {
			t.Fatalf("upsert %s/%s: %v", r.TenantID, r.SourceID, err)
		}
	}

	modes := []RetrievalMode{ModeOrdinaryHybrid, ModeLocal, ModeGlobal, ModeHybrid}

	// 1. Existence-oracle test: tenant-a [role:eng] searching for OMEGA_SECRET_99 must get ZERO results.
	for _, m := range modes {
		res, err := eng.Search(ctx, Query{
			TenantID:        "tenant-a",
			Principals:      []string{"role:eng"},
			Text:            "OMEGA_SECRET_99 SecretKeyVault",
			Mode:            m,
			TopK:            5,
			IncludeSubgraph: true,
			BypassCache:     true,
		})
		if err != nil {
			t.Fatalf("search (%s): %v", m, err)
		}
		if len(res.Chunks) != 0 || len(res.Entities) != 0 || len(res.Relations) != 0 {
			var names []string
			for _, re := range res.Entities {
				names = append(names, fmt.Sprintf("%s(%.3f,h=%d)", re.Entity.Name, re.Score, re.Hop))
			}
			t.Fatalf("mode %s leaked secret existence on OMEGA_SECRET_99: chunks=%d ents=%d (%v) rels=%d",
				m, len(res.Chunks), len(res.Entities), names, len(res.Relations))
		}
	}

	// 2. Shared-entity bridge test: tenant-a [role:eng] searching posting.List must NOT see secret description, SecretKeyVault, or tenant-b.
	for _, m := range modes {
		res, err := eng.Search(ctx, Query{
			TenantID:        "tenant-a",
			Principals:      []string{"role:eng"},
			Text:            "posting.List",
			Mode:            m,
			TopK:            5,
			IncludeSubgraph: true,
			BypassCache:     true,
		})
		if err != nil {
			t.Fatalf("search posting.List (%s): %v", m, err)
		}
		if containsAnyForbidden(res, []string{"omega_secret_99", "secret/posting-cve.md", "secretkeyvault", "beta_only_token_42", "betaprivatevault"}) {
			t.Fatalf("mode %s leaked restricted or cross-tenant data via shared entity posting.List: %+v", m, res)
		}
	}

	// 3. Direct GetEntity on shared posting.List as [role:eng] must filter provenance and descriptions.
	entID, _ := canonicalEntityID(EntityTypeDecl, "posting.List")
	entEng, ok := eng.GetEntity("tenant-a", entID, []string{"role:eng"})
	if !ok {
		t.Fatalf("expected role:eng to read authorized view of posting.List")
	}
	if len(entEng.SourceIDs) != 1 || entEng.SourceIDs[0] != "shared/posting.md" {
		t.Fatalf("expected only [shared/posting.md] in role:eng entity view, got %v", entEng.SourceIDs)
	}
	if strings.Contains(strings.ToLower(entEng.Description), "omega_secret_99") || strings.Contains(strings.ToLower(entEng.Description), "secret") {
		t.Fatalf("secret description leaked into role:eng GetEntity: %q", entEng.Description)
	}

	// 4. Authorized [role:sec] caller DOES see OMEGA_SECRET_99.
	resSec, err := eng.Search(ctx, Query{
		TenantID:        "tenant-a",
		Principals:      []string{"role:sec"},
		Text:            "OMEGA_SECRET_99 SecretKeyVault",
		Mode:            ModeHybrid,
		TopK:            5,
		IncludeSubgraph: true,
	})
	if err != nil || !resultContainsToken(resSec, "omega_secret_99") {
		t.Fatalf("expected role:sec to retrieve authorized secret doc, err=%v", err)
	}

	// 5. RequireExplicitACL=true denies empty-ACL docs to callers.
	strictCfg := DefaultEngineConfig()
	strictCfg.RequireExplicitACL = true
	strictEng := NewEngine(strictCfg)
	if _, err := strictEng.UpsertSource(ctx, SourceRecord{
		SourceID: "no-acl.md",
		Version:  1,
		TenantID: "t-strict",
		Content:  "Symbols: StrictOnlyNode\nNo ACL specified.",
	}); err != nil {
		t.Fatalf("upsert strict: %v", err)
	}
	strictRes, err := strictEng.Search(ctx, Query{TenantID: "t-strict", Principals: []string{"role:eng"}, Text: "StrictOnlyNode", Mode: ModeHybrid})
	if err != nil || len(strictRes.Chunks) != 0 {
		t.Fatalf("expected RequireExplicitACL=true to deny empty-ACL document, got %d chunks", len(strictRes.Chunks))
	}
}

func TestDualLevelRetrievalModes(t *testing.T) {
	ctx := context.Background()
	eng := NewEngine(DefaultEngineConfig())

	// 2-hop chain where query mentions INC-901, bridge ADR-902 links to target COMMIT-903.
	chain := []SourceRecord{
		{
			SourceID: "chain/inc-901.md",
			Version:  1,
			TenantID: "t1",
			Kind:     KindIncident,
			Title:    "INC-901: Compaction checksum mismatch",
			Content:  "Incident: INC-901\nRelated: ADR-902\nSymptoms: Checksum mismatch detected on compaction startup.",
		},
		{
			SourceID: "chain/adr-902.md",
			Version:  1,
			TenantID: "t1",
			Kind:     KindDecision,
			Title:    "ADR-902: Deterministic CRC64 block verifier",
			Content:  "Decision: ADR-902\nMitigates: INC-901\nRelated: COMMIT-903\nMandates CRC64 verification across posting blocks.",
		},
		{
			SourceID: "chain/commit-903.md",
			Version:  1,
			TenantID: "t1",
			Kind:     KindCommit,
			Title:    "COMMIT-903: Add VerifyCRC64Block in codec/codec.go",
			Content:  "Commit: COMMIT-903\nAuthor: zoe-storage\nDepends-On: ADR-902\nFiles: codec/codec.go\nSymbols: codec.VerifyCRC64Block\nAdds VerifyCRC64Block guard with zero allocation overhead.",
		},
	}
	for _, r := range chain {
		if _, err := eng.UpsertSource(ctx, r); err != nil {
			t.Fatalf("upsert %s: %v", r.SourceID, err)
		}
	}

	// Local 2-hop traversal starting from INC-901 must reach COMMIT-903 even though COMMIT-903 never mentions INC-901.
	locRes, err := eng.Search(ctx, Query{
		TenantID:        "t1",
		Text:            "Starting from INC-901, which commit and author added the block verifier?",
		Mode:            ModeLocal,
		TopK:            5,
		MaxHops:         2,
		IncludeSubgraph: true,
	})
	if err != nil {
		t.Fatalf("local search: %v", err)
	}
	if !resultContainsToken(locRes, "commit-903") || !resultContainsToken(locRes, "verifycrc64block") {
		t.Fatalf("expected ModeLocal 2-hop expansion to surface COMMIT-903 & VerifyCRC64Block, got chunks=%d trails=%d",
			len(locRes.Chunks), len(locRes.Trails))
	}

	// Global mode matches thematic relationships.
	globRes, err := eng.Search(ctx, Query{
		TenantID:        "t1",
		Text:            "How do architectural decisions mitigate checksum incidents across codec blocks?",
		Mode:            ModeGlobal,
		TopK:            5,
		IncludeSubgraph: true,
	})
	if err != nil || len(globRes.Chunks) == 0 || len(globRes.Relations) == 0 {
		t.Fatalf("expected ModeGlobal to return thematic relations & chunks, err=%v chunks=%d rels=%d",
			err, len(globRes.Chunks), len(globRes.Relations))
	}

	// Hybrid fuses both levels.
	hybRes, err := eng.Search(ctx, Query{
		TenantID:        "t1",
		Text:            "Trace INC-901 through ADR-902 to the implementing commit and codec verifier.",
		Mode:            ModeHybrid,
		TopK:            5,
		IncludeSubgraph: true,
	})
	if err != nil || !resultContainsToken(hybRes, "commit-903") {
		t.Fatalf("expected ModeHybrid to retrieve full chain, err=%v", err)
	}
}

func TestGoASTExtractorOnRealRepoFiles(t *testing.T) {
	ctx := context.Background()
	ext := NewDefaultExtractor(0.15, 0.60)

	for _, rel := range []string{"algo/uidlist.go", "posting/oracle.go"} {
		abs := filepath.Join("..", filepath.FromSlash(rel))
		raw, err := os.ReadFile(abs)
		if err != nil {
			t.Fatalf("read %s: %v", abs, err)
		}
		res, err := ext.Extract(ctx, SourceRecord{
			SourceID: rel,
			Version:  1,
			TenantID: "hanzo-graph",
			Kind:     KindCodeBlob,
			Path:     rel,
			Content:  string(raw),
		})
		if err != nil {
			t.Fatalf("extract %s: %v", rel, err)
		}
		if len(res.Entities) < 5 || len(res.Relations) < 5 {
			t.Fatalf("expected rich Go AST extraction for %s, got ents=%d rels=%d", rel, len(res.Entities), len(res.Relations))
		}
		if res.Metrics.InputTokens <= 0 || res.Metrics.EstimatedCostUSD <= 0 {
			t.Fatalf("expected positive extraction metrics for %s: %+v", rel, res.Metrics)
		}
	}
}

func TestConcurrentReadWriteRebuildRaceSafety(t *testing.T) {
	ctx := context.Background()
	eng := NewEngine(DefaultEngineConfig())

	for i := 0; i < 10; i++ {
		if _, err := eng.UpsertSource(ctx, SourceRecord{
			SourceID: fmt.Sprintf("conc/doc_%d.md", i),
			Version:  1,
			TenantID: "t-race",
			ACL:      []string{"role:eng"},
			Kind:     KindDocument,
			Content:  fmt.Sprintf("Symbols: posting.List, Node_%d\nInitial concurrent doc %d.", i, i),
		}); err != nil {
			t.Fatalf("seed %d: %v", i, err)
		}
	}

	var wg sync.WaitGroup
	errCh := make(chan error, 64)

	// 4 concurrent reader goroutines across all 4 modes.
	modes := []RetrievalMode{ModeOrdinaryHybrid, ModeLocal, ModeGlobal, ModeHybrid}
	for r := 0; r < 4; r++ {
		mode := modes[r]
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 40; i++ {
				_, err := eng.Search(ctx, Query{
					TenantID:        "t-race",
					Principals:      []string{"role:eng"},
					Text:            fmt.Sprintf("posting.List Node_%d", i%10),
					Mode:            mode,
					TopK:            5,
					IncludeSubgraph: true,
				})
				if err != nil && !errors.Is(err, ErrIndexesNotBuilt) {
					errCh <- fmt.Errorf("reader %s iter %d: %w", mode, i, err)
					return
				}
				_ = eng.Stats("t-race")
				_, _ = eng.GetSource("t-race", fmt.Sprintf("conc/doc_%d.md", i%10))
			}
		}()
	}

	// 2 concurrent writer goroutines performing incremental updates and deletes/re-inserts.
	for w := 0; w < 2; w++ {
		wid := w
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 20; i++ {
				sid := fmt.Sprintf("conc/writer_%d_%d.md", wid, i%4)
				ver := int64(i/4 + 1)
				if _, err := eng.UpsertSource(ctx, SourceRecord{
					SourceID: sid,
					Version:  ver,
					TenantID: "t-race",
					ACL:      []string{"role:eng"},
					Kind:     KindCommit,
					Content:  fmt.Sprintf("Commit: COMMIT-%d-%d\nSymbols: posting.List, WriterSym_%d_%d\nUpdate %d", wid, i, wid, i, i),
				}); err != nil {
					errCh <- fmt.Errorf("writer %d iter %d upsert: %w", wid, i, err)
					return
				}
				if i%5 == 4 {
					if _, err := eng.DeleteSource(ctx, "t-race", sid); err != nil && !errors.Is(err, ErrSourceNotFound) {
						errCh <- fmt.Errorf("writer %d iter %d delete: %w", wid, i, err)
						return
					}
				}
			}
		}()
	}

	// 1 concurrent rebuilder goroutine.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 4; i++ {
			if _, err := eng.RebuildAll(ctx); err != nil {
				errCh <- fmt.Errorf("rebuild %d: %w", i, err)
				return
			}
		}
	}()

	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Fatal(err)
	}
}

func TestUpsertAndDeleteWhileIndexesDropped(t *testing.T) {
	ctx := context.Background()
	eng := NewEngine(DefaultEngineConfig())

	if _, err := eng.UpsertSource(ctx, SourceRecord{
		SourceID: "d1.md",
		Version:  1,
		TenantID: "t1",
		Content:  "Symbols: DroppedAlphaNode\nInitial doc 1.",
	}); err != nil {
		t.Fatalf("upsert d1: %v", err)
	}
	if _, err := eng.UpsertSource(ctx, SourceRecord{
		SourceID: "d2.md",
		Version:  1,
		TenantID: "t1",
		Content:  "Symbols: DroppedBetaNode\nInitial doc 2.",
	}); err != nil {
		t.Fatalf("upsert d2: %v", err)
	}

	eng.DropDerivedIndexes()

	// Mutate canonical store while derived indexes are dropped: derivedBuilt must stay false.
	if _, err := eng.DeleteSource(ctx, "t1", "d1.md"); err != nil {
		t.Fatalf("delete d1 while dropped: %v", err)
	}
	if _, err := eng.UpsertSource(ctx, SourceRecord{
		SourceID: "d3.md",
		Version:  1,
		TenantID: "t1",
		Content:  "Symbols: DroppedGammaNode\nAdded while dropped.",
	}); err != nil {
		t.Fatalf("upsert d3 while dropped: %v", err)
	}
	if _, err := eng.Search(ctx, Query{TenantID: "t1", Text: "DroppedGammaNode", Mode: ModeHybrid}); !errors.Is(err, ErrIndexesNotBuilt) {
		t.Fatalf("expected ErrIndexesNotBuilt while still dropped, got %v", err)
	}

	if _, err := eng.RebuildAll(ctx); err != nil {
		t.Fatalf("RebuildAll: %v", err)
	}
	rDel, err := eng.Search(ctx, Query{TenantID: "t1", Text: "DroppedAlphaNode", Mode: ModeHybrid, IncludeSubgraph: true})
	if err != nil || resultContainsToken(rDel, "droppedalphanode") {
		t.Fatalf("deleted source d1 should not reappear after RebuildAll, err=%v", err)
	}
	rNew, err := eng.Search(ctx, Query{TenantID: "t1", Text: "DroppedGammaNode", Mode: ModeHybrid, IncludeSubgraph: true})
	if err != nil || !resultContainsToken(rNew, "droppedgammanode") {
		t.Fatalf("expected d3 added while dropped to appear after RebuildAll, err=%v", err)
	}
}

func TestEntityRelationListSourcesDeepCopy(t *testing.T) {
	ctx := context.Background()
	eng := NewEngine(DefaultEngineConfig())

	if _, err := eng.UpsertSource(ctx, SourceRecord{
		SourceID: "copy/doc.md",
		Version:  1,
		TenantID: "t1",
		ACL:      []string{"role:eng"},
		Kind:     KindCommit,
		Metadata: map[string]string{"env": "prod"},
		Content:  "Commit: COMMIT-77\nFiles: posting/list.go\nSymbols: posting.List\nCopy isolation check.",
	}); err != nil {
		t.Fatalf("upsert: %v", err)
	}

	// Mutate ListSources return value.
	list := eng.ListSources("t1")
	if len(list) != 1 {
		t.Fatalf("expected 1 source, got %d", len(list))
	}
	list[0].Metadata["env"] = "tampered"
	list[0].ACL[0] = "tampered"

	checkSrc, _ := eng.GetSource("t1", "copy/doc.md")
	if checkSrc.Metadata["env"] != "prod" || checkSrc.ACL[0] != "role:eng" {
		t.Fatalf("ListSources mutation leaked into canonical store: %+v", checkSrc)
	}

	// Mutate GetEntity and GetRelation return values.
	entID, _ := canonicalEntityID(EntityFile, "posting/list.go")
	ent, ok := eng.GetEntity("t1", entID, []string{"role:eng"})
	if !ok || len(ent.SourceIDs) == 0 {
		t.Fatalf("expected entity posting/list.go")
	}
	ent.SourceIDs[0] = "tampered"
	if len(ent.Keywords) > 0 {
		ent.Keywords[0] = "tampered"
	}

	ent2, _ := eng.GetEntity("t1", entID, []string{"role:eng"})
	if ent2.SourceIDs[0] != "copy/doc.md" {
		t.Fatalf("GetEntity mutation leaked into derived graph: %v", ent2.SourceIDs)
	}

	commitID, _ := canonicalEntityID(EntityCommit, "copy/doc.md")
	relID := canonicalRelationID(commitID, RelModifies, entID)
	rel, ok := eng.GetRelation("t1", relID, []string{"role:eng"})
	if !ok || len(rel.SourceIDs) == 0 {
		t.Fatalf("expected relation COMMIT-77 -> posting/list.go")
	}
	rel.SourceIDs[0] = "tampered"
	rel2, _ := eng.GetRelation("t1", relID, []string{"role:eng"})
	if rel2.SourceIDs[0] != "copy/doc.md" {
		t.Fatalf("GetRelation mutation leaked into derived graph: %v", rel2.SourceIDs)
	}
}

func TestBenchmarkHarnessEndToEnd(t *testing.T) {
	ctx := context.Background()
	cfg := BenchmarkConfig{
		RepoRoot: "..",
		TopK:     5,
		Iters:    2,
		Seed:     1347,
	}
	rep, err := RunBenchmarkSuite(ctx, cfg)
	if err != nil {
		t.Fatalf("RunBenchmarkSuite: %v", err)
	}
	if rep.Corpus.RealGoSourceFiles < 50 {
		t.Fatalf("expected >= 50 real Go files from repo, got %d", rep.Corpus.RealGoSourceFiles)
	}
	if !rep.Incremental.BitExactEquivalence {
		t.Fatalf("expected bit-exact equivalence between incremental and rebuild: %s vs %s",
			rep.Incremental.DigestAfterIncremental, rep.Incremental.DigestAfterFullRebuild)
	}
	if !rep.Incremental.DropEnforcesNotBuiltErr {
		t.Fatalf("expected DropDerivedIndexes to enforce ErrIndexesNotBuilt")
	}
	if rep.Incremental.SpeedupP50 <= 10.0 {
		t.Fatalf("expected >10x incremental speedup, got %.2fx", rep.Incremental.SpeedupP50)
	}
	if !rep.Freshness.CacheInvalidatedOnEdit || rep.Freshness.UpdatedRecallRate != 1.0 || rep.Freshness.DeletedGhostRecallRate != 0.0 {
		t.Fatalf("freshness check failed: %+v", rep.Freshness)
	}
	if rep.Security.LightRAGLeakCount != 0 || rep.Security.UnfilteredControlLeaks == 0 {
		t.Fatalf("security isolation check failed: %+v", rep.Security)
	}
	if rep.Corpus.DuplicateEntityRatePct <= 0 || rep.Corpus.RelationsPerDollar <= 0 {
		t.Fatalf("expected positive deduplication and cost telemetry: %+v", rep.Corpus)
	}
}
