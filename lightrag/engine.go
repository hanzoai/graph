/*
 * SPDX-FileCopyrightText: © 2026 Hanzo AI, Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

package lightrag

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"time"
)

type sourceKey struct {
	tenantID string
	sourceID string
}

type entityContrib struct {
	name         string
	descriptions []string
	keywords     []string
	aliases      []string
	weight       float64
	mentions     int
}

type storedEntity struct {
	id           string
	canonicalKey string
	typ          EntityType
	contribs     map[string]entityContrib // keyed by sourceID

	// Derived merged fields recomputed deterministically in sorted sourceID order.
	name         string
	description  string
	keywords     []string
	aliases      []string
	sourceIDs    []string
	mentionCount int
	weight       float64
}

func (se *storedEntity) recompute() {
	if len(se.contribs) == 0 {
		se.name = ""
		se.description = ""
		se.keywords = nil
		se.aliases = nil
		se.sourceIDs = nil
		se.mentionCount = 0
		se.weight = 0
		return
	}
	sids := make([]string, 0, len(se.contribs))
	for sid := range se.contribs {
		sids = append(sids, sid)
	}
	sort.Strings(sids)
	se.sourceIDs = sids

	var bestName string
	var descParts []string
	seenDesc := make(map[string]struct{})
	seenKW := make(map[string]struct{})
	seenAlias := make(map[string]struct{})
	var kws []string
	var aliases []string
	mentions := 0
	weight := 0.0

	for _, sid := range sids {
		c := se.contribs[sid]
		if bestName == "" || (c.name != "" && len(c.name) < len(bestName)) {
			bestName = c.name
		}
		for _, d := range c.descriptions {
			d = strings.TrimSpace(d)
			if d == "" {
				continue
			}
			if _, dup := seenDesc[d]; !dup {
				seenDesc[d] = struct{}{}
				descParts = append(descParts, d)
			}
		}
		for _, kw := range c.keywords {
			kw = strings.TrimSpace(kw)
			if kw == "" {
				continue
			}
			kl := strings.ToLower(kw)
			if _, dup := seenKW[kl]; !dup {
				seenKW[kl] = struct{}{}
				kws = append(kws, kw)
			}
		}
		for _, al := range c.aliases {
			al = strings.TrimSpace(al)
			if al == "" {
				continue
			}
			alNorm := normalizeAliasToken(al)
			if alNorm == "" {
				continue
			}
			if _, dup := seenAlias[alNorm]; !dup {
				seenAlias[alNorm] = struct{}{}
				aliases = append(aliases, al)
			}
		}
		mentions += c.mentions
		weight += c.weight
	}
	sort.Strings(kws)
	sort.Strings(aliases)

	if len(descParts) > 6 {
		descParts = descParts[:6]
	}
	se.name = bestName
	se.description = strings.Join(descParts, "; ")
	se.keywords = kws
	se.aliases = aliases
	se.mentionCount = mentions
	se.weight = weight
}

func (se *storedEntity) indexText() string {
	var b strings.Builder
	b.WriteString(se.name)
	b.WriteByte(' ')
	b.WriteString(string(se.typ))
	if len(se.aliases) > 0 {
		b.WriteByte(' ')
		b.WriteString(strings.Join(se.aliases, " "))
	}
	if len(se.keywords) > 0 {
		b.WriteByte(' ')
		b.WriteString(strings.Join(se.keywords, " "))
	}
	if se.description != "" {
		b.WriteByte(' ')
		b.WriteString(se.description)
	}
	return b.String()
}

type relationContrib struct {
	descriptions []string
	keywords     []string
	weight       float64
}

type storedRelation struct {
	id          string
	srcEntityID string
	dstEntityID string
	relType     RelationType
	contribs    map[string]relationContrib // keyed by sourceID

	// Derived merged fields recomputed deterministically in sorted sourceID order.
	description string
	keywords    []string
	sourceIDs   []string
	weight      float64
}

func (sr *storedRelation) recompute() {
	if len(sr.contribs) == 0 {
		sr.description = ""
		sr.keywords = nil
		sr.sourceIDs = nil
		sr.weight = 0
		return
	}
	sids := make([]string, 0, len(sr.contribs))
	for sid := range sr.contribs {
		sids = append(sids, sid)
	}
	sort.Strings(sids)
	sr.sourceIDs = sids

	var descParts []string
	seenDesc := make(map[string]struct{})
	seenKW := make(map[string]struct{})
	var kws []string
	weight := 0.0

	for _, sid := range sids {
		c := sr.contribs[sid]
		for _, d := range c.descriptions {
			d = strings.TrimSpace(d)
			if d == "" {
				continue
			}
			if _, dup := seenDesc[d]; !dup {
				seenDesc[d] = struct{}{}
				descParts = append(descParts, d)
			}
		}
		for _, kw := range c.keywords {
			kw = strings.TrimSpace(kw)
			if kw == "" {
				continue
			}
			kl := strings.ToLower(kw)
			if _, dup := seenKW[kl]; !dup {
				seenKW[kl] = struct{}{}
				kws = append(kws, kw)
			}
		}
		weight += c.weight
	}
	sort.Strings(kws)
	if len(descParts) > 5 {
		descParts = descParts[:5]
	}
	sr.description = strings.Join(descParts, "; ")
	sr.keywords = kws
	sr.weight = weight
}

func (sr *storedRelation) indexText(srcName, dstName string) string {
	var b strings.Builder
	b.WriteString(srcName)
	b.WriteByte(' ')
	b.WriteString(string(sr.relType))
	b.WriteByte(' ')
	b.WriteString(dstName)
	if len(sr.keywords) > 0 {
		b.WriteByte(' ')
		b.WriteString(strings.Join(sr.keywords, " "))
	}
	if sr.description != "" {
		b.WriteByte(' ')
		b.WriteString(sr.description)
	}
	return b.String()
}

type sourceProvenance struct {
	sourceID     string
	version      int64
	chunkIDs     []string
	entityIDs    []string
	relationIDs  []string
	rawEntities  int
	rawRelations int
	extractCost  float64
}

type tenantIndex struct {
	tenantID      string
	generation    uint64
	chunks        map[string]Chunk
	chunkLex      *bm25Index
	chunkVec      *vectorIndex
	entities      map[string]*storedEntity
	entityLex     *bm25Index
	entityVec     *vectorIndex
	aliasToEntity map[string]map[string]struct{}
	relations     map[string]*storedRelation
	relLex        *bm25Index
	relVec        *vectorIndex
	adj           map[string]map[string]struct{} // entityID -> set of relationIDs
	sourceProv    map[string]*sourceProvenance
	queryCache    map[string]SearchResult
}

func newTenantIndex(tenantID string, dim int) *tenantIndex {
	return &tenantIndex{
		tenantID:      tenantID,
		chunks:        make(map[string]Chunk),
		chunkLex:      newBM25Index(),
		chunkVec:      newVectorIndex(dim),
		entities:      make(map[string]*storedEntity),
		entityLex:     newBM25Index(),
		entityVec:     newVectorIndex(dim),
		aliasToEntity: make(map[string]map[string]struct{}),
		relations:     make(map[string]*storedRelation),
		relLex:        newBM25Index(),
		relVec:        newVectorIndex(dim),
		adj:           make(map[string]map[string]struct{}),
		sourceProv:    make(map[string]*sourceProvenance),
		queryCache:    make(map[string]SearchResult),
	}
}

func (t *tenantIndex) invalidateCache() bool {
	hadEntries := len(t.queryCache) > 0
	if hadEntries {
		clear(t.queryCache)
	}
	t.generation++
	return hadEntries
}

func (t *tenantIndex) registerEntityAliases(se *storedEntity) {
	register := func(raw string) {
		norm := normalizeAliasToken(raw)
		if norm == "" {
			return
		}
		if t.aliasToEntity[norm] == nil {
			t.aliasToEntity[norm] = make(map[string]struct{})
		}
		t.aliasToEntity[norm][se.id] = struct{}{}
	}
	register(se.name)
	for _, al := range se.aliases {
		register(al)
	}
}

func (t *tenantIndex) unregisterEntityAliases(se *storedEntity) {
	unregister := func(raw string) {
		norm := normalizeAliasToken(raw)
		if norm == "" {
			return
		}
		if set := t.aliasToEntity[norm]; set != nil {
			delete(set, se.id)
			if len(set) == 0 {
				delete(t.aliasToEntity, norm)
			}
		}
	}
	unregister(se.name)
	for _, al := range se.aliases {
		unregister(al)
	}
}

func (t *tenantIndex) addAdj(entID, relID string) {
	if t.adj[entID] == nil {
		t.adj[entID] = make(map[string]struct{})
	}
	t.adj[entID][relID] = struct{}{}
}

func (t *tenantIndex) removeAdj(entID, relID string) {
	if set := t.adj[entID]; set != nil {
		delete(set, relID)
		if len(set) == 0 {
			delete(t.adj, entID)
		}
	}
}

// Engine is the thread-safe LightRAG incremental entity-relationship indexing
// and dual-level retrieval engine.
type Engine struct {
	mu           sync.RWMutex
	cfg          EngineConfig
	sources      map[sourceKey]SourceRecord
	tenants      map[string]*tenantIndex
	derivedBuilt bool
}

// NewEngine creates a new Engine with validated default settings.
func NewEngine(cfg EngineConfig) *Engine {
	cfg = cfg.withDefaults()
	return &Engine{
		cfg:          cfg,
		sources:      make(map[sourceKey]SourceRecord),
		tenants:      make(map[string]*tenantIndex),
		derivedBuilt: true,
	}
}

// UpsertSource ingests or incrementally updates a canonical SourceRecord.
// When derived indexes are active, only the subgraph (chunks, entities, and
// relationships) attributed to (TenantID, SourceID) is retracted, re-extracted,
// deduplicated, merged, and re-indexed; unaffected nodes/edges remain untouched
// and the tenant query cache is invalidated atomically.
func (e *Engine) UpsertSource(ctx context.Context, rec SourceRecord) (UpdateTelemetry, error) {
	start := time.Now()
	if err := ctx.Err(); err != nil {
		return UpdateTelemetry{}, err
	}
	if err := validateSource(rec); err != nil {
		return UpdateTelemetry{}, err
	}
	cloned := rec.Clone()
	if cloned.Kind == "" {
		cloned.Kind = KindDocument
	}

	// Perform deterministic extraction and chunking outside the write lock so
	// concurrent readers are never stalled during extraction.
	extracted, err := e.cfg.Extractor.Extract(ctx, cloned)
	if err != nil {
		return UpdateTelemetry{}, err
	}
	chunks := splitChunks(cloned, e.cfg.ChunkSizeChars, e.cfg.ChunkOverlapChars)

	e.mu.Lock()
	defer e.mu.Unlock()

	key := sourceKey{tenantID: cloned.TenantID, sourceID: cloned.SourceID}
	existing, existed := e.sources[key]
	if existed && cloned.Version < existing.Version {
		return UpdateTelemetry{}, fmt.Errorf("%w: got v%d < stored v%d for %s",
			ErrStaleVersion, cloned.Version, existing.Version, cloned.SourceID)
	}

	e.sources[key] = cloned

	op := "insert"
	if existed {
		op = "update"
	}

	if !e.derivedBuilt {
		return UpdateTelemetry{
			Operation:           op,
			TenantID:            cloned.TenantID,
			SourceID:            cloned.SourceID,
			Version:             cloned.Version,
			Duration:            time.Since(start),
			ExtractionTokensIn:  extracted.Metrics.InputTokens,
			ExtractionTokensOut: extracted.Metrics.OutputTokens,
			EstimatedCostUSD:    extracted.Metrics.EstimatedCostUSD,
			CacheInvalidated:    true,
		}, nil
	}

	t := e.getOrCreateTenantLocked(cloned.TenantID)
	tel := e.applyIncrementalDeltaLocked(t, cloned, chunks, extracted, op, start)
	return tel, nil
}

// DeleteSource removes a canonical SourceRecord and incrementally retracts ONLY
// its chunks, entity contributions, and relationship contributions from the
// tenant's derived graph and lexical/vector projections.
func (e *Engine) DeleteSource(ctx context.Context, tenantID, sourceID string) (UpdateTelemetry, error) {
	start := time.Now()
	if err := ctx.Err(); err != nil {
		return UpdateTelemetry{}, err
	}
	if strings.TrimSpace(tenantID) == "" {
		return UpdateTelemetry{}, ErrEmptyTenant
	}
	if strings.TrimSpace(sourceID) == "" {
		return UpdateTelemetry{}, ErrInvalidSource
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	key := sourceKey{tenantID: tenantID, sourceID: sourceID}
	prev, ok := e.sources[key]
	if !ok {
		return UpdateTelemetry{}, ErrSourceNotFound
	}
	delete(e.sources, key)

	if !e.derivedBuilt {
		return UpdateTelemetry{
			Operation:        "delete",
			TenantID:         tenantID,
			SourceID:         sourceID,
			Version:          prev.Version,
			Duration:         time.Since(start),
			CacheInvalidated: true,
		}, nil
	}

	t := e.tenants[tenantID]
	if t == nil {
		return UpdateTelemetry{
			Operation:        "delete",
			TenantID:         tenantID,
			SourceID:         sourceID,
			Version:          prev.Version,
			Duration:         time.Since(start),
			CacheInvalidated: false,
		}, nil
	}

	preTotal := len(t.entities) + len(t.relations)
	touchedEnt := make(map[string]struct{})
	touchedRel := make(map[string]struct{})
	removedEnt, updatedEnt, removedRel, updatedRel := retractSourceLocked(t, sourceID, touchedEnt, touchedRel)
	cacheInv := t.invalidateCache()

	affectedEnt := len(touchedEnt)
	affectedRel := len(touchedRel)
	denom := preTotal
	if cur := len(t.entities) + len(t.relations); cur > denom {
		denom = cur
	}
	churn := 0.0
	if denom > 0 {
		churn = float64(affectedEnt+affectedRel) / float64(denom)
	}

	return UpdateTelemetry{
		Operation:          "delete",
		TenantID:           tenantID,
		SourceID:           sourceID,
		Version:            prev.Version,
		Duration:           time.Since(start),
		UpdatedEntities:    updatedEnt,
		RemovedEntities:    removedEnt,
		AffectedEntities:   affectedEnt,
		UpdatedRelations:   updatedRel,
		RemovedRelations:   removedRel,
		AffectedRelations:  affectedRel,
		TotalEntities:      len(t.entities),
		TotalRelations:     len(t.relations),
		TotalChunks:        len(t.chunks),
		SubgraphChurnRatio: churn,
		CacheInvalidated:   cacheInv,
	}, nil
}

// DropDerivedIndexes discards all derived graphs, chunks, lexical postings,
// dense vectors, alias tables, and query caches while leaving every canonical
// SourceRecord completely intact. Subsequent Search calls return ErrIndexesNotBuilt
// until RebuildAll succeeds.
func (e *Engine) DropDerivedIndexes() {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.tenants = make(map[string]*tenantIndex)
	e.derivedBuilt = false
}

// RebuildAll deterministically reconstructs all derived projections from the
// canonical source store in sorted (TenantID, SourceID) order.
func (e *Engine) RebuildAll(ctx context.Context) (UpdateTelemetry, error) {
	start := time.Now()
	if err := ctx.Err(); err != nil {
		return UpdateTelemetry{}, err
	}

	// Snapshot canonical sources in deterministic order so extraction can run
	// cleanly while holding no stale references.
	e.mu.RLock()
	keys := make([]sourceKey, 0, len(e.sources))
	for k := range e.sources {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].tenantID != keys[j].tenantID {
			return keys[i].tenantID < keys[j].tenantID
		}
		return keys[i].sourceID < keys[j].sourceID
	})
	snapshot := make([]SourceRecord, len(keys))
	for i, k := range keys {
		snapshot[i] = e.sources[k].Clone()
	}
	e.mu.RUnlock()

	type prepared struct {
		src       SourceRecord
		chunks    []Chunk
		extracted ExtractionResult
	}
	items := make([]prepared, 0, len(snapshot))
	for _, src := range snapshot {
		if err := ctx.Err(); err != nil {
			return UpdateTelemetry{}, err
		}
		ext, err := e.cfg.Extractor.Extract(ctx, src)
		if err != nil {
			return UpdateTelemetry{}, err
		}
		items = append(items, prepared{
			src:       src,
			chunks:    splitChunks(src, e.cfg.ChunkSizeChars, e.cfg.ChunkOverlapChars),
			extracted: ext,
		})
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	// Reconcile against any canonical mutations that occurred while extracting.
	e.tenants = make(map[string]*tenantIndex)
	totalRawEnt := 0
	totalDupes := 0
	totalRawRel := 0
	totalTokIn := 0
	totalTokOut := 0
	totalCost := 0.0

	preparedMap := make(map[sourceKey]prepared, len(items))
	for _, it := range items {
		preparedMap[sourceKey{tenantID: it.src.TenantID, sourceID: it.src.SourceID}] = it
	}

	curKeys := make([]sourceKey, 0, len(e.sources))
	for k := range e.sources {
		curKeys = append(curKeys, k)
	}
	sort.Slice(curKeys, func(i, j int) bool {
		if curKeys[i].tenantID != curKeys[j].tenantID {
			return curKeys[i].tenantID < curKeys[j].tenantID
		}
		return curKeys[i].sourceID < curKeys[j].sourceID
	})

	for _, k := range curKeys {
		curSrc := e.sources[k]
		prep, ok := preparedMap[k]
		if !ok || prep.src.Version != curSrc.Version || prep.src.Content != curSrc.Content {
			ext, err := e.cfg.Extractor.Extract(ctx, curSrc)
			if err != nil {
				return UpdateTelemetry{}, err
			}
			prep = prepared{
				src:       curSrc,
				chunks:    splitChunks(curSrc, e.cfg.ChunkSizeChars, e.cfg.ChunkOverlapChars),
				extracted: ext,
			}
		}
		t := e.getOrCreateTenantLocked(curSrc.TenantID)
		subTel := e.applyIncrementalDeltaLocked(t, curSrc, prep.chunks, prep.extracted, "rebuild", start)
		totalRawEnt += subTel.RawEntitiesExtracted
		totalDupes += subTel.DuplicateEntitiesCollapsed
		totalRawRel += subTel.RawRelationsExtracted
		totalTokIn += subTel.ExtractionTokensIn
		totalTokOut += subTel.ExtractionTokensOut
		totalCost += subTel.EstimatedCostUSD
	}

	e.derivedBuilt = true

	totalEnt, totalRel, totalChunks := 0, 0, 0
	for _, t := range e.tenants {
		totalEnt += len(t.entities)
		totalRel += len(t.relations)
		totalChunks += len(t.chunks)
	}
	dupeRate := 0.0
	if totalRawEnt > 0 {
		dupeRate = float64(totalDupes) / float64(totalRawEnt)
	}

	return UpdateTelemetry{
		Operation:                  "rebuild",
		Duration:                   time.Since(start),
		AddedEntities:              totalEnt,
		AffectedEntities:           totalEnt,
		AddedRelations:             totalRel,
		AffectedRelations:          totalRel,
		TotalEntities:              totalEnt,
		TotalRelations:             totalRel,
		TotalChunks:                totalChunks,
		SubgraphChurnRatio:         1.0,
		RawEntitiesExtracted:       totalRawEnt,
		DuplicateEntitiesCollapsed: totalDupes,
		DuplicateEntityRate:        dupeRate,
		RawRelationsExtracted:      totalRawRel,
		UniqueRelationsMerged:      totalRel,
		ExtractionTokensIn:         totalTokIn,
		ExtractionTokensOut:        totalTokOut,
		EstimatedCostUSD:           totalCost,
		CacheInvalidated:           true,
	}, nil
}

// applyIncrementalDeltaLocked performs the O(|ΔG|) subgraph delta retraction and
// deterministic re-indexing for a single source under e.mu.Lock().
func (e *Engine) applyIncrementalDeltaLocked(
	t *tenantIndex,
	src SourceRecord,
	chunks []Chunk,
	extracted ExtractionResult,
	op string,
	start time.Time,
) UpdateTelemetry {
	preGraphSize := len(t.entities) + len(t.relations)

	// 1. Retract ONLY prior subgraph contributions of src.SourceID (if updating).
	touchedEnt := make(map[string]struct{})
	touchedRel := make(map[string]struct{})
	removedEnt, retractUpdatedEnt, removedRel, retractUpdatedRel := retractSourceLocked(t, src.SourceID, touchedEnt, touchedRel)

	// 2. Insert new chunks into chunk store, chunk BM25, and chunk vector table.
	chunkIDs := make([]string, 0, len(chunks))
	for _, ch := range chunks {
		cClone := ch.Clone()
		t.chunks[cClone.ChunkID] = cClone
		chunkText := strings.TrimSpace(cClone.Title + " " + cClone.Path + "\n" + cClone.Text)
		t.chunkLex.Add(cClone.ChunkID, chunkText)
		t.chunkVec.Set(cClone.ChunkID, chunkText)
		chunkIDs = append(chunkIDs, cClone.ChunkID)
	}

	// 3. Deduplicate extracted entities within this source into per-entity contributions.
	type pendingEnt struct {
		id           string
		canonicalKey string
		typ          EntityType
		contrib      entityContrib
	}
	entMap := make(map[string]*pendingEnt)
	rawEntCount := len(extracted.Entities)
	intraDupes := 0

	addPendingEnt := func(raw ExtractedEntity) string {
		if strings.TrimSpace(raw.Name) == "" {
			return ""
		}
		typ := raw.Type
		if typ == "" {
			typ = EntityConcept
		}
		eid, ckey := canonicalEntityID(typ, raw.Name)
		pe, exists := entMap[eid]
		if exists {
			intraDupes++
			if raw.Description != "" {
				pe.contrib.descriptions = append(pe.contrib.descriptions, raw.Description)
			}
			pe.contrib.keywords = append(pe.contrib.keywords, raw.Keywords...)
			pe.contrib.aliases = append(pe.contrib.aliases, raw.Name)
			pe.contrib.aliases = append(pe.contrib.aliases, raw.Aliases...)
			w := raw.Weight
			if w <= 0 {
				w = 1.0
			}
			pe.contrib.weight += w
			pe.contrib.mentions++
			return eid
		}
		w := raw.Weight
		if w <= 0 {
			w = 1.0
		}
		var descs []string
		if raw.Description != "" {
			descs = []string{raw.Description}
		}
		aliases := make([]string, 0, len(raw.Aliases)+1)
		aliases = append(aliases, raw.Name)
		aliases = append(aliases, raw.Aliases...)
		entMap[eid] = &pendingEnt{
			id:           eid,
			canonicalKey: ckey,
			typ:          typ,
			contrib: entityContrib{
				name:         strings.TrimSpace(raw.Name),
				descriptions: descs,
				keywords:     cloneStringSlice(raw.Keywords),
				aliases:      aliases,
				weight:       w,
				mentions:     1,
			},
		}
		return eid
	}

	for _, rawEnt := range extracted.Entities {
		addPendingEnt(rawEnt)
	}

	// 4. Deduplicate extracted relations and guarantee endpoint referential integrity.
	type pendingRel struct {
		id          string
		srcEntityID string
		dstEntityID string
		relType     RelationType
		contrib     relationContrib
	}
	relMap := make(map[string]*pendingRel)
	rawRelCount := len(extracted.Relations)

	for _, rawRel := range extracted.Relations {
		if strings.TrimSpace(rawRel.SourceName) == "" || strings.TrimSpace(rawRel.TargetName) == "" {
			continue
		}
		srcType := rawRel.SourceType
		if srcType == "" {
			srcType = inferEntityType(rawRel.SourceName)
		}
		dstType := rawRel.TargetType
		if dstType == "" {
			dstType = inferEntityType(rawRel.TargetName)
		}
		srcID, _ := canonicalEntityID(srcType, rawRel.SourceName)
		if _, ok := entMap[srcID]; !ok {
			addPendingEnt(ExtractedEntity{
				Name:        rawRel.SourceName,
				Type:        srcType,
				Description: fmt.Sprintf("Entity %s referenced by %s", rawRel.SourceName, src.SourceID),
				Keywords:    []string{rawRel.SourceName},
				Weight:      0.8,
			})
		}
		dstID, _ := canonicalEntityID(dstType, rawRel.TargetName)
		if _, ok := entMap[dstID]; !ok {
			addPendingEnt(ExtractedEntity{
				Name:        rawRel.TargetName,
				Type:        dstType,
				Description: fmt.Sprintf("Entity %s referenced by %s", rawRel.TargetName, src.SourceID),
				Keywords:    []string{rawRel.TargetName},
				Weight:      0.8,
			})
		}
		if srcID == dstID {
			continue
		}
		rtype := rawRel.Type
		if rtype == "" {
			rtype = RelRelatedTo
		}
		rid := canonicalRelationID(srcID, rtype, dstID)
		w := rawRel.Weight
		if w <= 0 {
			w = 1.0
		}
		if pr, exists := relMap[rid]; exists {
			if rawRel.Description != "" {
				pr.contrib.descriptions = append(pr.contrib.descriptions, rawRel.Description)
			}
			pr.contrib.keywords = append(pr.contrib.keywords, rawRel.Keywords...)
			pr.contrib.weight += w
		} else {
			var descs []string
			if rawRel.Description != "" {
				descs = []string{rawRel.Description}
			}
			relMap[rid] = &pendingRel{
				id:          rid,
				srcEntityID: srcID,
				dstEntityID: dstID,
				relType:     rtype,
				contrib: relationContrib{
					descriptions: descs,
					keywords:     cloneStringSlice(rawRel.Keywords),
					weight:       w,
				},
			}
		}
	}

	// 5. Merge entity contributions into tenant graph and re-index ONLY touched entities.
	addedEnt := 0
	updatedEnt := retractUpdatedEnt
	crossDupes := 0
	provEntIDs := make([]string, 0, len(entMap))
	for eid := range entMap {
		provEntIDs = append(provEntIDs, eid)
	}
	sort.Strings(provEntIDs)

	for _, eid := range provEntIDs {
		pe := entMap[eid]
		se, exists := t.entities[eid]
		oldName := ""
		if !exists {
			se = &storedEntity{
				id:           pe.id,
				canonicalKey: pe.canonicalKey,
				typ:          pe.typ,
				contribs:     make(map[string]entityContrib),
			}
			t.entities[eid] = se
			addedEnt++
		} else {
			oldName = se.name
			t.unregisterEntityAliases(se)
			updatedEnt++
			crossDupes++
		}
		se.contribs[src.SourceID] = pe.contrib
		se.recompute()
		t.registerEntityAliases(se)
		idxTxt := se.indexText()
		t.entityLex.Add(se.id, idxTxt)
		t.entityVec.Set(se.id, idxTxt)
		touchedEnt[eid] = struct{}{}
		if exists && oldName != se.name {
			refreshIncidentRelationsLocked(t, se.id)
		}
	}

	// 6. Merge relation contributions into tenant graph and re-index ONLY touched edges.
	addedRel := 0
	updatedRel := retractUpdatedRel
	provRelIDs := make([]string, 0, len(relMap))
	for rid := range relMap {
		provRelIDs = append(provRelIDs, rid)
	}
	sort.Strings(provRelIDs)

	for _, rid := range provRelIDs {
		pr := relMap[rid]
		sr, exists := t.relations[rid]
		if !exists {
			sr = &storedRelation{
				id:          pr.id,
				srcEntityID: pr.srcEntityID,
				dstEntityID: pr.dstEntityID,
				relType:     pr.relType,
				contribs:    make(map[string]relationContrib),
			}
			t.relations[rid] = sr
			t.addAdj(sr.srcEntityID, sr.id)
			t.addAdj(sr.dstEntityID, sr.id)
			addedRel++
		} else {
			updatedRel++
		}
		sr.contribs[src.SourceID] = pr.contrib
		sr.recompute()
		reindexRelationLocked(t, sr)
		touchedRel[rid] = struct{}{}
	}

	// 7. Record provenance for future O(|ΔG|) incremental updates or deletion.
	t.sourceProv[src.SourceID] = &sourceProvenance{
		sourceID:     src.SourceID,
		version:      src.Version,
		chunkIDs:     chunkIDs,
		entityIDs:    provEntIDs,
		relationIDs:  provRelIDs,
		rawEntities:  rawEntCount,
		rawRelations: rawRelCount,
		extractCost:  extracted.Metrics.EstimatedCostUSD,
	}

	// 8. Atomically invalidate cached search results for this tenant.
	cacheInv := t.invalidateCache()

	affectedEnt := len(touchedEnt)
	affectedRel := len(touchedRel)
	postGraphSize := len(t.entities) + len(t.relations)
	denom := postGraphSize
	if preGraphSize > denom {
		denom = preGraphSize
	}
	churn := 0.0
	if denom > 0 {
		churn = float64(affectedEnt+affectedRel) / float64(denom)
		if churn > 1.0 {
			churn = 1.0
		}
	}

	collapsedDupes := intraDupes + crossDupes
	dupeRate := 0.0
	if rawEntCount > 0 {
		dupeRate = float64(collapsedDupes) / float64(rawEntCount)
	}

	return UpdateTelemetry{
		Operation:                  op,
		TenantID:                   src.TenantID,
		SourceID:                   src.SourceID,
		Version:                    src.Version,
		Duration:                   time.Since(start),
		AddedEntities:              addedEnt,
		UpdatedEntities:            updatedEnt,
		RemovedEntities:            removedEnt,
		AffectedEntities:           affectedEnt,
		AddedRelations:             addedRel,
		UpdatedRelations:           updatedRel,
		RemovedRelations:           removedRel,
		AffectedRelations:          affectedRel,
		TotalEntities:              len(t.entities),
		TotalRelations:             len(t.relations),
		TotalChunks:                len(t.chunks),
		SubgraphChurnRatio:         churn,
		RawEntitiesExtracted:       rawEntCount,
		DuplicateEntitiesCollapsed: collapsedDupes,
		DuplicateEntityRate:        dupeRate,
		RawRelationsExtracted:      rawRelCount,
		UniqueRelationsMerged:      len(provRelIDs),
		ExtractionTokensIn:         extracted.Metrics.InputTokens,
		ExtractionTokensOut:        extracted.Metrics.OutputTokens,
		EstimatedCostUSD:           extracted.Metrics.EstimatedCostUSD,
		CacheInvalidated:           cacheInv,
	}
}

func reindexRelationLocked(t *tenantIndex, sr *storedRelation) {
	srcName := sr.srcEntityID
	if se := t.entities[sr.srcEntityID]; se != nil {
		srcName = se.name
	}
	dstName := sr.dstEntityID
	if de := t.entities[sr.dstEntityID]; de != nil {
		dstName = de.name
	}
	idxTxt := sr.indexText(srcName, dstName)
	t.relLex.Add(sr.id, idxTxt)
	t.relVec.Set(sr.id, idxTxt)
}

func refreshIncidentRelationsLocked(t *tenantIndex, entityID string) {
	for rid := range t.adj[entityID] {
		if sr := t.relations[rid]; sr != nil {
			reindexRelationLocked(t, sr)
		}
	}
}

// retractSourceLocked removes ONLY chunks, relation contributions, and entity
// contributions attributed to sourceID, updating surviving shared nodes/edges
// in place and deleting orphans.
func retractSourceLocked(t *tenantIndex, sourceID string, touchedEnt, touchedRel map[string]struct{}) (removedEnt, updatedEnt, removedRel, updatedRel int) {
	prov, ok := t.sourceProv[sourceID]
	if !ok {
		return 0, 0, 0, 0
	}

	// Remove chunks for this source.
	for _, cid := range prov.chunkIDs {
		delete(t.chunks, cid)
		t.chunkLex.Remove(cid)
		t.chunkVec.Delete(cid)
	}

	// Retract relations BEFORE entities so endpoint names remain valid during recompute.
	for _, rid := range prov.relationIDs {
		sr, exists := t.relations[rid]
		if !exists {
			continue
		}
		if touchedRel != nil {
			touchedRel[rid] = struct{}{}
		}
		delete(sr.contribs, sourceID)
		if len(sr.contribs) == 0 {
			t.removeAdj(sr.srcEntityID, rid)
			t.removeAdj(sr.dstEntityID, rid)
			delete(t.relations, rid)
			t.relLex.Remove(rid)
			t.relVec.Delete(rid)
			removedRel++
		} else {
			sr.recompute()
			reindexRelationLocked(t, sr)
			updatedRel++
		}
	}

	// Retract entity contributions.
	for _, eid := range prov.entityIDs {
		se, exists := t.entities[eid]
		if !exists {
			continue
		}
		if touchedEnt != nil {
			touchedEnt[eid] = struct{}{}
		}
		oldName := se.name
		t.unregisterEntityAliases(se)
		delete(se.contribs, sourceID)
		if len(se.contribs) == 0 {
			delete(t.entities, eid)
			delete(t.adj, eid)
			t.entityLex.Remove(eid)
			t.entityVec.Delete(eid)
			removedEnt++
		} else {
			se.recompute()
			t.registerEntityAliases(se)
			idxTxt := se.indexText()
			t.entityLex.Add(se.id, idxTxt)
			t.entityVec.Set(se.id, idxTxt)
			if oldName != se.name {
				refreshIncidentRelationsLocked(t, se.id)
			}
			updatedEnt++
		}
	}

	delete(t.sourceProv, sourceID)
	return removedEnt, updatedEnt, removedRel, updatedRel
}

// GetSource returns a deep copy of the canonical SourceRecord if it exists.
func (e *Engine) GetSource(tenantID, sourceID string) (SourceRecord, bool) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	rec, ok := e.sources[sourceKey{tenantID: tenantID, sourceID: sourceID}]
	if !ok {
		return SourceRecord{}, false
	}
	return rec.Clone(), true
}

// ListSources returns deep copies of all canonical SourceRecords for tenantID
// sorted deterministically by SourceID.
func (e *Engine) ListSources(tenantID string) []SourceRecord {
	e.mu.RLock()
	defer e.mu.RUnlock()
	var out []SourceRecord
	for k, rec := range e.sources {
		if k.tenantID == tenantID {
			out = append(out, rec.Clone())
		}
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].SourceID < out[j].SourceID
	})
	return out
}

// GetEntity returns an ACL-scoped, deep-copied view of a derived entity.
// Descriptions, keywords, aliases, and sourceIDs are strictly filtered to
// contributing sources authorized for principals.
func (e *Engine) GetEntity(tenantID, entityID string, principals []string) (Entity, bool) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if !e.derivedBuilt {
		return Entity{}, false
	}
	t := e.tenants[tenantID]
	if t == nil {
		return Entity{}, false
	}
	se, ok := t.entities[entityID]
	if !ok {
		return Entity{}, false
	}
	pSet := makePrincipalSet(principals)
	ent, allowed := e.materializeEntityLocked(t, se, pSet)
	if !allowed {
		return Entity{}, false
	}
	return ent.Clone(), true
}

// GetRelation returns an ACL-scoped, deep-copied view of a derived relation.
func (e *Engine) GetRelation(tenantID, relationID string, principals []string) (Relation, bool) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if !e.derivedBuilt {
		return Relation{}, false
	}
	t := e.tenants[tenantID]
	if t == nil {
		return Relation{}, false
	}
	sr, ok := t.relations[relationID]
	if !ok {
		return Relation{}, false
	}
	pSet := makePrincipalSet(principals)
	rel, allowed := e.materializeRelationLocked(t, sr, pSet)
	if !allowed {
		return Relation{}, false
	}
	return rel.Clone(), true
}

// Stats returns derived index and deduplication statistics for tenantID.
func (e *Engine) Stats(tenantID string) TenantStats {
	e.mu.RLock()
	defer e.mu.RUnlock()

	srcCount := 0
	for k := range e.sources {
		if k.tenantID == tenantID {
			srcCount++
		}
	}
	st := TenantStats{
		TenantID:     tenantID,
		Sources:      srcCount,
		DerivedBuilt: e.derivedBuilt,
	}
	t := e.tenants[tenantID]
	if t == nil {
		return st
	}
	st.Chunks = len(t.chunks)
	st.Entities = len(t.entities)
	st.Relations = len(t.relations)
	st.Aliases = len(t.aliasToEntity)
	st.CacheEntries = len(t.queryCache)
	st.Generation = t.generation
	for _, p := range t.sourceProv {
		st.TotalRawEntities += p.rawEntities
		st.TotalRawRelations += p.rawRelations
		st.TotalExtractionCostUS += p.extractCost
	}
	if st.TotalRawEntities > st.Entities {
		st.TotalCollapsedDupes = st.TotalRawEntities - st.Entities
		st.AggregateDupeRate = float64(st.TotalCollapsedDupes) / float64(st.TotalRawEntities)
	}
	return st
}

// IndexDigest computes a deterministic SHA-256 hex digest over all derived
// chunks, entities, relations, aliases, and adjacency lists for tenantID.
// Used to verify bit-exact equivalence between incremental updates and RebuildAll.
func (e *Engine) IndexDigest(tenantID string) (string, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if !e.derivedBuilt {
		return "", ErrIndexesNotBuilt
	}
	h := sha256.New()
	t := e.tenants[tenantID]
	if t == nil {
		return hex.EncodeToString(h.Sum(nil))[:16], nil
	}

	// Chunks in sorted ID order.
	cids := make([]string, 0, len(t.chunks))
	for cid := range t.chunks {
		cids = append(cids, cid)
	}
	sort.Strings(cids)
	for _, cid := range cids {
		ch := t.chunks[cid]
		_, _ = fmt.Fprintf(h, "C|%s|%s|%d|%s|%s\n", ch.ChunkID, ch.SourceID, ch.Version, strings.Join(ch.ACL, ","), ch.Text)
	}

	// Entities in sorted ID order.
	eids := make([]string, 0, len(t.entities))
	for eid := range t.entities {
		eids = append(eids, eid)
	}
	sort.Strings(eids)
	for _, eid := range eids {
		se := t.entities[eid]
		_, _ = fmt.Fprintf(h, "E|%s|%s|%s|%s|%s|%s|%s|%d|%.6f\n",
			se.id, se.canonicalKey, se.typ, se.name, se.description,
			strings.Join(se.keywords, ","), strings.Join(se.sourceIDs, ","),
			se.mentionCount, se.weight)
	}

	// Relations in sorted ID order.
	rids := make([]string, 0, len(t.relations))
	for rid := range t.relations {
		rids = append(rids, rid)
	}
	sort.Strings(rids)
	for _, rid := range rids {
		sr := t.relations[rid]
		_, _ = fmt.Fprintf(h, "R|%s|%s|%s|%s|%s|%s|%s|%.6f\n",
			sr.id, sr.srcEntityID, sr.relType, sr.dstEntityID, sr.description,
			strings.Join(sr.keywords, ","), strings.Join(sr.sourceIDs, ","), sr.weight)
	}

	// Aliases in sorted key order.
	alKeys := make([]string, 0, len(t.aliasToEntity))
	for k := range t.aliasToEntity {
		alKeys = append(alKeys, k)
	}
	sort.Strings(alKeys)
	for _, k := range alKeys {
		set := t.aliasToEntity[k]
		ids := make([]string, 0, len(set))
		for id := range set {
			ids = append(ids, id)
		}
		sort.Strings(ids)
		_, _ = fmt.Fprintf(h, "A|%s|%s\n", k, strings.Join(ids, ","))
	}

	return hex.EncodeToString(h.Sum(nil))[:16], nil
}

// Search executes Local, Global, Hybrid (dual-level LightRAG), or OrdinaryHybrid
// retrieval strictly scoped to q.TenantID and q.Principals.
func (e *Engine) Search(ctx context.Context, q Query) (SearchResult, error) {
	start := time.Now()
	if err := ctx.Err(); err != nil {
		return SearchResult{}, err
	}
	if strings.TrimSpace(q.TenantID) == "" {
		return SearchResult{}, ErrEmptyTenant
	}
	if strings.TrimSpace(q.Text) == "" {
		return SearchResult{}, ErrInvalidQuery
	}
	if q.Mode == "" {
		q.Mode = ModeHybrid
	}
	if q.TopK <= 0 {
		q.TopK = 5
	}
	if q.MaxHops <= 0 {
		q.MaxHops = 2
	}
	if q.MaxExpandedNodes <= 0 {
		q.MaxExpandedNodes = 64
	}

	cacheKey := buildQueryCacheKey(q)

	e.mu.RLock()
	if !e.derivedBuilt {
		e.mu.RUnlock()
		return SearchResult{}, ErrIndexesNotBuilt
	}
	t := e.tenants[q.TenantID]
	if t == nil {
		e.mu.RUnlock()
		lowKW, highKW := DecomposeQuery(q.Text)
		return SearchResult{
			Mode:              q.Mode,
			Query:             q.Text,
			LowLevelKeywords:  lowKW,
			HighLevelKeywords: highKW,
			Latency:           time.Since(start),
		}, nil
	}

	if e.cfg.EnableQueryCache && !q.BypassCache {
		if cached, ok := t.queryCache[cacheKey]; ok {
			res := cached.Clone()
			res.CacheHit = true
			res.Latency = time.Since(start)
			e.mu.RUnlock()
			return res, nil
		}
	}

	genBefore := t.generation
	pSet := makePrincipalSet(q.Principals)
	var res SearchResult
	switch q.Mode {
	case ModeOrdinaryHybrid:
		res = e.searchOrdinaryHybridLocked(t, q, pSet)
	case ModeLocal:
		res = e.searchLocalLocked(t, q, pSet)
	case ModeGlobal:
		res = e.searchGlobalLocked(t, q, pSet)
	case ModeHybrid:
		res = e.searchHybridLocked(t, q, pSet)
	default:
		e.mu.RUnlock()
		return SearchResult{}, fmt.Errorf("%w: unsupported mode %q", ErrInvalidQuery, q.Mode)
	}
	e.mu.RUnlock()

	res.Latency = time.Since(start)
	res.CacheHit = false

	// Populate cache only if tenant generation has not changed concurrently.
	if e.cfg.EnableQueryCache && !q.BypassCache {
		e.mu.Lock()
		if e.derivedBuilt {
			if curT := e.tenants[q.TenantID]; curT == t && curT.generation == genBefore {
				if len(curT.queryCache) >= e.cfg.MaxCacheEntries {
					clear(curT.queryCache)
				}
				curT.queryCache[cacheKey] = res.Clone()
			}
		}
		e.mu.Unlock()
	}

	return res.Clone(), nil
}

// searchOrdinaryHybridLocked implements the matched chunk-only BM25 + dense
// vector RRF baseline with strict pre-return ACL filtering.
func (e *Engine) searchOrdinaryHybridLocked(t *tenantIndex, q Query, pSet map[string]struct{}) SearchResult {
	chunkAllowed := func(cid string) bool {
		ch, ok := t.chunks[cid]
		if !ok {
			return false
		}
		return e.aclAllowed(ch.ACL, pSet)
	}

	candK := q.TopK * 6
	if candK < 24 {
		candK = 24
	}
	lexHits, lexFiltered := t.chunkLex.Search(q.Text, chunkAllowed, candK)
	vecHits, vecFiltered := t.chunkVec.Search(q.Text, chunkAllowed, candK)

	const rrfK = 60.0
	type agg struct {
		score    float64
		lexScore float64
		vecScore float64
	}
	combined := make(map[string]*agg)
	for rank, h := range lexHits {
		a := combined[h.id]
		if a == nil {
			a = &agg{}
			combined[h.id] = a
		}
		a.lexScore = h.score
		a.score += 0.55 / (rrfK + float64(rank+1))
	}
	for rank, h := range vecHits {
		a := combined[h.id]
		if a == nil {
			a = &agg{}
			combined[h.id] = a
		}
		a.vecScore = h.score
		a.score += 0.45 / (rrfK + float64(rank+1))
	}

	scores := make(map[string]float64, len(combined))
	for cid, a := range combined {
		scores[cid] = a.score
	}
	ranked := topScored(scores, q.TopK)
	chunks := make([]RetrievedChunk, 0, len(ranked))
	totalTok := 0
	for _, r := range ranked {
		ch := t.chunks[r.id].Clone()
		a := combined[r.id]
		chunks = append(chunks, RetrievedChunk{
			Chunk:        ch,
			Score:        r.score,
			LexicalScore: a.lexScore,
			VectorScore:  a.vecScore,
		})
		totalTok += ch.TokenCount
	}

	aclFiltered := lexFiltered
	if vecFiltered > aclFiltered {
		aclFiltered = vecFiltered
	}
	return SearchResult{
		Mode:           ModeOrdinaryHybrid,
		Query:          q.Text,
		Chunks:         chunks,
		TotalTokens:    totalTok,
		ACLFilteredOut: aclFiltered,
	}
}

type localChannelOut struct {
	chunkScores   map[string]float64
	chunkEntities map[string][]string
	entities      []RetrievedEntity
	relations     []RetrievedRelation
	trails        []EvidenceTrail
	inspectedEnt  int
	inspectedRel  int
	aclFiltered   int
}

func (e *Engine) runLocalChannelLocked(t *tenantIndex, q Query, lowKW []string, pSet map[string]struct{}) localChannelOut {
	lowQuery := strings.Join(lowKW, " ")
	if lowQuery == "" {
		lowQuery = q.Text
	}
	lowQueryTokens := makeTokenSet(lowQuery + " " + q.Text)

	entityAllowed := func(eid string) bool {
		se, ok := t.entities[eid]
		if !ok {
			return false
		}
		return e.entityAuthorizedForQueryLocked(t, se, pSet, lowQueryTokens)
	}

	seedK := q.TopK * 4
	if seedK < 16 {
		seedK = 16
	}
	lexEnt, lexFilt := t.entityLex.Search(lowQuery+" "+q.Text, entityAllowed, seedK)
	vecEnt, vecFilt := t.entityVec.Search(lowQuery, entityAllowed, seedK)

	const rrfK = 60.0
	seedScores := make(map[string]float64)
	for rank, h := range lexEnt {
		seedScores[h.id] += 1.2 / (rrfK + float64(rank+1))
	}
	for rank, h := range vecEnt {
		seedScores[h.id] += 0.6 / (rrfK + float64(rank+1))
	}

	// Exact/alias matches from low-level keywords get a high-precision seed boost.
	for _, kw := range lowKW {
		norm := normalizeAliasToken(kw)
		if norm == "" {
			continue
		}
		for eid := range t.aliasToEntity[norm] {
			if !entityAllowed(eid) {
				continue
			}
			// Guard against cross-ACL alias leakage: verify alias exists in authorized contribs.
			if se := t.entities[eid]; se != nil && e.entityHasAuthorizedAliasLocked(t, se, pSet, norm) {
				seedScores[eid] += 0.055
			}
		}
	}

	seedRanked := topScored(seedScores, seedK)
	if len(seedRanked) > 12 {
		seedRanked = seedRanked[:12]
	}

	// Bounded 1-hop / 2-hop neighborhood expansion with hub-degree damping.
	entScore := make(map[string]float64)
	entHop := make(map[string]int)
	relScore := make(map[string]float64)
	relHop := make(map[string]int)

	type frontierItem struct {
		id    string
		score float64
		hop   int
	}
	var frontier []frontierItem
	for _, s := range seedRanked {
		entScore[s.id] = s.score
		entHop[s.id] = 0
		frontier = append(frontier, frontierItem{id: s.id, score: s.score, hop: 0})
	}

	inspectedEnt := len(seedRanked)
	inspectedRel := 0
	aclFiltered := lexFilt
	if vecFilt > aclFiltered {
		aclFiltered = vecFilt
	}

	for hop := 0; hop < q.MaxHops && len(frontier) > 0 && len(entScore) < q.MaxExpandedNodes; hop++ {
		var nextFrontier []frontierItem
		// Sort frontier deterministically by score desc, id asc.
		sort.Slice(frontier, func(i, j int) bool {
			if math.Abs(frontier[i].score-frontier[j].score) > 1e-12 {
				return frontier[i].score > frontier[j].score
			}
			return frontier[i].id < frontier[j].id
		})

		for _, node := range frontier {
			adjSet := t.adj[node.id]
			if len(adjSet) == 0 {
				continue
			}
			deg := float64(len(adjSet))
			// Hub damping prevents high-degree generic nodes from flooding expansion.
			hubDamp := 1.0 / math.Sqrt(1.0+deg)

			rids := make([]string, 0, len(adjSet))
			for rid := range adjSet {
				rids = append(rids, rid)
			}
			sort.Strings(rids)

			for _, rid := range rids {
				sr := t.relations[rid]
				if sr == nil {
					continue
				}
				if !e.relationHasAuthorizedSourceLocked(t, sr, pSet) {
					aclFiltered++
					continue
				}
				inspectedRel++

				neighborID := sr.dstEntityID
				if neighborID == node.id {
					neighborID = sr.srcEntityID
				}
				neighborSE := t.entities[neighborID]
				if neighborSE == nil || !e.entityHasAuthorizedSourceLocked(t, neighborSE, pSet) {
					aclFiltered++
					continue
				}

				edgeBoost := 1.0 + 0.15*math.Log1p(sr.weight)
				// Reward specific structural relations (CALLS, IMPORTS, MODIFIES, CAUSES, MITIGATES, DEFINES, DECIDES).
				if sr.relType == RelCalls || sr.relType == RelModifies || sr.relType == RelCauses ||
					sr.relType == RelMitigates || sr.relType == RelDecides || sr.relType == RelDependsOn {
					edgeBoost *= 1.35
				} else if sr.relType == RelImports || sr.relType == RelDefines || sr.relType == RelReferences {
					edgeBoost *= 1.15
				}
				propScore := node.score * e.cfg.HopDecay * hubDamp * edgeBoost

				relScore[rid] += propScore
				if prevHop, ok := relHop[rid]; !ok || hop+1 < prevHop {
					relHop[rid] = hop + 1
				}

				prevEnt, seen := entScore[neighborID]
				if !seen && len(entScore) >= q.MaxExpandedNodes {
					continue
				}
				entScore[neighborID] = prevEnt + propScore
				if !seen {
					entHop[neighborID] = hop + 1
					inspectedEnt++
					nextFrontier = append(nextFrontier, frontierItem{
						id:    neighborID,
						score: propScore,
						hop:   hop + 1,
					})
				}
			}
		}
		frontier = nextFrontier
	}

	// Materialize top entities and relations strictly filtered by ACL.
	topEnts := topScored(entScore, 16)
	retEnts := make([]RetrievedEntity, 0, len(topEnts))
	for _, te := range topEnts {
		se := t.entities[te.id]
		if se == nil {
			continue
		}
		mat, ok := e.materializeEntityLocked(t, se, pSet)
		if !ok {
			continue
		}
		retEnts = append(retEnts, RetrievedEntity{
			Entity: mat,
			Score:  te.score,
			Hop:    entHop[te.id],
		})
	}

	topRels := topScored(relScore, 16)
	retRels := make([]RetrievedRelation, 0, len(topRels))
	trails := make([]EvidenceTrail, 0, len(topRels))
	for _, tr := range topRels {
		sr := t.relations[tr.id]
		if sr == nil {
			continue
		}
		mat, ok := e.materializeRelationLocked(t, sr, pSet)
		if !ok {
			continue
		}
		retRels = append(retRels, RetrievedRelation{
			Relation: mat,
			Score:    tr.score,
			Hop:      relHop[tr.id],
		})
		trails = append(trails, EvidenceTrail{
			SourceEntity:      mat.SourceEntityName,
			Relation:          mat.Type,
			TargetEntity:      mat.TargetEntityName,
			Description:       mat.Description,
			SupportingSources: cloneStringSlice(mat.SourceIDs),
			Score:             tr.score,
		})
	}

	// Project local entity + relation support onto ACL-authorized provenance chunks.
	sourceSupport := make(map[string]float64)
	sourceMatchedEnts := make(map[string][]string)

	for _, re := range retEnts {
		hopScale := 1.0 / float64(1+re.Hop)
		// Normalize per-entity source spread so mega-hub entities don't dominate every file.
		spread := 1.0 / math.Sqrt(float64(maxInt(1, len(re.Entity.SourceIDs))))
		for _, sid := range re.Entity.SourceIDs {
			sourceSupport[sid] += re.Score * hopScale * spread
			if len(sourceMatchedEnts[sid]) < 6 {
				sourceMatchedEnts[sid] = appendUnique(sourceMatchedEnts[sid], re.Entity.Name)
			}
		}
	}
	for _, rr := range retRels {
		spread := 1.0 / math.Sqrt(float64(maxInt(1, len(rr.Relation.SourceIDs))))
		for _, sid := range rr.Relation.SourceIDs {
			sourceSupport[sid] += rr.Score * 1.15 * spread
		}
	}

	chunkScores, chunkEntities := e.projectSourcesToChunksLocked(t, q.Text, sourceSupport, sourceMatchedEnts, pSet)

	return localChannelOut{
		chunkScores:   chunkScores,
		chunkEntities: chunkEntities,
		entities:      retEnts,
		relations:     retRels,
		trails:        trails,
		inspectedEnt:  inspectedEnt,
		inspectedRel:  inspectedRel,
		aclFiltered:   aclFiltered,
	}
}

type globalChannelOut struct {
	chunkScores    map[string]float64
	chunkRelations map[string][]string
	entities       []RetrievedEntity
	relations      []RetrievedRelation
	trails         []EvidenceTrail
	inspectedRel   int
	aclFiltered    int
}

func (e *Engine) runGlobalChannelLocked(t *tenantIndex, q Query, highKW []string, pSet map[string]struct{}) globalChannelOut {
	highQuery := strings.Join(highKW, " ")
	if highQuery == "" {
		highQuery = q.Text
	}
	combinedQuery := highQuery + " " + q.Text
	queryTokens := makeTokenSet(combinedQuery)

	relAllowed := func(rid string) bool {
		sr, ok := t.relations[rid]
		if !ok {
			return false
		}
		return e.relationAuthorizedForQueryLocked(t, sr, pSet, queryTokens)
	}

	candK := q.TopK * 6
	if candK < 24 {
		candK = 24
	}
	lexRels, lexFilt := t.relLex.Search(combinedQuery, relAllowed, candK)
	vecRels, vecFilt := t.relVec.Search(highQuery, relAllowed, candK)

	const rrfK = 60.0
	relScores := make(map[string]float64)
	for rank, h := range lexRels {
		relScores[h.id] += 1.0 / (rrfK + float64(rank+1))
	}
	for rank, h := range vecRels {
		relScores[h.id] += 0.85 / (rrfK + float64(rank+1))
	}

	topRels := topScored(relScores, 20)
	retRels := make([]RetrievedRelation, 0, len(topRels))
	trails := make([]EvidenceTrail, 0, len(topRels))
	entScores := make(map[string]float64)
	sourceSupport := make(map[string]float64)
	sourceMatchedRels := make(map[string][]string)

	for _, tr := range topRels {
		sr := t.relations[tr.id]
		if sr == nil {
			continue
		}
		mat, ok := e.materializeRelationLocked(t, sr, pSet)
		if !ok {
			continue
		}
		retRels = append(retRels, RetrievedRelation{
			Relation: mat,
			Score:    tr.score,
			Hop:      0,
		})
		trails = append(trails, EvidenceTrail{
			SourceEntity:      mat.SourceEntityName,
			Relation:          mat.Type,
			TargetEntity:      mat.TargetEntityName,
			Description:       mat.Description,
			SupportingSources: cloneStringSlice(mat.SourceIDs),
			Score:             tr.score,
		})
		entScores[sr.srcEntityID] += tr.score * 0.7
		entScores[sr.dstEntityID] += tr.score * 0.7

		relLabel := fmt.Sprintf("%s -[%s]-> %s", mat.SourceEntityName, mat.Type, mat.TargetEntityName)
		spread := 1.0 / math.Sqrt(float64(maxInt(1, len(mat.SourceIDs))))
		for _, sid := range mat.SourceIDs {
			sourceSupport[sid] += tr.score * spread
			if len(sourceMatchedRels[sid]) < 4 {
				sourceMatchedRels[sid] = appendUnique(sourceMatchedRels[sid], relLabel)
			}
		}
	}

	// Gather endpoint entities of matched global relationships.
	topEnts := topScored(entScores, 16)
	retEnts := make([]RetrievedEntity, 0, len(topEnts))
	for _, te := range topEnts {
		se := t.entities[te.id]
		if se == nil {
			continue
		}
		mat, ok := e.materializeEntityLocked(t, se, pSet)
		if !ok {
			continue
		}
		retEnts = append(retEnts, RetrievedEntity{
			Entity: mat,
			Score:  te.score,
			Hop:    1,
		})
		// Avoid spraying mega-hub EntityConcept nodes across dozens of unrelated files;
		// the matched relationships themselves already project exact participating sources.
		if mat.Type == EntityConcept {
			continue
		}
		spread := 1.0 / math.Sqrt(float64(maxInt(1, len(mat.SourceIDs))))
		for _, sid := range mat.SourceIDs {
			sourceSupport[sid] += te.score * 0.35 * spread
		}
	}

	chunkScores, chunkRels := e.projectSourcesToChunksLocked(t, q.Text, sourceSupport, sourceMatchedRels, pSet)

	aclFilt := lexFilt
	if vecFilt > aclFilt {
		aclFilt = vecFilt
	}
	return globalChannelOut{
		chunkScores:    chunkScores,
		chunkRelations: chunkRels,
		entities:       retEnts,
		relations:      retRels,
		trails:         trails,
		inspectedRel:   len(relScores),
		aclFiltered:    aclFilt,
	}
}

func (e *Engine) searchLocalLocked(t *tenantIndex, q Query, pSet map[string]struct{}) SearchResult {
	lowKW, highKW := DecomposeQuery(q.Text)
	loc := e.runLocalChannelLocked(t, q, lowKW, pSet)

	ranked := topScored(loc.chunkScores, q.TopK)
	chunks := make([]RetrievedChunk, 0, len(ranked))
	totalTok := 0
	for _, r := range ranked {
		ch := t.chunks[r.id].Clone()
		chunks = append(chunks, RetrievedChunk{
			Chunk:           ch,
			Score:           r.score,
			LocalScore:      r.score,
			MatchedEntities: cloneStringSlice(loc.chunkEntities[r.id]),
		})
		totalTok += ch.TokenCount
	}

	res := SearchResult{
		Mode:               ModeLocal,
		Query:              q.Text,
		LowLevelKeywords:   lowKW,
		HighLevelKeywords:  highKW,
		Chunks:             chunks,
		Trails:             capTrails(loc.trails, 12),
		TotalTokens:        totalTok,
		InspectedEntities:  loc.inspectedEnt,
		InspectedRelations: loc.inspectedRel,
		ACLFilteredOut:     loc.aclFiltered,
	}
	if q.IncludeSubgraph {
		res.Entities = loc.entities
		res.Relations = loc.relations
	}
	return res
}

func (e *Engine) searchGlobalLocked(t *tenantIndex, q Query, pSet map[string]struct{}) SearchResult {
	lowKW, highKW := DecomposeQuery(q.Text)
	glob := e.runGlobalChannelLocked(t, q, highKW, pSet)

	ranked := topScored(glob.chunkScores, q.TopK)
	chunks := make([]RetrievedChunk, 0, len(ranked))
	totalTok := 0
	for _, r := range ranked {
		ch := t.chunks[r.id].Clone()
		chunks = append(chunks, RetrievedChunk{
			Chunk:            ch,
			Score:            r.score,
			GlobalScore:      r.score,
			MatchedRelations: cloneStringSlice(glob.chunkRelations[r.id]),
		})
		totalTok += ch.TokenCount
	}

	res := SearchResult{
		Mode:               ModeGlobal,
		Query:              q.Text,
		LowLevelKeywords:   lowKW,
		HighLevelKeywords:  highKW,
		Chunks:             chunks,
		Trails:             capTrails(glob.trails, 12),
		TotalTokens:        totalTok,
		InspectedEntities:  len(glob.entities),
		InspectedRelations: glob.inspectedRel,
		ACLFilteredOut:     glob.aclFiltered,
	}
	if q.IncludeSubgraph {
		res.Entities = glob.entities
		res.Relations = glob.relations
	}
	return res
}

func (e *Engine) searchHybridLocked(t *tenantIndex, q Query, pSet map[string]struct{}) SearchResult {
	lowKW, highKW := DecomposeQuery(q.Text)
	loc := e.runLocalChannelLocked(t, q, lowKW, pSet)
	glob := e.runGlobalChannelLocked(t, q, highKW, pSet)

	// Direct chunk grounding (pre-filtered by ACL).
	baseQ := q
	baseQ.TopK = maxInt(q.TopK*3, 15)
	base := e.searchOrdinaryHybridLocked(t, baseQ, pSet)

	const rrfK = 60.0
	type fused struct {
		score    float64
		locScore float64
		glbScore float64
		lexScore float64
		vecScore float64
		ents     []string
		rels     []string
	}
	byChunk := make(map[string]*fused)
	get := func(cid string) *fused {
		f := byChunk[cid]
		if f == nil {
			f = &fused{}
			byChunk[cid] = f
		}
		return f
	}

	locRanked := topScored(loc.chunkScores, maxInt(q.TopK*4, 20))
	for rank, h := range locRanked {
		f := get(h.id)
		f.locScore = h.score
		f.score += e.cfg.LocalWeight / (rrfK + float64(rank+1))
		f.ents = loc.chunkEntities[h.id]
	}

	globRanked := topScored(glob.chunkScores, maxInt(q.TopK*4, 20))
	for rank, h := range globRanked {
		f := get(h.id)
		f.glbScore = h.score
		f.score += e.cfg.GlobalWeight / (rrfK + float64(rank+1))
		f.rels = glob.chunkRelations[h.id]
	}

	for rank, rc := range base.Chunks {
		f := get(rc.Chunk.ChunkID)
		f.lexScore = rc.LexicalScore
		f.vecScore = rc.VectorScore
		f.score += e.cfg.ChunkWeight / (rrfK + float64(rank+1))
	}

	finalScores := make(map[string]float64, len(byChunk))
	for cid, f := range byChunk {
		finalScores[cid] = f.score
	}
	topChunks := topScored(finalScores, q.TopK)

	chunks := make([]RetrievedChunk, 0, len(topChunks))
	totalTok := 0
	for _, tc := range topChunks {
		ch := t.chunks[tc.id].Clone()
		f := byChunk[tc.id]
		chunks = append(chunks, RetrievedChunk{
			Chunk:            ch,
			Score:            tc.score,
			LocalScore:       f.locScore,
			GlobalScore:      f.glbScore,
			LexicalScore:     f.lexScore,
			VectorScore:      f.vecScore,
			MatchedEntities:  cloneStringSlice(f.ents),
			MatchedRelations: cloneStringSlice(f.rels),
		})
		totalTok += ch.TokenCount
	}

	// Merge trails from both channels deterministically.
	trailMap := make(map[string]EvidenceTrail)
	for _, tr := range append(loc.trails, glob.trails...) {
		tk := fmt.Sprintf("%s|%s|%s", tr.SourceEntity, tr.Relation, tr.TargetEntity)
		if prev, ok := trailMap[tk]; !ok || tr.Score > prev.Score {
			trailMap[tk] = tr
		}
	}
	trails := make([]EvidenceTrail, 0, len(trailMap))
	for _, tr := range trailMap {
		trails = append(trails, tr)
	}
	sort.Slice(trails, func(i, j int) bool {
		if math.Abs(trails[i].Score-trails[j].Score) > 1e-12 {
			return trails[i].Score > trails[j].Score
		}
		return trails[i].SourceEntity < trails[j].SourceEntity
	})

	res := SearchResult{
		Mode:               ModeHybrid,
		Query:              q.Text,
		LowLevelKeywords:   lowKW,
		HighLevelKeywords:  highKW,
		Chunks:             chunks,
		Trails:             capTrails(trails, 16),
		TotalTokens:        totalTok,
		InspectedEntities:  loc.inspectedEnt + len(glob.entities),
		InspectedRelations: loc.inspectedRel + glob.inspectedRel,
		ACLFilteredOut:     loc.aclFiltered + glob.aclFiltered + base.ACLFilteredOut,
	}
	if q.IncludeSubgraph {
		res.Entities = mergeRetrievedEntities(loc.entities, glob.entities, 16)
		res.Relations = mergeRetrievedRelations(loc.relations, glob.relations, 16)
	}
	return res
}

// projectSourcesToChunksLocked selects the highest-relevance chunk within each
// graph-supported source, strictly verifying pre-return ACL authorization.
func (e *Engine) projectSourcesToChunksLocked(
	t *tenantIndex,
	queryText string,
	sourceSupport map[string]float64,
	sourceLabels map[string][]string,
	pSet map[string]struct{},
) (map[string]float64, map[string][]string) {
	chunkScores := make(map[string]float64)
	chunkLabels := make(map[string][]string)
	qVec := EmbedText(queryText, e.cfg.VectorDim)
	qToks := makeTokenSet(queryText)

	for sid, supScore := range sourceSupport {
		if !e.sourceAllowedLocked(t.tenantID, sid, pSet) {
			continue
		}
		prov := t.sourceProv[sid]
		if prov == nil || len(prov.chunkIDs) == 0 {
			continue
		}
		bestCID := ""
		bestScore := -1.0
		for _, cid := range prov.chunkIDs {
			ch, ok := t.chunks[cid]
			if !ok || !e.aclAllowed(ch.ACL, pSet) {
				continue
			}
			// Combine graph support score with chunk semantic alignment so multi-chunk
			// source files surface the most pertinent chunk.
			sim := 0.0
			if vec, ok := t.chunkVec.vectors[cid]; ok {
				sim = CosineSimilarity(qVec, vec)
				if sim < 0 {
					sim = 0
				}
			}
			overlap := tfOverlapRatio(qToks, t.chunkLex.docs[cid].tf)
			sc := supScore * (1.0 + 0.35*sim + 0.25*overlap)
			if sc > bestScore || (math.Abs(sc-bestScore) <= 1e-12 && (bestCID == "" || cid < bestCID)) {
				bestScore = sc
				bestCID = cid
			}
		}
		if bestCID != "" {
			chunkScores[bestCID] = bestScore
			if labels := sourceLabels[sid]; len(labels) > 0 {
				chunkLabels[bestCID] = cloneStringSlice(labels)
			}
		}
	}
	return chunkScores, chunkLabels
}

func tfOverlapRatio(queryTokens map[string]struct{}, tf map[string]int) float64 {
	if len(queryTokens) == 0 || len(tf) == 0 {
		return 0
	}
	hits := 0
	for tok := range queryTokens {
		if tf[tok] > 0 {
			hits++
		}
	}
	return float64(hits) / float64(len(queryTokens))
}

// --- ACL & Sub-Document Authorization Helpers ---

func (e *Engine) aclAllowed(acl []string, pSet map[string]struct{}) bool {
	if len(acl) == 0 {
		return !e.cfg.RequireExplicitACL
	}
	if len(pSet) == 0 {
		return false
	}
	for _, p := range acl {
		if _, ok := pSet[p]; ok {
			return true
		}
	}
	return false
}

func (e *Engine) sourceAllowedLocked(tenantID, sourceID string, pSet map[string]struct{}) bool {
	src, ok := e.sources[sourceKey{tenantID: tenantID, sourceID: sourceID}]
	if !ok {
		return false
	}
	return e.aclAllowed(src.ACL, pSet)
}

func (e *Engine) entityHasAuthorizedSourceLocked(t *tenantIndex, se *storedEntity, pSet map[string]struct{}) bool {
	for _, sid := range se.sourceIDs {
		if e.sourceAllowedLocked(t.tenantID, sid, pSet) {
			return true
		}
	}
	return false
}

func (e *Engine) relationHasAuthorizedSourceLocked(t *tenantIndex, sr *storedRelation, pSet map[string]struct{}) bool {
	srcAllowed := false
	for _, sid := range sr.sourceIDs {
		if e.sourceAllowedLocked(t.tenantID, sid, pSet) {
			srcAllowed = true
			break
		}
	}
	if !srcAllowed {
		return false
	}
	se := t.entities[sr.srcEntityID]
	de := t.entities[sr.dstEntityID]
	return se != nil && de != nil &&
		e.entityHasAuthorizedSourceLocked(t, se, pSet) &&
		e.entityHasAuthorizedSourceLocked(t, de, pSet)
}

// entityAuthorizedForQueryLocked closes cross-ACL existence side-channels:
// if an entity has contributions from BOTH authorized and unauthorized sources,
// it only matches the query if the caller's authorized-only view shares signal
// with queryTokens.
func (e *Engine) entityAuthorizedForQueryLocked(t *tenantIndex, se *storedEntity, pSet map[string]struct{}, queryTokens map[string]struct{}) bool {
	allowedCount := 0
	for _, sid := range se.sourceIDs {
		if e.sourceAllowedLocked(t.tenantID, sid, pSet) {
			allowedCount++
		}
	}
	if allowedCount == 0 {
		return false
	}
	if allowedCount == len(se.sourceIDs) {
		return true
	}
	// Straddles an ACL boundary: require match strictly on authorized text.
	mat, ok := e.materializeEntityLocked(t, se, pSet)
	if !ok {
		return false
	}
	authText := mat.Name + " " + strings.Join(mat.Aliases, " ") + " " + strings.Join(mat.Keywords, " ") + " " + mat.Description
	return hasAnyTokenOverlap(queryTokens, authText)
}

func (e *Engine) entityHasAuthorizedAliasLocked(t *tenantIndex, se *storedEntity, pSet map[string]struct{}, normAlias string) bool {
	for _, sid := range se.sourceIDs {
		if !e.sourceAllowedLocked(t.tenantID, sid, pSet) {
			continue
		}
		c := se.contribs[sid]
		if normalizeAliasToken(c.name) == normAlias {
			return true
		}
		for _, al := range c.aliases {
			if normalizeAliasToken(al) == normAlias {
				return true
			}
		}
	}
	return false
}

func (e *Engine) relationAuthorizedForQueryLocked(t *tenantIndex, sr *storedRelation, pSet map[string]struct{}, queryTokens map[string]struct{}) bool {
	if !e.relationHasAuthorizedSourceLocked(t, sr, pSet) {
		return false
	}
	allowedCount := 0
	for _, sid := range sr.sourceIDs {
		if e.sourceAllowedLocked(t.tenantID, sid, pSet) {
			allowedCount++
		}
	}
	if allowedCount == len(sr.sourceIDs) {
		return true
	}
	mat, ok := e.materializeRelationLocked(t, sr, pSet)
	if !ok {
		return false
	}
	authText := mat.SourceEntityName + " " + string(mat.Type) + " " + mat.TargetEntityName + " " +
		strings.Join(mat.Keywords, " ") + " " + mat.Description
	return hasAnyTokenOverlap(queryTokens, authText)
}

// materializeEntityLocked builds a caller-safe Entity containing ONLY
// descriptions, keywords, aliases, and SourceIDs from ACL-authorized sources.
func (e *Engine) materializeEntityLocked(t *tenantIndex, se *storedEntity, pSet map[string]struct{}) (Entity, bool) {
	allAllowed := true
	allowedSids := make([]string, 0, len(se.sourceIDs))
	for _, sid := range se.sourceIDs {
		if e.sourceAllowedLocked(t.tenantID, sid, pSet) {
			allowedSids = append(allowedSids, sid)
		} else {
			allAllowed = false
		}
	}
	if len(allowedSids) == 0 {
		return Entity{}, false
	}

	// Count ACL-authorized incident degree.
	deg := 0
	for rid := range t.adj[se.id] {
		if sr := t.relations[rid]; sr != nil && e.relationHasAuthorizedSourceLocked(t, sr, pSet) {
			deg++
		}
	}

	if allAllowed {
		return Entity{
			ID:           se.id,
			CanonicalKey: se.canonicalKey,
			Name:         se.name,
			Type:         se.typ,
			Description:  se.description,
			Keywords:     cloneStringSlice(se.keywords),
			Aliases:      cloneStringSlice(se.aliases),
			SourceIDs:    cloneStringSlice(se.sourceIDs),
			MentionCount: se.mentionCount,
			Degree:       deg,
			Weight:       se.weight,
		}, true
	}

	// Recompute strictly from authorized contributing sources.
	temp := &storedEntity{
		id:           se.id,
		canonicalKey: se.canonicalKey,
		typ:          se.typ,
		contribs:     make(map[string]entityContrib, len(allowedSids)),
	}
	for _, sid := range allowedSids {
		temp.contribs[sid] = se.contribs[sid]
	}
	temp.recompute()
	return Entity{
		ID:           temp.id,
		CanonicalKey: temp.canonicalKey,
		Name:         temp.name,
		Type:         temp.typ,
		Description:  temp.description,
		Keywords:     cloneStringSlice(temp.keywords),
		Aliases:      cloneStringSlice(temp.aliases),
		SourceIDs:    cloneStringSlice(temp.sourceIDs),
		MentionCount: temp.mentionCount,
		Degree:       deg,
		Weight:       temp.weight,
	}, true
}

// materializeRelationLocked builds a caller-safe Relation strictly filtered to
// ACL-authorized contributing sources and authorized endpoints.
func (e *Engine) materializeRelationLocked(t *tenantIndex, sr *storedRelation, pSet map[string]struct{}) (Relation, bool) {
	se := t.entities[sr.srcEntityID]
	de := t.entities[sr.dstEntityID]
	if se == nil || de == nil {
		return Relation{}, false
	}
	srcEnt, srcOK := e.materializeEntityLocked(t, se, pSet)
	dstEnt, dstOK := e.materializeEntityLocked(t, de, pSet)
	if !srcOK || !dstOK {
		return Relation{}, false
	}

	allAllowed := true
	allowedSids := make([]string, 0, len(sr.sourceIDs))
	for _, sid := range sr.sourceIDs {
		if e.sourceAllowedLocked(t.tenantID, sid, pSet) {
			allowedSids = append(allowedSids, sid)
		} else {
			allAllowed = false
		}
	}
	if len(allowedSids) == 0 {
		return Relation{}, false
	}

	if allAllowed {
		return Relation{
			ID:               sr.id,
			SourceEntityID:   sr.srcEntityID,
			TargetEntityID:   sr.dstEntityID,
			SourceEntityName: srcEnt.Name,
			TargetEntityName: dstEnt.Name,
			Type:             sr.relType,
			Description:      sr.description,
			Keywords:         cloneStringSlice(sr.keywords),
			SourceIDs:        cloneStringSlice(sr.sourceIDs),
			Weight:           sr.weight,
		}, true
	}

	temp := &storedRelation{
		id:          sr.id,
		srcEntityID: sr.srcEntityID,
		dstEntityID: sr.dstEntityID,
		relType:     sr.relType,
		contribs:    make(map[string]relationContrib, len(allowedSids)),
	}
	for _, sid := range allowedSids {
		temp.contribs[sid] = sr.contribs[sid]
	}
	temp.recompute()
	return Relation{
		ID:               temp.id,
		SourceEntityID:   temp.srcEntityID,
		TargetEntityID:   temp.dstEntityID,
		SourceEntityName: srcEnt.Name,
		TargetEntityName: dstEnt.Name,
		Type:             temp.relType,
		Description:      temp.description,
		Keywords:         cloneStringSlice(temp.keywords),
		SourceIDs:        cloneStringSlice(temp.sourceIDs),
		Weight:           temp.weight,
	}, true
}

func (e *Engine) getOrCreateTenantLocked(tenantID string) *tenantIndex {
	t := e.tenants[tenantID]
	if t == nil {
		t = newTenantIndex(tenantID, e.cfg.VectorDim)
		e.tenants[tenantID] = t
	}
	return t
}

func validateSource(rec SourceRecord) error {
	if strings.TrimSpace(rec.TenantID) == "" {
		return ErrEmptyTenant
	}
	if strings.TrimSpace(rec.SourceID) == "" {
		return fmt.Errorf("%w: empty source_id", ErrInvalidSource)
	}
	if rec.Version < 1 {
		return fmt.Errorf("%w: version must be >= 1, got %d", ErrInvalidSource, rec.Version)
	}
	if strings.TrimSpace(rec.Content) == "" {
		return fmt.Errorf("%w: empty content", ErrInvalidSource)
	}
	return nil
}

func makePrincipalSet(principals []string) map[string]struct{} {
	if len(principals) == 0 {
		return nil
	}
	out := make(map[string]struct{}, len(principals))
	for _, p := range principals {
		p = strings.TrimSpace(p)
		if p != "" {
			out[p] = struct{}{}
		}
	}
	return out
}

func buildQueryCacheKey(q Query) string {
	prins := cloneStringSlice(q.Principals)
	sort.Strings(prins)
	return fmt.Sprintf("%s|%s|%s|%d|%d|%d|%t|%s",
		q.TenantID, strings.Join(prins, ","), q.Mode, q.TopK, q.MaxHops, q.MaxExpandedNodes, q.IncludeSubgraph, q.Text)
}

func makeTokenSet(text string) map[string]struct{} {
	toks := Tokenize(text)
	out := make(map[string]struct{}, len(toks))
	for _, t := range toks {
		out[t] = struct{}{}
	}
	return out
}

func hasAnyTokenOverlap(queryTokens map[string]struct{}, text string) bool {
	if len(queryTokens) == 0 {
		return false
	}
	for _, tok := range Tokenize(text) {
		if _, ok := queryTokens[tok]; ok {
			return true
		}
	}
	return false
}

func appendUnique(slice []string, item string) []string {
	for _, s := range slice {
		if s == item {
			return slice
		}
	}
	return append(slice, item)
}

func mergeRetrievedEntities(a, b []RetrievedEntity, limit int) []RetrievedEntity {
	byID := make(map[string]RetrievedEntity)
	for _, x := range append(a, b...) {
		if prev, ok := byID[x.Entity.ID]; !ok || x.Score > prev.Score {
			byID[x.Entity.ID] = x
		}
	}
	out := make([]RetrievedEntity, 0, len(byID))
	for _, v := range byID {
		out = append(out, v)
	}
	sort.Slice(out, func(i, j int) bool {
		if math.Abs(out[i].Score-out[j].Score) > 1e-12 {
			return out[i].Score > out[j].Score
		}
		return out[i].Entity.ID < out[j].Entity.ID
	})
	if len(out) > limit {
		out = out[:limit]
	}
	return out
}

func mergeRetrievedRelations(a, b []RetrievedRelation, limit int) []RetrievedRelation {
	byID := make(map[string]RetrievedRelation)
	for _, x := range append(a, b...) {
		if prev, ok := byID[x.Relation.ID]; !ok || x.Score > prev.Score {
			byID[x.Relation.ID] = x
		}
	}
	out := make([]RetrievedRelation, 0, len(byID))
	for _, v := range byID {
		out = append(out, v)
	}
	sort.Slice(out, func(i, j int) bool {
		if math.Abs(out[i].Score-out[j].Score) > 1e-12 {
			return out[i].Score > out[j].Score
		}
		return out[i].Relation.ID < out[j].Relation.ID
	})
	if len(out) > limit {
		out = out[:limit]
	}
	return out
}

func capTrails(trails []EvidenceTrail, limit int) []EvidenceTrail {
	if len(trails) > limit {
		return trails[:limit]
	}
	return trails
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
