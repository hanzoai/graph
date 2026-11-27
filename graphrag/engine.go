package graphrag

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

type derivedChunk struct {
	SourceRef SourceRef
	Title     string
	Snippet   string
	Content   string
	Themes    []string
	Tokens    []string
	TF        map[string]int
	Embedding [embeddingDim]float64
}

type derivedPartition struct {
	tenantID string
	scopeKey string
	acl      []string

	// Per-document derived contributions for O(|delta|) exact retraction.
	docExtractions   map[string]ExtractionResult
	docChunks        map[string]*derivedChunk
	chunkDF          map[string]int
	chunkTotalTokens int
	entityContribs   map[string]map[string]Entity   // entityID -> sourceID -> Entity
	relContribs      map[string]map[string]Relation // relKey -> sourceID -> Relation

	// Merged partition graph.
	entities  map[string]*Entity
	relations map[string]*Relation

	// Hierarchical community tree (L0 -> L1 -> L2) and cached summaries.
	communities  map[string]*Community
	byLevel      map[int][]string
	entityToL0   map[string]string
	summaries    map[string]*CommunitySummary
	modularityL0 float64
	modularityL1 float64
}

func newDerivedPartition(tenantID, scopeKey string, acl []string) *derivedPartition {
	return &derivedPartition{
		tenantID:       tenantID,
		scopeKey:       scopeKey,
		acl:            append([]string(nil), acl...),
		docExtractions: make(map[string]ExtractionResult),
		docChunks:      make(map[string]*derivedChunk),
		chunkDF:        make(map[string]int),
		entityContribs: make(map[string]map[string]Entity),
		relContribs:    make(map[string]map[string]Relation),
		entities:       make(map[string]*Entity),
		relations:      make(map[string]*Relation),
		communities:    make(map[string]*Community),
		byLevel:        make(map[int][]string),
		entityToL0:     make(map[string]string),
		summaries:      make(map[string]*CommunitySummary),
	}
}

// Engine manages canonical versioned documents and derived ACL-partitioned
// GraphRAG community hierarchies, hierarchical summaries, incremental
// invalidation, ordinary hybrid baseline chunks, and global-vs-local query routing.
type Engine struct {
	mu         sync.RWMutex
	cfg        Config
	extractor  Extractor
	summarizer Summarizer
	router     *QueryRouter

	// Canonical source-of-truth store: tenantID -> sourceID -> CanonicalDocument.
	canonical map[string]map[string]CanonicalDocument

	// Derived projections: tenantID -> scopeKey -> *derivedPartition.
	derived      map[string]map[string]*derivedPartition
	derivedBuilt bool

	lastBuild BuildStats
}

// NewEngine creates a new GraphRAG engine with the given Config.
func NewEngine(cfg Config) *Engine {
	c := cfg.normalized()
	ext := c.Extractor
	if ext == nil {
		ext = NewDefaultExtractor()
	}
	sum := c.Summarizer
	if sum == nil {
		sum = NewExtractiveHierarchicalSummarizer()
	}
	return &Engine{
		cfg:        c,
		extractor:  ext,
		summarizer: sum,
		router:     NewQueryRouter(c.DefaultGlobalLevel),
		canonical:  make(map[string]map[string]CanonicalDocument),
		derived:    make(map[string]map[string]*derivedPartition),
	}
}

// Config returns a copy of the normalized engine configuration.
func (e *Engine) Config() Config {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.cfg
}

// Router returns the engine's QueryRouter for inspection or pre-flight routing.
func (e *Engine) Router() *QueryRouter {
	return e.router
}

// UpsertDocument stores or updates a CanonicalDocument.
// If derived indexes have not yet been built (initial offline ingest), it
// records only the canonical truth so BuildAll can construct the initial
// community hierarchy in one pass.
// If derived indexes are already built, it performs targeted incremental
// entity/edge retraction + leaf-and-ancestor community summary invalidation and
// returns exact IncrementalRefreshStats.
func (e *Engine) UpsertDocument(doc CanonicalDocument) (IncrementalRefreshStats, error) {
	if strings.TrimSpace(doc.TenantID) == "" || strings.TrimSpace(doc.SourceID) == "" || strings.TrimSpace(doc.Version) == "" {
		return IncrementalRefreshStats{}, fmt.Errorf("%w: tenant_id, source_id, and version are required", ErrInvalidInput)
	}
	clean := doc.Clone()
	clean.TenantID = strings.TrimSpace(clean.TenantID)
	clean.SourceID = strings.TrimSpace(clean.SourceID)
	clean.Version = strings.TrimSpace(clean.Version)
	clean.ACL = NormalizeACL(clean.ACL)
	if clean.UpdatedAt.IsZero() {
		clean.UpdatedAt = time.Now().UTC()
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	tStore, ok := e.canonical[clean.TenantID]
	if !ok {
		tStore = make(map[string]CanonicalDocument)
		e.canonical[clean.TenantID] = tStore
	}
	oldDoc, existed := tStore[clean.SourceID]
	tStore[clean.SourceID] = clean

	if !e.derivedBuilt {
		return IncrementalRefreshStats{
			Operation: "stage_canonical",
			SourceID:  clean.SourceID,
		}, nil
	}

	start := time.Now()
	newScope := ACLScopeKey(clean.TenantID, clean.ACL)
	var stats IncrementalRefreshStats
	stats.SourceID = clean.SourceID

	if existed {
		oldScope := ACLScopeKey(oldDoc.TenantID, oldDoc.ACL)
		if oldScope == newScope {
			stats.Operation = "incremental_update"
			p := e.ensurePartitionLocked(clean.TenantID, newScope, clean.ACL)
			e.applyIncrementalUpsertLocked(p, clean, &stats)
		} else {
			// ACL scope changed (e.g., restriction or revocation): retract from old
			// partition and incrementally insert into new partition.
			stats.Operation = "incremental_acl_move"
			if tDerived, ok := e.derived[clean.TenantID]; ok {
				if oldP, ok := tDerived[oldScope]; ok {
					e.applyIncrementalDeleteLocked(oldP, oldDoc.SourceID, &stats)
					if len(oldP.docExtractions) == 0 {
						delete(tDerived, oldScope)
					}
				}
			}
			newP := e.ensurePartitionLocked(clean.TenantID, newScope, clean.ACL)
			if len(newP.communities) == 0 {
				e.ingestDocIntoPartitionLocked(newP, clean)
				e.fullBuildPartitionLocked(newP, &stats)
			} else {
				e.applyIncrementalUpsertLocked(newP, clean, &stats)
			}
		}
	} else {
		stats.Operation = "incremental_insert"
		p := e.ensurePartitionLocked(clean.TenantID, newScope, clean.ACL)
		if len(p.communities) == 0 {
			e.ingestDocIntoPartitionLocked(p, clean)
			e.fullBuildPartitionLocked(p, &stats)
		} else {
			e.applyIncrementalUpsertLocked(p, clean, &stats)
		}
	}

	stats.DurationMs = float64(time.Since(start).Nanoseconds()) / 1e6
	stats.TotalTokens = stats.PromptTokens + stats.CompletionTokens
	stats.EstimatedCostUSD = e.cfg.EstimateCostUSD(stats.PromptTokens, stats.CompletionTokens)
	return stats, nil
}

// DeleteDocument removes a CanonicalDocument by (tenantID, sourceID) and, if
// derived indexes are built, incrementally retracts its contributions and
// refreshes only affected leaf + ancestor community summaries.
func (e *Engine) DeleteDocument(tenantID, sourceID string) (IncrementalRefreshStats, error) {
	tenantID = strings.TrimSpace(tenantID)
	sourceID = strings.TrimSpace(sourceID)
	if tenantID == "" || sourceID == "" {
		return IncrementalRefreshStats{}, fmt.Errorf("%w: tenant_id and source_id are required", ErrInvalidInput)
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	tStore, ok := e.canonical[tenantID]
	if !ok {
		return IncrementalRefreshStats{}, ErrNotFound
	}
	oldDoc, existed := tStore[sourceID]
	if !existed {
		return IncrementalRefreshStats{}, ErrNotFound
	}
	delete(tStore, sourceID)
	if len(tStore) == 0 {
		delete(e.canonical, tenantID)
	}

	if !e.derivedBuilt {
		return IncrementalRefreshStats{
			Operation: "delete_canonical",
			SourceID:  sourceID,
		}, nil
	}

	start := time.Now()
	stats := IncrementalRefreshStats{
		Operation: "incremental_delete",
		SourceID:  sourceID,
	}
	oldScope := ACLScopeKey(oldDoc.TenantID, oldDoc.ACL)
	if tDerived, ok := e.derived[tenantID]; ok {
		if p, ok := tDerived[oldScope]; ok {
			e.applyIncrementalDeleteLocked(p, sourceID, &stats)
			if len(p.docExtractions) == 0 {
				delete(tDerived, oldScope)
			}
		}
		if len(tDerived) == 0 {
			delete(e.derived, tenantID)
		}
	}

	stats.DurationMs = float64(time.Since(start).Nanoseconds()) / 1e6
	stats.TotalTokens = stats.PromptTokens + stats.CompletionTokens
	stats.EstimatedCostUSD = e.cfg.EstimateCostUSD(stats.PromptTokens, stats.CompletionTokens)
	return stats, nil
}

// GetDocument returns a deep copy of a CanonicalDocument iff it belongs to
// caller.TenantID and caller.Principals satisfy its ACL. Otherwise returns
// ErrNotFound (zero existence-oracle leakage).
func (e *Engine) GetDocument(caller CallerContext, sourceID string) (CanonicalDocument, error) {
	if strings.TrimSpace(caller.TenantID) == "" {
		return CanonicalDocument{}, ErrUnauthorized
	}
	e.mu.RLock()
	defer e.mu.RUnlock()

	tStore, ok := e.canonical[caller.TenantID]
	if !ok {
		return CanonicalDocument{}, ErrNotFound
	}
	doc, ok := tStore[sourceID]
	if !ok || !ACLAllows(doc.ACL, caller.Principals) {
		return CanonicalDocument{}, ErrNotFound
	}
	return doc.Clone(), nil
}

// CanonicalCount returns the total number of canonical documents stored for a tenant.
func (e *Engine) CanonicalCount(tenantID string) int {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return len(e.canonical[tenantID])
}

// DropDerivedIndexes deletes all derived entities, relations, chunks,
// communities, and summaries while preserving 100% of canonical documents.
func (e *Engine) DropDerivedIndexes() {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.derived = make(map[string]map[string]*derivedPartition)
	e.derivedBuilt = false
}

// BuildAll builds all derived ACL-partitioned graphs, multi-level communities
// (L0 -> L1 -> L2), hierarchical summaries, and local hybrid indexes from
// canonical documents.
func (e *Engine) BuildAll() BuildStats {
	return e.RebuildAll()
}

// RebuildAll drops any existing derived projections and deterministically
// reconstructs all derived indexes from canonical truth.
func (e *Engine) RebuildAll() BuildStats {
	e.mu.Lock()
	defer e.mu.Unlock()

	start := time.Now()
	e.derived = make(map[string]map[string]*derivedPartition)

	stats := BuildStats{
		CommunitiesByLevel: make(map[int]int),
	}

	tenantIDs := make([]string, 0, len(e.canonical))
	for tid := range e.canonical {
		tenantIDs = append(tenantIDs, tid)
	}
	sort.Strings(tenantIDs)

	var mod0Weighted, mod1Weighted float64
	var totalModEntities int

	for _, tid := range tenantIDs {
		tStore := e.canonical[tid]
		docIDs := make([]string, 0, len(tStore))
		for sid := range tStore {
			docIDs = append(docIDs, sid)
		}
		sort.Strings(docIDs)

		for _, sid := range docIDs {
			doc := tStore[sid]
			scopeKey := ACLScopeKey(doc.TenantID, doc.ACL)
			p := e.ensurePartitionLocked(doc.TenantID, scopeKey, doc.ACL)
			e.ingestDocIntoPartitionLocked(p, doc)
			stats.DocumentsIndexed++
		}

		// Build community hierarchy and summaries per ACL partition in sorted order.
		tDerived := e.derived[tid]
		scopeKeys := make([]string, 0, len(tDerived))
		for sk := range tDerived {
			scopeKeys = append(scopeKeys, sk)
		}
		sort.Strings(scopeKeys)

		for _, sk := range scopeKeys {
			p := tDerived[sk]
			e.rebuildPartitionHierarchyAndSummariesLocked(p)
			stats.PartitionsBuilt++
			stats.EntitiesTotal += len(p.entities)
			stats.RelationsTotal += len(p.relations)
			for lvl, cids := range p.byLevel {
				stats.CommunitiesByLevel[lvl] += len(cids)
			}
			for _, s := range p.summaries {
				stats.SummariesBuilt++
				stats.PromptTokens += s.PromptTokens
				stats.CompletionTokens += s.CompletionTokens
			}
			w := len(p.entities)
			mod0Weighted += p.modularityL0 * float64(w)
			mod1Weighted += p.modularityL1 * float64(w)
			totalModEntities += w
		}
	}

	e.derivedBuilt = true
	if totalModEntities > 0 {
		stats.MeanModularityL0 = mod0Weighted / float64(totalModEntities)
		stats.MeanModularityL1 = mod1Weighted / float64(totalModEntities)
	}
	stats.DurationMs = float64(time.Since(start).Nanoseconds()) / 1e6
	stats.TotalTokens = stats.PromptTokens + stats.CompletionTokens
	stats.EstimatedCostUSD = e.cfg.EstimateCostUSD(stats.PromptTokens, stats.CompletionTokens)
	stats.DerivedDigest = e.derivedDigestLocked()
	e.lastBuild = stats
	return stats
}

// DerivedDigest returns a deterministic SHA-256 hex digest over all derived
// graphs, hierarchies, and summaries to verify bit-exact drop/rebuild parity.
func (e *Engine) DerivedDigest() (string, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if !e.derivedBuilt {
		return "", ErrDerivedIndexNotBuilt
	}
	return e.derivedDigestLocked(), nil
}

func (e *Engine) derivedDigestLocked() string {
	h := sha256.New()
	write := func(s string) {
		h.Write([]byte(s))
		h.Write([]byte{0})
	}

	tids := make([]string, 0, len(e.derived))
	for tid := range e.derived {
		tids = append(tids, tid)
	}
	sort.Strings(tids)

	for _, tid := range tids {
		write("T:" + tid)
		tDerived := e.derived[tid]
		sks := make([]string, 0, len(tDerived))
		for sk := range tDerived {
			sks = append(sks, sk)
		}
		sort.Strings(sks)
		for _, sk := range sks {
			p := tDerived[sk]
			write("S:" + sk + ":" + strings.Join(p.acl, ","))

			eids := make([]string, 0, len(p.entities))
			for id := range p.entities {
				eids = append(eids, id)
			}
			sort.Strings(eids)
			for _, id := range eids {
				ent := p.entities[id]
				write(fmt.Sprintf("E:%s|%s|%.6f|%s", ent.ID, ent.Type, ent.Weight, strings.Join(ent.Themes, ",")))
			}

			rks := make([]string, 0, len(p.relations))
			for k := range p.relations {
				rks = append(rks, k)
			}
			sort.Strings(rks)
			for _, k := range rks {
				rel := p.relations[k]
				write(fmt.Sprintf("R:%s|%s|%s|%.6f", rel.SourceEntityID, rel.TargetEntityID, rel.Kind, rel.Weight))
			}

			for lvl := 0; lvl < e.cfg.MaxHierarchyLevels; lvl++ {
				cids := append([]string(nil), p.byLevel[lvl]...)
				sort.Strings(cids)
				for _, cid := range cids {
					c := p.communities[cid]
					write(fmt.Sprintf("C:L%d|%s|P=%s|Ch=%s|Ent=%s", c.Level, c.ID, c.ParentID, strings.Join(c.ChildIDs, ","), strings.Join(c.EntityIDs, ",")))
					if sum, ok := p.summaries[cid]; ok {
						write(fmt.Sprintf("SUM:%s|%s|%s|%d|%d", sum.CommunityID, sum.Title, sum.Summary, sum.PromptTokens, sum.CompletionTokens))
					}
				}
			}
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}

// GetCommunitySummary returns a deep copy of a community summary iff it exists
// in caller.TenantID and caller.Principals satisfy its ACL. Returns ErrNotFound
// for foreign tenants or unauthorized scopes (zero existence-oracle leakage).
func (e *Engine) GetCommunitySummary(caller CallerContext, communityID string) (CommunitySummary, error) {
	if strings.TrimSpace(caller.TenantID) == "" {
		return CommunitySummary{}, ErrUnauthorized
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	if !e.derivedBuilt {
		return CommunitySummary{}, ErrDerivedIndexNotBuilt
	}
	tDerived, ok := e.derived[caller.TenantID]
	if !ok {
		return CommunitySummary{}, ErrNotFound
	}
	scopeKeys := make([]string, 0, len(tDerived))
	for sk := range tDerived {
		scopeKeys = append(scopeKeys, sk)
	}
	sort.Strings(scopeKeys)
	for _, sk := range scopeKeys {
		p := tDerived[sk]
		if !ACLAllows(p.acl, caller.Principals) {
			continue
		}
		if sum, exists := p.summaries[communityID]; exists {
			return sum.Clone(), nil
		}
	}
	return CommunitySummary{}, ErrNotFound
}

// ListCommunities returns deep-copied communities at the requested hierarchy
// level (or all levels if level < 0) across all partitions authorized for caller.
func (e *Engine) ListCommunities(caller CallerContext, level int) ([]Community, error) {
	if strings.TrimSpace(caller.TenantID) == "" {
		return nil, ErrUnauthorized
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	if !e.derivedBuilt {
		return nil, ErrDerivedIndexNotBuilt
	}
	tDerived, ok := e.derived[caller.TenantID]
	if !ok {
		return nil, nil
	}
	scopeKeys := make([]string, 0, len(tDerived))
	for sk := range tDerived {
		scopeKeys = append(scopeKeys, sk)
	}
	sort.Strings(scopeKeys)

	var out []Community
	for _, sk := range scopeKeys {
		p := tDerived[sk]
		if !ACLAllows(p.acl, caller.Principals) {
			continue
		}
		for lvl := 0; lvl < e.cfg.MaxHierarchyLevels; lvl++ {
			if level >= 0 && lvl != level {
				continue
			}
			for _, cid := range p.byLevel[lvl] {
				if c, ok := p.communities[cid]; ok {
					out = append(out, c.Clone())
				}
			}
		}
	}
	return out, nil
}

// Query executes a query under caller's strict TenantID + ACL scope using
// ModeAuto (QueryRouter gate), ModeGlobalL0/L1/L2 (GraphRAG Map-Reduce), or
// ModeLocalHybrid (ordinary top-K BM25 + dense chunk baseline).
func (e *Engine) Query(ctx context.Context, caller CallerContext, req QueryRequest) (QueryResponse, error) {
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return QueryResponse{}, err
		}
	}
	if strings.TrimSpace(caller.TenantID) == "" {
		return QueryResponse{}, ErrUnauthorized
	}
	if strings.TrimSpace(req.Query) == "" {
		return QueryResponse{}, fmt.Errorf("%w: query cannot be empty", ErrInvalidInput)
	}

	e.mu.RLock()
	defer e.mu.RUnlock()

	if !e.derivedBuilt {
		return QueryResponse{}, ErrDerivedIndexNotBuilt
	}

	start := time.Now()
	requestedMode := req.Mode
	if requestedMode == "" {
		requestedMode = ModeAuto
	}
	topK := req.TopK
	if topK <= 0 {
		topK = e.cfg.DefaultTopK
	}
	tokenBudget := req.TokenBudget
	if tokenBudget <= 0 {
		tokenBudget = e.cfg.DefaultTokenBudget
	}

	routing := e.router.Route(req.Query)
	effectiveMode := requestedMode
	if requestedMode == ModeAuto {
		effectiveMode = routing.RecommendedMode
	}

	// Select ONLY partitions belonging to caller.TenantID whose ACL allows caller.
	var authPartitions []*derivedPartition
	var partitionsDenied int
	if tDerived, ok := e.derived[caller.TenantID]; ok {
		scopeKeys := make([]string, 0, len(tDerived))
		for sk := range tDerived {
			scopeKeys = append(scopeKeys, sk)
		}
		sort.Strings(scopeKeys)
		for _, sk := range scopeKeys {
			p := tDerived[sk]
			if ACLAllows(p.acl, caller.Principals) {
				authPartitions = append(authPartitions, p)
			} else {
				partitionsDenied++
			}
		}
	}

	resp := QueryResponse{
		Query:              req.Query,
		RequestedMode:      requestedMode,
		EffectiveMode:      effectiveMode,
		Routing:            routing.Clone(),
		CommunityLevelUsed: -1,
		Metrics: QueryMetrics{
			PartitionsAuthorized: len(authPartitions),
			PartitionsDenied:     partitionsDenied,
		},
	}

	switch effectiveMode {
	case ModeLocalHybrid:
		e.execLocalHybridLocked(ctx, caller, req.Query, topK, tokenBudget, authPartitions, &resp)
	case ModeGlobalL0:
		e.execGlobalMapReduceLocked(ctx, caller, req.Query, 0, tokenBudget, req.MaxMapGroups, authPartitions, &resp)
	case ModeGlobalL1:
		e.execGlobalMapReduceLocked(ctx, caller, req.Query, 1, tokenBudget, req.MaxMapGroups, authPartitions, &resp)
	case ModeGlobalL2:
		e.execGlobalMapReduceLocked(ctx, caller, req.Query, 2, tokenBudget, req.MaxMapGroups, authPartitions, &resp)
	default:
		return QueryResponse{}, fmt.Errorf("%w: unsupported mode %q", ErrInvalidInput, effectiveMode)
	}

	resp.Metrics.LatencyMs = float64(time.Since(start).Nanoseconds()) / 1e6
	resp.Metrics.TotalTokens = resp.Metrics.PromptTokens + resp.Metrics.CompletionTokens
	resp.Metrics.EstimatedCostUSD = e.cfg.EstimateCostUSD(resp.Metrics.PromptTokens, resp.Metrics.CompletionTokens)
	return resp.Clone(), nil
}

func (e *Engine) execGlobalMapReduceLocked(
	ctx context.Context,
	caller CallerContext,
	query string,
	level int,
	tokenBudget int,
	maxMapGroups int,
	authPartitions []*derivedPartition,
	resp *QueryResponse,
) {
	if level >= e.cfg.MaxHierarchyLevels {
		level = e.cfg.MaxHierarchyLevels - 1
	}
	resp.CommunityLevelUsed = level

	var summaries []CommunitySummary
	for _, p := range authPartitions {
		// Defensive re-verification of tenant & ACL boundary.
		if p.tenantID != caller.TenantID || !ACLAllows(p.acl, caller.Principals) {
			continue
		}
		cids := p.byLevel[level]
		for _, cid := range cids {
			if sum, ok := p.summaries[cid]; ok {
				if sum.TenantID != caller.TenantID || !ACLAllows(sum.ACL, caller.Principals) {
					continue
				}
				summaries = append(summaries, sum.Clone())
			}
		}
	}

	sort.Slice(summaries, func(i, j int) bool {
		return summaries[i].CommunityID < summaries[j].CommunityID
	})
	resp.Metrics.CommunitiesScanned = len(summaries)

	if len(summaries) == 0 {
		resp.Answer = "No authorized community summaries available for this caller."
		return
	}

	// Build summary-corpus DF and avgLen for Map-phase BM25 scoring.
	df := make(map[string]int)
	totalLen := 0
	for _, sum := range summaries {
		txt := sum.Title + "\n" + sum.Summary + "\n" + strings.Join(sum.Findings, "\n")
		toks := Tokenize(txt)
		totalLen += len(toks)
		seen := make(map[string]struct{}, len(toks))
		for _, t := range toks {
			if _, dup := seen[t]; !dup {
				seen[t] = struct{}{}
				df[t]++
			}
		}
	}
	avgLen := float64(totalLen) / float64(len(summaries))

	// Map phase over authorized community summaries.
	var partials []PartialAnswer
	for _, sum := range summaries {
		if ctx != nil && ctx.Err() != nil {
			break
		}
		pa := e.summarizer.MapCommunity(query, sum, df, len(summaries), avgLen)
		resp.Metrics.PromptTokens += pa.PromptTokens
		resp.Metrics.CompletionTokens += pa.CompletionTokens
		resp.Metrics.CommunitiesMapped++
		if len(pa.KeyPoints) > 0 {
			partials = append(partials, pa)
		}
	}

	sort.Slice(partials, func(i, j int) bool {
		if partials[i].RelevanceScore != partials[j].RelevanceScore {
			return partials[i].RelevanceScore > partials[j].RelevanceScore
		}
		return partials[i].CommunityID < partials[j].CommunityID
	})
	if maxMapGroups > 0 && len(partials) > maxMapGroups {
		partials = partials[:maxMapGroups]
	}

	// Reduce phase.
	ans, themes, citations, redPrompt, redComp := e.summarizer.ReducePartials(query, partials, tokenBudget)
	resp.Metrics.PromptTokens += redPrompt
	resp.Metrics.CompletionTokens += redComp

	// Defensive pre-return canonical ACL check on every emitted citation.
	resp.Citations = e.filterAuthorizedCitationsLocked(caller, citations)
	resp.PartialAnswers = partials
	resp.DetectedThemes = themes
	resp.Answer = ans
}

func (e *Engine) execLocalHybridLocked(
	ctx context.Context,
	caller CallerContext,
	query string,
	topK int,
	tokenBudget int,
	authPartitions []*derivedPartition,
	resp *QueryResponse,
) {
	var chunks []*derivedChunk
	for _, p := range authPartitions {
		if p.tenantID != caller.TenantID || !ACLAllows(p.acl, caller.Principals) {
			continue
		}
		sids := make([]string, 0, len(p.docChunks))
		for sid := range p.docChunks {
			sids = append(sids, sid)
		}
		sort.Strings(sids)
		for _, sid := range sids {
			chunks = append(chunks, p.docChunks[sid])
		}
	}

	resp.Metrics.ChunksScanned = len(chunks)
	if len(chunks) == 0 {
		resp.Answer = "No authorized canonical documents available for this caller."
		return
	}

	qTokens := Tokenize(query)
	qVec := EmbedText(query)
	qLower := strings.ToLower(query)

	// Aggregate DF strictly for query terms across authorized partitions only
	// (O(|qTokens| * |authPartitions|), zero leakage of unauthorized DF stats).
	df := make(map[string]int, len(qTokens))
	totalLen := 0
	for _, p := range authPartitions {
		if p.tenantID != caller.TenantID || !ACLAllows(p.acl, caller.Principals) {
			continue
		}
		totalLen += p.chunkTotalTokens
		for _, qt := range qTokens {
			if c := p.chunkDF[qt]; c > 0 {
				df[qt] += c
			}
		}
	}
	avgLen := float64(totalLen) / float64(len(chunks))

	hits := make([]SourceHit, 0, len(chunks))
	for _, ch := range chunks {
		if ctx != nil && ctx.Err() != nil {
			break
		}
		bm25 := BM25Score(qTokens, ch.TF, len(ch.Tokens), avgLen, df, len(chunks))
		cos := CosineSimilarity(qVec, ch.Embedding)

		// Exact source path / title / symbol boost for exact lookups.
		exactBoost := 0.0
		if strings.Contains(qLower, strings.ToLower(ch.SourceRef.SourceID)) {
			exactBoost += 2.5
		}
		if ch.Title != "" && strings.Contains(qLower, strings.ToLower(ch.Title)) {
			exactBoost += 1.5
		}
		score := 0.55*math.Tanh(bm25/4.0) + 0.35*cos + exactBoost
		hits = append(hits, SourceHit{
			SourceRef: ch.SourceRef,
			Title:     ch.Title,
			Snippet:   ch.Snippet,
			Score:     score,
			Themes:    append([]string(nil), ch.Themes...),
		})
	}

	sort.Slice(hits, func(i, j int) bool {
		if hits[i].Score != hits[j].Score {
			return hits[i].Score > hits[j].Score
		}
		return hits[i].SourceRef.SourceID < hits[j].SourceRef.SourceID
	})
	if len(hits) > topK {
		hits = hits[:topK]
	}

	ans, themes, citations, promptTok, compTok := e.summarizer.SynthesizeLocal(query, hits, tokenBudget)
	resp.Metrics.PromptTokens += promptTok
	resp.Metrics.CompletionTokens += compTok
	resp.SourceHits = hits
	resp.DetectedThemes = themes
	resp.Citations = e.filterAuthorizedCitationsLocked(caller, citations)
	resp.Answer = ans
}

func (e *Engine) filterAuthorizedCitationsLocked(caller CallerContext, refs []SourceRef) []SourceRef {
	tStore, ok := e.canonical[caller.TenantID]
	if !ok {
		return nil
	}
	var allowed []SourceRef
	for _, r := range refs {
		if r.TenantID != caller.TenantID {
			continue
		}
		doc, exists := tStore[r.SourceID]
		if !exists || !ACLAllows(doc.ACL, caller.Principals) {
			continue
		}
		allowed = append(allowed, SourceRef{
			SourceID: doc.SourceID,
			Version:  doc.Version,
			TenantID: doc.TenantID,
		})
	}
	return sortSourceRefs(allowed)
}

func (e *Engine) ensurePartitionLocked(tenantID, scopeKey string, acl []string) *derivedPartition {
	tDerived, ok := e.derived[tenantID]
	if !ok {
		tDerived = make(map[string]*derivedPartition)
		e.derived[tenantID] = tDerived
	}
	p, ok := tDerived[scopeKey]
	if !ok {
		p = newDerivedPartition(tenantID, scopeKey, acl)
		tDerived[scopeKey] = p
	}
	return p
}

func (e *Engine) ingestDocIntoPartitionLocked(p *derivedPartition, doc CanonicalDocument) {
	ext := e.extractor.Extract(doc)
	p.docExtractions[doc.SourceID] = ext

	if prevChunk, existed := p.docChunks[doc.SourceID]; existed {
		p.removeChunkStats(prevChunk)
	}

	fullText := doc.SourceID + "\n" + doc.Title + "\n" + doc.Content
	toks := Tokenize(fullText)
	ch := &derivedChunk{
		SourceRef: SourceRef{SourceID: doc.SourceID, Version: doc.Version, TenantID: doc.TenantID},
		Title:     doc.Title,
		Snippet:   firstSentence(doc.Content, 260),
		Content:   doc.Content,
		Themes:    append([]string(nil), ext.Themes...),
		Tokens:    toks,
		TF:        TermFreq(toks),
		Embedding: EmbedText(fullText),
	}
	p.docChunks[doc.SourceID] = ch
	p.addChunkStats(ch)

	for _, ent := range ext.Entities {
		if _, ok := p.entityContribs[ent.ID]; !ok {
			p.entityContribs[ent.ID] = make(map[string]Entity)
		}
		p.entityContribs[ent.ID][doc.SourceID] = ent.Clone()
		p.recomputeEntity(ent.ID)
	}
	for _, rel := range ext.Relations {
		rk := relationKey(rel)
		if _, ok := p.relContribs[rk]; !ok {
			p.relContribs[rk] = make(map[string]Relation)
		}
		p.relContribs[rk][doc.SourceID] = rel.Clone()
		p.recomputeRelation(rk)
	}
}

func (e *Engine) rebuildPartitionHierarchyAndSummariesLocked(p *derivedPartition) {
	h := buildCommunityHierarchy(p.tenantID, p.scopeKey, p.acl, p.entities, p.relations, e.cfg)
	p.communities = h.Communities
	p.byLevel = h.ByLevel
	p.entityToL0 = h.EntityToL0
	p.modularityL0 = h.ModularityL0
	p.modularityL1 = h.ModularityL1
	p.summaries = make(map[string]*CommunitySummary, len(p.communities))

	// Level 0 summaries from entities & internal relations.
	for _, cid := range p.byLevel[0] {
		c := p.communities[cid]
		ents, rels := p.communitySubgraph(c.EntityIDs)
		sum := e.summarizer.SummarizeLeaf(c.Clone(), ents, rels, 0)
		p.summaries[cid] = &sum
	}

	// Level 1 and Level 2 hierarchical rollups from child summaries.
	for lvl := 1; lvl < e.cfg.MaxHierarchyLevels; lvl++ {
		for _, cid := range p.byLevel[lvl] {
			c := p.communities[cid]
			children := make([]CommunitySummary, 0, len(c.ChildIDs))
			for _, childID := range c.ChildIDs {
				if cs, ok := p.summaries[childID]; ok {
					children = append(children, cs.Clone())
				}
			}
			sum := e.summarizer.SummarizeParent(c.Clone(), children, 0)
			p.summaries[cid] = &sum
		}
	}
}

func (e *Engine) fullBuildPartitionLocked(p *derivedPartition, stats *IncrementalRefreshStats) {
	e.rebuildPartitionHierarchyAndSummariesLocked(p)
	stats.DirtyL0 += len(p.byLevel[0])
	stats.TotalL0 += len(p.byLevel[0])
	stats.DirtyL1 += len(p.byLevel[1])
	stats.TotalL1 += len(p.byLevel[1])
	stats.DirtyL2 += len(p.byLevel[2])
	stats.TotalL2 += len(p.byLevel[2])
	for _, s := range p.summaries {
		stats.CommunitiesResummarized++
		stats.PromptTokens += s.PromptTokens
		stats.CompletionTokens += s.CompletionTokens
	}
}

// applyIncrementalUpsertLocked retracts old document contributions, merges new
// contributions, updates only affected L0 communities, and invalidates +
// re-summarizes strictly the dirty L0 communities and their L1 -> L2 ancestors.
func (e *Engine) applyIncrementalUpsertLocked(p *derivedPartition, doc CanonicalDocument, stats *IncrementalRefreshStats) {
	dirtyEntities := make(map[string]struct{})
	dirtyL0 := make(map[string]struct{})

	// 1. Retract previous contributions if the document already existed in p.
	if oldExt, ok := p.docExtractions[doc.SourceID]; ok {
		for _, ent := range oldExt.Entities {
			dirtyEntities[ent.ID] = struct{}{}
			if l0, ok := p.entityToL0[ent.ID]; ok {
				dirtyL0[l0] = struct{}{}
			}
			if m, ok := p.entityContribs[ent.ID]; ok {
				delete(m, doc.SourceID)
				p.recomputeEntity(ent.ID)
			}
		}
		for _, rel := range oldExt.Relations {
			dirtyEntities[rel.SourceEntityID] = struct{}{}
			dirtyEntities[rel.TargetEntityID] = struct{}{}
			if l0, ok := p.entityToL0[rel.SourceEntityID]; ok {
				dirtyL0[l0] = struct{}{}
			}
			if l0, ok := p.entityToL0[rel.TargetEntityID]; ok {
				dirtyL0[l0] = struct{}{}
			}
			rk := relationKey(rel)
			if m, ok := p.relContribs[rk]; ok {
				delete(m, doc.SourceID)
				p.recomputeRelation(rk)
			}
		}
	}

	// 2. Apply new document extraction and chunk.
	e.ingestDocIntoPartitionLocked(p, doc)
	newExt := p.docExtractions[doc.SourceID]
	for _, ent := range newExt.Entities {
		dirtyEntities[ent.ID] = struct{}{}
		if l0, ok := p.entityToL0[ent.ID]; ok {
			dirtyL0[l0] = struct{}{}
		}
	}
	for _, rel := range newExt.Relations {
		dirtyEntities[rel.SourceEntityID] = struct{}{}
		dirtyEntities[rel.TargetEntityID] = struct{}{}
		if l0, ok := p.entityToL0[rel.SourceEntityID]; ok {
			dirtyL0[l0] = struct{}{}
		}
		if l0, ok := p.entityToL0[rel.TargetEntityID]; ok {
			dirtyL0[l0] = struct{}{}
		}
	}

	e.refreshDirtyHierarchyLocked(p, dirtyEntities, dirtyL0, stats)
}

// applyIncrementalDeleteLocked retracts a deleted document's entities and
// relations and refreshes strictly affected leaf + ancestor summaries.
func (e *Engine) applyIncrementalDeleteLocked(p *derivedPartition, sourceID string, stats *IncrementalRefreshStats) {
	dirtyEntities := make(map[string]struct{})
	dirtyL0 := make(map[string]struct{})

	oldExt, ok := p.docExtractions[sourceID]
	if !ok {
		return
	}
	delete(p.docExtractions, sourceID)
	if oldChunk, ok := p.docChunks[sourceID]; ok {
		p.removeChunkStats(oldChunk)
		delete(p.docChunks, sourceID)
	}

	for _, ent := range oldExt.Entities {
		dirtyEntities[ent.ID] = struct{}{}
		if l0, ok := p.entityToL0[ent.ID]; ok {
			dirtyL0[l0] = struct{}{}
		}
		if m, ok := p.entityContribs[ent.ID]; ok {
			delete(m, sourceID)
			p.recomputeEntity(ent.ID)
		}
	}
	for _, rel := range oldExt.Relations {
		dirtyEntities[rel.SourceEntityID] = struct{}{}
		dirtyEntities[rel.TargetEntityID] = struct{}{}
		if l0, ok := p.entityToL0[rel.SourceEntityID]; ok {
			dirtyL0[l0] = struct{}{}
		}
		if l0, ok := p.entityToL0[rel.TargetEntityID]; ok {
			dirtyL0[l0] = struct{}{}
		}
		rk := relationKey(rel)
		if m, ok := p.relContribs[rk]; ok {
			delete(m, sourceID)
			p.recomputeRelation(rk)
		}
	}

	e.refreshDirtyHierarchyLocked(p, dirtyEntities, dirtyL0, stats)
}

// refreshDirtyHierarchyLocked updates L0 membership for dirtyEntities (removing
// dead entities, attaching brand-new entities to their strongest neighboring L0
// or a new L0 community), then re-summarizes ONLY dirty L0 communities and
// their strict L1 -> L2 ancestors.
func (e *Engine) refreshDirtyHierarchyLocked(
	p *derivedPartition,
	dirtyEntities map[string]struct{},
	dirtyL0 map[string]struct{},
	stats *IncrementalRefreshStats,
) {
	stats.DirtyEntities += len(dirtyEntities)

	sortedDirtyEnts := make([]string, 0, len(dirtyEntities))
	for eid := range dirtyEntities {
		sortedDirtyEnts = append(sortedDirtyEnts, eid)
	}
	sort.Strings(sortedDirtyEnts)

	// Build fast neighbor lookup for dirty entities.
	entityNeighbors := make(map[string]map[string]float64)
	for _, rel := range p.relations {
		_, d1 := dirtyEntities[rel.SourceEntityID]
		_, d2 := dirtyEntities[rel.TargetEntityID]
		if !d1 && !d2 {
			continue
		}
		if d1 {
			if _, ok := entityNeighbors[rel.SourceEntityID]; !ok {
				entityNeighbors[rel.SourceEntityID] = make(map[string]float64)
			}
			entityNeighbors[rel.SourceEntityID][rel.TargetEntityID] += rel.Weight
		}
		if d2 {
			if _, ok := entityNeighbors[rel.TargetEntityID]; !ok {
				entityNeighbors[rel.TargetEntityID] = make(map[string]float64)
			}
			entityNeighbors[rel.TargetEntityID][rel.SourceEntityID] += rel.Weight
		}
	}

	var unattachedNew []string
	for _, eid := range sortedDirtyEnts {
		_, stillExists := p.entities[eid]
		oldL0, hadL0 := p.entityToL0[eid]

		if !stillExists {
			if hadL0 {
				dirtyL0[oldL0] = struct{}{}
				if c, ok := p.communities[oldL0]; ok {
					c.EntityIDs = removeString(c.EntityIDs, eid)
				}
				delete(p.entityToL0, eid)
			}
			continue
		}

		if !hadL0 {
			// New entity: attach to the existing L0 community with strongest edge weight.
			commWeights := make(map[string]float64)
			for nb, w := range entityNeighbors[eid] {
				if nbL0, ok := p.entityToL0[nb]; ok {
					commWeights[nbL0] += w
				}
			}
			bestL0 := ""
			bestW := 0.0
			for _, cid := range sortedKeysFloat(commWeights) {
				if commWeights[cid] > bestW {
					bestW = commWeights[cid]
					bestL0 = cid
				}
			}
			if bestL0 != "" {
				p.entityToL0[eid] = bestL0
				if c, ok := p.communities[bestL0]; ok {
					c.EntityIDs = dedupStringsSorted(append(c.EntityIDs, eid))
				}
				dirtyL0[bestL0] = struct{}{}
			} else {
				unattachedNew = append(unattachedNew, eid)
			}
		}
	}

	// Any newly introduced disconnected cluster forms a new L0 community and
	// attaches to an existing or new L1 -> L2 ancestor.
	dirtyL1 := make(map[string]struct{})
	dirtyL2 := make(map[string]struct{})

	if len(unattachedNew) > 0 {
		members := dedupStringsSorted(unattachedNew)
		newL0ID := canonicalCommunityID(p.tenantID, p.scopeKey, 0, members)
		// Avoid accidental collision with an existing community ID.
		for _, exists := p.communities[newL0ID]; exists; _, exists = p.communities[newL0ID] {
			newL0ID += "-inc"
		}
		newL0 := &Community{
			ID:         newL0ID,
			TenantID:   p.tenantID,
			ScopeKey:   p.scopeKey,
			ACL:        append([]string(nil), p.acl...),
			Level:      0,
			EntityIDs:  members,
			SourceRefs: collectEntitySourceRefs(p.entities, members),
		}
		p.communities[newL0ID] = newL0
		p.byLevel[0] = dedupStringsSorted(append(p.byLevel[0], newL0ID))
		for _, eid := range members {
			p.entityToL0[eid] = newL0ID
		}
		dirtyL0[newL0ID] = struct{}{}

		if e.cfg.MaxHierarchyLevels > 1 {
			// Create a dedicated L1 parent (and L2 grandparent if enabled) for isolated new cluster.
			newL1ID := canonicalCommunityID(p.tenantID, p.scopeKey, 1, members)
			for _, exists := p.communities[newL1ID]; exists; _, exists = p.communities[newL1ID] {
				newL1ID += "-inc"
			}
			newL1 := &Community{
				ID:         newL1ID,
				TenantID:   p.tenantID,
				ScopeKey:   p.scopeKey,
				ACL:        append([]string(nil), p.acl...),
				Level:      1,
				ChildIDs:   []string{newL0ID},
				EntityIDs:  append([]string(nil), members...),
				SourceRefs: append([]SourceRef(nil), newL0.SourceRefs...),
			}
			newL0.ParentID = newL1ID
			p.communities[newL1ID] = newL1
			p.byLevel[1] = dedupStringsSorted(append(p.byLevel[1], newL1ID))
			dirtyL1[newL1ID] = struct{}{}

			if e.cfg.MaxHierarchyLevels > 2 {
				var l2ParentID string
				if len(p.byLevel[2]) > 0 {
					// Attach to smallest existing L2 macro community so L2 stays compact.
					l2ParentID = p.byLevel[2][0]
					minSize := len(p.communities[l2ParentID].EntityIDs)
					for _, cand := range p.byLevel[2][1:] {
						if sz := len(p.communities[cand].EntityIDs); sz < minSize {
							minSize = sz
							l2ParentID = cand
						}
					}
					l2 := p.communities[l2ParentID]
					l2.ChildIDs = dedupStringsSorted(append(l2.ChildIDs, newL1ID))
					newL1.ParentID = l2ParentID
					dirtyL2[l2ParentID] = struct{}{}
				} else {
					l2ParentID = canonicalCommunityID(p.tenantID, p.scopeKey, 2, members)
					newL2 := &Community{
						ID:         l2ParentID,
						TenantID:   p.tenantID,
						ScopeKey:   p.scopeKey,
						ACL:        append([]string(nil), p.acl...),
						Level:      2,
						ChildIDs:   []string{newL1ID},
						EntityIDs:  append([]string(nil), members...),
						SourceRefs: append([]SourceRef(nil), newL0.SourceRefs...),
					}
					newL1.ParentID = l2ParentID
					p.communities[l2ParentID] = newL2
					p.byLevel[2] = dedupStringsSorted(append(p.byLevel[2], l2ParentID))
					dirtyL2[l2ParentID] = struct{}{}
				}
			}
		}
	}

	// Step A: Refresh ONLY dirty L0 communities (prune empty L0s and propagate
	// dirtiness strictly to their L1 parents).
	sortedDirtyL0 := sortedSetKeys(dirtyL0)
	for _, cid := range sortedDirtyL0 {
		c, ok := p.communities[cid]
		if !ok {
			continue
		}
		if c.ParentID != "" {
			dirtyL1[c.ParentID] = struct{}{}
		}
		if len(c.EntityIDs) == 0 {
			// Prune empty L0 community.
			if parent, pok := p.communities[c.ParentID]; pok {
				parent.ChildIDs = removeString(parent.ChildIDs, cid)
			}
			delete(p.communities, cid)
			delete(p.summaries, cid)
			p.byLevel[0] = removeString(p.byLevel[0], cid)
			continue
		}
		c.SourceRefs = collectEntitySourceRefs(p.entities, c.EntityIDs)
		ents, rels := p.communitySubgraph(c.EntityIDs)
		var prevRev uint64
		if oldSum, ok := p.summaries[cid]; ok {
			prevRev = oldSum.Revision
		}
		sum := e.summarizer.SummarizeLeaf(c.Clone(), ents, rels, prevRev)
		p.summaries[cid] = &sum
		stats.DirtyL0++
		stats.CommunitiesResummarized++
		stats.PromptTokens += sum.PromptTokens
		stats.CompletionTokens += sum.CompletionTokens
	}

	// Step B: Refresh ONLY dirty L1 ancestors (prune empty L1s and propagate
	// dirtiness strictly to their L2 parents).
	if e.cfg.MaxHierarchyLevels > 1 {
		for _, cid := range sortedSetKeys(dirtyL1) {
			c, ok := p.communities[cid]
			if !ok {
				continue
			}
			if c.ParentID != "" {
				dirtyL2[c.ParentID] = struct{}{}
			}
			if len(c.ChildIDs) == 0 {
				if parent, pok := p.communities[c.ParentID]; pok {
					parent.ChildIDs = removeString(parent.ChildIDs, cid)
				}
				delete(p.communities, cid)
				delete(p.summaries, cid)
				p.byLevel[1] = removeString(p.byLevel[1], cid)
				continue
			}
			var childSums []CommunitySummary
			var allEnts []string
			for _, childID := range c.ChildIDs {
				if cc, ok := p.communities[childID]; ok {
					allEnts = append(allEnts, cc.EntityIDs...)
				}
				if cs, ok := p.summaries[childID]; ok {
					childSums = append(childSums, cs.Clone())
				}
			}
			c.EntityIDs = dedupStringsSorted(allEnts)
			c.SourceRefs = collectEntitySourceRefs(p.entities, c.EntityIDs)
			var prevRev uint64
			if oldSum, ok := p.summaries[cid]; ok {
				prevRev = oldSum.Revision
			}
			sum := e.summarizer.SummarizeParent(c.Clone(), childSums, prevRev)
			p.summaries[cid] = &sum
			stats.DirtyL1++
			stats.CommunitiesResummarized++
			stats.PromptTokens += sum.PromptTokens
			stats.CompletionTokens += sum.CompletionTokens
		}
	}

	// Step C: Refresh ONLY dirty L2 grandparents.
	if e.cfg.MaxHierarchyLevels > 2 {
		for _, cid := range sortedSetKeys(dirtyL2) {
			c, ok := p.communities[cid]
			if !ok {
				continue
			}
			if len(c.ChildIDs) == 0 {
				delete(p.communities, cid)
				delete(p.summaries, cid)
				p.byLevel[2] = removeString(p.byLevel[2], cid)
				continue
			}
			var childSums []CommunitySummary
			var allEnts []string
			for _, childID := range c.ChildIDs {
				if cc, ok := p.communities[childID]; ok {
					allEnts = append(allEnts, cc.EntityIDs...)
				}
				if cs, ok := p.summaries[childID]; ok {
					childSums = append(childSums, cs.Clone())
				}
			}
			c.EntityIDs = dedupStringsSorted(allEnts)
			c.SourceRefs = collectEntitySourceRefs(p.entities, c.EntityIDs)
			var prevRev uint64
			if oldSum, ok := p.summaries[cid]; ok {
				prevRev = oldSum.Revision
			}
			sum := e.summarizer.SummarizeParent(c.Clone(), childSums, prevRev)
			p.summaries[cid] = &sum
			stats.DirtyL2++
			stats.CommunitiesResummarized++
			stats.PromptTokens += sum.PromptTokens
			stats.CompletionTokens += sum.CompletionTokens
		}
	}

	totalL0 := len(p.byLevel[0])
	totalL1 := len(p.byLevel[1])
	totalL2 := len(p.byLevel[2])
	stats.TotalL0 += totalL0
	stats.TotalL1 += totalL1
	stats.TotalL2 += totalL2
	totalComms := totalL0 + totalL1 + totalL2
	reused := totalComms - (stats.DirtyL0 + stats.DirtyL1 + stats.DirtyL2)
	if reused < 0 {
		reused = 0
	}
	stats.CommunitiesReused += reused
}

func (p *derivedPartition) recomputeEntity(entityID string) {
	contribs := p.entityContribs[entityID]
	if len(contribs) == 0 {
		delete(p.entityContribs, entityID)
		delete(p.entities, entityID)
		return
	}
	sids := make([]string, 0, len(contribs))
	for sid := range contribs {
		sids = append(sids, sid)
	}
	sort.Strings(sids)

	var merged Entity
	var allThemes []string
	var allRefs []SourceRef
	var descs []string
	descSeen := make(map[string]struct{})

	for i, sid := range sids {
		c := contribs[sid]
		if i == 0 {
			merged.ID = c.ID
			merged.Name = c.Name
			merged.Type = c.Type
		}
		merged.Weight += c.Weight
		allThemes = append(allThemes, c.Themes...)
		allRefs = append(allRefs, c.SourceRefs...)
		if c.Description != "" && len(descs) < 3 {
			if _, dup := descSeen[c.Description]; !dup {
				descSeen[c.Description] = struct{}{}
				descs = append(descs, c.Description)
			}
		}
	}
	merged.Description = strings.Join(descs, "; ")
	merged.Themes = dedupStringsSorted(allThemes)
	merged.SourceRefs = sortSourceRefs(allRefs)
	p.entities[entityID] = &merged
}

func (p *derivedPartition) recomputeRelation(rk string) {
	contribs := p.relContribs[rk]
	if len(contribs) == 0 {
		delete(p.relContribs, rk)
		delete(p.relations, rk)
		return
	}
	sids := make([]string, 0, len(contribs))
	for sid := range contribs {
		sids = append(sids, sid)
	}
	sort.Strings(sids)

	var merged Relation
	var allRefs []SourceRef
	for i, sid := range sids {
		c := contribs[sid]
		if i == 0 {
			merged.SourceEntityID = c.SourceEntityID
			merged.TargetEntityID = c.TargetEntityID
			merged.Kind = c.Kind
			merged.Description = c.Description
		}
		merged.Weight += c.Weight
		allRefs = append(allRefs, c.SourceRefs...)
	}
	merged.SourceRefs = sortSourceRefs(allRefs)
	p.relations[rk] = &merged
}

func (p *derivedPartition) communitySubgraph(entityIDs []string) ([]Entity, []Relation) {
	memberSet := make(map[string]struct{}, len(entityIDs))
	ents := make([]Entity, 0, len(entityIDs))
	for _, eid := range entityIDs {
		memberSet[eid] = struct{}{}
		if ent, ok := p.entities[eid]; ok {
			ents = append(ents, ent.Clone())
		}
	}
	rks := make([]string, 0, len(p.relations))
	for k := range p.relations {
		rks = append(rks, k)
	}
	sort.Strings(rks)
	var rels []Relation
	for _, k := range rks {
		r := p.relations[k]
		if _, ok1 := memberSet[r.SourceEntityID]; !ok1 {
			continue
		}
		if _, ok2 := memberSet[r.TargetEntityID]; !ok2 {
			continue
		}
		rels = append(rels, r.Clone())
	}
	return ents, rels
}

func (p *derivedPartition) addChunkStats(ch *derivedChunk) {
	if ch == nil {
		return
	}
	p.chunkTotalTokens += len(ch.Tokens)
	for tok := range ch.TF {
		p.chunkDF[tok]++
	}
}

func (p *derivedPartition) removeChunkStats(ch *derivedChunk) {
	if ch == nil {
		return
	}
	p.chunkTotalTokens -= len(ch.Tokens)
	if p.chunkTotalTokens < 0 {
		p.chunkTotalTokens = 0
	}
	for tok := range ch.TF {
		if c := p.chunkDF[tok]; c <= 1 {
			delete(p.chunkDF, tok)
		} else {
			p.chunkDF[tok] = c - 1
		}
	}
}

func relationKey(r Relation) string {
	a, b := r.SourceEntityID, r.TargetEntityID
	if a > b {
		a, b = b, a
	}
	return a + "|" + b + "|" + r.Kind
}

func removeString(slice []string, target string) []string {
	out := make([]string, 0, len(slice))
	for _, s := range slice {
		if s != target {
			out = append(out, s)
		}
	}
	return out
}

func sortedSetKeys(m map[string]struct{}) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
