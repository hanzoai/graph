/*
 * SPDX-License-Identifier: Apache-2.0
 */

package hipporag

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"strings"
	"sync"
)

type indexedDoc struct {
	Record SourceRecord
	Tokens []string
	TF     map[string]int
	DocLen int
	Vec    DenseVec
	NodeID string
}

// tenantIndex contains all strictly derived indexes for a single tenant.
// Partitioning derived structures per tenant structurally prevents entity-name
// bridges from ever connecting different tenants.
type tenantIndex struct {
	tenantID     string
	docs         map[string]*indexedDoc
	docIDs       []string
	postings     map[string][]string
	avgDocLen    float64
	nodes        map[string]*GraphNode
	nodeVecs     map[string]DenseVec
	adj          map[string][]GraphEdge
	facts        []FactRecord
	factVecs     []DenseVec
	factTokens   [][]string
	entityIDs    []string
	entityKeyMap map[string]string // canonical entity key -> node ID
	synonymyCnt  int
}

// Engine manages canonical source records and strictly derived HippoRAG 2 indexes.
type Engine struct {
	mu           sync.RWMutex
	cfg          PPRConfig
	canonical    map[string]map[string]SourceRecord   // tenantID -> sourceID -> latest version
	history      map[string]map[string][]SourceRecord // tenantID -> sourceID -> all versions
	derivedBuilt bool
	tenants      map[string]*tenantIndex
}

// NewEngine creates a new HippoRAG 2 associative memory engine.
func NewEngine(cfg PPRConfig) *Engine {
	return &Engine{
		cfg:       cfg,
		canonical: make(map[string]map[string]SourceRecord),
		history:   make(map[string]map[string][]SourceRecord),
		tenants:   make(map[string]*tenantIndex),
	}
}

// Config returns a copy of the engine's default PPRConfig.
func (e *Engine) Config() PPRConfig {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.cfg
}

// UpsertSource stores a canonical versioned SourceRecord.
// Calling UpsertSource marks derived indexes dirty until RebuildIndex is called
// (or use UpsertSourcesAndRebuild for batch ingestion).
func (e *Engine) UpsertSource(rec SourceRecord) error {
	if strings.TrimSpace(rec.TenantID) == "" {
		return ErrEmptyTenant
	}
	if strings.TrimSpace(rec.SourceID) == "" {
		return ErrEmptySourceID
	}
	if rec.Version == 0 {
		return ErrInvalidVersion
	}
	if rec.Reliability <= 0 {
		rec.Reliability = 0.85
	}
	if rec.Reliability > 1.0 {
		rec.Reliability = 1.0
	}
	if rec.Kind == "" {
		rec.Kind = SourceDocument
	}

	cp := rec.Clone()

	e.mu.Lock()
	defer e.mu.Unlock()

	if _, ok := e.canonical[cp.TenantID]; !ok {
		e.canonical[cp.TenantID] = make(map[string]SourceRecord)
		e.history[cp.TenantID] = make(map[string][]SourceRecord)
	}
	e.history[cp.TenantID][cp.SourceID] = append(e.history[cp.TenantID][cp.SourceID], cp)
	if existing, exists := e.canonical[cp.TenantID][cp.SourceID]; !exists || cp.Version >= existing.Version {
		e.canonical[cp.TenantID][cp.SourceID] = cp
		e.derivedBuilt = false
		e.tenants = make(map[string]*tenantIndex)
	}
	return nil
}

// UpsertSourcesAndRebuild ingests a batch of canonical records and builds all derived indexes.
func (e *Engine) UpsertSourcesAndRebuild(records []SourceRecord) error {
	for _, r := range records {
		if err := e.UpsertSource(r); err != nil {
			return err
		}
	}
	e.RebuildIndex()
	return nil
}

// GetSource returns the latest canonical SourceRecord if authorized under ctx.
func (e *Engine) GetSource(ctx AccessContext, sourceID string) (SourceRecord, bool) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if ctx.TenantID == "" {
		return SourceRecord{}, false
	}
	tMap, ok := e.canonical[ctx.TenantID]
	if !ok {
		return SourceRecord{}, false
	}
	rec, ok := tMap[sourceID]
	if !ok || !ctx.IsAuthorized(rec.TenantID, rec.ACL) {
		return SourceRecord{}, false
	}
	return rec.Clone(), true
}

// SourceCount returns the total number of canonical source records across all tenants.
func (e *Engine) SourceCount() int {
	e.mu.RLock()
	defer e.mu.RUnlock()
	n := 0
	for _, m := range e.canonical {
		n += len(m)
	}
	return n
}

// SourceVersionCount returns the total number of versioned canonical records stored.
func (e *Engine) SourceVersionCount(tenantID, sourceID string) int {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return len(e.history[tenantID][sourceID])
}

// DropDerivedIndexes discards all lexical, dense vector, entity-passage graph,
// and recognition-memory fact indexes while preserving 100% of canonical source records.
func (e *Engine) DropDerivedIndexes() {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.tenants = make(map[string]*tenantIndex)
	e.derivedBuilt = false
}

// HasDerivedIndexes reports whether derived indexes are currently materialized.
func (e *Engine) HasDerivedIndexes() bool {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.derivedBuilt
}

func sourceNodeKind(kind SourceKind) NodeKind {
	switch kind {
	case SourceGitCommit:
		return NodeCommit
	case SourceAgentEvent:
		return NodeEvent
	case SourceDecision:
		return NodeDecision
	default:
		return NodePassage
	}
}

func makeSourceNodeID(tenantID string, kind SourceKind, sourceID string) string {
	return tenantID + "::" + string(sourceNodeKind(kind)) + "::" + sourceID
}

func makeEntityNodeID(tenantID string, entityKey string) string {
	return tenantID + "::entity::" + entityKey
}

// RebuildIndex deterministically rebuilds all derived indexes (BM25 lexical,
// dense vector, entity-passage graph, synonymy edges, structural edges, and
// recognition-memory facts) strictly from canonical source records.
func (e *Engine) RebuildIndex() {
	e.mu.Lock()
	defer e.mu.Unlock()

	newTenants := make(map[string]*tenantIndex, len(e.canonical))

	tenantIDs := make([]string, 0, len(e.canonical))
	for tid := range e.canonical {
		tenantIDs = append(tenantIDs, tid)
	}
	sort.Strings(tenantIDs)

	for _, tid := range tenantIDs {
		recMap := e.canonical[tid]
		ti := buildTenantIndex(tid, recMap, e.cfg)
		newTenants[tid] = ti
	}

	e.tenants = newTenants
	e.derivedBuilt = true
}

func buildTenantIndex(tenantID string, recMap map[string]SourceRecord, cfg PPRConfig) *tenantIndex {
	ti := &tenantIndex{
		tenantID:     tenantID,
		docs:         make(map[string]*indexedDoc, len(recMap)),
		postings:     make(map[string][]string),
		nodes:        make(map[string]*GraphNode),
		nodeVecs:     make(map[string]DenseVec),
		adj:          make(map[string][]GraphEdge),
		entityKeyMap: make(map[string]string),
	}

	docIDs := make([]string, 0, len(recMap))
	for sid := range recMap {
		docIDs = append(docIDs, sid)
	}
	sort.Strings(docIDs)
	ti.docIDs = docIDs

	totalLen := 0
	entitySupport := make(map[string]map[string]struct{}) // entityNodeID -> set of SourceIDs

	ensureEntity := func(rawLabel string, sourceID string, reliability float64) (string, string) {
		key := NormalizeEntity(rawLabel)
		if key == "" {
			return "", ""
		}
		nid := makeEntityNodeID(tenantID, key)
		if _, ok := ti.nodes[nid]; !ok {
			ti.nodes[nid] = &GraphNode{
				ID:          nid,
				Kind:        NodeEntity,
				TenantID:    tenantID,
				Label:       key,
				Reliability: reliability,
			}
			ti.nodeVecs[nid] = EmbedText(strings.ReplaceAll(key, "_", " "))
			ti.entityKeyMap[key] = nid
			entitySupport[nid] = make(map[string]struct{})
		} else if reliability > ti.nodes[nid].Reliability {
			ti.nodes[nid].Reliability = reliability
		}
		entitySupport[nid][sourceID] = struct{}{}
		return nid, key
	}

	addEdge := func(edge GraphEdge) {
		ti.adj[edge.From] = append(ti.adj[edge.From], edge)
	}

	// Pass 1: Index documents (BM25 + DenseVec), create source nodes, facts, entity nodes, and edges.
	for _, sid := range docIDs {
		rec := recMap[sid]
		fullText := strings.TrimSpace(rec.Title + " " + rec.Text)
		toks := Tokenize(fullText)
		tf := make(map[string]int, len(toks))
		for _, t := range toks {
			tf[t]++
		}
		for term := range tf {
			ti.postings[term] = append(ti.postings[term], sid)
		}
		docLen := len(toks)
		if docLen == 0 {
			docLen = 1
		}
		totalLen += docLen

		srcNodeID := makeSourceNodeID(tenantID, rec.Kind, rec.SourceID)
		vec := EmbedText(fullText)
		ti.docs[sid] = &indexedDoc{
			Record: rec,
			Tokens: toks,
			TF:     tf,
			DocLen: docLen,
			Vec:    vec,
			NodeID: srcNodeID,
		}
		ti.nodes[srcNodeID] = &GraphNode{
			ID:                srcNodeID,
			Kind:              sourceNodeKind(rec.Kind),
			TenantID:          tenantID,
			Label:             rec.Title,
			SourceID:          rec.SourceID,
			Version:           rec.Version,
			Reliability:       rec.Reliability,
			SupportingSources: []string{rec.SourceID},
		}
		ti.nodeVecs[srcNodeID] = vec

		// Gather triples: explicit structured triples plus auto-extracted triples when empty.
		triples := rec.Triples
		autoExtracted := false
		if len(triples) == 0 {
			triples = ExtractAutoTriples(rec)
			autoExtracted = true
		}

		mentionedEntities := make(map[string]float64) // entityNodeID -> best relevance
		if autoExtracted {
			if titleKey := NormalizeEntity(rec.Title); titleKey != "" {
				if nid, _ := ensureEntity(titleKey, rec.SourceID, rec.Reliability); nid != "" {
					mentionedEntities[nid] = 0.95
				}
			}
		}

		for idx, tr := range triples {
			rel := tr.Relevance
			if rel <= 0 {
				rel = 0.85
			}
			if rel > 1.0 {
				rel = 1.0
			}
			reliab := tr.Reliability
			if reliab <= 0 {
				reliab = rec.Reliability
			}
			if reliab > 1.0 {
				reliab = 1.0
			}

			subID, subKey := ensureEntity(tr.Subject, rec.SourceID, reliab)
			objID, objKey := ensureEntity(tr.Object, rec.SourceID, reliab)
			if subID == "" || objID == "" || subID == objID {
				continue
			}

			pred := NormalizeEntity(tr.Predicate)
			if pred == "" {
				pred = "related_to"
			}

			// Directed and reverse entity-entity fact edges so multi-hop PPR can traverse
			// both forward (failure -> kernel -> allocator -> PR) and backward.
			addEdge(GraphEdge{
				From:        subID,
				To:          objID,
				Kind:        EdgeFactLink,
				Predicate:   pred,
				TenantID:    tenantID,
				SourceID:    rec.SourceID,
				ACL:         append([]string(nil), rec.ACL...),
				Relevance:   rel,
				Reliability: reliab,
			})
			addEdge(GraphEdge{
				From:        objID,
				To:          subID,
				Kind:        EdgeFactLink,
				Predicate:   "inv_" + pred,
				TenantID:    tenantID,
				SourceID:    rec.SourceID,
				ACL:         append([]string(nil), rec.ACL...),
				Relevance:   rel * 0.90,
				Reliability: reliab,
			})

			if prev := mentionedEntities[subID]; rel > prev {
				mentionedEntities[subID] = rel
			}
			if prev := mentionedEntities[objID]; rel > prev {
				mentionedEntities[objID] = rel
			}

			factText := strings.ReplaceAll(subKey, "_", " ") + " " +
				strings.ReplaceAll(pred, "_", " ") + " " +
				strings.ReplaceAll(objKey, "_", " ")
			factRec := FactRecord{
				ID:          fmt.Sprintf("%s::fact::%s::%d", tenantID, rec.SourceID, idx),
				TenantID:    tenantID,
				SourceID:    rec.SourceID,
				Version:     rec.Version,
				ACL:         append([]string(nil), rec.ACL...),
				Subject:     subKey,
				SubjectID:   subID,
				Predicate:   pred,
				Object:      objKey,
				ObjectID:    objID,
				Relevance:   rel,
				Reliability: reliab,
				Text:        factText,
			}
			ti.facts = append(ti.facts, factRec)
			ti.factVecs = append(ti.factVecs, EmbedText(factText))
			ti.factTokens = append(ti.factTokens, Tokenize(factText))
		}

		// Connect mentioned entities <-> source node (bipartite entity-passage graph).
		entIDs := make([]string, 0, len(mentionedEntities))
		for eid := range mentionedEntities {
			entIDs = append(entIDs, eid)
		}
		sort.Strings(entIDs)
		for _, eid := range entIDs {
			rel := mentionedEntities[eid]
			addEdge(GraphEdge{
				From:        eid,
				To:          srcNodeID,
				Kind:        EdgePassageMention,
				Predicate:   "evidenced_in",
				TenantID:    tenantID,
				SourceID:    rec.SourceID,
				ACL:         append([]string(nil), rec.ACL...),
				Relevance:   rel,
				Reliability: rec.Reliability,
			})
			addEdge(GraphEdge{
				From:        srcNodeID,
				To:          eid,
				Kind:        EdgePassageMention,
				Predicate:   "mentions",
				TenantID:    tenantID,
				SourceID:    rec.SourceID,
				ACL:         append([]string(nil), rec.ACL...),
				Relevance:   rel * 0.92,
				Reliability: rec.Reliability,
			})
		}
	}

	if len(docIDs) > 0 {
		ti.avgDocLen = float64(totalLen) / float64(len(docIDs))
	}

	// Pass 2: Structural cross-source metadata links strictly within the same tenant.
	for _, sid := range docIDs {
		rec := recMap[sid]
		fromID := ti.docs[sid].NodeID
		linksRaw := rec.Metadata["links_to"]
		if linksRaw == "" {
			continue
		}
		for _, part := range strings.Split(linksRaw, ",") {
			targetSID := strings.TrimSpace(part)
			if targetSID == "" || targetSID == sid {
				continue
			}
			targetDoc, ok := ti.docs[targetSID]
			if !ok {
				continue
			}
			// Combine reliability conservatively across linked records.
			reliab := rec.Reliability
			if targetDoc.Record.Reliability < reliab {
				reliab = targetDoc.Record.Reliability
			}
			addEdge(GraphEdge{
				From:        fromID,
				To:          targetDoc.NodeID,
				Kind:        EdgeStructural,
				Predicate:   "links_to",
				TenantID:    tenantID,
				SourceID:    rec.SourceID,
				ACL:         append([]string(nil), rec.ACL...),
				Relevance:   0.92,
				Reliability: reliab,
			})
			addEdge(GraphEdge{
				From:        targetDoc.NodeID,
				To:          fromID,
				Kind:        EdgeStructural,
				Predicate:   "linked_from",
				TenantID:    tenantID,
				SourceID:    targetDoc.Record.SourceID,
				ACL:         append([]string(nil), targetDoc.Record.ACL...),
				Relevance:   0.85,
				Reliability: reliab,
			})
		}
	}

	// Finalize entity supporting sources in deterministic order.
	entityIDs := make([]string, 0, len(entitySupport))
	for eid, sup := range entitySupport {
		entityIDs = append(entityIDs, eid)
		sids := make([]string, 0, len(sup))
		for s := range sup {
			sids = append(sids, s)
		}
		sort.Strings(sids)
		ti.nodes[eid].SupportingSources = sids
	}
	sort.Strings(entityIDs)
	ti.entityIDs = entityIDs

	// Pass 3: HippoRAG 2 dense synonymy edges strictly within the same tenant.
	// Only add synonymy edges between distinct entities whose cosine similarity >= SynonymyThreshold.
	if cfg.SynonymyThreshold > 0 && cfg.SynonymyTopK > 0 && len(entityIDs) > 1 {
		type synCand struct {
			target string
			sim    float64
		}
		for i, eidA := range entityIDs {
			vecA := ti.nodeVecs[eidA]
			var cands []synCand
			for j, eidB := range entityIDs {
				if i == j {
					continue
				}
				sim := CosineSim(vecA, ti.nodeVecs[eidB])
				if sim >= cfg.SynonymyThreshold {
					cands = append(cands, synCand{target: eidB, sim: sim})
				}
			}
			sort.Slice(cands, func(a, b int) bool {
				if cands[a].sim != cands[b].sim {
					return cands[a].sim > cands[b].sim
				}
				return cands[a].target < cands[b].target
			})
			if len(cands) > cfg.SynonymyTopK {
				cands = cands[:cfg.SynonymyTopK]
			}
			for _, c := range cands {
				reliab := ti.nodes[eidA].Reliability
				if ti.nodes[c.target].Reliability < reliab {
					reliab = ti.nodes[c.target].Reliability
				}
				addEdge(GraphEdge{
					From:        eidA,
					To:          c.target,
					Kind:        EdgeSynonymy,
					Predicate:   "synonymous_with",
					TenantID:    tenantID,
					Relevance:   c.sim * 0.80,
					Reliability: reliab,
				})
				ti.synonymyCnt++
			}
		}
	}

	// Deterministic sorting of postings and adjacency lists for bit-exact rebuilds.
	for term := range ti.postings {
		sort.Strings(ti.postings[term])
	}
	for nid := range ti.adj {
		edges := ti.adj[nid]
		sort.Slice(edges, func(i, j int) bool {
			if edges[i].To != edges[j].To {
				return edges[i].To < edges[j].To
			}
			if edges[i].Kind != edges[j].Kind {
				return edges[i].Kind < edges[j].Kind
			}
			if edges[i].Predicate != edges[j].Predicate {
				return edges[i].Predicate < edges[j].Predicate
			}
			return edges[i].SourceID < edges[j].SourceID
		})
		ti.adj[nid] = edges
	}

	return ti
}

// Snapshot computes a deterministic digest of the current derived indexes.
func (e *Engine) Snapshot() IndexSnapshot {
	e.mu.RLock()
	defer e.mu.RUnlock()

	snap := IndexSnapshot{}
	for _, m := range e.canonical {
		snap.CanonicalRecords += len(m)
	}
	if !e.derivedBuilt {
		return snap
	}

	h := sha256.New()
	tids := make([]string, 0, len(e.tenants))
	for tid := range e.tenants {
		tids = append(tids, tid)
	}
	sort.Strings(tids)

	for _, tid := range tids {
		ti := e.tenants[tid]
		snap.GraphNodes += len(ti.nodes)
		snap.FactRecords += len(ti.facts)
		snap.SynonymyEdges += ti.synonymyCnt
		snap.VocabTerms += len(ti.postings)

		fmt.Fprintf(h, "TENANT:%s\n", tid)

		nids := make([]string, 0, len(ti.nodes))
		for nid := range ti.nodes {
			nids = append(nids, nid)
		}
		sort.Strings(nids)
		for _, nid := range nids {
			n := ti.nodes[nid]
			fmt.Fprintf(h, "N:%s|%s|%s|%d|%.6f|%s\n",
				n.ID, n.Kind, n.SourceID, n.Version, n.Reliability, strings.Join(n.SupportingSources, ","))
			for _, ed := range ti.adj[nid] {
				snap.GraphEdges++
				fmt.Fprintf(h, "E:%s->%s|%s|%s|%s|%.6f|%.6f\n",
					ed.From, ed.To, ed.Kind, ed.Predicate, ed.SourceID, ed.Relevance, ed.Reliability)
			}
		}
		for _, f := range ti.facts {
			fmt.Fprintf(h, "F:%s|%s|%s|%s|%.6f|%.6f\n",
				f.ID, f.Subject, f.Predicate, f.Object, f.Relevance, f.Reliability)
		}
	}
	snap.DigestSHA256 = hex.EncodeToString(h.Sum(nil))
	return snap
}
