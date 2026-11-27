/*
 * SPDX-License-Identifier: Apache-2.0
 */

package hipporag

import (
	"container/heap"
	"math"
	"sort"
	"strings"
)

// SearchHippoRAG2 executes the full HippoRAG 2 bounded Personalized PageRank
// retrieval pipeline using the engine's configured PPRConfig.
func (e *Engine) SearchHippoRAG2(req QueryRequest) (QueryResult, error) {
	return e.SearchHippoRAG2WithConfig(req, e.Config())
}

// SearchHippoRAG2WithConfig executes bounded PPR with an explicit PPRConfig
// (useful for ablations and negative boundary regimes such as hub collapse).
func (e *Engine) SearchHippoRAG2WithConfig(req QueryRequest, cfg PPRConfig) (QueryResult, error) {
	if strings.TrimSpace(req.Access.TenantID) == "" {
		return QueryResult{}, ErrEmptyTenant
	}
	if req.TopK > 0 {
		cfg.TopK = req.TopK
	}
	if cfg.TopK <= 0 {
		cfg.TopK = 5
	}
	if cfg.Damping <= 0 || cfg.Damping >= 1 {
		cfg.Damping = 0.75
	}
	if cfg.MaxIterations <= 0 {
		cfg.MaxIterations = 25
	}
	if cfg.Epsilon <= 0 {
		cfg.Epsilon = 1e-6
	}
	if cfg.MaxHops <= 0 {
		cfg.MaxHops = 4
	}
	if cfg.MaxExpandedNodes <= 0 {
		cfg.MaxExpandedNodes = 256
	}

	e.mu.RLock()
	defer e.mu.RUnlock()

	if !e.derivedBuilt {
		return QueryResult{}, ErrIndexesNotBuilt
	}

	res := QueryResult{
		Method: "hipporag2_bounded_ppr",
		Query:  req.Query,
	}

	// Pre-return tenant partitioning: only ever touch the caller's tenant index.
	ti, ok := e.tenants[req.Access.TenantID]
	if !ok {
		return res, nil
	}

	// Strict pre-return ACL authorization map before any seed, fact, or graph step.
	authDocs := authorizedDocMap(ti, req.Access)
	if len(authDocs) == 0 {
		return res, nil
	}

	// Compute authorized degree of every authorized entity node for hub damping.
	authEntityDeg := make(map[string]int, len(ti.entityIDs))
	authNodes := make(map[string]*GraphNode, len(ti.nodes))
	for nid, node := range ti.nodes {
		if isNodeAuthorized(node, authDocs) {
			authNodes[nid] = node
		}
	}
	for nid, node := range authNodes {
		if node.Kind != NodeEntity {
			continue
		}
		deg := 0
		for _, ed := range ti.adj[nid] {
			if isEdgeAuthorized(ed, req.Access, authDocs, authNodes) {
				deg++
			}
		}
		authEntityDeg[nid] = deg
	}

	qToks := Tokenize(req.Query)
	qTokSet := make(map[string]struct{}, len(qToks))
	for _, t := range qToks {
		qTokSet[t] = struct{}{}
	}
	qVec := EmbedText(req.Query)
	qNorm := NormalizeEntity(req.Query)
	qCandEntities := ExtractCandidateEntities(req.Query)

	// Step 1: Identify direct query entity matches strictly within authorized entity nodes.
	seedMass := make(map[string]float64)
	entitySeedSet := make(map[string]struct{})
	directMatchedEntities := make(map[string]float64)

	for _, eid := range ti.entityIDs {
		node, ok := authNodes[eid]
		if !ok {
			continue
		}
		label := node.Label
		matchScore := matchEntityToQuery(label, qNorm, qTokSet, qCandEntities)
		if matchScore <= 0 {
			continue
		}
		w := matchScore
		if cfg.EnableHubDamping {
			deg := float64(authEntityDeg[eid])
			w *= entitySpecificity(deg, cfg.HubDegreePivot, cfg.HubPenaltyAlpha)
		}
		if cfg.EnableReliabilityWeighting {
			w *= math.Pow(node.Reliability, cfg.ReliabilityExponent)
		}
		if w > 0 {
			directMatchedEntities[eid] = w
		}
	}

	// Step 2: HippoRAG 2 Fact Retrieval + Recognition-Memory Triple Filtering.
	type scoredFact struct {
		fact  FactRecord
		score float64
	}
	var candFacts []scoredFact
	for i, f := range ti.facts {
		// Pre-return ACL check on fact's canonical source record.
		if authDocs[f.SourceID] == nil || !req.Access.IsAuthorized(f.TenantID, f.ACL) {
			continue
		}
		if authNodes[f.SubjectID] == nil || authNodes[f.ObjectID] == nil {
			continue
		}
		res.Audit.CandidateFactsEvaluated++

		cos := CosineSim(qVec, ti.factVecs[i])
		if cos < 0 {
			cos = 0
		}
		lex := lexicalOverlapTokens(qTokSet, ti.factTokens[i])
		entBonus := 0.0
		if directMatchedEntities[f.SubjectID] > 0 {
			entBonus += 0.45
		}
		if directMatchedEntities[f.ObjectID] > 0 {
			entBonus += 0.45
		}
		rawAlign := 0.45*cos + 0.45*lex + entBonus
		if rawAlign <= 0.04 {
			continue
		}

		score := rawAlign
		if cfg.EnableReliabilityWeighting {
			score *= math.Pow(f.Reliability, cfg.ReliabilityExponent)
		}
		if cfg.EnableHubDamping {
			degS := float64(authEntityDeg[f.SubjectID])
			degO := float64(authEntityDeg[f.ObjectID])
			// Penalize triples anchored on generic high-degree hubs.
			spec := math.Sqrt(entitySpecificity(degS, cfg.HubDegreePivot, cfg.HubPenaltyAlpha) *
				entitySpecificity(degO, cfg.HubDegreePivot, cfg.HubPenaltyAlpha))
			score *= spec
		}
		score *= f.Relevance

		candFacts = append(candFacts, scoredFact{fact: f, score: score})
	}

	sort.Slice(candFacts, func(i, j int) bool {
		if candFacts[i].score != candFacts[j].score {
			return candFacts[i].score > candFacts[j].score
		}
		return candFacts[i].fact.ID < candFacts[j].fact.ID
	})

	maxFacts := cfg.MaxSeedFacts
	if maxFacts <= 0 {
		maxFacts = 8
	}
	if !cfg.EnableRecognitionMemory {
		maxFacts *= 3
	}

	for _, sf := range candFacts {
		if cfg.EnableRecognitionMemory {
			if sf.score < cfg.RecognitionThreshold || len(res.FilteredFacts) >= maxFacts {
				res.Audit.FactsRejectedByRecognition++
				continue
			}
		} else if len(res.FilteredFacts) >= maxFacts {
			break
		}
		res.Audit.FactsAcceptedByRecognition++
		res.FilteredFacts = append(res.FilteredFacts, sf.fact.Subject+" --["+sf.fact.Predicate+"]--> "+sf.fact.Object)

		// Seed subject and object phrase nodes from recognition-accepted triples.
		seedMass[sf.fact.SubjectID] += sf.score
		seedMass[sf.fact.ObjectID] += sf.score * 0.90
		entitySeedSet[sf.fact.SubjectID] = struct{}{}
		entitySeedSet[sf.fact.ObjectID] = struct{}{}
	}

	// Add direct matched query entity seeds.
	for eid, w := range directMatchedEntities {
		seedMass[eid] += w * 1.25
		entitySeedSet[eid] = struct{}{}
	}

	seedEntities := make([]string, 0, len(entitySeedSet))
	for eid := range entitySeedSet {
		seedEntities = append(seedEntities, authNodes[eid].Label)
	}
	sort.Strings(seedEntities)
	res.SeedEntities = seedEntities
	res.Audit.SeedEntityCount = len(entitySeedSet)

	// Step 3: HippoRAG 2 Dual Passage Seeding (dense + lexical passage seeds scaled by PassageSeedWeight).
	type docCand struct {
		doc   *indexedDoc
		score float64
	}
	passageCands := make([]docCand, 0, len(authDocs))
	bm25Scores := computeBM25Authorized(ti, authDocs, qToks)
	maxBM25 := 0.0
	for _, s := range bm25Scores {
		if s > maxBM25 {
			maxBM25 = s
		}
	}
	for _, sid := range ti.docIDs {
		doc := authDocs[sid]
		if doc == nil {
			continue
		}
		cos := CosineSim(qVec, doc.Vec)
		if cos < 0 {
			cos = 0
		}
		bNorm := 0.0
		if maxBM25 > 0 {
			bNorm = bm25Scores[sid] / maxBM25
		}
		comb := 0.55*cos + 0.45*bNorm
		if cfg.EnableReliabilityWeighting {
			comb *= math.Pow(doc.Record.Reliability, cfg.ReliabilityExponent)
		}
		if comb > 0 {
			passageCands = append(passageCands, docCand{doc: doc, score: comb})
		}
	}
	sort.Slice(passageCands, func(i, j int) bool {
		if passageCands[i].score != passageCands[j].score {
			return passageCands[i].score > passageCands[j].score
		}
		return passageCands[i].doc.Record.SourceID < passageCands[j].doc.Record.SourceID
	})

	passageSeedLimit := cfg.TopK
	if passageSeedLimit < 3 {
		passageSeedLimit = 3
	}
	if len(passageCands) < passageSeedLimit {
		passageSeedLimit = len(passageCands)
	}
	passageWeight := cfg.PassageSeedWeight
	if len(entitySeedSet) == 0 {
		// If no entity matched at all, rely on passage seeds.
		passageWeight = 1.0
	}
	for i := 0; i < passageSeedLimit; i++ {
		pc := passageCands[i]
		w := pc.score * passageWeight
		if w > 0 {
			seedMass[pc.doc.NodeID] += w
			res.Audit.SeedPassageCount++
		}
	}

	if len(seedMass) == 0 {
		return res, nil
	}

	// Normalize personalization vector p.
	personalization := make(map[string]float64, len(seedMass))
	var sumP float64
	for _, w := range seedMass {
		if w > 0 {
			sumP += w
		}
	}
	for nid, w := range seedMass {
		if w > 0 {
			personalization[nid] = w / sumP
		}
	}

	// Step 4: Bounded Ego-Subgraph Expansion (MaxHops, MaxExpandedNodes) strictly over authorized nodes/edges.
	subNodes, hopDist := expandBoundedSubgraph(ti, personalization, authDocs, authNodes, authEntityDeg, req.Access, cfg)
	res.Audit.ExpandedNodes = len(subNodes)

	// Build normalized outgoing transitions strictly within subNodes.
	type weightedEdge struct {
		edge GraphEdge
		prob float64
		rawW float64
	}
	transitions := make(map[string][]weightedEdge, len(subNodes))
	edgeCount := 0
	for nid := range subNodes {
		rawEdges := ti.adj[nid]
		if len(rawEdges) == 0 {
			continue
		}
		var kept []weightedEdge
		var totalW float64
		for _, ed := range rawEdges {
			if !subNodes[ed.To] {
				continue
			}
			if !isEdgeAuthorized(ed, req.Access, authDocs, authNodes) {
				continue
			}
			w := effectiveEdgeWeight(ed, authNodes[ed.To], authEntityDeg, cfg)
			if w <= 0 {
				continue
			}
			kept = append(kept, weightedEdge{edge: ed, rawW: w})
			totalW += w
		}
		if totalW > 0 {
			for i := range kept {
				kept[i].prob = kept[i].rawW / totalW
			}
			transitions[nid] = kept
			edgeCount += len(kept)
		}
	}
	res.Audit.ExpandedEdges = edgeCount

	// Step 5: Bounded Personalized PageRank Power Iteration.
	nodeOrder := make([]string, 0, len(subNodes))
	for nid := range subNodes {
		nodeOrder = append(nodeOrder, nid)
	}
	sort.Strings(nodeOrder)

	// Filter personalization to subNodes and renormalize.
	pVec := make(map[string]float64, len(subNodes))
	var pSum float64
	for nid, w := range personalization {
		if subNodes[nid] {
			pVec[nid] = w
			pSum += w
		}
	}
	if pSum > 0 {
		for nid := range pVec {
			pVec[nid] /= pSum
		}
	}

	pi := make(map[string]float64, len(subNodes))
	for nid, w := range pVec {
		pi[nid] = w
	}

	alpha := cfg.Damping
	teleport := 1.0 - alpha

	for iter := 1; iter <= cfg.MaxIterations; iter++ {
		next := make(map[string]float64, len(subNodes))
		for nid, w := range pVec {
			next[nid] = teleport * w
		}

		var danglingMass float64
		for _, u := range nodeOrder {
			mass := pi[u]
			if mass == 0 {
				continue
			}
			outs := transitions[u]
			if len(outs) == 0 {
				danglingMass += mass
				continue
			}
			for _, we := range outs {
				next[we.edge.To] += alpha * mass * we.prob
			}
		}

		if danglingMass > 0 {
			for nid, w := range pVec {
				next[nid] += alpha * danglingMass * w
			}
		}

		var l1 float64
		for _, nid := range nodeOrder {
			l1 += math.Abs(next[nid] - pi[nid])
		}
		pi = next
		res.Audit.PPRIterations = iter
		res.Audit.PPRResidualL1 = l1
		if l1 < cfg.Epsilon {
			res.Audit.Converged = true
			break
		}
	}

	// Step 6: Rank authorized source nodes by bounded PPR mass.
	type rankedSource struct {
		nodeID string
		doc    *indexedDoc
		score  float64
	}
	ranked := make([]rankedSource, 0, len(subNodes))
	for _, nid := range nodeOrder {
		node := authNodes[nid]
		if node == nil || node.Kind == NodeEntity || node.SourceID == "" {
			continue
		}
		doc := authDocs[node.SourceID]
		if doc == nil {
			continue
		}
		s := pi[nid]
		if cfg.EnableReliabilityWeighting {
			s *= math.Pow(doc.Record.Reliability, 0.5)
		}
		ranked = append(ranked, rankedSource{
			nodeID: nid,
			doc:    doc,
			score:  s,
		})
	}

	sort.Slice(ranked, func(i, j int) bool {
		if ranked[i].score != ranked[j].score {
			return ranked[i].score > ranked[j].score
		}
		return ranked[i].doc.Record.SourceID < ranked[j].doc.Record.SourceID
	})

	if len(ranked) > cfg.TopK {
		ranked = ranked[:cfg.TopK]
	}

	// Step 7: Deterministic path reconstruction from primary query entity seeds to each retrieved hit.
	// Anchoring on direct query entity matches ensures multi-hop trails trace the full causal chain
	// from the entity asked about in the query (e.g., cuda_graph_memory_alias -> kernel -> allocator -> PR -> test).
	pathSeeds := make(map[string]float64)
	maxDirect := 0.0
	for eid, w := range directMatchedEntities {
		if subNodes[eid] && w > maxDirect {
			maxDirect = w
		}
	}
	if maxDirect > 0 {
		for eid, w := range directMatchedEntities {
			if subNodes[eid] && w >= maxDirect*0.85 {
				pathSeeds[eid] = w
			}
		}
	}
	if len(pathSeeds) == 0 {
		for eid := range entitySeedSet {
			if subNodes[eid] {
				pathSeeds[eid] = pVec[eid]
			}
		}
	}
	if len(pathSeeds) == 0 {
		pathSeeds = pVec
	}

	adjForPath := make(map[string][]pathEdge, len(transitions))
	for u, outs := range transitions {
		pe := make([]pathEdge, len(outs))
		for i, we := range outs {
			pe[i] = pathEdge{
				edge:   we.edge,
				weight: we.prob * (0.15 + pi[we.edge.To]),
			}
		}
		adjForPath[u] = pe
	}

	hits := make([]RetrievedHit, 0, len(ranked))
	allPathsOK := len(ranked) > 0
	for _, rs := range ranked {
		path := reconstructEvidencePath(pathSeeds, pVec, rs.nodeID, authNodes, adjForPath)
		if path.TargetNode == "" {
			allPathsOK = false
		}
		toks := EstimateTokens(rs.doc.Record.Title + " " + rs.doc.Record.Text)
		res.TotalTokens += toks
		hits = append(hits, RetrievedHit{
			SourceID:    rs.doc.Record.SourceID,
			Version:     rs.doc.Record.Version,
			Kind:        rs.doc.Record.Kind,
			TenantID:    rs.doc.Record.TenantID,
			Title:       rs.doc.Record.Title,
			Text:        rs.doc.Record.Text,
			Score:       rs.score,
			Reliability: rs.doc.Record.Reliability,
			HopDistance: hopDist[rs.nodeID],
			Path:        path,
			Tokens:      toks,
		})
	}
	res.Hits = hits
	res.ReconstructedOK = allPathsOK
	return res, nil
}

// SearchBM25 executes the matched Okapi BM25 lexical baseline over the exact
// same canonical corpus and strict pre-return tenant/ACL filter.
func (e *Engine) SearchBM25(req QueryRequest) (QueryResult, error) {
	if strings.TrimSpace(req.Access.TenantID) == "" {
		return QueryResult{}, ErrEmptyTenant
	}
	topK := req.TopK
	if topK <= 0 {
		topK = e.cfg.TopK
	}
	if topK <= 0 {
		topK = 5
	}

	e.mu.RLock()
	defer e.mu.RUnlock()
	if !e.derivedBuilt {
		return QueryResult{}, ErrIndexesNotBuilt
	}

	res := QueryResult{
		Method: "bm25_lexical_baseline",
		Query:  req.Query,
	}
	ti, ok := e.tenants[req.Access.TenantID]
	if !ok {
		return res, nil
	}
	authDocs := authorizedDocMap(ti, req.Access)
	if len(authDocs) == 0 {
		return res, nil
	}

	qToks := Tokenize(req.Query)
	scores := computeBM25Authorized(ti, authDocs, qToks)

	type scored struct {
		doc   *indexedDoc
		score float64
	}
	cands := make([]scored, 0, len(scores))
	for sid, sc := range scores {
		if sc > 0 {
			cands = append(cands, scored{doc: authDocs[sid], score: sc})
		}
	}
	sort.Slice(cands, func(i, j int) bool {
		if cands[i].score != cands[j].score {
			return cands[i].score > cands[j].score
		}
		return cands[i].doc.Record.SourceID < cands[j].doc.Record.SourceID
	})
	if len(cands) > topK {
		cands = cands[:topK]
	}

	res.Audit.ExpandedNodes = len(authDocs)
	for _, c := range cands {
		toks := EstimateTokens(c.doc.Record.Title + " " + c.doc.Record.Text)
		res.TotalTokens += toks
		res.Hits = append(res.Hits, RetrievedHit{
			SourceID:    c.doc.Record.SourceID,
			Version:     c.doc.Record.Version,
			Kind:        c.doc.Record.Kind,
			TenantID:    c.doc.Record.TenantID,
			Title:       c.doc.Record.Title,
			Text:        c.doc.Record.Text,
			Score:       c.score,
			Reliability: c.doc.Record.Reliability,
			Tokens:      toks,
		})
	}
	return res, nil
}

// SearchDenseRAG executes the matched dense vector RAG baseline over the exact
// same embedding index and strict pre-return tenant/ACL filter.
func (e *Engine) SearchDenseRAG(req QueryRequest) (QueryResult, error) {
	if strings.TrimSpace(req.Access.TenantID) == "" {
		return QueryResult{}, ErrEmptyTenant
	}
	topK := req.TopK
	if topK <= 0 {
		topK = e.cfg.TopK
	}
	if topK <= 0 {
		topK = 5
	}

	e.mu.RLock()
	defer e.mu.RUnlock()
	if !e.derivedBuilt {
		return QueryResult{}, ErrIndexesNotBuilt
	}

	res := QueryResult{
		Method: "dense_vector_rag_baseline",
		Query:  req.Query,
	}
	ti, ok := e.tenants[req.Access.TenantID]
	if !ok {
		return res, nil
	}
	authDocs := authorizedDocMap(ti, req.Access)
	if len(authDocs) == 0 {
		return res, nil
	}

	qVec := EmbedText(req.Query)
	type scored struct {
		doc   *indexedDoc
		score float64
	}
	cands := make([]scored, 0, len(authDocs))
	for _, sid := range ti.docIDs {
		doc := authDocs[sid]
		if doc == nil {
			continue
		}
		sim := CosineSim(qVec, doc.Vec)
		if sim > 0 {
			cands = append(cands, scored{doc: doc, score: sim})
		}
	}
	sort.Slice(cands, func(i, j int) bool {
		if cands[i].score != cands[j].score {
			return cands[i].score > cands[j].score
		}
		return cands[i].doc.Record.SourceID < cands[j].doc.Record.SourceID
	})
	if len(cands) > topK {
		cands = cands[:topK]
	}

	res.Audit.ExpandedNodes = len(authDocs)
	for _, c := range cands {
		toks := EstimateTokens(c.doc.Record.Title + " " + c.doc.Record.Text)
		res.TotalTokens += toks
		res.Hits = append(res.Hits, RetrievedHit{
			SourceID:    c.doc.Record.SourceID,
			Version:     c.doc.Record.Version,
			Kind:        c.doc.Record.Kind,
			TenantID:    c.doc.Record.TenantID,
			Title:       c.doc.Record.Title,
			Text:        c.doc.Record.Text,
			Score:       c.score,
			Reliability: c.doc.Record.Reliability,
			Tokens:      toks,
		})
	}
	return res, nil
}

// SearchUnfiltered1Hop executes the matched 1-hop entity-graph baseline:
// entity-seeded 1-hop expansion to directly attached source passages without
// bounded multi-hop PPR, reliability weighting, hub damping, or recognition memory.
// Pre-return tenant and ACL filtering is still enforced.
func (e *Engine) SearchUnfiltered1Hop(req QueryRequest) (QueryResult, error) {
	if strings.TrimSpace(req.Access.TenantID) == "" {
		return QueryResult{}, ErrEmptyTenant
	}
	topK := req.TopK
	if topK <= 0 {
		topK = e.cfg.TopK
	}
	if topK <= 0 {
		topK = 5
	}

	e.mu.RLock()
	defer e.mu.RUnlock()
	if !e.derivedBuilt {
		return QueryResult{}, ErrIndexesNotBuilt
	}

	res := QueryResult{
		Method: "unfiltered_1hop_graph_baseline",
		Query:  req.Query,
	}
	ti, ok := e.tenants[req.Access.TenantID]
	if !ok {
		return res, nil
	}
	authDocs := authorizedDocMap(ti, req.Access)
	if len(authDocs) == 0 {
		return res, nil
	}
	authNodes := make(map[string]*GraphNode, len(ti.nodes))
	for nid, n := range ti.nodes {
		if isNodeAuthorized(n, authDocs) {
			authNodes[nid] = n
		}
	}

	qToks := Tokenize(req.Query)
	qTokSet := make(map[string]struct{}, len(qToks))
	for _, t := range qToks {
		qTokSet[t] = struct{}{}
	}
	qNorm := NormalizeEntity(req.Query)
	qCands := ExtractCandidateEntities(req.Query)

	docScores := make(map[string]float64)
	seedForDoc := make(map[string]string)
	visited := make(map[string]bool)

	for _, eid := range ti.entityIDs {
		node, ok := authNodes[eid]
		if !ok {
			continue
		}
		m := matchEntityToQuery(node.Label, qNorm, qTokSet, qCands)
		if m <= 0 {
			continue
		}
		res.SeedEntities = append(res.SeedEntities, node.Label)
		visited[eid] = true
		for _, ed := range ti.adj[eid] {
			if !isEdgeAuthorized(ed, req.Access, authDocs, authNodes) {
				continue
			}
			visited[ed.To] = true
			target := authNodes[ed.To]
			if target != nil && target.Kind != NodeEntity && target.SourceID != "" {
				// Unfiltered 1-hop sums raw relevance without reliability/hub damping.
				docScores[target.SourceID] += m * ed.Relevance
				if seedForDoc[target.SourceID] == "" {
					seedForDoc[target.SourceID] = eid
				}
			}
		}
	}
	sort.Strings(res.SeedEntities)
	res.Audit.SeedEntityCount = len(res.SeedEntities)
	res.Audit.ExpandedNodes = len(visited)

	type scored struct {
		doc   *indexedDoc
		score float64
	}
	cands := make([]scored, 0, len(docScores))
	for sid, sc := range docScores {
		if doc := authDocs[sid]; doc != nil {
			cands = append(cands, scored{doc: doc, score: sc})
		}
	}
	sort.Slice(cands, func(i, j int) bool {
		if cands[i].score != cands[j].score {
			return cands[i].score > cands[j].score
		}
		return cands[i].doc.Record.SourceID < cands[j].doc.Record.SourceID
	})
	if len(cands) > topK {
		cands = cands[:topK]
	}

	for _, c := range cands {
		toks := EstimateTokens(c.doc.Record.Title + " " + c.doc.Record.Text)
		res.TotalTokens += toks
		seedID := seedForDoc[c.doc.Record.SourceID]
		seedLabel := ""
		if n := authNodes[seedID]; n != nil {
			seedLabel = n.Label
		}
		res.Hits = append(res.Hits, RetrievedHit{
			SourceID:    c.doc.Record.SourceID,
			Version:     c.doc.Record.Version,
			Kind:        c.doc.Record.Kind,
			TenantID:    c.doc.Record.TenantID,
			Title:       c.doc.Record.Title,
			Text:        c.doc.Record.Text,
			Score:       c.score,
			Reliability: c.doc.Record.Reliability,
			HopDistance: 1,
			Path: EvidencePath{
				SeedNode:   seedID,
				TargetNode: c.doc.NodeID,
				Hops:       1,
				Trail:      seedLabel + " --[evidenced_in]--> " + c.doc.Record.Title,
			},
			Tokens: toks,
		})
	}
	res.ReconstructedOK = len(res.Hits) > 0
	return res, nil
}

// --- Internal helpers ---

func authorizedDocMap(ti *tenantIndex, ctx AccessContext) map[string]*indexedDoc {
	out := make(map[string]*indexedDoc, len(ti.docs))
	for sid, doc := range ti.docs {
		if ctx.IsAuthorized(doc.Record.TenantID, doc.Record.ACL) {
			out[sid] = doc
		}
	}
	return out
}

func isNodeAuthorized(node *GraphNode, authDocs map[string]*indexedDoc) bool {
	if node == nil {
		return false
	}
	if node.Kind != NodeEntity {
		return authDocs[node.SourceID] != nil
	}
	for _, sid := range node.SupportingSources {
		if authDocs[sid] != nil {
			return true
		}
	}
	return false
}

func isEdgeAuthorized(ed GraphEdge, ctx AccessContext, authDocs map[string]*indexedDoc, authNodes map[string]*GraphNode) bool {
	if ed.TenantID != ctx.TenantID {
		return false
	}
	if ed.SourceID != "" && authDocs[ed.SourceID] == nil {
		return false
	}
	if len(ed.ACL) > 0 && !ctx.IsAuthorized(ed.TenantID, ed.ACL) {
		return false
	}
	if authNodes[ed.To] == nil {
		return false
	}
	return true
}

func entitySpecificity(deg float64, pivot float64, alpha float64) float64 {
	if deg <= pivot {
		return 1.0 / math.Log2(2.0+0.25*deg)
	}
	return 1.0 / (math.Log2(2.0+0.25*pivot) * (1.0 + alpha*(deg-pivot)))
}

func effectiveEdgeWeight(ed GraphEdge, toNode *GraphNode, authEntityDeg map[string]int, cfg PPRConfig) float64 {
	w := ed.Relevance
	if w <= 0 {
		w = 0.5
	}
	if cfg.EnableReliabilityWeighting {
		w *= math.Pow(ed.Reliability, cfg.ReliabilityExponent)
	}
	if cfg.EnableHubDamping && toNode != nil && toNode.Kind == NodeEntity {
		deg := float64(authEntityDeg[ed.To])
		if deg > cfg.HubDegreePivot {
			w /= (1.0 + cfg.HubPenaltyAlpha*(deg-cfg.HubDegreePivot))
		}
	}
	return w
}

func matchEntityToQuery(entityLabel string, qNorm string, qTokSet map[string]struct{}, qCands []string) float64 {
	if entityLabel == "" {
		return 0
	}
	// Exact normalized phrase substring match (e.g., "cuda_graph_memory_alias" in query).
	if len(entityLabel) >= 4 && strings.Contains(qNorm, entityLabel) {
		return 1.0
	}
	for _, c := range qCands {
		if c == entityLabel {
			return 1.0
		}
	}
	// Multi-token partial overlap for compound entities.
	parts := strings.Split(entityLabel, "_")
	var contentParts int
	var matched int
	for _, p := range parts {
		if len(p) <= 1 {
			continue
		}
		if _, stop := stopWords[p]; stop {
			continue
		}
		contentParts++
		if _, ok := qTokSet[p]; ok {
			matched++
		}
	}
	if contentParts == 0 || matched == 0 {
		return 0
	}
	ratio := float64(matched) / float64(contentParts)
	if contentParts == 1 {
		if matched == 1 && len(entityLabel) >= 4 {
			return 0.65
		}
		return 0
	}
	if ratio >= 0.66 {
		return 0.85 * ratio
	}
	if matched >= 2 && ratio >= 0.50 {
		return 0.60 * ratio
	}
	return 0
}

func lexicalOverlapTokens(qTokSet map[string]struct{}, toks []string) float64 {
	if len(qTokSet) == 0 || len(toks) == 0 {
		return 0
	}
	hits := 0
	unique := 0
	for i, t := range toks {
		dup := false
		for j := 0; j < i; j++ {
			if toks[j] == t {
				dup = true
				break
			}
		}
		if dup {
			continue
		}
		unique++
		if _, ok := qTokSet[t]; ok {
			hits++
		}
	}
	if hits == 0 || unique == 0 {
		return 0
	}
	denom := math.Sqrt(float64(len(qTokSet)) * float64(unique))
	if denom <= 0 {
		return 0
	}
	return float64(hits) / denom
}

func computeBM25Authorized(ti *tenantIndex, authDocs map[string]*indexedDoc, qToks []string) map[string]float64 {
	scores := make(map[string]float64)
	totalAuth := len(authDocs)
	if totalAuth == 0 || len(qToks) == 0 {
		return scores
	}

	// Compute average doc length strictly over authorized docs so foreign docs
	// cannot influence even BM25 IDF/length statistics.
	sumLen := 0
	for _, doc := range authDocs {
		sumLen += doc.DocLen
	}
	avgLen := float64(sumLen) / float64(totalAuth)

	seenQ := make(map[string]struct{}, len(qToks))
	for _, term := range qToks {
		if _, dup := seenQ[term]; dup {
			continue
		}
		seenQ[term] = struct{}{}

		postings := ti.postings[term]
		if len(postings) == 0 {
			continue
		}
		var authPostings []*indexedDoc
		for _, sid := range postings {
			if doc := authDocs[sid]; doc != nil {
				authPostings = append(authPostings, doc)
			}
		}
		df := len(authPostings)
		if df == 0 {
			continue
		}
		for _, doc := range authPostings {
			tf := doc.TF[term]
			scores[doc.Record.SourceID] += bm25Score(tf, doc.DocLen, avgLen, df, totalAuth)
		}
	}
	return scores
}

// expandBoundedSubgraph expands a priority-bounded ego-subgraph up to MaxHops
// and MaxExpandedNodes strictly within authorized nodes and edges.
func expandBoundedSubgraph(
	ti *tenantIndex,
	seeds map[string]float64,
	authDocs map[string]*indexedDoc,
	authNodes map[string]*GraphNode,
	authEntityDeg map[string]int,
	ctx AccessContext,
	cfg PPRConfig,
) (map[string]bool, map[string]int) {
	included := make(map[string]bool, cfg.MaxExpandedNodes)
	hopDist := make(map[string]int, cfg.MaxExpandedNodes)

	type frontierItem struct {
		nodeID string
		hop    int
		score  float64
	}
	var current []frontierItem
	for nid, w := range seeds {
		if authNodes[nid] != nil {
			current = append(current, frontierItem{nodeID: nid, hop: 0, score: w})
		}
	}
	sort.Slice(current, func(i, j int) bool {
		if current[i].score != current[j].score {
			return current[i].score > current[j].score
		}
		return current[i].nodeID < current[j].nodeID
	})

	for _, item := range current {
		if len(included) >= cfg.MaxExpandedNodes {
			break
		}
		included[item.nodeID] = true
		hopDist[item.nodeID] = 0
	}

	// Each entity-entity hop traverses 1 edge, and reaching a source passage from an entity
	// adds 1 bipartite hop; allow up to MaxHops+1 graph edges so a MaxHops-entity chain can
	// always land on its supporting source node.
	edgeHorizon := cfg.MaxHops + 1

	for hop := 0; hop < edgeHorizon && len(current) > 0 && len(included) < cfg.MaxExpandedNodes; hop++ {
		nextMap := make(map[string]float64)
		for _, item := range current {
			if !included[item.nodeID] {
				continue
			}
			for _, ed := range ti.adj[item.nodeID] {
				if included[ed.To] {
					continue
				}
				if !isEdgeAuthorized(ed, ctx, authDocs, authNodes) {
					continue
				}
				w := effectiveEdgeWeight(ed, authNodes[ed.To], authEntityDeg, cfg)
				prop := item.score * w
				if prop > nextMap[ed.To] {
					nextMap[ed.To] = prop
				}
			}
		}
		if len(nextMap) == 0 {
			break
		}
		nextSlice := make([]frontierItem, 0, len(nextMap))
		for nid, sc := range nextMap {
			nextSlice = append(nextSlice, frontierItem{nodeID: nid, hop: hop + 1, score: sc})
		}
		sort.Slice(nextSlice, func(i, j int) bool {
			if nextSlice[i].score != nextSlice[j].score {
				return nextSlice[i].score > nextSlice[j].score
			}
			return nextSlice[i].nodeID < nextSlice[j].nodeID
		})
		var admitted []frontierItem
		for _, item := range nextSlice {
			if len(included) >= cfg.MaxExpandedNodes {
				break
			}
			included[item.nodeID] = true
			hopDist[item.nodeID] = item.hop
			admitted = append(admitted, item)
		}
		current = admitted
	}

	return included, hopDist
}

type pathEdge struct {
	edge   GraphEdge
	weight float64
}

type dijkstraItem struct {
	nodeID string
	cost   float64
	hops   int
}

type dijkstraHeap []dijkstraItem

func (h dijkstraHeap) Len() int { return len(h) }
func (h dijkstraHeap) Less(i, j int) bool {
	if h[i].cost != h[j].cost {
		return h[i].cost < h[j].cost
	}
	if h[i].hops != h[j].hops {
		return h[i].hops < h[j].hops
	}
	return h[i].nodeID < h[j].nodeID
}
func (h dijkstraHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *dijkstraHeap) Push(x any)   { *h = append(*h, x.(dijkstraItem)) }
func (h *dijkstraHeap) Pop() any {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[:n-1]
	return item
}

// reconstructEvidencePath finds the deterministic highest-probability path from
// preferredSeeds (or fallbackSeeds) to targetNode inside the authorized PPR subgraph.
func reconstructEvidencePath(
	preferredSeeds map[string]float64,
	fallbackSeeds map[string]float64,
	targetNode string,
	authNodes map[string]*GraphNode,
	adj map[string][]pathEdge,
) EvidencePath {
	p := dijkstraPath(preferredSeeds, targetNode, authNodes, adj)
	if p.TargetNode != "" {
		return p
	}
	return dijkstraPath(fallbackSeeds, targetNode, authNodes, adj)
}

func dijkstraPath(
	seeds map[string]float64,
	targetNode string,
	authNodes map[string]*GraphNode,
	adj map[string][]pathEdge,
) EvidencePath {
	if len(seeds) == 0 || authNodes[targetNode] == nil {
		return EvidencePath{}
	}

	dist := make(map[string]float64)
	prevNode := make(map[string]string)
	prevEdge := make(map[string]GraphEdge)
	prevWeight := make(map[string]float64)

	h := &dijkstraHeap{}
	heap.Init(h)

	seedIDs := make([]string, 0, len(seeds))
	for sid := range seeds {
		seedIDs = append(seedIDs, sid)
	}
	sort.Strings(seedIDs)
	for _, sid := range seedIDs {
		w := seeds[sid]
		if w <= 1e-9 {
			w = 1e-9
		}
		initCost := -math.Log(w) * 0.25
		dist[sid] = initCost
		heap.Push(h, dijkstraItem{nodeID: sid, cost: initCost, hops: 0})
	}

	found := false
	for h.Len() > 0 {
		cur := heap.Pop(h).(dijkstraItem)
		if cur.cost > dist[cur.nodeID]+1e-12 {
			continue
		}
		if cur.nodeID == targetNode {
			found = true
			break
		}
		for _, pe := range adj[cur.nodeID] {
			w := pe.weight
			if w < 1e-9 {
				w = 1e-9
			}
			// Add a small hop regularizer so shorter high-signal paths beat needlessly cyclic detours.
			stepCost := -math.Log(w) + 0.15
			nd := cur.cost + stepCost
			old, seen := dist[pe.edge.To]
			if !seen || nd < old-1e-12 || (math.Abs(nd-old) <= 1e-12 && cur.nodeID < prevNode[pe.edge.To]) {
				dist[pe.edge.To] = nd
				prevNode[pe.edge.To] = cur.nodeID
				prevEdge[pe.edge.To] = pe.edge
				prevWeight[pe.edge.To] = pe.weight
				heap.Push(h, dijkstraItem{nodeID: pe.edge.To, cost: nd, hops: cur.hops + 1})
			}
		}
	}

	if !found {
		return EvidencePath{}
	}

	var stepsRev []PathStep
	curr := targetNode
	for {
		p, ok := prevNode[curr]
		if !ok {
			break
		}
		ed := prevEdge[curr]
		fn := authNodes[p]
		tn := authNodes[curr]
		stepsRev = append(stepsRev, PathStep{
			FromNode:    p,
			FromLabel:   fn.Label,
			FromKind:    fn.Kind,
			Predicate:   ed.Predicate,
			ToNode:      curr,
			ToLabel:     tn.Label,
			ToKind:      tn.Kind,
			EdgeKind:    ed.Kind,
			EdgeWeight:  prevWeight[curr],
			Relevance:   ed.Relevance,
			Reliability: ed.Reliability,
			SourceID:    ed.SourceID,
		})
		curr = p
	}

	steps := make([]PathStep, len(stepsRev))
	for i := range stepsRev {
		steps[i] = stepsRev[len(stepsRev)-1-i]
	}

	var trail strings.Builder
	if len(steps) == 0 {
		trail.WriteString(authNodes[targetNode].Label)
	} else {
		trail.WriteString(steps[0].FromLabel)
		for _, st := range steps {
			trail.WriteString(" --[")
			trail.WriteString(st.Predicate)
			trail.WriteString("]--> ")
			trail.WriteString(st.ToLabel)
		}
	}

	return EvidencePath{
		SeedNode:   curr,
		TargetNode: targetNode,
		Score:      math.Exp(-dist[targetNode]),
		Hops:       len(steps),
		Steps:      steps,
		Trail:      trail.String(),
	}
}
