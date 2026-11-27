/*
 * SPDX-License-Identifier: Apache-2.0
 */

package graphplanner

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"time"
)

// PlannerConfig tunes the cost-based query planner thresholds and cost weights.
type PlannerConfig struct {
	DefaultLatencyBudgetMs float64
	HubDegreeThreshold     int
	MaxGraphFirstFrontier  int
	PureTraversalMaxNodes  int
	VectorShortlistK       int
	MaxTraversalNodes      int
	MaxTraversalEdges      int
	CacheWarmOnTraversal   bool
}

// DefaultPlannerConfig returns balanced cost-model parameters for embedded CSR
// graph + vector retrieval.
func DefaultPlannerConfig() PlannerConfig {
	return PlannerConfig{
		DefaultLatencyBudgetMs: 10.0,
		HubDegreeThreshold:     40,
		MaxGraphFirstFrontier:  180,
		PureTraversalMaxNodes:  250,
		VectorShortlistK:       32,
		MaxTraversalNodes:      400,
		MaxTraversalEdges:      1200,
		CacheWarmOnTraversal:   true,
	}
}

var multiHopCueWords = []string{
	"chain", "transitive", "depends", "dependency", "dependencies",
	"imports", "imported", "calls", "caller", "callers", "downstream",
	"upstream", "caused", "cause", "root cause", "blast", "impact",
	"path", "via", "through", "connected", "neighbors", "neighborhood",
	"multi-hop", "related", "linked", "references", "referenced",
	"mitigates", "defines", "defined", "belongs",
}

func hasMultiHopCue(query string) bool {
	q := strings.ToLower(query)
	for _, cue := range multiHopCueWords {
		if strings.Contains(q, cue) {
			return true
		}
	}
	return false
}

func isValidPlan(p ExecutionPlan) bool {
	switch p {
	case PlanPlainRAGBypass, PlanCachedNeighborhood, PlanPureTraversal, PlanGraphFirst, PlanVectorFirst:
		return true
	default:
		return false
	}
}

type resolvedQueryContext struct {
	csr                 *tenantCSR
	scopeSet            map[string]struct{}
	aclFP               string
	queryTokens         []string
	queryVec            []float32
	anchors             []uint32
	explicitAnchors     bool
	edgeTypeFilter      map[uint16]struct{}
	edgeTypesKey        string
	targetLabelFilter   map[string]struct{}
	targetLabelKey      string
	effHops             int
	topK                int
	budgetMs            float64
	simpleFactual       bool
	lexCandidateCount   int
	estSelectivity      float64
	edgeTypeSelectivity float64
	localDegree         float64
	maxLocalDegree      int
	estFrontierNodes    int
	estEdgesVisited     int
	cacheHitProb        float64
}

func (s *GraphStore) prepareContextLocked(req QueryRequest) (*resolvedQueryContext, error) {
	if strings.TrimSpace(req.Access.TenantID) == "" {
		return nil, ErrInvalidAccessContext
	}
	if req.ForcePlan != "" && !isValidPlan(req.ForcePlan) {
		return nil, fmt.Errorf("%w: %q", ErrInvalidPlan, req.ForcePlan)
	}
	if !s.derivedBuilt {
		return nil, ErrDerivedIndexNotBuilt
	}

	csr := s.tenants[req.Access.TenantID]
	topK := req.TopK
	if topK <= 0 {
		topK = 5
	}
	budgetMs := req.LatencyBudgetMs
	if budgetMs <= 0 {
		budgetMs = s.cfg.Planner.DefaultLatencyBudgetMs
	}

	ctx := &resolvedQueryContext{
		csr:                 csr,
		scopeSet:            scopeSetFromSlice(req.Access.Scopes),
		aclFP:               canonicalScopeFingerprint(req.Access.Scopes),
		queryTokens:         Tokenize(req.Query),
		queryVec:            ComputeEmbedding(req.Query),
		edgeTypesKey:        canonicalStringListKey(req.EdgeTypes),
		targetLabelKey:      canonicalStringListKey(req.TargetLabels),
		effHops:             req.MaxHops,
		topK:                topK,
		budgetMs:            budgetMs,
		estSelectivity:      1.0,
		edgeTypeSelectivity: 1.0,
	}

	if csr == nil || csr.nodeCount() == 0 {
		ctx.simpleFactual = !req.RequireStructuralMatch && len(req.EdgeTypes) == 0 && req.MaxHops <= 0
		return ctx, nil
	}

	// Build target label filter.
	if len(req.TargetLabels) > 0 {
		ctx.targetLabelFilter = make(map[string]struct{}, len(req.TargetLabels))
		matchedLabelNodes := 0
		for _, lb := range req.TargetLabels {
			if lb == "" {
				continue
			}
			ctx.targetLabelFilter[lb] = struct{}{}
			matchedLabelNodes += csr.LabelCounts[lb]
		}
		if csr.nodeCount() > 0 {
			sel := float64(matchedLabelNodes) / float64(csr.nodeCount())
			if sel > 1.0 {
				sel = 1.0
			}
			ctx.estSelectivity = sel
		}
	}

	// Build edge type filter using interned uint16 IDs.
	if len(req.EdgeTypes) > 0 {
		ctx.edgeTypeFilter = make(map[uint16]struct{}, len(req.EdgeTypes))
		matchedEdges := 0
		for _, et := range req.EdgeTypes {
			if et == "" {
				continue
			}
			if etID, ok := csr.EdgeTypeToInt[et]; ok {
				ctx.edgeTypeFilter[etID] = struct{}{}
				matchedEdges += csr.EdgeTypeCounts[etID]
			}
		}
		if csr.edgeCount() > 0 {
			ctx.edgeTypeSelectivity = float64(matchedEdges) / float64(csr.edgeCount())
		} else {
			ctx.edgeTypeSelectivity = 0.0
		}
	}

	// Resolve explicit authorized anchors (strictly within caller's tenant and ACL).
	seenAnchor := make(map[uint32]bool)
	for _, aid := range req.AnchorIDs {
		u, ok := csr.IDToInt[aid]
		if !ok || seenAnchor[u] {
			continue
		}
		if !csr.isNodeAuthorized(u, ctx.scopeSet) {
			continue
		}
		seenAnchor[u] = true
		ctx.anchors = append(ctx.anchors, u)
		ctx.explicitAnchors = true
	}

	// Determine whether this is a simple factual single-hop query where graph
	// expansion adds latency/cost without evidence gain.
	hasRelCue := hasMultiHopCue(req.Query)
	if !req.RequireStructuralMatch && len(req.EdgeTypes) == 0 && len(ctx.anchors) == 0 {
		if req.MaxHops <= 0 || (req.MaxHops == 1 && !hasRelCue) {
			ctx.simpleFactual = true
		}
	}
	if !req.RequireStructuralMatch && len(req.EdgeTypes) == 0 && req.MaxHops <= 0 && !hasRelCue {
		ctx.simpleFactual = true
	}

	// Estimate authorized lexical candidate cardinality via BM25 inverted postings.
	lexSeen := make(map[uint32]struct{})
	for _, tok := range ctx.queryTokens {
		for _, p := range csr.Postings[tok] {
			if !csr.isNodeAuthorized(p.NodeID, ctx.scopeSet) {
				continue
			}
			if !csr.nodeMatchesLabels(p.NodeID, ctx.targetLabelFilter) {
				continue
			}
			lexSeen[p.NodeID] = struct{}{}
		}
	}
	ctx.lexCandidateCount = len(lexSeen)

	// If not simple factual and no explicit anchors, try exact identifier
	// anchor resolution (exact SourceID or compound underscore/path identifier)
	// strictly over authorized nodes with non-zero degree, avoiding false
	// matches on common English words like "score", "token", or "error".
	if !ctx.simpleFactual && len(ctx.anchors) == 0 && len(req.AnchorIDs) == 0 {
		for _, field := range strings.Fields(req.Query) {
			clean := strings.Trim(strings.ToLower(field), "?,.;:!\"'()[]{}")
			if clean == "" {
				continue
			}
			isCompound := strings.ContainsAny(clean, "_:/.")
			if !isCompound {
				continue
			}
			if nodes, ok := csr.TitleToNodes[clean]; ok && len(nodes) == 1 {
				u := nodes[0]
				if !seenAnchor[u] && csr.isNodeAuthorized(u, ctx.scopeSet) && csr.totalDegree(u) > 0 {
					seenAnchor[u] = true
					ctx.anchors = append(ctx.anchors, u)
				}
			}
		}
		sort.Slice(ctx.anchors, func(i, j int) bool { return ctx.anchors[i] < ctx.anchors[j] })
	}

	if ctx.effHops <= 0 && !ctx.simpleFactual {
		ctx.effHops = 1
	}
	if ctx.effHops > 4 {
		ctx.effHops = 4
	}

	// Compute local graph degree and bounded frontier estimate for resolved anchors.
	if len(ctx.anchors) > 0 {
		sumDeg := 0
		maxDeg := 0
		hotCount := 0
		for _, a := range ctx.anchors {
			d := csr.totalDegree(a)
			sumDeg += d
			if d > maxDeg {
				maxDeg = d
			}
			if ctx.effHops > 0 && s.cache.PeekHot(csr.TenantID, ctx.aclFP, a, ctx.edgeTypesKey, ctx.targetLabelKey, ctx.effHops) {
				hotCount++
			}
		}
		ctx.localDegree = float64(sumDeg) / float64(len(ctx.anchors))
		ctx.maxLocalDegree = maxDeg
		ctx.cacheHitProb = float64(hotCount) / float64(len(ctx.anchors))

		// Branching factor estimate adjusted by edge-type selectivity.
		effBranch := ctx.localDegree * math.Max(0.15, ctx.edgeTypeSelectivity)
		if effBranch < 1.0 && ctx.localDegree > 0 {
			effBranch = 1.0
		}
		frontier := float64(len(ctx.anchors))
		edgesEst := 0.0
		meanNext := math.Max(1.0, csr.MeanDegree*math.Max(0.2, ctx.edgeTypeSelectivity))
		for h := 0; h < ctx.effHops; h++ {
			branch := effBranch
			if h > 0 {
				branch = meanNext
			}
			edgesEst += frontier * branch
			frontier *= branch
			if frontier > float64(csr.nodeCount()) {
				frontier = float64(csr.nodeCount())
				break
			}
		}
		ctx.estFrontierNodes = int(math.Ceil(frontier))
		ctx.estEdgesVisited = int(math.Ceil(edgesEst))
	} else {
		ctx.localDegree = csr.MeanDegree
	}

	// Combine predicate selectivity with lexical selectivity when available.
	if csr.nodeCount() > 0 && ctx.lexCandidateCount > 0 {
		lexSel := float64(ctx.lexCandidateCount) / float64(csr.nodeCount())
		if lexSel < ctx.estSelectivity {
			ctx.estSelectivity = lexSel
		}
	}

	return ctx, nil
}

// PlanQuery evaluates all five physical execution plans without executing them
// and returns the chosen plan and full cost breakdown.
func (s *GraphStore) PlanQuery(req QueryRequest) (QueryExplanation, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	ctx, err := s.prepareContextLocked(req)
	if err != nil {
		return QueryExplanation{}, err
	}
	return s.planFromContextLocked(req, ctx), nil
}

func (s *GraphStore) planFromContextLocked(req QueryRequest, ctx *resolvedQueryContext) QueryExplanation {
	cfg := s.cfg.Planner
	nNodes := 0
	if ctx.csr != nil {
		nNodes = ctx.csr.nodeCount()
	}

	alts := make([]PlanCostEstimate, 0, len(AllExecutionPlans))

	// 1. plain_rag_bypass
	{
		cand := ctx.lexCandidateCount
		if cand == 0 {
			cand = minInt(nNodes, ctx.topK*4)
		}
		lat := 0.014 + 0.00028*float64(cand)
		feasible := true
		cost := lat * 10.0
		rationale := "direct BM25 + lightweight dense scoring with zero graph traversal"
		if ctx.simpleFactual {
			// Strongly preferred on simple factual single-hop queries.
			cost *= 0.35
			rationale = "simple factual single-hop query detected; bypassing graph avoids traversal latency without evidence loss"
		} else if req.RequireStructuralMatch || len(req.EdgeTypes) > 0 {
			// Cannot satisfy explicit typed structural/edge predicate without graph.
			feasible = false
			cost += 100.0
			rationale = "infeasible for explicit structural/edge-type graph constraints"
		} else if len(ctx.anchors) > 0 && ctx.effHops >= 1 {
			cost += 18.0
			rationale = "multi-hop/anchored relational query loses structural evidence under plain RAG bypass"
		} else {
			cost += 6.0
			rationale = "relational query without graph verification risks lower multi-hop precision"
		}
		if lat > ctx.budgetMs {
			cost += (lat - ctx.budgetMs) * 20.0
		}
		alts = append(alts, PlanCostEstimate{
			Plan:                 PlanPlainRAGBypass,
			EstimatedLatencyMs:   round4(lat),
			EstimatedCandidates:  cand,
			EstimatedSelectivity: round4(ctx.estSelectivity),
			EstimatedLocalDegree: round4(ctx.localDegree),
			CacheHitProbability:  round4(ctx.cacheHitProb),
			EstimatedCostScore:   round4(cost),
			Feasible:             feasible,
			Rationale:            rationale,
		})
	}

	// 2. cached_neighborhood
	{
		hasAnchors := len(ctx.anchors) > 0 && ctx.effHops >= 1
		feasible := hasAnchors && ctx.cacheHitProb >= 0.5
		cand := minInt(ctx.estFrontierNodes, cfg.MaxTraversalNodes)
		if cand < ctx.topK {
			cand = ctx.topK
		}
		lat := 0.010 + 0.00018*float64(cand)
		cost := lat * 10.0
		rationale := "O(1) ACL-scoped LRU neighborhood lookup + compact local rerank"
		if !hasAnchors {
			rationale = "requires resolved anchor node(s) and hops >= 1"
			cost += 100.0
		} else if ctx.cacheHitProb < 0.5 {
			rationale = "anchor neighborhood cold in LRU cache (hit_prob < 0.5)"
			cost += 50.0
		} else {
			// Hot cache! Huge speedup especially under tight budgets or moderate degree.
			cost *= (1.0 - 0.65*ctx.cacheHitProb)
			if ctx.budgetMs <= 0.5 {
				cost *= 0.5
				rationale = "hot LRU neighborhood cache meets tight latency budget with zero edge traversal"
			}
		}
		if ctx.simpleFactual {
			cost += 12.0
		}
		alts = append(alts, PlanCostEstimate{
			Plan:                 PlanCachedNeighborhood,
			EstimatedLatencyMs:   round4(lat),
			EstimatedCandidates:  cand,
			EstimatedSelectivity: round4(ctx.estSelectivity),
			EstimatedLocalDegree: round4(ctx.localDegree),
			CacheHitProbability:  round4(ctx.cacheHitProb),
			EstimatedCostScore:   round4(cost),
			Feasible:             feasible,
			Rationale:            rationale,
		})
	}

	// 3. pure_traversal
	{
		hasAnchors := len(ctx.anchors) > 0 && ctx.effHops >= 1
		isHub := ctx.maxLocalDegree > cfg.HubDegreeThreshold
		cand := minInt(ctx.estFrontierNodes, cfg.MaxTraversalNodes)
		edges := minInt(ctx.estEdgesVisited, cfg.MaxTraversalEdges)
		lat := 0.016 + 0.00055*float64(edges) + 0.0002*float64(cand)
		feasible := hasAnchors && !isHub && ctx.estFrontierNodes <= cfg.PureTraversalMaxNodes
		cost := lat * 10.0
		rationale := "bounded k-hop typed CSR traversal without vector scoring overhead"

		if !hasAnchors {
			rationale = "requires exact/resolved anchor node(s) and hops >= 1"
			cost += 100.0
		} else if isHub {
			rationale = fmt.Sprintf("anchor degree %d exceeds hub threshold %d; pure traversal risks hub explosion", ctx.maxLocalDegree, cfg.HubDegreeThreshold)
			cost += 45.0
		} else if req.RequireStructuralMatch || (len(req.EdgeTypes) > 0 && ctx.edgeTypeSelectivity <= 0.45 && len(ctx.queryTokens) <= 6) {
			cost *= 0.55
			rationale = "exact anchor with selective typed edge/structural predicate favors pure CSR traversal"
		} else if len(ctx.queryTokens) >= 4 && !req.RequireStructuralMatch {
			// Semantic query needs vector/lexical scoring beyond pure topology.
			cost += 8.5
			rationale = "rich textual query benefits from semantic/lexical reranking beyond pure structural traversal"
		}
		if ctx.simpleFactual {
			cost += 15.0
		}
		if lat > ctx.budgetMs {
			cost += (lat - ctx.budgetMs) * 25.0
		}
		alts = append(alts, PlanCostEstimate{
			Plan:                 PlanPureTraversal,
			EstimatedLatencyMs:   round4(lat),
			EstimatedCandidates:  cand,
			EstimatedSelectivity: round4(ctx.estSelectivity),
			EstimatedLocalDegree: round4(ctx.localDegree),
			CacheHitProbability:  round4(ctx.cacheHitProb),
			EstimatedCostScore:   round4(cost),
			Feasible:             feasible,
			Rationale:            rationale,
		})
	}

	// 4. graph_first
	{
		hasAnchors := len(ctx.anchors) > 0 && ctx.effHops >= 1
		isHub := ctx.maxLocalDegree > cfg.HubDegreeThreshold
		cand := minInt(ctx.estFrontierNodes, cfg.MaxTraversalNodes)
		edges := minInt(ctx.estEdgesVisited, cfg.MaxTraversalEdges)
		lat := 0.022 + 0.00060*float64(edges) + 0.0014*float64(cand)
		feasible := hasAnchors && !isHub && ctx.estFrontierNodes <= cfg.MaxGraphFirstFrontier
		cost := lat * 10.0
		rationale := "seed entity CSR expansion followed by hybrid vector + lexical reranking"

		if !hasAnchors {
			rationale = "requires resolved anchor node(s) and hops >= 1"
			cost += 100.0
		} else if isHub {
			rationale = fmt.Sprintf("anchor local degree %d exceeds hub threshold %d (hub explosion); defer to vector_first", ctx.maxLocalDegree, cfg.HubDegreeThreshold)
			cost += 50.0
		} else if ctx.estFrontierNodes > cfg.MaxGraphFirstFrontier {
			rationale = fmt.Sprintf("estimated frontier %d exceeds graph_first cap %d", ctx.estFrontierNodes, cfg.MaxGraphFirstFrontier)
			cost += 35.0
		} else if !ctx.simpleFactual {
			// Ideal regime: manageable anchor neighborhood + textual/semantic query.
			cost *= 0.65
			if req.RequireStructuralMatch {
				cost += 2.0
			}
			rationale = "bounded local degree around anchor(s) with rich query semantics favors graph_first expansion + hybrid rerank"
		}
		if ctx.simpleFactual {
			cost += 14.0
		}
		if lat > ctx.budgetMs {
			cost += (lat - ctx.budgetMs) * 25.0
		}
		alts = append(alts, PlanCostEstimate{
			Plan:                 PlanGraphFirst,
			EstimatedLatencyMs:   round4(lat),
			EstimatedCandidates:  cand,
			EstimatedSelectivity: round4(ctx.estSelectivity),
			EstimatedLocalDegree: round4(ctx.localDegree),
			CacheHitProbability:  round4(ctx.cacheHitProb),
			EstimatedCostScore:   round4(cost),
			Feasible:             feasible,
			Rationale:            rationale,
		})
	}

	// 5. vector_first
	{
		shortlist := minInt(nNodes, cfg.VectorShortlistK)
		if shortlist < ctx.topK {
			shortlist = ctx.topK
		}
		verifyEdges := shortlist * minInt(8, int(math.Ceil(ctx.csrMeanDegree()))+1)
		lat := 0.028 + 0.00018*float64(nNodes) + 0.00045*float64(verifyEdges)
		feasible := true
		cost := lat * 10.0
		rationale := "hybrid vector/BM25 shortlist followed by bounded graph neighborhood verification"

		if ctx.maxLocalDegree > cfg.HubDegreeThreshold {
			cost *= 0.60
			rationale = fmt.Sprintf("high-degree hub anchor (degree=%d > %d) triggers vector_first to avoid hub explosion", ctx.maxLocalDegree, cfg.HubDegreeThreshold)
		} else if len(ctx.anchors) == 0 && !ctx.simpleFactual {
			cost *= 0.65
			rationale = "no exact entity anchor on relational/multi-hop query; vector_first seeds shortlist before graph verification"
		} else if ctx.simpleFactual {
			cost += 5.0
			rationale = "simple factual query does not need post-shortlist graph verification"
		} else if req.RequireStructuralMatch && len(ctx.anchors) > 0 && ctx.maxLocalDegree <= cfg.HubDegreeThreshold {
			cost += 9.0
			rationale = "strict structural traversal from low-degree anchor is more direct than global vector shortlist"
		}
		if lat > ctx.budgetMs {
			cost += (lat - ctx.budgetMs) * 20.0
		}
		alts = append(alts, PlanCostEstimate{
			Plan:                 PlanVectorFirst,
			EstimatedLatencyMs:   round4(lat),
			EstimatedCandidates:  shortlist,
			EstimatedSelectivity: round4(ctx.estSelectivity),
			EstimatedLocalDegree: round4(ctx.localDegree),
			CacheHitProbability:  round4(ctx.cacheHitProb),
			EstimatedCostScore:   round4(cost),
			Feasible:             feasible,
			Rationale:            rationale,
		})
	}

	// Choose winning plan (or respect explicit ForcePlan for controlled baselines).
	selected := PlanVectorFirst
	reason := ""
	forced := false

	if req.ForcePlan != "" {
		selected = req.ForcePlan
		forced = true
		reason = fmt.Sprintf("plan forced by caller override (%s)", req.ForcePlan)
	} else {
		bestIdx := -1
		for i, a := range alts {
			if !a.Feasible {
				continue
			}
			if bestIdx == -1 || a.EstimatedCostScore < alts[bestIdx].EstimatedCostScore {
				bestIdx = i
			}
		}
		if bestIdx == -1 {
			// Fallback to lowest overall cost if no plan marked feasible.
			bestIdx = 0
			for i := 1; i < len(alts); i++ {
				if alts[i].EstimatedCostScore < alts[bestIdx].EstimatedCostScore {
					bestIdx = i
				}
			}
		}
		selected = alts[bestIdx].Plan
		reason = alts[bestIdx].Rationale
	}

	return QueryExplanation{
		SelectedPlan:          selected,
		Forced:                forced,
		LatencyBudgetMs:       round4(ctx.budgetMs),
		AnchorCount:           len(ctx.anchors),
		RequestedHops:         ctx.effHops,
		EstimatedSelectivity:  round4(ctx.estSelectivity),
		EstimatedLocalDegree:  round4(ctx.localDegree),
		CacheHitProbability:   round4(ctx.cacheHitProb),
		SimpleFactualDetected: ctx.simpleFactual,
		Alternatives:          alts,
		SelectionReason:       reason,
	}
}

func (ctx *resolvedQueryContext) csrMeanDegree() float64 {
	if ctx.csr == nil {
		return 0
	}
	return ctx.csr.MeanDegree
}

// ExecuteQuery plans and executes a query with strict pre-return tenant/ACL
// enforcement, returning scored hits, full plan explanation, and telemetry.
func (s *GraphStore) ExecuteQuery(req QueryRequest) (QueryResponse, error) {
	start := time.Now()

	s.mu.RLock()
	defer s.mu.RUnlock()

	ctx, err := s.prepareContextLocked(req)
	if err != nil {
		return QueryResponse{}, err
	}

	expl := s.planFromContextLocked(req, ctx)
	var hits []ScoredHit
	var tel QueryTelemetry

	if ctx.csr != nil && ctx.csr.nodeCount() > 0 {
		switch expl.SelectedPlan {
		case PlanPlainRAGBypass:
			hits, tel = s.execPlainRAGBypassLocked(req, ctx)
		case PlanCachedNeighborhood:
			hits, tel = s.execCachedNeighborhoodLocked(req, ctx)
		case PlanPureTraversal:
			hits, tel = s.execPureTraversalLocked(req, ctx)
		case PlanGraphFirst:
			hits, tel = s.execGraphFirstLocked(req, ctx)
		case PlanVectorFirst:
			hits, tel = s.execVectorFirstLocked(req, ctx)
		}
	}

	// Final defense-in-depth ACL check & token accounting before returning.
	tokCount := 0
	for _, h := range hits {
		tokCount += EstimateTokens(h.Title + " " + h.Text)
	}
	tel.EstimatedTokens = tokCount
	tel.ActualLatencyMs = round4(float64(time.Since(start).Nanoseconds()) / 1e6)

	return QueryResponse{
		Hits:        hits,
		Plan:        expl.SelectedPlan,
		Explanation: expl,
		Telemetry:   tel,
	}, nil
}

// --- Physical Executor 1: plain_rag_bypass ---

func (s *GraphStore) execPlainRAGBypassLocked(req QueryRequest, ctx *resolvedQueryContext) ([]ScoredHit, QueryTelemetry) {
	csr := ctx.csr
	var tel QueryTelemetry

	bm25Scores, visited := computeBM25Scores(csr, ctx.queryTokens, ctx.scopeSet, ctx.targetLabelFilter)
	tel.NodesVisited = visited

	type scoredCand struct {
		u     uint32
		lex   float64
		vec   float64
		total float64
	}
	cands := make([]scoredCand, 0, len(bm25Scores))
	maxLex := 0.0
	for _, sc := range bm25Scores {
		if sc > maxLex {
			maxLex = sc
		}
	}
	if maxLex <= 0 {
		maxLex = 1.0
	}

	if len(bm25Scores) > 0 {
		for u, sc := range bm25Scores {
			normLex := sc / maxLex
			vec := positiveCosine(ctx.queryVec, csr.vectorSlice(u))
			total := 0.70*normLex + 0.30*vec
			cands = append(cands, scoredCand{u: u, lex: normLex, vec: vec, total: total})
		}
	} else {
		// Fallback to dense scan only if zero lexical postings matched.
		n := csr.nodeCount()
		for i := 0; i < n; i++ {
			u := uint32(i)
			if !csr.isNodeAuthorized(u, ctx.scopeSet) || !csr.nodeMatchesLabels(u, ctx.targetLabelFilter) {
				continue
			}
			tel.NodesVisited++
			vec := positiveCosine(ctx.queryVec, csr.vectorSlice(u))
			if vec > 0 {
				cands = append(cands, scoredCand{u: u, vec: vec, total: vec})
			}
		}
	}

	tel.CandidatesEvaluated = len(cands)
	sort.Slice(cands, func(i, j int) bool {
		if cands[i].total != cands[j].total {
			return cands[i].total > cands[j].total
		}
		return csr.SourceIDs[cands[i].u] < csr.SourceIDs[cands[j].u]
	})

	limit := minInt(ctx.topK, len(cands))
	hits := make([]ScoredHit, 0, limit)
	for i := 0; i < limit; i++ {
		c := cands[i]
		hits = append(hits, buildScoredHit(csr, c.u, c.total, c.vec, c.lex, 0, 0, nil))
	}
	return hits, tel
}

// --- Physical Executor 2: cached_neighborhood ---

func (s *GraphStore) execCachedNeighborhoodLocked(req QueryRequest, ctx *resolvedQueryContext) ([]ScoredHit, QueryTelemetry) {
	csr := ctx.csr
	var tel QueryTelemetry

	if len(ctx.anchors) == 0 {
		// Graceful fallback to vector_first if caller forced cached_neighborhood without anchors.
		return s.execVectorFirstLocked(req, ctx)
	}

	hops := ctx.effHops
	if hops <= 0 {
		hops = 1
	}

	merged := make(map[uint32]cachedNeighbor)
	anchorSet := make(map[uint32]bool, len(ctx.anchors))
	for _, a := range ctx.anchors {
		anchorSet[a] = true
		if cached, ok := s.cache.Get(csr.TenantID, ctx.aclFP, a, ctx.edgeTypesKey, ctx.targetLabelKey, hops); ok {
			tel.CacheHits++
			for _, cn := range cached {
				if !csr.isNodeAuthorized(cn.NodeID, ctx.scopeSet) {
					continue
				}
				if prev, exists := merged[cn.NodeID]; !exists || cn.GraphScore > prev.GraphScore {
					merged[cn.NodeID] = cn
				}
			}
		} else {
			tel.CacheMisses++
			neighbors, visited, edges := s.traverseSingleAnchorLocked(csr, a, hops, ctx)
			tel.NodesVisited += visited
			tel.EdgesTraversed += edges
			s.cache.Put(csr.TenantID, ctx.aclFP, a, ctx.edgeTypesKey, ctx.targetLabelKey, hops, neighbors)
			for _, cn := range neighbors {
				if prev, exists := merged[cn.NodeID]; !exists || cn.GraphScore > prev.GraphScore {
					merged[cn.NodeID] = cn
				}
			}
		}
	}

	return s.rankNeighborhoodCandidatesLocked(req, ctx, merged, anchorSet, tel)
}

// --- Physical Executor 3: pure_traversal ---

func (s *GraphStore) execPureTraversalLocked(req QueryRequest, ctx *resolvedQueryContext) ([]ScoredHit, QueryTelemetry) {
	csr := ctx.csr
	var tel QueryTelemetry

	if len(ctx.anchors) == 0 {
		return s.execPlainRAGBypassLocked(req, ctx)
	}

	hops := ctx.effHops
	if hops <= 0 {
		hops = 1
	}

	merged := make(map[uint32]cachedNeighbor)
	anchorSet := make(map[uint32]bool, len(ctx.anchors))
	for _, a := range ctx.anchors {
		anchorSet[a] = true
		neighbors, visited, edges := s.traverseSingleAnchorLocked(csr, a, hops, ctx)
		tel.NodesVisited += visited
		tel.EdgesTraversed += edges
		if s.cfg.Planner.CacheWarmOnTraversal {
			s.cache.Put(csr.TenantID, ctx.aclFP, a, ctx.edgeTypesKey, ctx.targetLabelKey, hops, neighbors)
		}
		for _, cn := range neighbors {
			if prev, exists := merged[cn.NodeID]; !exists || cn.GraphScore > prev.GraphScore {
				merged[cn.NodeID] = cn
			}
		}
	}

	// Pure traversal ranks strictly by structural hop distance & edge weight (no vector math!).
	list := make([]cachedNeighbor, 0, len(merged))
	for _, cn := range merged {
		if !req.IncludeAnchors && anchorSet[cn.NodeID] {
			continue
		}
		if !csr.nodeMatchesLabels(cn.NodeID, ctx.targetLabelFilter) {
			continue
		}
		list = append(list, cn)
	}
	tel.CandidatesEvaluated = len(list)

	// Lightweight lexical tie-breaker only if pure graph scores are equal.
	sort.Slice(list, func(i, j int) bool {
		if list[i].GraphScore != list[j].GraphScore {
			return list[i].GraphScore > list[j].GraphScore
		}
		if list[i].HopDistance != list[j].HopDistance {
			return list[i].HopDistance < list[j].HopDistance
		}
		return csr.SourceIDs[list[i].NodeID] < csr.SourceIDs[list[j].NodeID]
	})

	limit := minInt(ctx.topK, len(list))
	hits := make([]ScoredHit, 0, limit)
	for i := 0; i < limit; i++ {
		cn := list[i]
		hits = append(hits, buildScoredHit(csr, cn.NodeID, cn.GraphScore, 0, 0, cn.GraphScore, cn.HopDistance, pathToSourceIDs(csr, cn.Path)))
	}
	return hits, tel
}

// --- Physical Executor 4: graph_first ---

func (s *GraphStore) execGraphFirstLocked(req QueryRequest, ctx *resolvedQueryContext) ([]ScoredHit, QueryTelemetry) {
	csr := ctx.csr
	var tel QueryTelemetry

	if len(ctx.anchors) == 0 {
		return s.execVectorFirstLocked(req, ctx)
	}

	hops := ctx.effHops
	if hops <= 0 {
		hops = 1
	}

	merged := make(map[uint32]cachedNeighbor)
	anchorSet := make(map[uint32]bool, len(ctx.anchors))
	for _, a := range ctx.anchors {
		anchorSet[a] = true
		neighbors, visited, edges := s.traverseSingleAnchorLocked(csr, a, hops, ctx)
		tel.NodesVisited += visited
		tel.EdgesTraversed += edges
		if s.cfg.Planner.CacheWarmOnTraversal {
			s.cache.Put(csr.TenantID, ctx.aclFP, a, ctx.edgeTypesKey, ctx.targetLabelKey, hops, neighbors)
		}
		for _, cn := range neighbors {
			if prev, exists := merged[cn.NodeID]; !exists || cn.GraphScore > prev.GraphScore {
				merged[cn.NodeID] = cn
			}
		}
	}

	return s.rankNeighborhoodCandidatesLocked(req, ctx, merged, anchorSet, tel)
}

func (s *GraphStore) rankNeighborhoodCandidatesLocked(
	req QueryRequest,
	ctx *resolvedQueryContext,
	merged map[uint32]cachedNeighbor,
	anchorSet map[uint32]bool,
	tel QueryTelemetry,
) ([]ScoredHit, QueryTelemetry) {
	csr := ctx.csr

	// Compute lexical overlap scores strictly for nodes inside the reachable subgraph.
	bm25All, _ := computeBM25Scores(csr, ctx.queryTokens, ctx.scopeSet, ctx.targetLabelFilter)
	maxLex := 0.0
	for u := range merged {
		if sc := bm25All[u]; sc > maxLex {
			maxLex = sc
		}
	}
	if maxLex <= 0 {
		maxLex = 1.0
	}

	type scoredItem struct {
		cn    cachedNeighbor
		lex   float64
		vec   float64
		total float64
	}
	items := make([]scoredItem, 0, len(merged))
	for _, cn := range merged {
		if !req.IncludeAnchors && anchorSet[cn.NodeID] {
			continue
		}
		if !csr.nodeMatchesLabels(cn.NodeID, ctx.targetLabelFilter) {
			continue
		}
		lex := bm25All[cn.NodeID] / maxLex
		vec := positiveCosine(ctx.queryVec, csr.vectorSlice(cn.NodeID))
		// Combine graph topology score with hybrid lexical + semantic score.
		total := 0.45*cn.GraphScore + 0.35*lex + 0.20*vec
		items = append(items, scoredItem{cn: cn, lex: lex, vec: vec, total: total})
	}
	tel.CandidatesEvaluated = len(items)

	sort.Slice(items, func(i, j int) bool {
		if items[i].total != items[j].total {
			return items[i].total > items[j].total
		}
		if items[i].cn.HopDistance != items[j].cn.HopDistance {
			return items[i].cn.HopDistance < items[j].cn.HopDistance
		}
		return csr.SourceIDs[items[i].cn.NodeID] < csr.SourceIDs[items[j].cn.NodeID]
	})

	limit := minInt(ctx.topK, len(items))
	hits := make([]ScoredHit, 0, limit)
	for i := 0; i < limit; i++ {
		it := items[i]
		hits = append(hits, buildScoredHit(csr, it.cn.NodeID, it.total, it.vec, it.lex, it.cn.GraphScore, it.cn.HopDistance, pathToSourceIDs(csr, it.cn.Path)))
	}
	return hits, tel
}

// --- Physical Executor 5: vector_first ---

func (s *GraphStore) execVectorFirstLocked(req QueryRequest, ctx *resolvedQueryContext) ([]ScoredHit, QueryTelemetry) {
	csr := ctx.csr
	var tel QueryTelemetry

	bm25Scores, _ := computeBM25Scores(csr, ctx.queryTokens, ctx.scopeSet, ctx.targetLabelFilter)
	maxLex := 0.0
	for _, sc := range bm25Scores {
		if sc > maxLex {
			maxLex = sc
		}
	}
	if maxLex <= 0 {
		maxLex = 1.0
	}

	anchorSet := make(map[uint32]bool, len(ctx.anchors))
	for _, a := range ctx.anchors {
		anchorSet[a] = true
	}

	type shortlistItem struct {
		u     uint32
		vec   float64
		lex   float64
		base  float64
		graph float64
		hop   int
		path  []uint32
		total float64
	}

	n := csr.nodeCount()
	all := make([]shortlistItem, 0, minInt(n, 128))
	for i := 0; i < n; i++ {
		u := uint32(i)
		if !csr.isNodeAuthorized(u, ctx.scopeSet) {
			continue
		}
		if !req.IncludeAnchors && anchorSet[u] {
			continue
		}
		if !csr.nodeMatchesLabels(u, ctx.targetLabelFilter) {
			continue
		}
		tel.NodesVisited++
		vec := positiveCosine(ctx.queryVec, csr.vectorSlice(u))
		lex := bm25Scores[u] / maxLex
		base := 0.50*vec + 0.50*lex
		if base > 0 || len(ctx.anchors) > 0 {
			all = append(all, shortlistItem{u: u, vec: vec, lex: lex, base: base})
		}
	}

	sort.Slice(all, func(i, j int) bool {
		if all[i].base != all[j].base {
			return all[i].base > all[j].base
		}
		return csr.SourceIDs[all[i].u] < csr.SourceIDs[all[j].u]
	})

	shortlistK := minInt(len(all), s.cfg.Planner.VectorShortlistK)
	if shortlistK < ctx.topK {
		shortlistK = minInt(len(all), ctx.topK)
	}
	shortlist := all[:shortlistK]
	shortlistMap := make(map[uint32]float64, shortlistK)
	for _, it := range shortlist {
		shortlistMap[it.u] = it.base
	}

	// Bounded graph verification/rerank on ONLY the top-K vector shortlist!
	// This avoids hub explosion because we inspect edges from each shortlisted
	// candidate (bounded by shortlistK) rather than fanning out from a giant hub.
	for i := range shortlist {
		u := shortlist[i].u
		bestGraph := 0.0
		bestHop := 0
		var bestPath []uint32

		// Check forward edges from u.
		for ei := csr.FwdOffsets[u]; ei < csr.FwdOffsets[u+1]; ei++ {
			edgeIdx := csr.FwdEdgeIdx[ei]
			if !csr.isEdgeAuthorized(edgeIdx, ctx.scopeSet) {
				continue
			}
			if len(ctx.edgeTypeFilter) > 0 {
				if _, ok := ctx.edgeTypeFilter[csr.FwdEdgeTypes[ei]]; !ok {
					continue
				}
			}
			v := csr.FwdTargets[ei]
			if !csr.isNodeAuthorized(v, ctx.scopeSet) {
				continue
			}
			tel.EdgesTraversed++
			w := float64(csr.FwdWeights[ei])
			if anchorSet[v] {
				sc := math.Min(1.0, 0.85*w)
				if sc > bestGraph {
					bestGraph = sc
					bestHop = 1
					bestPath = []uint32{v, u}
				}
			} else if peerBase, ok := shortlistMap[v]; ok {
				sc := math.Min(0.60, 0.45*peerBase*w)
				if sc > bestGraph {
					bestGraph = sc
					bestHop = 1
					bestPath = []uint32{v, u}
				}
			}
		}

		// Check reverse edges into u (crucial when anchor -> u!).
		for ei := csr.RevOffsets[u]; ei < csr.RevOffsets[u+1]; ei++ {
			edgeIdx := csr.RevEdgeIdx[ei]
			if !csr.isEdgeAuthorized(edgeIdx, ctx.scopeSet) {
				continue
			}
			if len(ctx.edgeTypeFilter) > 0 {
				if _, ok := ctx.edgeTypeFilter[csr.RevEdgeTypes[ei]]; !ok {
					continue
				}
			}
			src := csr.RevSources[ei]
			if !csr.isNodeAuthorized(src, ctx.scopeSet) {
				continue
			}
			tel.EdgesTraversed++
			w := float64(csr.RevWeights[ei])
			if anchorSet[src] {
				sc := math.Min(1.0, 0.90*w)
				if sc > bestGraph {
					bestGraph = sc
					bestHop = 1
					bestPath = []uint32{src, u}
				}
			} else if peerBase, ok := shortlistMap[src]; ok {
				sc := math.Min(0.60, 0.45*peerBase*w)
				if sc > bestGraph {
					bestGraph = sc
					bestHop = 1
					bestPath = []uint32{src, u}
				}
			}
		}

		shortlist[i].graph = bestGraph
		shortlist[i].hop = bestHop
		shortlist[i].path = bestPath
		shortlist[i].total = 0.60*shortlist[i].base + 0.40*bestGraph
	}

	// If shortlist has multi-hop query with explicit anchors and no 1-hop match,
	// expand 1 hop out from the top 4 vector candidates to surface linked bridge targets.
	extra := make([]shortlistItem, 0)
	if ctx.effHops >= 2 && len(ctx.anchors) == 0 && len(shortlist) > 0 {
		seedN := minInt(4, len(shortlist))
		seenExtra := make(map[uint32]bool, shortlistK)
		for _, it := range shortlist {
			seenExtra[it.u] = true
		}
		for i := 0; i < seedN; i++ {
			seed := shortlist[i]
			for ei := csr.FwdOffsets[seed.u]; ei < csr.FwdOffsets[seed.u+1]; ei++ {
				if !csr.isEdgeAuthorized(csr.FwdEdgeIdx[ei], ctx.scopeSet) {
					continue
				}
				if len(ctx.edgeTypeFilter) > 0 {
					if _, ok := ctx.edgeTypeFilter[csr.FwdEdgeTypes[ei]]; !ok {
						continue
					}
				}
				v := csr.FwdTargets[ei]
				if seenExtra[v] || !csr.isNodeAuthorized(v, ctx.scopeSet) || !csr.nodeMatchesLabels(v, ctx.targetLabelFilter) {
					continue
				}
				seenExtra[v] = true
				tel.EdgesTraversed++
				vec := positiveCosine(ctx.queryVec, csr.vectorSlice(v))
				lex := bm25Scores[v] / maxLex
				base := 0.50*vec + 0.50*lex
				gScore := math.Min(1.0, 0.70*seed.base*float64(csr.FwdWeights[ei]))
				extra = append(extra, shortlistItem{
					u:     v,
					vec:   vec,
					lex:   lex,
					base:  base,
					graph: gScore,
					hop:   1,
					path:  []uint32{seed.u, v},
					total: 0.50*base + 0.50*gScore,
				})
			}
		}
	}

	combined := append(shortlist, extra...)
	tel.CandidatesEvaluated = len(combined)

	sort.Slice(combined, func(i, j int) bool {
		if combined[i].total != combined[j].total {
			return combined[i].total > combined[j].total
		}
		return csr.SourceIDs[combined[i].u] < csr.SourceIDs[combined[j].u]
	})

	limit := minInt(ctx.topK, len(combined))
	hits := make([]ScoredHit, 0, limit)
	for i := 0; i < limit; i++ {
		c := combined[i]
		hits = append(hits, buildScoredHit(csr, c.u, c.total, c.vec, c.lex, c.graph, c.hop, pathToSourceIDs(csr, c.path)))
	}
	return hits, tel
}

// --- Bounded CSR Traversal Helper (Pre-Queue ACL Enforcement) ---

func (s *GraphStore) traverseSingleAnchorLocked(
	csr *tenantCSR,
	anchor uint32,
	maxHops int,
	ctx *resolvedQueryContext,
) ([]cachedNeighbor, int, int) {
	if !csr.isNodeAuthorized(anchor, ctx.scopeSet) {
		return nil, 0, 0
	}

	maxNodes := s.cfg.Planner.MaxTraversalNodes
	maxEdges := s.cfg.Planner.MaxTraversalEdges

	type qNode struct {
		u     uint32
		hop   int
		score float64
		path  []uint32
	}

	visited := map[uint32]cachedNeighbor{
		anchor: {
			NodeID:      anchor,
			HopDistance: 0,
			GraphScore:  1.0,
			Path:        []uint32{anchor},
		},
	}
	queue := []qNode{{u: anchor, hop: 0, score: 1.0, path: []uint32{anchor}}}
	nodesVisited := 1
	edgesTraversed := 0

	tryVisit := func(cur qNode, next uint32, edgeIdx uint32, edgeType uint16, weight float32, dirFactor float64) bool {
		if edgesTraversed >= maxEdges {
			return false
		}
		if !csr.isEdgeAuthorized(edgeIdx, ctx.scopeSet) {
			return true
		}
		if len(ctx.edgeTypeFilter) > 0 {
			if _, ok := ctx.edgeTypeFilter[edgeType]; !ok {
				return true
			}
		}
		if !csr.isNodeAuthorized(next, ctx.scopeSet) {
			return true
		}
		edgesTraversed++

		nextHop := cur.hop + 1
		// Hop-decayed weighted path score with mild hub damping.
		deg := csr.totalDegree(next)
		hubDamp := 1.0 / (1.0 + 0.08*math.Log1p(float64(deg)))
		nextScore := cur.score * 0.72 * dirFactor * math.Min(2.0, float64(weight)) * hubDamp
		if prev, ok := visited[next]; ok {
			if nextScore > prev.GraphScore {
				newPath := make([]uint32, len(cur.path)+1)
				copy(newPath, cur.path)
				newPath[len(cur.path)] = next
				visited[next] = cachedNeighbor{
					NodeID:      next,
					HopDistance: nextHop,
					GraphScore:  nextScore,
					Path:        newPath,
				}
			}
			return true
		}
		if len(visited) >= maxNodes {
			return false
		}
		newPath := make([]uint32, len(cur.path)+1)
		copy(newPath, cur.path)
		newPath[len(cur.path)] = next
		cn := cachedNeighbor{
			NodeID:      next,
			HopDistance: nextHop,
			GraphScore:  nextScore,
			Path:        newPath,
		}
		visited[next] = cn
		nodesVisited++
		if nextHop < maxHops {
			queue = append(queue, qNode{u: next, hop: nextHop, score: nextScore, path: newPath})
		}
		return true
	}

	for head := 0; head < len(queue); head++ {
		cur := queue[head]
		if cur.hop >= maxHops {
			continue
		}
		// Forward CSR edges.
		for ei := csr.FwdOffsets[cur.u]; ei < csr.FwdOffsets[cur.u+1]; ei++ {
			if !tryVisit(cur, csr.FwdTargets[ei], csr.FwdEdgeIdx[ei], csr.FwdEdgeTypes[ei], csr.FwdWeights[ei], 1.0) {
				break
			}
		}
		// Reverse CSR edges (slightly discounted vs forward direction).
		for ei := csr.RevOffsets[cur.u]; ei < csr.RevOffsets[cur.u+1]; ei++ {
			if !tryVisit(cur, csr.RevSources[ei], csr.RevEdgeIdx[ei], csr.RevEdgeTypes[ei], csr.RevWeights[ei], 0.90) {
				break
			}
		}
	}

	out := make([]cachedNeighbor, 0, len(visited))
	for _, cn := range visited {
		out = append(out, cn)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].GraphScore != out[j].GraphScore {
			return out[i].GraphScore > out[j].GraphScore
		}
		return out[i].NodeID < out[j].NodeID
	})
	return out, nodesVisited, edgesTraversed
}

// --- Lexical & Hit Construction Helpers ---

func computeBM25Scores(
	csr *tenantCSR,
	queryTokens []string,
	scopeSet map[string]struct{},
	wantLabels map[string]struct{},
) (map[uint32]float64, int) {
	scores := make(map[uint32]float64)
	if len(queryTokens) == 0 || csr.nodeCount() == 0 {
		return scores, 0
	}
	const k1 = 1.2
	const b = 0.75
	N := float64(csr.nodeCount())
	avgDL := csr.AvgDocLn
	if avgDL <= 0 {
		avgDL = 1.0
	}

	seenTok := make(map[string]bool, len(queryTokens))
	visited := make(map[uint32]struct{})
	for _, tok := range queryTokens {
		if seenTok[tok] {
			continue
		}
		seenTok[tok] = true
		postings := csr.Postings[tok]
		if len(postings) == 0 {
			continue
		}
		df := float64(len(postings))
		idf := math.Log(1.0 + (N-df+0.5)/(df+0.5))
		for _, p := range postings {
			if !csr.isNodeAuthorized(p.NodeID, scopeSet) {
				continue
			}
			if !csr.nodeMatchesLabels(p.NodeID, wantLabels) {
				continue
			}
			visited[p.NodeID] = struct{}{}
			tf := float64(p.TF)
			dl := float64(csr.DocLens[p.NodeID])
			denom := tf + k1*(1.0-b+b*(dl/avgDL))
			if denom > 0 {
				scores[p.NodeID] += idf * (tf * (k1 + 1.0) / denom)
			}
		}
	}
	return scores, len(visited)
}

func positiveCosine(a, b []float32) float64 {
	c := CosineSimilarity(a, b)
	if c < 0 {
		return 0
	}
	return c
}

func pathToSourceIDs(csr *tenantCSR, path []uint32) []string {
	if len(path) == 0 {
		return nil
	}
	out := make([]string, len(path))
	for i, u := range path {
		if int(u) < len(csr.SourceIDs) {
			out[i] = csr.SourceIDs[u]
		}
	}
	return out
}

func buildScoredHit(
	csr *tenantCSR,
	u uint32,
	score, vecScore, lexScore, graphScore float64,
	hop int,
	path []string,
) ScoredHit {
	node := csr.Nodes[u].Clone()
	var pathCopy []string
	if path != nil {
		pathCopy = make([]string, len(path))
		copy(pathCopy, path)
	}
	return ScoredHit{
		SourceID:     node.SourceID,
		Version:      node.Version,
		Kind:         node.Kind,
		Title:        node.Title,
		Text:         node.Text,
		Labels:       node.Labels,
		Score:        round4(score),
		VectorScore:  round4(vecScore),
		LexicalScore: round4(lexScore),
		GraphScore:   round4(graphScore),
		HopDistance:  hop,
		Path:         pathCopy,
		Metadata:     node.Metadata,
	}
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func round4(v float64) float64 {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return 0
	}
	return math.Round(v*10000) / 10000
}
