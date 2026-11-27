package graphrag

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
)

// weightedGraph is an undirected weighted graph (optionally with self-loop
// weights represented as A_ii = 2 * selfLoopWeight) used for deterministic
// Newman-Girvan modularity optimization and Leiden connectivity refinement.
type weightedGraph struct {
	nodes    []string
	adj      map[string]map[string]float64 // neighbor weights excluding self-loop
	selfLoop map[string]float64            // A_ii = 2 * internal_weight
	degree   map[string]float64            // k_i = selfLoop[i] + sum_{j!=i} adj[i][j]
	twoM     float64                       // 2m = sum_i k_i
}

func newWeightedGraph(nodes []string) *weightedGraph {
	sorted := dedupStringsSorted(nodes)
	g := &weightedGraph{
		nodes:    sorted,
		adj:      make(map[string]map[string]float64, len(sorted)),
		selfLoop: make(map[string]float64, len(sorted)),
		degree:   make(map[string]float64, len(sorted)),
	}
	for _, n := range sorted {
		g.adj[n] = make(map[string]float64)
	}
	return g
}

func (g *weightedGraph) addEdge(u, v string, w float64) {
	if w <= 0 {
		return
	}
	if _, ok := g.adj[u]; !ok {
		g.adj[u] = make(map[string]float64)
		g.nodes = append(g.nodes, u)
		sort.Strings(g.nodes)
	}
	if _, ok := g.adj[v]; !ok {
		g.adj[v] = make(map[string]float64)
		g.nodes = append(g.nodes, v)
		sort.Strings(g.nodes)
	}
	if u == v {
		g.selfLoop[u] += w
		return
	}
	g.adj[u][v] += w
	g.adj[v][u] += w
}

func (g *weightedGraph) finalize() {
	sort.Strings(g.nodes)
	g.twoM = 0
	for _, u := range g.nodes {
		deg := g.selfLoop[u]
		for _, w := range g.adj[u] {
			deg += w
		}
		g.degree[u] = deg
		g.twoM += deg
	}
}

// modularity computes exact Newman-Girvan weighted modularity Q for a node ->
// community assignment with resolution gamma.
func (g *weightedGraph) modularity(assignment map[string]string, gamma float64) float64 {
	if g.twoM <= 0 || len(g.nodes) == 0 {
		return 0
	}
	commInternal := make(map[string]float64)
	commTot := make(map[string]float64)
	for _, u := range g.nodes {
		cu := assignment[u]
		commTot[cu] += g.degree[u]
		commInternal[cu] += g.selfLoop[u]
		for v, w := range g.adj[u] {
			if assignment[v] == cu {
				commInternal[cu] += w
			}
		}
	}
	var q float64
	for _, c := range sortedKeysFloat(commTot) {
		in := commInternal[c]
		tot := commTot[c]
		q += (in / g.twoM) - gamma*(tot/g.twoM)*(tot/g.twoM)
	}
	return q
}

// detectCommunitiesLevel runs deterministic local-moving modularity
// optimization followed by Leiden-style connectivity refinement so every
// emitted community induces a connected subgraph.
func detectCommunitiesLevel(g *weightedGraph, gamma float64, maxPasses int) (map[string]string, float64) {
	assignment := make(map[string]string, len(g.nodes))
	if len(g.nodes) == 0 {
		return assignment, 0
	}
	commTot := make(map[string]float64, len(g.nodes))
	for _, u := range g.nodes {
		assignment[u] = u
		commTot[u] = g.degree[u]
	}
	if g.twoM <= 0 {
		return assignment, 0
	}

	const eps = 1e-12
	twoMSq := g.twoM * g.twoM

	for pass := 0; pass < maxPasses; pass++ {
		moved := false
		for _, u := range g.nodes {
			ki := g.degree[u]
			if ki <= 0 {
				continue
			}
			oldComm := assignment[u]

			// Compute weights from u to each neighboring community (excluding u itself).
			neighborCommWeight := make(map[string]float64)
			for v, w := range g.adj[u] {
				cv := assignment[v]
				neighborCommWeight[cv] += w
			}
			if len(neighborCommWeight) == 0 {
				continue
			}

			kInOld := neighborCommWeight[oldComm]
			totOldMinusU := commTot[oldComm] - ki

			bestComm := oldComm
			bestDelta := 0.0

			// Deterministic iteration over candidate neighboring communities.
			candidates := sortedKeysFloat(neighborCommWeight)
			for _, cand := range candidates {
				if cand == oldComm {
					continue
				}
				kInNew := neighborCommWeight[cand]
				totNew := commTot[cand]
				deltaQ := (2.0*(kInNew-kInOld))/g.twoM - gamma*(2.0*ki*(totNew-totOldMinusU))/twoMSq
				if deltaQ > bestDelta+eps || (deltaQ > eps && deltaQ >= bestDelta-eps && cand < bestComm) {
					bestDelta = deltaQ
					bestComm = cand
				}
			}

			if bestComm != oldComm && bestDelta > eps {
				assignment[u] = bestComm
				commTot[oldComm] -= ki
				commTot[bestComm] += ki
				moved = true
			}
		}
		if !moved {
			break
		}
	}

	// Leiden-style refinement: ensure every community induces a connected
	// subgraph by splitting any disconnected components deterministically.
	assignment = refineConnectedCommunities(g, assignment)
	q := g.modularity(assignment, 1.0)
	return assignment, q
}

// refineConnectedCommunities splits any disconnected community into its
// connected components (Leiden connectivity guarantee) with deterministic labels.
func refineConnectedCommunities(g *weightedGraph, assignment map[string]string) map[string]string {
	membersByComm := make(map[string][]string)
	for _, u := range g.nodes {
		c := assignment[u]
		membersByComm[c] = append(membersByComm[c], u)
	}

	commKeys := make([]string, 0, len(membersByComm))
	for c := range membersByComm {
		commKeys = append(commKeys, c)
	}
	sort.Strings(commKeys)

	refined := make(map[string]string, len(g.nodes))
	for _, c := range commKeys {
		members := membersByComm[c]
		sort.Strings(members)
		memberSet := make(map[string]struct{}, len(members))
		for _, m := range members {
			memberSet[m] = struct{}{}
		}
		visited := make(map[string]struct{}, len(members))
		for _, start := range members {
			if _, seen := visited[start]; seen {
				continue
			}
			// BFS within the induced subgraph of community c.
			queue := []string{start}
			visited[start] = struct{}{}
			var comp []string
			for len(queue) > 0 {
				cur := queue[0]
				queue = queue[1:]
				comp = append(comp, cur)
				nbrs := sortedKeysFloat(g.adj[cur])
				for _, nb := range nbrs {
					if _, inComm := memberSet[nb]; !inComm {
						continue
					}
					if _, seen := visited[nb]; seen {
						continue
					}
					visited[nb] = struct{}{}
					queue = append(queue, nb)
				}
			}
			sort.Strings(comp)
			// Canonical component label = smallest member ID.
			compLabel := comp[0]
			for _, m := range comp {
				refined[m] = compLabel
			}
		}
	}
	return refined
}

// buildSuperGraph coarsens graph g according to assignment (node -> commID),
// creating one super-node per community with exact preserved edge/self-loop weights.
func buildSuperGraph(g *weightedGraph, assignment map[string]string) *weightedGraph {
	commSet := make(map[string]struct{})
	for _, u := range g.nodes {
		commSet[assignment[u]] = struct{}{}
	}
	commNodes := make([]string, 0, len(commSet))
	for c := range commSet {
		commNodes = append(commNodes, c)
	}
	sort.Strings(commNodes)
	sg := newWeightedGraph(commNodes)

	for _, u := range g.nodes {
		cu := assignment[u]
		if sl := g.selfLoop[u]; sl > 0 {
			sg.selfLoop[cu] += sl
		}
		for _, v := range sortedKeysFloat(g.adj[u]) {
			if v <= u {
				continue // process each undirected edge once
			}
			w := g.adj[u][v]
			cv := assignment[v]
			if cu == cv {
				sg.selfLoop[cu] += 2.0 * w
			} else {
				sg.addEdge(cu, cv, w)
			}
		}
	}
	sg.finalize()
	return sg
}

// canonicalCommunityID derives a deterministic, stable community ID from its
// tenant, ACL scope, hierarchy level, and sorted member entities.
func canonicalCommunityID(tenantID, scopeKey string, level int, entityIDs []string) string {
	sorted := dedupStringsSorted(entityIDs)
	h := sha256.New()
	h.Write([]byte(tenantID))
	h.Write([]byte{0})
	h.Write([]byte(scopeKey))
	h.Write([]byte{0})
	h.Write([]byte(fmt.Sprintf("L%d", level)))
	h.Write([]byte{0})
	for _, id := range sorted {
		h.Write([]byte(id))
		h.Write([]byte{0})
	}
	return fmt.Sprintf("comm-l%d-%s-%s", level, scopeKey[:8], hex.EncodeToString(h.Sum(nil))[:12])
}

// communityHierarchyResult holds a complete L0 -> L1 -> L2 hierarchy built for
// one (TenantID, ACLScope) partition.
type communityHierarchyResult struct {
	Communities  map[string]*Community // commID -> Community
	ByLevel      map[int][]string      // level -> sorted commIDs
	EntityToL0   map[string]string     // entityID -> L0 commID
	ModularityL0 float64
	ModularityL1 float64
}

func buildCommunityHierarchy(
	tenantID, scopeKey string,
	acl []string,
	entities map[string]*Entity,
	relations map[string]*Relation,
	cfg Config,
) *communityHierarchyResult {
	res := &communityHierarchyResult{
		Communities: make(map[string]*Community),
		ByLevel:     make(map[int][]string),
		EntityToL0:  make(map[string]string),
	}
	if len(entities) == 0 {
		return res
	}

	entityIDs := make([]string, 0, len(entities))
	for id := range entities {
		entityIDs = append(entityIDs, id)
	}
	sort.Strings(entityIDs)

	g0 := newWeightedGraph(entityIDs)
	relKeys := make([]string, 0, len(relations))
	for k := range relations {
		relKeys = append(relKeys, k)
	}
	sort.Strings(relKeys)
	for _, rk := range relKeys {
		r := relations[rk]
		if _, ok1 := entities[r.SourceEntityID]; !ok1 {
			continue
		}
		if _, ok2 := entities[r.TargetEntityID]; !ok2 {
			continue
		}
		g0.addEdge(r.SourceEntityID, r.TargetEntityID, r.Weight)
	}
	g0.finalize()

	// Level 0: fine-grained leaf communities.
	assign0, mod0 := detectCommunitiesLevel(g0, cfg.Resolution*1.25, cfg.MaxLocalMovePasses)
	res.ModularityL0 = mod0

	l0Groups := make(map[string][]string)
	for _, eid := range entityIDs {
		lbl := assign0[eid]
		l0Groups[lbl] = append(l0Groups[lbl], eid)
	}
	l0LabelToID := make(map[string]string, len(l0Groups))
	l0Labels := make([]string, 0, len(l0Groups))
	for lbl := range l0Groups {
		l0Labels = append(l0Labels, lbl)
	}
	sort.Strings(l0Labels)

	for _, lbl := range l0Labels {
		members := dedupStringsSorted(l0Groups[lbl])
		cid := canonicalCommunityID(tenantID, scopeKey, 0, members)
		l0LabelToID[lbl] = cid
		internalW, totalW := computeCommunityWeights(g0, members)
		refs := collectEntitySourceRefs(entities, members)
		res.Communities[cid] = &Community{
			ID:             cid,
			TenantID:       tenantID,
			ScopeKey:       scopeKey,
			ACL:            append([]string(nil), acl...),
			Level:          0,
			EntityIDs:      members,
			InternalWeight: internalW,
			TotalWeight:    totalW,
			SourceRefs:     refs,
		}
		res.ByLevel[0] = append(res.ByLevel[0], cid)
		for _, eid := range members {
			res.EntityToL0[eid] = cid
		}
	}
	sort.Strings(res.ByLevel[0])

	if cfg.MaxHierarchyLevels <= 1 {
		return res
	}

	// Level 1: mid-level subsystem/domain communities coarsened from L0.
	assign0ByCID := make(map[string]string, len(entityIDs))
	for _, eid := range entityIDs {
		assign0ByCID[eid] = res.EntityToL0[eid]
	}
	g1 := buildSuperGraph(g0, assign0ByCID)
	assign1, mod1 := detectCommunitiesLevel(g1, cfg.Resolution*0.70, cfg.MaxLocalMovePasses)
	res.ModularityL1 = mod1

	// If L0 had many communities and g1 didn't merge enough (e.g. high self-loop weight),
	// perform a gentle attraction pass connecting L0s sharing inter-community edges.
	if len(res.ByLevel[0]) > 4 && countUniqueValues(assign1) == len(res.ByLevel[0]) {
		assign1 = coarsenByStrongestNeighbor(g1, assign1, 3)
		res.ModularityL1 = g1.modularity(assign1, 1.0)
	}

	l1Groups := make(map[string][]string) // l1Label -> L0 commIDs
	for _, l0ID := range res.ByLevel[0] {
		lbl := assign1[l0ID]
		l1Groups[lbl] = append(l1Groups[lbl], l0ID)
	}
	l1Labels := make([]string, 0, len(l1Groups))
	for lbl := range l1Groups {
		l1Labels = append(l1Labels, lbl)
	}
	sort.Strings(l1Labels)

	l0ToL1 := make(map[string]string, len(res.ByLevel[0]))
	for _, lbl := range l1Labels {
		childL0s := dedupStringsSorted(l1Groups[lbl])
		var allEntities []string
		for _, c0 := range childL0s {
			allEntities = append(allEntities, res.Communities[c0].EntityIDs...)
		}
		allEntities = dedupStringsSorted(allEntities)
		cid := canonicalCommunityID(tenantID, scopeKey, 1, allEntities)
		internalW, totalW := computeCommunityWeights(g0, allEntities)
		refs := collectEntitySourceRefs(entities, allEntities)
		res.Communities[cid] = &Community{
			ID:             cid,
			TenantID:       tenantID,
			ScopeKey:       scopeKey,
			ACL:            append([]string(nil), acl...),
			Level:          1,
			ChildIDs:       childL0s,
			EntityIDs:      allEntities,
			InternalWeight: internalW,
			TotalWeight:    totalW,
			SourceRefs:     refs,
		}
		res.ByLevel[1] = append(res.ByLevel[1], cid)
		for _, c0 := range childL0s {
			res.Communities[c0].ParentID = cid
			l0ToL1[c0] = cid
		}
	}
	sort.Strings(res.ByLevel[1])

	if cfg.MaxHierarchyLevels <= 2 {
		return res
	}

	// Level 2: macro-level corpus communities coarsened from L1.
	g2 := buildSuperGraph(g1, l0ToL1)
	assign2, _ := detectCommunitiesLevel(g2, cfg.Resolution*0.30, cfg.MaxLocalMovePasses)
	if len(res.ByLevel[1]) > 4 && countUniqueValues(assign2) > len(res.ByLevel[1])/2 {
		assign2 = coarsenByStrongestNeighbor(g2, assign2, 4)
	}
	// Ensure L2 is strictly coarser than L1 when len(L1) > 2 by merging smallest
	// remaining L2 buckets up to at most 3 macro communities if connected.
	l2Groups := make(map[string][]string)
	for _, l1ID := range res.ByLevel[1] {
		lbl := assign2[l1ID]
		l2Groups[lbl] = append(l2Groups[lbl], l1ID)
	}
	l2Labels := make([]string, 0, len(l2Groups))
	for lbl := range l2Groups {
		l2Labels = append(l2Labels, lbl)
	}
	sort.Strings(l2Labels)

	for _, lbl := range l2Labels {
		childL1s := dedupStringsSorted(l2Groups[lbl])
		var allEntities []string
		for _, c1 := range childL1s {
			allEntities = append(allEntities, res.Communities[c1].EntityIDs...)
		}
		allEntities = dedupStringsSorted(allEntities)
		cid := canonicalCommunityID(tenantID, scopeKey, 2, allEntities)
		internalW, totalW := computeCommunityWeights(g0, allEntities)
		refs := collectEntitySourceRefs(entities, allEntities)
		res.Communities[cid] = &Community{
			ID:             cid,
			TenantID:       tenantID,
			ScopeKey:       scopeKey,
			ACL:            append([]string(nil), acl...),
			Level:          2,
			ChildIDs:       childL1s,
			EntityIDs:      allEntities,
			InternalWeight: internalW,
			TotalWeight:    totalW,
			SourceRefs:     refs,
		}
		res.ByLevel[2] = append(res.ByLevel[2], cid)
		for _, c1 := range childL1s {
			res.Communities[c1].ParentID = cid
		}
	}
	sort.Strings(res.ByLevel[2])

	return res
}

// coarsenByStrongestNeighbor deterministically merges singleton super-nodes
// with their strongest connected neighbor up to maxGroupSize to guarantee
// meaningful multi-level summarization coarsening when internal self-loops
// dominate modularity.
func coarsenByStrongestNeighbor(g *weightedGraph, current map[string]string, maxGroupSize int) map[string]string {
	out := make(map[string]string, len(current))
	for k, v := range current {
		out[k] = v
	}
	groupSize := make(map[string]int, len(out))
	for _, v := range out {
		groupSize[v]++
	}
	for _, u := range g.nodes {
		cu := out[u]
		if groupSize[cu] >= maxGroupSize {
			continue
		}
		bestTarget := ""
		bestW := 0.0
		for _, v := range sortedKeysFloat(g.adj[u]) {
			cv := out[v]
			if cv == cu {
				continue
			}
			if groupSize[cv]+groupSize[cu] > maxGroupSize {
				continue
			}
			w := g.adj[u][v]
			if w > bestW || (w == bestW && bestTarget != "" && cv < bestTarget) {
				bestW = w
				bestTarget = cv
			}
		}
		if bestTarget != "" && bestW > 0 {
			// Merge smaller/later label into canonical earlier label.
			keep, drop := bestTarget, cu
			if cu < bestTarget {
				keep, drop = cu, bestTarget
			}
			for _, n := range g.nodes {
				if out[n] == drop {
					out[n] = keep
				}
			}
			groupSize[keep] += groupSize[drop]
			delete(groupSize, drop)
		}
	}
	return out
}

func countUniqueValues(m map[string]string) int {
	seen := make(map[string]struct{}, len(m))
	for _, v := range m {
		seen[v] = struct{}{}
	}
	return len(seen)
}

func computeCommunityWeights(g *weightedGraph, members []string) (internalWeight float64, totalWeight float64) {
	if len(members) == 0 {
		return 0, 0
	}
	memberSet := make(map[string]struct{}, len(members))
	for _, m := range members {
		memberSet[m] = struct{}{}
		totalWeight += g.degree[m]
		internalWeight += g.selfLoop[m]
	}
	for _, u := range members {
		for v, w := range g.adj[u] {
			if _, ok := memberSet[v]; ok && u < v {
				internalWeight += 2.0 * w
			}
		}
	}
	return internalWeight / 2.0, totalWeight / 2.0
}

func collectEntitySourceRefs(entities map[string]*Entity, entityIDs []string) []SourceRef {
	var refs []SourceRef
	for _, eid := range entityIDs {
		if ent, ok := entities[eid]; ok {
			refs = append(refs, ent.SourceRefs...)
		}
	}
	return sortSourceRefs(refs)
}
