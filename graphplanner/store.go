/*
 * SPDX-License-Identifier: Apache-2.0
 */

package graphplanner

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
)

type canonicalKey struct {
	TenantID string
	ID       string
}

type lexicalPosting struct {
	NodeID uint32
	TF     uint16
}

// tenantCSR holds the compact derived CSR adjacency, vector, and lexical
// indexes strictly partitioned for a single tenant.
type tenantCSR struct {
	TenantID string

	// Integer-interned node identity tables (0 .. N-1).
	SourceIDs []string
	IDToInt   map[string]uint32
	Nodes     []CanonicalNode // indexed by uint32 intern ID (canonical snapshot)
	NodeACL   []map[string]struct{}
	NodeLabel []map[string]struct{}

	// Integer-interned edge-type tables (0 .. T-1).
	EdgeTypeNames []string
	EdgeTypeToInt map[string]uint16

	// Edge ACL metadata indexed by uint32 edge index (0 .. E-1).
	EdgeACLs []map[string]struct{}
	EdgeIDs  []string

	// Forward CSR (u -> v).
	FwdOffsets   []uint32  // len = N + 1
	FwdTargets   []uint32  // len = E
	FwdEdgeTypes []uint16  // len = E
	FwdWeights   []float32 // len = E
	FwdEdgeIdx   []uint32  // len = E

	// Reverse CSR (v -> u, storing incoming edges).
	RevOffsets   []uint32  // len = N + 1
	RevSources   []uint32  // len = E
	RevEdgeTypes []uint16  // len = E
	RevWeights   []float32 // len = E
	RevEdgeIdx   []uint32  // len = E

	// Flat contiguous L2-normalized dense embeddings (len = N * EmbeddingDim).
	Vectors []float32

	// Derived BM25 lexical inverted index.
	Postings map[string][]lexicalPosting
	DocLens  []uint32
	AvgDocLn float64

	// Exact lowercase title/identifier lookup to resolve entity anchors in queries.
	TitleToNodes map[string][]uint32

	// Precomputed cardinality and degree statistics for O(1) cost planning.
	LabelCounts    map[string]int
	EdgeTypeCounts map[uint16]int
	MeanDegree     float64
	P90Degree      float64
	MaxDegree      int

	// Measured byte footprints for storage telemetry.
	CSRForwardBytes   int64
	CSRReverseBytes   int64
	InternTableBytes  int64
	VectorIndexBytes  int64
	LexicalIndexBytes int64
}

func (t *tenantCSR) nodeCount() int {
	return len(t.SourceIDs)
}

func (t *tenantCSR) edgeCount() int {
	return len(t.FwdTargets)
}

func (t *tenantCSR) totalDegree(u uint32) int {
	if int(u)+1 >= len(t.FwdOffsets) {
		return 0
	}
	outDeg := int(t.FwdOffsets[u+1] - t.FwdOffsets[u])
	inDeg := int(t.RevOffsets[u+1] - t.RevOffsets[u])
	return outDeg + inDeg
}

func (t *tenantCSR) vectorSlice(u uint32) []float32 {
	start := int(u) * EmbeddingDim
	end := start + EmbeddingDim
	if start < 0 || end > len(t.Vectors) {
		return nil
	}
	return t.Vectors[start:end]
}

func (t *tenantCSR) isNodeAuthorized(u uint32, scopeSet map[string]struct{}) bool {
	if int(u) >= len(t.NodeACL) {
		return false
	}
	req := t.NodeACL[u]
	if len(req) == 0 {
		return true
	}
	for s := range req {
		if _, ok := scopeSet[s]; ok {
			return true
		}
	}
	return false
}

func (t *tenantCSR) isEdgeAuthorized(edgeIdx uint32, scopeSet map[string]struct{}) bool {
	if int(edgeIdx) >= len(t.EdgeACLs) {
		return false
	}
	req := t.EdgeACLs[edgeIdx]
	if len(req) == 0 {
		return true
	}
	for s := range req {
		if _, ok := scopeSet[s]; ok {
			return true
		}
	}
	return false
}

func (t *tenantCSR) nodeMatchesLabels(u uint32, wantLabels map[string]struct{}) bool {
	if len(wantLabels) == 0 {
		return true
	}
	if int(u) >= len(t.NodeLabel) {
		return false
	}
	nl := t.NodeLabel[u]
	for w := range wantLabels {
		if _, ok := nl[w]; ok {
			return true
		}
	}
	return false
}

// StoreConfig configures the embedded graph store and cost-based planner.
type StoreConfig struct {
	CacheCapacity int
	Planner       PlannerConfig
}

// DefaultStoreConfig returns sensible production defaults for a single-process
// embedded graph store.
func DefaultStoreConfig() StoreConfig {
	return StoreConfig{
		CacheCapacity: 512,
		Planner:       DefaultPlannerConfig(),
	}
}

// GraphStore is a concurrency-safe embedded native graph store pairing
// canonical versioned records with compact CSR adjacency indexes, vector and
// lexical indexes, an ACL-isolated LRU neighborhood cache, and a cost-based
// query planner.
type GraphStore struct {
	mu sync.RWMutex

	cfg StoreConfig

	// Canonical source-of-truth records (untouched by DropDerivedIndexes).
	nodes       map[canonicalKey]CanonicalNode
	nodeHistory map[canonicalKey][]CanonicalNode
	edges       map[canonicalKey]CanonicalEdge
	edgeHistory map[canonicalKey][]CanonicalEdge

	// Derived per-tenant compact CSR + vector + lexical indexes.
	derivedBuilt bool
	tenants      map[string]*tenantCSR
	indexDigest  string

	// Bounded ACL-scoped LRU neighborhood cache.
	cache *neighborhoodCache
}

// NewGraphStore initializes an empty embedded GraphStore.
func NewGraphStore(cfg StoreConfig) *GraphStore {
	if cfg.CacheCapacity <= 0 {
		cfg.CacheCapacity = DefaultStoreConfig().CacheCapacity
	}
	if cfg.Planner.DefaultLatencyBudgetMs <= 0 {
		cfg.Planner = DefaultPlannerConfig()
	}
	return &GraphStore{
		cfg:         cfg,
		nodes:       make(map[canonicalKey]CanonicalNode),
		nodeHistory: make(map[canonicalKey][]CanonicalNode),
		edges:       make(map[canonicalKey]CanonicalEdge),
		edgeHistory: make(map[canonicalKey][]CanonicalEdge),
		tenants:     make(map[string]*tenantCSR),
		cache:       newNeighborhoodCache(cfg.CacheCapacity),
	}
}

func validateNode(n CanonicalNode) error {
	if strings.TrimSpace(n.TenantID) == "" || strings.TrimSpace(n.SourceID) == "" {
		return fmt.Errorf("%w: tenant_id and source_id are required", ErrInvalidRecord)
	}
	if n.Version == 0 {
		return fmt.Errorf("%w: node %s version must be >= 1", ErrInvalidRecord, n.SourceID)
	}
	if len(n.Embedding) != 0 {
		if len(n.Embedding) != EmbeddingDim {
			return fmt.Errorf("%w: node %s embedding dim %d != %d", ErrInvalidRecord, n.SourceID, len(n.Embedding), EmbeddingDim)
		}
		for i, v := range n.Embedding {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return fmt.Errorf("%w: node %s embedding[%d] is non-finite", ErrInvalidRecord, n.SourceID, i)
			}
		}
	}
	return nil
}

func validateEdge(e CanonicalEdge) error {
	if strings.TrimSpace(e.TenantID) == "" || strings.TrimSpace(e.EdgeID) == "" {
		return fmt.Errorf("%w: tenant_id and edge_id are required", ErrInvalidRecord)
	}
	if strings.TrimSpace(e.FromSourceID) == "" || strings.TrimSpace(e.ToSourceID) == "" {
		return fmt.Errorf("%w: edge %s requires from_source_id and to_source_id", ErrInvalidRecord, e.EdgeID)
	}
	if strings.TrimSpace(e.Type) == "" {
		return fmt.Errorf("%w: edge %s requires non-empty type", ErrInvalidRecord, e.EdgeID)
	}
	if e.Version == 0 {
		return fmt.Errorf("%w: edge %s version must be >= 1", ErrInvalidRecord, e.EdgeID)
	}
	if math.IsNaN(float64(e.Weight)) || math.IsInf(float64(e.Weight), 0) || e.Weight < 0 {
		return fmt.Errorf("%w: edge %s weight must be finite and >= 0", ErrInvalidRecord, e.EdgeID)
	}
	return nil
}

// UpsertNode inserts or updates a canonical node record.
// Updating canonical state immediately invalidates derived indexes and the
// tenant's LRU neighborhood cache so revoked ACLs or modified records cannot
// leak stale state. Backfilling a strictly older version records history
// without mutating canonical state.
func (s *GraphStore) UpsertNode(node CanonicalNode) error {
	if err := validateNode(node); err != nil {
		return err
	}
	cloned := node.Clone()
	if cloned.Kind == "" {
		cloned.Kind = KindDocument
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	key := canonicalKey{TenantID: cloned.TenantID, ID: cloned.SourceID}
	s.nodeHistory[key] = append(s.nodeHistory[key], cloned.Clone())
	cur, exists := s.nodes[key]
	if !exists || cloned.Version >= cur.Version {
		s.nodes[key] = cloned
		s.derivedBuilt = false
		s.cache.InvalidateTenant(cloned.TenantID)
	}
	return nil
}

// UpsertEdge inserts or updates a canonical directed edge record.
func (s *GraphStore) UpsertEdge(edge CanonicalEdge) error {
	if err := validateEdge(edge); err != nil {
		return err
	}
	cloned := edge.Clone()
	if cloned.Weight == 0 {
		cloned.Weight = 1.0
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	key := canonicalKey{TenantID: cloned.TenantID, ID: cloned.EdgeID}
	s.edgeHistory[key] = append(s.edgeHistory[key], cloned.Clone())
	cur, exists := s.edges[key]
	if !exists || cloned.Version >= cur.Version {
		s.edges[key] = cloned
		s.derivedBuilt = false
		s.cache.InvalidateTenant(cloned.TenantID)
	}
	return nil
}

// DeleteNode removes a canonical node and immediately invalidates derived
// indexes and the tenant's neighborhood cache.
func (s *GraphStore) DeleteNode(tenantID, sourceID string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	key := canonicalKey{TenantID: tenantID, ID: sourceID}
	if _, ok := s.nodes[key]; !ok {
		return false
	}
	delete(s.nodes, key)
	s.derivedBuilt = false
	s.cache.InvalidateTenant(tenantID)
	return true
}

// DeleteEdge removes a canonical edge and immediately invalidates derived
// indexes and the tenant's neighborhood cache.
func (s *GraphStore) DeleteEdge(tenantID, edgeID string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	key := canonicalKey{TenantID: tenantID, ID: edgeID}
	if _, ok := s.edges[key]; !ok {
		return false
	}
	delete(s.edges, key)
	s.derivedBuilt = false
	s.cache.InvalidateTenant(tenantID)
	return true
}

// UpsertBatchAndRebuild loads canonical nodes and edges and rebuilds all
// derived indexes atomically.
func (s *GraphStore) UpsertBatchAndRebuild(nodes []CanonicalNode, edges []CanonicalEdge) error {
	for _, n := range nodes {
		if err := s.UpsertNode(n); err != nil {
			return err
		}
	}
	for _, e := range edges {
		if err := s.UpsertEdge(e); err != nil {
			return err
		}
	}
	return s.RebuildIndex()
}

// GetNode returns a deep copy of the authorized canonical node.
func (s *GraphStore) GetNode(access AccessContext, sourceID string) (CanonicalNode, bool) {
	if strings.TrimSpace(access.TenantID) == "" {
		return CanonicalNode{}, false
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	n, ok := s.nodes[canonicalKey{TenantID: access.TenantID, ID: sourceID}]
	if !ok {
		return CanonicalNode{}, false
	}
	if !aclAllows(n.ACL, scopeSetFromSlice(access.Scopes)) {
		return CanonicalNode{}, false
	}
	return n.Clone(), true
}

// GetEdge returns a deep copy of the authorized canonical edge.
func (s *GraphStore) GetEdge(access AccessContext, edgeID string) (CanonicalEdge, bool) {
	if strings.TrimSpace(access.TenantID) == "" {
		return CanonicalEdge{}, false
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	e, ok := s.edges[canonicalKey{TenantID: access.TenantID, ID: edgeID}]
	if !ok {
		return CanonicalEdge{}, false
	}
	if !aclAllows(e.ACL, scopeSetFromSlice(access.Scopes)) {
		return CanonicalEdge{}, false
	}
	return e.Clone(), true
}

// NodeVersionCount returns the number of historical versions recorded for a node.
func (s *GraphStore) NodeVersionCount(tenantID, sourceID string) int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.nodeHistory[canonicalKey{TenantID: tenantID, ID: sourceID}])
}

// DropDerivedIndexes wipes all CSR adjacency arrays, vector/lexical indexes,
// and cached neighborhoods without modifying any canonical records.
func (s *GraphStore) DropDerivedIndexes() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.derivedBuilt = false
	s.tenants = make(map[string]*tenantCSR)
	s.indexDigest = ""
	s.cache.Clear()
}

// HasDerivedIndexes reports whether derived indexes are built and up-to-date.
func (s *GraphStore) HasDerivedIndexes() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.derivedBuilt
}

// RebuildIndex deterministically constructs per-tenant compact CSR forward and
// reverse adjacency indexes, flat vector tables, BM25 lexical postings, and
// planner statistics from canonical records.
func (s *GraphStore) RebuildIndex() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	nodesByTenant := make(map[string][]CanonicalNode)
	edgesByTenant := make(map[string][]CanonicalEdge)
	tenantSet := make(map[string]struct{})

	for _, n := range s.nodes {
		nodesByTenant[n.TenantID] = append(nodesByTenant[n.TenantID], n.Clone())
		tenantSet[n.TenantID] = struct{}{}
	}
	for _, e := range s.edges {
		edgesByTenant[e.TenantID] = append(edgesByTenant[e.TenantID], e.Clone())
		tenantSet[e.TenantID] = struct{}{}
	}

	tenantIDs := make([]string, 0, len(tenantSet))
	for t := range tenantSet {
		tenantIDs = append(tenantIDs, t)
	}
	sort.Strings(tenantIDs)

	built := make(map[string]*tenantCSR, len(tenantIDs))
	digestHasher := sha256.New()

	for _, tid := range tenantIDs {
		tNodes := nodesByTenant[tid]
		sort.Slice(tNodes, func(i, j int) bool {
			return tNodes[i].SourceID < tNodes[j].SourceID
		})
		if len(tNodes) > math.MaxUint32-1 {
			return fmt.Errorf("%w: tenant %s exceeds uint32 node limit", ErrCapacityExceeded, tid)
		}

		tEdges := edgesByTenant[tid]
		sort.Slice(tEdges, func(i, j int) bool {
			if tEdges[i].FromSourceID != tEdges[j].FromSourceID {
				return tEdges[i].FromSourceID < tEdges[j].FromSourceID
			}
			if tEdges[i].ToSourceID != tEdges[j].ToSourceID {
				return tEdges[i].ToSourceID < tEdges[j].ToSourceID
			}
			if tEdges[i].Type != tEdges[j].Type {
				return tEdges[i].Type < tEdges[j].Type
			}
			return tEdges[i].EdgeID < tEdges[j].EdgeID
		})

		csr, err := buildTenantCSR(tid, tNodes, tEdges, digestHasher)
		if err != nil {
			return err
		}
		built[tid] = csr
	}

	s.tenants = built
	s.indexDigest = hex.EncodeToString(digestHasher.Sum(nil))[:16]
	s.derivedBuilt = true
	s.cache.Clear()
	return nil
}

func buildTenantCSR(tid string, tNodes []CanonicalNode, tEdges []CanonicalEdge, digestHasher interface{ Write([]byte) (int, error) }) (*tenantCSR, error) {
	nCount := len(tNodes)
	if nCount > math.MaxInt/EmbeddingDim {
		return nil, fmt.Errorf("%w: tenant %s vector buffer size overflow", ErrCapacityExceeded, tid)
	}
	csr := &tenantCSR{
		TenantID:       tid,
		SourceIDs:      make([]string, nCount),
		IDToInt:        make(map[string]uint32, nCount),
		Nodes:          make([]CanonicalNode, nCount),
		NodeACL:        make([]map[string]struct{}, nCount),
		NodeLabel:      make([]map[string]struct{}, nCount),
		EdgeTypeToInt:  make(map[string]uint16),
		Vectors:        make([]float32, nCount*EmbeddingDim),
		Postings:       make(map[string][]lexicalPosting),
		DocLens:        make([]uint32, nCount),
		TitleToNodes:   make(map[string][]uint32),
		LabelCounts:    make(map[string]int),
		EdgeTypeCounts: make(map[uint16]int),
	}

	var internBytes int64
	var totalDocLen uint64

	fmt.Fprintf(digestHasher, "T:%d:%s\n", len(tid), tid)

	for i, node := range tNodes {
		u := uint32(i)
		csr.SourceIDs[i] = node.SourceID
		csr.IDToInt[node.SourceID] = u
		csr.Nodes[i] = node
		internBytes += int64(len(node.SourceID) + 24)

		if len(node.ACL) > 0 {
			aclMap := make(map[string]struct{}, len(node.ACL))
			for _, sc := range node.ACL {
				if sc != "" {
					aclMap[sc] = struct{}{}
				}
			}
			csr.NodeACL[i] = aclMap
		}
		if len(node.Labels) > 0 {
			lblMap := make(map[string]struct{}, len(node.Labels))
			for _, lb := range node.Labels {
				if lb != "" {
					lblMap[lb] = struct{}{}
				}
			}
			for lb := range lblMap {
				csr.LabelCounts[lb]++
			}
			csr.NodeLabel[i] = lblMap
		}

		// Register normalized title and sourceID for entity anchor resolution.
		if titleKey := strings.ToLower(strings.TrimSpace(node.Title)); titleKey != "" {
			csr.TitleToNodes[titleKey] = append(csr.TitleToNodes[titleKey], u)
		}
		idKey := strings.ToLower(strings.TrimSpace(node.SourceID))
		if idKey != "" {
			csr.TitleToNodes[idKey] = append(csr.TitleToNodes[idKey], u)
		}

		// Dense vector embedding (flat contiguous storage).
		var vec []float32
		if len(node.Embedding) == EmbeddingDim {
			vec = make([]float32, EmbeddingDim)
			copy(vec, node.Embedding)
			NormalizeVector(vec)
		} else {
			combined := node.Title + " " + strings.Join(node.Labels, " ") + " " + node.Text
			vec = ComputeEmbedding(combined)
		}
		copy(csr.Vectors[i*EmbeddingDim:(i+1)*EmbeddingDim], vec)

		// Lexical inverted index postings (Title weighted 2x + Labels + Text).
		toks := Tokenize(node.Title + " " + node.Title + " " + strings.Join(node.Labels, " ") + " " + node.Text)
		if len(toks) > math.MaxUint32 {
			toks = toks[:math.MaxUint32]
		}
		csr.DocLens[i] = uint32(len(toks))
		totalDocLen += uint64(len(toks))
		tfMap := make(map[string]uint16, len(toks))
		for _, tok := range toks {
			if tfMap[tok] < math.MaxUint16 {
				tfMap[tok]++
			}
		}
		for tok, tf := range tfMap {
			csr.Postings[tok] = append(csr.Postings[tok], lexicalPosting{NodeID: u, TF: tf})
		}

		// Hash deterministic node digest.
		fmt.Fprintf(digestHasher, "N:%d:%s:%d:%s\n", u, node.SourceID, node.Version, canonicalScopeFingerprint(node.ACL))
	}

	if nCount > 0 {
		csr.AvgDocLn = float64(totalDocLen) / float64(nCount)
		if csr.AvgDocLn < 1.0 {
			csr.AvgDocLn = 1.0
		}
	}

	// Filter edges whose endpoints both exist in this tenant and intern edge types.
	type resolvedEdge struct {
		from   uint32
		to     uint32
		etype  uint16
		weight float32
		edgeID string
		acl    []string
	}
	resolved := make([]resolvedEdge, 0, len(tEdges))
	for _, e := range tEdges {
		u, okU := csr.IDToInt[e.FromSourceID]
		v, okV := csr.IDToInt[e.ToSourceID]
		if !okU || !okV {
			continue
		}
		etID, ok := csr.EdgeTypeToInt[e.Type]
		if !ok {
			if len(csr.EdgeTypeNames) >= math.MaxUint16 {
				return nil, fmt.Errorf("%w: tenant %s exceeds uint16 edge type limit", ErrCapacityExceeded, tid)
			}
			etID = uint16(len(csr.EdgeTypeNames))
			csr.EdgeTypeNames = append(csr.EdgeTypeNames, e.Type)
			csr.EdgeTypeToInt[e.Type] = etID
			internBytes += int64(len(e.Type) + 18)
		}
		resolved = append(resolved, resolvedEdge{
			from:   u,
			to:     v,
			etype:  etID,
			weight: e.Weight,
			edgeID: e.EdgeID,
			acl:    e.ACL,
		})
	}
	if len(resolved) > math.MaxUint32-1 {
		return nil, fmt.Errorf("%w: tenant %s exceeds uint32 edge limit", ErrCapacityExceeded, tid)
	}

	eCount := len(resolved)
	csr.EdgeACLs = make([]map[string]struct{}, eCount)
	csr.EdgeIDs = make([]string, eCount)
	csr.FwdOffsets = make([]uint32, nCount+1)
	csr.RevOffsets = make([]uint32, nCount+1)

	for i, re := range resolved {
		csr.EdgeIDs[i] = re.edgeID
		if len(re.acl) > 0 {
			am := make(map[string]struct{}, len(re.acl))
			for _, sc := range re.acl {
				if sc != "" {
					am[sc] = struct{}{}
				}
			}
			csr.EdgeACLs[i] = am
		}
		csr.FwdOffsets[re.from+1]++
		csr.RevOffsets[re.to+1]++
		csr.EdgeTypeCounts[re.etype]++
	}

	// Prefix sum for CSR row offsets.
	for i := 0; i < nCount; i++ {
		csr.FwdOffsets[i+1] += csr.FwdOffsets[i]
		csr.RevOffsets[i+1] += csr.RevOffsets[i]
	}

	csr.FwdTargets = make([]uint32, eCount)
	csr.FwdEdgeTypes = make([]uint16, eCount)
	csr.FwdWeights = make([]float32, eCount)
	csr.FwdEdgeIdx = make([]uint32, eCount)

	csr.RevSources = make([]uint32, eCount)
	csr.RevEdgeTypes = make([]uint16, eCount)
	csr.RevWeights = make([]float32, eCount)
	csr.RevEdgeIdx = make([]uint32, eCount)

	fwdCursor := make([]uint32, nCount)
	revCursor := make([]uint32, nCount)
	copy(fwdCursor, csr.FwdOffsets[:nCount])
	copy(revCursor, csr.RevOffsets[:nCount])

	var buf [16]byte
	for idx, re := range resolved {
		eIdx := uint32(idx)
		fp := fwdCursor[re.from]
		fwdCursor[re.from]++
		csr.FwdTargets[fp] = re.to
		csr.FwdEdgeTypes[fp] = re.etype
		csr.FwdWeights[fp] = re.weight
		csr.FwdEdgeIdx[fp] = eIdx

		rp := revCursor[re.to]
		revCursor[re.to]++
		csr.RevSources[rp] = re.from
		csr.RevEdgeTypes[rp] = re.etype
		csr.RevWeights[rp] = re.weight
		csr.RevEdgeIdx[rp] = eIdx

		binary.LittleEndian.PutUint32(buf[0:4], re.from)
		binary.LittleEndian.PutUint32(buf[4:8], re.to)
		binary.LittleEndian.PutUint16(buf[8:10], re.etype)
		binary.LittleEndian.PutUint32(buf[10:14], math.Float32bits(re.weight))
		_, _ = digestHasher.Write(buf[:14])
	}

	// Degree distribution statistics.
	if nCount > 0 {
		degs := make([]int, nCount)
		sumDeg := 0
		maxDeg := 0
		for u := 0; u < nCount; u++ {
			d := csr.totalDegree(uint32(u))
			degs[u] = d
			sumDeg += d
			if d > maxDeg {
				maxDeg = d
			}
		}
		sort.Ints(degs)
		csr.MeanDegree = float64(sumDeg) / float64(nCount)
		p90Idx := int(math.Floor(0.90 * float64(nCount-1)))
		csr.P90Degree = float64(degs[p90Idx])
		csr.MaxDegree = maxDeg
	}

	// Exact compact byte accounting for CSR + vectors + lexical index.
	// Forward CSR: offsets(4*(N+1)) + targets(4*E) + edgeTypes(2*E) + weights(4*E) + edgeIdx(4*E)
	csr.CSRForwardBytes = int64(len(csr.FwdOffsets))*4 + int64(eCount)*(4+2+4+4)
	// Reverse CSR: offsets(4*(N+1)) + sources(4*E) + edgeTypes(2*E) + weights(4*E) + edgeIdx(4*E)
	csr.CSRReverseBytes = int64(len(csr.RevOffsets))*4 + int64(eCount)*(4+2+4+4)
	csr.InternTableBytes = internBytes
	csr.VectorIndexBytes = int64(len(csr.Vectors)) * 4
	var lexBytes int64
	for tok, plist := range csr.Postings {
		lexBytes += int64(len(tok)+24) + int64(len(plist))*6
	}
	lexBytes += int64(len(csr.DocLens)) * 4
	csr.LexicalIndexBytes = lexBytes

	return csr, nil
}

// StorageStats reports compact embedded CSR memory footprint and cache stats.
func (s *GraphStore) StorageStats() StorageStats {
	s.mu.RLock()
	defer s.mu.RUnlock()

	entries, capSize, hits, misses, evictions := s.cache.Stats()
	hitRate := 0.0
	if hits+misses > 0 {
		hitRate = float64(hits) / float64(hits+misses)
	}

	st := StorageStats{
		CanonicalNodes:    len(s.nodes),
		CanonicalEdges:    len(s.edges),
		CacheEntries:      entries,
		CacheCapacity:     capSize,
		CacheHits:         hits,
		CacheMisses:       misses,
		CacheEvictions:    evictions,
		CacheHitRate:      hitRate,
		IndexDigestSHA256: s.indexDigest,
		DerivedBuilt:      s.derivedBuilt,
	}

	for _, t := range s.tenants {
		st.IndexedNodes += t.nodeCount()
		st.IndexedEdges += t.edgeCount()
		st.CSRForwardBytes += t.CSRForwardBytes
		st.CSRReverseBytes += t.CSRReverseBytes
		st.InternTableBytes += t.InternTableBytes
		st.VectorIndexBytes += t.VectorIndexBytes
		st.LexicalIndexBytes += t.LexicalIndexBytes
	}
	// TotalGraphBytes reflects the compact native CSR graph representation
	// (forward CSR + reverse CSR + integer intern tables).
	st.TotalGraphBytes = st.CSRForwardBytes + st.CSRReverseBytes + st.InternTableBytes
	if st.IndexedNodes > 0 {
		// Node-proportional footprint: forward/reverse CSR offsets (8B/node) + intern table per node.
		nodeFootprint := int64(st.IndexedNodes)*8 + st.InternTableBytes
		st.BytesPerNode = float64(nodeFootprint) / float64(st.IndexedNodes)
	}
	if st.IndexedEdges > 0 {
		// Edge-proportional footprint: forward + reverse CSR edge arrays (28 bytes/edge exact).
		edgeFootprint := int64(st.IndexedEdges) * 28
		st.BytesPerEdge = float64(edgeFootprint) / float64(st.IndexedEdges)
	}
	return st
}

func scopeSetFromSlice(scopes []string) map[string]struct{} {
	if len(scopes) == 0 {
		return nil
	}
	m := make(map[string]struct{}, len(scopes))
	for _, sc := range scopes {
		if sc != "" {
			m[sc] = struct{}{}
		}
	}
	return m
}

func aclAllows(required []string, callerScopes map[string]struct{}) bool {
	if len(required) == 0 {
		return true
	}
	for _, sc := range required {
		if _, ok := callerScopes[sc]; ok {
			return true
		}
	}
	return false
}
