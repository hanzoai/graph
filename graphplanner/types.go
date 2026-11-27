/*
 * SPDX-License-Identifier: Apache-2.0
 */

package graphplanner

import "errors"

// SourceKind classifies canonical versioned records stored in the native graph.
type SourceKind string

const (
	KindDocument   SourceKind = "document"
	KindRepoBlob   SourceKind = "repo_blob"
	KindGitCommit  SourceKind = "git_commit"
	KindAgentEvent SourceKind = "agent_event"
	KindDecision   SourceKind = "decision"
	KindEntity     SourceKind = "entity"
)

// ExecutionPlan names one of the five physical query execution strategies
// evaluated by the cost-based planner.
type ExecutionPlan string

const (
	PlanPlainRAGBypass     ExecutionPlan = "plain_rag_bypass"
	PlanCachedNeighborhood ExecutionPlan = "cached_neighborhood"
	PlanPureTraversal      ExecutionPlan = "pure_traversal"
	PlanGraphFirst         ExecutionPlan = "graph_first"
	PlanVectorFirst        ExecutionPlan = "vector_first"
)

// AllExecutionPlans lists all five supported physical plans in canonical order.
var AllExecutionPlans = []ExecutionPlan{
	PlanPlainRAGBypass,
	PlanCachedNeighborhood,
	PlanPureTraversal,
	PlanGraphFirst,
	PlanVectorFirst,
}

var (
	// ErrDerivedIndexNotBuilt is returned when a query runs while derived CSR,
	// vector, or lexical indexes are unbuilt or invalidated after a canonical
	// mutation.
	ErrDerivedIndexNotBuilt = errors.New("graphplanner: derived indexes not built; call RebuildIndex")
	// ErrInvalidRecord is returned when a canonical node or edge fails
	// validation (missing identity, tenant, version, or invalid float values).
	ErrInvalidRecord = errors.New("graphplanner: invalid canonical record")
	// ErrInvalidAccessContext is returned when a query lacks a tenant ID.
	ErrInvalidAccessContext = errors.New("graphplanner: missing tenant in access context")
	// ErrInvalidPlan is returned when ForcePlan specifies an unknown plan name.
	ErrInvalidPlan = errors.New("graphplanner: unknown execution plan")
	// ErrCapacityExceeded is returned if integer interning limits would overflow.
	ErrCapacityExceeded = errors.New("graphplanner: compact intern table capacity exceeded")
)

// AccessContext identifies the calling tenant and granted ACL scopes.
// Tenant isolation is strictly enforced before any candidate is inspected.
type AccessContext struct {
	TenantID string   `json:"tenant_id"`
	Scopes   []string `json:"scopes,omitempty"`
}

// CanonicalNode is an immutable-versioned canonical entity/document record.
type CanonicalNode struct {
	SourceID  string            `json:"source_id"`
	Version   uint64            `json:"version"`
	TenantID  string            `json:"tenant_id"`
	ACL       []string          `json:"acl,omitempty"`
	Kind      SourceKind        `json:"kind"`
	Title     string            `json:"title"`
	Text      string            `json:"text"`
	Labels    []string          `json:"labels,omitempty"`
	Metadata  map[string]string `json:"metadata,omitempty"`
	Embedding []float32         `json:"embedding,omitempty"`
	Timestamp int64             `json:"timestamp"`
}

// Clone returns a complete deep copy of CanonicalNode, including empty non-nil
// slices and maps so external mutations can never alias internal store state.
func (n CanonicalNode) Clone() CanonicalNode {
	out := n
	if n.ACL != nil {
		out.ACL = make([]string, len(n.ACL))
		copy(out.ACL, n.ACL)
	}
	if n.Labels != nil {
		out.Labels = make([]string, len(n.Labels))
		copy(out.Labels, n.Labels)
	}
	if n.Metadata != nil {
		out.Metadata = make(map[string]string, len(n.Metadata))
		for k, v := range n.Metadata {
			out.Metadata[k] = v
		}
	}
	if n.Embedding != nil {
		out.Embedding = make([]float32, len(n.Embedding))
		copy(out.Embedding, n.Embedding)
	}
	return out
}

// CanonicalEdge is a versioned directed typed relationship between two
// canonical nodes within the same tenant.
type CanonicalEdge struct {
	EdgeID       string            `json:"edge_id"`
	Version      uint64            `json:"version"`
	TenantID     string            `json:"tenant_id"`
	ACL          []string          `json:"acl,omitempty"`
	FromSourceID string            `json:"from_source_id"`
	ToSourceID   string            `json:"to_source_id"`
	Type         string            `json:"type"`
	Weight       float32           `json:"weight"`
	Metadata     map[string]string `json:"metadata,omitempty"`
	Timestamp    int64             `json:"timestamp"`
}

// Clone returns a complete deep copy of CanonicalEdge.
func (e CanonicalEdge) Clone() CanonicalEdge {
	out := e
	if e.ACL != nil {
		out.ACL = make([]string, len(e.ACL))
		copy(out.ACL, e.ACL)
	}
	if e.Metadata != nil {
		out.Metadata = make(map[string]string, len(e.Metadata))
		for k, v := range e.Metadata {
			out.Metadata[k] = v
		}
	}
	return out
}

// QueryRequest specifies a hybrid graph/vector/lexical query and optional
// constraints and latency budget for the cost-based planner.
type QueryRequest struct {
	Query                  string        `json:"query"`
	AnchorIDs              []string      `json:"anchor_ids,omitempty"`
	EdgeTypes              []string      `json:"edge_types,omitempty"`
	TargetLabels           []string      `json:"target_labels,omitempty"`
	MaxHops                int           `json:"max_hops"`
	TopK                   int           `json:"top_k"`
	LatencyBudgetMs        float64       `json:"latency_budget_ms"`
	Access                 AccessContext `json:"access"`
	ForcePlan              ExecutionPlan `json:"force_plan,omitempty"`
	RequireStructuralMatch bool          `json:"require_structural_match,omitempty"`
	// IncludeAnchors controls whether anchor nodes themselves can appear in hits.
	IncludeAnchors bool `json:"include_anchors,omitempty"`
}

// ScoredHit is a single authorized retrieval result with score decomposition
// and provenance trail.
type ScoredHit struct {
	SourceID     string            `json:"source_id"`
	Version      uint64            `json:"version"`
	Kind         SourceKind        `json:"kind"`
	Title        string            `json:"title"`
	Text         string            `json:"text"`
	Labels       []string          `json:"labels,omitempty"`
	Score        float64           `json:"score"`
	VectorScore  float64           `json:"vector_score"`
	LexicalScore float64           `json:"lexical_score"`
	GraphScore   float64           `json:"graph_score"`
	HopDistance  int               `json:"hop_distance"`
	Path         []string          `json:"path,omitempty"`
	Metadata     map[string]string `json:"metadata,omitempty"`
}

// PlanCostEstimate captures the planner's pre-execution cost estimation for a
// candidate physical execution plan.
type PlanCostEstimate struct {
	Plan                 ExecutionPlan `json:"plan"`
	EstimatedLatencyMs   float64       `json:"estimated_latency_ms"`
	EstimatedCandidates  int           `json:"estimated_candidates"`
	EstimatedSelectivity float64       `json:"estimated_selectivity"`
	EstimatedLocalDegree float64       `json:"estimated_local_degree"`
	CacheHitProbability  float64       `json:"cache_hit_probability"`
	EstimatedCostScore   float64       `json:"estimated_cost_score"`
	Feasible             bool          `json:"feasible"`
	Rationale            string        `json:"rationale"`
}

// QueryExplanation provides full transparency into why the planner chose a
// specific execution plan over the alternatives.
type QueryExplanation struct {
	SelectedPlan          ExecutionPlan      `json:"selected_plan"`
	Forced                bool               `json:"forced"`
	LatencyBudgetMs       float64            `json:"latency_budget_ms"`
	AnchorCount           int                `json:"anchor_count"`
	RequestedHops         int                `json:"requested_hops"`
	EstimatedSelectivity  float64            `json:"estimated_selectivity"`
	EstimatedLocalDegree  float64            `json:"estimated_local_degree"`
	CacheHitProbability   float64            `json:"cache_hit_probability"`
	SimpleFactualDetected bool               `json:"simple_factual_detected"`
	Alternatives          []PlanCostEstimate `json:"alternatives"`
	SelectionReason       string             `json:"selection_reason"`
}

// QueryTelemetry captures measured execution metrics alongside security counters.
type QueryTelemetry struct {
	ActualLatencyMs       float64 `json:"actual_latency_ms"`
	CandidatesEvaluated   int     `json:"candidates_evaluated"`
	NodesVisited          int     `json:"nodes_visited"`
	EdgesTraversed        int     `json:"edges_traversed"`
	ForeignNodesInspected int     `json:"foreign_nodes_inspected"`
	ForeignEdgesTraversed int     `json:"foreign_edges_traversed"`
	CacheHits             int     `json:"cache_hits"`
	CacheMisses           int     `json:"cache_misses"`
	EstimatedTokens       int     `json:"estimated_tokens"`
}

// QueryResponse is the complete result returned by ExecuteQuery.
type QueryResponse struct {
	Hits        []ScoredHit      `json:"hits"`
	Plan        ExecutionPlan    `json:"plan"`
	Explanation QueryExplanation `json:"explanation"`
	Telemetry   QueryTelemetry   `json:"telemetry"`
}

// StorageStats reports compact embedded CSR memory footprint and cache health.
type StorageStats struct {
	CanonicalNodes    int     `json:"canonical_nodes"`
	CanonicalEdges    int     `json:"canonical_edges"`
	IndexedNodes      int     `json:"indexed_nodes"`
	IndexedEdges      int     `json:"indexed_edges"`
	CSRForwardBytes   int64   `json:"csr_forward_bytes"`
	CSRReverseBytes   int64   `json:"csr_reverse_bytes"`
	InternTableBytes  int64   `json:"intern_table_bytes"`
	VectorIndexBytes  int64   `json:"vector_index_bytes"`
	LexicalIndexBytes int64   `json:"lexical_index_bytes"`
	TotalGraphBytes   int64   `json:"total_graph_bytes"`
	BytesPerNode      float64 `json:"bytes_per_node"`
	BytesPerEdge      float64 `json:"bytes_per_edge"`
	CacheEntries      int     `json:"cache_entries"`
	CacheCapacity     int     `json:"cache_capacity"`
	CacheHits         uint64  `json:"cache_hits"`
	CacheMisses       uint64  `json:"cache_misses"`
	CacheEvictions    uint64  `json:"cache_evictions"`
	CacheHitRate      float64 `json:"cache_hit_rate"`
	IndexDigestSHA256 string  `json:"index_digest_sha256"`
	DerivedBuilt      bool    `json:"derived_built"`
}
