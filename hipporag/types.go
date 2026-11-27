/*
 * SPDX-License-Identifier: Apache-2.0
 */

package hipporag

import "errors"

var (
	// ErrEmptyTenant is returned when a record or query omits TenantID.
	ErrEmptyTenant = errors.New("hipporag: tenant ID is required")
	// ErrEmptySourceID is returned when a canonical source record omits SourceID.
	ErrEmptySourceID = errors.New("hipporag: source ID is required")
	// ErrInvalidVersion is returned when a canonical source record has Version == 0.
	ErrInvalidVersion = errors.New("hipporag: source version must be >= 1")
	// ErrIndexesNotBuilt is returned when querying before derived indexes are built,
	// after DropDerivedIndexes, or after UpsertSource mutates canonical state before RebuildIndex.
	ErrIndexesNotBuilt = errors.New("hipporag: derived indexes are dropped; call RebuildIndex")
)

// SourceKind classifies a canonical source record.
type SourceKind string

const (
	SourceDocument   SourceKind = "document"
	SourceRepoBlob   SourceKind = "repo_blob"
	SourceGitCommit  SourceKind = "git_commit"
	SourceAgentEvent SourceKind = "agent_event"
	SourceDecision   SourceKind = "decision"
)

// ExtractedTriple represents a directional relational fact grounded in a source record.
type ExtractedTriple struct {
	Subject     string  `json:"subject"`
	Predicate   string  `json:"predicate"`
	Object      string  `json:"object"`
	Relevance   float64 `json:"relevance"`   // (0, 1], defaults to 0.85 if <= 0
	Reliability float64 `json:"reliability"` // (0, 1], defaults to SourceRecord.Reliability if <= 0
}

// SourceRecord is an immutable versioned canonical record. All lexical, vector,
// entity-passage graph, and associative memory indexes are derived from these records.
type SourceRecord struct {
	SourceID    string            `json:"source_id"`
	Version     uint64            `json:"version"`
	Kind        SourceKind        `json:"kind"`
	TenantID    string            `json:"tenant_id"`
	ACL         []string          `json:"acl,omitempty"`
	Title       string            `json:"title"`
	Text        string            `json:"text"`
	Timestamp   int64             `json:"timestamp"`
	Reliability float64           `json:"reliability"` // (0, 1], defaults to 0.85 if <= 0
	Metadata    map[string]string `json:"metadata,omitempty"`
	Triples     []ExtractedTriple `json:"triples,omitempty"`
}

// Clone returns a deep copy of the canonical source record.
func (r SourceRecord) Clone() SourceRecord {
	cp := r
	if r.ACL != nil {
		cp.ACL = make([]string, len(r.ACL))
		copy(cp.ACL, r.ACL)
	}
	if r.Metadata != nil {
		cp.Metadata = make(map[string]string, len(r.Metadata))
		for k, v := range r.Metadata {
			cp.Metadata[k] = v
		}
	}
	if r.Triples != nil {
		cp.Triples = make([]ExtractedTriple, len(r.Triples))
		copy(cp.Triples, r.Triples)
	}
	return cp
}

// AccessContext carries the caller's tenant identity and authorized ACL scopes.
// It is evaluated strictly before seed selection, graph traversal, and result return.
type AccessContext struct {
	TenantID string   `json:"tenant_id"`
	Scopes   []string `json:"scopes,omitempty"`
}

// IsAuthorized reports whether ctx may read a canonical record with the given tenant and ACL.
// A non-empty record ACL requires at least one matching scope in ctx.Scopes.
func (ctx AccessContext) IsAuthorized(tenantID string, acl []string) bool {
	if ctx.TenantID == "" || tenantID == "" || ctx.TenantID != tenantID {
		return false
	}
	if len(acl) == 0 {
		return true
	}
	if len(ctx.Scopes) == 0 {
		return false
	}
	for _, required := range acl {
		for _, held := range ctx.Scopes {
			if held == required {
				return true
			}
		}
	}
	return false
}

// NodeKind identifies the role of a derived node in the associative graph.
type NodeKind string

const (
	NodeEntity   NodeKind = "entity"
	NodeFact     NodeKind = "fact"
	NodePassage  NodeKind = "passage"
	NodeCommit   NodeKind = "commit"
	NodeEvent    NodeKind = "event"
	NodeDecision NodeKind = "decision"
)

// EdgeKind identifies the semantic type of a derived edge in the associative graph.
type EdgeKind string

const (
	EdgeFactLink       EdgeKind = "fact_link"
	EdgePassageMention EdgeKind = "passage_mention"
	EdgeSynonymy       EdgeKind = "synonymy"
	EdgeStructural     EdgeKind = "structural"
)

// GraphNode is a derived node inside the tenant-partitioned associative graph.
type GraphNode struct {
	ID          string   `json:"id"`
	Kind        NodeKind `json:"kind"`
	TenantID    string   `json:"tenant_id"`
	Label       string   `json:"label"`
	SourceID    string   `json:"source_id,omitempty"`
	Version     uint64   `json:"version,omitempty"`
	Reliability float64  `json:"reliability"`
	// SupportingSources lists every canonical SourceID that contributes to this node.
	// At query time, a node is visible iff at least one supporting source is authorized.
	SupportingSources []string `json:"supporting_sources,omitempty"`
}

// GraphEdge is a directed weighted edge in the derived associative graph.
type GraphEdge struct {
	From        string   `json:"from"`
	To          string   `json:"to"`
	Kind        EdgeKind `json:"kind"`
	Predicate   string   `json:"predicate"`
	TenantID    string   `json:"tenant_id"`
	SourceID    string   `json:"source_id,omitempty"`
	ACL         []string `json:"acl,omitempty"`
	Relevance   float64  `json:"relevance"`
	Reliability float64  `json:"reliability"`
}

// FactRecord is a derived reified triple used during HippoRAG 2 recognition-memory filtering.
type FactRecord struct {
	ID          string   `json:"id"`
	TenantID    string   `json:"tenant_id"`
	SourceID    string   `json:"source_id"`
	Version     uint64   `json:"version"`
	ACL         []string `json:"acl,omitempty"`
	Subject     string   `json:"subject"`
	SubjectID   string   `json:"subject_id"`
	Predicate   string   `json:"predicate"`
	Object      string   `json:"object"`
	ObjectID    string   `json:"object_id"`
	Relevance   float64  `json:"relevance"`
	Reliability float64  `json:"reliability"`
	Text        string   `json:"text"`
}

// PPRConfig controls bounded Personalized PageRank, recognition-memory filtering,
// and edge weighting.
type PPRConfig struct {
	// Damping is the PPR continuation probability alpha in (0, 1).
	// Teleport probability back to the personalized seed vector is (1 - Damping).
	Damping float64 `json:"damping"`
	// MaxIterations caps power-iteration steps.
	MaxIterations int `json:"max_iterations"`
	// Epsilon is the L1 convergence threshold between consecutive PPR vectors.
	Epsilon float64 `json:"epsilon"`
	// MaxHops bounds the authorized ego-subgraph distance from initial seed nodes.
	MaxHops int `json:"max_hops"`
	// MaxExpandedNodes bounds the maximum number of nodes included in the PPR walk.
	MaxExpandedNodes int `json:"max_expanded_nodes"`
	// TopK is the maximum number of source hits returned.
	TopK int `json:"top_k"`

	// PassageSeedWeight controls the HippoRAG 2 dual-seeding weight placed on
	// dense-matched passage nodes relative to phrase/entity seed nodes.
	PassageSeedWeight float64 `json:"passage_seed_weight"`
	// MaxSeedFacts caps the number of candidate facts admitted to recognition memory.
	MaxSeedFacts int `json:"max_seed_facts"`
	// RecognitionThreshold filters out low-alignment / low-reliability triples.
	RecognitionThreshold float64 `json:"recognition_threshold"`
	// EnableRecognitionMemory toggles HippoRAG 2 triple recognition-memory filtering.
	EnableRecognitionMemory bool `json:"enable_recognition_memory"`

	// EnableReliabilityWeighting scales edge transitions by reliability^ReliabilityExponent.
	EnableReliabilityWeighting bool    `json:"enable_reliability_weighting"`
	ReliabilityExponent        float64 `json:"reliability_exponent"`
	// EnableHubDamping penalizes high-degree generic hub nodes during seeding and transitions.
	EnableHubDamping bool    `json:"enable_hub_damping"`
	HubDegreePivot   float64 `json:"hub_degree_pivot"`
	HubPenaltyAlpha  float64 `json:"hub_penalty_alpha"`

	// SynonymyThreshold is the cosine similarity cutoff for derived entity synonymy edges.
	SynonymyThreshold float64 `json:"synonymy_threshold"`
	// SynonymyTopK limits outgoing synonymy edges per entity node.
	SynonymyTopK int `json:"synonymy_top_k"`
}

// DefaultPPRConfig returns calibrated HippoRAG 2 bounded PPR defaults.
func DefaultPPRConfig() PPRConfig {
	return PPRConfig{
		Damping:                    0.82,
		MaxIterations:              25,
		Epsilon:                    1e-6,
		MaxHops:                    5,
		MaxExpandedNodes:           256,
		TopK:                       5,
		PassageSeedWeight:          0.20,
		MaxSeedFacts:               8,
		RecognitionThreshold:       0.22,
		EnableRecognitionMemory:    true,
		EnableReliabilityWeighting: true,
		ReliabilityExponent:        2.0,
		EnableHubDamping:           true,
		HubDegreePivot:             6.0,
		HubPenaltyAlpha:            0.65,
		SynonymyThreshold:          0.74,
		SynonymyTopK:               3,
	}
}

// UnweightedNoRecogConfig disables reliability weighting, hub damping, and
// recognition-memory filtering for the hub-collapse negative boundary regime.
func UnweightedNoRecogConfig(topK int) PPRConfig {
	cfg := DefaultPPRConfig()
	cfg.TopK = topK
	cfg.EnableRecognitionMemory = false
	cfg.EnableReliabilityWeighting = false
	cfg.EnableHubDamping = false
	cfg.PassageSeedWeight = 0.05
	cfg.RecognitionThreshold = 0.0
	return cfg
}

// QueryRequest specifies a retrieval query and its mandatory AccessContext.
type QueryRequest struct {
	Query  string        `json:"query"`
	Access AccessContext `json:"access"`
	TopK   int           `json:"top_k,omitempty"`
}

// PathStep is one directed hop in a reconstructed evidence path.
type PathStep struct {
	FromNode    string   `json:"from_node"`
	FromLabel   string   `json:"from_label"`
	FromKind    NodeKind `json:"from_kind"`
	Predicate   string   `json:"predicate"`
	ToNode      string   `json:"to_node"`
	ToLabel     string   `json:"to_label"`
	ToKind      NodeKind `json:"to_kind"`
	EdgeKind    EdgeKind `json:"edge_kind"`
	EdgeWeight  float64  `json:"edge_weight"`
	Relevance   float64  `json:"relevance"`
	Reliability float64  `json:"reliability"`
	SourceID    string   `json:"source_id,omitempty"`
}

// EvidencePath traces how a retrieved hit connects back to a query seed node.
type EvidencePath struct {
	SeedNode   string     `json:"seed_node"`
	TargetNode string     `json:"target_node"`
	Score      float64    `json:"score"`
	Hops       int        `json:"hops"`
	Steps      []PathStep `json:"steps"`
	Trail      string     `json:"trail"`
}

// RetrievedHit is a single authorized source returned by a retriever.
type RetrievedHit struct {
	SourceID    string       `json:"source_id"`
	Version     uint64       `json:"version"`
	Kind        SourceKind   `json:"kind"`
	TenantID    string       `json:"tenant_id"`
	Title       string       `json:"title"`
	Text        string       `json:"text"`
	Score       float64      `json:"score"`
	Reliability float64      `json:"reliability"`
	HopDistance int          `json:"hop_distance"`
	Path        EvidencePath `json:"path"`
	Tokens      int          `json:"tokens"`
}

// AuditStats provides verifiable pre-return isolation and bounded-walk metrics.
type AuditStats struct {
	ForeignTenantNodesInspected   int     `json:"foreign_tenant_nodes_inspected"`
	UnauthorizedACLNodesInspected int     `json:"unauthorized_acl_nodes_inspected"`
	ForeignTenantEdgesTraversed   int     `json:"foreign_tenant_edges_traversed"`
	SeedEntityCount               int     `json:"seed_entity_count"`
	SeedPassageCount              int     `json:"seed_passage_count"`
	CandidateFactsEvaluated       int     `json:"candidate_facts_evaluated"`
	FactsAcceptedByRecognition    int     `json:"facts_accepted_by_recognition"`
	FactsRejectedByRecognition    int     `json:"facts_rejected_by_recognition"`
	ExpandedNodes                 int     `json:"expanded_nodes"`
	ExpandedEdges                 int     `json:"expanded_edges"`
	PPRIterations                 int     `json:"ppr_iterations"`
	PPRResidualL1                 float64 `json:"ppr_residual_l1"`
	Converged                     bool    `json:"converged"`
}

// QueryResult is the output of a retrieval call.
type QueryResult struct {
	Method          string         `json:"method"`
	Query           string         `json:"query"`
	Hits            []RetrievedHit `json:"hits"`
	SeedEntities    []string       `json:"seed_entities,omitempty"`
	FilteredFacts   []string       `json:"filtered_facts,omitempty"`
	TotalTokens     int            `json:"total_tokens"`
	Audit           AuditStats     `json:"audit"`
	ReconstructedOK bool           `json:"reconstructed_ok"`
}

// IndexSnapshot captures a deterministic structural digest of derived indexes
// used to verify bit-exact rebuild equivalence after DropDerivedIndexes -> RebuildIndex.
type IndexSnapshot struct {
	CanonicalRecords int    `json:"canonical_records"`
	GraphNodes       int    `json:"graph_nodes"`
	GraphEdges       int    `json:"graph_edges"`
	FactRecords      int    `json:"fact_records"`
	SynonymyEdges    int    `json:"synonymy_edges"`
	VocabTerms       int    `json:"vocab_terms"`
	DigestSHA256     string `json:"digest_sha256"`
}
