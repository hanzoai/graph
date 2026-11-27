/*
 * SPDX-FileCopyrightText: © 2026 Hanzo AI, Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

package lightrag

import (
	"context"
	"errors"
	"time"
)

var (
	// ErrInvalidSource is returned when a SourceRecord fails validation.
	ErrInvalidSource = errors.New("lightrag: invalid source record")
	// ErrSourceNotFound is returned when a requested source does not exist.
	ErrSourceNotFound = errors.New("lightrag: source not found")
	// ErrIndexesNotBuilt is returned when Search is invoked after DropDerivedIndexes
	// before RebuildAll has reconstructed the derived projections.
	ErrIndexesNotBuilt = errors.New("lightrag: derived indexes not built; call RebuildAll")
	// ErrInvalidQuery is returned when a retrieval query is malformed.
	ErrInvalidQuery = errors.New("lightrag: invalid query")
	// ErrEmptyTenant is returned when TenantID is omitted.
	ErrEmptyTenant = errors.New("lightrag: tenant ID is required")
	// ErrStaleVersion is returned when an upsert supplies a version older than
	// the currently stored canonical version for the same (TenantID, SourceID).
	ErrStaleVersion = errors.New("lightrag: source version is older than stored version")
)

// SourceKind classifies the canonical origin of an ingested artifact.
type SourceKind string

const (
	KindDocument   SourceKind = "document"
	KindCodeBlob   SourceKind = "code_blob"
	KindCommit     SourceKind = "commit"
	KindAgentEvent SourceKind = "agent_event"
	KindDecision   SourceKind = "decision"
	KindIncident   SourceKind = "incident"
)

// EntityType classifies typed nodes in the derived entity-relationship graph.
type EntityType string

const (
	EntityPackage    EntityType = "package"
	EntityTypeDecl   EntityType = "type"
	EntityFunction   EntityType = "function"
	EntityMethod     EntityType = "method"
	EntityFile       EntityType = "file"
	EntityComponent  EntityType = "component"
	EntityCommit     EntityType = "commit"
	EntityDecision   EntityType = "decision"
	EntityIncident   EntityType = "incident"
	EntityAuthor     EntityType = "author"
	EntityConcept    EntityType = "concept"
	EntityDependency EntityType = "dependency"
)

// RelationType classifies typed edges in the derived entity-relationship graph.
type RelationType string

const (
	RelBelongsTo  RelationType = "BELONGS_TO"
	RelDefines    RelationType = "DEFINES"
	RelImports    RelationType = "IMPORTS"
	RelCalls      RelationType = "CALLS"
	RelReferences RelationType = "REFERENCES"
	RelModifies   RelationType = "MODIFIES"
	RelDependsOn  RelationType = "DEPENDS_ON"
	RelCauses     RelationType = "CAUSES"
	RelMitigates  RelationType = "MITIGATES"
	RelDecides    RelationType = "DECIDES"
	RelOwnedBy    RelationType = "OWNED_BY"
	RelMentions   RelationType = "MENTIONS"
	RelRelatedTo  RelationType = "RELATED_TO"
)

// RetrievalMode selects the LightRAG retrieval strategy or baseline.
type RetrievalMode string

const (
	// ModeLocal performs LightRAG low-level entity-focused retrieval with
	// bounded 1-hop/2-hop neighborhood expansion.
	ModeLocal RetrievalMode = "lightrag_local"
	// ModeGlobal performs LightRAG high-level relationship and theme retrieval.
	ModeGlobal RetrievalMode = "lightrag_global"
	// ModeHybrid performs LightRAG dual-level retrieval fusing Local, Global,
	// and direct chunk evidence via Reciprocal Rank Fusion.
	ModeHybrid RetrievalMode = "lightrag_hybrid"
	// ModeOrdinaryHybrid is the matched baseline: ordinary chunk-level hybrid
	// lexical (BM25) + dense vector search without entity-relationship graph traversal.
	ModeOrdinaryHybrid RetrievalMode = "ordinary_hybrid"
)

// SourceRecord is the canonical, immutable-identity record stored in the engine.
// All graph nodes, edges, chunks, lexical postings, and vectors are strictly
// derived from SourceRecords.
type SourceRecord struct {
	SourceID  string            `json:"source_id"`
	Version   int64             `json:"version"`
	TenantID  string            `json:"tenant_id"`
	ACL       []string          `json:"acl,omitempty"`
	Kind      SourceKind        `json:"kind"`
	Title     string            `json:"title,omitempty"`
	Path      string            `json:"path,omitempty"`
	Content   string            `json:"content"`
	Timestamp time.Time         `json:"timestamp,omitempty"`
	Metadata  map[string]string `json:"metadata,omitempty"`
}

// Clone returns a deep copy of the SourceRecord, preserving nil vs non-nil
// empty slices and maps so callers cannot mutate engine internal state.
func (s SourceRecord) Clone() SourceRecord {
	out := s
	out.ACL = cloneStringSlice(s.ACL)
	out.Metadata = cloneStringMap(s.Metadata)
	return out
}

// ExtractedEntity is a typed entity emitted by an Extractor for a single source.
type ExtractedEntity struct {
	Name        string     `json:"name"`
	Type        EntityType `json:"type"`
	Description string     `json:"description"`
	Keywords    []string   `json:"keywords,omitempty"`
	Aliases     []string   `json:"aliases,omitempty"`
	Weight      float64    `json:"weight"`
}

// ExtractedRelation is a typed relationship emitted by an Extractor for a single source.
type ExtractedRelation struct {
	SourceName  string       `json:"source_name"`
	SourceType  EntityType   `json:"source_type"`
	TargetName  string       `json:"target_name"`
	TargetType  EntityType   `json:"target_type"`
	Type        RelationType `json:"type"`
	Description string       `json:"description"`
	Keywords    []string     `json:"keywords,omitempty"`
	Weight      float64      `json:"weight"`
}

// ExtractionMetrics captures deterministic token and cost accounting for extraction.
type ExtractionMetrics struct {
	InputTokens      int     `json:"input_tokens"`
	OutputTokens     int     `json:"output_tokens"`
	EstimatedCostUSD float64 `json:"estimated_cost_usd"`
	RawEntities      int     `json:"raw_entities"`
	RawRelations     int     `json:"raw_relations"`
}

// ExtractionResult is the output of an Extractor for one SourceRecord.
type ExtractionResult struct {
	Entities  []ExtractedEntity   `json:"entities"`
	Relations []ExtractedRelation `json:"relations"`
	Metrics   ExtractionMetrics   `json:"metrics"`
}

// Extractor is the pluggable entity-relationship extraction interface.
// Implementations may be deterministic AST/NLP parsers or LLM-backed extractors.
type Extractor interface {
	Extract(ctx context.Context, src SourceRecord) (ExtractionResult, error)
}

// FuncExtractor adapts a plain function to the Extractor interface.
type FuncExtractor func(ctx context.Context, src SourceRecord) (ExtractionResult, error)

// Extract invokes the wrapped extraction function.
func (f FuncExtractor) Extract(ctx context.Context, src SourceRecord) (ExtractionResult, error) {
	return f(ctx, src)
}

// Chunk is a derived text segment of a canonical SourceRecord.
type Chunk struct {
	ChunkID    string            `json:"chunk_id"`
	SourceID   string            `json:"source_id"`
	Version    int64             `json:"version"`
	TenantID   string            `json:"tenant_id"`
	ACL        []string          `json:"acl,omitempty"`
	Kind       SourceKind        `json:"kind"`
	Title      string            `json:"title,omitempty"`
	Path       string            `json:"path,omitempty"`
	Text       string            `json:"text"`
	TokenCount int               `json:"token_count"`
	Metadata   map[string]string `json:"metadata,omitempty"`
}

// Clone returns a deep copy of Chunk.
func (c Chunk) Clone() Chunk {
	out := c
	out.ACL = cloneStringSlice(c.ACL)
	out.Metadata = cloneStringMap(c.Metadata)
	return out
}

// Entity is a deduplicated, ACL-scoped view of a derived graph entity.
type Entity struct {
	ID           string     `json:"id"`
	CanonicalKey string     `json:"canonical_key"`
	Name         string     `json:"name"`
	Type         EntityType `json:"type"`
	Description  string     `json:"description"`
	Keywords     []string   `json:"keywords,omitempty"`
	Aliases      []string   `json:"aliases,omitempty"`
	SourceIDs    []string   `json:"source_ids,omitempty"`
	MentionCount int        `json:"mention_count"`
	Degree       int        `json:"degree"`
	Weight       float64    `json:"weight"`
}

// Clone returns a deep copy of Entity.
func (e Entity) Clone() Entity {
	out := e
	out.Keywords = cloneStringSlice(e.Keywords)
	out.Aliases = cloneStringSlice(e.Aliases)
	out.SourceIDs = cloneStringSlice(e.SourceIDs)
	return out
}

// Relation is a deduplicated, ACL-scoped view of a derived graph relationship.
type Relation struct {
	ID               string       `json:"id"`
	SourceEntityID   string       `json:"source_entity_id"`
	TargetEntityID   string       `json:"target_entity_id"`
	SourceEntityName string       `json:"source_entity_name"`
	TargetEntityName string       `json:"target_entity_name"`
	Type             RelationType `json:"type"`
	Description      string       `json:"description"`
	Keywords         []string     `json:"keywords,omitempty"`
	SourceIDs        []string     `json:"source_ids,omitempty"`
	Weight           float64      `json:"weight"`
}

// Clone returns a deep copy of Relation.
func (r Relation) Clone() Relation {
	out := r
	out.Keywords = cloneStringSlice(r.Keywords)
	out.SourceIDs = cloneStringSlice(r.SourceIDs)
	return out
}

// UpdateTelemetry records exact subgraph delta and deduplication metrics for
// an UpsertSource, DeleteSource, or RebuildAll operation.
type UpdateTelemetry struct {
	Operation                  string        `json:"operation"`
	TenantID                   string        `json:"tenant_id"`
	SourceID                   string        `json:"source_id,omitempty"`
	Version                    int64         `json:"version,omitempty"`
	Duration                   time.Duration `json:"duration_ns"`
	AddedEntities              int           `json:"added_entities"`
	UpdatedEntities            int           `json:"updated_entities"`
	RemovedEntities            int           `json:"removed_entities"`
	AffectedEntities           int           `json:"affected_entities"`
	AddedRelations             int           `json:"added_relations"`
	UpdatedRelations           int           `json:"updated_relations"`
	RemovedRelations           int           `json:"removed_relations"`
	AffectedRelations          int           `json:"affected_relations"`
	TotalEntities              int           `json:"total_entities"`
	TotalRelations             int           `json:"total_relations"`
	TotalChunks                int           `json:"total_chunks"`
	SubgraphChurnRatio         float64       `json:"subgraph_churn_ratio"`
	RawEntitiesExtracted       int           `json:"raw_entities_extracted"`
	DuplicateEntitiesCollapsed int           `json:"duplicate_entities_collapsed"`
	DuplicateEntityRate        float64       `json:"duplicate_entity_rate"`
	RawRelationsExtracted      int           `json:"raw_relations_extracted"`
	UniqueRelationsMerged      int           `json:"unique_relations_merged"`
	ExtractionTokensIn         int           `json:"extraction_tokens_in"`
	ExtractionTokensOut        int           `json:"extraction_tokens_out"`
	EstimatedCostUSD           float64       `json:"estimated_cost_usd"`
	CacheInvalidated           bool          `json:"cache_invalidated"`
}

// Query specifies a dual-level or baseline retrieval request.
type Query struct {
	Text             string        `json:"text"`
	TenantID         string        `json:"tenant_id"`
	Principals       []string      `json:"principals,omitempty"`
	Mode             RetrievalMode `json:"mode"`
	TopK             int           `json:"top_k"`
	MaxHops          int           `json:"max_hops,omitempty"`
	MaxExpandedNodes int           `json:"max_expanded_nodes,omitempty"`
	IncludeSubgraph  bool          `json:"include_subgraph,omitempty"`
	BypassCache      bool          `json:"bypass_cache,omitempty"`
}

// RetrievedChunk pairs a retrieved Chunk with per-channel relevance scores.
type RetrievedChunk struct {
	Chunk            Chunk    `json:"chunk"`
	Score            float64  `json:"score"`
	LocalScore       float64  `json:"local_score"`
	GlobalScore      float64  `json:"global_score"`
	LexicalScore     float64  `json:"lexical_score"`
	VectorScore      float64  `json:"vector_score"`
	MatchedEntities  []string `json:"matched_entities,omitempty"`
	MatchedRelations []string `json:"matched_relations,omitempty"`
}

// Clone returns a deep copy of RetrievedChunk.
func (rc RetrievedChunk) Clone() RetrievedChunk {
	out := rc
	out.Chunk = rc.Chunk.Clone()
	out.MatchedEntities = cloneStringSlice(rc.MatchedEntities)
	out.MatchedRelations = cloneStringSlice(rc.MatchedRelations)
	return out
}

// RetrievedEntity pairs a retrieved Entity with its relevance score and hop depth.
type RetrievedEntity struct {
	Entity Entity  `json:"entity"`
	Score  float64 `json:"score"`
	Hop    int     `json:"hop"`
}

// Clone returns a deep copy of RetrievedEntity.
func (re RetrievedEntity) Clone() RetrievedEntity {
	out := re
	out.Entity = re.Entity.Clone()
	return out
}

// RetrievedRelation pairs a retrieved Relation with its relevance score and hop depth.
type RetrievedRelation struct {
	Relation Relation `json:"relation"`
	Score    float64  `json:"score"`
	Hop      int      `json:"hop"`
}

// Clone returns a deep copy of RetrievedRelation.
func (rr RetrievedRelation) Clone() RetrievedRelation {
	out := rr
	out.Relation = rr.Relation.Clone()
	return out
}

// EvidenceTrail summarizes a grounded graph triple supporting a retrieved answer.
type EvidenceTrail struct {
	SourceEntity      string       `json:"source_entity"`
	Relation          RelationType `json:"relation"`
	TargetEntity      string       `json:"target_entity"`
	Description       string       `json:"description"`
	SupportingSources []string     `json:"supporting_sources,omitempty"`
	Score             float64      `json:"score"`
}

// Clone returns a deep copy of EvidenceTrail.
func (et EvidenceTrail) Clone() EvidenceTrail {
	out := et
	out.SupportingSources = cloneStringSlice(et.SupportingSources)
	return out
}

// SearchResult is the immutable response returned by Engine.Search.
type SearchResult struct {
	Mode               RetrievalMode       `json:"mode"`
	Query              string              `json:"query"`
	LowLevelKeywords   []string            `json:"low_level_keywords,omitempty"`
	HighLevelKeywords  []string            `json:"high_level_keywords,omitempty"`
	Chunks             []RetrievedChunk    `json:"chunks"`
	Entities           []RetrievedEntity   `json:"entities,omitempty"`
	Relations          []RetrievedRelation `json:"relations,omitempty"`
	Trails             []EvidenceTrail     `json:"trails,omitempty"`
	Latency            time.Duration       `json:"latency_ns"`
	CacheHit           bool                `json:"cache_hit"`
	TotalTokens        int                 `json:"total_tokens"`
	InspectedEntities  int                 `json:"inspected_entities"`
	InspectedRelations int                 `json:"inspected_relations"`
	ACLFilteredOut     int                 `json:"acl_filtered_out"`
}

// Clone returns a deep copy of SearchResult so caller mutations cannot corrupt
// cached or internal state.
func (sr SearchResult) Clone() SearchResult {
	out := sr
	out.LowLevelKeywords = cloneStringSlice(sr.LowLevelKeywords)
	out.HighLevelKeywords = cloneStringSlice(sr.HighLevelKeywords)
	if sr.Chunks != nil {
		out.Chunks = make([]RetrievedChunk, len(sr.Chunks))
		for i := range sr.Chunks {
			out.Chunks[i] = sr.Chunks[i].Clone()
		}
	}
	if sr.Entities != nil {
		out.Entities = make([]RetrievedEntity, len(sr.Entities))
		for i := range sr.Entities {
			out.Entities[i] = sr.Entities[i].Clone()
		}
	}
	if sr.Relations != nil {
		out.Relations = make([]RetrievedRelation, len(sr.Relations))
		for i := range sr.Relations {
			out.Relations[i] = sr.Relations[i].Clone()
		}
	}
	if sr.Trails != nil {
		out.Trails = make([]EvidenceTrail, len(sr.Trails))
		for i := range sr.Trails {
			out.Trails[i] = sr.Trails[i].Clone()
		}
	}
	return out
}

// TenantStats summarizes derived index cardinality and deduplication health.
type TenantStats struct {
	TenantID              string  `json:"tenant_id"`
	Sources               int     `json:"sources"`
	Chunks                int     `json:"chunks"`
	Entities              int     `json:"entities"`
	Relations             int     `json:"relations"`
	Aliases               int     `json:"aliases"`
	CacheEntries          int     `json:"cache_entries"`
	Generation            uint64  `json:"generation"`
	TotalRawEntities      int     `json:"total_raw_entities"`
	TotalCollapsedDupes   int     `json:"total_collapsed_dupes"`
	AggregateDupeRate     float64 `json:"aggregate_dupe_rate"`
	TotalRawRelations     int     `json:"total_raw_relations"`
	TotalExtractionCostUS float64 `json:"total_extraction_cost_usd"`
	DerivedBuilt          bool    `json:"derived_built"`
}

// EngineConfig configures the LightRAG incremental indexing and retrieval engine.
type EngineConfig struct {
	ChunkSizeChars             int
	ChunkOverlapChars          int
	VectorDim                  int
	RequireExplicitACL         bool
	EnableQueryCache           bool
	MaxCacheEntries            int
	LocalWeight                float64
	GlobalWeight               float64
	ChunkWeight                float64
	HopDecay                   float64
	InputTokenPricePerMillion  float64
	OutputTokenPricePerMillion float64
	Extractor                  Extractor
}

// DefaultEngineConfig returns production defaults aligned with LightRAG dual-level
// retrieval and deterministic zero-dependency embeddings.
func DefaultEngineConfig() EngineConfig {
	return EngineConfig{
		ChunkSizeChars:             1400,
		ChunkOverlapChars:          200,
		VectorDim:                  128,
		RequireExplicitACL:         false,
		EnableQueryCache:           true,
		MaxCacheEntries:            256,
		LocalWeight:                0.45,
		GlobalWeight:               0.35,
		ChunkWeight:                0.20,
		HopDecay:                   0.55,
		InputTokenPricePerMillion:  0.15,
		OutputTokenPricePerMillion: 0.60,
	}
}

func (c EngineConfig) withDefaults() EngineConfig {
	def := DefaultEngineConfig()
	if c.ChunkSizeChars <= 0 {
		c.ChunkSizeChars = def.ChunkSizeChars
	}
	if c.ChunkOverlapChars < 0 || c.ChunkOverlapChars >= c.ChunkSizeChars {
		c.ChunkOverlapChars = def.ChunkOverlapChars
		if c.ChunkOverlapChars >= c.ChunkSizeChars {
			c.ChunkOverlapChars = c.ChunkSizeChars / 4
		}
	}
	if c.VectorDim <= 0 {
		c.VectorDim = def.VectorDim
	}
	if c.MaxCacheEntries <= 0 {
		c.MaxCacheEntries = def.MaxCacheEntries
	}
	if c.LocalWeight == 0 && c.GlobalWeight == 0 && c.ChunkWeight == 0 {
		c.LocalWeight = def.LocalWeight
		c.GlobalWeight = def.GlobalWeight
		c.ChunkWeight = def.ChunkWeight
	}
	if c.HopDecay <= 0 || c.HopDecay >= 1 {
		c.HopDecay = def.HopDecay
	}
	if c.InputTokenPricePerMillion <= 0 {
		c.InputTokenPricePerMillion = def.InputTokenPricePerMillion
	}
	if c.OutputTokenPricePerMillion <= 0 {
		c.OutputTokenPricePerMillion = def.OutputTokenPricePerMillion
	}
	if c.Extractor == nil {
		c.Extractor = NewDefaultExtractor(c.InputTokenPricePerMillion, c.OutputTokenPricePerMillion)
	}
	return c
}

// cloneStringSlice preserves nil vs non-nil empty slices.
func cloneStringSlice(s []string) []string {
	if s == nil {
		return nil
	}
	out := make([]string, len(s))
	copy(out, s)
	return out
}

// cloneStringMap preserves nil vs non-nil empty maps.
func cloneStringMap(m map[string]string) map[string]string {
	if m == nil {
		return nil
	}
	out := make(map[string]string, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}
