package graphrag

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"sort"
	"strings"
	"time"
)

var (
	// ErrInvalidInput is returned when required canonical identity or query
	// fields are missing or malformed.
	ErrInvalidInput = errors.New("graphrag: invalid input")

	// ErrUnauthorized is returned when CallerContext is missing a TenantID.
	ErrUnauthorized = errors.New("graphrag: missing caller tenant context")

	// ErrNotFound is returned uniformly for both nonexistent IDs and
	// unauthorized/cross-tenant IDs to prevent existence-oracle leakage.
	ErrNotFound = errors.New("graphrag: not found")

	// ErrDerivedIndexNotBuilt is returned when a query is executed after
	// DropDerivedIndexes before RebuildAll/BuildAll has restored derived state.
	ErrDerivedIndexNotBuilt = errors.New("graphrag: derived index not built")
)

// QueryMode controls how Engine.Query routes and executes a query.
type QueryMode string

const (
	// ModeAuto uses QueryRouter to dispatch corpus-wide synthesis questions to
	// GraphRAG hierarchical map-reduce and exact/local lookups to the ordinary
	// top-K hybrid retrieval baseline.
	ModeAuto QueryMode = "auto"

	// ModeGlobalL0 forces GraphRAG map-reduce over leaf (L0) community summaries.
	ModeGlobalL0 QueryMode = "global_l0"

	// ModeGlobalL1 forces GraphRAG map-reduce over mid-level (L1) community summaries.
	ModeGlobalL1 QueryMode = "global_l1"

	// ModeGlobalL2 forces GraphRAG map-reduce over macro-level (L2) community summaries.
	ModeGlobalL2 QueryMode = "global_l2"

	// ModeLocalHybrid forces ordinary top-K hybrid (BM25 + dense vector) retrieval
	// over canonical document chunks without GraphRAG community map-reduce.
	ModeLocalHybrid QueryMode = "local_hybrid"
)

// RouteTarget identifies the execution path chosen for a query.
type RouteTarget string

const (
	RouteGlobalGraphRAG RouteTarget = "global_graphrag"
	RouteLocalHybrid    RouteTarget = "local_hybrid"
)

// SourceRef is a canonical provenance pointer linking derived entities,
// communities, summaries, and answers back to immutable source identity.
type SourceRef struct {
	SourceID string `json:"source_id"`
	Version  string `json:"version"`
	TenantID string `json:"tenant_id"`
}

// CanonicalDocument is the immutable source-of-truth record owned by the store.
// All derived graphs, communities, summaries, and search indexes can be dropped
// and reconstructed bit-for-bit from CanonicalDocument records.
type CanonicalDocument struct {
	SourceID  string            `json:"source_id"`
	Version   string            `json:"version"`
	TenantID  string            `json:"tenant_id"`
	ACL       []string          `json:"acl"`
	Kind      string            `json:"kind"`
	Title     string            `json:"title"`
	Content   string            `json:"content"`
	Metadata  map[string]string `json:"metadata,omitempty"`
	UpdatedAt time.Time         `json:"updated_at"`
}

// Clone returns a deep copy of the CanonicalDocument.
func (d CanonicalDocument) Clone() CanonicalDocument {
	out := d
	if d.ACL != nil {
		out.ACL = append([]string(nil), d.ACL...)
	}
	if d.Metadata != nil {
		out.Metadata = make(map[string]string, len(d.Metadata))
		for k, v := range d.Metadata {
			out.Metadata[k] = v
		}
	}
	return out
}

// CallerContext authenticates the caller's tenant and principal allow-list.
type CallerContext struct {
	TenantID   string   `json:"tenant_id"`
	Principals []string `json:"principals"`
}

// Entity represents a derived graph node extracted from one or more canonical
// documents within a single (TenantID, ACLScope) partition.
type Entity struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	Type        string            `json:"type"`
	Description string            `json:"description"`
	Themes      []string          `json:"themes,omitempty"`
	Weight      float64           `json:"weight"`
	SourceRefs  []SourceRef       `json:"source_refs"`
	Attributes  map[string]string `json:"attributes,omitempty"`
}

// Clone returns a deep copy of Entity.
func (e Entity) Clone() Entity {
	out := e
	if e.Themes != nil {
		out.Themes = append([]string(nil), e.Themes...)
	}
	if e.SourceRefs != nil {
		out.SourceRefs = append([]SourceRef(nil), e.SourceRefs...)
	}
	if e.Attributes != nil {
		out.Attributes = make(map[string]string, len(e.Attributes))
		for k, v := range e.Attributes {
			out.Attributes[k] = v
		}
	}
	return out
}

// Relation represents a derived undirected weighted relationship between two
// entities within the same (TenantID, ACLScope) partition.
type Relation struct {
	SourceEntityID string      `json:"source_entity_id"`
	TargetEntityID string      `json:"target_entity_id"`
	Kind           string      `json:"kind"`
	Description    string      `json:"description"`
	Weight         float64     `json:"weight"`
	SourceRefs     []SourceRef `json:"source_refs"`
}

// Clone returns a deep copy of Relation.
func (r Relation) Clone() Relation {
	out := r
	if r.SourceRefs != nil {
		out.SourceRefs = append([]SourceRef(nil), r.SourceRefs...)
	}
	return out
}

// ExtractionResult holds the entities, relations, and themes extracted from a
// single CanonicalDocument.
type ExtractionResult struct {
	Entities  []Entity   `json:"entities"`
	Relations []Relation `json:"relations"`
	Themes    []string   `json:"themes"`
}

// Community represents a node in the multi-level (L0 -> L1 -> L2) hierarchy
// within a single (TenantID, ACLScope) partition.
type Community struct {
	ID             string      `json:"id"`
	TenantID       string      `json:"tenant_id"`
	ScopeKey       string      `json:"scope_key"`
	ACL            []string    `json:"acl"`
	Level          int         `json:"level"`
	ParentID       string      `json:"parent_id,omitempty"`
	ChildIDs       []string    `json:"child_ids,omitempty"`
	EntityIDs      []string    `json:"entity_ids"`
	InternalWeight float64     `json:"internal_weight"`
	TotalWeight    float64     `json:"total_weight"`
	SourceRefs     []SourceRef `json:"source_refs"`
}

// Clone returns a deep copy of Community.
func (c Community) Clone() Community {
	out := c
	if c.ACL != nil {
		out.ACL = append([]string(nil), c.ACL...)
	}
	if c.ChildIDs != nil {
		out.ChildIDs = append([]string(nil), c.ChildIDs...)
	}
	if c.EntityIDs != nil {
		out.EntityIDs = append([]string(nil), c.EntityIDs...)
	}
	if c.SourceRefs != nil {
		out.SourceRefs = append([]SourceRef(nil), c.SourceRefs...)
	}
	return out
}

// CommunitySummary is the derived hierarchical summary attached to a Community.
type CommunitySummary struct {
	CommunityID      string         `json:"community_id"`
	TenantID         string         `json:"tenant_id"`
	ScopeKey         string         `json:"scope_key"`
	ACL              []string       `json:"acl"`
	Level            int            `json:"level"`
	Title            string         `json:"title"`
	Summary          string         `json:"summary"`
	Findings         []string       `json:"findings"`
	TopEntities      []string       `json:"top_entities"`
	ThemeCounts      map[string]int `json:"theme_counts"`
	SourceRefs       []SourceRef    `json:"source_refs"`
	PromptTokens     int            `json:"prompt_tokens"`
	CompletionTokens int            `json:"completion_tokens"`
	Revision         uint64         `json:"revision"`
	Dirty            bool           `json:"dirty"`
}

// Clone returns a deep copy of CommunitySummary.
func (s CommunitySummary) Clone() CommunitySummary {
	out := s
	if s.ACL != nil {
		out.ACL = append([]string(nil), s.ACL...)
	}
	if s.Findings != nil {
		out.Findings = append([]string(nil), s.Findings...)
	}
	if s.TopEntities != nil {
		out.TopEntities = append([]string(nil), s.TopEntities...)
	}
	if s.ThemeCounts != nil {
		out.ThemeCounts = make(map[string]int, len(s.ThemeCounts))
		for k, v := range s.ThemeCounts {
			out.ThemeCounts[k] = v
		}
	}
	if s.SourceRefs != nil {
		out.SourceRefs = append([]SourceRef(nil), s.SourceRefs...)
	}
	return out
}

// PartialAnswer is the output of the GraphRAG Map step for a single community
// summary.
type PartialAnswer struct {
	CommunityID      string      `json:"community_id"`
	Level            int         `json:"level"`
	RelevanceScore   float64     `json:"relevance_score"`
	KeyPoints        []string    `json:"key_points"`
	Themes           []string    `json:"themes"`
	SourceRefs       []SourceRef `json:"source_refs"`
	PromptTokens     int         `json:"prompt_tokens"`
	CompletionTokens int         `json:"completion_tokens"`
}

// Clone returns a deep copy of PartialAnswer.
func (p PartialAnswer) Clone() PartialAnswer {
	out := p
	if p.KeyPoints != nil {
		out.KeyPoints = append([]string(nil), p.KeyPoints...)
	}
	if p.Themes != nil {
		out.Themes = append([]string(nil), p.Themes...)
	}
	if p.SourceRefs != nil {
		out.SourceRefs = append([]SourceRef(nil), p.SourceRefs...)
	}
	return out
}

// SourceHit represents a retrieved canonical document chunk in the local hybrid
// retrieval path.
type SourceHit struct {
	SourceRef SourceRef `json:"source_ref"`
	Title     string    `json:"title"`
	Snippet   string    `json:"snippet"`
	Score     float64   `json:"score"`
	Themes    []string  `json:"themes,omitempty"`
}

// Clone returns a deep copy of SourceHit.
func (h SourceHit) Clone() SourceHit {
	out := h
	if h.Themes != nil {
		out.Themes = append([]string(nil), h.Themes...)
	}
	return out
}

// QueryRequest specifies a user or agent query and execution budget.
type QueryRequest struct {
	Query        string    `json:"query"`
	Mode         QueryMode `json:"mode,omitempty"`
	TopK         int       `json:"top_k,omitempty"`
	TokenBudget  int       `json:"token_budget,omitempty"`
	MaxMapGroups int       `json:"max_map_groups,omitempty"`
}

// RoutingDecision explains why QueryRouter selected Global GraphRAG or Local Hybrid.
type RoutingDecision struct {
	Target            RouteTarget `json:"target"`
	RecommendedMode   QueryMode   `json:"recommended_mode"`
	RecommendedLevel  int         `json:"recommended_level"`
	GlobalScore       float64     `json:"global_score"`
	LocalScore        float64     `json:"local_score"`
	Reason            string      `json:"reason"`
	MatchedGlobalCues []string    `json:"matched_global_cues,omitempty"`
	MatchedLocalCues  []string    `json:"matched_local_cues,omitempty"`
}

// Clone returns a deep copy of RoutingDecision.
func (r RoutingDecision) Clone() RoutingDecision {
	out := r
	if r.MatchedGlobalCues != nil {
		out.MatchedGlobalCues = append([]string(nil), r.MatchedGlobalCues...)
	}
	if r.MatchedLocalCues != nil {
		out.MatchedLocalCues = append([]string(nil), r.MatchedLocalCues...)
	}
	return out
}

// QueryMetrics records exact latency, token usage, cost, and ACL audit counters.
type QueryMetrics struct {
	LatencyMs            float64 `json:"latency_ms"`
	PromptTokens         int     `json:"prompt_tokens"`
	CompletionTokens     int     `json:"completion_tokens"`
	TotalTokens          int     `json:"total_tokens"`
	EstimatedCostUSD     float64 `json:"estimated_cost_usd"`
	CommunitiesScanned   int     `json:"communities_scanned"`
	CommunitiesMapped    int     `json:"communities_mapped"`
	ChunksScanned        int     `json:"chunks_scanned"`
	PartitionsAuthorized int     `json:"partitions_authorized"`
	PartitionsDenied     int     `json:"partitions_denied"`
}

// QueryResponse is the deep-copied response returned by Engine.Query.
type QueryResponse struct {
	Query              string          `json:"query"`
	RequestedMode      QueryMode       `json:"requested_mode"`
	EffectiveMode      QueryMode       `json:"effective_mode"`
	Routing            RoutingDecision `json:"routing"`
	CommunityLevelUsed int             `json:"community_level_used"`
	Answer             string          `json:"answer"`
	DetectedThemes     []string        `json:"detected_themes"`
	PartialAnswers     []PartialAnswer `json:"partial_answers,omitempty"`
	SourceHits         []SourceHit     `json:"source_hits,omitempty"`
	Citations          []SourceRef     `json:"citations"`
	Metrics            QueryMetrics    `json:"metrics"`
}

// Clone returns a deep copy of QueryResponse.
func (r QueryResponse) Clone() QueryResponse {
	out := r
	out.Routing = r.Routing.Clone()
	if r.DetectedThemes != nil {
		out.DetectedThemes = append([]string(nil), r.DetectedThemes...)
	}
	if r.PartialAnswers != nil {
		out.PartialAnswers = make([]PartialAnswer, len(r.PartialAnswers))
		for i := range r.PartialAnswers {
			out.PartialAnswers[i] = r.PartialAnswers[i].Clone()
		}
	}
	if r.SourceHits != nil {
		out.SourceHits = make([]SourceHit, len(r.SourceHits))
		for i := range r.SourceHits {
			out.SourceHits[i] = r.SourceHits[i].Clone()
		}
	}
	if r.Citations != nil {
		out.Citations = append([]SourceRef(nil), r.Citations...)
	}
	return out
}

// BuildStats records exact telemetry for a full offline community build.
type BuildStats struct {
	DurationMs         float64     `json:"duration_ms"`
	PartitionsBuilt    int         `json:"partitions_built"`
	DocumentsIndexed   int         `json:"documents_indexed"`
	EntitiesTotal      int         `json:"entities_total"`
	RelationsTotal     int         `json:"relations_total"`
	CommunitiesByLevel map[int]int `json:"communities_by_level"`
	SummariesBuilt     int         `json:"summaries_built"`
	PromptTokens       int         `json:"prompt_tokens"`
	CompletionTokens   int         `json:"completion_tokens"`
	TotalTokens        int         `json:"total_tokens"`
	EstimatedCostUSD   float64     `json:"estimated_cost_usd"`
	MeanModularityL0   float64     `json:"mean_modularity_l0"`
	MeanModularityL1   float64     `json:"mean_modularity_l1"`
	DerivedDigest      string      `json:"derived_digest"`
}

// IncrementalRefreshStats records exact telemetry for an incremental document
// update, ACL change, or deletion showing targeted leaf + ancestor invalidation.
type IncrementalRefreshStats struct {
	Operation               string  `json:"operation"`
	SourceID                string  `json:"source_id"`
	DurationMs              float64 `json:"duration_ms"`
	DirtyEntities           int     `json:"dirty_entities"`
	DirtyL0                 int     `json:"dirty_l0"`
	TotalL0                 int     `json:"total_l0"`
	DirtyL1                 int     `json:"dirty_l1"`
	TotalL1                 int     `json:"total_l1"`
	DirtyL2                 int     `json:"dirty_l2"`
	TotalL2                 int     `json:"total_l2"`
	CommunitiesResummarized int     `json:"communities_resummarized"`
	CommunitiesReused       int     `json:"communities_reused"`
	PromptTokens            int     `json:"prompt_tokens"`
	CompletionTokens        int     `json:"completion_tokens"`
	TotalTokens             int     `json:"total_tokens"`
	EstimatedCostUSD        float64 `json:"estimated_cost_usd"`
}

// Config configures the GraphRAG engine, community detector, and pricing model.
type Config struct {
	// Resolution controls Newman-Girvan modularity resolution gamma (default 1.0).
	Resolution float64
	// MaxHierarchyLevels is the number of hierarchy levels (default 3: L0, L1, L2).
	MaxHierarchyLevels int
	// MaxLocalMovePasses caps local modularity passes per level (default 20).
	MaxLocalMovePasses int
	// DefaultGlobalLevel is the default hierarchy level for ModeAuto global queries (default 1 = L1).
	DefaultGlobalLevel int
	// DefaultTopK is the default TopK for local hybrid retrieval (default 5).
	DefaultTopK int
	// DefaultTokenBudget is the default context token budget for synthesis (default 4000).
	DefaultTokenBudget int
	// InputTokenPricePerMillion USD per 1M prompt tokens (default $3.00).
	InputTokenPricePerMillion float64
	// OutputTokenPricePerMillion USD per 1M completion tokens (default $15.00).
	OutputTokenPricePerMillion float64
	// Extractor overrides the default Go AST + technical-text extractor.
	Extractor Extractor
	// Summarizer overrides the default deterministic hierarchical summarizer.
	Summarizer Summarizer
}

// DefaultConfig returns production defaults for the GraphRAG engine.
func DefaultConfig() Config {
	return Config{
		Resolution:                 1.0,
		MaxHierarchyLevels:         3,
		MaxLocalMovePasses:         20,
		DefaultGlobalLevel:         1,
		DefaultTopK:                5,
		DefaultTokenBudget:         4000,
		InputTokenPricePerMillion:  3.00,
		OutputTokenPricePerMillion: 15.00,
	}
}

func (c Config) normalized() Config {
	def := DefaultConfig()
	if c.Resolution <= 0 {
		c.Resolution = def.Resolution
	}
	if c.MaxHierarchyLevels <= 0 || c.MaxHierarchyLevels > 3 {
		c.MaxHierarchyLevels = def.MaxHierarchyLevels
	}
	if c.MaxLocalMovePasses <= 0 {
		c.MaxLocalMovePasses = def.MaxLocalMovePasses
	}
	if c.DefaultGlobalLevel < 0 || c.DefaultGlobalLevel >= c.MaxHierarchyLevels {
		c.DefaultGlobalLevel = def.DefaultGlobalLevel
		if c.DefaultGlobalLevel >= c.MaxHierarchyLevels {
			c.DefaultGlobalLevel = c.MaxHierarchyLevels - 1
		}
	}
	if c.DefaultTopK <= 0 {
		c.DefaultTopK = def.DefaultTopK
	}
	if c.DefaultTokenBudget <= 0 {
		c.DefaultTokenBudget = def.DefaultTokenBudget
	}
	if c.InputTokenPricePerMillion <= 0 {
		c.InputTokenPricePerMillion = def.InputTokenPricePerMillion
	}
	if c.OutputTokenPricePerMillion <= 0 {
		c.OutputTokenPricePerMillion = def.OutputTokenPricePerMillion
	}
	return c
}

// EstimateCostUSD calculates USD cost using configured input/output token rates.
func (c Config) EstimateCostUSD(promptTokens, completionTokens int) float64 {
	inCost := (float64(promptTokens) / 1_000_000.0) * c.InputTokenPricePerMillion
	outCost := (float64(completionTokens) / 1_000_000.0) * c.OutputTokenPricePerMillion
	return inCost + outCost
}

// NormalizeACL trims, deduplicates, and lexicographically sorts an ACL list.
// An empty ACL is preserved as empty (deny-by-default).
func NormalizeACL(acl []string) []string {
	if len(acl) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(acl))
	out := make([]string, 0, len(acl))
	for _, raw := range acl {
		p := strings.TrimSpace(raw)
		if p == "" {
			continue
		}
		if _, ok := seen[p]; ok {
			continue
		}
		seen[p] = struct{}{}
		out = append(out, p)
	}
	if len(out) == 0 {
		return nil
	}
	sort.Strings(out)
	return out
}

// ACLScopeKey derives a canonical deterministic scope identifier for a
// normalized ACL within a TenantID.
func ACLScopeKey(tenantID string, acl []string) string {
	norm := NormalizeACL(acl)
	h := sha256.New()
	h.Write([]byte(tenantID))
	h.Write([]byte{0})
	if len(norm) == 0 {
		h.Write([]byte("<deny-all>"))
	} else {
		for _, p := range norm {
			h.Write([]byte(p))
			h.Write([]byte{0})
		}
	}
	return hex.EncodeToString(h.Sum(nil))[:16]
}

// ACLAllows returns true iff the caller's principals satisfy the resource ACL.
// Security posture:
//   - Empty resource ACL is deny-by-default (returns false).
//   - Wildcard "*" in resource ACL permits any caller within the SAME TenantID
//     (cross-tenant checks are always enforced separately before ACLAllows).
//   - Otherwise, at least one principal in callerPrincipals must match an entry
//     in resourceACL.
func ACLAllows(resourceACL []string, callerPrincipals []string) bool {
	if len(resourceACL) == 0 {
		return false
	}
	for _, p := range resourceACL {
		if p == "*" {
			return true
		}
	}
	if len(callerPrincipals) == 0 {
		return false
	}
	callerSet := make(map[string]struct{}, len(callerPrincipals))
	for _, cp := range callerPrincipals {
		cp = strings.TrimSpace(cp)
		if cp != "" {
			callerSet[cp] = struct{}{}
		}
	}
	for _, rp := range resourceACL {
		if _, ok := callerSet[rp]; ok {
			return true
		}
	}
	return false
}

func sortSourceRefs(refs []SourceRef) []SourceRef {
	if len(refs) == 0 {
		return nil
	}
	seen := make(map[SourceRef]struct{}, len(refs))
	out := make([]SourceRef, 0, len(refs))
	for _, r := range refs {
		if _, ok := seen[r]; ok {
			continue
		}
		seen[r] = struct{}{}
		out = append(out, r)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].TenantID != out[j].TenantID {
			return out[i].TenantID < out[j].TenantID
		}
		if out[i].SourceID != out[j].SourceID {
			return out[i].SourceID < out[j].SourceID
		}
		return out[i].Version < out[j].Version
	})
	return out
}
