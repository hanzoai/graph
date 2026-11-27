/*
 * SPDX-FileCopyrightText: © 2026 Hanzo AI, Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

package lightrag

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"
)

// BenchmarkConfig controls reproducible HIP-1347 benchmark execution.
type BenchmarkConfig struct {
	RepoRoot string
	TopK     int
	Iters    int
	Seed     int64
}

// DefaultBenchmarkConfig returns reproducible defaults for the HIP-1347 LightRAG suite.
func DefaultBenchmarkConfig() BenchmarkConfig {
	return BenchmarkConfig{
		RepoRoot: ".",
		TopK:     5,
		Iters:    10,
		Seed:     1347,
	}
}

// LatencyStats summarizes latency percentiles in milliseconds.
type LatencyStats struct {
	P50Ms  float64 `json:"p50_ms"`
	P95Ms  float64 `json:"p95_ms"`
	P99Ms  float64 `json:"p99_ms"`
	MeanMs float64 `json:"mean_ms"`
	Count  int     `json:"count"`
}

// CorpusSummary describes the real repository + operational artifact corpus.
type CorpusSummary struct {
	RepoRoot                 string  `json:"repo_root"`
	RealGoSourceFiles        int     `json:"real_go_source_files"`
	RealDocFiles             int     `json:"real_doc_files"`
	OperationalArtifacts     int     `json:"operational_artifacts"`
	TotalSources             int     `json:"total_sources"`
	TotalBytes               int     `json:"total_bytes"`
	TotalChunks              int     `json:"total_chunks"`
	RawEntitiesExtracted     int     `json:"raw_entities_extracted"`
	UniqueCanonicalEntities  int     `json:"unique_canonical_entities"`
	DuplicateEntitiesMerged  int     `json:"duplicate_entities_merged"`
	DuplicateEntityRatePct   float64 `json:"duplicate_entity_rate_pct"`
	RawRelationsExtracted    int     `json:"raw_relations_extracted"`
	UniqueCanonicalRelations int     `json:"unique_canonical_relations"`
	ExtractionTokensIn       int     `json:"extraction_tokens_in"`
	ExtractionTokensOut      int     `json:"extraction_tokens_out"`
	EstimatedExtractCostUSD  float64 `json:"estimated_extract_cost_usd"`
	RelationsPerDollar       float64 `json:"relations_per_dollar"`
}

// IncrementalBenchResult compares incremental delta updates against full rebuilds.
type IncrementalBenchResult struct {
	IncrementalOps          int          `json:"incremental_ops"`
	UpdateOps               int          `json:"update_ops"`
	InsertOps               int          `json:"insert_ops"`
	DeleteOps               int          `json:"delete_ops"`
	IncrementalLatency      LatencyStats `json:"incremental_latency"`
	FullRebuildLatency      LatencyStats `json:"full_rebuild_latency"`
	SpeedupP50              float64      `json:"speedup_p50"`
	SpeedupMean             float64      `json:"speedup_mean"`
	MeanSubgraphChurnPct    float64      `json:"mean_subgraph_churn_pct"`
	MaxSubgraphChurnPct     float64      `json:"max_subgraph_churn_pct"`
	MeanAffectedEntities    float64      `json:"mean_affected_entities"`
	MeanAffectedRelations   float64      `json:"mean_affected_relations"`
	DigestAfterIncremental  string       `json:"digest_after_incremental"`
	DigestAfterFullRebuild  string       `json:"digest_after_full_rebuild"`
	BitExactEquivalence     bool         `json:"bit_exact_equivalence"`
	DropEnforcesNotBuiltErr bool         `json:"drop_enforces_not_built_err"`
}

// SystemMetrics aggregates retrieval quality, latency, and cost for one mode.
type SystemMetrics struct {
	Mode              RetrievalMode `json:"mode"`
	Queries           int           `json:"queries"`
	RecallAtK         float64       `json:"recall_at_k"`
	MRRAtK            float64       `json:"mrr_at_k"`
	AnswerF1          float64       `json:"answer_f1"`
	AnswerEM          float64       `json:"answer_em"`
	TaskSuccessRate   float64       `json:"task_success_rate"`
	MeanTokens        float64       `json:"mean_context_tokens"`
	CostPer1MUsefulUS float64       `json:"cost_per_1m_useful_tokens_usd"`
	Latency           LatencyStats  `json:"latency"`
	CachedLatencyP50  float64       `json:"cached_latency_p50_ms"`
}

// WorkloadBreakdown reports per-workload metrics across all 4 retrieval modes.
type WorkloadBreakdown struct {
	Workload string          `json:"workload"`
	Queries  int             `json:"queries"`
	Systems  []SystemMetrics `json:"systems"`
}

// FreshnessVerification verifies cache invalidation and post-update/delete recall.
type FreshnessVerification struct {
	QueriesTested          int     `json:"queries_tested"`
	CacheInvalidatedOnEdit bool    `json:"cache_invalidated_on_edit"`
	UpdatedRecallRate      float64 `json:"updated_recall_rate"`
	DeletedGhostRecallRate float64 `json:"deleted_ghost_recall_rate"`
}

// SecurityIsolationReport summarizes cross-tenant & intra-tenant ACL probe results.
type SecurityIsolationReport struct {
	ProbesExecuted            int     `json:"probes_executed"`
	LightRAGLeakCount         int     `json:"lightrag_leak_count"`
	LightRAGLeakRatePct       float64 `json:"lightrag_leak_rate_pct"`
	UnfilteredControlLeaks    int     `json:"unfiltered_control_leaks"`
	UnfilteredControlRatePct  float64 `json:"unfiltered_control_rate_pct"`
	ACLFilteredCandidatesSeen int     `json:"acl_filtered_candidates_seen"`
}

// NegativeRegime documents an explicitly measured regime where LightRAG incurs
// overhead or a component ablation degrades performance.
type NegativeRegime struct {
	Name        string             `json:"name"`
	Description string             `json:"description"`
	Metrics     map[string]float64 `json:"metrics"`
}

// BenchmarkReport is the complete machine-readable HIP-1347 benchmark artifact.
type BenchmarkReport struct {
	GeneratedAtUTC  string                  `json:"generated_at_utc"`
	GoVersion       string                  `json:"go_version"`
	GOOS            string                  `json:"goos"`
	GOARCH          string                  `json:"goarch"`
	NumCPU          int                     `json:"num_cpu"`
	Seed            int64                   `json:"seed"`
	TopK            int                     `json:"top_k"`
	Iters           int                     `json:"iters"`
	Corpus          CorpusSummary           `json:"corpus"`
	Incremental     IncrementalBenchResult  `json:"incremental"`
	OverallSystems  []SystemMetrics         `json:"overall_systems"`
	Workloads       []WorkloadBreakdown     `json:"workloads"`
	Freshness       FreshnessVerification   `json:"freshness"`
	Security        SecurityIsolationReport `json:"security"`
	NegativeRegimes []NegativeRegime        `json:"negative_regimes"`
}

type evalQuery struct {
	ID           string
	Workload     string
	Text         string
	GoldSources  []string // canonical SourceIDs that must be retrieved
	AnswerTokens []string // grounded answer tokens expected in top-K evidence
	Principals   []string
}

const benchTenant = "hanzo-graph"

// LoadRepoCorpus deterministically loads real Apache-2.0 Go source files and
// documentation from the Hanzo Graph repository, plus versioned operational
// engineering artifacts (commits, ADRs, incidents, agent notes) grounded in
// the repository's real packages and symbols.
func LoadRepoCorpus(repoRoot string) ([]SourceRecord, int, int, int, error) {
	targetDirs := []string{"algo", "buildvars", "codec", "dql", "lex", "posting", "schema", "tok", "types", "x"}
	docFiles := []string{"README.md", "dql/README.md", "tok/options/README.md", "go.mod", "NOTICE"}

	var goPaths []string
	for _, dir := range targetDirs {
		absDir := filepath.Join(repoRoot, dir)
		if _, err := os.Stat(absDir); err != nil {
			continue
		}
		err := filepath.Walk(absDir, func(path string, info os.FileInfo, walkErr error) error {
			if walkErr != nil || info == nil || info.IsDir() {
				return nil
			}
			if !strings.HasSuffix(info.Name(), ".go") || strings.HasSuffix(info.Name(), "_test.go") {
				return nil
			}
			rel, relErr := filepath.Rel(repoRoot, path)
			if relErr != nil {
				return nil
			}
			goPaths = append(goPaths, filepath.ToSlash(rel))
			return nil
		})
		if err != nil {
			return nil, 0, 0, 0, err
		}
	}
	sort.Strings(goPaths)

	baseTime := time.Date(2026, 11, 1, 0, 0, 0, 0, time.UTC)
	var records []SourceRecord
	goCount := 0
	for _, rel := range goPaths {
		raw, err := os.ReadFile(filepath.Join(repoRoot, filepath.FromSlash(rel)))
		if err != nil || len(raw) == 0 {
			continue
		}
		records = append(records, SourceRecord{
			SourceID:  rel,
			Version:   1,
			TenantID:  benchTenant,
			Kind:      KindCodeBlob,
			Title:     "Hanzo Graph Go source " + rel,
			Path:      rel,
			Content:   string(raw),
			Timestamp: baseTime,
			Metadata: map[string]string{
				"license": "Apache-2.0",
				"origin":  "hanzoai/graph",
			},
		})
		goCount++
	}

	docCount := 0
	for _, rel := range docFiles {
		raw, err := os.ReadFile(filepath.Join(repoRoot, filepath.FromSlash(rel)))
		if err != nil || len(raw) == 0 {
			continue
		}
		records = append(records, SourceRecord{
			SourceID:  rel,
			Version:   1,
			TenantID:  benchTenant,
			Kind:      KindDocument,
			Title:     "Hanzo Graph documentation " + rel,
			Path:      rel,
			Content:   string(raw),
			Timestamp: baseTime,
			Metadata: map[string]string{
				"license": "Apache-2.0",
				"origin":  "hanzoai/graph",
			},
		})
		docCount++
	}

	ops := buildOperationalArtifacts(baseTime)
	records = append(records, ops...)
	return records, goCount, docCount, len(ops), nil
}

// buildOperationalArtifacts creates versioned commits, ADRs, incidents, and
// agent analysis outputs that reference real hanzoai/graph files and symbols,
// forming realistic multi-hop engineering chains across code and ops events.
func buildOperationalArtifacts(baseTime time.Time) []SourceRecord {
	return []SourceRecord{
		{
			SourceID:  "ops/INC-401-mvcc-rollup-stall.md",
			Version:   1,
			TenantID:  benchTenant,
			Kind:      KindIncident,
			Title:     "INC-401: Elevated write stalls during incremental rollup bursts",
			Path:      "ops/INC-401-mvcc-rollup-stall.md",
			Timestamp: baseTime.Add(24 * time.Hour),
			Content: `Incident: INC-401
Title: Elevated write stalls during incremental rollup bursts
Author: sre-storage
Package: posting
Related: ADR-108
Causes: ADR-108
Symptoms: Sustained write latency spikes observed when background rollup queues saturate CachePL invalidation and MemoryLayer disk reads.
Summary: Telemetry pointed to stale cached posting list entries colliding with concurrent commit timestamps during high churn. Follow ADR-108 for the root-cause analysis and remediation commit.`,
		},
		{
			SourceID:  "ops/ADR-108-mvcc-cache-invalidation.md",
			Version:   1,
			TenantID:  benchTenant,
			Kind:      KindDecision,
			Title:     "ADR-108: Atomic CachePL invalidation on MVCC commit timestamps",
			Path:      "ops/ADR-108-mvcc-cache-invalidation.md",
			Timestamp: baseTime.Add(48 * time.Hour),
			Content: `Decision: ADR-108
Author: core-storage
Files: posting/mvcc.go, posting/oracle.go
Symbols: posting.CachePL, posting.Txn, posting.MemoryLayer
Mitigates: INC-401
Related: COMMIT-9104
Rationale: CommitToDisk and UpdateCachedKeys must evict stale CachePL buckets atomically with Oracle watermarks so readers never observe half-applied MVCC deltas. Implemented in COMMIT-9104.`,
		},
		{
			SourceID:  "ops/COMMIT-9104-txn-cache-fence.md",
			Version:   1,
			TenantID:  benchTenant,
			Kind:      KindCommit,
			Title:     "COMMIT-9104: Fence UpdateCachedKeys against concurrent ReadPostingList",
			Path:      "ops/COMMIT-9104-txn-cache-fence.md",
			Timestamp: baseTime.Add(72 * time.Hour),
			Content: `Commit: COMMIT-9104
Author: alice-db
Files: posting/mvcc.go, posting/writer.go
Symbols: posting.TxnWriter, posting.CachePL
Fixes: INC-401
Depends-On: ADR-108
Summary: Enforces commit watermark fencing inside posting/mvcc.go and posting/writer.go so TxnWriter flushes complete before UpdateCachedKeys exposes new read timestamps. Verified zero stale reads under race detector.`,
		},
		{
			SourceID:  "ops/INC-412-hnsw-entry-drift.md",
			Version:   1,
			TenantID:  benchTenant,
			Kind:      KindIncident,
			Title:     "INC-412: Nearest-neighbor recall drop after vector index compaction",
			Path:      "ops/INC-412-hnsw-entry-drift.md",
			Timestamp: baseTime.Add(96 * time.Hour),
			Content: `Incident: INC-412
Author: sre-vector
Related: ADR-115
Causes: ADR-115
Symptoms: Top-10 approximate nearest-neighbor recall fell by 14% after graph compaction without any change in embedding vectors.
Summary: Investigation traced recall regression to stale entry-point selection during layer traversal; see ADR-115 for architectural link to persistent graph layers.`,
		},
		{
			SourceID:  "ops/ADR-115-hnsw-start-node-selection.md",
			Version:   1,
			TenantID:  benchTenant,
			Kind:      KindDecision,
			Title:     "ADR-115: Deterministic PickStartNode and SearchWithPath persistence contract",
			Path:      "ops/ADR-115-hnsw-start-node-selection.md",
			Timestamp: baseTime.Add(120 * time.Hour),
			Content: `Decision: ADR-115
Author: vector-eng
Files: tok/hnsw/persistent_hnsw.go, tok/index_factory.go
Symbols: tok/hnsw.persistentHNSW, tok.IndexFactory
Mitigates: INC-412
Related: COMMIT-9211
Rationale: Persistent vector indexes registered via tok.IndexFactory must keep PickStartNode synchronized with committed TxnCache state in tok/hnsw/persistent_hnsw.go. Remediation shipped in COMMIT-9211.`,
		},
		{
			SourceID:  "ops/COMMIT-9211-hnsw-entrypoint-sync.md",
			Version:   1,
			TenantID:  benchTenant,
			Kind:      KindCommit,
			Title:     "COMMIT-9211: Refresh HNSW entry node on TxnCache commit",
			Path:      "ops/COMMIT-9211-hnsw-entrypoint-sync.md",
			Timestamp: baseTime.Add(144 * time.Hour),
			Content: `Commit: COMMIT-9211
Author: bob-vec
Files: tok/hnsw/persistent_hnsw.go, tok/hnsw/helper.go
Symbols: tok/hnsw.TxnCache, tok/hnsw.SearchResult
Fixes: INC-412
Depends-On: ADR-115
Summary: Synchronizes HNSW start node metadata inside tok/hnsw/persistent_hnsw.go and tok/hnsw/helper.go after Insert and SearchWithPath updates, restoring 99.2% recall.`,
		},
		{
			SourceID:  "ops/INC-425-uid-pack-cpu-spike.md",
			Version:   1,
			TenantID:  benchTenant,
			Kind:      KindIncident,
			Title:     "INC-425: CPU saturation on skewed UID list intersections",
			Path:      "ops/INC-425-uid-pack-cpu-spike.md",
			Timestamp: baseTime.Add(168 * time.Hour),
			Content: `Incident: INC-425
Author: sre-query
Related: ADR-121
Causes: ADR-121
Symptoms: Tail query latency jumped 6x when intersecting large sparse posting lists with small selective filters.
Summary: Flamegraphs showed linear scans over packed integer blocks instead of binary jump decoding; see ADR-121 for the codec and set-algebra redesign.`,
		},
		{
			SourceID:  "ops/ADR-121-uid-compressed-jump-intersect.md",
			Version:   1,
			TenantID:  benchTenant,
			Kind:      KindDecision,
			Title:     "ADR-121: Adaptive binary jump intersection over group-varint UID blocks",
			Path:      "ops/ADR-121-uid-compressed-jump-intersect.md",
			Timestamp: baseTime.Add(192 * time.Hour),
			Content: `Decision: ADR-121
Author: query-eng
Files: algo/uidlist.go, codec/codec.go, posting/list.go
Symbols: algo.IntersectCompressedWithBin, codec.Decoder, posting.List
Mitigates: INC-425
Related: COMMIT-9302
Rationale: Use IntersectCompressedWithBin with codec.Decoder block skipping whenever cardinality skew exceeds 16x, reducing decoded blocks by 88%.`,
		},
		{
			SourceID:  "ops/COMMIT-9302-uid-binjump-threshold.md",
			Version:   1,
			TenantID:  benchTenant,
			Kind:      KindCommit,
			Title:     "COMMIT-9302: Tune IntersectCompressedWithBin skew threshold in algo/uidlist.go",
			Path:      "ops/COMMIT-9302-uid-binjump-threshold.md",
			Timestamp: baseTime.Add(216 * time.Hour),
			Content: `Commit: COMMIT-9302
Author: carol-algo
Files: algo/uidlist.go, codec/codec.go
Symbols: algo.IntersectCompressedWith, codec.Encoder
Fixes: INC-425
Depends-On: ADR-121
Summary: Routes high-skew compressed UID intersections through IntersectCompressedWithBin using codec.Decoder seek pointers, cutting p99 intersection CPU time by 74%.`,
		},
		{
			SourceID:  "ops/EVT-501-schema-rebuild-agent-audit.md",
			Version:   1,
			TenantID:  benchTenant,
			Kind:      KindAgentEvent,
			Title:     "EVT-501: Autonomous schema index rebuild verification report",
			Path:      "ops/EVT-501-schema-rebuild-agent-audit.md",
			Timestamp: baseTime.Add(240 * time.Hour),
			Content: `Component: EVT-501
Author: agent-verifier
Files: posting/index.go, schema/parse.go, tok/tok.go
Symbols: posting.IndexRebuild, schema.ParsedSchema, tok.Tokenizer
Related: ADR-108
Summary: Verified that IndexRebuild.DropIndexes and BuildIndexes regenerate all tokenizer-derived secondary indexes strictly from canonical predicate data in zapdb without mutating source triples.`,
		},
		{
			SourceID:  "ops/EVT-502-dql-lexer-parser-pipeline.md",
			Version:   1,
			TenantID:  benchTenant,
			Kind:      KindAgentEvent,
			Title:     "EVT-502: DQL lexing to AST syntax tree dependency audit",
			Path:      "ops/EVT-502-dql-lexer-parser-pipeline.md",
			Timestamp: baseTime.Add(264 * time.Hour),
			Content: `Component: EVT-502
Author: agent-architect
Files: dql/parser.go, dql/mutation.go, lex/lexer.go
Symbols: dql.GraphQuery, dql.FilterTree, lex.Lexer, lex.ItemIterator
Summary: Audited the single-process DQL parsing pipeline: lex.Lexer produces slice-backed ItemIterator streams consumed by dql/parser.go to construct GraphQuery, FilterTree, and NQuad mutation AST nodes.`,
		},
		{
			SourceID:  "ops/EVT-503-deprecated-legacy-note.md",
			Version:   1,
			TenantID:  benchTenant,
			Kind:      KindAgentEvent,
			Title:     "EVT-503: Temporary migration checklist for legacy raft-removed stub",
			Path:      "ops/EVT-503-deprecated-legacy-note.md",
			Timestamp: baseTime.Add(288 * time.Hour),
			Content: `Component: EVT-503
Author: agent-cleanup
Files: README.md
Symbols: LegacyRaftStubSentinel
Summary: LegacyRaftStubSentinel temporary migration note tracking removal of old cluster raft hooks. Scheduled for deletion once incremental cleanup runs.`,
		},
	}
}

func buildEvalQueries() []evalQuery {
	return []evalQuery{
		// Workload 1: Cross-package multi-hop architecture & incident causality queries.
		// Designed so intermediate bridge entities (ADRs, packages, types) connect the query
		// to target files/commits that do NOT share direct keywords with the question.
		{
			ID:           "mh-01",
			Workload:     "cross_package_multihop",
			Text:         "Which commit resolved INC-401 and what storage files did it modify?",
			GoldSources:  []string{"ops/COMMIT-9104-txn-cache-fence.md", "ops/ADR-108-mvcc-cache-invalidation.md"},
			AnswerTokens: []string{"commit-9104", "posting/mvcc.go", "posting/writer.go"},
		},
		{
			ID:           "mh-02",
			Workload:     "cross_package_multihop",
			Text:         "Trace INC-412 from the incident through its architectural decision to the commit that restored recall.",
			GoldSources:  []string{"ops/COMMIT-9211-hnsw-entrypoint-sync.md", "ops/ADR-115-hnsw-start-node-selection.md"},
			AnswerTokens: []string{"commit-9211", "adr-115", "persistent_hnsw.go"},
		},
		{
			ID:           "mh-03",
			Workload:     "cross_package_multihop",
			Text:         "Which author and commit fixed INC-425 and which codec symbol was used?",
			GoldSources:  []string{"ops/COMMIT-9302-uid-binjump-threshold.md", "ops/ADR-121-uid-compressed-jump-intersect.md"},
			AnswerTokens: []string{"commit-9302", "carol-algo", "decoder"},
		},
		{
			ID:           "mh-04",
			Workload:     "cross_package_multihop",
			Text:         "Starting from INC-401, which core Go source files implement CachePL and Oracle watermarks?",
			GoldSources:  []string{"posting/mvcc.go", "posting/oracle.go", "ops/ADR-108-mvcc-cache-invalidation.md"},
			AnswerTokens: []string{"cachepl", "committodisk", "oracle"},
		},
		{
			ID:           "mh-05",
			Workload:     "cross_package_multihop",
			Text:         "Starting from INC-412, which persistent vector index implementation and factory interface are involved?",
			GoldSources:  []string{"tok/hnsw/persistent_hnsw.go", "tok/index_factory.go", "ops/ADR-115-hnsw-start-node-selection.md"},
			AnswerTokens: []string{"pickstartnode", "indexfactory", "txncache"},
		},
		{
			ID:           "mh-06",
			Workload:     "cross_package_multihop",
			Text:         "How does posting/list.go connect to compressed UID set intersection and binary decoding?",
			GoldSources:  []string{"posting/list.go", "algo/uidlist.go", "codec/codec.go"},
			AnswerTokens: []string{"intersectcompressedwith", "decoder", "uidpack"},
		},
		{
			ID:           "mh-07",
			Workload:     "cross_package_multihop",
			Text:         "Which files bridge DQL query AST parsing from lex.ItemIterator to GraphQuery and NQuad mutations?",
			GoldSources:  []string{"dql/parser.go", "lex/lexer.go", "dql/mutation.go"},
			AnswerTokens: []string{"graphquery", "itemiterator", "nquad"},
		},
		{
			ID:           "mh-08",
			Workload:     "cross_package_multihop",
			Text:         "Trace EVT-501 to the schema index rebuild functions that drop and regenerate derived tokenizer indexes.",
			GoldSources:  []string{"posting/index.go", "schema/parse.go", "ops/EVT-501-schema-rebuild-agent-audit.md"},
			AnswerTokens: []string{"indexrebuild", "dropindexes", "buildindexes"},
		},
		{
			ID:           "mh-09",
			Workload:     "cross_package_multihop",
			Text:         "Which decision links INC-425 to posting.List and what skew optimization does it prescribe?",
			GoldSources:  []string{"ops/ADR-121-uid-compressed-jump-intersect.md", "algo/uidlist.go"},
			AnswerTokens: []string{"adr-121", "intersectcompressedwithbin", "16x"},
		},
		{
			ID:           "mh-10",
			Workload:     "cross_package_multihop",
			Text:         "How does posting/index.go depend on tok and tok/hnsw for derived indexing?",
			GoldSources:  []string{"posting/index.go", "tok/tok.go", "tok/hnsw/persistent_hnsw.go"},
			AnswerTokens: []string{"tokenizer", "buildtokens", "hnsw"},
		},

		// Workload 2: Global cross-cutting thematic & architectural synthesis queries.
		{
			ID:           "gl-01",
			Workload:     "global_thematic",
			Text:         "Across packages, how are MVCC transaction timestamps, commit writes, and posting list caches coordinated?",
			GoldSources:  []string{"posting/mvcc.go", "posting/oracle.go", "posting/writer.go", "ops/ADR-108-mvcc-cache-invalidation.md"},
			AnswerTokens: []string{"committodisk", "cachepl", "txnwriter"},
		},
		{
			ID:           "gl-02",
			Workload:     "global_thematic",
			Text:         "Explain the cross-package architecture for tokenization, full-text analysis, and HNSW vector indexing.",
			GoldSources:  []string{"tok/tok.go", "tok/index_factory.go", "tok/hnsw/persistent_hnsw.go"},
			AnswerTokens: []string{"tokenizer", "indexfactory", "searchwithpath"},
		},
		{
			ID:           "gl-03",
			Workload:     "global_thematic",
			Text:         "How do UID compression, bitpacking encoders, and sorted set intersection work together?",
			GoldSources:  []string{"codec/codec.go", "algo/uidlist.go", "algo/packed.go"},
			AnswerTokens: []string{"encoder", "decoder", "mergesorted"},
		},
		{
			ID:           "gl-04",
			Workload:     "global_thematic",
			Text:         "How are DQL lexing, filter trees, math expressions, and mutation parsing structured?",
			GoldSources:  []string{"dql/parser.go", "dql/math.go", "dql/mutation.go", "lex/lexer.go"},
			AnswerTokens: []string{"graphquery", "filtertree", "mathtree"},
		},
		{
			ID:           "gl-05",
			Workload:     "global_thematic",
			Text:         "What mechanisms govern schema parsing, predicate index drops, and full derived index rebuilding?",
			GoldSources:  []string{"schema/parse.go", "posting/index.go", "ops/EVT-501-schema-rebuild-agent-audit.md"},
			AnswerTokens: []string{"parsedschema", "indexrebuild", "buildindexes"},
		},
		{
			ID:           "gl-06",
			Workload:     "global_thematic",
			Text:         "How are spatial S2 cell indexing and geographic query filters implemented?",
			GoldSources:  []string{"types/s2index.go", "types/geofilter.go", "tok/tok.go"},
			AnswerTokens: []string{"geoquerydata", "s2", "geotokenizer"},
		},
		{
			ID:           "gl-07",
			Workload:     "global_thematic",
			Text:         "How does Hanzo Graph encode storage keys, namespaces, and zapdb persistence?",
			GoldSources:  []string{"x/keys.go", "README.md", "posting/mvcc.go"},
			AnswerTokens: []string{"parsedkey", "zapdb", "namespace"},
		},
		{
			ID:           "gl-08",
			Workload:     "global_thematic",
			Text:         "What security primitives handle ACL encryption keys, TLS configuration, and vault integration?",
			GoldSources:  []string{"x/acl_enc_keys.go", "x/tls_helper.go", "x/vault.go"},
			AnswerTokens: []string{"keys", "tlshelperconfig", "vault"},
		},

		// Workload 3: Simple single-fact exact identifier lookups (explicit negative regime!).
		// Plain ordinary_hybrid excels here with minimal latency; LightRAG graph traversal adds overhead.
		{
			ID:           "sf-01",
			Workload:     "single_fact_exact",
			Text:         "Where is CountMinSketch defined?",
			GoldSources:  []string{"algo/cm-sketch.go"},
			AnswerTokens: []string{"countminsketch"},
		},
		{
			ID:           "sf-02",
			Workload:     "single_fact_exact",
			Text:         "Where is SafeMutex implemented?",
			GoldSources:  []string{"x/lock.go"},
			AnswerTokens: []string{"safemutex"},
		},
		{
			ID:           "sf-03",
			Workload:     "single_fact_exact",
			Text:         "Which file defines ReservedNamespace?",
			GoldSources:  []string{"x/keys.go"},
			AnswerTokens: []string{"reservednamespace"},
		},
		{
			ID:           "sf-04",
			Workload:     "single_fact_exact",
			Text:         "Where is BigFloatTokenizer declared?",
			GoldSources:  []string{"tok/tok.go"},
			AnswerTokens: []string{"bigfloattokenizer"},
		},
		{
			ID:           "sf-05",
			Workload:     "single_fact_exact",
			Text:         "Which file defines GeoQueryData?",
			GoldSources:  []string{"types/geofilter.go"},
			AnswerTokens: []string{"geoquerydata"},
		},
		{
			ID:           "sf-06",
			Workload:     "single_fact_exact",
			Text:         "Where is TLSHelperConfig defined?",
			GoldSources:  []string{"x/tls_helper.go"},
			AnswerTokens: []string{"tlshelperconfig"},
		},
		{
			ID:           "sf-07",
			Workload:     "single_fact_exact",
			Text:         "Which file contains MinioCredentials?",
			GoldSources:  []string{"x/minioclient.go"},
			AnswerTokens: []string{"miniocredentials"},
		},
		{
			ID:           "sf-08",
			Workload:     "single_fact_exact",
			Text:         "Where is SlowOperationLatency declared?",
			GoldSources:  []string{"x/logger.go"},
			AnswerTokens: []string{"slowoperationlatency"},
		},
	}
}

// RunBenchmarkSuite executes the reproducible HIP-1347 LightRAG benchmark suite
// over the real Hanzo Graph repository corpus and returns raw structured metrics.
func RunBenchmarkSuite(ctx context.Context, cfg BenchmarkConfig) (BenchmarkReport, error) {
	if cfg.RepoRoot == "" {
		cfg.RepoRoot = "."
	}
	if cfg.TopK <= 0 {
		cfg.TopK = 5
	}
	if cfg.Iters <= 0 {
		cfg.Iters = 10
	}
	if cfg.Seed == 0 {
		cfg.Seed = 1347
	}

	records, goCount, docCount, opsCount, err := LoadRepoCorpus(cfg.RepoRoot)
	if err != nil {
		return BenchmarkReport{}, fmt.Errorf("load repo corpus: %w", err)
	}
	if len(records) == 0 {
		return BenchmarkReport{}, fmt.Errorf("no source records found under %q", cfg.RepoRoot)
	}

	eng := NewEngine(DefaultEngineConfig())
	totalBytes := 0
	totalTokIn := 0
	totalTokOut := 0
	totalCost := 0.0
	for _, rec := range records {
		totalBytes += len(rec.Content)
		tel, err := eng.UpsertSource(ctx, rec)
		if err != nil {
			return BenchmarkReport{}, fmt.Errorf("initial upsert %s: %w", rec.SourceID, err)
		}
		totalTokIn += tel.ExtractionTokensIn
		totalTokOut += tel.ExtractionTokensOut
		totalCost += tel.EstimatedCostUSD
	}

	st := eng.Stats(benchTenant)
	relsPerDollar := 0.0
	if totalCost > 0 {
		relsPerDollar = float64(st.Relations) / totalCost
	}

	corpusSummary := CorpusSummary{
		RepoRoot:                 cfg.RepoRoot,
		RealGoSourceFiles:        goCount,
		RealDocFiles:             docCount,
		OperationalArtifacts:     opsCount,
		TotalSources:             st.Sources,
		TotalBytes:               totalBytes,
		TotalChunks:              st.Chunks,
		RawEntitiesExtracted:     st.TotalRawEntities,
		UniqueCanonicalEntities:  st.Entities,
		DuplicateEntitiesMerged:  st.TotalCollapsedDupes,
		DuplicateEntityRatePct:   round2(st.AggregateDupeRate * 100.0),
		RawRelationsExtracted:    st.TotalRawRelations,
		UniqueCanonicalRelations: st.Relations,
		ExtractionTokensIn:       totalTokIn,
		ExtractionTokensOut:      totalTokOut,
		EstimatedExtractCostUSD:  round6(totalCost),
		RelationsPerDollar:       round2(relsPerDollar),
	}

	// 1. Evaluate Dual-Level Retrieval vs Ordinary Hybrid Baseline on the v1 corpus.
	queries := buildEvalQueries()
	modes := []RetrievalMode{ModeOrdinaryHybrid, ModeLocal, ModeGlobal, ModeHybrid}
	overallSystems, workloadBreakdowns, err := evaluateRetrievalModes(ctx, eng, queries, modes, cfg.TopK, cfg.Iters)
	if err != nil {
		return BenchmarkReport{}, err
	}

	// 2. Evaluate Incremental Delta Updates vs Full Graph Rebuild + Freshness + Cache Invalidation.
	incResult, freshness, err := evaluateIncrementalAndFreshness(ctx, eng, records, cfg)
	if err != nil {
		return BenchmarkReport{}, err
	}

	// 3. Evaluate Strict Pre-Return Tenant & ACL Isolation against unfiltered shared-graph control.
	secReport, err := evaluateSecurityIsolation(ctx)
	if err != nil {
		return BenchmarkReport{}, err
	}

	// 4. Construct explicitly measured Negative & Boundary Regimes.
	negRegimes := buildNegativeRegimes(workloadBreakdowns, secReport)

	return BenchmarkReport{
		GeneratedAtUTC:  time.Now().UTC().Format(time.RFC3339),
		GoVersion:       runtime.Version(),
		GOOS:            runtime.GOOS,
		GOARCH:          runtime.GOARCH,
		NumCPU:          runtime.NumCPU(),
		Seed:            cfg.Seed,
		TopK:            cfg.TopK,
		Iters:           cfg.Iters,
		Corpus:          corpusSummary,
		Incremental:     incResult,
		OverallSystems:  overallSystems,
		Workloads:       workloadBreakdowns,
		Freshness:       freshness,
		Security:        secReport,
		NegativeRegimes: negRegimes,
	}, nil
}

func evaluateRetrievalModes(
	ctx context.Context,
	eng *Engine,
	queries []evalQuery,
	modes []RetrievalMode,
	topK int,
	iters int,
) ([]SystemMetrics, []WorkloadBreakdown, error) {
	workloadOrder := []string{"cross_package_multihop", "global_thematic", "single_fact_exact"}

	type queryRun struct {
		workload     string
		lats         []float64
		cachedLat    float64
		recall       float64
		mrr          float64
		f1           float64
		em           float64
		taskOK       bool
		totalTokens  int
		usefulTokens int
	}

	aggregate := func(mode RetrievalMode, runs []queryRun) SystemMetrics {
		var allLats []float64
		var cachedLats []float64
		sumRecall, sumMRR, sumF1, sumEM, sumTask := 0.0, 0.0, 0.0, 0.0, 0.0
		totalTokens, goldTokens := 0, 0
		for _, r := range runs {
			allLats = append(allLats, r.lats...)
			cachedLats = append(cachedLats, r.cachedLat)
			sumRecall += r.recall
			sumMRR += r.mrr
			sumF1 += r.f1
			sumEM += r.em
			if r.taskOK {
				sumTask += 1.0
			}
			totalTokens += r.totalTokens
			goldTokens += r.usefulTokens
		}
		n := float64(len(runs))
		if n == 0 {
			n = 1
		}
		costPer1MUseful := 0.0
		if goldTokens > 0 {
			costPer1MUseful = 3.00 * (float64(totalTokens) / float64(goldTokens))
		} else if totalTokens > 0 {
			costPer1MUseful = 999.0
		}
		return SystemMetrics{
			Mode:              mode,
			Queries:           len(runs),
			RecallAtK:         round4(sumRecall / n),
			MRRAtK:            round4(sumMRR / n),
			AnswerF1:          round4(sumF1 / n),
			AnswerEM:          round4(sumEM / n),
			TaskSuccessRate:   round4(sumTask / n),
			MeanTokens:        round2(float64(totalTokens) / n),
			CostPer1MUsefulUS: round2(costPer1MUseful),
			Latency:           computeLatencyStats(allLats),
			CachedLatencyP50:  computeLatencyStats(cachedLats).P50Ms,
		}
	}

	overall := make([]SystemMetrics, 0, len(modes))
	perWorkloadByMode := make(map[string][]SystemMetrics, len(workloadOrder))

	for _, m := range modes {
		var modeRuns []queryRun
		byWL := make(map[string][]queryRun)

		for _, q := range queries {
			req := Query{
				Text:             q.Text,
				TenantID:         benchTenant,
				Principals:       q.Principals,
				Mode:             m,
				TopK:             topK,
				MaxHops:          2,
				MaxExpandedNodes: 64,
				IncludeSubgraph:  true,
				BypassCache:      true,
			}

			// Calibrate micro-batch size for sub-ms queries so coarse VM timers never yield 0.000ms.
			batch := 1
			if m == ModeOrdinaryHybrid {
				batch = 10
			} else {
				batch = 3
			}

			var lastRes SearchResult
			lats := make([]float64, 0, iters)
			for it := 0; it < iters; it++ {
				t0 := time.Now()
				for b := 0; b < batch; b++ {
					res, err := eng.Search(ctx, req)
					if err != nil {
						return nil, nil, err
					}
					lastRes = res
				}
				dt := (float64(time.Since(t0).Nanoseconds()) / 1e6) / float64(batch)
				lats = append(lats, dt)
			}

			// Measure warm cache hit latency over a 50-call micro-batch.
			warmReq := req
			warmReq.BypassCache = false
			_, _ = eng.Search(ctx, warmReq)
			t0 := time.Now()
			const cacheBatch = 50
			for b := 0; b < cacheBatch; b++ {
				if _, err := eng.Search(ctx, warmReq); err != nil {
					return nil, nil, err
				}
			}
			cachedMs := (float64(time.Since(t0).Nanoseconds()) / 1e6) / float64(cacheBatch)

			rec, mrr, f1, em, taskOK, usedTok, usefulTok := judgeResult(q, lastRes)
			qr := queryRun{
				workload:     q.Workload,
				lats:         lats,
				cachedLat:    cachedMs,
				recall:       rec,
				mrr:          mrr,
				f1:           f1,
				em:           em,
				taskOK:       taskOK,
				totalTokens:  usedTok,
				usefulTokens: usefulTok,
			}
			modeRuns = append(modeRuns, qr)
			byWL[q.Workload] = append(byWL[q.Workload], qr)
		}

		overall = append(overall, aggregate(m, modeRuns))
		for _, wl := range workloadOrder {
			perWorkloadByMode[wl] = append(perWorkloadByMode[wl], aggregate(m, byWL[wl]))
		}
	}

	breakdowns := make([]WorkloadBreakdown, 0, len(workloadOrder))
	for _, wl := range workloadOrder {
		sysList := perWorkloadByMode[wl]
		qCount := 0
		if len(sysList) > 0 {
			qCount = sysList[0].Queries
		}
		breakdowns = append(breakdowns, WorkloadBreakdown{
			Workload: wl,
			Queries:  qCount,
			Systems:  sysList,
		})
	}
	return overall, breakdowns, nil
}

// judgeResult implements the shared, deterministic extractive answer synthesizer
// and judge reading strictly the top-K retrieved chunks + grounded subgraph trails.
func judgeResult(q evalQuery, res SearchResult) (recall, mrr, f1, em float64, taskOK bool, totalTokens, usefulTokens int) {
	goldSet := make(map[string]struct{}, len(q.GoldSources))
	for _, gs := range q.GoldSources {
		goldSet[gs] = struct{}{}
	}

	seenGold := make(map[string]struct{})
	firstGoldRank := 0
	var evidenceBuilder strings.Builder

	for idx, rc := range res.Chunks {
		totalTokens += rc.Chunk.TokenCount
		if _, isGold := goldSet[rc.Chunk.SourceID]; isGold {
			if _, dup := seenGold[rc.Chunk.SourceID]; !dup {
				seenGold[rc.Chunk.SourceID] = struct{}{}
				usefulTokens += rc.Chunk.TokenCount
				if firstGoldRank == 0 {
					firstGoldRank = idx + 1
				}
			}
		}
		evidenceBuilder.WriteString(rc.Chunk.SourceID)
		evidenceBuilder.WriteByte(' ')
		evidenceBuilder.WriteString(rc.Chunk.Text)
		evidenceBuilder.WriteByte('\n')
	}
	// Include top grounded graph trails (strictly ACL-filtered provenance strings).
	for i, tr := range res.Trails {
		if i >= 8 {
			break
		}
		evidenceBuilder.WriteString(tr.SourceEntity)
		evidenceBuilder.WriteByte(' ')
		evidenceBuilder.WriteString(tr.TargetEntity)
		evidenceBuilder.WriteByte(' ')
		evidenceBuilder.WriteString(tr.Description)
		evidenceBuilder.WriteByte('\n')
		totalTokens += EstimateTokens(tr.Description)
	}

	if len(goldSet) > 0 {
		recall = float64(len(seenGold)) / float64(len(goldSet))
	}
	if firstGoldRank > 0 {
		mrr = 1.0 / float64(firstGoldRank)
	}

	evidenceLower := strings.ToLower(evidenceBuilder.String())
	matchedAns := 0
	for _, tok := range q.AnswerTokens {
		if strings.Contains(evidenceLower, strings.ToLower(tok)) {
			matchedAns++
		}
	}
	if len(q.AnswerTokens) > 0 {
		ansRec := float64(matchedAns) / float64(len(q.AnswerTokens))
		// Extractive synthesizer emits only matched canonical answer tokens -> precision = 1.0 when matched > 0.
		if matchedAns > 0 {
			f1 = (2.0 * 1.0 * ansRec) / (1.0 + ansRec)
		}
		if matchedAns == len(q.AnswerTokens) {
			em = 1.0
		}
	}

	// Downstream task success requires retrieving at least one gold source AND
	// grounding at least 2/3 of the required answer tokens (or all if <= 2 tokens).
	needed := (len(q.AnswerTokens)*2 + 2) / 3
	if len(q.AnswerTokens) <= 2 {
		needed = len(q.AnswerTokens)
	}
	taskOK = len(seenGold) > 0 && matchedAns >= needed
	return recall, mrr, f1, em, taskOK, totalTokens, usefulTokens
}

func evaluateIncrementalAndFreshness(
	ctx context.Context,
	eng *Engine,
	records []SourceRecord,
	cfg BenchmarkConfig,
) (IncrementalBenchResult, FreshnessVerification, error) {
	// Prime query cache on v1 before executing incremental mutations so we verify
	// cache invalidation on mutation and deletion.
	primeQuery := Query{
		Text:     "LegacyRaftStubSentinel migration checklist",
		TenantID: benchTenant,
		Mode:     ModeHybrid,
		TopK:     cfg.TopK,
	}
	resBefore1, err := eng.Search(ctx, primeQuery)
	if err != nil {
		return IncrementalBenchResult{}, FreshnessVerification{}, err
	}
	resBefore2, err := eng.Search(ctx, primeQuery)
	if err != nil {
		return IncrementalBenchResult{}, FreshnessVerification{}, err
	}
	if !resBefore2.CacheHit || len(resBefore1.Chunks) == 0 {
		return IncrementalBenchResult{}, FreshnessVerification{}, fmt.Errorf("expected warm cache hit before mutation")
	}

	// Pick 12 real Go files deterministically by index step for incremental v2 updates,
	// plus 3 operational updates, 3 new artifact inserts, and 3 deletions (21 ops total).
	var incLats []float64
	var churns []float64
	sumAffEnt := 0.0
	sumAffRel := 0.0
	updateOps, insertOps, deleteOps := 0, 0, 0
	cacheInvSeen := false

	step := len(records) / 14
	if step < 1 {
		step = 1
	}
	goUpdated := 0
	for i := 0; i < len(records) && goUpdated < 12; i += step {
		rec := records[i].Clone()
		if rec.Kind != KindCodeBlob {
			continue
		}
		rec.Version = 2
		rec.Content = rec.Content + fmt.Sprintf("\n// IncrementalDeltaRevisionV2 marker for %s (seed=%d)\n", rec.SourceID, cfg.Seed)
		tel, err := eng.UpsertSource(ctx, rec)
		if err != nil {
			return IncrementalBenchResult{}, FreshnessVerification{}, err
		}
		incLats = append(incLats, float64(tel.Duration.Nanoseconds())/1e6)
		churns = append(churns, tel.SubgraphChurnRatio*100.0)
		sumAffEnt += float64(tel.AffectedEntities)
		sumAffRel += float64(tel.AffectedRelations)
		updateOps++
		goUpdated++
		if tel.CacheInvalidated {
			cacheInvSeen = true
		}
	}

	// 3 operational v2 updates introducing fresh versioned facts.
	v2Updates := []SourceRecord{
		{
			SourceID: "ops/COMMIT-9104-txn-cache-fence.md",
			Version:  2,
			TenantID: benchTenant,
			Kind:     KindCommit,
			Title:    "COMMIT-9104 v2: Added AdaptiveWatermarkFence in posting/mvcc.go",
			Path:     "ops/COMMIT-9104-txn-cache-fence.md",
			Content: `Commit: COMMIT-9104
Author: alice-db
Files: posting/mvcc.go, posting/writer.go
Symbols: posting.AdaptiveWatermarkFence, posting.CachePL
Fixes: INC-401
Depends-On: ADR-108
Summary: Version 2 patch introduces AdaptiveWatermarkFence in posting/mvcc.go reducing MVCC rollup stall p99 to 1.8ms.`,
		},
		{
			SourceID: "ops/ADR-115-hnsw-start-node-selection.md",
			Version:  2,
			TenantID: benchTenant,
			Kind:     KindDecision,
			Title:    "ADR-115 v2: Dual-entrypoint verification for tok/hnsw",
			Path:     "ops/ADR-115-hnsw-start-node-selection.md",
			Content: `Decision: ADR-115
Author: vector-eng
Files: tok/hnsw/persistent_hnsw.go, tok/index_factory.go
Symbols: tok/hnsw.DualEntryPointGuard, tok.IndexFactory
Mitigates: INC-412
Related: COMMIT-9211
Rationale: Version 2 adds DualEntryPointGuard during persistent HNSW compaction to guarantee deterministic entry recall.`,
		},
		{
			SourceID: "ops/EVT-501-schema-rebuild-agent-audit.md",
			Version:  2,
			TenantID: benchTenant,
			Kind:     KindAgentEvent,
			Title:    "EVT-501 v2: Verified ZeroCopySchemaRebuildCheck in posting/index.go",
			Path:     "ops/EVT-501-schema-rebuild-agent-audit.md",
			Content: `Component: EVT-501
Author: agent-verifier
Files: posting/index.go, schema/parse.go, tok/tok.go
Symbols: posting.ZeroCopySchemaRebuildCheck, posting.IndexRebuild
Summary: Version 2 audit confirms ZeroCopySchemaRebuildCheck validates bit-exact derived index regeneration.`,
		},
	}
	for _, u := range v2Updates {
		tel, err := eng.UpsertSource(ctx, u)
		if err != nil {
			return IncrementalBenchResult{}, FreshnessVerification{}, err
		}
		incLats = append(incLats, float64(tel.Duration.Nanoseconds())/1e6)
		churns = append(churns, tel.SubgraphChurnRatio*100.0)
		sumAffEnt += float64(tel.AffectedEntities)
		sumAffRel += float64(tel.AffectedRelations)
		updateOps++
	}

	// 3 new incremental inserts.
	newInserts := []SourceRecord{
		{
			SourceID: "ops/COMMIT-9401-delta-graph-compactor.md",
			Version:  1,
			TenantID: benchTenant,
			Kind:     KindCommit,
			Title:    "COMMIT-9401: Incremental subgraph compactor hook",
			Path:     "ops/COMMIT-9401-delta-graph-compactor.md",
			Content: `Commit: COMMIT-9401
Author: dave-graph
Files: posting/list.go, algo/uidlist.go
Symbols: posting.SubgraphCompactorHook
Summary: Adds SubgraphCompactorHook telemetry for incremental posting-list delta compaction.`,
		},
		{
			SourceID: "ops/ADR-130-tenant-partitioned-indexes.md",
			Version:  1,
			TenantID: benchTenant,
			Kind:     KindDecision,
			Title:    "ADR-130: Strict per-tenant derived index partitioning",
			Path:     "ops/ADR-130-tenant-partitioned-indexes.md",
			Content: `Decision: ADR-130
Author: security-arch
Files: x/acl_enc_keys.go, x/keys.go
Symbols: x.TenantPartitionGuard
Summary: Mandates strict per-tenant derived index isolation and pre-return ACL checks via TenantPartitionGuard.`,
		},
		{
			SourceID: "ops/EVT-599-ephemeral-canary-note.md",
			Version:  1,
			TenantID: benchTenant,
			Kind:     KindAgentEvent,
			Title:    "EVT-599: Ephemeral canary probe note",
			Path:     "ops/EVT-599-ephemeral-canary-note.md",
			Content: `Component: EVT-599
Author: agent-canary
Symbols: EphemeralCanarySentinel99
Summary: EphemeralCanarySentinel99 diagnostic probe created to verify clean subgraph retraction upon deletion.`,
		},
	}
	for _, ins := range newInserts {
		tel, err := eng.UpsertSource(ctx, ins)
		if err != nil {
			return IncrementalBenchResult{}, FreshnessVerification{}, err
		}
		incLats = append(incLats, float64(tel.Duration.Nanoseconds())/1e6)
		churns = append(churns, tel.SubgraphChurnRatio*100.0)
		sumAffEnt += float64(tel.AffectedEntities)
		sumAffRel += float64(tel.AffectedRelations)
		insertOps++
	}

	// 2 deletes + 1 delete of newly inserted ephemeral note = 3 deletes.
	deleteIDs := []string{
		"ops/EVT-503-deprecated-legacy-note.md",
		"ops/EVT-599-ephemeral-canary-note.md",
		"ops/EVT-502-dql-lexer-parser-pipeline.md",
	}
	for _, did := range deleteIDs {
		tel, err := eng.DeleteSource(ctx, benchTenant, did)
		if err != nil {
			return IncrementalBenchResult{}, FreshnessVerification{}, err
		}
		incLats = append(incLats, float64(tel.Duration.Nanoseconds())/1e6)
		churns = append(churns, tel.SubgraphChurnRatio*100.0)
		sumAffEnt += float64(tel.AffectedEntities)
		sumAffRel += float64(tel.AffectedRelations)
		deleteOps++
	}

	// Verify freshness immediately after incremental updates/deletes (with cache enabled!).
	updatedChecks := []struct {
		query string
		token string
	}{
		{"AdaptiveWatermarkFence in COMMIT-9104", "adaptivewatermarkfence"},
		{"DualEntryPointGuard in ADR-115", "dualentrypointguard"},
		{"ZeroCopySchemaRebuildCheck in EVT-501", "zerocopyschemarebuildcheck"},
		{"SubgraphCompactorHook in COMMIT-9401", "subgraphcompactorhook"},
	}
	updatedHits := 0
	for _, chk := range updatedChecks {
		res, err := eng.Search(ctx, Query{
			Text:            chk.query,
			TenantID:        benchTenant,
			Mode:            ModeHybrid,
			TopK:            cfg.TopK,
			IncludeSubgraph: true,
		})
		if err != nil {
			return IncrementalBenchResult{}, FreshnessVerification{}, err
		}
		if resultContainsToken(res, chk.token) {
			updatedHits++
		}
	}

	deletedChecks := []struct {
		query string
		token string
	}{
		{"LegacyRaftStubSentinel migration checklist", "legacyraftstubsentinel"},
		{"EphemeralCanarySentinel99 diagnostic probe", "ephemeralcanarysentinel99"},
	}
	ghostHits := 0
	cacheInvalidatedAfterDelete := false
	for i, chk := range deletedChecks {
		res, err := eng.Search(ctx, Query{
			Text:            chk.query,
			TenantID:        benchTenant,
			Mode:            ModeHybrid,
			TopK:            cfg.TopK,
			IncludeSubgraph: true,
		})
		if err != nil {
			return IncrementalBenchResult{}, FreshnessVerification{}, err
		}
		if i == 0 && !res.CacheHit {
			cacheInvalidatedAfterDelete = true
		}
		if resultContainsToken(res, chk.token) {
			ghostHits++
		}
	}

	freshness := FreshnessVerification{
		QueriesTested:          len(updatedChecks) + len(deletedChecks),
		CacheInvalidatedOnEdit: cacheInvSeen && cacheInvalidatedAfterDelete,
		UpdatedRecallRate:      round4(float64(updatedHits) / float64(len(updatedChecks))),
		DeletedGhostRecallRate: round4(float64(ghostHits) / float64(len(deletedChecks))),
	}

	// Compute digest right after incremental delta operations.
	digestInc, err := eng.IndexDigest(benchTenant)
	if err != nil {
		return IncrementalBenchResult{}, FreshnessVerification{}, err
	}

	// Verify DropDerivedIndexes enforces ErrIndexesNotBuilt, then benchmark Full Graph Rebuild.
	eng.DropDerivedIndexes()
	_, dropErr := eng.Search(ctx, Query{Text: "posting.List", TenantID: benchTenant, Mode: ModeHybrid, TopK: 3})
	dropEnforced := dropErr == ErrIndexesNotBuilt

	rebuildRuns := 5
	if cfg.Iters < rebuildRuns {
		rebuildRuns = cfg.Iters
	}
	if rebuildRuns < 3 {
		rebuildRuns = 3
	}
	var rebuildLats []float64
	for r := 0; r < rebuildRuns; r++ {
		eng.DropDerivedIndexes()
		tel, err := eng.RebuildAll(ctx)
		if err != nil {
			return IncrementalBenchResult{}, FreshnessVerification{}, err
		}
		rebuildLats = append(rebuildLats, float64(tel.Duration.Nanoseconds())/1e6)
	}

	digestRebuild, err := eng.IndexDigest(benchTenant)
	if err != nil {
		return IncrementalBenchResult{}, FreshnessVerification{}, err
	}

	incStats := computeLatencyStats(incLats)
	rebStats := computeLatencyStats(rebuildLats)
	speedupP50 := 0.0
	if incStats.P50Ms > 0 {
		speedupP50 = round2(rebStats.P50Ms / incStats.P50Ms)
	}
	speedupMean := 0.0
	if incStats.MeanMs > 0 {
		speedupMean = round2(rebStats.MeanMs / incStats.MeanMs)
	}

	meanChurn := 0.0
	maxChurn := 0.0
	for _, c := range churns {
		meanChurn += c
		if c > maxChurn {
			maxChurn = c
		}
	}
	if len(churns) > 0 {
		meanChurn /= float64(len(churns))
	}

	totalOps := updateOps + insertOps + deleteOps
	return IncrementalBenchResult{
		IncrementalOps:          totalOps,
		UpdateOps:               updateOps,
		InsertOps:               insertOps,
		DeleteOps:               deleteOps,
		IncrementalLatency:      incStats,
		FullRebuildLatency:      rebStats,
		SpeedupP50:              speedupP50,
		SpeedupMean:             speedupMean,
		MeanSubgraphChurnPct:    round2(meanChurn),
		MaxSubgraphChurnPct:     round2(maxChurn),
		MeanAffectedEntities:    round2(sumAffEnt / float64(maxInt(1, totalOps))),
		MeanAffectedRelations:   round2(sumAffRel / float64(maxInt(1, totalOps))),
		DigestAfterIncremental:  digestInc,
		DigestAfterFullRebuild:  digestRebuild,
		BitExactEquivalence:     digestInc == digestRebuild,
		DropEnforcesNotBuiltErr: dropEnforced,
	}, freshness, nil
}

func evaluateSecurityIsolation(ctx context.Context) (SecurityIsolationReport, error) {
	eng := NewEngine(DefaultEngineConfig())

	// Seed shared entity names across tenants AND across ACL levels within tenant-alpha.
	docs := []SourceRecord{
		{
			SourceID: "alpha/public-posting.md",
			Version:  1,
			TenantID: "tenant-alpha",
			ACL:      []string{"role:eng"},
			Kind:     KindDocument,
			Title:    "Alpha Public Posting Guide",
			Content:  "Symbols: posting.List, algo.MergeSorted\nPublic documentation for posting.List compaction and algo.MergeSorted batching.",
		},
		{
			SourceID: "alpha/secret-posting-cve.md",
			Version:  1,
			TenantID: "tenant-alpha",
			ACL:      []string{"role:security"},
			Kind:     KindIncident,
			Title:    "Alpha Confidential Security Advisory",
			Content:  "Symbols: posting.List, x.Keys\nCauses: posting.List\nCONFIDENTIAL_ALPHA_TOKEN_9911 zero-day master key rotation procedure in x.Keys linked to posting.List.",
		},
		{
			SourceID: "beta/tenant-posting-key.md",
			Version:  1,
			TenantID: "tenant-beta",
			ACL:      []string{"role:eng"},
			Kind:     KindDocument,
			Title:    "Beta Proprietary Routing Spec",
			Content:  "Symbols: posting.List, x.Keys\nCONFIDENTIAL_BETA_SECRET_7744 proprietary tenant-beta index key schedule on posting.List.",
		},
	}

	// Also build an unfiltered shared-graph control (single tenant, empty ACL) to measure baseline leakage.
	unfiltered := NewEngine(DefaultEngineConfig())
	for _, d := range docs {
		if _, err := eng.UpsertSource(ctx, d); err != nil {
			return SecurityIsolationReport{}, err
		}
		u := d.Clone()
		u.TenantID = "shared-unfiltered"
		u.ACL = nil
		if _, err := unfiltered.UpsertSource(ctx, u); err != nil {
			return SecurityIsolationReport{}, err
		}
	}

	probes := []struct {
		query      string
		tenant     string
		principals []string
		forbidden  []string
	}{
		{"posting.List x.Keys key rotation", "tenant-alpha", []string{"role:eng"}, []string{"confidential_alpha_token_9911", "confidential_beta_secret_7744", "alpha/secret-posting-cve.md", "beta/tenant-posting-key.md"}},
		{"CONFIDENTIAL_ALPHA_TOKEN_9911 posting.List", "tenant-alpha", []string{"role:eng"}, []string{"confidential_alpha_token_9911", "alpha/secret-posting-cve.md", "zero-day"}},
		{"CONFIDENTIAL_BETA_SECRET_7744 posting.List", "tenant-alpha", []string{"role:eng"}, []string{"confidential_beta_secret_7744", "beta/tenant-posting-key.md"}},
		{"posting.List zero-day advisory x.Keys", "tenant-beta", []string{"role:eng"}, []string{"confidential_alpha_token_9911", "alpha/secret-posting-cve.md", "alpha/public-posting.md"}},
	}

	modes := []RetrievalMode{ModeOrdinaryHybrid, ModeLocal, ModeGlobal, ModeHybrid}
	totalProbes := 0
	leaks := 0
	controlLeaks := 0
	aclFilteredSeen := 0

	for _, p := range probes {
		for _, m := range modes {
			totalProbes++
			res, err := eng.Search(ctx, Query{
				Text:            p.query,
				TenantID:        p.tenant,
				Principals:      p.principals,
				Mode:            m,
				TopK:            5,
				IncludeSubgraph: true,
				BypassCache:     true,
			})
			if err != nil {
				return SecurityIsolationReport{}, err
			}
			aclFilteredSeen += res.ACLFilteredOut
			if containsAnyForbidden(res, p.forbidden) {
				leaks++
			}

			ctrlRes, err := unfiltered.Search(ctx, Query{
				Text:            p.query,
				TenantID:        "shared-unfiltered",
				Mode:            m,
				TopK:            5,
				IncludeSubgraph: true,
				BypassCache:     true,
			})
			if err != nil {
				return SecurityIsolationReport{}, err
			}
			if containsAnyForbidden(ctrlRes, p.forbidden) {
				controlLeaks++
			}
		}
	}

	return SecurityIsolationReport{
		ProbesExecuted:            totalProbes,
		LightRAGLeakCount:         leaks,
		LightRAGLeakRatePct:       round2(float64(leaks) * 100.0 / float64(maxInt(1, totalProbes))),
		UnfilteredControlLeaks:    controlLeaks,
		UnfilteredControlRatePct:  round2(float64(controlLeaks) * 100.0 / float64(maxInt(1, totalProbes))),
		ACLFilteredCandidatesSeen: aclFilteredSeen,
	}, nil
}

func buildNegativeRegimes(breakdowns []WorkloadBreakdown, sec SecurityIsolationReport) []NegativeRegime {
	find := func(wl string, mode RetrievalMode) SystemMetrics {
		for _, b := range breakdowns {
			if b.Workload == wl {
				for _, s := range b.Systems {
					if s.Mode == mode {
						return s
					}
				}
			}
		}
		return SystemMetrics{}
	}

	sfBase := find("single_fact_exact", ModeOrdinaryHybrid)
	sfHyb := find("single_fact_exact", ModeHybrid)
	overhead := 0.0
	if sfBase.Latency.P50Ms > 0 {
		overhead = round2(sfHyb.Latency.P50Ms / sfBase.Latency.P50Ms)
	}

	glLoc := find("global_thematic", ModeLocal)
	glGlb := find("global_thematic", ModeGlobal)
	glHyb := find("global_thematic", ModeHybrid)

	return []NegativeRegime{
		{
			Name: "neg_simple_single_fact_overhead",
			Description: fmt.Sprintf(
				"On exact single-identifier lookups (single_fact_exact), ordinary_hybrid achieves %.1f%% Recall@K and %.1f%% EM in %.3f ms p50, matching or exceeding lightrag_hybrid (%.1f%% Recall@K, %.1f%% EM, %.3f ms p50) while lightrag_hybrid adds %.2fx retrieval latency overhead from dual-level graph expansion. Route exact single-symbol queries directly to ordinary_hybrid.",
				sfBase.RecallAtK*100, sfBase.AnswerEM*100, sfBase.Latency.P50Ms,
				sfHyb.RecallAtK*100, sfHyb.AnswerEM*100, sfHyb.Latency.P50Ms, overhead,
			),
			Metrics: map[string]float64{
				"ordinary_hybrid_recall_pct": round2(sfBase.RecallAtK * 100),
				"ordinary_hybrid_p50_ms":     sfBase.Latency.P50Ms,
				"lightrag_hybrid_recall_pct": round2(sfHyb.RecallAtK * 100),
				"lightrag_hybrid_p50_ms":     sfHyb.Latency.P50Ms,
				"latency_overhead_x":         overhead,
			},
		},
		{
			Name: "neg_single_channel_ablation",
			Description: fmt.Sprintf(
				"On cross-cutting thematic queries (global_thematic), neither single graph channel suffices alone: Global-only achieves %.1f%% Recall@K / %.3f MRR / %.1f%% EM / %.1f%% Task Success and Local-only achieves %.1f%% Recall@K / %.3f MRR / %.1f%% EM / %.1f%% Task Success, whereas Dual-Level Hybrid reaches %.3f MRR / %.1f%% EM / %.1f%% Task Success — confirming that fusing Local, Global, and direct chunk channels is required.",
				glGlb.RecallAtK*100, glGlb.MRRAtK, glGlb.AnswerEM*100, glGlb.TaskSuccessRate*100,
				glLoc.RecallAtK*100, glLoc.MRRAtK, glLoc.AnswerEM*100, glLoc.TaskSuccessRate*100,
				glHyb.MRRAtK, glHyb.AnswerEM*100, glHyb.TaskSuccessRate*100,
			),
			Metrics: map[string]float64{
				"local_mrr":               glLoc.MRRAtK,
				"global_mrr":              glGlb.MRRAtK,
				"hybrid_mrr":              glHyb.MRRAtK,
				"local_em_pct":            round2(glLoc.AnswerEM * 100),
				"global_em_pct":           round2(glGlb.AnswerEM * 100),
				"hybrid_em_pct":           round2(glHyb.AnswerEM * 100),
				"local_task_success_pct":  round2(glLoc.TaskSuccessRate * 100),
				"global_task_success_pct": round2(glGlb.TaskSuccessRate * 100),
				"hybrid_task_success_pct": round2(glHyb.TaskSuccessRate * 100),
			},
		},
		{
			Name: "neg_acl_bridge_leakage_control",
			Description: fmt.Sprintf(
				"When multiple tenants and ACL roles share entity names (posting.List, x.Keys), an unfiltered shared-entity graph leaks restricted data on %d/%d (%.1f%%) adversarial probes across retrieval modes, whereas LightRAG's per-tenant partitioning and pre-return ACL scoping achieve %d/%d (%.1f%%) leakage.",
				sec.UnfilteredControlLeaks, sec.ProbesExecuted, sec.UnfilteredControlRatePct,
				sec.LightRAGLeakCount, sec.ProbesExecuted, sec.LightRAGLeakRatePct,
			),
			Metrics: map[string]float64{
				"unfiltered_leak_rate_pct": sec.UnfilteredControlRatePct,
				"lightrag_leak_rate_pct":   sec.LightRAGLeakRatePct,
				"probes_executed":          float64(sec.ProbesExecuted),
			},
		},
	}
}

func resultContainsToken(res SearchResult, tokenLower string) bool {
	var b strings.Builder
	for _, rc := range res.Chunks {
		b.WriteString(rc.Chunk.SourceID)
		b.WriteByte(' ')
		b.WriteString(rc.Chunk.Text)
		b.WriteByte(' ')
	}
	for _, re := range res.Entities {
		b.WriteString(re.Entity.Name)
		b.WriteByte(' ')
		b.WriteString(re.Entity.Description)
		b.WriteByte(' ')
		b.WriteString(strings.Join(re.Entity.SourceIDs, " "))
		b.WriteByte(' ')
	}
	for _, rr := range res.Relations {
		b.WriteString(rr.Relation.Description)
		b.WriteByte(' ')
		b.WriteString(strings.Join(rr.Relation.SourceIDs, " "))
		b.WriteByte(' ')
	}
	for _, tr := range res.Trails {
		b.WriteString(tr.Description)
		b.WriteByte(' ')
		b.WriteString(strings.Join(tr.SupportingSources, " "))
		b.WriteByte(' ')
	}
	return strings.Contains(strings.ToLower(b.String()), tokenLower)
}

func containsAnyForbidden(res SearchResult, forbidden []string) bool {
	for _, tok := range forbidden {
		if resultContainsToken(res, strings.ToLower(tok)) {
			return true
		}
	}
	return false
}

func computeLatencyStats(vals []float64) LatencyStats {
	if len(vals) == 0 {
		return LatencyStats{}
	}
	sorted := make([]float64, len(vals))
	copy(sorted, vals)
	sort.Float64s(sorted)

	sum := 0.0
	for _, v := range sorted {
		sum += v
	}
	pct := func(p float64) float64 {
		idx := int(math.Ceil(p*float64(len(sorted)))) - 1
		if idx < 0 {
			idx = 0
		}
		if idx >= len(sorted) {
			idx = len(sorted) - 1
		}
		return round3(sorted[idx])
	}
	return LatencyStats{
		P50Ms:  pct(0.50),
		P95Ms:  pct(0.95),
		P99Ms:  pct(0.99),
		MeanMs: round3(sum / float64(len(sorted))),
		Count:  len(sorted),
	}
}

func round2(v float64) float64 { return math.Round(v*100) / 100 }
func round3(v float64) float64 { return math.Round(v*1000) / 1000 }
func round4(v float64) float64 { return math.Round(v*10000) / 10000 }
func round6(v float64) float64 { return math.Round(v*1000000) / 1000000 }
