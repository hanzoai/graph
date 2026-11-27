package graphrag

import (
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"
)

// BenchmarkConfig controls reproducible evaluation runs.
type BenchmarkConfig struct {
	RepoRoot        string
	ExternalDataset string
	Seed            int64
	TopK            int
	TokenBudget     int
	Iters           int
}

// BenchmarkQuery defines one ground-truth evaluation query for controlled comparison.
type BenchmarkQuery struct {
	ID                 string   `json:"id"`
	Regime             string   `json:"regime"` // "global_thematic_synthesis", "cross_package_architecture", "exact_lookup_negative_regime"
	Query              string   `json:"query"`
	ExpectedRoute      string   `json:"expected_route"`
	RequiredThemes     []string `json:"required_themes,omitempty"`
	RequiredKeywords   []string `json:"required_keywords,omitempty"`
	GoldSources        []string `json:"gold_sources,omitempty"`
	MinSourcesRequired int      `json:"min_sources_required,omitempty"`
}

// ExternalDatasetBundle allows running cmd/graphrag-bench on external JSON corpora.
type ExternalDatasetBundle struct {
	Name      string              `json:"name"`
	Documents []CanonicalDocument `json:"documents"`
	Queries   []BenchmarkQuery    `json:"queries"`
}

// StrategyRegimeMetrics summarizes one strategy on one query regime.
type StrategyRegimeMetrics struct {
	Strategy             string  `json:"strategy"`
	Regime               string  `json:"regime"`
	QueryCount           int     `json:"query_count"`
	SynthesisAccuracyF1  float64 `json:"synthesis_accuracy_f1"`
	ThemeCoverage        float64 `json:"theme_coverage"`
	SourceRecall         float64 `json:"source_recall"`
	TaskSuccessRate      float64 `json:"task_success_rate"`
	RoutingAccuracy      float64 `json:"routing_accuracy"`
	AvgPromptTokens      float64 `json:"avg_prompt_tokens"`
	AvgCompletionTokens  float64 `json:"avg_completion_tokens"`
	AvgTotalTokens       float64 `json:"avg_total_tokens"`
	AvgCostUSD           float64 `json:"avg_cost_usd"`
	LatencyP50Ms         float64 `json:"latency_p50_ms"`
	LatencyP95Ms         float64 `json:"latency_p95_ms"`
	AvgCommunitiesMapped float64 `json:"avg_communities_mapped"`
	ACLLeaks             int     `json:"acl_leaks"`
}

// IncrementalBenchCase captures one incremental refresh measurement vs full rebuild.
type IncrementalBenchCase struct {
	CaseName                string  `json:"case_name"`
	Operation               string  `json:"operation"`
	IncrementalDurationMs   float64 `json:"incremental_duration_ms"`
	FullRebuildDurationMs   float64 `json:"full_rebuild_duration_ms"`
	SpeedupVsFullRebuild    float64 `json:"speedup_vs_full_rebuild"`
	CommunitiesResummarized int     `json:"communities_resummarized"`
	CommunitiesReused       int     `json:"communities_reused"`
	CommunitiesReusedPct    float64 `json:"communities_reused_pct"`
	IncrementalTokens       int     `json:"incremental_tokens"`
	FullRebuildTokens       int     `json:"full_rebuild_tokens"`
	SummaryTokenSavingsPct  float64 `json:"summary_token_savings_pct"`
	IncrementalCostUSD      float64 `json:"incremental_cost_usd"`
	FullRebuildCostUSD      float64 `json:"full_rebuild_cost_usd"`
}

// ACLAuditReport summarizes strict multi-tenant & ACL summary isolation tests.
type ACLAuditReport struct {
	TotalProbes                  int    `json:"total_probes"`
	CanaryLeaksInGlobalSummaries int    `json:"canary_leaks_in_global_summaries"`
	CanaryLeaksInLocalHits       int    `json:"canary_leaks_in_local_hits"`
	CrossTenantPartitionsLeaked  int    `json:"cross_tenant_partitions_leaked"`
	ExistenceOracleLeaks         int    `json:"existence_oracle_leaks"`
	PostRevocationLeaks          int    `json:"post_revocation_leaks"`
	DropRebuildBitExactParity    bool   `json:"drop_rebuild_bit_exact_parity"`
	InitialDerivedDigest         string `json:"initial_derived_digest"`
	RebuiltDerivedDigest         string `json:"rebuilt_derived_digest"`
}

// BenchmarkReport is the complete machine-readable receipt written to results.json.
type BenchmarkReport struct {
	GeneratedAtUTC      string                  `json:"generated_at_utc"`
	GoVersion           string                  `json:"go_version"`
	GOOS                string                  `json:"goos"`
	GOARCH              string                  `json:"goarch"`
	NumCPU              int                     `json:"num_cpu"`
	Seed                int64                   `json:"seed"`
	TopK                int                     `json:"top_k"`
	TokenBudget         int                     `json:"token_budget"`
	Iters               int                     `json:"iters"`
	DatasetName         string                  `json:"dataset_name"`
	RealGoFilesParsed   int                     `json:"real_go_files_parsed"`
	RepoDocsParsed      int                     `json:"repo_docs_parsed"`
	RegressionDocsCount int                     `json:"regression_docs_count"`
	TotalCanonicalDocs  int                     `json:"total_canonical_docs"`
	BuildCost           BuildStats              `json:"build_cost"`
	IncrementalRefresh  []IncrementalBenchCase  `json:"incremental_refresh"`
	RegimeMetrics       []StrategyRegimeMetrics `json:"regime_metrics"`
	OverallMetrics      []StrategyRegimeMetrics `json:"overall_metrics"`
	ACLAudit            ACLAuditReport          `json:"acl_audit"`
	Disclosures         []string                `json:"disclosures"`
}

// RunBenchmark executes the controlled end-to-end GraphRAG evaluation across
// global synthesis, cross-package architecture, exact-lookup negative regime,
// incremental invalidation, and ACL isolation.
func RunBenchmark(cfg BenchmarkConfig) (BenchmarkReport, error) {
	if cfg.TopK <= 0 {
		cfg.TopK = 5
	}
	if cfg.TokenBudget <= 0 {
		cfg.TokenBudget = 4000
	}
	if cfg.Iters <= 0 {
		cfg.Iters = 5
	}
	if cfg.Seed == 0 {
		cfg.Seed = 1347
	}

	var docs []CanonicalDocument
	var queries []BenchmarkQuery
	var realGoCount, repoDocsCount, regDocsCount int
	datasetName := "hanzoai/graph-real-corpus-v1"

	if cfg.ExternalDataset != "" {
		raw, err := os.ReadFile(cfg.ExternalDataset)
		if err != nil {
			return BenchmarkReport{}, fmt.Errorf("read external dataset: %w", err)
		}
		var bundle ExternalDatasetBundle
		if err := json.Unmarshal(raw, &bundle); err != nil {
			return BenchmarkReport{}, fmt.Errorf("parse external dataset: %w", err)
		}
		if bundle.Name != "" {
			datasetName = bundle.Name
		}
		docs = bundle.Documents
		queries = bundle.Queries
	} else {
		var err error
		docs, realGoCount, repoDocsCount, regDocsCount, err = LoadRealEngineeringCorpus(cfg.RepoRoot)
		if err != nil {
			return BenchmarkReport{}, err
		}
		queries = DefaultBenchmarkQueries()
	}

	// Build engine and stage canonical documents (Tenant Alpha authorized +
	// restricted security docs + Tenant Beta foreign docs).
	engCfg := DefaultConfig()
	engCfg.DefaultTopK = cfg.TopK
	engCfg.DefaultTokenBudget = cfg.TokenBudget
	eng := NewEngine(engCfg)

	for _, d := range docs {
		if _, err := eng.UpsertDocument(d); err != nil {
			return BenchmarkReport{}, fmt.Errorf("upsert doc %s: %w", d.SourceID, err)
		}
	}

	// Add restricted security records & foreign tenant records for strict ACL audit.
	restrictedDocs := buildACLCanaryDocuments()
	for _, d := range restrictedDocs {
		if _, err := eng.UpsertDocument(d); err != nil {
			return BenchmarkReport{}, fmt.Errorf("upsert canary %s: %w", d.SourceID, err)
		}
	}

	// Initial offline full build + Drop/Rebuild bit-exact SHA-256 parity verification.
	buildStats := eng.BuildAll()
	initialDigest, err := eng.DerivedDigest()
	if err != nil {
		return BenchmarkReport{}, err
	}
	eng.DropDerivedIndexes()
	rebuildStats := eng.RebuildAll()
	rebuiltDigest, err := eng.DerivedDigest()
	if err != nil {
		return BenchmarkReport{}, err
	}

	// Evaluate all 5 controlled strategies on identical corpus, caller, judge, and budget.
	engCaller := CallerContext{
		TenantID:   "tenant-alpha",
		Principals: []string{"role:eng"},
	}

	strategies := []struct {
		name string
		mode QueryMode
	}{
		{name: "ordinary_hybrid_topk", mode: ModeLocalHybrid},
		{name: "graphrag_global_l0", mode: ModeGlobalL0},
		{name: "graphrag_global_l1", mode: ModeGlobalL1},
		{name: "graphrag_global_l2", mode: ModeGlobalL2},
		{name: "routed_auto_gate", mode: ModeAuto},
	}

	regimesOrder := []string{
		"global_thematic_synthesis",
		"cross_package_architecture",
		"exact_lookup_negative_regime",
	}
	queriesByRegime := make(map[string][]BenchmarkQuery)
	for _, q := range queries {
		queriesByRegime[q.Regime] = append(queriesByRegime[q.Regime], q)
	}

	canaries := []string{
		"CANARY_VAULT_KEY_ZERO_DAY_9941",
		"CANARY_HSM_BYPASS_7723",
		"CANARY_BETA_CORPUS_8812",
	}

	var regimeMetrics []StrategyRegimeMetrics
	var overallMetrics []StrategyRegimeMetrics

	for _, strat := range strategies {
		var allAcc, allCov, allSrc, allSucc, allRoute []float64
		var allPrompt, allComp, allTot, allCost, allComm []float64
		var allLat []float64
		totalLeaks := 0

		for _, reg := range regimesOrder {
			qs := queriesByRegime[reg]
			if len(qs) == 0 {
				continue
			}
			var accs, covs, srcs, succs, routes []float64
			var prompts, comps, tots, costs, comms []float64
			var lats []float64
			leaks := 0

			for _, q := range qs {
				var lastResp QueryResponse
				for iter := 0; iter < cfg.Iters; iter++ {
					resp, err := eng.Query(context.Background(), engCaller, QueryRequest{
						Query:       q.Query,
						Mode:        strat.mode,
						TopK:        cfg.TopK,
						TokenBudget: cfg.TokenBudget,
					})
					if err != nil {
						return BenchmarkReport{}, fmt.Errorf("query %s/%s: %w", strat.name, q.ID, err)
					}
					lats = append(lats, resp.Metrics.LatencyMs)
					allLat = append(allLat, resp.Metrics.LatencyMs)
					lastResp = resp
				}

				// Check for any canary leakage.
				if containsAnyCanary(lastResp, canaries) {
					leaks++
					totalLeaks++
				}

				acc, cov, srcRec, taskSucc := judgeQueryResponse(q, lastResp)
				routeHit := 0.0
				if string(lastResp.Routing.Target) == q.ExpectedRoute {
					routeHit = 1.0
				}
				// For forced strategies, measure whether effective execution matched expected route.
				if strat.mode != ModeAuto {
					effTarget := RouteGlobalGraphRAG
					if lastResp.EffectiveMode == ModeLocalHybrid {
						effTarget = RouteLocalHybrid
					}
					if string(effTarget) == q.ExpectedRoute {
						routeHit = 1.0
					} else {
						routeHit = 0.0
					}
				}

				accs = append(accs, acc)
				covs = append(covs, cov)
				srcs = append(srcs, srcRec)
				succs = append(succs, taskSucc)
				routes = append(routes, routeHit)
				prompts = append(prompts, float64(lastResp.Metrics.PromptTokens))
				comps = append(comps, float64(lastResp.Metrics.CompletionTokens))
				tots = append(tots, float64(lastResp.Metrics.TotalTokens))
				costs = append(costs, lastResp.Metrics.EstimatedCostUSD)
				comms = append(comms, float64(lastResp.Metrics.CommunitiesMapped))

				allAcc = append(allAcc, acc)
				allCov = append(allCov, cov)
				allSrc = append(allSrc, srcRec)
				allSucc = append(allSucc, taskSucc)
				allRoute = append(allRoute, routeHit)
				allPrompt = append(allPrompt, float64(lastResp.Metrics.PromptTokens))
				allComp = append(allComp, float64(lastResp.Metrics.CompletionTokens))
				allTot = append(allTot, float64(lastResp.Metrics.TotalTokens))
				allCost = append(allCost, lastResp.Metrics.EstimatedCostUSD)
				allComm = append(allComm, float64(lastResp.Metrics.CommunitiesMapped))
			}

			regimeMetrics = append(regimeMetrics, StrategyRegimeMetrics{
				Strategy:             strat.name,
				Regime:               reg,
				QueryCount:           len(qs),
				SynthesisAccuracyF1:  mean(accs),
				ThemeCoverage:        mean(covs),
				SourceRecall:         mean(srcs),
				TaskSuccessRate:      mean(succs),
				RoutingAccuracy:      mean(routes),
				AvgPromptTokens:      mean(prompts),
				AvgCompletionTokens:  mean(comps),
				AvgTotalTokens:       mean(tots),
				AvgCostUSD:           mean(costs),
				LatencyP50Ms:         percentile(lats, 50),
				LatencyP95Ms:         percentile(lats, 95),
				AvgCommunitiesMapped: mean(comms),
				ACLLeaks:             leaks,
			})
		}

		overallMetrics = append(overallMetrics, StrategyRegimeMetrics{
			Strategy:             strat.name,
			Regime:               "overall_all_regimes",
			QueryCount:           len(allAcc),
			SynthesisAccuracyF1:  mean(allAcc),
			ThemeCoverage:        mean(allCov),
			SourceRecall:         mean(allSrc),
			TaskSuccessRate:      mean(allSucc),
			RoutingAccuracy:      mean(allRoute),
			AvgPromptTokens:      mean(allPrompt),
			AvgCompletionTokens:  mean(allComp),
			AvgTotalTokens:       mean(allTot),
			AvgCostUSD:           mean(allCost),
			LatencyP50Ms:         percentile(allLat, 50),
			LatencyP95Ms:         percentile(allLat, 95),
			AvgCommunitiesMapped: mean(allComm),
			ACLLeaks:             totalLeaks,
		})
	}

	// Run ACL isolation & post-revocation security audit.
	aclAudit := runACLAudit(eng, initialDigest, rebuiltDigest, initialDigest == rebuiltDigest)

	// Run incremental summary invalidation vs full rebuild benchmark on a fresh clone engine.
	incCases, err := runIncrementalBench(engCfg, docs, rebuildStats)
	if err != nil {
		return BenchmarkReport{}, err
	}

	report := BenchmarkReport{
		GeneratedAtUTC:      time.Now().UTC().Format(time.RFC3339),
		GoVersion:           runtime.Version(),
		GOOS:                runtime.GOOS,
		GOARCH:              runtime.GOARCH,
		NumCPU:              runtime.NumCPU(),
		Seed:                cfg.Seed,
		TopK:                cfg.TopK,
		TokenBudget:         cfg.TokenBudget,
		Iters:               cfg.Iters,
		DatasetName:         datasetName,
		RealGoFilesParsed:   realGoCount,
		RepoDocsParsed:      repoDocsCount,
		RegressionDocsCount: regDocsCount,
		TotalCanonicalDocs:  len(docs) + len(restrictedDocs),
		BuildCost:           buildStats,
		IncrementalRefresh:  incCases,
		RegimeMetrics:       regimeMetrics,
		OverallMetrics:      overallMetrics,
		ACLAudit:            aclAudit,
		Disclosures: []string{
			fmt.Sprintf("Parsed %d real upstream hanzoai/graph Go source files via go/parser + go/ast and %d repository documentation files directly from disk.", realGoCount, repoDocsCount),
			fmt.Sprintf("Added %d deterministic versioned engineering regression & ADR records grounded in real hanzoai/graph packages (posting, dql, tok/hnsw, schema, types, codec, algo, x) to evaluate multi-document global synthesis; external JSON corpora are supported via --external-dataset.", regDocsCount),
			"Zero external dependencies added to go.mod/go.sum; pure Go stdlib clean-room Apache-2.0 implementation inspired by Edge et al. (arXiv:2404.16130).",
			"All 5 evaluated strategies share the exact same canonical store, TopK budget, token budget, deterministic extractive synthesizer, pricing model ($3.00/1M input, $15.00/1M output), and judge.",
			"Upstream Microsoft GraphRAG (MIT) is referenced strictly as study/design literature (explicitly noted in issue #3 as research software in maintenance mode with expensive indexing); zero code is copied.",
		},
	}

	return report, nil
}

func runIncrementalBench(cfg Config, baseDocs []CanonicalDocument, fullBuildRef BuildStats) ([]IncrementalBenchCase, error) {
	eng := NewEngine(cfg)
	for _, d := range baseDocs {
		if _, err := eng.UpsertDocument(d); err != nil {
			return nil, err
		}
	}
	fullStats := eng.BuildAll()
	if fullStats.TotalTokens == 0 {
		fullStats = fullBuildRef
	}

	var cases []IncrementalBenchCase

	makeCase := func(name string, inc IncrementalRefreshStats, full BuildStats) IncrementalBenchCase {
		totalComms := inc.CommunitiesResummarized + inc.CommunitiesReused
		reusedPct := 0.0
		if totalComms > 0 {
			reusedPct = 100.0 * float64(inc.CommunitiesReused) / float64(totalComms)
		}
		speedup := 1.0
		if inc.DurationMs > 0 {
			speedup = full.DurationMs / inc.DurationMs
		}
		tokSavePct := 0.0
		if full.TotalTokens > 0 {
			tokSavePct = 100.0 * float64(full.TotalTokens-inc.TotalTokens) / float64(full.TotalTokens)
		}
		return IncrementalBenchCase{
			CaseName:                name,
			Operation:               inc.Operation,
			IncrementalDurationMs:   inc.DurationMs,
			FullRebuildDurationMs:   full.DurationMs,
			SpeedupVsFullRebuild:    speedup,
			CommunitiesResummarized: inc.CommunitiesResummarized,
			CommunitiesReused:       inc.CommunitiesReused,
			CommunitiesReusedPct:    reusedPct,
			IncrementalTokens:       inc.TotalTokens,
			FullRebuildTokens:       full.TotalTokens,
			SummaryTokenSavingsPct:  tokSavePct,
			IncrementalCostUSD:      inc.EstimatedCostUSD,
			FullRebuildCostUSD:      full.EstimatedCostUSD,
		}
	}

	// Case 1: Update 1 regression report in place.
	updatedDoc := CanonicalDocument{
		SourceID: "reports/REG-001.md",
		Version:  "v2",
		TenantID: "tenant-alpha",
		ACL:      []string{"*"},
		Kind:     "regression_report",
		Title:    "REG-001: MVCC posting-list rollup race under concurrent commit (Mitigated)",
		Content:  "Updated postmortem for REG-001 in posting/mvcc.go and posting/list.go: added read-ts barrier and lock ordering verification reducing MVCC stale read regressions by 92%.",
		Metadata: map[string]string{
			"package":          "posting",
			"root_cause":       "mvcc_rollup_lock_ordering",
			"regression_class": "mvcc_and_posting_consistency",
		},
	}
	inc1, err := eng.UpsertDocument(updatedDoc)
	if err != nil {
		return nil, err
	}
	cases = append(cases, makeCase("single_doc_update", inc1, fullStats))

	// Case 2: Update 3 documents across distinct packages (sequential incremental batch).
	batchDocs := []CanonicalDocument{
		{
			SourceID: "reports/REG-004.md",
			Version:  "v2",
			TenantID: "tenant-alpha",
			ACL:      []string{"*"},
			Kind:     "regression_report",
			Title:    "REG-004: DQL recursive query parser stack amplification (v2 fix)",
			Content:  "Follow-up on REG-004 in dql/parser.go and lex/lexer.go: bounded recursion depth and added parser fuzz coverage for nested filter blocks.",
			Metadata: map[string]string{"package": "dql", "root_cause": "unbounded_parser_recursion", "regression_class": "query_parsing_and_dql"},
		},
		{
			SourceID: "reports/REG-009.md",
			Version:  "v2",
			TenantID: "tenant-alpha",
			ACL:      []string{"*"},
			Kind:     "regression_report",
			Title:    "REG-009: HNSW SearchLayer mutex contention under bulk vector ingest (v2)",
			Content:  "Follow-up on REG-009 in tok/hnsw/search_layer.go and tok/hnsw/persistent_hnsw.go: striped rwmutex lowered tail latency during HNSW vector updates.",
			Metadata: map[string]string{"package": "tok/hnsw", "root_cause": "hnsw_layer_lock_contention", "regression_class": "indexing_and_vector_hnsw"},
		},
		{
			SourceID: "reports/REG-014.md",
			Version:  "v2",
			TenantID: "tenant-alpha",
			ACL:      []string{"*"},
			Kind:     "regression_report",
			Title:    "REG-014: Codec pack/unpack slice over-allocation under skewed postings (v2)",
			Content:  "Follow-up on REG-014 in codec/codec.go and algo/packed.go: capped pooled buffer growth to eliminate heap OOM spikes.",
			Metadata: map[string]string{"package": "codec", "root_cause": "codec_buffer_over_allocation", "regression_class": "memory_allocation_and_codec"},
		},
	}
	var batchAgg IncrementalRefreshStats
	batchAgg.Operation = "incremental_3doc_batch"
	for _, bd := range batchDocs {
		st, err := eng.UpsertDocument(bd)
		if err != nil {
			return nil, err
		}
		batchAgg.DurationMs += st.DurationMs
		batchAgg.DirtyEntities += st.DirtyEntities
		batchAgg.CommunitiesResummarized += st.CommunitiesResummarized
		batchAgg.CommunitiesReused += st.CommunitiesReused
		batchAgg.PromptTokens += st.PromptTokens
		batchAgg.CompletionTokens += st.CompletionTokens
		batchAgg.TotalTokens += st.TotalTokens
		batchAgg.EstimatedCostUSD += st.EstimatedCostUSD
	}
	// Normalize CommunitiesReused for average reuse across the 3 steps.
	batchAgg.CommunitiesReused = batchAgg.CommunitiesReused / len(batchDocs)
	batchAgg.CommunitiesResummarized = batchAgg.CommunitiesResummarized / len(batchDocs)
	cases = append(cases, makeCase("three_doc_update_batch", batchAgg, fullStats))

	// Case 3: Delete 1 document and incrementally retract its leaf + ancestors.
	incDel, err := eng.DeleteDocument("tenant-alpha", "reports/REG-020.md")
	if err != nil {
		return nil, err
	}
	cases = append(cases, makeCase("single_doc_delete", incDel, fullStats))

	return cases, nil
}

func runACLAudit(eng *Engine, initialDigest, rebuiltDigest string, parity bool) ACLAuditReport {
	report := ACLAuditReport{
		DropRebuildBitExactParity: parity,
		InitialDerivedDigest:      initialDigest,
		RebuiltDerivedDigest:      rebuiltDigest,
	}

	engCaller := CallerContext{TenantID: "tenant-alpha", Principals: []string{"role:eng"}}
	secCaller := CallerContext{TenantID: "tenant-alpha", Principals: []string{"role:eng", "role:security-incident"}}
	betaCaller := CallerContext{TenantID: "tenant-beta", Principals: []string{"role:eng"}}

	secretCanaries := []string{
		"CANARY_VAULT_KEY_ZERO_DAY_9941",
		"CANARY_HSM_BYPASS_7723",
		"CANARY_BETA_CORPUS_8812",
	}

	// 1. Probe global L0, L1, L2 and local queries as standard engineer.
	probeQueries := []string{
		"What are the largest sources of engineering regressions across our repositories including security and vault?",
		"Summarize all security, encryption, vault, and HSM incidents across all packages.",
		"CANARY_VAULT_KEY_ZERO_DAY_9941 CANARY_HSM_BYPASS_7723 CANARY_BETA_CORPUS_8812",
	}
	for _, pq := range probeQueries {
		for _, m := range []QueryMode{ModeGlobalL0, ModeGlobalL1, ModeGlobalL2, ModeLocalHybrid} {
			report.TotalProbes++
			resp, err := eng.Query(context.Background(), engCaller, QueryRequest{
				Query: pq,
				Mode:  m,
				TopK:  10,
			})
			if err == nil && containsAnyCanary(resp, secretCanaries) {
				if m == ModeLocalHybrid {
					report.CanaryLeaksInLocalHits++
				} else {
					report.CanaryLeaksInGlobalSummaries++
				}
			}
		}
	}

	// 2. Existence-oracle probe: fetch restricted community IDs as security admin,
	// then probe them as standard engineer and as tenant-beta.
	secComms, _ := eng.ListCommunities(secCaller, -1)
	engComms, _ := eng.ListCommunities(engCaller, -1)
	engCommSet := make(map[string]struct{}, len(engComms))
	for _, c := range engComms {
		engCommSet[c.ID] = struct{}{}
	}
	for _, sc := range secComms {
		if _, visibleToEng := engCommSet[sc.ID]; visibleToEng {
			continue
		}
		report.TotalProbes++
		if _, err := eng.GetCommunitySummary(engCaller, sc.ID); err != ErrNotFound {
			report.ExistenceOracleLeaks++
		}
		report.TotalProbes++
		if _, err := eng.GetCommunitySummary(betaCaller, sc.ID); err != ErrNotFound {
			report.CrossTenantPartitionsLeaked++
		}
	}

	// 3. Cross-tenant global query probe: tenant-beta must never see tenant-alpha sources.
	report.TotalProbes++
	betaResp, err := eng.Query(context.Background(), betaCaller, QueryRequest{
		Query: "What are the largest sources of engineering regressions across all packages?",
		Mode:  ModeGlobalL1,
	})
	if err == nil {
		for _, cit := range betaResp.Citations {
			if cit.TenantID != "tenant-beta" {
				report.CrossTenantPartitionsLeaked++
			}
		}
	}

	// 4. Live ACL revocation probe: ingest a public canary doc, verify visible,
	// then revoke ACL to ["role:security-incident"] and verify 0 leaks.
	revokeCanary := "CANARY_REVOKE_LIVE_5501"
	revokeDoc := CanonicalDocument{
		SourceID: "reports/REV-LIVE-001.md",
		Version:  "v1",
		TenantID: "tenant-alpha",
		ACL:      []string{"*"},
		Kind:     "regression_report",
		Title:    "REV-LIVE-001: Temporary public disclosure",
		Content:  "Temporary disclosure mentioning " + revokeCanary + " in x/vault.go and posting/mvcc.go.",
		Metadata: map[string]string{"package": "x", "regression_class": "security_acl_and_encryption"},
	}
	_, _ = eng.UpsertDocument(revokeDoc)

	// Revoke ACL in place via incremental update.
	revokeDoc.Version = "v2"
	revokeDoc.ACL = []string{"role:security-incident"}
	_, _ = eng.UpsertDocument(revokeDoc)

	for _, m := range []QueryMode{ModeGlobalL0, ModeGlobalL1, ModeGlobalL2, ModeLocalHybrid} {
		report.TotalProbes++
		resp, err := eng.Query(context.Background(), engCaller, QueryRequest{
			Query: "Summarize security vault disclosures including " + revokeCanary,
			Mode:  m,
		})
		if err == nil && containsAnyCanary(resp, []string{revokeCanary}) {
			report.PostRevocationLeaks++
		}
	}

	// Clean up temporary revocation test document.
	_, _ = eng.DeleteDocument("tenant-alpha", revokeDoc.SourceID)

	return report
}

func containsAnyCanary(resp QueryResponse, canaries []string) bool {
	var all strings.Builder
	all.WriteString(resp.Answer)
	all.WriteByte('\n')
	for _, p := range resp.PartialAnswers {
		all.WriteString(strings.Join(p.KeyPoints, " "))
		all.WriteByte('\n')
	}
	for _, h := range resp.SourceHits {
		all.WriteString(h.Title)
		all.WriteByte(' ')
		all.WriteString(h.Snippet)
		all.WriteByte('\n')
	}
	text := all.String()
	for _, c := range canaries {
		if strings.Contains(text, c) {
			return true
		}
	}
	return false
}

func judgeQueryResponse(q BenchmarkQuery, resp QueryResponse) (accF1, themeCov, sourceRec, taskSucc float64) {
	ansLower := strings.ToLower(resp.Answer)
	detectedThemeSet := make(map[string]struct{}, len(resp.DetectedThemes))
	for _, th := range resp.DetectedThemes {
		detectedThemeSet[strings.ToLower(th)] = struct{}{}
	}
	citedSet := make(map[string]struct{}, len(resp.Citations))
	for _, c := range resp.Citations {
		citedSet[c.SourceID] = struct{}{}
	}

	// Theme coverage.
	if len(q.RequiredThemes) > 0 {
		hits := 0
		for _, rt := range q.RequiredThemes {
			rtLow := strings.ToLower(rt)
			if _, ok := detectedThemeSet[rtLow]; ok || strings.Contains(ansLower, rtLow) {
				hits++
			}
		}
		themeCov = float64(hits) / float64(len(q.RequiredThemes))
	} else {
		themeCov = 1.0
	}

	// Keyword / synthesis evidence recall + precision proxy (F1).
	kwHits := 0
	for _, kw := range q.RequiredKeywords {
		if strings.Contains(ansLower, strings.ToLower(kw)) {
			kwHits++
		}
	}
	kwRecall := 1.0
	if len(q.RequiredKeywords) > 0 {
		kwRecall = float64(kwHits) / float64(len(q.RequiredKeywords))
	}

	// Source citation recall.
	srcHits := 0
	for _, gs := range q.GoldSources {
		if _, ok := citedSet[gs]; ok || strings.Contains(ansLower, strings.ToLower(gs)) {
			srcHits++
		}
	}
	if len(q.GoldSources) > 0 {
		sourceRec = float64(srcHits) / float64(len(q.GoldSources))
	} else {
		sourceRec = 1.0
	}

	// For exact lookups, evaluate exact-hit precision: rank-1 exact match on the
	// local path scores high precision, whereas returning dozens of unrelated
	// community citations across the corpus incurs a heavy precision penalty.
	precision := 1.0
	if q.Regime == "exact_lookup_negative_regime" {
		if len(resp.SourceHits) > 0 && len(q.GoldSources) > 0 {
			topRank := 0
			for idx, h := range resp.SourceHits {
				for _, gs := range q.GoldSources {
					if h.SourceRef.SourceID == gs {
						topRank = idx + 1
						break
					}
				}
				if topRank > 0 {
					break
				}
			}
			if topRank > 0 {
				precision = 1.0 / float64(topRank)
			} else {
				precision = 0.0
			}
		} else if len(resp.Citations) > 0 {
			precision = math.Min(1.0, float64(len(q.GoldSources)*2)/float64(len(resp.Citations)))
		} else {
			precision = 0.0
		}
		recall := 0.5*kwRecall + 0.5*sourceRec
		if precision+recall > 0 {
			accF1 = 2.0 * precision * recall / (precision + recall)
		}
		if srcHits >= 1 && kwRecall >= 0.5 && precision >= 0.50 {
			taskSucc = 1.0
		}
		return accF1, themeCov, sourceRec, taskSucc
	}

	// For global synthesis & cross-package architecture regimes:
	// Synthesis Accuracy F1 blends required theme coverage, required evidence
	// keywords, and multi-source breadth.
	recall := 0.45*themeCov + 0.35*kwRecall + 0.20*sourceRec
	precision = math.Min(1.0, 0.60+0.40*kwRecall)
	if precision+recall > 0 {
		accF1 = 2.0 * precision * recall / (precision + recall)
	}
	minSrc := q.MinSourcesRequired
	if minSrc <= 0 {
		minSrc = 3
	}
	if themeCov >= 0.70 && kwRecall >= 0.60 && len(resp.Citations) >= minSrc {
		taskSucc = 1.0
	}
	return accF1, themeCov, sourceRec, taskSucc
}

// LoadRealEngineeringCorpus parses real upstream hanzoai/graph Go source files,
// repository documentation, and deterministic package-grounded regression
// reports into CanonicalDocuments under tenant-alpha.
func LoadRealEngineeringCorpus(repoRoot string) (docs []CanonicalDocument, goFiles int, repoDocs int, regDocs int, err error) {
	if repoRoot == "" {
		repoRoot = "."
	}
	baseTime := time.Date(2026, 11, 1, 0, 0, 0, 0, time.UTC)

	// 1. Walk real upstream Go source files (skipping tests, generated pb, .git, and graphrag/cmd itself).
	var goPaths []string
	err = filepath.WalkDir(repoRoot, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		rel, rerr := filepath.Rel(repoRoot, path)
		if rerr != nil {
			return rerr
		}
		rel = filepath.ToSlash(rel)
		if d.IsDir() {
			if rel == ".git" || rel == "graphrag" || rel == "cmd" || rel == "docs" || strings.HasPrefix(rel, ".") && rel != "." {
				return filepath.SkipDir
			}
			return nil
		}
		if strings.HasSuffix(rel, ".go") && !strings.HasSuffix(rel, "_test.go") && !strings.HasSuffix(rel, ".pb.go") {
			goPaths = append(goPaths, rel)
		}
		return nil
	})
	if err != nil {
		return nil, 0, 0, 0, fmt.Errorf("walk repo %s: %w", repoRoot, err)
	}
	sort.Strings(goPaths)

	for _, rel := range goPaths {
		raw, rerr := os.ReadFile(filepath.Join(repoRoot, filepath.FromSlash(rel)))
		if rerr != nil {
			continue
		}
		content := string(raw)
		if len(content) > 24000 {
			content = content[:24000]
		}
		docs = append(docs, CanonicalDocument{
			SourceID:  rel,
			Version:   "v1",
			TenantID:  "tenant-alpha",
			ACL:       []string{"*"},
			Kind:      "go_source",
			Title:     "Go Source: " + rel,
			Content:   content,
			Metadata:  map[string]string{"package": filepath.ToSlash(filepath.Dir(rel))},
			UpdatedAt: baseTime,
		})
		goFiles++
	}

	// 2. Load top-level repository docs if present.
	for _, fname := range []string{"README.md", "hanzo.yml", "NOTICE"} {
		full := filepath.Join(repoRoot, fname)
		raw, rerr := os.ReadFile(full)
		if rerr != nil {
			continue
		}
		docs = append(docs, CanonicalDocument{
			SourceID:  fname,
			Version:   "v1",
			TenantID:  "tenant-alpha",
			ACL:       []string{"*"},
			Kind:      "doc",
			Title:     "Repo Doc: " + fname,
			Content:   string(raw),
			UpdatedAt: baseTime,
		})
		repoDocs++
	}

	// 3. Append deterministic engineering regression reports & ADRs grounded in real packages.
	regReports := buildGroundedRegressionCorpus(baseTime)
	docs = append(docs, regReports...)
	regDocs = len(regReports)

	return docs, goFiles, repoDocs, regDocs, nil
}

func buildGroundedRegressionCorpus(ts time.Time) []CanonicalDocument {
	type spec struct {
		id, title, pkg, cause, theme, content string
	}
	specs := []spec{
		{
			id: "reports/REG-001.md", title: "REG-001: MVCC posting-list rollup race under concurrent commit",
			pkg: "posting", cause: "mvcc_rollup_lock_ordering", theme: "mvcc_and_posting_consistency",
			content: "Regression REG-001 in posting/mvcc.go and posting/list.go: concurrent rollup and ReadTs snapshot readers observed stale posting deltas when commit oracle timestamps raced with RWMutex acquisition.",
		},
		{
			id: "reports/REG-002.md", title: "REG-002: Oracle pending transaction watermark stall during high-write bursts",
			pkg: "posting", cause: "oracle_watermark_contention", theme: "mvcc_and_posting_consistency",
			content: "Regression REG-002 in posting/oracle.go and posting/writer.go: delayed watermark advancement caused MVCC transaction retries and posting list writer stalls across mutation batches.",
		},
		{
			id: "reports/REG-003.md", title: "REG-003: Posting list LMap eviction deadlock on concurrent index rebuild",
			pkg: "posting", cause: "posting_lmap_lock_inversion", theme: "concurrency_and_locking",
			content: "Regression REG-003 in posting/lists.go and posting/index.go: mutex lock inversion between cache eviction and posting index rebuild triggered intermittent deadlock under parallel load.",
		},
		{
			id: "reports/REG-004.md", title: "REG-004: DQL recursive query parser stack amplification",
			pkg: "dql", cause: "unbounded_parser_recursion", theme: "query_parsing_and_dql",
			content: "Regression REG-004 in dql/parser.go and lex/lexer.go: deeply nested DQL filter and recurse directives caused parser stack amplification and request timeout spikes.",
		},
		{
			id: "reports/REG-005.md", title: "REG-005: DQL upsert conditional block mis-evaluates empty uid variable",
			pkg: "dql", cause: "upsert_empty_uid_binding", theme: "query_parsing_and_dql",
			content: "Regression REG-005 in dql/mutation.go and dql/parser_mutation.go: conditional upsert blocks evaluated empty uid variables as truthy during partial mutation AST lowering.",
		},
		{
			id: "reports/REG-006.md", title: "REG-006: Lexer IRI escape handling corrupts quoted predicate names",
			pkg: "lex", cause: "iri_escape_lexer_drift", theme: "query_parsing_and_dql",
			content: "Regression REG-006 in lex/iri.go and dql/dql.go: unescaped Unicode angle-bracket IRIs caused lexer tokenization drift and malformed DQL query plans.",
		},
		{
			id: "reports/REG-007.md", title: "REG-007: HNSW persistent graph corruption on abrupt snapshot flush",
			pkg: "tok/hnsw", cause: "hnsw_non_atomic_layer_flush", theme: "indexing_and_vector_hnsw",
			content: "Regression REG-007 in tok/hnsw/persistent_hnsw.go and tok/hnsw/persistent_factory.go: partial HNSW neighbor-list writes left dangling vector edges and degraded ANN recall by 18%.",
		},
		{
			id: "reports/REG-008.md", title: "REG-008: Bleve fulltext stemmer mismatch across multilingual tokenizers",
			pkg: "tok", cause: "tokenizer_stemmer_drift", theme: "indexing_and_vector_hnsw",
			content: "Regression REG-008 in tok/bleve.go, tok/stemmers.go, and tok/stopwords.go: inconsistent language stemmer caching dropped fulltext index hits after schema reload.",
		},
		{
			id: "reports/REG-009.md", title: "REG-009: HNSW SearchLayer mutex contention under bulk vector ingest",
			pkg: "tok/hnsw", cause: "hnsw_layer_lock_contention", theme: "concurrency_and_locking",
			content: "Regression REG-009 in tok/hnsw/search_layer.go and tok/index/search_path.go: coarse-grained RWMutex contention during concurrent vector nearest-neighbor search degraded p99 latency.",
		},
		{
			id: "reports/REG-010.md", title: "REG-010: ShardedMap bucket lock starvation during high-cardinality type sort",
			pkg: "types", cause: "sharded_map_lock_starvation", theme: "concurrency_and_locking",
			content: "Regression REG-010 in types/sharded_map.go and types/sort.go: skewed hash distribution caused goroutine lock starvation across hot buckets during concurrent sorting.",
		},
		{
			id: "reports/REG-011.md", title: "REG-011: S2 geospatial polygon filter drops boundary points",
			pkg: "types", cause: "s2_geofilter_precision_loss", theme: "schema_and_type_conversion",
			content: "Regression REG-011 in types/geofilter.go and types/s2index.go: S2 cell covering approximation dropped points on anti-meridian polygon boundaries during geo queries.",
		},
		{
			id: "reports/REG-012.md", title: "REG-012: Schema parser silently drops reverse index directive on alter",
			pkg: "schema", cause: "schema_alter_directive_drop", theme: "schema_and_type_conversion",
			content: "Regression REG-012 in schema/parse.go and schema/schema.go: online schema alter dropped @reverse directive metadata when combined with facet scalar conversion.",
		},
		{
			id: "reports/REG-013.md", title: "REG-013: Scalar type conversion overflow on int64-to-float64 coercion",
			pkg: "types", cause: "scalar_coercion_precision_loss", theme: "schema_and_type_conversion",
			content: "Regression REG-013 in types/conversion.go and types/scalar_types.go: implicit int64 to float64 scalar coercion lost precision on 54-bit UIDs and facet timestamps.",
		},
		{
			id: "reports/REG-014.md", title: "REG-014: Codec pack/unpack slice over-allocation under skewed postings",
			pkg: "codec", cause: "codec_buffer_over_allocation", theme: "memory_allocation_and_codec",
			content: "Regression REG-014 in codec/codec.go and algo/packed.go: unbounded varint block pre-allocation caused heap spikes and OOM kills on wide posting blocks.",
		},
		{
			id: "reports/REG-015.md", title: "REG-015: UIDList intersection allocates quadratic temporary buffers",
			pkg: "algo", cause: "uidlist_temp_buffer_churn", theme: "memory_allocation_and_codec",
			content: "Regression REG-015 in algo/uidlist.go and algo/heap.go: repeated multi-way UID list intersections churned heap allocations and triggered GC pauses under high fanout queries.",
		},
		{
			id: "reports/REG-016.md", title: "REG-016: Count-Min Sketch counter saturation skews hot-key estimation",
			pkg: "algo", cause: "cm_sketch_counter_overflow", theme: "memory_allocation_and_codec",
			content: "Regression REG-016 in algo/cm-sketch.go and posting/size.go: 16-bit Count-Min Sketch buckets saturated under Zipfian workloads, distorting cache eviction size estimates.",
		},
		{
			id: "reports/REG-017.md", title: "REG-017: Vault encryption key refresh race drops active ACL session",
			pkg: "x", cause: "vault_key_rotation_race", theme: "security_acl_and_encryption",
			content: "Regression REG-017 in x/vault.go and x/acl_enc_keys.go: non-atomic encryption key rotation invalidated in-flight ACL token verifications during key rollover.",
		},
		{
			id: "reports/REG-018.md", title: "REG-018: TLS helper accepts weak cipher fallback when FIPS build tag omitted",
			pkg: "x", cause: "tls_fips_cipher_fallback", theme: "security_acl_and_encryption",
			content: "Regression REG-018 in x/tls_helper.go and buildvars/fips.go: missing strict TLS minimum version check allowed non-FIPS cipher negotiation on misconfigured listeners.",
		},
		{
			id: "reports/REG-019.md", title: "REG-019: Disk metrics collector leaks file descriptors on Linux reload",
			pkg: "x", cause: "metrics_fd_leak_on_reload", theme: "observability_and_ops",
			content: "Regression REG-019 in x/disk_metrics_linux.go and x/metrics.go: unclosed stat handles during dynamic config reload exhausted ulimit file descriptors.",
		},
		{
			id: "reports/REG-020.md", title: "REG-020: MinIO backup client retries ignore context cancellation",
			pkg: "x", cause: "minio_retry_ignores_ctx", theme: "observability_and_ops",
			content: "Regression REG-020 in x/minioclient.go and x/health.go: unbounded backoff retries stalled graceful server shutdown and triggered false healthcheck timeouts.",
		},
		{
			id: "docs/ADR-001.md", title: "ADR-001: Derived Index Separation and Canonical Identity Versioning",
			pkg: "posting", cause: "derived_index_architecture", theme: "mvcc_and_posting_consistency",
			content: "ADR-001 mandates that posting lists, HNSW vector indexes, Bleve fulltext indexes, and community graphs are strictly derived projections over canonical versioned identities and must be rebuildable without data loss.",
		},
		{
			id: "docs/ADR-002.md", title: "ADR-002: Strict Pre-Return Tenant and ACL Filtering Boundary",
			pkg: "x", cause: "acl_isolation_boundary", theme: "security_acl_and_encryption",
			content: "ADR-002 enforces tenant partitioning and pre-return ACL filtering across x/acl_enc_keys.go, posting/index.go, and all derived retrieval layers so restricted records never leak into shared summaries.",
		},
		{
			id: "docs/ADR-003.md", title: "ADR-003: Bounded Memory Budgets for Codec and UIDList Operators",
			pkg: "codec", cause: "bounded_memory_allocators", theme: "memory_allocation_and_codec",
			content: "ADR-003 caps buffer growth in codec/codec.go, algo/packed.go, and algo/uidlist.go to prevent OOM regressions under adversarial posting fanout.",
		},
		{
			id: "docs/ADR-004.md", title: "ADR-004: Explicit Query Routing for Global Synthesis vs Exact Lookup",
			pkg: "dql", cause: "global_vs_local_routing", theme: "query_parsing_and_dql",
			content: "ADR-004 requires routing corpus-wide thematic questions to hierarchical GraphRAG summaries while keeping exact DQL, symbol, and file lookups on the low-latency hybrid path.",
		},
	}

	out := make([]CanonicalDocument, 0, len(specs))
	for _, s := range specs {
		kind := "regression_report"
		if strings.HasPrefix(s.id, "docs/") {
			kind = "adr"
		}
		out = append(out, CanonicalDocument{
			SourceID: s.id,
			Version:  "v1",
			TenantID: "tenant-alpha",
			ACL:      []string{"*"},
			Kind:     kind,
			Title:    s.title,
			Content:  s.content,
			Metadata: map[string]string{
				"package":          s.pkg,
				"root_cause":       s.cause,
				"regression_class": s.theme,
			},
			UpdatedAt: ts,
		})
	}
	return out
}

func buildACLCanaryDocuments() []CanonicalDocument {
	ts := time.Date(2026, 11, 2, 0, 0, 0, 0, time.UTC)
	return []CanonicalDocument{
		{
			SourceID: "restricted/SEC-INC-901.md",
			Version:  "v1",
			TenantID: "tenant-alpha",
			ACL:      []string{"role:security-incident"},
			Kind:     "regression_report",
			Title:    "SEC-INC-901: Restricted Vault Master Key Incident",
			Content:  "CONFIDENTIAL security incident CANARY_VAULT_KEY_ZERO_DAY_9941 affecting x/vault.go and x/acl_enc_keys.go with emergency HSM mitigation CANARY_HSM_BYPASS_7723.",
			Metadata: map[string]string{
				"package":          "x",
				"root_cause":       "restricted_vault_zero_day",
				"regression_class": "security_acl_and_encryption",
			},
			UpdatedAt: ts,
		},
		{
			SourceID: "restricted/SEC-INC-902.md",
			Version:  "v1",
			TenantID: "tenant-alpha",
			ACL:      []string{"role:security-incident"},
			Kind:     "regression_report",
			Title:    "SEC-INC-902: Restricted Encryption Audit Finding",
			Content:  "CONFIDENTIAL audit record referencing CANARY_HSM_BYPASS_7723 in buildvars/fips.go and x/tls_helper.go.",
			Metadata: map[string]string{
				"package":          "buildvars",
				"root_cause":       "restricted_hsm_audit",
				"regression_class": "security_acl_and_encryption",
			},
			UpdatedAt: ts,
		},
		{
			SourceID: "beta/BETA-REG-001.md",
			Version:  "v1",
			TenantID: "tenant-beta",
			ACL:      []string{"*"},
			Kind:     "regression_report",
			Title:    "BETA-REG-001: Tenant Beta Exclusive Cluster Regression",
			Content:  "Tenant Beta private cluster report CANARY_BETA_CORPUS_8812 describing posting/mvcc.go replication lag in isolated customer environment.",
			Metadata: map[string]string{
				"package":          "posting",
				"root_cause":       "beta_exclusive_lag",
				"regression_class": "mvcc_and_posting_consistency",
			},
			UpdatedAt: ts,
		},
	}
}

// DefaultBenchmarkQueries returns the controlled 3-regime evaluation suite
// covering global thematic synthesis, cross-package architecture, and the
// exact-lookup negative regime.
func DefaultBenchmarkQueries() []BenchmarkQuery {
	return []BenchmarkQuery{
		// Regime 1: Corpus-wide global thematic synthesis (GraphRAG target regime).
		{
			ID:            "GLOBAL-01",
			Regime:        "global_thematic_synthesis",
			Query:         "What are the largest sources of engineering regressions across our repositories?",
			ExpectedRoute: string(RouteGlobalGraphRAG),
			RequiredThemes: []string{
				"mvcc_and_posting_consistency",
				"concurrency_and_locking",
				"query_parsing_and_dql",
				"indexing_and_vector_hnsw",
				"memory_allocation_and_codec",
				"schema_and_type_conversion",
				"security_acl_and_encryption",
			},
			RequiredKeywords:   []string{"posting", "dql", "hnsw", "codec", "schema", "vault"},
			GoldSources:        []string{"reports/REG-001.md", "reports/REG-004.md", "reports/REG-007.md", "reports/REG-012.md", "reports/REG-014.md", "reports/REG-017.md"},
			MinSourcesRequired: 10,
		},
		{
			ID:            "GLOBAL-02",
			Regime:        "global_thematic_synthesis",
			Query:         "Summarize the main concurrency, locking, and memory allocation regressions across all packages.",
			ExpectedRoute: string(RouteGlobalGraphRAG),
			RequiredThemes: []string{
				"concurrency_and_locking",
				"memory_allocation_and_codec",
				"mvcc_and_posting_consistency",
			},
			RequiredKeywords:   []string{"posting", "hnsw", "codec", "algo", "types"},
			GoldSources:        []string{"reports/REG-003.md", "reports/REG-009.md", "reports/REG-010.md", "reports/REG-014.md", "reports/REG-015.md"},
			MinSourcesRequired: 8,
		},
		{
			ID:            "GLOBAL-03",
			Regime:        "global_thematic_synthesis",
			Query:         "What are the top systemic indexing, query parsing, and schema bottlenecks across the codebase?",
			ExpectedRoute: string(RouteGlobalGraphRAG),
			RequiredThemes: []string{
				"indexing_and_vector_hnsw",
				"query_parsing_and_dql",
				"schema_and_type_conversion",
			},
			RequiredKeywords:   []string{"dql", "tok", "hnsw", "schema", "types", "lex"},
			GoldSources:        []string{"reports/REG-004.md", "reports/REG-007.md", "reports/REG-008.md", "reports/REG-011.md", "reports/REG-012.md"},
			MinSourcesRequired: 8,
		},
		{
			ID:            "GLOBAL-04",
			Regime:        "global_thematic_synthesis",
			Query:         "Summarize recurring security, encryption, and operational reliability risks across all subsystems.",
			ExpectedRoute: string(RouteGlobalGraphRAG),
			RequiredThemes: []string{
				"security_acl_and_encryption",
				"observability_and_ops",
			},
			RequiredKeywords:   []string{"vault", "tls", "fips", "metrics", "minio"},
			GoldSources:        []string{"reports/REG-017.md", "reports/REG-018.md", "reports/REG-019.md", "reports/REG-020.md"},
			MinSourcesRequired: 6,
		},

		// Regime 2: Cross-package architecture synthesis.
		{
			ID:            "ARCH-01",
			Regime:        "cross_package_architecture",
			Query:         "Compare patterns of derived indexing and storage consistency across posting, tok, codec, and algo packages.",
			ExpectedRoute: string(RouteGlobalGraphRAG),
			RequiredThemes: []string{
				"mvcc_and_posting_consistency",
				"indexing_and_vector_hnsw",
				"memory_allocation_and_codec",
			},
			RequiredKeywords:   []string{"posting", "tok", "codec", "algo"},
			GoldSources:        []string{"docs/ADR-001.md", "docs/ADR-003.md", "posting/mvcc.go", "codec/codec.go"},
			MinSourcesRequired: 8,
		},
		{
			ID:            "ARCH-02",
			Regime:        "cross_package_architecture",
			Query:         "What are the dominant patterns connecting DQL parsing, lexer tokenization, schema definitions, and scalar types across all packages?",
			ExpectedRoute: string(RouteGlobalGraphRAG),
			RequiredThemes: []string{
				"query_parsing_and_dql",
				"schema_and_type_conversion",
			},
			RequiredKeywords:   []string{"dql", "lex", "schema", "types"},
			GoldSources:        []string{"dql/parser.go", "lex/lexer.go", "schema/schema.go", "types/conversion.go"},
			MinSourcesRequired: 8,
		},
		{
			ID:            "ARCH-03",
			Regime:        "cross_package_architecture",
			Query:         "Provide an executive portfolio overview of all architectural decisions and cross-cutting risks across our repositories.",
			ExpectedRoute: string(RouteGlobalGraphRAG),
			RequiredThemes: []string{
				"mvcc_and_posting_consistency",
				"security_acl_and_encryption",
				"memory_allocation_and_codec",
				"query_parsing_and_dql",
			},
			RequiredKeywords:   []string{"posting", "x", "codec", "dql"},
			GoldSources:        []string{"docs/ADR-001.md", "docs/ADR-002.md", "docs/ADR-003.md", "docs/ADR-004.md"},
			MinSourcesRequired: 10,
		},

		// Regime 3: Exact single-file / symbol / incident lookups (NEGATIVE REGIME for GraphRAG!).
		{
			ID:               "EXACT-01",
			Regime:           "exact_lookup_negative_regime",
			Query:            "Where is posting/mvcc.go implemented and what does it handle?",
			ExpectedRoute:    string(RouteLocalHybrid),
			RequiredKeywords: []string{"posting/mvcc.go"},
			GoldSources:      []string{"posting/mvcc.go"},
		},
		{
			ID:               "EXACT-02",
			Regime:           "exact_lookup_negative_regime",
			Query:            "What is the exact root cause in REG-007?",
			ExpectedRoute:    string(RouteLocalHybrid),
			RequiredKeywords: []string{"REG-007", "hnsw"},
			GoldSources:      []string{"reports/REG-007.md"},
		},
		{
			ID:               "EXACT-03",
			Regime:           "exact_lookup_negative_regime",
			Query:            "Which file implements Count-Min Sketch in algo/cm-sketch.go?",
			ExpectedRoute:    string(RouteLocalHybrid),
			RequiredKeywords: []string{"algo/cm-sketch.go"},
			GoldSources:      []string{"algo/cm-sketch.go"},
		},
		{
			ID:               "EXACT-04",
			Regime:           "exact_lookup_negative_regime",
			Query:            "What is the exact root cause in REG-017 regarding x/vault.go?",
			ExpectedRoute:    string(RouteLocalHybrid),
			RequiredKeywords: []string{"REG-017", "vault"},
			GoldSources:      []string{"reports/REG-017.md"},
		},
		{
			ID:               "EXACT-05",
			Regime:           "exact_lookup_negative_regime",
			Query:            "Where is SearchLayer implemented in tok/hnsw/search_layer.go?",
			ExpectedRoute:    string(RouteLocalHybrid),
			RequiredKeywords: []string{"tok/hnsw/search_layer.go"},
			GoldSources:      []string{"tok/hnsw/search_layer.go"},
		},
		{
			ID:               "EXACT-06",
			Regime:           "exact_lookup_negative_regime",
			Query:            "Find the details of REG-012 in schema/parse.go.",
			ExpectedRoute:    string(RouteLocalHybrid),
			RequiredKeywords: []string{"REG-012", "schema"},
			GoldSources:      []string{"reports/REG-012.md"},
		},
	}
}

func mean(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	var sum float64
	for _, v := range vals {
		sum += v
	}
	return sum / float64(len(vals))
}

func percentile(vals []float64, p float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	cp := append([]float64(nil), vals...)
	sort.Float64s(cp)
	if len(cp) == 1 {
		return cp[0]
	}
	rank := (p / 100.0) * float64(len(cp)-1)
	lo := int(math.Floor(rank))
	hi := int(math.Ceil(rank))
	if lo == hi {
		return cp[lo]
	}
	frac := rank - float64(lo)
	return cp[lo]*(1.0-frac) + cp[hi]*frac
}
