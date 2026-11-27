/*
 * SPDX-FileCopyrightText: © 2026 Hanzo AI, Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

// Command lightrag-bench runs the reproducible HIP-1347 LightRAG incremental
// entity-relationship indexing and dual-level retrieval benchmark suite over
// the real Hanzo Graph repository corpus and writes raw JSON measurements.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/hanzoai/graph/lightrag"
)

func main() {
	cfg := lightrag.DefaultBenchmarkConfig()
	outJSON := "docs/research/hip-1347-lightrag/results.json"

	flag.StringVar(&cfg.RepoRoot, "repo-root", cfg.RepoRoot, "Path to Hanzo Graph repository root")
	flag.StringVar(&outJSON, "out", outJSON, "Output path for raw JSON benchmark results")
	flag.IntVar(&cfg.TopK, "top-k", cfg.TopK, "Retrieval Top-K budget (shared across all systems)")
	flag.IntVar(&cfg.Iters, "iters", cfg.Iters, "Timing iterations per query for latency percentiles")
	flag.Int64Var(&cfg.Seed, "seed", cfg.Seed, "Deterministic benchmark seed")
	flag.Parse()

	report, err := lightrag.RunBenchmarkSuite(context.Background(), cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "lightrag-bench error: %v\n", err)
		os.Exit(1)
	}

	if outJSON != "" {
		if err := os.MkdirAll(filepath.Dir(outJSON), 0o755); err != nil {
			fmt.Fprintf(os.Stderr, "mkdir %s: %v\n", filepath.Dir(outJSON), err)
			os.Exit(1)
		}
		raw, err := json.MarshalIndent(report, "", "  ")
		if err != nil {
			fmt.Fprintf(os.Stderr, "marshal results: %v\n", err)
			os.Exit(1)
		}
		raw = append(raw, '\n')
		if err := os.WriteFile(outJSON, raw, 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "write %s: %v\n", outJSON, err)
			os.Exit(1)
		}
	}

	printSummary(report, outJSON)
}

func printSummary(r lightrag.BenchmarkReport, outPath string) {
	fmt.Printf("=== HIP-1347 LightRAG Incremental Indexing Benchmark ===\n")
	fmt.Printf("Host: %s (%s/%s, %d CPUs) | Seed: %d | TopK: %d | Iters: %d\n",
		r.GoVersion, r.GOOS, r.GOARCH, r.NumCPU, r.Seed, r.TopK, r.Iters)
	fmt.Printf("Corpus: %d sources (%d Go files, %d docs, %d ops artifacts; %d bytes, %d chunks)\n",
		r.Corpus.TotalSources, r.Corpus.RealGoSourceFiles, r.Corpus.RealDocFiles,
		r.Corpus.OperationalArtifacts, r.Corpus.TotalBytes, r.Corpus.TotalChunks)
	fmt.Printf("Entities: %d raw -> %d canonical (%d collapsed, %.2f%% duplicate rate)\n",
		r.Corpus.RawEntitiesExtracted, r.Corpus.UniqueCanonicalEntities,
		r.Corpus.DuplicateEntitiesMerged, r.Corpus.DuplicateEntityRatePct)
	fmt.Printf("Relations: %d raw -> %d canonical | Proxy Extract Cost: $%.6f | Rels/$: %.2f\n\n",
		r.Corpus.RawRelationsExtracted, r.Corpus.UniqueCanonicalRelations,
		r.Corpus.EstimatedExtractCostUSD, r.Corpus.RelationsPerDollar)

	fmt.Printf("Incremental Delta (%d ops: %d upd / %d ins / %d del) vs Full Rebuild:\n",
		r.Incremental.IncrementalOps, r.Incremental.UpdateOps, r.Incremental.InsertOps, r.Incremental.DeleteOps)
	fmt.Printf("  Incremental latency (ms): p50=%.3f  p95=%.3f  p99=%.3f  mean=%.3f\n",
		r.Incremental.IncrementalLatency.P50Ms, r.Incremental.IncrementalLatency.P95Ms,
		r.Incremental.IncrementalLatency.P99Ms, r.Incremental.IncrementalLatency.MeanMs)
	fmt.Printf("  Full rebuild latency (ms): p50=%.3f  p95=%.3f  p99=%.3f  mean=%.3f\n",
		r.Incremental.FullRebuildLatency.P50Ms, r.Incremental.FullRebuildLatency.P95Ms,
		r.Incremental.FullRebuildLatency.P99Ms, r.Incremental.FullRebuildLatency.MeanMs)
	fmt.Printf("  Speedup: %.2fx (p50) / %.2fx (mean) | Mean Churn: %.2f%% (max %.2f%%)\n",
		r.Incremental.SpeedupP50, r.Incremental.SpeedupMean,
		r.Incremental.MeanSubgraphChurnPct, r.Incremental.MaxSubgraphChurnPct)
	fmt.Printf("  Bit-Exact Digest Parity: %v (inc=%s == rebuild=%s) | Drop Enforced: %v\n\n",
		r.Incremental.BitExactEquivalence, r.Incremental.DigestAfterIncremental,
		r.Incremental.DigestAfterFullRebuild, r.Incremental.DropEnforcesNotBuiltErr)

	fmt.Printf("Overall Retrieval Comparison (%d queries, TopK=%d):\n", r.OverallSystems[0].Queries, r.TopK)
	fmt.Printf("%-18s %9s %8s %8s %8s %9s %9s %9s %10s\n",
		"System", "Recall@K", "MRR@K", "AnsF1", "AnsEM", "TaskSucc", "p50(ms)", "p99(ms)", "$/1MUse")
	for _, s := range r.OverallSystems {
		fmt.Printf("%-18s %8.1f%% %7.3f %7.1f%% %7.1f%% %8.1f%% %9.3f %9.3f %9.2f\n",
			s.Mode, s.RecallAtK*100, s.MRRAtK, s.AnswerF1*100, s.AnswerEM*100,
			s.TaskSuccessRate*100, s.Latency.P50Ms, s.Latency.P99Ms, s.CostPer1MUsefulUS)
	}
	fmt.Println()

	for _, wl := range r.Workloads {
		fmt.Printf("Workload: %s (%d queries)\n", wl.Workload, wl.Queries)
		for _, s := range wl.Systems {
			fmt.Printf("  %-18s Recall@K=%5.1f%% MRR=%.3f F1=%5.1f%% EM=%5.1f%% Task=%5.1f%% p50=%.3fms p99=%.3fms\n",
				s.Mode, s.RecallAtK*100, s.MRRAtK, s.AnswerF1*100, s.AnswerEM*100,
				s.TaskSuccessRate*100, s.Latency.P50Ms, s.Latency.P99Ms)
		}
	}
	fmt.Printf("\nFreshness: cache_invalidated=%v | updated_recall=%.1f%% | deleted_ghost_recall=%.1f%%\n",
		r.Freshness.CacheInvalidatedOnEdit, r.Freshness.UpdatedRecallRate*100, r.Freshness.DeletedGhostRecallRate*100)
	fmt.Printf("Security ACL Isolation: lightrag_leaks=%d/%d (%.1f%%) vs unfiltered_control=%d/%d (%.1f%%)\n",
		r.Security.LightRAGLeakCount, r.Security.ProbesExecuted, r.Security.LightRAGLeakRatePct,
		r.Security.UnfilteredControlLeaks, r.Security.ProbesExecuted, r.Security.UnfilteredControlRatePct)
	if outPath != "" {
		fmt.Printf("Wrote raw JSON results to %s\n", outPath)
	}
}
