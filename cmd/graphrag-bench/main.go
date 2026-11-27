package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/hanzoai/graph/graphrag"
)

func main() {
	repoRoot := flag.String("repo-root", ".", "Path to hanzoai/graph repository root")
	externalDataset := flag.String("external-dataset", "", "Optional path to external JSON corpus/query bundle")
	seed := flag.Int64("seed", 1347, "Deterministic benchmark seed")
	topK := flag.Int("top-k", 5, "TopK retrieval budget for local hybrid baseline")
	tokenBudget := flag.Int("token-budget", 4000, "Synthesis context token budget")
	iters := flag.Int("iters", 5, "Iterations per query for p50/p95 latency measurement")
	outJSON := flag.String("out-json", "docs/research/hip-1347-graphrag/results.json", "Path to write raw JSON benchmark receipts")
	flag.Parse()

	report, err := graphrag.RunBenchmark(graphrag.BenchmarkConfig{
		RepoRoot:        *repoRoot,
		ExternalDataset: *externalDataset,
		Seed:            *seed,
		TopK:            *topK,
		TokenBudget:     *tokenBudget,
		Iters:           *iters,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "graphrag-bench failed: %v\n", err)
		os.Exit(1)
	}

	if *outJSON != "" {
		if err := os.MkdirAll(filepath.Dir(*outJSON), 0o755); err != nil {
			fmt.Fprintf(os.Stderr, "create output dir failed: %v\n", err)
			os.Exit(1)
		}
		raw, err := json.MarshalIndent(report, "", "  ")
		if err != nil {
			fmt.Fprintf(os.Stderr, "marshal report failed: %v\n", err)
			os.Exit(1)
		}
		raw = append(raw, '\n')
		if err := os.WriteFile(*outJSON, raw, 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "write %s failed: %v\n", *outJSON, err)
			os.Exit(1)
		}
	}

	fmt.Printf("GraphRAG HIP-1347 Benchmark (%s, %s/%s, seed=%d)\n", report.DatasetName, report.GoVersion, report.GOOS+"/"+report.GOARCH, report.Seed)
	fmt.Printf("Corpus: %d real Go files + %d repo docs + %d regression/ADR records = %d canonical docs (incl. ACL canaries)\n",
		report.RealGoFilesParsed, report.RepoDocsParsed, report.RegressionDocsCount, report.TotalCanonicalDocs)
	fmt.Printf("Offline Build: %.2f ms | Entities=%d Relations=%d | L0=%d L1=%d L2=%d | BuildTokens=%d ($%.5f) | Q_L0=%.3f Q_L1=%.3f\n",
		report.BuildCost.DurationMs,
		report.BuildCost.EntitiesTotal,
		report.BuildCost.RelationsTotal,
		report.BuildCost.CommunitiesByLevel[0],
		report.BuildCost.CommunitiesByLevel[1],
		report.BuildCost.CommunitiesByLevel[2],
		report.BuildCost.TotalTokens,
		report.BuildCost.EstimatedCostUSD,
		report.BuildCost.MeanModularityL0,
		report.BuildCost.MeanModularityL1,
	)

	fmt.Println("\n--- Regime Results ---")
	fmt.Printf("%-28s %-22s %7s %7s %7s %7s %8s %9s %8s %5s\n",
		"Regime", "Strategy", "AccF1", "Cov", "SrcRec", "TaskOK", "AvgTok", "p50/p95ms", "Cost($)", "Leaks")
	for _, m := range report.RegimeMetrics {
		fmt.Printf("%-28s %-22s %6.1f%% %6.1f%% %6.1f%% %6.1f%% %8.0f %4.2f/%4.2f %8.5f %5d\n",
			m.Regime, m.Strategy,
			m.SynthesisAccuracyF1*100, m.ThemeCoverage*100, m.SourceRecall*100, m.TaskSuccessRate*100,
			m.AvgTotalTokens, m.LatencyP50Ms, m.LatencyP95Ms, m.AvgCostUSD, m.ACLLeaks)
	}

	fmt.Println("\n--- Incremental Refresh vs Full Rebuild ---")
	fmt.Printf("%-24s %9s %9s %8s %9s %10s\n", "Case", "Inc(ms)", "Full(ms)", "Speedup", "Reused%", "TokSave%")
	for _, c := range report.IncrementalRefresh {
		fmt.Printf("%-24s %9.3f %9.3f %7.2fx %8.1f%% %9.1f%%\n",
			c.CaseName, c.IncrementalDurationMs, c.FullRebuildDurationMs, c.SpeedupVsFullRebuild, c.CommunitiesReusedPct, c.SummaryTokenSavingsPct)
	}

	fmt.Printf("\n--- ACL & Parity Audit ---\n")
	fmt.Printf("Probes=%d | GlobalLeaks=%d | LocalLeaks=%d | CrossTenant=%d | OracleLeaks=%d | PostRevokeLeaks=%d | DropRebuildParity=%v\n",
		report.ACLAudit.TotalProbes,
		report.ACLAudit.CanaryLeaksInGlobalSummaries,
		report.ACLAudit.CanaryLeaksInLocalHits,
		report.ACLAudit.CrossTenantPartitionsLeaked,
		report.ACLAudit.ExistenceOracleLeaks,
		report.ACLAudit.PostRevocationLeaks,
		report.ACLAudit.DropRebuildBitExactParity,
	)
	if *outJSON != "" {
		fmt.Printf("\nWrote raw JSON receipts to %s\n", *outJSON)
	}
}
