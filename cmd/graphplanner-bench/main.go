/*
 * SPDX-License-Identifier: Apache-2.0
 */

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/hanzoai/graph/graphplanner"
)

func main() {
	repoRoot := flag.String("repo-root", ".", "Path to canonical hanzoai/graph repository root for real Go AST code-graph extraction")
	externalDataset := flag.String("external-dataset", "", "Optional path to external STaRK/GraphRAG-Bench JSON WorkloadBundle")
	seed := flag.Int64("seed", 1347, "Deterministic PRNG seed")
	topK := flag.Int("top-k", 5, "Retrieval TopK budget shared across all compared strategies")
	iters := flag.Int("iters", 10, "Timed iterations per query (median latency recorded)")
	outDir := flag.String("out-dir", "docs/research/hip-1347-graphplanner", "Output directory for results.json and optional README.md")
	writeDossier := flag.Bool("write-dossier", true, "Write results.json and 6-point HIP-1347 README.md into out-dir")
	flag.Parse()

	opts := graphplanner.BenchmarkOptions{
		RepoRoot:            *repoRoot,
		ExternalDatasetPath: *externalDataset,
		Seed:                *seed,
		TopK:                *topK,
		Iterations:          *iters,
	}

	report, err := graphplanner.RunBenchmarkSuite(opts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "graphplanner-bench failed: %v\n", err)
		os.Exit(1)
	}

	if *writeDossier {
		if err := os.MkdirAll(*outDir, 0o755); err != nil {
			fmt.Fprintf(os.Stderr, "mkdir %s failed: %v\n", *outDir, err)
			os.Exit(1)
		}
		jsonBytes, err := json.MarshalIndent(report, "", "  ")
		if err != nil {
			fmt.Fprintf(os.Stderr, "marshal results.json failed: %v\n", err)
			os.Exit(1)
		}
		jsonPath := filepath.Join(*outDir, "results.json")
		if err := os.WriteFile(jsonPath, append(jsonBytes, '\n'), 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "write %s failed: %v\n", jsonPath, err)
			os.Exit(1)
		}
		md := graphplanner.RenderMarkdownDossier(report)
		mdPath := filepath.Join(*outDir, "README.md")
		if err := os.WriteFile(mdPath, []byte(md), 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "write %s failed: %v\n", mdPath, err)
			os.Exit(1)
		}
		fmt.Printf("Wrote %s and %s\n", jsonPath, mdPath)
	}

	fmt.Printf("Corpus: %d nodes, %d edges (%d real Go files) | CSR %.1f B/node, %.1f B/edge | Rebuild bit-exact=%v (%.3f ms)\n",
		report.Storage.IndexedNodes, report.Storage.IndexedEdges, report.RealCodeFilesParsed,
		report.Storage.BytesPerNode, report.Storage.BytesPerEdge,
		report.Rebuild.BitExactDigest, report.Rebuild.RebuildTimeMs)
	fmt.Printf("ACL Audit: %d probes, %d leaks, foreign_nodes=%d, foreign_edges=%d, cross_scope_cache_isolated=%v, post_revocation_invalidated=%v\n",
		report.Security.ProbesExecuted, report.Security.LeaksDetected,
		report.Security.ForeignNodesInspected, report.Security.ForeignEdgesTraversed,
		report.Security.CrossScopeCacheIsolated, report.Security.PostRevocationInvalidated)
	fmt.Println("\nOverall Strategy Comparison:")
	for _, s := range report.OverallStrategies {
		fmt.Printf("  %-28s Recall@K=%5.1f%%  MRR=%.3f  Task=%5.1f%%  SLA=%5.1f%%  p50=%.4fms  p95=%.4fms  Cost=$%6.2f  Leaks=%d\n",
			s.Strategy, s.RecallAtK*100, s.MRR, s.TaskSuccessRate*100, s.SLAComplianceRate*100,
			s.LatencyP50Ms, s.LatencyP95Ms, s.CostPer1MUsefulTokens, s.ACLLeaks)
	}
}
