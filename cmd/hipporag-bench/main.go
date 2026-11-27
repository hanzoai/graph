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

	"github.com/hanzoai/graph/hipporag"
)

func main() {
	outDir := flag.String("out-dir", "docs/research/hip-1347-hipporag2", "Directory where dossier files are written when --write-dossier is set")
	iters := flag.Int("iters", 15, "Number of timed latency iterations per query")
	topK := flag.Int("top-k", 5, "Matched retrieval budget (TopK) across all systems")
	pricePer1M := flag.Float64("price-per-1m-tokens", 3.00, "Unit price (USD) per 1M input tokens for cost/1M useful tokens metric")
	externalDataset := flag.String("external-dataset", "", "Optional path to an external HotpotQA JSON, MuSiQue JSONL, or HIP-1347 JSON dataset")
	writeDossier := flag.Bool("write-dossier", false, "Write README.md, score_table.md, and raw_results.json into --out-dir")
	flag.Parse()

	opts := hipporag.BenchmarkOptions{
		TopK:                *topK,
		TimedIterations:     *iters,
		InputPricePer1MUSD:  *pricePer1M,
		ExternalDatasetPath: *externalDataset,
	}

	report, err := hipporag.RunBenchmark(opts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "hipporag-bench failed: %v\n", err)
		os.Exit(1)
	}

	rawJSON, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "marshal raw_results.json: %v\n", err)
		os.Exit(1)
	}
	rawJSON = append(rawJSON, '\n')

	scoreTableMD := hipporag.RenderScoreTableMarkdown(report)
	readmeMD := hipporag.RenderReportREADME(report)

	if *writeDossier {
		if err := os.MkdirAll(*outDir, 0o755); err != nil {
			fmt.Fprintf(os.Stderr, "mkdir %s: %v\n", *outDir, err)
			os.Exit(1)
		}
		files := map[string][]byte{
			filepath.Join(*outDir, "raw_results.json"): rawJSON,
			filepath.Join(*outDir, "score_table.md"):   []byte(scoreTableMD),
			filepath.Join(*outDir, "README.md"):        []byte(readmeMD),
		}
		for path, content := range files {
			if err := os.WriteFile(path, content, 0o644); err != nil {
				fmt.Fprintf(os.Stderr, "write %s: %v\n", path, err)
				os.Exit(1)
			}
		}
		fmt.Printf("Wrote HIP-1347 dossier to %s\n\n", *outDir)
	}

	fmt.Print(scoreTableMD)
}
