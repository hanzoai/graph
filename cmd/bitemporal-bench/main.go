/*
 * SPDX-FileCopyrightText: © 2026 Hanzo AI, Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/hanzoai/graph/bitemporal"
)

func main() {
	outDir := flag.String("out-dir", "docs/research/hip-1347-graphiti", "directory to write README.md, score_table.md, and raw_results.json")
	iters := flag.Int("iters", 25, "timed iterations per query for latency measurement")
	flag.Parse()

	report, err := bitemporal.RunComparativeBenchmark(*iters)
	if err != nil {
		fmt.Fprintf(os.Stderr, "benchmark failed: %v\n", err)
		os.Exit(1)
	}

	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "failed to create output directory %q: %v\n", *outDir, err)
		os.Exit(1)
	}

	rawJSON, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to marshal raw results: %v\n", err)
		os.Exit(1)
	}
	rawJSON = append(rawJSON, '\n')

	scoreTable := bitemporal.RenderScoreTableMarkdown(report)
	readme := bitemporal.RenderResearchReadmeMarkdown(report)

	files := map[string][]byte{
		"raw_results.json": rawJSON,
		"score_table.md":   []byte(scoreTable),
		"README.md":        []byte(readme),
	}
	for name, content := range files {
		target := filepath.Join(*outDir, name)
		if err := os.WriteFile(target, content, 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "failed to write %s: %v\n", target, err)
			os.Exit(1)
		}
	}

	fmt.Print(scoreTable)
	fmt.Printf("\nWrote artifacts to %s (README.md, score_table.md, raw_results.json)\n", *outDir)
}
