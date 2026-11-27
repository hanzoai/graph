/*
 * SPDX-License-Identifier: Apache-2.0
 */

package graphplanner

import (
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const (
	DefaultTenantID = "tenant-hanzo"
	ForeignTenantID = "tenant-external-beta"
)

// BenchmarkQuery defines a single reproducible evaluation query with ground-truth
// evidence identities and expected answer token(s).
type BenchmarkQuery struct {
	ID            string        `json:"id"`
	Category      string        `json:"category"`
	ExpectedPlan  ExecutionPlan `json:"expected_plan,omitempty"`
	Request       QueryRequest  `json:"request"`
	GoldSourceIDs []string      `json:"gold_source_ids"`
	GoldAnswer    string        `json:"gold_answer"`
	// WarmBeforeRun indicates that the benchmark should pre-warm this anchor
	// once before timed measurement (modeling a hot operational neighborhood).
	WarmBeforeRun bool `json:"warm_before_run,omitempty"`
}

// WorkloadBundle packages canonical nodes, canonical edges, and benchmark
// queries for deterministic evaluation.
type WorkloadBundle struct {
	Name          string           `json:"name"`
	Seed          int64            `json:"seed"`
	RepoRoot      string           `json:"repo_root,omitempty"`
	RealCodeFiles int              `json:"real_code_files"`
	Nodes         []CanonicalNode  `json:"nodes"`
	Edges         []CanonicalEdge  `json:"edges"`
	Queries       []BenchmarkQuery `json:"queries"`
}

// LoadExternalWorkload reads a JSON WorkloadBundle from disk so researchers can
// evaluate external STaRK or GraphRAG-Bench formatted datasets at runtime.
func LoadExternalWorkload(path string) (WorkloadBundle, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return WorkloadBundle{}, fmt.Errorf("graphplanner: read external workload %s: %w", path, err)
	}
	var bundle WorkloadBundle
	if err := json.Unmarshal(raw, &bundle); err != nil {
		return WorkloadBundle{}, fmt.Errorf("graphplanner: parse external workload %s: %w", path, err)
	}
	if len(bundle.Nodes) == 0 || len(bundle.Queries) == 0 {
		return WorkloadBundle{}, fmt.Errorf("graphplanner: external workload %s must contain nodes and queries", path)
	}
	if bundle.Name == "" {
		bundle.Name = filepath.Base(path)
	}
	return bundle, nil
}

// BuildCombinedBenchmarkSuite constructs the full HIP-1347 benchmark suite:
//  1. Real versioned Go engineering code graph extracted via go/ast from repoRoot
//  2. STaRK-structured semi-structured multi-hop workload
//  3. GraphRAG-Bench simple factual single-hop negative regime
//  4. High-degree hub explosion regime
//  5. Tight latency budget hot-cache regime
//  6. Multi-tenant & ACL isolation probes
func BuildCombinedBenchmarkSuite(repoRoot string, seed int64) (WorkloadBundle, error) {
	if seed == 0 {
		seed = 1347
	}
	rng := rand.New(rand.NewSource(seed))

	bundle := WorkloadBundle{
		Name:     "hip-1347-graphplanner-suite",
		Seed:     seed,
		RepoRoot: repoRoot,
	}

	// 1. Real Go engineering code graph extracted directly from hanzoai/graph.
	codeNodes, codeEdges, codeQueries, codeFileCount, err := buildRealHanzoGraphCodeCorpus(repoRoot)
	if err != nil {
		return WorkloadBundle{}, err
	}
	bundle.RealCodeFiles = codeFileCount
	bundle.Nodes = append(bundle.Nodes, codeNodes...)
	bundle.Edges = append(bundle.Edges, codeEdges...)
	bundle.Queries = append(bundle.Queries, codeQueries...)

	// 2. STaRK-structured semi-structured multi-hop workload.
	stNodes, stEdges, stQueries := buildSTaRKSemiStructuredWorkload(rng)
	bundle.Nodes = append(bundle.Nodes, stNodes...)
	bundle.Edges = append(bundle.Edges, stEdges...)
	bundle.Queries = append(bundle.Queries, stQueries...)

	// 3. GraphRAG-Bench simple factual single-hop negative regime.
	factNodes, factEdges, factQueries := buildSimpleFactualNegativeRegime()
	bundle.Nodes = append(bundle.Nodes, factNodes...)
	bundle.Edges = append(bundle.Edges, factEdges...)
	bundle.Queries = append(bundle.Queries, factQueries...)

	// 4. High-degree hub explosion regime.
	hubNodes, hubEdges, hubQueries := buildHubExplosionRegime(rng)
	bundle.Nodes = append(bundle.Nodes, hubNodes...)
	bundle.Edges = append(bundle.Edges, hubEdges...)
	bundle.Queries = append(bundle.Queries, hubQueries...)

	// 5. Tight latency budget hot-cache regime.
	hotNodes, hotEdges, hotQueries := buildTightBudgetHotCacheRegime()
	bundle.Nodes = append(bundle.Nodes, hotNodes...)
	bundle.Edges = append(bundle.Edges, hotEdges...)
	bundle.Queries = append(bundle.Queries, hotQueries...)

	// 6. Multi-tenant & ACL isolation probes.
	aclNodes, aclEdges, aclQueries := buildACLIsolationProbes()
	bundle.Nodes = append(bundle.Nodes, aclNodes...)
	bundle.Edges = append(bundle.Edges, aclEdges...)
	bundle.Queries = append(bundle.Queries, aclQueries...)

	return bundle, nil
}

// buildRealHanzoGraphCodeCorpus parses actual Go source files from the
// canonical hanzoai/graph repository into versioned nodes and typed edges.
func buildRealHanzoGraphCodeCorpus(repoRoot string) ([]CanonicalNode, []CanonicalEdge, []BenchmarkQuery, int, error) {
	if strings.TrimSpace(repoRoot) == "" {
		return nil, nil, nil, 0, nil
	}
	targetDirs := []string{"algo", "buildvars", "codec", "dql", "lex", "posting", "schema", "tok", "types", "x"}
	fset := token.NewFileSet()

	var nodes []CanonicalNode
	var edges []CanonicalEdge
	var queries []BenchmarkQuery

	seenPkg := make(map[string]bool)
	pkgFiles := make(map[string][]string)
	pkgSymbols := make(map[string][]string)
	pkgImports := make(map[string]map[string]bool)
	fileSymbols := make(map[string][]string)
	symbolSummary := make(map[string]string)

	fileCount := 0
	edgeSeq := 0
	nextEdgeID := func(prefix string) string {
		edgeSeq++
		return fmt.Sprintf("code_edge_%s_%05d", prefix, edgeSeq)
	}

	const modPrefix = "github.com/hanzoai/graph/"

	for _, dir := range targetDirs {
		fullDir := filepath.Join(repoRoot, dir)
		if info, err := os.Stat(fullDir); err != nil || !info.IsDir() {
			continue
		}
		var goFiles []string
		err := filepath.Walk(fullDir, func(path string, info os.FileInfo, walkErr error) error {
			if walkErr != nil {
				return nil
			}
			if info.IsDir() {
				return nil
			}
			if strings.HasSuffix(path, ".go") && !strings.HasSuffix(path, "_test.go") {
				goFiles = append(goFiles, path)
			}
			return nil
		})
		if err != nil {
			return nil, nil, nil, 0, err
		}
		sort.Strings(goFiles)

		for _, path := range goFiles {
			rel, err := filepath.Rel(repoRoot, path)
			if err != nil {
				continue
			}
			rel = filepath.ToSlash(rel)
			pkgPath := filepath.ToSlash(filepath.Dir(rel))
			pkgID := "pkg:" + pkgPath

			parsed, err := parser.ParseFile(fset, path, nil, parser.ParseComments)
			if err != nil {
				continue
			}
			fileCount++

			if !seenPkg[pkgID] {
				seenPkg[pkgID] = true
				nodes = append(nodes, CanonicalNode{
					SourceID:  pkgID,
					Version:   1,
					TenantID:  DefaultTenantID,
					Kind:      KindRepoBlob,
					Title:     "package " + pkgPath,
					Text:      fmt.Sprintf("Hanzo Graph Go package %s (%s) in github.com/hanzoai/graph.", pkgPath, parsed.Name.Name),
					Labels:    []string{"go_package", "real_repo"},
					Metadata:  map[string]string{"pkg_path": pkgPath, "pkg_name": parsed.Name.Name},
					Timestamp: 1740000000 + int64(len(nodes)),
				})
			}

			fileID := "file:" + rel
			var exportedNames []string
			var localImports []string

			for _, imp := range parsed.Imports {
				impPath := strings.Trim(imp.Path.Value, `"`)
				if strings.HasPrefix(impPath, modPrefix) {
					targetPkg := strings.TrimPrefix(impPath, modPrefix)
					localImports = append(localImports, targetPkg)
					if pkgImports[pkgID] == nil {
						pkgImports[pkgID] = make(map[string]bool)
					}
					pkgImports[pkgID]["pkg:"+targetPkg] = true
				}
			}

			for _, decl := range parsed.Decls {
				switch d := decl.(type) {
				case *ast.FuncDecl:
					if d.Name != nil && d.Name.IsExported() && d.Recv == nil {
						symName := d.Name.Name
						exportedNames = append(exportedNames, symName)
						symID := fmt.Sprintf("sym:%s.%s", pkgPath, symName)
						docText := ""
						if d.Doc != nil {
							docText = strings.TrimSpace(d.Doc.Text())
						}
						if len(docText) > 220 {
							docText = docText[:220]
						}
						if _, dup := symbolSummary[symID]; !dup {
							summary := fmt.Sprintf("Exported Go function %s in package %s file %s. %s", symName, pkgPath, rel, docText)
							symbolSummary[symID] = summary
							nodes = append(nodes, CanonicalNode{
								SourceID:  symID,
								Version:   1,
								TenantID:  DefaultTenantID,
								Kind:      KindEntity,
								Title:     symName,
								Text:      summary,
								Labels:    []string{"go_symbol", "go_func", "real_repo"},
								Metadata:  map[string]string{"file": rel, "package": pkgPath},
								Timestamp: 1740000000 + int64(len(nodes)),
							})
							edges = append(edges, CanonicalEdge{
								EdgeID:       nextEdgeID("defines"),
								Version:      1,
								TenantID:     DefaultTenantID,
								FromSourceID: fileID,
								ToSourceID:   symID,
								Type:         "defines",
								Weight:       1.0,
							})
							fileSymbols[fileID] = append(fileSymbols[fileID], symID)
							pkgSymbols[pkgID] = append(pkgSymbols[pkgID], symID)
						}
					}
				case *ast.GenDecl:
					if d.Tok == token.TYPE {
						for _, spec := range d.Specs {
							ts, ok := spec.(*ast.TypeSpec)
							if !ok || ts.Name == nil || !ts.Name.IsExported() {
								continue
							}
							symName := ts.Name.Name
							exportedNames = append(exportedNames, symName)
							symID := fmt.Sprintf("sym:%s.%s", pkgPath, symName)
							docText := ""
							if d.Doc != nil {
								docText = strings.TrimSpace(d.Doc.Text())
							}
							if len(docText) > 220 {
								docText = docText[:220]
							}
							if _, dup := symbolSummary[symID]; !dup {
								summary := fmt.Sprintf("Exported Go type %s in package %s file %s. %s", symName, pkgPath, rel, docText)
								symbolSummary[symID] = summary
								nodes = append(nodes, CanonicalNode{
									SourceID:  symID,
									Version:   1,
									TenantID:  DefaultTenantID,
									Kind:      KindEntity,
									Title:     symName,
									Text:      summary,
									Labels:    []string{"go_symbol", "go_type", "real_repo"},
									Metadata:  map[string]string{"file": rel, "package": pkgPath},
									Timestamp: 1740000000 + int64(len(nodes)),
								})
								edges = append(edges, CanonicalEdge{
									EdgeID:       nextEdgeID("defines"),
									Version:      1,
									TenantID:     DefaultTenantID,
									FromSourceID: fileID,
									ToSourceID:   symID,
									Type:         "defines",
									Weight:       1.0,
								})
								fileSymbols[fileID] = append(fileSymbols[fileID], symID)
								pkgSymbols[pkgID] = append(pkgSymbols[pkgID], symID)
							}
						}
					}
				}
			}

			fileDoc := ""
			if parsed.Doc != nil {
				fileDoc = strings.TrimSpace(parsed.Doc.Text())
				if len(fileDoc) > 240 {
					fileDoc = fileDoc[:240]
				}
			}
			sort.Strings(exportedNames)
			if len(exportedNames) > 12 {
				exportedNames = exportedNames[:12]
			}
			fileText := fmt.Sprintf("Go source file %s in package %s. Exports: %s. Local imports: %s. %s",
				rel, pkgPath, strings.Join(exportedNames, ", "), strings.Join(localImports, ", "), fileDoc)
			nodes = append(nodes, CanonicalNode{
				SourceID:  fileID,
				Version:   1,
				TenantID:  DefaultTenantID,
				Kind:      KindRepoBlob,
				Title:     rel,
				Text:      fileText,
				Labels:    []string{"go_file", "real_repo"},
				Metadata:  map[string]string{"package": pkgPath},
				Timestamp: 1740000000 + int64(len(nodes)),
			})
			edges = append(edges, CanonicalEdge{
				EdgeID:       nextEdgeID("belongs"),
				Version:      1,
				TenantID:     DefaultTenantID,
				FromSourceID: fileID,
				ToSourceID:   pkgID,
				Type:         "belongs_to",
				Weight:       1.0,
			})
			pkgFiles[pkgID] = append(pkgFiles[pkgID], fileID)
		}
	}

	// Materialize package-level internal import edges.
	pkgIDs := make([]string, 0, len(seenPkg))
	for pid := range seenPkg {
		pkgIDs = append(pkgIDs, pid)
	}
	sort.Strings(pkgIDs)
	for _, fromPkg := range pkgIDs {
		imps := pkgImports[fromPkg]
		targets := make([]string, 0, len(imps))
		for toPkg := range imps {
			if seenPkg[toPkg] && toPkg != fromPkg {
				targets = append(targets, toPkg)
			}
		}
		sort.Strings(targets)
		for _, toPkg := range targets {
			edges = append(edges, CanonicalEdge{
				EdgeID:       nextEdgeID("imports"),
				Version:      1,
				TenantID:     DefaultTenantID,
				FromSourceID: fromPkg,
				ToSourceID:   toPkg,
				Type:         "imports",
				Weight:       1.0,
			})
		}
	}

	// Build deterministic verified structural queries on the real code graph.
	qIdx := 0
	// Query type A: exact typed `defines` traversal from real files with 2..6 exported symbols.
	fileKeys := make([]string, 0, len(fileSymbols))
	for fk := range fileSymbols {
		if len(fileSymbols[fk]) >= 2 && len(fileSymbols[fk]) <= 8 {
			fileKeys = append(fileKeys, fk)
		}
	}
	sort.Strings(fileKeys)
	for _, fk := range fileKeys {
		if qIdx >= 10 {
			break
		}
		syms := append([]string(nil), fileSymbols[fk]...)
		sort.Strings(syms)
		gold := syms
		if len(gold) > 3 {
			gold = gold[:3]
		}
		ans := strings.TrimPrefix(gold[0], "sym:")
		if dot := strings.LastIndex(ans, "."); dot >= 0 {
			ans = ans[dot+1:]
		}
		qIdx++
		queries = append(queries, BenchmarkQuery{
			ID:           fmt.Sprintf("code_struct_defines_%02d", qIdx),
			Category:     "real_codegraph_structural",
			ExpectedPlan: PlanPureTraversal,
			Request: QueryRequest{
				Query:                  fmt.Sprintf("symbols defined in %s", strings.TrimPrefix(fk, "file:")),
				AnchorIDs:              []string{fk},
				EdgeTypes:              []string{"defines"},
				TargetLabels:           []string{"go_symbol"},
				MaxHops:                1,
				TopK:                   5,
				LatencyBudgetMs:        5.0,
				Access:                 AccessContext{TenantID: DefaultTenantID},
				RequireStructuralMatch: true,
			},
			GoldSourceIDs: gold,
			GoldAnswer:    ans,
		})
	}

	// Query type B: exact package import dependency traversal from real packages.
	for _, pid := range pkgIDs {
		if qIdx >= 16 {
			break
		}
		imps := pkgImports[pid]
		var targets []string
		for tp := range imps {
			if seenPkg[tp] && tp != pid {
				targets = append(targets, tp)
			}
		}
		if len(targets) == 0 {
			continue
		}
		sort.Strings(targets)
		qIdx++
		queries = append(queries, BenchmarkQuery{
			ID:           fmt.Sprintf("code_struct_imports_%02d", qIdx),
			Category:     "real_codegraph_structural",
			ExpectedPlan: PlanPureTraversal,
			Request: QueryRequest{
				Query:                  fmt.Sprintf("packages imported by %s", strings.TrimPrefix(pid, "pkg:")),
				AnchorIDs:              []string{pid},
				EdgeTypes:              []string{"imports"},
				TargetLabels:           []string{"go_package"},
				MaxHops:                1,
				TopK:                   5,
				LatencyBudgetMs:        5.0,
				Access:                 AccessContext{TenantID: DefaultTenantID},
				RequireStructuralMatch: true,
			},
			GoldSourceIDs: targets,
			GoldAnswer:    strings.TrimPrefix(targets[0], "pkg:"),
		})
	}

	return nodes, edges, queries, fileCount, nil
}

// buildSTaRKSemiStructuredWorkload constructs STaRK-style semi-structured
// relational + textual retrieval workloads (both anchored graph_first and
// unanchored semantic vector_first cases).
func buildSTaRKSemiStructuredWorkload(rng *rand.Rand) ([]CanonicalNode, []CanonicalEdge, []BenchmarkQuery) {
	var nodes []CanonicalNode
	var edges []CanonicalEdge
	var queries []BenchmarkQuery

	domains := []struct {
		service  string
		module   string
		vuln     string
		patch    string
		protocol string
		symptom  string
	}{
		{"ledger_router", "merkle_batcher", "nonce_replay_drift", "pr_7101_nonce_epoch", "quorum_bft", "duplicate commit receipt during leader failover"},
		{"vector_gateway", "hnsw_quantizer", "centroid_skew_bug", "pr_7102_pq_rebalance", "cosine_simd", "recall drop after shard compaction"},
		{"inference_mesh", "kv_slab_pool", "page_alias_hazard", "pr_7103_slab_guard", "paged_attn", "corrupted prefix cache under concurrent batching"},
		{"storage_engine", "wal_compactor", "crc_truncation_race", "pr_7104_wal_fsync", "nvme_ring", "tail checkpoint mismatch on power loss"},
		{"policy_broker", "acl_evaluator", "scope_cache_stale", "pr_7105_epoch_bust", "zero_trust", "stale permission window after role revocation"},
		{"telemetry_bus", "span_sampler", "ring_wrap_overflow", "pr_7106_atomic_head", "otlp_grpc", "dropped latency histograms at burst peak"},
		{"workflow_orch", "dag_scheduler", "cycle_deadlock_edge", "pr_7107_topo_check", "async_queue", "stalled pipeline workers on dynamic retry"},
		{"billing_meter", "token_counter", "rounding_drift_err", "pr_7108_fixed_point", "ledger_iso", "sub-cent invoice variance on streaming calls"},
		{"index_builder", "csr_compressor", "offset_uint32_wrap", "pr_7109_bounds_chk", "compact_csr", "adjacency slice panic on dense hub rebuild"},
		{"search_ranker", "bm25_scorer", "idf_zero_division", "pr_7110_smooth_idf", "hybrid_rrf", "NaN score propagation on singleton corpus"},
		{"sandbox_runner", "cgroup_limiter", "oom_signal_missed", "pr_7111_mem_watch", "seccomp_v2", "unbounded heap growth in untrusted plugin"},
		{"model_registry", "weight_loader", "mmap_alignment_err", "pr_7112_page_align", "safetensor", "bus error loading quantized checkpoint shard"},
	}

	eSeq := 0
	mkEdge := func(from, to, etype string, w float32) CanonicalEdge {
		eSeq++
		return CanonicalEdge{
			EdgeID:       fmt.Sprintf("stark_edge_%04d", eSeq),
			Version:      1,
			TenantID:     DefaultTenantID,
			FromSourceID: from,
			ToSourceID:   to,
			Type:         etype,
			Weight:       w,
		}
	}

	for i, d := range domains {
		svcID := "stark_svc_" + d.service
		modID := "stark_mod_" + d.module
		incID := "stark_inc_" + d.vuln
		patID := "stark_pat_" + d.patch

		nodes = append(nodes,
			CanonicalNode{
				SourceID:  svcID,
				Version:   1,
				TenantID:  DefaultTenantID,
				Kind:      KindEntity,
				Title:     d.service,
				Text:      fmt.Sprintf("Production service %s operating on protocol %s with component %s.", d.service, d.protocol, d.module),
				Labels:    []string{"stark_service"},
				Timestamp: 1740100000 + int64(i*4),
			},
			CanonicalNode{
				SourceID:  modID,
				Version:   1,
				TenantID:  DefaultTenantID,
				Kind:      KindRepoBlob,
				Title:     d.module,
				Text:      fmt.Sprintf("Core library module %s implementing %s internals for %s.", d.module, d.protocol, d.service),
				Labels:    []string{"stark_module"},
				Timestamp: 1740100001 + int64(i*4),
			},
			CanonicalNode{
				SourceID:  incID,
				Version:   1,
				TenantID:  DefaultTenantID,
				Kind:      KindAgentEvent,
				Title:     d.vuln,
				Text:      fmt.Sprintf("Incident %s observed %s in subsystem %s.", d.vuln, d.symptom, d.module),
				Labels:    []string{"stark_incident"},
				Timestamp: 1740100002 + int64(i*4),
			},
			CanonicalNode{
				SourceID:  patID,
				Version:   1,
				TenantID:  DefaultTenantID,
				Kind:      KindGitCommit,
				Title:     d.patch,
				Text:      fmt.Sprintf("Remediation commit %s resolves %s (%s) via deterministic guardrails in %s.", d.patch, d.vuln, d.symptom, d.module),
				Labels:    []string{"stark_patch"},
				Timestamp: 1740100003 + int64(i*4),
			},
		)

		edges = append(edges,
			mkEdge(svcID, modID, "depends_on", 1.0),
			mkEdge(modID, incID, "causes", 1.0),
			mkEdge(patID, incID, "mitigates", 1.2),
			mkEdge(patID, modID, "modifies", 1.0),
		)

		// Add 2 distracting sibling modules per service (low degree ~5, no hub explosion).
		for j := 0; j < 2; j++ {
			distID := fmt.Sprintf("stark_dist_%02d_%d", i, j)
			nodes = append(nodes, CanonicalNode{
				SourceID:  distID,
				Version:   1,
				TenantID:  DefaultTenantID,
				Kind:      KindRepoBlob,
				Title:     fmt.Sprintf("%s_helper_%d", d.service, j),
				Text:      fmt.Sprintf("Auxiliary helper %d for %s handling routine metrics serialization (%d).", j, d.service, rng.Intn(1000)),
				Labels:    []string{"stark_module"},
				Timestamp: 1740150000 + int64(i*2+j),
			})
			edges = append(edges, mkEdge(svcID, distID, "depends_on", 0.6))
		}

		// Anchored STaRK multi-hop query -> ideal for graph_first!
		if i < 8 {
			queries = append(queries, BenchmarkQuery{
				ID:           fmt.Sprintf("stark_graph_first_%02d", i+1),
				Category:     "stark_semistructured_multihop",
				ExpectedPlan: PlanGraphFirst,
				Request: QueryRequest{
					Query:           fmt.Sprintf("Which remediation patch mitigates the %s incident affecting %s?", d.symptom, d.service),
					AnchorIDs:       []string{svcID},
					TargetLabels:    []string{"stark_patch"},
					MaxHops:         3,
					TopK:            5,
					LatencyBudgetMs: 8.0,
					Access:          AccessContext{TenantID: DefaultTenantID},
				},
				GoldSourceIDs: []string{patID},
				GoldAnswer:    d.patch,
			})
		} else {
			// Unanchored semi-structured multi-hop query -> ideal for vector_first!
			queries = append(queries, BenchmarkQuery{
				ID:           fmt.Sprintf("stark_vector_first_%02d", i+1),
				Category:     "stark_semistructured_multihop",
				ExpectedPlan: PlanVectorFirst,
				Request: QueryRequest{
					Query:           fmt.Sprintf("Find the remediation commit linked to %s under %s protocol", d.symptom, d.protocol),
					TargetLabels:    []string{"stark_patch"},
					MaxHops:         2,
					TopK:            5,
					LatencyBudgetMs: 8.0,
					Access:          AccessContext{TenantID: DefaultTenantID},
				},
				GoldSourceIDs: []string{patID},
				GoldAnswer:    d.patch,
			})
		}
	}

	return nodes, edges, queries
}

// buildSimpleFactualNegativeRegime creates single-hop factual queries where
// graph expansion adds latency and noise without evidence gain (explicitly
// required by hanzoai/graph#5: plain RAG beats forced graph expansion here).
func buildSimpleFactualNegativeRegime() ([]CanonicalNode, []CanonicalEdge, []BenchmarkQuery) {
	facts := []struct {
		id     string
		title  string
		text   string
		query  string
		answer string
	}{
		{"fact_01", "HIP-1347 Embedding Dimension", "The default zero-dependency feature-hashing vector dimension in Hanzo GraphPlanner is 128 floats.", "What is the default feature-hashing vector dimension in Hanzo GraphPlanner?", "128"},
		{"fact_02", "CSR Forward Edge Width", "Each forward CSR edge record packs target uint32, edge type uint16, weight float32, and edge index uint32 for 14 bytes.", "How many bytes does each forward CSR edge record pack?", "14"},
		{"fact_03", "ZapDB Storage Backend", "Hanzo Graph stores keys and posting lists in luxfi zapdb v1.10.6 opened in managed timestamp mode.", "Which storage backend holds keys and posting lists in Hanzo Graph?", "zapdb"},
		{"fact_04", "BM25 Default Tuning", "The embedded Okapi BM25 lexical scorer sets saturation k1 to 1.2 and length normalization b to 0.75.", "What saturation k1 and length normalization b values does Okapi BM25 use?", "1.2"},
		{"fact_05", "Apache 2.0 License", "Hanzo Graph is distributed under the Apache License 2.0 matching upstream Dgraph commit 64804bd2.", "Under which open-source license is Hanzo Graph distributed?", "Apache"},
		{"fact_06", "LRU Cache Default Capacity", "The bounded ACL-scoped neighborhood LRU cache defaults to 512 entries per embedded store.", "What is the default capacity of the bounded neighborhood LRU cache?", "512"},
		{"fact_07", "Tenant Isolation Rule", "Tenant partitioning and ACL scope checks run before any hit is scored, cached, or returned.", "When are tenant partitioning and ACL scope checks applied?", "before"},
		{"fact_08", "Hub Degree Threshold", "The cost-based planner pivots from graph_first to vector_first when anchor local degree exceeds 40.", "At what anchor degree threshold does the planner pivot away from graph_first?", "40"},
		{"fact_09", "Derived Index Rebuild", "Dropping derived CSR and lexical indexes never deletes canonical versioned records and RebuildIndex restores a bit-exact digest.", "Does dropping derived indexes delete canonical versioned records?", "never"},
		{"fact_10", "Maximum Hop Guard", "Bounded structural traversals clamp MaxHops to 4 hops to prevent unbounded memory blowup.", "What maximum hop limit clamps bounded structural traversals?", "4"},
		{"fact_11", "ZAP Native Transport", "Hanzo control plane RPC uses native ZAP schemas rather than gRPC or protobuf.", "Which RPC transport protocol does the Hanzo control plane use instead of gRPC?", "ZAP"},
		{"fact_12", "Simple Factual Bypass", "When a single-hop factual query has no structural graph constraint, plain_rag_bypass skips CSR expansion.", "Which execution plan skips CSR expansion on simple factual queries?", "plain_rag_bypass"},
	}

	var nodes []CanonicalNode
	var edges []CanonicalEdge
	var queries []BenchmarkQuery

	// Add a factual index hub connected to all fact nodes to simulate a dense
	// documentation index where forced graph traversal drags in unrelated facts.
	nodes = append(nodes, CanonicalNode{
		SourceID:  "fact_handbook_index",
		Version:   1,
		TenantID:  DefaultTenantID,
		Kind:      KindDocument,
		Title:     "Engineering Handbook Index",
		Text:      "Table of contents linking all Hanzo engineering reference facts.",
		Labels:    []string{"fact_doc"},
		Timestamp: 1740200000,
	})

	for i, f := range facts {
		nodes = append(nodes, CanonicalNode{
			SourceID:  f.id,
			Version:   1,
			TenantID:  DefaultTenantID,
			Kind:      KindDocument,
			Title:     f.title,
			Text:      f.text,
			Labels:    []string{"fact_doc"},
			Timestamp: 1740200001 + int64(i),
		})
		edges = append(edges, CanonicalEdge{
			EdgeID:       fmt.Sprintf("fact_edge_%02d", i+1),
			Version:      1,
			TenantID:     DefaultTenantID,
			FromSourceID: "fact_handbook_index",
			ToSourceID:   f.id,
			Type:         "references",
			Weight:       1.0,
		})
		queries = append(queries, BenchmarkQuery{
			ID:           fmt.Sprintf("simple_factual_%02d", i+1),
			Category:     "graphrag_simple_factual",
			ExpectedPlan: PlanPlainRAGBypass,
			Request: QueryRequest{
				Query:           f.query,
				TargetLabels:    []string{"fact_doc"},
				MaxHops:         0,
				TopK:            3,
				LatencyBudgetMs: 2.0,
				Access:          AccessContext{TenantID: DefaultTenantID},
			},
			GoldSourceIDs: []string{f.id},
			GoldAnswer:    f.answer,
		})
	}
	return nodes, edges, queries
}

// buildHubExplosionRegime constructs a high-degree hub (96 incident edges)
// connected to 90 noisy telemetry nodes and 6 distinct gold incident targets.
// Forced graph_first / pure_traversal explodes across the hub, while the
// adaptive cost-based planner detects degree > 40 and selects vector_first.
func buildHubExplosionRegime(rng *rand.Rand) ([]CanonicalNode, []CanonicalEdge, []BenchmarkQuery) {
	var nodes []CanonicalNode
	var edges []CanonicalEdge
	var queries []BenchmarkQuery

	hubID := "hub_gateway_monolith"
	nodes = append(nodes, CanonicalNode{
		SourceID:  hubID,
		Version:   1,
		TenantID:  DefaultTenantID,
		Kind:      KindEntity,
		Title:     "gateway_monolith_hub",
		Text:      "Central API gateway monolith connected to ninety telemetry streams and six critical subsystems.",
		Labels:    []string{"hub_service"},
		Timestamp: 1740300000,
	})

	// 90 noisy telemetry nodes with high edge weights to tempt naive pure traversal.
	for i := 0; i < 90; i++ {
		nid := fmt.Sprintf("hub_noise_%03d", i)
		nodes = append(nodes, CanonicalNode{
			SourceID:  nid,
			Version:   1,
			TenantID:  DefaultTenantID,
			Kind:      KindAgentEvent,
			Title:     fmt.Sprintf("heartbeat_metric_%03d", i),
			Text:      fmt.Sprintf("Routine gateway_monolith_hub heartbeat probe %03d ok latency_ms=%d cpu_pct=%d", i, 5+rng.Intn(40), 10+rng.Intn(50)),
			Labels:    []string{"hub_target"},
			Timestamp: 1740300100 + int64(i),
		})
		edges = append(edges, CanonicalEdge{
			EdgeID:       fmt.Sprintf("hub_noise_edge_%03d", i),
			Version:      1,
			TenantID:     DefaultTenantID,
			FromSourceID: hubID,
			ToSourceID:   nid,
			Type:         "emits",
			Weight:       1.5, // higher static edge weight so pure_traversal prefers noise over signal!
		})
	}

	goldCases := []struct {
		id     string
		title  string
		text   string
		query  string
		answer string
	}{
		{"hub_gold_tls", "tls_alpn_downgrade_alert", "Critical TLS ALPN negotiation downgrade detected on gateway_monolith_hub terminating HTTP/3 handshakes with cipher_suite_x25519.", "Which critical TLS ALPN downgrade alert affected HTTP/3 cipher_suite_x25519 on gateway_monolith_hub?", "cipher_suite_x25519"},
		{"hub_gold_jwt", "jwt_jwks_rotation_fault", "JWKS key rotation race rejected valid EdDSA bearer tokens during kid_2026_q4 rollover on gateway_monolith_hub.", "What JWKS key rotation fault rejected EdDSA bearer tokens during kid_2026_q4 rollover?", "kid_2026_q4"},
		{"hub_gold_rl", "redis_token_bucket_skew", "Distributed rate limiter token bucket clock skew throttled burst_tier_enterprise tenants on gateway_monolith_hub.", "Which rate limiter token bucket clock skew throttled burst_tier_enterprise tenants?", "burst_tier_enterprise"},
		{"hub_gold_ws", "websocket_frame_mask_oom", "Malformed fragmented WebSocket frame mask triggered arena_pool_exhaustion inside gateway_monolith_hub proxy.", "What malformed WebSocket frame mask fault triggered arena_pool_exhaustion?", "arena_pool_exhaustion"},
		{"hub_gold_dns", "coredns_srv_ttl_flap", "Zero-TTL SRV record flapping caused upstream_pool_churn across regional edge pop nodes from gateway_monolith_hub.", "Which CoreDNS SRV TTL flapping fault caused upstream_pool_churn?", "upstream_pool_churn"},
		{"hub_gold_grpc", "h2_window_starvation", "HTTP/2 flow control receive window starvation stalled bidirectional_stream_sync RPCs through gateway_monolith_hub.", "What HTTP/2 flow control window starvation stalled bidirectional_stream_sync RPCs?", "bidirectional_stream_sync"},
	}

	for i, gc := range goldCases {
		nodes = append(nodes, CanonicalNode{
			SourceID:  gc.id,
			Version:   1,
			TenantID:  DefaultTenantID,
			Kind:      KindAgentEvent,
			Title:     gc.title,
			Text:      gc.text,
			Labels:    []string{"hub_target"},
			Timestamp: 1740300500 + int64(i),
		})
		edges = append(edges, CanonicalEdge{
			EdgeID:       fmt.Sprintf("hub_gold_edge_%02d", i+1),
			Version:      1,
			TenantID:     DefaultTenantID,
			FromSourceID: hubID,
			ToSourceID:   gc.id,
			Type:         "emits",
			Weight:       0.9,
		})
		queries = append(queries, BenchmarkQuery{
			ID:           fmt.Sprintf("hub_explosion_%02d", i+1),
			Category:     "hub_degree_explosion",
			ExpectedPlan: PlanVectorFirst,
			Request: QueryRequest{
				Query:           gc.query,
				AnchorIDs:       []string{hubID},
				EdgeTypes:       []string{"emits"},
				TargetLabels:    []string{"hub_target"},
				MaxHops:         1,
				TopK:            3,
				LatencyBudgetMs: 6.0,
				Access:          AccessContext{TenantID: DefaultTenantID},
			},
			GoldSourceIDs: []string{gc.id},
			GoldAnswer:    gc.answer,
		})
	}

	return nodes, edges, queries
}

// buildTightBudgetHotCacheRegime models repeated operational neighborhood
// queries over warm anchors under tight latency budgets (0.25 ms), where the
// planner detects CacheHitProbability = 1.0 and selects cached_neighborhood.
func buildTightBudgetHotCacheRegime() ([]CanonicalNode, []CanonicalEdge, []BenchmarkQuery) {
	var nodes []CanonicalNode
	var edges []CanonicalEdge
	var queries []BenchmarkQuery

	anchors := []struct {
		anchorID string
		title    string
		childID  string
		childTxt string
		answer   string
	}{
		{"hot_cluster_us_east", "cluster_us_east_1", "hot_runbook_failover_east", "Active-active failover runbook runbook_east_01 drains ingress in 12 seconds.", "runbook_east_01"},
		{"hot_cluster_eu_west", "cluster_eu_west_1", "hot_runbook_gdpr_vault", "Regional KMS key rotation procedure kms_eu_vault_02 enforces sovereign HSM residency.", "kms_eu_vault_02"},
		{"hot_cluster_ap_south", "cluster_ap_south_1", "hot_runbook_cdn_purge", "Edge cache invalidation procedure cdn_ap_purge_03 flushes stale manifests globally.", "cdn_ap_purge_03"},
		{"hot_cluster_gpu_west", "cluster_gpu_west_2", "hot_runbook_nvlink_reset", "Fabric recovery procedure nvlink_reset_04 isolates degraded H100 interconnects.", "nvlink_reset_04"},
		{"hot_cluster_db_core", "cluster_db_core_0", "hot_runbook_raft_quorum", "Consensus quorum step-down procedure raft_quorum_05 transfers leadership cleanly.", "raft_quorum_05"},
		{"hot_cluster_auth_edge", "cluster_auth_edge_1", "hot_runbook_jwks_pin", "Emergency JWKS revocation runbook jwks_pin_06 rotates signing keys without downtime.", "jwks_pin_06"},
	}

	for i, a := range anchors {
		nodes = append(nodes,
			CanonicalNode{
				SourceID:  a.anchorID,
				Version:   1,
				TenantID:  DefaultTenantID,
				Kind:      KindEntity,
				Title:     a.title,
				Text:      fmt.Sprintf("Operational cluster %s with hot SRE runbook neighborhood.", a.title),
				Labels:    []string{"hot_cluster"},
				Timestamp: 1740400000 + int64(i*4),
			},
			CanonicalNode{
				SourceID:  a.childID,
				Version:   1,
				TenantID:  DefaultTenantID,
				Kind:      KindDecision,
				Title:     a.childID,
				Text:      a.childTxt,
				Labels:    []string{"sre_runbook"},
				Timestamp: 1740400001 + int64(i*4),
			},
		)
		edges = append(edges, CanonicalEdge{
			EdgeID:       fmt.Sprintf("hot_edge_%02d", i+1),
			Version:      1,
			TenantID:     DefaultTenantID,
			FromSourceID: a.anchorID,
			ToSourceID:   a.childID,
			Type:         "governed_by",
			Weight:       1.2,
		})
		// Add 3 neighboring telemetry specs around each hot cluster.
		for j := 0; j < 3; j++ {
			sibID := fmt.Sprintf("%s_node_%d", a.anchorID, j)
			nodes = append(nodes, CanonicalNode{
				SourceID:  sibID,
				Version:   1,
				TenantID:  DefaultTenantID,
				Kind:      KindEntity,
				Title:     sibID,
				Text:      fmt.Sprintf("Worker node %d attached to %s.", j, a.title),
				Labels:    []string{"cluster_worker"},
				Timestamp: 1740400100 + int64(i*3+j),
			})
			edges = append(edges, CanonicalEdge{
				EdgeID:       fmt.Sprintf("hot_sib_edge_%02d_%d", i+1, j),
				Version:      1,
				TenantID:     DefaultTenantID,
				FromSourceID: a.anchorID,
				ToSourceID:   sibID,
				Type:         "governed_by",
				Weight:       0.7,
			})
		}

		queries = append(queries, BenchmarkQuery{
			ID:            fmt.Sprintf("tight_budget_hot_%02d", i+1),
			Category:      "tight_latency_budget_hot",
			ExpectedPlan:  PlanCachedNeighborhood,
			WarmBeforeRun: true,
			Request: QueryRequest{
				Query:           fmt.Sprintf("What SRE recovery runbook governs %s?", a.title),
				AnchorIDs:       []string{a.anchorID},
				EdgeTypes:       []string{"governed_by"},
				TargetLabels:    []string{"sre_runbook"},
				MaxHops:         1,
				TopK:            3,
				LatencyBudgetMs: 0.25,
				Access:          AccessContext{TenantID: DefaultTenantID},
			},
			GoldSourceIDs: []string{a.childID},
			GoldAnswer:    a.answer,
		})
	}

	return nodes, edges, queries
}

// buildACLIsolationProbes constructs cross-tenant shared-entity and
// within-tenant restricted-ACL bridge probes to verify 0% leak rate.
func buildACLIsolationProbes() ([]CanonicalNode, []CanonicalEdge, []BenchmarkQuery) {
	var nodes []CanonicalNode
	var edges []CanonicalEdge
	var queries []BenchmarkQuery

	for i := 0; i < 6; i++ {
		anchorID := fmt.Sprintf("acl_bridge_anchor_%02d", i+1)
		pubID := fmt.Sprintf("acl_pub_doc_%02d", i+1)
		secID := fmt.Sprintf("acl_secret_doc_%02d", i+1)
		foreignID := fmt.Sprintf("acl_foreign_doc_%02d", i+1)

		// DefaultTenantID: public anchor -> public doc AND restricted sec-ops secret doc.
		nodes = append(nodes,
			CanonicalNode{
				SourceID:  anchorID,
				Version:   1,
				TenantID:  DefaultTenantID,
				ACL:       []string{"team:public", "team:sec-ops"},
				Kind:      KindEntity,
				Title:     fmt.Sprintf("shared_bridge_entity_%02d", i+1),
				Text:      fmt.Sprintf("Shared bridge entity %02d linking public status and restricted audit docs.", i+1),
				Labels:    []string{"acl_bridge"},
				Timestamp: 1740500000 + int64(i*4),
			},
			CanonicalNode{
				SourceID:  pubID,
				Version:   1,
				TenantID:  DefaultTenantID,
				ACL:       []string{"team:public"},
				Kind:      KindDocument,
				Title:     fmt.Sprintf("public_release_note_%02d", i+1),
				Text:      fmt.Sprintf("Public release note pub_token_%02d for shared_bridge_entity_%02d.", i+1, i+1),
				Labels:    []string{"acl_doc"},
				Timestamp: 1740500001 + int64(i*4),
			},
			CanonicalNode{
				SourceID:  secID,
				Version:   1,
				TenantID:  DefaultTenantID,
				ACL:       []string{"team:sec-ops"},
				Kind:      KindDocument,
				Title:     fmt.Sprintf("classified_secops_audit_%02d", i+1),
				Text:      fmt.Sprintf("CONFIDENTIAL secret_canary_%02d for shared_bridge_entity_%02d restricted strictly to sec-ops.", i+1, i+1),
				Labels:    []string{"acl_doc"},
				Timestamp: 1740500002 + int64(i*4),
			},
			// ForeignTenantID with IDENTICAL anchor SourceID!
			CanonicalNode{
				SourceID:  anchorID,
				Version:   1,
				TenantID:  ForeignTenantID,
				ACL:       []string{"team:public"},
				Kind:      KindEntity,
				Title:     fmt.Sprintf("shared_bridge_entity_%02d", i+1),
				Text:      fmt.Sprintf("Foreign tenant beta entity %02d.", i+1),
				Labels:    []string{"acl_bridge"},
				Timestamp: 1740500003 + int64(i*4),
			},
			CanonicalNode{
				SourceID:  foreignID,
				Version:   1,
				TenantID:  ForeignTenantID,
				ACL:       []string{"team:public"},
				Kind:      KindDocument,
				Title:     fmt.Sprintf("foreign_beta_secret_%02d", i+1),
				Text:      fmt.Sprintf("FOREIGN_TENANT_CANARY_%02d strictly belonging to tenant-external-beta.", i+1),
				Labels:    []string{"acl_doc"},
				Timestamp: 1740500004 + int64(i*4),
			},
		)

		edges = append(edges,
			CanonicalEdge{
				EdgeID:       fmt.Sprintf("acl_e_pub_%02d", i+1),
				Version:      1,
				TenantID:     DefaultTenantID,
				ACL:          []string{"team:public", "team:sec-ops"},
				FromSourceID: anchorID,
				ToSourceID:   pubID,
				Type:         "links_to",
				Weight:       1.0,
			},
			CanonicalEdge{
				EdgeID:       fmt.Sprintf("acl_e_sec_%02d", i+1),
				Version:      1,
				TenantID:     DefaultTenantID,
				ACL:          []string{"team:sec-ops"},
				FromSourceID: anchorID,
				ToSourceID:   secID,
				Type:         "links_to",
				Weight:       1.5,
			},
			CanonicalEdge{
				EdgeID:       fmt.Sprintf("acl_e_foreign_%02d", i+1),
				Version:      1,
				TenantID:     ForeignTenantID,
				ACL:          []string{"team:public"},
				FromSourceID: anchorID,
				ToSourceID:   foreignID,
				Type:         "links_to",
				Weight:       1.5,
			},
		)

		// Public caller probing DefaultTenantID: must see ONLY pubID, NEVER secID or foreignID!
		queries = append(queries, BenchmarkQuery{
			ID:           fmt.Sprintf("acl_probe_public_%02d", i+1),
			Category:     "acl_isolation_probes",
			ExpectedPlan: PlanGraphFirst,
			Request: QueryRequest{
				Query:           fmt.Sprintf("Which documents relate to shared_bridge_entity_%02d secret_canary_%02d?", i+1, i+1),
				AnchorIDs:       []string{anchorID},
				EdgeTypes:       []string{"links_to"},
				TargetLabels:    []string{"acl_doc"},
				MaxHops:         1,
				TopK:            5,
				LatencyBudgetMs: 5.0,
				Access:          AccessContext{TenantID: DefaultTenantID, Scopes: []string{"team:public"}},
			},
			GoldSourceIDs: []string{pubID},
			GoldAnswer:    fmt.Sprintf("pub_token_%02d", i+1),
		})
	}

	return nodes, edges, queries
}
