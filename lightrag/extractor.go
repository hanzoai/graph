/*
 * SPDX-FileCopyrightText: © 2026 Hanzo AI, Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

package lightrag

import (
	"context"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"unicode"
)

const hanzoGraphModulePrefix = "github.com/hanzoai/graph/"

// DefaultExtractor is a deterministic, clean-room Go AST + technical-text
// entity-relationship extractor. It parses real Go source files with go/parser
// and extracts structured entities and relationships from commits, ADRs,
// incident reports, agent events, and markdown documentation.
type DefaultExtractor struct {
	InputTokenPricePerMillion  float64
	OutputTokenPricePerMillion float64
}

// NewDefaultExtractor constructs a DefaultExtractor with transparent token
// pricing parameters for cost-per-relationship accounting.
func NewDefaultExtractor(inputPricePerM, outputPricePerM float64) *DefaultExtractor {
	if inputPricePerM <= 0 {
		inputPricePerM = 0.15
	}
	if outputPricePerM <= 0 {
		outputPricePerM = 0.60
	}
	return &DefaultExtractor{
		InputTokenPricePerMillion:  inputPricePerM,
		OutputTokenPricePerMillion: outputPricePerM,
	}
}

var (
	reFilePath    = regexp.MustCompile(`\b([a-zA-Z0-9_-]+(?:/[a-zA-Z0-9_.-]+)+\.(?:go|md|proto|yaml|yml|json|txt|sh))\b`)
	rePkgSymbol   = regexp.MustCompile(`\b([a-z][a-z0-9_]{1,18})\.([A-Z][A-Za-z0-9_]{1,48})\b`)
	reBacktickSym = regexp.MustCompile("`([^`\\n]{2,64})`")
	reArtifactID  = regexp.MustCompile(`\b((?:ADR|INC|PR|COMMIT|EVT|RFC|HIP)-[A-Za-z0-9_-]{2,24})\b`)
)

// knownConcepts maps canonical domain themes to surface trigger phrases for
// LightRAG high-level thematic bridging across code and operational documents.
var knownConcepts = []struct {
	Canonical string
	Triggers  []string
	Keywords  []string
}{
	{
		Canonical: "MVCC Transactions",
		Triggers:  []string{"mvcc", "readts", "committs", "shouldabort", "txnwriter", "oracle", "transaction", "rollup"},
		Keywords:  []string{"mvcc", "transaction", "timestamp", "commit", "abort", "rollup", "oracle", "snapshot"},
	},
	{
		Canonical: "Posting List Engine",
		Triggers:  []string{"posting list", "postinglist", "mutablelayer", "cachepl", "memorylayer", "readpostinglist"},
		Keywords:  []string{"posting", "list", "mutation", "layer", "cache", "persistence", "zapdb"},
	},
	{
		Canonical: "UID Compression & Set Algebra",
		Triggers:  []string{"uidpack", "uidblock", "intersectcompressed", "mergesorted", "groupvarint", "bitpacking", "codec.encoder", "codec.decoder"},
		Keywords:  []string{"uid", "compression", "codec", "encoder", "decoder", "intersection", "merge", "packed"},
	},
	{
		Canonical: "HNSW Vector Indexing",
		Triggers:  []string{"hnsw", "persistenthnsw", "searchwithpath", "pickstartnode", "vec_entry", "efconstruction", "nearest neighbor", "vector index"},
		Keywords:  []string{"hnsw", "vector", "similarity", "graph", "nearest", "neighbor", "embedding", "search"},
	},
	{
		Canonical: "Tokenization & Full-Text Indexing",
		Triggers:  []string{"tokenizer", "bleve", "fulltext", "ngram", "trigram", "buildtokens", "stemmer", "stopword", "indexfactory"},
		Keywords:  []string{"tokenizer", "token", "fulltext", "bleve", "index", "stemmer", "ngram", "trigram"},
	},
	{
		Canonical: "DQL Query & Mutation Parsing",
		Triggers:  []string{"graphquery", "filtertree", "mathtree", "shortestpathargs", "nquad", "dql", "lexer", "itemiterator"},
		Keywords:  []string{"dql", "parser", "query", "lexer", "ast", "mutation", "nquad", "filter"},
	},
	{
		Canonical: "Schema Management & Rebuilding",
		Triggers:  []string{"parsedschema", "indexrebuild", "needindexrebuild", "schemaupdate", "dropindexes", "buildindexes"},
		Keywords:  []string{"schema", "predicate", "rebuild", "index", "type", "drop"},
	},
	{
		Canonical: "Spatial S2 & Geo Indexing",
		Triggers:  []string{"geofilter", "geoquerydata", "s2index", "s2.cellunion", "geotokenizer"},
		Keywords:  []string{"geo", "s2", "spatial", "polygon", "location", "index"},
	},
	{
		Canonical: "Storage & Key Encoding",
		Triggers:  []string{"zapdb", "badger", "parsedkey", "reservednamespace", "datakey", "indexkey", "schemakey"},
		Keywords:  []string{"storage", "zapdb", "badger", "key", "encoding", "namespace", "disk"},
	},
	{
		Canonical: "Security, ACL & Namespace Isolation",
		Triggers:  []string{"acl", "tenant", "vault", "tls", "namespace", "encryption", "sensitive", "ratelimiter"},
		Keywords:  []string{"security", "acl", "tenant", "isolation", "namespace", "encryption", "tls", "vault"},
	},
}

// Extract parses src into typed entities, relationships, and deterministic
// extraction cost metrics.
func (e *DefaultExtractor) Extract(ctx context.Context, src SourceRecord) (ExtractionResult, error) {
	if err := ctx.Err(); err != nil {
		return ExtractionResult{}, err
	}

	var res ExtractionResult
	isGo := src.Kind == KindCodeBlob || strings.HasSuffix(strings.ToLower(src.Path), ".go") || strings.HasSuffix(strings.ToLower(src.SourceID), ".go")
	if isGo {
		if parsed, ok := e.extractGoAST(src); ok {
			res = parsed
		} else {
			res = e.extractStructuredText(src)
		}
	} else {
		res = e.extractStructuredText(src)
	}

	inTok := EstimateTokens(src.Title + "\n" + src.Content)
	outChars := 0
	for _, ent := range res.Entities {
		outChars += len(ent.Name) + len(ent.Description) + 24
	}
	for _, rel := range res.Relations {
		outChars += len(rel.SourceName) + len(rel.TargetName) + len(rel.Description) + 28
	}
	outTok := (outChars + 3) / 4
	cost := (float64(inTok)/1_000_000.0)*e.InputTokenPricePerMillion +
		(float64(outTok)/1_000_000.0)*e.OutputTokenPricePerMillion

	res.Metrics = ExtractionMetrics{
		InputTokens:      inTok,
		OutputTokens:     outTok,
		EstimatedCostUSD: cost,
		RawEntities:      len(res.Entities),
		RawRelations:     len(res.Relations),
	}
	return res, nil
}

func (e *DefaultExtractor) extractGoAST(src SourceRecord) (ExtractionResult, bool) {
	fset := token.NewFileSet()
	fname := src.Path
	if fname == "" {
		fname = src.SourceID
	}
	file, err := parser.ParseFile(fset, fname, src.Content, parser.ParseComments)
	if err != nil || file == nil {
		return ExtractionResult{}, false
	}

	var entities []ExtractedEntity
	var relations []ExtractedRelation

	pkgPath := deriveGoPackagePath(fname, file.Name.Name)
	pkgName := file.Name.Name

	fileDoc := firstCommentSentence(file.Doc)
	if fileDoc == "" {
		fileDoc = fmt.Sprintf("Go source file %s belonging to package %s", fname, pkgPath)
	}
	entities = append(entities, ExtractedEntity{
		Name:        fname,
		Type:        EntityFile,
		Description: fileDoc,
		Keywords:    []string{fname, pkgPath, pkgName, "go", "source"},
		Aliases:     []string{filepath.Base(fname)},
		Weight:      1.0,
	})
	entities = append(entities, ExtractedEntity{
		Name:        pkgPath,
		Type:        EntityPackage,
		Description: fmt.Sprintf("Hanzo Graph Go package %s (%s)", pkgPath, pkgName),
		Keywords:    []string{pkgPath, pkgName, "package"},
		Aliases:     []string{pkgName, hanzoGraphModulePrefix + pkgPath},
		Weight:      1.2,
	})
	relations = append(relations, ExtractedRelation{
		SourceName:  fname,
		SourceType:  EntityFile,
		TargetName:  pkgPath,
		TargetType:  EntityPackage,
		Type:        RelBelongsTo,
		Description: fmt.Sprintf("File %s implements part of package %s", fname, pkgPath),
		Keywords:    []string{fname, pkgPath, "package", "implementation"},
		Weight:      1.0,
	})

	importMap := make(map[string]importTarget)

	for _, imp := range file.Imports {
		impPath := strings.Trim(imp.Path.Value, `"`)
		if impPath == "" {
			continue
		}
		var alias string
		if imp.Name != nil && imp.Name.Name != "_" && imp.Name.Name != "." {
			alias = imp.Name.Name
		} else {
			parts := strings.Split(impPath, "/")
			alias = parts[len(parts)-1]
			// Strip major version suffix (e.g., /v2) for alias lookup.
			if len(parts) >= 2 && strings.HasPrefix(alias, "v") && len(alias) <= 3 {
				alias = parts[len(parts)-2]
			}
		}

		if strings.HasPrefix(impPath, hanzoGraphModulePrefix) {
			subPkg := strings.TrimPrefix(impPath, hanzoGraphModulePrefix)
			importMap[alias] = importTarget{name: subPkg, typ: EntityPackage}
			entities = append(entities, ExtractedEntity{
				Name:        subPkg,
				Type:        EntityPackage,
				Description: fmt.Sprintf("Internal Hanzo Graph package %s imported by %s", subPkg, fname),
				Keywords:    []string{subPkg, alias, "internal", "package"},
				Aliases:     []string{impPath, alias},
				Weight:      1.1,
			})
			relations = append(relations,
				ExtractedRelation{
					SourceName:  fname,
					SourceType:  EntityFile,
					TargetName:  subPkg,
					TargetType:  EntityPackage,
					Type:        RelImports,
					Description: fmt.Sprintf("File %s imports internal package %s", fname, subPkg),
					Keywords:    []string{fname, pkgPath, subPkg, "import", "dependency"},
					Weight:      1.0,
				},
				ExtractedRelation{
					SourceName:  pkgPath,
					SourceType:  EntityPackage,
					TargetName:  subPkg,
					TargetType:  EntityPackage,
					Type:        RelImports,
					Description: fmt.Sprintf("Package %s depends on internal package %s", pkgPath, subPkg),
					Keywords:    []string{pkgPath, subPkg, "package dependency", "architecture"},
					Weight:      1.1,
				},
			)
		} else if strings.Contains(impPath, ".") {
			// External third-party dependency (skip stdlib noise to keep graph high-signal).
			depName := impPath
			importMap[alias] = importTarget{name: depName, typ: EntityDependency}
			entities = append(entities, ExtractedEntity{
				Name:        depName,
				Type:        EntityDependency,
				Description: fmt.Sprintf("External Go dependency %s used by %s", depName, pkgPath),
				Keywords:    []string{depName, alias, "dependency", "external"},
				Aliases:     []string{alias},
				Weight:      0.9,
			})
			relations = append(relations, ExtractedRelation{
				SourceName:  pkgPath,
				SourceType:  EntityPackage,
				TargetName:  depName,
				TargetType:  EntityDependency,
				Type:        RelImports,
				Description: fmt.Sprintf("Package %s uses external dependency %s in %s", pkgPath, depName, fname),
				Keywords:    []string{pkgPath, depName, alias, "external dependency"},
				Weight:      0.9,
			})
		}
	}

	// Collect exported type declarations and functions/methods.
	declaredTypes := make(map[string]string) // shortName -> qualifiedName

	for _, decl := range file.Decls {
		gen, ok := decl.(*ast.GenDecl)
		if !ok || gen.Tok != token.TYPE {
			continue
		}
		for _, spec := range gen.Specs {
			ts, ok := spec.(*ast.TypeSpec)
			if !ok || ts.Name == nil {
				continue
			}
			typeName := ts.Name.Name
			if !ast.IsExported(typeName) && typeName != "persistentHNSW" && typeName != "oracle" && typeName != "rebuilder" {
				continue
			}
			qualName := pkgPath + "." + typeName
			declaredTypes[typeName] = qualName

			kindStr := describeTypeExpr(ts.Type)
			doc := firstCommentSentence(ts.Doc)
			if doc == "" {
				doc = firstCommentSentence(gen.Doc)
			}
			if doc == "" {
				doc = fmt.Sprintf("Go %s %s defined in %s", kindStr, qualName, fname)
			} else {
				doc = fmt.Sprintf("%s (%s in %s): %s", qualName, kindStr, fname, doc)
			}

			fieldNames, refTargets := inspectTypeRefs(ts.Type, importMap)
			kws := append([]string{qualName, typeName, pkgPath, kindStr}, fieldNames...)

			entities = append(entities, ExtractedEntity{
				Name:        qualName,
				Type:        EntityTypeDecl,
				Description: doc,
				Keywords:    kws,
				Aliases:     []string{typeName, pkgName + "." + typeName},
				Weight:      1.25,
			})
			relations = append(relations,
				ExtractedRelation{
					SourceName:  fname,
					SourceType:  EntityFile,
					TargetName:  qualName,
					TargetType:  EntityTypeDecl,
					Type:        RelDefines,
					Description: fmt.Sprintf("File %s defines %s %s", fname, kindStr, qualName),
					Keywords:    []string{fname, qualName, typeName, kindStr, "defines"},
					Weight:      1.1,
				},
				ExtractedRelation{
					SourceName:  pkgPath,
					SourceType:  EntityPackage,
					TargetName:  qualName,
					TargetType:  EntityTypeDecl,
					Type:        RelDefines,
					Description: fmt.Sprintf("Package %s exports %s %s", pkgPath, kindStr, qualName),
					Keywords:    []string{pkgPath, qualName, typeName, kindStr},
					Weight:      1.0,
				},
			)
			for _, ref := range refTargets {
				entities = append(entities, ExtractedEntity{
					Name:        ref.qualName,
					Type:        EntityTypeDecl,
					Description: fmt.Sprintf("Referenced type %s from package %s", ref.qualName, ref.pkgName),
					Keywords:    []string{ref.qualName, ref.shortName, ref.pkgName},
					Aliases:     []string{ref.shortName},
					Weight:      0.9,
				})
				relations = append(relations, ExtractedRelation{
					SourceName:  qualName,
					SourceType:  EntityTypeDecl,
					TargetName:  ref.qualName,
					TargetType:  EntityTypeDecl,
					Type:        RelReferences,
					Description: fmt.Sprintf("Type %s references %s in %s", qualName, ref.qualName, fname),
					Keywords:    []string{qualName, ref.qualName, pkgPath, ref.pkgName, "type reference"},
					Weight:      1.0,
				})
			}
		}
	}

	// Functions and methods + bounded cross-package call graph extraction.
	funcCount := 0
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Name == nil {
			continue
		}
		fnName := fn.Name.Name
		if !ast.IsExported(fnName) || strings.HasPrefix(fnName, "Test") || strings.HasPrefix(fnName, "Benchmark") {
			continue
		}
		if funcCount >= 40 {
			break
		}
		funcCount++

		doc := firstCommentSentence(fn.Doc)
		recvType := receiverTypeName(fn.Recv)
		var entName string
		var entType EntityType
		var aliases []string
		if recvType != "" {
			entName = pkgPath + "." + recvType + "." + fnName
			entType = EntityMethod
			aliases = []string{recvType + "." + fnName, fnName}
		} else {
			entName = pkgPath + "." + fnName
			entType = EntityFunction
			aliases = []string{pkgName + "." + fnName, fnName}
		}
		if doc == "" {
			doc = fmt.Sprintf("Exported %s %s in %s", entType, entName, fname)
		} else {
			doc = fmt.Sprintf("%s in %s: %s", entName, fname, doc)
		}

		entities = append(entities, ExtractedEntity{
			Name:        entName,
			Type:        entType,
			Description: doc,
			Keywords:    []string{entName, fnName, pkgPath, recvType},
			Aliases:     aliases,
			Weight:      1.05,
		})
		relations = append(relations, ExtractedRelation{
			SourceName:  fname,
			SourceType:  EntityFile,
			TargetName:  entName,
			TargetType:  entType,
			Type:        RelDefines,
			Description: fmt.Sprintf("File %s defines %s %s", fname, entType, entName),
			Keywords:    []string{fname, entName, fnName, pkgPath},
			Weight:      1.0,
		})
		if recvType != "" {
			recvQual := pkgPath + "." + recvType
			entities = append(entities, ExtractedEntity{
				Name:        recvQual,
				Type:        EntityTypeDecl,
				Description: fmt.Sprintf("Receiver type %s in package %s", recvQual, pkgPath),
				Keywords:    []string{recvQual, recvType, pkgPath},
				Aliases:     []string{recvType},
				Weight:      1.0,
			})
			relations = append(relations, ExtractedRelation{
				SourceName:  recvQual,
				SourceType:  EntityTypeDecl,
				TargetName:  entName,
				TargetType:  entType,
				Type:        RelDefines,
				Description: fmt.Sprintf("Type %s implements method %s", recvQual, entName),
				Keywords:    []string{recvQual, entName, fnName, "method"},
				Weight:      1.1,
			})
		}

		// Extract up to 8 distinct cross-package selector calls inside the function body.
		if fn.Body != nil {
			seenCalls := make(map[string]struct{})
			ast.Inspect(fn.Body, func(n ast.Node) bool {
				if len(seenCalls) >= 8 {
					return false
				}
				call, ok := n.(*ast.CallExpr)
				if !ok {
					return true
				}
				sel, ok := call.Fun.(*ast.SelectorExpr)
				if !ok || sel.Sel == nil || !ast.IsExported(sel.Sel.Name) {
					return true
				}
				ident, ok := sel.X.(*ast.Ident)
				if !ok {
					return true
				}
				targetPkg, isImported := importMap[ident.Name]
				if !isImported || targetPkg.typ != EntityPackage {
					return true
				}
				calleeQual := targetPkg.name + "." + sel.Sel.Name
				if _, dup := seenCalls[calleeQual]; dup {
					return true
				}
				seenCalls[calleeQual] = struct{}{}
				entities = append(entities, ExtractedEntity{
					Name:        calleeQual,
					Type:        EntityFunction,
					Description: fmt.Sprintf("Function %s in package %s invoked by %s", calleeQual, targetPkg.name, entName),
					Keywords:    []string{calleeQual, sel.Sel.Name, targetPkg.name},
					Aliases:     []string{sel.Sel.Name},
					Weight:      0.95,
				})
				relations = append(relations, ExtractedRelation{
					SourceName:  entName,
					SourceType:  entType,
					TargetName:  calleeQual,
					TargetType:  EntityFunction,
					Type:        RelCalls,
					Description: fmt.Sprintf("%s invokes %s across packages %s -> %s", entName, calleeQual, pkgPath, targetPkg.name),
					Keywords:    []string{entName, calleeQual, pkgPath, targetPkg.name, "call", "cross-package"},
					Weight:      1.15,
				})
				return true
			})
		}
	}

	// Bridge Go file & package to high-level architectural concepts for Global retrieval.
	conceptEntities, conceptRels := extractDomainConcepts(fname, EntityFile, pkgPath, src.Content)
	entities = append(entities, conceptEntities...)
	relations = append(relations, conceptRels...)

	return ExtractionResult{
		Entities:  entities,
		Relations: relations,
	}, true
}

func (e *DefaultExtractor) extractStructuredText(src SourceRecord) ExtractionResult {
	var entities []ExtractedEntity
	var relations []ExtractedRelation

	anchorName := src.Path
	if anchorName == "" {
		anchorName = src.Title
	}
	if anchorName == "" {
		anchorName = src.SourceID
	}
	anchorType := entityTypeForSourceKind(src.Kind)
	anchorDesc := src.Title
	if anchorDesc == "" {
		anchorDesc = firstNonEmptyLine(src.Content)
	}
	if len(anchorDesc) > 220 {
		anchorDesc = anchorDesc[:220]
	}

	entities = append(entities, ExtractedEntity{
		Name:        anchorName,
		Type:        anchorType,
		Description: anchorDesc,
		Keywords:    append([]string{anchorName, string(src.Kind)}, topTerms(src.Title+" "+src.Content, 8)...),
		Aliases:     nonEmptyStrings(src.SourceID, src.Title),
		Weight:      1.2,
	})

	// 1. Structured engineering headers (Commits, ADRs, Incidents, Agent Events).
	lines := strings.Split(src.Content, "\n")
	for _, rawLine := range lines {
		line := strings.TrimSpace(rawLine)
		if line == "" || !strings.Contains(line, ":") {
			continue
		}
		idx := strings.IndexByte(line, ':')
		if idx <= 0 || idx > 24 {
			continue
		}
		key := strings.ToLower(strings.TrimSpace(line[:idx]))
		val := strings.TrimSpace(line[idx+1:])
		if val == "" {
			continue
		}
		switch key {
		case "files", "modifies", "file":
			for _, item := range splitListValue(val) {
				entities = append(entities, ExtractedEntity{
					Name:        item,
					Type:        EntityFile,
					Description: fmt.Sprintf("Source file %s modified or referenced by %s", item, anchorName),
					Keywords:    []string{item, filepath.Base(item), "file"},
					Aliases:     []string{filepath.Base(item)},
					Weight:      1.1,
				})
				relations = append(relations, ExtractedRelation{
					SourceName:  anchorName,
					SourceType:  anchorType,
					TargetName:  item,
					TargetType:  EntityFile,
					Type:        RelModifies,
					Description: fmt.Sprintf("%s modifies or targets file %s: %s", anchorName, item, anchorDesc),
					Keywords:    []string{anchorName, item, "modifies", "file"},
					Weight:      1.2,
				})
			}
		case "package", "packages":
			for _, item := range splitListValue(val) {
				entities = append(entities, ExtractedEntity{
					Name:        item,
					Type:        EntityPackage,
					Description: fmt.Sprintf("Package %s associated with %s", item, anchorName),
					Keywords:    []string{item, "package"},
					Weight:      1.1,
				})
				relations = append(relations, ExtractedRelation{
					SourceName:  anchorName,
					SourceType:  anchorType,
					TargetName:  item,
					TargetType:  EntityPackage,
					Type:        RelReferences,
					Description: fmt.Sprintf("%s references package %s", anchorName, item),
					Keywords:    []string{anchorName, item, "package"},
					Weight:      1.0,
				})
			}
		case "depends-on", "depends_on", "requires":
			for _, item := range splitListValue(val) {
				t := inferEntityType(item)
				entities = append(entities, ExtractedEntity{
					Name:        item,
					Type:        t,
					Description: fmt.Sprintf("Dependency %s required by %s", item, anchorName),
					Keywords:    []string{item, "dependency"},
					Weight:      1.1,
				})
				relations = append(relations, ExtractedRelation{
					SourceName:  anchorName,
					SourceType:  anchorType,
					TargetName:  item,
					TargetType:  t,
					Type:        RelDependsOn,
					Description: fmt.Sprintf("%s depends on %s", anchorName, item),
					Keywords:    []string{anchorName, item, "depends_on"},
					Weight:      1.2,
				})
			}
		case "causes", "root-cause", "root_cause":
			for _, item := range splitListValue(val) {
				t := inferEntityType(item)
				entities = append(entities, ExtractedEntity{
					Name:        item,
					Type:        t,
					Description: fmt.Sprintf("Causal factor %s linked to %s", item, anchorName),
					Keywords:    []string{item, "root cause", "incident"},
					Weight:      1.25,
				})
				relations = append(relations, ExtractedRelation{
					SourceName:  anchorName,
					SourceType:  anchorType,
					TargetName:  item,
					TargetType:  t,
					Type:        RelCauses,
					Description: fmt.Sprintf("%s is causally linked to %s: %s", anchorName, item, anchorDesc),
					Keywords:    []string{anchorName, item, "causes", "root cause"},
					Weight:      1.3,
				})
			}
		case "mitigates", "fixes", "resolved-by":
			for _, item := range splitListValue(val) {
				t := inferEntityType(item)
				entities = append(entities, ExtractedEntity{
					Name:        item,
					Type:        t,
					Description: fmt.Sprintf("Mitigation or fix target %s linked to %s", item, anchorName),
					Keywords:    []string{item, "fix", "mitigation"},
					Weight:      1.2,
				})
				relations = append(relations, ExtractedRelation{
					SourceName:  anchorName,
					SourceType:  anchorType,
					TargetName:  item,
					TargetType:  t,
					Type:        RelMitigates,
					Description: fmt.Sprintf("%s fixes or mitigates %s: %s", anchorName, item, anchorDesc),
					Keywords:    []string{anchorName, item, "mitigates", "fixes"},
					Weight:      1.25,
				})
			}
		case "decides", "decision":
			for _, item := range splitListValue(val) {
				t := inferEntityType(item)
				entities = append(entities, ExtractedEntity{
					Name:        item,
					Type:        t,
					Description: fmt.Sprintf("Architectural decision target %s in %s", item, anchorName),
					Keywords:    []string{item, "decision", "architecture"},
					Weight:      1.15,
				})
				relations = append(relations, ExtractedRelation{
					SourceName:  anchorName,
					SourceType:  anchorType,
					TargetName:  item,
					TargetType:  t,
					Type:        RelDecides,
					Description: fmt.Sprintf("%s establishes decision for %s: %s", anchorName, item, anchorDesc),
					Keywords:    []string{anchorName, item, "decides", "architecture"},
					Weight:      1.2,
				})
			}
		case "author", "owner":
			for _, item := range splitListValue(val) {
				entities = append(entities, ExtractedEntity{
					Name:        item,
					Type:        EntityAuthor,
					Description: fmt.Sprintf("Author or owner %s of %s", item, anchorName),
					Keywords:    []string{item, "author", "owner"},
					Weight:      0.9,
				})
				relations = append(relations, ExtractedRelation{
					SourceName:  anchorName,
					SourceType:  anchorType,
					TargetName:  item,
					TargetType:  EntityAuthor,
					Type:        RelOwnedBy,
					Description: fmt.Sprintf("%s is authored or owned by %s", anchorName, item),
					Keywords:    []string{anchorName, item, "owner", "author"},
					Weight:      0.9,
				})
			}
		case "symbols", "components", "related":
			for _, item := range splitListValue(val) {
				t := inferEntityType(item)
				entities = append(entities, ExtractedEntity{
					Name:        item,
					Type:        t,
					Description: fmt.Sprintf("Symbol or component %s referenced by %s", item, anchorName),
					Keywords:    []string{item, "component", "symbol"},
					Weight:      1.05,
				})
				relations = append(relations, ExtractedRelation{
					SourceName:  anchorName,
					SourceType:  anchorType,
					TargetName:  item,
					TargetType:  t,
					Type:        RelReferences,
					Description: fmt.Sprintf("%s references %s", anchorName, item),
					Keywords:    []string{anchorName, item, "references"},
					Weight:      1.05,
				})
			}
		}
	}

	// 2. Regex extraction of file paths, package-qualified Go symbols, artifact IDs, and backticked symbols.
	seenMentions := make(map[string]EntityType)

	for _, m := range reFilePath.FindAllStringSubmatch(src.Content, -1) {
		fp := strings.TrimPrefix(m[1], "./")
		if fp == anchorName {
			continue
		}
		seenMentions[fp] = EntityFile
		entities = append(entities, ExtractedEntity{
			Name:        fp,
			Type:        EntityFile,
			Description: fmt.Sprintf("Repository file %s mentioned in %s", fp, anchorName),
			Keywords:    []string{fp, filepath.Base(fp)},
			Aliases:     []string{filepath.Base(fp)},
			Weight:      1.0,
		})
		relations = append(relations, ExtractedRelation{
			SourceName:  anchorName,
			SourceType:  anchorType,
			TargetName:  fp,
			TargetType:  EntityFile,
			Type:        RelReferences,
			Description: fmt.Sprintf("%s mentions repository file %s", anchorName, fp),
			Keywords:    []string{anchorName, fp, "file reference"},
			Weight:      1.0,
		})
	}

	for _, m := range rePkgSymbol.FindAllStringSubmatch(src.Content, -1) {
		pkg := m[1]
		sym := m[2]
		qual := pkg + "." + sym
		seenMentions[qual] = EntityTypeDecl
		entities = append(entities, ExtractedEntity{
			Name:        qual,
			Type:        EntityTypeDecl,
			Description: fmt.Sprintf("Code symbol %s mentioned in %s", qual, anchorName),
			Keywords:    []string{qual, pkg, sym},
			Aliases:     []string{sym},
			Weight:      1.05,
		})
		relations = append(relations, ExtractedRelation{
			SourceName:  anchorName,
			SourceType:  anchorType,
			TargetName:  qual,
			TargetType:  EntityTypeDecl,
			Type:        RelMentions,
			Description: fmt.Sprintf("%s discusses symbol %s: %s", anchorName, qual, anchorDesc),
			Keywords:    []string{anchorName, qual, pkg, sym, "symbol"},
			Weight:      1.05,
		})
	}

	for _, m := range reArtifactID.FindAllStringSubmatch(src.Content, -1) {
		aid := m[1]
		if strings.EqualFold(aid, anchorName) {
			continue
		}
		t := inferEntityType(aid)
		seenMentions[aid] = t
		entities = append(entities, ExtractedEntity{
			Name:        aid,
			Type:        t,
			Description: fmt.Sprintf("Engineering artifact %s linked from %s", aid, anchorName),
			Keywords:    []string{aid, string(t)},
			Weight:      1.1,
		})
		relations = append(relations, ExtractedRelation{
			SourceName:  anchorName,
			SourceType:  anchorType,
			TargetName:  aid,
			TargetType:  t,
			Type:        RelRelatedTo,
			Description: fmt.Sprintf("%s links to engineering artifact %s", anchorName, aid),
			Keywords:    []string{anchorName, aid, "artifact link"},
			Weight:      1.1,
		})
	}

	for _, m := range reBacktickSym.FindAllStringSubmatch(src.Content, -1) {
		sym := strings.TrimSpace(m[1])
		if len(sym) < 3 || strings.Contains(sym, " ") || strings.HasPrefix(sym, "-") {
			continue
		}
		if _, already := seenMentions[sym]; already || sym == anchorName {
			continue
		}
		t := inferEntityType(sym)
		seenMentions[sym] = t
		entities = append(entities, ExtractedEntity{
			Name:        sym,
			Type:        t,
			Description: fmt.Sprintf("Identifier %s referenced in %s", sym, anchorName),
			Keywords:    []string{sym},
			Weight:      0.9,
		})
		relations = append(relations, ExtractedRelation{
			SourceName:  anchorName,
			SourceType:  anchorType,
			TargetName:  sym,
			TargetType:  t,
			Type:        RelMentions,
			Description: fmt.Sprintf("%s references identifier %s", anchorName, sym),
			Keywords:    []string{anchorName, sym},
			Weight:      0.85,
		})
	}

	// 3. Domain concepts for Global high-level thematic retrieval.
	conceptEntities, conceptRels := extractDomainConcepts(anchorName, anchorType, "", src.Title+"\n"+src.Content)
	entities = append(entities, conceptEntities...)
	relations = append(relations, conceptRels...)

	// 4. Pairwise co-occurrence links between top high-signal mentioned entities in the same document
	// (bounded to keep graph sparse and deterministic).
	var sortedMentions []string
	for k := range seenMentions {
		sortedMentions = append(sortedMentions, k)
	}
	sort.Strings(sortedMentions)
	if len(sortedMentions) > 6 {
		sortedMentions = sortedMentions[:6]
	}
	for i := 0; i < len(sortedMentions); i++ {
		for j := i + 1; j < len(sortedMentions); j++ {
			a, b := sortedMentions[i], sortedMentions[j]
			relations = append(relations, ExtractedRelation{
				SourceName:  a,
				SourceType:  seenMentions[a],
				TargetName:  b,
				TargetType:  seenMentions[b],
				Type:        RelRelatedTo,
				Description: fmt.Sprintf("%s and %s co-occur in %s: %s", a, b, anchorName, anchorDesc),
				Keywords:    []string{a, b, anchorName},
				Weight:      0.75,
			})
		}
	}

	return ExtractionResult{
		Entities:  entities,
		Relations: relations,
	}
}

func extractDomainConcepts(anchorName string, anchorType EntityType, pkgPath, text string) ([]ExtractedEntity, []ExtractedRelation) {
	lower := strings.ToLower(text)
	var entities []ExtractedEntity
	var relations []ExtractedRelation

	for _, c := range knownConcepts {
		hits := 0
		for _, trig := range c.Triggers {
			if strings.Contains(lower, trig) {
				hits++
			}
		}
		if hits == 0 {
			continue
		}
		weight := 0.8 + 0.15*float64(hits)
		if weight > 1.5 {
			weight = 1.5
		}
		entities = append(entities, ExtractedEntity{
			Name:        c.Canonical,
			Type:        EntityConcept,
			Description: fmt.Sprintf("Architectural theme %s covering %s", c.Canonical, strings.Join(c.Keywords, ", ")),
			Keywords:    c.Keywords,
			Weight:      weight,
		})
		relations = append(relations, ExtractedRelation{
			SourceName:  anchorName,
			SourceType:  anchorType,
			TargetName:  c.Canonical,
			TargetType:  EntityConcept,
			Type:        RelRelatedTo,
			Description: fmt.Sprintf("%s participates in architectural theme %s (%s)", anchorName, c.Canonical, strings.Join(c.Keywords, ", ")),
			Keywords:    append([]string{anchorName, c.Canonical}, c.Keywords...),
			Weight:      weight,
		})
		if pkgPath != "" {
			relations = append(relations, ExtractedRelation{
				SourceName:  pkgPath,
				SourceType:  EntityPackage,
				TargetName:  c.Canonical,
				TargetType:  EntityConcept,
				Type:        RelRelatedTo,
				Description: fmt.Sprintf("Package %s contributes to %s (%s)", pkgPath, c.Canonical, strings.Join(c.Keywords, ", ")),
				Keywords:    append([]string{pkgPath, c.Canonical}, c.Keywords...),
				Weight:      weight,
			})
		}
	}
	return entities, relations
}

// DecomposeQuery implements LightRAG's dual-level query decomposition into
// low-level (entity/identifier-specific) and high-level (thematic/relational) keywords.
func DecomposeQuery(queryText string) (lowLevel []string, highLevel []string) {
	seenLow := make(map[string]struct{})
	seenHigh := make(map[string]struct{})
	addLow := func(s string) {
		s = strings.TrimSpace(s)
		if s == "" {
			return
		}
		k := strings.ToLower(s)
		if _, ok := seenLow[k]; ok {
			return
		}
		seenLow[k] = struct{}{}
		lowLevel = append(lowLevel, s)
	}
	addHigh := func(s string) {
		s = strings.TrimSpace(s)
		if s == "" {
			return
		}
		k := strings.ToLower(s)
		if _, ok := seenHigh[k]; ok {
			return
		}
		seenHigh[k] = struct{}{}
		highLevel = append(highLevel, s)
	}

	// File paths, qualified symbols, artifact IDs, and backticked identifiers are low-level entity targets.
	for _, m := range reFilePath.FindAllStringSubmatch(queryText, -1) {
		addLow(m[1])
	}
	for _, m := range rePkgSymbol.FindAllStringSubmatch(queryText, -1) {
		addLow(m[1] + "." + m[2])
		addLow(m[2])
	}
	for _, m := range reArtifactID.FindAllStringSubmatch(queryText, -1) {
		addLow(m[1])
	}
	for _, m := range reBacktickSym.FindAllStringSubmatch(queryText, -1) {
		addLow(m[1])
	}

	// Identify camelCase / PascalCase / snake_case / slash-path words as low-level.
	for _, field := range strings.Fields(queryText) {
		clean := strings.Trim(field, "?,.:;()[]{}\"'`")
		if len(clean) < 2 {
			continue
		}
		if isIdentifierLike(clean) {
			addLow(clean)
		}
	}

	// Match high-level domain themes and content tokens.
	lower := strings.ToLower(queryText)
	for _, c := range knownConcepts {
		for _, kw := range append(c.Triggers, c.Keywords...) {
			if len(kw) >= 3 && strings.Contains(lower, kw) {
				addHigh(c.Canonical)
				addHigh(kw)
				break
			}
		}
	}

	toks := Tokenize(queryText)
	for _, t := range toks {
		if len(t) <= 2 {
			continue
		}
		if _, isLow := seenLow[t]; !isLow {
			addHigh(t)
		}
	}
	// Ensure neither channel is empty so dual-level retrieval always has signal.
	if len(lowLevel) == 0 {
		for _, t := range toks {
			if len(t) > 2 {
				addLow(t)
			}
		}
	}
	if len(highLevel) == 0 {
		for _, t := range toks {
			if len(t) > 2 {
				addHigh(t)
			}
		}
	}
	return lowLevel, highLevel
}

func isIdentifierLike(s string) bool {
	if strings.ContainsAny(s, "_/.") {
		return true
	}
	hasUpper := false
	hasLower := false
	hasInternalUpper := false
	for i, r := range s {
		if unicode.IsUpper(r) {
			hasUpper = true
			if i > 0 {
				hasInternalUpper = true
			}
		} else if unicode.IsLower(r) {
			hasLower = true
		}
	}
	if hasInternalUpper && hasLower {
		return true
	}
	// Acronyms like MVCC, HNSW, DQL, ACL, UID.
	if hasUpper && !hasLower && len(s) >= 2 && len(s) <= 8 {
		return true
	}
	return false
}

func deriveGoPackagePath(fname, pkgName string) string {
	dir := filepath.ToSlash(filepath.Dir(fname))
	dir = strings.TrimPrefix(dir, "./")
	if dir == "." || dir == "" || dir == "/" {
		return pkgName
	}
	return dir
}

func firstCommentSentence(cg *ast.CommentGroup) string {
	if cg == nil {
		return ""
	}
	txt := strings.TrimSpace(cg.Text())
	if txt == "" {
		return ""
	}
	txt = strings.Join(strings.Fields(txt), " ")
	if len(txt) > 220 {
		txt = txt[:220]
	}
	return txt
}

func describeTypeExpr(expr ast.Expr) string {
	switch expr.(type) {
	case *ast.StructType:
		return "struct"
	case *ast.InterfaceType:
		return "interface"
	case *ast.FuncType:
		return "func type"
	case *ast.MapType:
		return "map type"
	case *ast.ArrayType:
		return "slice type"
	default:
		return "type"
	}
}

type importTarget struct {
	name string
	typ  EntityType
}

type typeRef struct {
	pkgName   string
	shortName string
	qualName  string
}

func inspectTypeRefs(expr ast.Expr, importMap map[string]importTarget) ([]string, []typeRef) {
	var fieldNames []string
	var refs []typeRef
	seenRef := make(map[string]struct{})

	collectExpr := func(e ast.Expr) {
		ast.Inspect(e, func(n ast.Node) bool {
			sel, ok := n.(*ast.SelectorExpr)
			if !ok || sel.Sel == nil {
				return true
			}
			ident, ok := sel.X.(*ast.Ident)
			if !ok {
				return true
			}
			imp, ok := importMap[ident.Name]
			if !ok || imp.typ != EntityPackage {
				return true
			}
			q := imp.name + "." + sel.Sel.Name
			if _, dup := seenRef[q]; dup {
				return true
			}
			seenRef[q] = struct{}{}
			refs = append(refs, typeRef{
				pkgName:   imp.name,
				shortName: sel.Sel.Name,
				qualName:  q,
			})
			return true
		})
	}

	switch t := expr.(type) {
	case *ast.StructType:
		if t.Fields != nil {
			for _, f := range t.Fields.List {
				for _, name := range f.Names {
					if ast.IsExported(name.Name) && len(fieldNames) < 12 {
						fieldNames = append(fieldNames, name.Name)
					}
				}
				collectExpr(f.Type)
			}
		}
	case *ast.InterfaceType:
		if t.Methods != nil {
			for _, m := range t.Methods.List {
				for _, name := range m.Names {
					if ast.IsExported(name.Name) && len(fieldNames) < 12 {
						fieldNames = append(fieldNames, name.Name)
					}
				}
				collectExpr(m.Type)
			}
		}
	default:
		collectExpr(expr)
	}
	return fieldNames, refs
}

func receiverTypeName(fl *ast.FieldList) string {
	if fl == nil || len(fl.List) == 0 {
		return ""
	}
	expr := fl.List[0].Type
	for {
		switch t := expr.(type) {
		case *ast.StarExpr:
			expr = t.X
		case *ast.IndexExpr:
			expr = t.X
		case *ast.IndexListExpr:
			expr = t.X
		case *ast.Ident:
			return t.Name
		default:
			return ""
		}
	}
}

func entityTypeForSourceKind(kind SourceKind) EntityType {
	switch kind {
	case KindCommit:
		return EntityCommit
	case KindDecision:
		return EntityDecision
	case KindIncident:
		return EntityIncident
	case KindCodeBlob:
		return EntityFile
	case KindAgentEvent:
		return EntityComponent
	default:
		return EntityFile
	}
}

func inferEntityType(s string) EntityType {
	upper := strings.ToUpper(s)
	switch {
	case strings.HasPrefix(upper, "ADR-"):
		return EntityDecision
	case strings.HasPrefix(upper, "INC-"):
		return EntityIncident
	case strings.HasPrefix(upper, "COMMIT-") || strings.HasPrefix(upper, "PR-"):
		return EntityCommit
	case strings.HasSuffix(s, ".go") || strings.HasSuffix(s, ".md") || strings.HasSuffix(s, ".proto"):
		return EntityFile
	case strings.Contains(s, ".") && !strings.Contains(s, "/"):
		return EntityTypeDecl
	case strings.Contains(s, "/"):
		return EntityPackage
	default:
		return EntityComponent
	}
}

func splitListValue(val string) []string {
	parts := strings.FieldsFunc(val, func(r rune) bool {
		return r == ',' || r == ';'
	})
	var out []string
	for _, p := range parts {
		p = strings.TrimSpace(strings.Trim(p, "`\"'"))
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

func firstNonEmptyLine(s string) string {
	for _, l := range strings.Split(s, "\n") {
		l = strings.TrimSpace(strings.TrimLeft(l, "#*>- "))
		if l != "" {
			return l
		}
	}
	return ""
}

func topTerms(text string, limit int) []string {
	toks := Tokenize(text)
	freq := make(map[string]int, len(toks))
	for _, t := range toks {
		if len(t) >= 3 {
			freq[t]++
		}
	}
	type kv struct {
		k string
		v int
	}
	arr := make([]kv, 0, len(freq))
	for k, v := range freq {
		arr = append(arr, kv{k, v})
	}
	sort.Slice(arr, func(i, j int) bool {
		if arr[i].v != arr[j].v {
			return arr[i].v > arr[j].v
		}
		return arr[i].k < arr[j].k
	})
	if len(arr) > limit {
		arr = arr[:limit]
	}
	out := make([]string, len(arr))
	for i := range arr {
		out[i] = arr[i].k
	}
	return out
}

func nonEmptyStrings(vals ...string) []string {
	var out []string
	seen := make(map[string]struct{})
	for _, v := range vals {
		v = strings.TrimSpace(v)
		if v == "" {
			continue
		}
		if _, ok := seen[v]; ok {
			continue
		}
		seen[v] = struct{}{}
		out = append(out, v)
	}
	return out
}
