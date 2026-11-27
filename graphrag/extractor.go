package graphrag

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// Extractor extracts derived entities, weighted relations, and high-level
// themes from a single CanonicalDocument.
type Extractor interface {
	Extract(doc CanonicalDocument) ExtractionResult
}

// DefaultExtractor combines a clean-room Go AST extractor (go/parser + go/ast)
// with a deterministic structured engineering-text extractor for regression
// reports, postmortems, ADRs, commits, and documentation.
type DefaultExtractor struct{}

// NewDefaultExtractor returns the standard deterministic stdlib extractor.
func NewDefaultExtractor() *DefaultExtractor {
	return &DefaultExtractor{}
}

type themeRule struct {
	Theme    string
	Keywords []string
}

var engineeringThemeRules = []themeRule{
	{
		Theme:    "concurrency_and_locking",
		Keywords: []string{"mutex", "rwmutex", "deadlock", "race", "lock", "atomic", "goroutine", "channel", "concurrency", "sharded", "contention", "sync"},
	},
	{
		Theme:    "mvcc_and_posting_consistency",
		Keywords: []string{"posting", "mvcc", "readts", "oracle", "rollup", "commit", "delta", "mutation", "transaction", "stale", "snapshot", "uidlist"},
	},
	{
		Theme:    "query_parsing_and_dql",
		Keywords: []string{"dql", "parser", "lexer", "lex", "ast", "upsert", "filter", "recurse", "pagination", "syntax", "iri", "directive"},
	},
	{
		Theme:    "indexing_and_vector_hnsw",
		Keywords: []string{"hnsw", "bleve", "tokenizer", "tok", "index", "stemmer", "stopword", "fulltext", "trigram", "vector", "nearest", "searchlayer"},
	},
	{
		Theme:    "memory_allocation_and_codec",
		Keywords: []string{"codec", "pack", "unpack", "buffer", "alloc", "heap", "oom", "memory", "cache", "sketch", "slice", "pool", "size"},
	},
	{
		Theme:    "schema_and_type_conversion",
		Keywords: []string{"schema", "conversion", "scalar", "facet", "geofilter", "s2", "earth", "password", "parse", "coercion", "type"},
	},
	{
		Theme:    "security_acl_and_encryption",
		Keywords: []string{"acl", "vault", "tls", "encrypt", "decrypt", "fips", "key", "auth", "token", "permission", "sensitive", "tenant"},
	},
	{
		Theme:    "observability_and_ops",
		Keywords: []string{"metrics", "histogram", "health", "profile", "logger", "ulimit", "disk", "minio", "server", "config", "flag", "telemetry"},
	},
}

var (
	reIncidentID = regexp.MustCompile(`\b(INC|REG|ADR|BUG|PR|HIP)-[0-9]{1,6}\b`)
	rePkgRef     = regexp.MustCompile(`\b(algo|buildvars|codec|dql|lex|posting|schema|tok|types|x)(/[a-z0-9_]+)?\b`)
	reGoFileRef  = regexp.MustCompile(`\b[a-z0-9_]+/[a-z0-9_/-]+\.go\b`)
	reCamelSym   = regexp.MustCompile(`\b[A-Z][a-zA-Z0-9]{3,}\b`)
)

// Extract deterministically produces entities, undirected relations, and themes
// grounded in doc's canonical SourceRef.
func (e *DefaultExtractor) Extract(doc CanonicalDocument) ExtractionResult {
	ref := SourceRef{
		SourceID: doc.SourceID,
		Version:  doc.Version,
		TenantID: doc.TenantID,
	}
	if doc.Kind == "go_source" || strings.HasSuffix(strings.ToLower(doc.SourceID), ".go") {
		if res, ok := e.extractGoAST(doc, ref); ok {
			return res
		}
	}
	return e.extractTechnicalText(doc, ref)
}

func (e *DefaultExtractor) extractGoAST(doc CanonicalDocument, ref SourceRef) (ExtractionResult, bool) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, doc.SourceID, doc.Content, parser.ParseComments)
	if err != nil {
		return ExtractionResult{}, false
	}

	entityMap := make(map[string]*Entity)
	edgeMap := make(map[string]*Relation)

	addEntity := func(id, name, typ, desc string, weight float64, themes []string) {
		if id == "" {
			return
		}
		if cur, ok := entityMap[id]; ok {
			cur.Weight += weight
			if desc != "" && !strings.Contains(cur.Description, desc) && len(cur.Description) < 220 {
				if cur.Description != "" {
					cur.Description += "; "
				}
				cur.Description += desc
			}
			cur.Themes = dedupStringsSorted(append(cur.Themes, themes...))
			return
		}
		entityMap[id] = &Entity{
			ID:          id,
			Name:        name,
			Type:        typ,
			Description: desc,
			Themes:      dedupStringsSorted(themes),
			Weight:      weight,
			SourceRefs:  []SourceRef{ref},
		}
	}

	addEdge := func(a, b, kind, desc string, weight float64) {
		if a == "" || b == "" || a == b {
			return
		}
		if a > b {
			a, b = b, a
		}
		key := a + "|" + b + "|" + kind
		if cur, ok := edgeMap[key]; ok {
			cur.Weight += weight
			return
		}
		edgeMap[key] = &Relation{
			SourceEntityID: a,
			TargetEntityID: b,
			Kind:           kind,
			Description:    desc,
			Weight:         weight,
			SourceRefs:     []SourceRef{ref},
		}
	}

	themes := detectThemes(doc.SourceID + " " + doc.Title + " " + doc.Content)
	pkgDir := filepath.ToSlash(filepath.Dir(doc.SourceID))
	pkgLabel := file.Name.Name
	if pkgDir != "" && pkgDir != "." {
		pkgLabel = pkgDir
	}
	pkgID := NormalizeEntityID("package", pkgLabel)
	fileID := NormalizeEntityID("file", filepath.ToSlash(doc.SourceID))

	addEntity(pkgID, pkgLabel, "package", "Go package "+pkgLabel, 2.0, themes)
	addEntity(fileID, filepath.ToSlash(doc.SourceID), "file", "Go source file "+doc.SourceID+" in package "+pkgLabel, 1.5, themes)
	addEdge(pkgID, fileID, "contains", pkgLabel+" contains "+doc.SourceID, 2.0)

	// Connect detected themes as thematic hub nodes so cross-package functional
	// communities naturally emerge while keeping package modularity high.
	for _, th := range themes {
		thID := NormalizeEntityID("theme", th)
		addEntity(thID, th, "theme", "Engineering theme "+th, 1.0, []string{th})
		addEdge(pkgID, thID, "exhibits_theme", pkgLabel+" exhibits "+th, 1.2)
		addEdge(fileID, thID, "exhibits_theme", doc.SourceID+" exhibits "+th, 0.8)
	}

	// Extract internal Hanzo Graph package imports.
	for _, imp := range file.Imports {
		pathVal := strings.Trim(imp.Path.Value, "\"")
		if strings.HasPrefix(pathVal, "github.com/hanzoai/graph/") {
			depPkg := strings.TrimPrefix(pathVal, "github.com/hanzoai/graph/")
			depID := NormalizeEntityID("package", depPkg)
			addEntity(depID, depPkg, "package", "Imported internal package "+depPkg, 1.0, nil)
			addEdge(pkgID, depID, "imports", pkgLabel+" imports "+depPkg, 1.5)
			addEdge(fileID, depID, "imports", doc.SourceID+" imports "+depPkg, 1.0)
		}
	}

	// Extract exported types and top-level functions (capped deterministically to
	// keep graphs compact and high-signal).
	var typeNames []string
	var funcNames []string
	for _, decl := range file.Decls {
		switch d := decl.(type) {
		case *ast.GenDecl:
			if d.Tok != token.TYPE {
				continue
			}
			for _, spec := range d.Specs {
				ts, ok := spec.(*ast.TypeSpec)
				if !ok || !ts.Name.IsExported() {
					continue
				}
				typeNames = append(typeNames, ts.Name.Name)
			}
		case *ast.FuncDecl:
			if d.Name == nil || !d.Name.IsExported() {
				continue
			}
			name := d.Name.Name
			if d.Recv != nil && len(d.Recv.List) > 0 {
				recv := exprTypeName(d.Recv.List[0].Type)
				if recv != "" {
					name = recv + "." + name
				}
			}
			funcNames = append(funcNames, name)
		}
	}

	sort.Strings(typeNames)
	sort.Strings(funcNames)
	if len(typeNames) > 12 {
		typeNames = typeNames[:12]
	}
	if len(funcNames) > 14 {
		funcNames = funcNames[:14]
	}

	for _, tn := range typeNames {
		symID := NormalizeEntityID("symbol", pkgLabel+"."+tn)
		addEntity(symID, pkgLabel+"."+tn, "type", "Exported Go type "+tn+" in "+doc.SourceID, 1.2, themes)
		addEdge(fileID, symID, "defines", doc.SourceID+" defines type "+tn, 1.5)
		addEdge(pkgID, symID, "exposes", pkgLabel+" exposes "+tn, 1.0)
	}
	for _, fn := range funcNames {
		symID := NormalizeEntityID("symbol", pkgLabel+"."+fn)
		addEntity(symID, pkgLabel+"."+fn, "function", "Exported Go function "+fn+" in "+doc.SourceID, 1.0, themes)
		addEdge(fileID, symID, "defines", doc.SourceID+" defines func "+fn, 1.2)
		if dot := strings.IndexByte(fn, '.'); dot > 0 {
			recvType := fn[:dot]
			recvID := NormalizeEntityID("symbol", pkgLabel+"."+recvType)
			if _, exists := entityMap[recvID]; exists {
				addEdge(recvID, symID, "has_method", recvType+" implements "+fn, 1.4)
			}
		}
	}

	return finalizeExtraction(entityMap, edgeMap, themes), true
}

func (e *DefaultExtractor) extractTechnicalText(doc CanonicalDocument, ref SourceRef) ExtractionResult {
	entityMap := make(map[string]*Entity)
	edgeMap := make(map[string]*Relation)

	addEntity := func(id, name, typ, desc string, weight float64, themes []string) {
		if id == "" {
			return
		}
		if cur, ok := entityMap[id]; ok {
			cur.Weight += weight
			if desc != "" && !strings.Contains(cur.Description, desc) && len(cur.Description) < 240 {
				if cur.Description != "" {
					cur.Description += "; "
				}
				cur.Description += desc
			}
			cur.Themes = dedupStringsSorted(append(cur.Themes, themes...))
			return
		}
		entityMap[id] = &Entity{
			ID:          id,
			Name:        name,
			Type:        typ,
			Description: desc,
			Themes:      dedupStringsSorted(themes),
			Weight:      weight,
			SourceRefs:  []SourceRef{ref},
		}
	}

	addEdge := func(a, b, kind, desc string, weight float64) {
		if a == "" || b == "" || a == b {
			return
		}
		if a > b {
			a, b = b, a
		}
		key := a + "|" + b + "|" + kind
		if cur, ok := edgeMap[key]; ok {
			cur.Weight += weight
			return
		}
		edgeMap[key] = &Relation{
			SourceEntityID: a,
			TargetEntityID: b,
			Kind:           kind,
			Description:    desc,
			Weight:         weight,
			SourceRefs:     []SourceRef{ref},
		}
	}

	combined := doc.Title + "\n" + doc.Content
	themes := detectThemes(combined)
	if metaTheme := strings.TrimSpace(doc.Metadata["theme"]); metaTheme != "" {
		themes = dedupStringsSorted(append(themes, strings.ToLower(metaTheme)))
	}
	if metaCat := strings.TrimSpace(doc.Metadata["regression_class"]); metaCat != "" {
		themes = dedupStringsSorted(append(themes, strings.ToLower(metaCat)))
	}

	docKind := doc.Kind
	if docKind == "" {
		docKind = "document"
	}
	docEntityID := NormalizeEntityID(docKind, doc.SourceID)
	summaryLine := firstSentence(doc.Content, 180)
	if summaryLine == "" {
		summaryLine = doc.Title
	}
	addEntity(docEntityID, doc.Title, docKind, summaryLine, 1.5, themes)

	var linkedIDs []string
	linkedIDs = append(linkedIDs, docEntityID)

	// Extract explicit metadata subsystem/root cause if provided.
	if pkg := strings.TrimSpace(doc.Metadata["package"]); pkg != "" {
		pkgID := NormalizeEntityID("package", pkg)
		addEntity(pkgID, pkg, "package", "Subsystem package "+pkg, 1.8, themes)
		addEdge(docEntityID, pkgID, "affects_package", doc.SourceID+" affects "+pkg, 2.2)
		linkedIDs = append(linkedIDs, pkgID)
	}
	if cause := strings.TrimSpace(doc.Metadata["root_cause"]); cause != "" {
		causeID := NormalizeEntityID("root_cause", cause)
		addEntity(causeID, cause, "root_cause", "Root cause "+cause+" ("+summaryLine+")", 1.8, themes)
		addEdge(docEntityID, causeID, "root_cause", doc.SourceID+" caused by "+cause, 2.5)
		linkedIDs = append(linkedIDs, causeID)
	}

	// Extract referenced incident/ADR/HIP IDs.
	for _, m := range dedupStringsSorted(reIncidentID.FindAllString(combined, -1)) {
		id := NormalizeEntityID("artifact", m)
		addEntity(id, m, "artifact", "Engineering record "+m+": "+doc.Title, 1.5, themes)
		addEdge(docEntityID, id, "references", doc.SourceID+" references "+m, 1.8)
		linkedIDs = append(linkedIDs, id)
	}

	// Extract referenced Go packages and files.
	for _, pkg := range dedupStringsSorted(rePkgRef.FindAllString(combined, -1)) {
		pkgID := NormalizeEntityID("package", pkg)
		addEntity(pkgID, pkg, "package", "Hanzo Graph package "+pkg, 1.4, themes)
		addEdge(docEntityID, pkgID, "mentions_package", doc.SourceID+" mentions "+pkg, 1.5)
		linkedIDs = append(linkedIDs, pkgID)
	}
	for _, gf := range dedupStringsSorted(reGoFileRef.FindAllString(combined, -1)) {
		gfID := NormalizeEntityID("file", gf)
		addEntity(gfID, gf, "file", "Referenced Go file "+gf, 1.2, themes)
		addEdge(docEntityID, gfID, "mentions_file", doc.SourceID+" references "+gf, 1.5)
		linkedIDs = append(linkedIDs, gfID)
	}

	// Extract salient CamelCase technical symbols (capped deterministically).
	syms := dedupStringsSorted(reCamelSym.FindAllString(combined, -1))
	if len(syms) > 8 {
		syms = syms[:8]
	}
	for _, sym := range syms {
		if _, isStop := stopWords[strings.ToLower(sym)]; isStop {
			continue
		}
		symID := NormalizeEntityID("symbol", sym)
		addEntity(symID, sym, "symbol", "Technical symbol "+sym+" referenced in "+doc.SourceID, 0.9, themes)
		addEdge(docEntityID, symID, "mentions_symbol", doc.SourceID+" mentions "+sym, 1.0)
	}

	// Connect detected themes and co-occurring entities so thematic communities form cleanly.
	for _, th := range themes {
		thID := NormalizeEntityID("theme", th)
		addEntity(thID, th, "theme", "Engineering theme "+th, 1.5, []string{th})
		addEdge(docEntityID, thID, "exhibits_theme", doc.SourceID+" exhibits "+th, 2.0)
		for _, lid := range linkedIDs {
			if lid != docEntityID && lid != thID {
				addEdge(lid, thID, "associated_theme", lid+" associated with "+th, 1.4)
			}
		}
	}

	// Pairwise co-occurrence between packages/root-causes inside the same report.
	linkedIDs = dedupStringsSorted(linkedIDs)
	for i := 0; i < len(linkedIDs); i++ {
		for j := i + 1; j < len(linkedIDs); j++ {
			if linkedIDs[i] == docEntityID || linkedIDs[j] == docEntityID {
				continue
			}
			addEdge(linkedIDs[i], linkedIDs[j], "co_occurs", "Co-occurs in "+doc.SourceID, 1.0)
		}
	}

	return finalizeExtraction(entityMap, edgeMap, themes)
}

func detectThemes(text string) []string {
	tokens := Tokenize(text)
	if len(tokens) == 0 {
		return nil
	}
	counts := make(map[string]int, len(tokens))
	for _, tok := range tokens {
		counts[tok]++
	}
	type scoredTheme struct {
		theme string
		hits  int
	}
	var scored []scoredTheme
	for _, rule := range engineeringThemeRules {
		hits := 0
		for _, kw := range rule.Keywords {
			if c := counts[kw]; c > 0 {
				hits += c
			}
		}
		if hits > 0 {
			scored = append(scored, scoredTheme{theme: rule.Theme, hits: hits})
		}
	}
	sort.Slice(scored, func(i, j int) bool {
		if scored[i].hits != scored[j].hits {
			return scored[i].hits > scored[j].hits
		}
		return scored[i].theme < scored[j].theme
	})
	limit := len(scored)
	if limit > 3 {
		limit = 3
	}
	out := make([]string, 0, limit)
	for i := 0; i < limit; i++ {
		out = append(out, scored[i].theme)
	}
	sort.Strings(out)
	return out
}

func finalizeExtraction(entityMap map[string]*Entity, edgeMap map[string]*Relation, themes []string) ExtractionResult {
	eKeys := make([]string, 0, len(entityMap))
	for k := range entityMap {
		eKeys = append(eKeys, k)
	}
	sort.Strings(eKeys)
	entities := make([]Entity, 0, len(eKeys))
	for _, k := range eKeys {
		ent := entityMap[k]
		ent.SourceRefs = sortSourceRefs(ent.SourceRefs)
		entities = append(entities, ent.Clone())
	}

	rKeys := make([]string, 0, len(edgeMap))
	for k := range edgeMap {
		rKeys = append(rKeys, k)
	}
	sort.Strings(rKeys)
	relations := make([]Relation, 0, len(rKeys))
	for _, k := range rKeys {
		rel := edgeMap[k]
		rel.SourceRefs = sortSourceRefs(rel.SourceRefs)
		relations = append(relations, rel.Clone())
	}

	return ExtractionResult{
		Entities:  entities,
		Relations: relations,
		Themes:    dedupStringsSorted(themes),
	}
}

func exprTypeName(expr ast.Expr) string {
	switch t := expr.(type) {
	case *ast.Ident:
		return t.Name
	case *ast.StarExpr:
		return exprTypeName(t.X)
	case *ast.IndexExpr:
		return exprTypeName(t.X)
	case *ast.IndexListExpr:
		return exprTypeName(t.X)
	default:
		return ""
	}
}

func firstSentence(text string, maxChars int) string {
	clean := strings.TrimSpace(strings.ReplaceAll(text, "\n", " "))
	for strings.Contains(clean, "  ") {
		clean = strings.ReplaceAll(clean, "  ", " ")
	}
	if clean == "" {
		return ""
	}
	if idx := strings.Index(clean, ". "); idx > 0 && idx < maxChars {
		return clean[:idx+1]
	}
	if len(clean) > maxChars {
		return strings.TrimSpace(clean[:maxChars]) + "..."
	}
	return clean
}
