package graphrag

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// Summarizer generates leaf (L0) community summaries, hierarchical parent
// (L1/L2) rollups from child summaries, Map-Reduce global synthesis answers,
// and matched Local Hybrid grounded answers.
type Summarizer interface {
	SummarizeLeaf(comm Community, entities []Entity, relations []Relation, prevRevision uint64) CommunitySummary
	SummarizeParent(comm Community, childSummaries []CommunitySummary, prevRevision uint64) CommunitySummary
	MapCommunity(query string, summary CommunitySummary, corpusDF map[string]int, totalSummaries int, avgLen float64) PartialAnswer
	ReducePartials(query string, partials []PartialAnswer, tokenBudget int) (answer string, themes []string, citations []SourceRef, promptTok int, completionTok int)
	SynthesizeLocal(query string, hits []SourceHit, tokenBudget int) (answer string, themes []string, citations []SourceRef, promptTok int, completionTok int)
}

// ExtractiveHierarchicalSummarizer is a deterministic, zero-dependency
// stdlib summarizer and map-reduce synthesizer with exact prompt/completion
// token accounting.
type ExtractiveHierarchicalSummarizer struct{}

// NewExtractiveHierarchicalSummarizer constructs the default deterministic summarizer.
func NewExtractiveHierarchicalSummarizer() *ExtractiveHierarchicalSummarizer {
	return &ExtractiveHierarchicalSummarizer{}
}

// SummarizeLeaf builds an L0 community summary grounded strictly in the
// community's entities, internal relations, and canonical SourceRefs.
func (s *ExtractiveHierarchicalSummarizer) SummarizeLeaf(
	comm Community,
	entities []Entity,
	relations []Relation,
	prevRevision uint64,
) CommunitySummary {
	sortedEnts := make([]Entity, len(entities))
	copy(sortedEnts, entities)
	sort.Slice(sortedEnts, func(i, j int) bool {
		if sortedEnts[i].Weight != sortedEnts[j].Weight {
			return sortedEnts[i].Weight > sortedEnts[j].Weight
		}
		return sortedEnts[i].ID < sortedEnts[j].ID
	})

	sortedRels := make([]Relation, len(relations))
	copy(sortedRels, relations)
	sort.Slice(sortedRels, func(i, j int) bool {
		if sortedRels[i].Weight != sortedRels[j].Weight {
			return sortedRels[i].Weight > sortedRels[j].Weight
		}
		if sortedRels[i].SourceEntityID != sortedRels[j].SourceEntityID {
			return sortedRels[i].SourceEntityID < sortedRels[j].SourceEntityID
		}
		return sortedRels[i].TargetEntityID < sortedRels[j].TargetEntityID
	})

	// Estimate prompt tokens from serialized community entities + relations.
	var promptBuf strings.Builder
	promptBuf.WriteString("Summarize community ")
	promptBuf.WriteString(comm.ID)
	promptBuf.WriteString(" entities and relations:\n")
	themeCounts := make(map[string]int)
	var allRefs []SourceRef
	allRefs = append(allRefs, comm.SourceRefs...)

	for _, e := range sortedEnts {
		promptBuf.WriteString(e.Type)
		promptBuf.WriteString(" ")
		promptBuf.WriteString(e.Name)
		promptBuf.WriteString(": ")
		promptBuf.WriteString(e.Description)
		promptBuf.WriteByte('\n')
		for _, th := range e.Themes {
			themeCounts[th]++
		}
		allRefs = append(allRefs, e.SourceRefs...)
	}
	for _, r := range sortedRels {
		promptBuf.WriteString(r.SourceEntityID)
		promptBuf.WriteString(" -[")
		promptBuf.WriteString(r.Kind)
		promptBuf.WriteString("]-> ")
		promptBuf.WriteString(r.TargetEntityID)
		promptBuf.WriteString(": ")
		promptBuf.WriteString(r.Description)
		promptBuf.WriteByte('\n')
		allRefs = append(allRefs, r.SourceRefs...)
	}

	topEntities := make([]string, 0, 8)
	var packages []string
	var rootCauses []string
	var findings []string
	findingSeen := make(map[string]struct{})

	for _, e := range sortedEnts {
		if len(topEntities) < 8 && e.Type != "theme" {
			topEntities = append(topEntities, e.Name)
		}
		if e.Type == "package" {
			packages = append(packages, e.Name)
		}
		if e.Type == "root_cause" {
			rootCauses = append(rootCauses, e.Name)
		}
		if e.Description != "" && len(findings) < 6 && e.Type != "theme" {
			line := fmt.Sprintf("[%s] %s: %s", e.Type, e.Name, e.Description)
			if _, dup := findingSeen[line]; !dup {
				findingSeen[line] = struct{}{}
				findings = append(findings, line)
			}
		}
	}
	for _, r := range sortedRels {
		if len(findings) >= 8 {
			break
		}
		if r.Description != "" && (r.Kind == "root_cause" || r.Kind == "affects_package" || r.Kind == "imports" || r.Kind == "exhibits_theme") {
			line := fmt.Sprintf("[rel:%s] %s", r.Kind, r.Description)
			if _, dup := findingSeen[line]; !dup {
				findingSeen[line] = struct{}{}
				findings = append(findings, line)
			}
		}
	}

	rankedThemes := rankThemeCounts(themeCounts)
	title := buildCommunityTitle(0, rankedThemes, packages, topEntities)

	var sumBuf strings.Builder
	sumBuf.WriteString(title)
	sumBuf.WriteString(". ")
	if len(rankedThemes) > 0 {
		sumBuf.WriteString("Primary themes: ")
		sumBuf.WriteString(strings.Join(limitSlice(rankedThemes, 4), ", "))
		sumBuf.WriteString(". ")
	}
	if len(packages) > 0 {
		sumBuf.WriteString("Subsystems: ")
		sumBuf.WriteString(strings.Join(limitSlice(dedupStringsSorted(packages), 5), ", "))
		sumBuf.WriteString(". ")
	}
	if len(rootCauses) > 0 {
		sumBuf.WriteString("Root causes: ")
		sumBuf.WriteString(strings.Join(limitSlice(dedupStringsSorted(rootCauses), 4), ", "))
		sumBuf.WriteString(". ")
	}
	if len(topEntities) > 0 {
		sumBuf.WriteString("Key entities: ")
		sumBuf.WriteString(strings.Join(limitSlice(topEntities, 6), ", "))
		sumBuf.WriteString(". ")
	}
	sortedRefs := sortSourceRefs(allRefs)
	sumBuf.WriteString(fmt.Sprintf("Grounded in %d canonical sources (%d entities, %d relations).", len(sortedRefs), len(sortedEnts), len(sortedRels)))

	summaryText := sumBuf.String()
	promptTokens := EstimateTokens(promptBuf.String())
	completionTokens := EstimateTokens(summaryText + "\n" + strings.Join(findings, "\n"))

	return CommunitySummary{
		CommunityID:      comm.ID,
		TenantID:         comm.TenantID,
		ScopeKey:         comm.ScopeKey,
		ACL:              append([]string(nil), comm.ACL...),
		Level:            0,
		Title:            title,
		Summary:          summaryText,
		Findings:         findings,
		TopEntities:      topEntities,
		ThemeCounts:      themeCounts,
		SourceRefs:       sortedRefs,
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
		Revision:         prevRevision + 1,
		Dirty:            false,
	}
}

// SummarizeParent builds an L1 or L2 community summary by hierarchically
// rolling up its child community summaries (without re-reading raw documents).
func (s *ExtractiveHierarchicalSummarizer) SummarizeParent(
	comm Community,
	childSummaries []CommunitySummary,
	prevRevision uint64,
) CommunitySummary {
	sortedChildren := make([]CommunitySummary, len(childSummaries))
	copy(sortedChildren, childSummaries)
	sort.Slice(sortedChildren, func(i, j int) bool {
		return sortedChildren[i].CommunityID < sortedChildren[j].CommunityID
	})

	var promptBuf strings.Builder
	promptBuf.WriteString(fmt.Sprintf("Roll up %d child summaries into L%d community %s:\n", len(sortedChildren), comm.Level, comm.ID))

	themeCounts := make(map[string]int)
	entityFreq := make(map[string]int)
	var findings []string
	findingSeen := make(map[string]struct{})
	var allRefs []SourceRef
	allRefs = append(allRefs, comm.SourceRefs...)

	for _, cs := range sortedChildren {
		promptBuf.WriteString(cs.Title)
		promptBuf.WriteString(": ")
		promptBuf.WriteString(cs.Summary)
		promptBuf.WriteByte('\n')
		for th, cnt := range cs.ThemeCounts {
			themeCounts[th] += cnt
		}
		for _, ent := range cs.TopEntities {
			entityFreq[ent]++
		}
		for _, f := range cs.Findings {
			if len(findings) >= 10 {
				break
			}
			if _, dup := findingSeen[f]; !dup {
				findingSeen[f] = struct{}{}
				findings = append(findings, f)
			}
		}
		allRefs = append(allRefs, cs.SourceRefs...)
	}

	rankedThemes := rankThemeCounts(themeCounts)
	topEntities := rankStringCounts(entityFreq, 10)
	title := buildCommunityTitle(comm.Level, rankedThemes, nil, topEntities)

	var sumBuf strings.Builder
	sumBuf.WriteString(title)
	sumBuf.WriteString(". ")
	if len(rankedThemes) > 0 {
		var parts []string
		for _, th := range limitSlice(rankedThemes, 5) {
			parts = append(parts, fmt.Sprintf("%s (%d)", th, themeCounts[th]))
		}
		sumBuf.WriteString("Aggregated themes: ")
		sumBuf.WriteString(strings.Join(parts, ", "))
		sumBuf.WriteString(". ")
	}
	if len(topEntities) > 0 {
		sumBuf.WriteString("Cross-community entities: ")
		sumBuf.WriteString(strings.Join(limitSlice(topEntities, 8), ", "))
		sumBuf.WriteString(". ")
	}
	sortedRefs := sortSourceRefs(allRefs)
	sumBuf.WriteString(fmt.Sprintf("Synthesizes %d child communities across %d canonical sources.", len(sortedChildren), len(sortedRefs)))

	summaryText := sumBuf.String()
	promptTokens := EstimateTokens(promptBuf.String())
	completionTokens := EstimateTokens(summaryText + "\n" + strings.Join(findings, "\n"))

	return CommunitySummary{
		CommunityID:      comm.ID,
		TenantID:         comm.TenantID,
		ScopeKey:         comm.ScopeKey,
		ACL:              append([]string(nil), comm.ACL...),
		Level:            comm.Level,
		Title:            title,
		Summary:          summaryText,
		Findings:         findings,
		TopEntities:      topEntities,
		ThemeCounts:      themeCounts,
		SourceRefs:       sortedRefs,
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
		Revision:         prevRevision + 1,
		Dirty:            false,
	}
}

// MapCommunity executes the GraphRAG Map phase on a single authorized community
// summary, scoring relevance and extracting grounded partial key points.
func (s *ExtractiveHierarchicalSummarizer) MapCommunity(
	query string,
	summary CommunitySummary,
	corpusDF map[string]int,
	totalSummaries int,
	avgLen float64,
) PartialAnswer {
	docText := summary.Title + "\n" + summary.Summary + "\n" + strings.Join(summary.Findings, "\n")
	qTokens := Tokenize(query)
	dTokens := Tokenize(docText)
	dTF := TermFreq(dTokens)

	bm25 := BM25Score(qTokens, dTF, len(dTokens), avgLen, corpusDF, totalSummaries)
	cos := CosineSimilarity(EmbedText(query), EmbedText(docText))

	// Check explicit theme overlap between query and community themes.
	qLower := strings.ToLower(query)
	themeBoost := 0.0
	rankedThemes := rankThemeCounts(summary.ThemeCounts)
	for _, th := range rankedThemes {
		parts := strings.Split(th, "_")
		for _, p := range parts {
			if len(p) >= 3 && strings.Contains(qLower, p) {
				themeBoost += 0.35
				break
			}
		}
	}

	// Corpus-wide thematic questions benefit from broad community coverage even
	// when phrasing uses high-level terms ("regressions", "risks", "bottlenecks").
	isBroadGlobal := strings.Contains(qLower, "across") ||
		strings.Contains(qLower, "largest") ||
		strings.Contains(qLower, "overall") ||
		strings.Contains(qLower, "all ") ||
		strings.Contains(qLower, "summarize") ||
		strings.Contains(qLower, "main ") ||
		strings.Contains(qLower, "systemic")

	score := 0.50*math.Tanh(bm25/4.0) + 0.35*cos + math.Min(themeBoost, 0.70)
	if isBroadGlobal && len(rankedThemes) > 0 {
		// Reward communities with rich thematic mass on broad corpus-wide questions.
		totalThemeMass := 0
		for _, cnt := range summary.ThemeCounts {
			totalThemeMass += cnt
		}
		score += 0.15 * math.Tanh(float64(totalThemeMass)/8.0)
	}

	promptText := fmt.Sprintf("Query: %s\nCommunity [%s L%d]: %s", query, summary.CommunityID, summary.Level, docText)
	promptTok := EstimateTokens(promptText)

	var keyPoints []string
	if score >= 0.08 {
		keyPoints = append(keyPoints, fmt.Sprintf("%s: %s", summary.Title, summary.Summary))
		for _, f := range summary.Findings {
			if len(keyPoints) >= 4 {
				break
			}
			keyPoints = append(keyPoints, f)
		}
	}
	completionTok := EstimateTokens(strings.Join(keyPoints, "\n"))

	return PartialAnswer{
		CommunityID:      summary.CommunityID,
		Level:            summary.Level,
		RelevanceScore:   score,
		KeyPoints:        keyPoints,
		Themes:           rankedThemes,
		SourceRefs:       sortSourceRefs(summary.SourceRefs),
		PromptTokens:     promptTok,
		CompletionTokens: completionTok,
	}
}

// ReducePartials executes the GraphRAG Reduce phase: merges ranked partial
// answers within tokenBudget, aggregates corpus-wide themes and citations, and
// emits a grounded global synthesis answer.
func (s *ExtractiveHierarchicalSummarizer) ReducePartials(
	query string,
	partials []PartialAnswer,
	tokenBudget int,
) (string, []string, []SourceRef, int, int) {
	if tokenBudget <= 0 {
		tokenBudget = 4000
	}
	ranked := make([]PartialAnswer, 0, len(partials))
	for _, p := range partials {
		if len(p.KeyPoints) > 0 && p.RelevanceScore > 0 {
			ranked = append(ranked, p)
		}
	}
	sort.Slice(ranked, func(i, j int) bool {
		if ranked[i].RelevanceScore != ranked[j].RelevanceScore {
			return ranked[i].RelevanceScore > ranked[j].RelevanceScore
		}
		return ranked[i].CommunityID < ranked[j].CommunityID
	})

	if len(ranked) == 0 {
		ans := "No authorized community summaries matched the query."
		return ans, nil, nil, EstimateTokens(query), EstimateTokens(ans)
	}

	var reduceInput strings.Builder
	reduceInput.WriteString("Synthesize global answer for: ")
	reduceInput.WriteString(query)
	reduceInput.WriteByte('\n')

	themeFreq := make(map[string]int)
	var allRefs []SourceRef
	var sections []string
	usedTokens := EstimateTokens(query) + 32

	for _, p := range ranked {
		block := fmt.Sprintf("- [%s score=%.2f] %s", p.CommunityID, p.RelevanceScore, strings.Join(p.KeyPoints, " | "))
		blkTok := EstimateTokens(block)
		if usedTokens+blkTok > tokenBudget && len(sections) > 0 {
			break
		}
		usedTokens += blkTok
		reduceInput.WriteString(block)
		reduceInput.WriteByte('\n')
		sections = append(sections, block)
		for idx, th := range p.Themes {
			weight := 1
			if idx == 0 {
				weight = 2
			}
			themeFreq[th] += weight
		}
		allRefs = append(allRefs, p.SourceRefs...)
	}

	rankedThemes := rankThemeCounts(themeFreq)
	sortedRefs := sortSourceRefs(allRefs)

	var out strings.Builder
	out.WriteString("Global synthesis across ")
	out.WriteString(fmt.Sprintf("%d communities (%d canonical sources): ", len(sections), len(sortedRefs)))
	if len(rankedThemes) > 0 {
		out.WriteString("Dominant themes: ")
		out.WriteString(strings.Join(rankedThemes, ", "))
		out.WriteString(". ")
	}
	out.WriteString("Evidence:\n")
	out.WriteString(strings.Join(sections, "\n"))

	answer := out.String()
	promptTok := EstimateTokens(reduceInput.String())
	completionTok := EstimateTokens(answer)
	return answer, rankedThemes, sortedRefs, promptTok, completionTok
}

// SynthesizeLocal generates a grounded answer from top-K local hybrid chunks
// using the same extractive style and token accounting as ReducePartials so
// GraphRAG and Ordinary Hybrid RAG comparisons are strictly controlled.
func (s *ExtractiveHierarchicalSummarizer) SynthesizeLocal(
	query string,
	hits []SourceHit,
	tokenBudget int,
) (string, []string, []SourceRef, int, int) {
	if tokenBudget <= 0 {
		tokenBudget = 4000
	}
	if len(hits) == 0 {
		ans := "No authorized canonical documents matched the query."
		return ans, nil, nil, EstimateTokens(query), EstimateTokens(ans)
	}

	var promptBuf strings.Builder
	promptBuf.WriteString("Answer query from top-K retrieved documents: ")
	promptBuf.WriteString(query)
	promptBuf.WriteByte('\n')

	themeFreq := make(map[string]int)
	var refs []SourceRef
	var lines []string
	usedTokens := EstimateTokens(query) + 24

	for _, h := range hits {
		line := fmt.Sprintf("- [%s@%s score=%.2f] %s: %s", h.SourceRef.SourceID, h.SourceRef.Version, h.Score, h.Title, h.Snippet)
		tok := EstimateTokens(line)
		if usedTokens+tok > tokenBudget && len(lines) > 0 {
			break
		}
		usedTokens += tok
		promptBuf.WriteString(line)
		promptBuf.WriteByte('\n')
		lines = append(lines, line)
		refs = append(refs, h.SourceRef)
		for _, th := range h.Themes {
			themeFreq[th]++
		}
	}

	rankedThemes := rankThemeCounts(themeFreq)
	sortedRefs := sortSourceRefs(refs)

	var out strings.Builder
	out.WriteString(fmt.Sprintf("Local top-%d grounded answer (%d canonical sources): ", len(lines), len(sortedRefs)))
	if len(rankedThemes) > 0 {
		out.WriteString("Observed themes: ")
		out.WriteString(strings.Join(rankedThemes, ", "))
		out.WriteString(". ")
	}
	out.WriteString("Evidence:\n")
	out.WriteString(strings.Join(lines, "\n"))

	answer := out.String()
	return answer, rankedThemes, sortedRefs, EstimateTokens(promptBuf.String()), EstimateTokens(answer)
}

func buildCommunityTitle(level int, themes, packages, entities []string) string {
	prefix := fmt.Sprintf("L%d Community", level)
	if len(themes) > 0 && len(packages) > 0 {
		return fmt.Sprintf("%s: %s (%s)", prefix, themes[0], strings.Join(limitSlice(dedupStringsSorted(packages), 2), ", "))
	}
	if len(themes) > 0 {
		return fmt.Sprintf("%s: %s", prefix, strings.Join(limitSlice(themes, 2), " & "))
	}
	if len(packages) > 0 {
		return fmt.Sprintf("%s: %s", prefix, strings.Join(limitSlice(dedupStringsSorted(packages), 2), ", "))
	}
	if len(entities) > 0 {
		return fmt.Sprintf("%s: %s", prefix, strings.Join(limitSlice(entities, 2), ", "))
	}
	return prefix
}

func rankThemeCounts(counts map[string]int) []string {
	type kv struct {
		k string
		v int
	}
	list := make([]kv, 0, len(counts))
	for k, v := range counts {
		if v > 0 {
			list = append(list, kv{k: k, v: v})
		}
	}
	sort.Slice(list, func(i, j int) bool {
		if list[i].v != list[j].v {
			return list[i].v > list[j].v
		}
		return list[i].k < list[j].k
	})
	out := make([]string, len(list))
	for i, item := range list {
		out[i] = item.k
	}
	return out
}

func rankStringCounts(counts map[string]int, limit int) []string {
	ranked := rankThemeCounts(counts)
	return limitSlice(ranked, limit)
}

func limitSlice(s []string, n int) []string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}
