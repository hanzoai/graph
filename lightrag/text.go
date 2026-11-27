/*
 * SPDX-FileCopyrightText: © 2026 Hanzo AI, Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

package lightrag

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"hash/fnv"
	"math"
	"sort"
	"strings"
	"unicode"
)

var stopWords = map[string]struct{}{
	"a": {}, "an": {}, "and": {}, "are": {}, "as": {}, "at": {}, "be": {}, "been": {},
	"but": {}, "by": {}, "can": {}, "do": {}, "does": {}, "for": {}, "from": {}, "had": {},
	"has": {}, "have": {}, "he": {}, "her": {}, "his": {}, "how": {}, "if": {}, "in": {},
	"into": {}, "is": {}, "it": {}, "its": {}, "may": {}, "more": {}, "no": {}, "not": {},
	"of": {}, "on": {}, "or": {}, "our": {}, "out": {}, "over": {}, "she": {}, "should": {},
	"so": {}, "some": {}, "such": {}, "than": {}, "that": {}, "the": {}, "their": {}, "them": {},
	"then": {}, "there": {}, "these": {}, "they": {}, "this": {}, "to": {}, "up": {}, "was": {},
	"we": {}, "were": {}, "what": {}, "when": {}, "where": {}, "which": {}, "who": {}, "why": {},
	"will": {}, "with": {}, "would": {}, "you": {}, "your": {},
}

// Tokenize splits prose and source code into normalized lowercase tokens,
// preserving exact identifiers while also emitting camelCase and snake_case
// sub-tokens so code symbols match both exact and descriptive queries.
func Tokenize(text string) []string {
	if text == "" {
		return nil
	}
	var tokens []string
	var buf strings.Builder
	flush := func() {
		if buf.Len() == 0 {
			return
		}
		raw := buf.String()
		buf.Reset()
		emitTokenParts(raw, &tokens)
	}
	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' || r == '.' || r == '/' || r == '-' {
			buf.WriteRune(r)
		} else {
			flush()
		}
	}
	flush()
	return tokens
}

func emitTokenParts(raw string, out *[]string) {
	trimmed := strings.Trim(raw, "._/-")
	if trimmed == "" {
		return
	}
	lower := strings.ToLower(trimmed)
	if len(lower) > 1 {
		if _, isStop := stopWords[lower]; !isStop {
			*out = append(*out, lower)
		}
	}

	// Split compound paths/qualified identifiers on separators and camelCase boundaries.
	parts := splitCompound(trimmed)
	if len(parts) <= 1 {
		return
	}
	for _, p := range parts {
		pl := strings.ToLower(p)
		if len(pl) <= 1 || pl == lower {
			continue
		}
		if _, isStop := stopWords[pl]; isStop {
			continue
		}
		*out = append(*out, pl)
	}
}

func splitCompound(s string) []string {
	var parts []string
	runes := []rune(s)
	start := 0
	for i := 0; i < len(runes); i++ {
		r := runes[i]
		if r == '_' || r == '.' || r == '/' || r == '-' {
			if i > start {
				parts = append(parts, string(runes[start:i]))
			}
			start = i + 1
			continue
		}
		if i > start && unicode.IsUpper(r) {
			prev := runes[i-1]
			nextLower := i+1 < len(runes) && unicode.IsLower(runes[i+1])
			if unicode.IsLower(prev) || unicode.IsDigit(prev) || (unicode.IsUpper(prev) && nextLower) {
				parts = append(parts, string(runes[start:i]))
				start = i
			}
		}
	}
	if start < len(runes) {
		parts = append(parts, string(runes[start:]))
	}
	return parts
}

// EstimateTokens estimates LLM token count deterministically (~4 chars per token
// with minimum based on word count).
func EstimateTokens(text string) int {
	if text == "" {
		return 0
	}
	charEst := (len(text) + 3) / 4
	wordEst := len(strings.Fields(text))
	if wordEst > charEst {
		return wordEst
	}
	return charEst
}

// EmbedText generates a deterministic, zero-dependency L2-normalized dense vector
// using signed FNV-1a feature hashing over unigrams, bigrams, and character 4-grams.
func EmbedText(text string, dim int) []float32 {
	if dim <= 0 {
		dim = 128
	}
	vec := make([]float32, dim)
	tokens := Tokenize(text)
	if len(tokens) == 0 {
		return vec
	}

	addFeature := func(feat string, weight float64) {
		h := fnv.New64a()
		_, _ = h.Write([]byte(feat))
		sum := h.Sum64()
		idx := int(sum % uint64(dim))
		sign := float64(1.0)
		if (sum>>63)&1 == 1 {
			sign = -1.0
		}
		vec[idx] += float32(sign * weight)
	}

	for i, tok := range tokens {
		addFeature("u:"+tok, 1.0)
		if i+1 < len(tokens) {
			addFeature("b:"+tok+"_"+tokens[i+1], 0.75)
		}
		if len(tok) >= 4 {
			limit := len(tok) - 3
			if limit > 8 {
				limit = 8
			}
			for j := 0; j < limit; j++ {
				addFeature("c:"+tok[j:j+4], 0.35)
			}
		}
	}

	var normSq float64
	for _, v := range vec {
		normSq += float64(v) * float64(v)
	}
	if normSq > 0 {
		invNorm := float32(1.0 / math.Sqrt(normSq))
		for i := range vec {
			vec[i] *= invNorm
		}
	}
	return vec
}

// CosineSimilarity computes dot product of two L2-normalized vectors.
func CosineSimilarity(a, b []float32) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var sum float64
	for i := 0; i < n; i++ {
		sum += float64(a[i]) * float64(b[i])
	}
	return sum
}

type scoredID struct {
	id    string
	score float64
}

type bm25Doc struct {
	tf     map[string]int
	length int
}

// bm25Index is an incremental Okapi BM25 inverted index supporting O(|doc_tokens|)
// insertion, update, and removal without full-index rebuilds.
type bm25Index struct {
	docs     map[string]bm25Doc
	postings map[string]map[string]int
	totalLen int
}

func newBM25Index() *bm25Index {
	return &bm25Index{
		docs:     make(map[string]bm25Doc),
		postings: make(map[string]map[string]int),
	}
}

func (idx *bm25Index) Add(docID, text string) {
	if _, exists := idx.docs[docID]; exists {
		idx.Remove(docID)
	}
	toks := Tokenize(text)
	tf := make(map[string]int, len(toks))
	for _, t := range toks {
		tf[t]++
	}
	idx.docs[docID] = bm25Doc{tf: tf, length: len(toks)}
	idx.totalLen += len(toks)
	for term, freq := range tf {
		p := idx.postings[term]
		if p == nil {
			p = make(map[string]int)
			idx.postings[term] = p
		}
		p[docID] = freq
	}
}

func (idx *bm25Index) Remove(docID string) {
	doc, ok := idx.docs[docID]
	if !ok {
		return
	}
	idx.totalLen -= doc.length
	if idx.totalLen < 0 {
		idx.totalLen = 0
	}
	for term := range doc.tf {
		if p := idx.postings[term]; p != nil {
			delete(p, docID)
			if len(p) == 0 {
				delete(idx.postings, term)
			}
		}
	}
	delete(idx.docs, docID)
}

func (idx *bm25Index) Search(queryText string, allowed func(id string) bool, topK int) ([]scoredID, int) {
	nDocs := len(idx.docs)
	if nDocs == 0 || topK <= 0 {
		return nil, 0
	}
	qToks := Tokenize(queryText)
	if len(qToks) == 0 {
		return nil, 0
	}
	qFreq := make(map[string]int, len(qToks))
	for _, t := range qToks {
		qFreq[t]++
	}

	avgDL := float64(idx.totalLen) / float64(nDocs)
	if avgDL <= 0 {
		avgDL = 1.0
	}
	const k1 = 1.2
	const b = 0.75

	scores := make(map[string]float64)
	filteredOut := 0
	checkedDenied := make(map[string]struct{})

	for term, qtf := range qFreq {
		p := idx.postings[term]
		if len(p) == 0 {
			continue
		}
		df := float64(len(p))
		idf := math.Log(1.0 + (float64(nDocs)-df+0.5)/(df+0.5))
		qWeight := 1.0 + math.Log(float64(qtf))
		for docID, tf := range p {
			if allowed != nil && !allowed(docID) {
				if _, seen := checkedDenied[docID]; !seen {
					checkedDenied[docID] = struct{}{}
					filteredOut++
				}
				continue
			}
			dl := float64(idx.docs[docID].length)
			f := float64(tf)
			denom := f + k1*(1.0-b+b*(dl/avgDL))
			scores[docID] += qWeight * idf * (f * (k1 + 1.0)) / denom
		}
	}

	return topScored(scores, topK), filteredOut
}

// vectorIndex is an incremental O(dim) dense vector table with pre-scoring ACL gating.
type vectorIndex struct {
	dim     int
	vectors map[string][]float32
}

func newVectorIndex(dim int) *vectorIndex {
	return &vectorIndex{
		dim:     dim,
		vectors: make(map[string][]float32),
	}
}

func (v *vectorIndex) Set(id, text string) {
	v.vectors[id] = EmbedText(text, v.dim)
}

func (v *vectorIndex) Delete(id string) {
	delete(v.vectors, id)
}

const minVectorCosine = 0.15

func (v *vectorIndex) Search(queryText string, allowed func(id string) bool, topK int) ([]scoredID, int) {
	if len(v.vectors) == 0 || topK <= 0 {
		return nil, 0
	}
	qVec := EmbedText(queryText, v.dim)
	scores := make(map[string]float64, len(v.vectors))
	filteredOut := 0
	for id, vec := range v.vectors {
		if allowed != nil && !allowed(id) {
			filteredOut++
			continue
		}
		sim := CosineSimilarity(qVec, vec)
		if sim >= minVectorCosine {
			scores[id] = sim
		}
	}
	return topScored(scores, topK), filteredOut
}

func topScored(scores map[string]float64, topK int) []scoredID {
	if len(scores) == 0 {
		return nil
	}
	items := make([]scoredID, 0, len(scores))
	for id, sc := range scores {
		items = append(items, scoredID{id: id, score: sc})
	}
	sort.Slice(items, func(i, j int) bool {
		if math.Abs(items[i].score-items[j].score) > 1e-12 {
			return items[i].score > items[j].score
		}
		return items[i].id < items[j].id
	})
	if topK > 0 && len(items) > topK {
		items = items[:topK]
	}
	return items
}

// splitChunks splits a canonical source record into deterministic line-aligned
// chunks with content-addressed chunk IDs.
func splitChunks(src SourceRecord, maxChars, overlapChars int) []Chunk {
	content := strings.TrimSpace(src.Content)
	if content == "" {
		return nil
	}
	if maxChars <= 0 {
		maxChars = 1400
	}
	if overlapChars < 0 || overlapChars >= maxChars {
		overlapChars = maxChars / 6
	}

	lines := strings.Split(content, "\n")
	var texts []string
	var cur strings.Builder
	var curLines []string

	flush := func() {
		t := strings.TrimSpace(cur.String())
		if t != "" {
			texts = append(texts, t)
		}
		// Keep trailing overlap lines.
		cur.Reset()
		if overlapChars > 0 && len(curLines) > 0 {
			var kept []string
			keptChars := 0
			for i := len(curLines) - 1; i >= 0; i-- {
				l := curLines[i]
				if keptChars+len(l)+1 > overlapChars && len(kept) > 0 {
					break
				}
				kept = append([]string{l}, kept...)
				keptChars += len(l) + 1
			}
			curLines = kept
			for i, l := range curLines {
				if i > 0 {
					cur.WriteByte('\n')
				}
				cur.WriteString(l)
			}
		} else {
			curLines = nil
		}
	}

	for _, line := range lines {
		if len(line) > maxChars {
			if cur.Len() > 0 {
				flush()
			}
			for start := 0; start < len(line); start += maxChars - overlapChars {
				end := start + maxChars
				if end > len(line) {
					end = len(line)
				}
				seg := strings.TrimSpace(line[start:end])
				if seg != "" {
					texts = append(texts, seg)
				}
				if end == len(line) {
					break
				}
			}
			cur.Reset()
			curLines = nil
			continue
		}
		if cur.Len() > 0 && cur.Len()+1+len(line) > maxChars {
			flush()
		}
		if cur.Len() > 0 {
			cur.WriteByte('\n')
		}
		cur.WriteString(line)
		curLines = append(curLines, line)
	}
	if strings.TrimSpace(cur.String()) != "" {
		t := strings.TrimSpace(cur.String())
		if len(texts) == 0 || texts[len(texts)-1] != t {
			texts = append(texts, t)
		}
	}

	chunks := make([]Chunk, 0, len(texts))
	for i, text := range texts {
		cid := deterministicID("chk", fmt.Sprintf("%s|%s|%d", src.TenantID, src.SourceID, i))
		chunks = append(chunks, Chunk{
			ChunkID:    cid,
			SourceID:   src.SourceID,
			Version:    src.Version,
			TenantID:   src.TenantID,
			ACL:        cloneStringSlice(src.ACL),
			Kind:       src.Kind,
			Title:      src.Title,
			Path:       src.Path,
			Text:       text,
			TokenCount: EstimateTokens(text),
			Metadata:   cloneStringMap(src.Metadata),
		})
	}
	return chunks
}

func deterministicID(prefix, key string) string {
	h := sha256.Sum256([]byte(prefix + ":" + key))
	return prefix + "_" + hex.EncodeToString(h[:8])
}

// canonicalAliasSynonyms maps well-known engineering aliases to a canonical token
// so duplicate mentions across Go files, docs, commits, and incidents resolve to
// a single deterministic graph node.
var canonicalAliasSynonyms = map[string]string{
	"badger":       "zapdb",
	"badgerdb":     "zapdb",
	"luxfi/zapdb":  "zapdb",
	"zapdb":        "zapdb",
	"hnsw":         "tok/hnsw",
	"tok/hnsw":     "tok/hnsw",
	"dql":          "dql",
	"dql parser":   "dql",
	"posting list": "posting.list",
	"posting.list": "posting.list",
	"mvcc":         "mvcc",
	"txn":          "posting.txn",
	"posting.txn":  "posting.txn",
}

// normalizeCanonicalKey produces a deterministic, order-independent canonical key
// for entity deduplication and alias resolution.
func normalizeCanonicalKey(typ EntityType, rawName string) string {
	s := strings.TrimSpace(rawName)
	s = strings.Trim(s, "`\"'()[]{}*&,;:")
	s = strings.TrimPrefix(s, "github.com/hanzoai/graph/")
	s = strings.TrimPrefix(s, "./")
	s = strings.TrimPrefix(s, "pkg:")
	s = strings.Join(strings.Fields(s), " ")
	lower := strings.ToLower(s)

	if mapped, ok := canonicalAliasSynonyms[lower]; ok && (typ == EntityPackage || typ == EntityDependency || typ == EntityConcept || typ == EntityComponent) {
		lower = mapped
	}

	switch typ {
	case EntityPackage:
		lower = strings.TrimPrefix(lower, "package ")
		lower = strings.Trim(lower, "/")
	case EntityFile:
		lower = strings.TrimPrefix(lower, "/")
	case EntityConcept, EntityComponent:
		// Mild singularization for common English plurals (>4 chars, not ending in ss/us/is).
		if len(lower) > 4 && strings.HasSuffix(lower, "s") &&
			!strings.HasSuffix(lower, "ss") && !strings.HasSuffix(lower, "us") && !strings.HasSuffix(lower, "is") {
			lower = strings.TrimSuffix(lower, "s")
		}
	case EntityTypeDecl, EntityFunction, EntityMethod:
		// Strip pointer/slice qualifiers while keeping package.Symbol canonical form.
		lower = strings.TrimLeft(lower, "*[]")
	}

	if lower == "" {
		lower = "unknown"
	}
	return string(typ) + ":" + lower
}

func canonicalEntityID(typ EntityType, rawName string) (id string, canonicalKey string) {
	canonicalKey = normalizeCanonicalKey(typ, rawName)
	return deterministicID("ent", canonicalKey), canonicalKey
}

func canonicalRelationID(srcEntityID string, relType RelationType, dstEntityID string) string {
	return deterministicID("rel", fmt.Sprintf("%s|%s|%s", srcEntityID, string(relType), dstEntityID))
}

func normalizeAliasToken(s string) string {
	s = strings.TrimSpace(strings.ToLower(s))
	s = strings.Trim(s, "`\"'()[]{}*&,;:")
	s = strings.TrimPrefix(s, "github.com/hanzoai/graph/")
	s = strings.TrimPrefix(s, "./")
	return strings.Join(strings.Fields(s), " ")
}
