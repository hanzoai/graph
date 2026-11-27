/*
 * SPDX-License-Identifier: Apache-2.0
 */

package hipporag

import (
	"hash/fnv"
	"math"
	"regexp"
	"sort"
	"strings"
	"unicode"
)

// EmbedDim is the dimensionality of the deterministic dense feature embeddings.
const EmbedDim = 128

// DenseVec is a fixed-size L2-normalized embedding vector.
type DenseVec [EmbedDim]float64

var stopWords = map[string]struct{}{
	"a": {}, "an": {}, "the": {}, "and": {}, "or": {}, "but": {}, "if": {},
	"in": {}, "on": {}, "at": {}, "to": {}, "for": {}, "of": {}, "with": {},
	"by": {}, "from": {}, "as": {}, "is": {}, "was": {}, "are": {}, "were": {},
	"be": {}, "been": {}, "being": {}, "have": {}, "has": {}, "had": {},
	"do": {}, "does": {}, "did": {}, "that": {}, "this": {}, "these": {},
	"those": {}, "it": {}, "its": {}, "they": {}, "them": {}, "their": {},
	"which": {}, "who": {}, "whom": {}, "what": {}, "where": {}, "when": {},
	"why": {}, "how": {}, "into": {}, "over": {}, "under": {}, "about": {},
	"between": {}, "through": {}, "during": {}, "before": {}, "after": {},
	"above": {}, "below": {}, "up": {}, "down": {}, "out": {}, "off": {},
	"than": {}, "then": {}, "so": {}, "such": {}, "can": {}, "could": {},
	"will": {}, "would": {}, "should": {}, "may": {}, "might": {}, "must": {},
	"not": {}, "no": {}, "nor": {}, "only": {}, "own": {}, "same": {},
	"too": {}, "very": {}, "just": {}, "also": {}, "there": {}, "here": {},
}

var (
	reProperPhrase = regexp.MustCompile(`\b[A-Z][a-zA-Z0-9_-]+(?:\s+[A-Z][a-zA-Z0-9_-]+){0,3}\b`)
	reCodeIdent    = regexp.MustCompile(`\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b`)
	rePRRef        = regexp.MustCompile(`\b(?:PR|pr|commit|issue)\s*#?([0-9a-fA-F]{3,12})\b`)
	reMultiSep     = regexp.MustCompile(`[\s\-/]+`)
)

// NormalizeEntity produces a canonical entity key suitable for graph node IDs.
func NormalizeEntity(raw string) string {
	s := strings.TrimSpace(strings.ToLower(raw))
	s = strings.Trim(s, ".,;:!?\"'()[]{}`")
	if s == "" {
		return ""
	}
	s = reMultiSep.ReplaceAllString(s, "_")
	var b strings.Builder
	b.Grow(len(s))
	prevUnderscore := false
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
			prevUnderscore = false
		} else if r == '_' {
			if !prevUnderscore && b.Len() > 0 {
				b.WriteRune('_')
				prevUnderscore = true
			}
		}
	}
	out := strings.Trim(b.String(), "_")
	return out
}

// Tokenize splits text into normalized content tokens (filtering common stopwords)
// while keeping compound technical identifiers and their constituent sub-tokens.
func Tokenize(text string) []string {
	if text == "" {
		return nil
	}
	rawFields := strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '_' && r != '-'
	})
	tokens := make([]string, 0, len(rawFields)*2)
	for _, f := range rawFields {
		f = strings.Trim(f, "_-")
		if f == "" {
			continue
		}
		if _, stop := stopWords[f]; !stop && len(f) > 1 {
			tokens = append(tokens, f)
		}
		if strings.ContainsAny(f, "_-") {
			parts := strings.FieldsFunc(f, func(r rune) bool {
				return r == '_' || r == '-'
			})
			for _, p := range parts {
				if len(p) <= 1 {
					continue
				}
				if _, stop := stopWords[p]; stop {
					continue
				}
				if p != f {
					tokens = append(tokens, p)
				}
			}
		}
	}
	return tokens
}

// EstimateTokens computes a deterministic token count for context budget accounting.
func EstimateTokens(text string) int {
	words := len(strings.Fields(text))
	if words == 0 {
		return 1
	}
	return int(math.Ceil(float64(words) * 1.3))
}

// hashFeature maps a feature string deterministically to (bucket, sign).
func hashFeature(feat string) (int, float64) {
	h := fnv.New64a()
	_, _ = h.Write([]byte(feat))
	sum := h.Sum64()
	bucket := int(sum % EmbedDim)
	sign := 1.0
	if (sum>>63)&1 == 1 {
		sign = -1.0
	}
	return bucket, sign
}

// EmbedText computes a deterministic 128-dim L2-normalized dense embedding over
// unigrams, bigrams, and character 4-grams. Zero external model dependencies.
func EmbedText(text string) DenseVec {
	var v DenseVec
	toks := Tokenize(text)
	if len(toks) == 0 {
		return v
	}

	for i, tok := range toks {
		// Unigram feature
		b, s := hashFeature("u:" + tok)
		v[b] += s * 1.0

		// Bigram feature
		if i+1 < len(toks) {
			b2, s2 := hashFeature("b:" + tok + "_" + toks[i+1])
			v[b2] += s2 * 0.85
		}

		// Character 4-gram morphological features for synonymy/fuzzy matching
		if len(tok) >= 4 {
			padded := "^" + tok + "$"
			for j := 0; j+4 <= len(padded); j++ {
				bg, sg := hashFeature("c:" + padded[j:j+4])
				v[bg] += sg * 0.35
			}
		}
	}

	return NormalizeVec(v)
}

// NormalizeVec L2-normalizes a DenseVec.
func NormalizeVec(v DenseVec) DenseVec {
	var normSq float64
	for i := 0; i < EmbedDim; i++ {
		normSq += v[i] * v[i]
	}
	if normSq <= 0 {
		return v
	}
	inv := 1.0 / math.Sqrt(normSq)
	for i := 0; i < EmbedDim; i++ {
		v[i] *= inv
	}
	return v
}

// CosineSim computes the cosine similarity between two L2-normalized vectors.
func CosineSim(a, b DenseVec) float64 {
	var dot float64
	for i := 0; i < EmbedDim; i++ {
		dot += a[i] * b[i]
	}
	return dot
}

// ExtractCandidateEntities extracts normalized entity keys from text (used for
// query seeding and automatic OpenIE triple extraction on plain-text datasets).
func ExtractCandidateEntities(text string) []string {
	seen := make(map[string]struct{})
	add := func(raw string) {
		k := NormalizeEntity(raw)
		if len(k) < 3 {
			return
		}
		if _, stop := stopWords[k]; stop {
			return
		}
		seen[k] = struct{}{}
	}

	for _, m := range reCodeIdent.FindAllString(text, -1) {
		add(m)
	}
	for _, m := range rePRRef.FindAllStringSubmatch(text, -1) {
		if len(m) >= 2 {
			add("pr_" + strings.ToLower(m[1]))
			add(m[0])
		}
	}
	for _, m := range reProperPhrase.FindAllString(text, -1) {
		parts := strings.Fields(m)
		if len(parts) == 1 {
			if _, stop := stopWords[strings.ToLower(parts[0])]; stop {
				continue
			}
		}
		add(m)
	}

	out := make([]string, 0, len(seen))
	for k := range seen {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// ExtractAutoTriples builds deterministic relational triples from a SourceRecord's
// Title and Text when explicit structured triples are not provided (for example,
// when loading external HotpotQA or MuSiQue JSON/JSONL corpora).
func ExtractAutoTriples(rec SourceRecord) []ExtractedTriple {
	rel := rec.Reliability
	if rel <= 0 {
		rel = 0.85
	}
	titleKey := NormalizeEntity(rec.Title)
	entities := ExtractCandidateEntities(rec.Text)

	var triples []ExtractedTriple
	if titleKey != "" {
		for _, ent := range entities {
			if ent == titleKey {
				continue
			}
			triples = append(triples, ExtractedTriple{
				Subject:     titleKey,
				Predicate:   "references",
				Object:      ent,
				Relevance:   0.80,
				Reliability: rel,
			})
		}
	}

	// Sentence-local entity co-occurrence triples for multi-hop bridge linkage.
	sentences := strings.FieldsFunc(rec.Text, func(r rune) bool {
		return r == '.' || r == '\n' || r == ';'
	})
	for _, sent := range sentences {
		sentEnts := ExtractCandidateEntities(sent)
		for i := 0; i < len(sentEnts) && i < 6; i++ {
			for j := i + 1; j < len(sentEnts) && j < 6; j++ {
				if sentEnts[i] == sentEnts[j] {
					continue
				}
				triples = append(triples, ExtractedTriple{
					Subject:     sentEnts[i],
					Predicate:   "co_occurs_with",
					Object:      sentEnts[j],
					Relevance:   0.72,
					Reliability: rel,
				})
			}
		}
	}
	return triples
}

// bm25Score computes standard Okapi BM25 term contribution.
func bm25Score(tf int, docLen int, avgDocLen float64, df int, totalDocs int) float64 {
	if tf <= 0 || totalDocs <= 0 || avgDocLen <= 0 {
		return 0
	}
	const k1 = 1.5
	const b = 0.75
	idf := math.Log(1.0 + (float64(totalDocs-df)+0.5)/(float64(df)+0.5))
	num := float64(tf) * (k1 + 1.0)
	den := float64(tf) + k1*(1.0-b+b*(float64(docLen)/avgDocLen))
	return idf * (num / den)
}
