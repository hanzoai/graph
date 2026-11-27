/*
 * SPDX-License-Identifier: Apache-2.0
 */

package graphplanner

import (
	"hash/fnv"
	"math"
	"strings"
	"unicode"
)

const (
	// EmbeddingDim is the fixed dimension of the deterministic stdlib dense vectors.
	EmbeddingDim = 128
)

var stopWords = map[string]bool{
	"a": true, "an": true, "the": true, "and": true, "or": true, "of": true,
	"to": true, "in": true, "on": true, "at": true, "for": true, "by": true,
	"with": true, "from": true, "as": true, "is": true, "are": true, "was": true,
	"were": true, "be": true, "been": true, "being": true, "that": true, "this": true,
	"these": true, "those": true, "it": true, "its": true, "which": true, "what": true,
	"who": true, "where": true, "when": true, "how": true, "does": true, "do": true,
	"did": true, "can": true, "could": true, "would": true, "should": true, "about": true,
	"into": true, "through": true, "between": true, "over": true, "under": true,
}

// Tokenize normalizes text into lowercase identifier/word tokens preserving
// snake_case/camelCase sub-tokens as well as full compound identifiers.
func Tokenize(text string) []string {
	if text == "" {
		return nil
	}
	var out []string
	var cur []rune
	flush := func() {
		if len(cur) == 0 {
			return
		}
		tok := strings.ToLower(string(cur))
		cur = cur[:0]
		if len(tok) < 2 || stopWords[tok] {
			return
		}
		out = append(out, tok)
		// Also emit underscore sub-parts for code identifiers like "posting_list".
		if strings.ContainsRune(tok, '_') {
			for _, part := range strings.Split(tok, "_") {
				if len(part) >= 2 && !stopWords[part] {
					out = append(out, part)
				}
			}
		}
	}
	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' {
			cur = append(cur, r)
		} else {
			flush()
		}
	}
	flush()
	return out
}

// ComputeEmbedding produces a deterministic L2-normalized 128-dim signed
// feature-hashing vector over unigrams, bigrams, and character 4-grams.
func ComputeEmbedding(text string) []float32 {
	vec := make([]float32, EmbeddingDim)
	tokens := Tokenize(text)
	if len(tokens) == 0 {
		return vec
	}

	addFeature := func(feat string, weight float32) {
		h := fnv.New64a()
		_, _ = h.Write([]byte(feat))
		sum := h.Sum64()
		idx := int(sum % EmbeddingDim)
		sign := float32(1.0)
		if (sum>>63)&1 == 1 {
			sign = -1.0
		}
		vec[idx] += sign * weight
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

	NormalizeVector(vec)
	return vec
}

// NormalizeVector performs in-place L2 normalization on a float32 slice.
func NormalizeVector(v []float32) {
	var sumSq float64
	for _, x := range v {
		sumSq += float64(x) * float64(x)
	}
	if sumSq <= 0 || math.IsNaN(sumSq) || math.IsInf(sumSq, 0) {
		return
	}
	inv := float32(1.0 / math.Sqrt(sumSq))
	for i := range v {
		v[i] *= inv
	}
}

// CosineSimilarity computes cosine similarity between two equal-length vectors.
func CosineSimilarity(a, b []float32) float64 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		fa := float64(a[i])
		fb := float64(b[i])
		dot += fa * fb
		na += fa * fa
		nb += fb * fb
	}
	if na == 0 || nb == 0 {
		return 0
	}
	sim := dot / math.Sqrt(na*nb)
	if math.IsNaN(sim) || math.IsInf(sim, 0) {
		return 0
	}
	return sim
}

// EstimateTokens approximates token count for retrieved context cost reporting.
func EstimateTokens(text string) int {
	if text == "" {
		return 0
	}
	words := len(strings.Fields(text))
	chars := (len(text) + 3) / 4
	if words > chars {
		return words
	}
	return chars
}
