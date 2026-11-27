package graphrag

import (
	"hash/fnv"
	"math"
	"sort"
	"strings"
	"unicode"
)

const embeddingDim = 128

var stopWords = map[string]struct{}{
	"a": {}, "an": {}, "and": {}, "are": {}, "as": {}, "at": {}, "be": {}, "been": {},
	"by": {}, "can": {}, "did": {}, "do": {}, "does": {}, "for": {}, "from": {}, "had": {},
	"has": {}, "have": {}, "how": {}, "in": {}, "into": {}, "is": {}, "it": {}, "its": {},
	"of": {}, "on": {}, "or": {}, "our": {}, "that": {}, "the": {}, "their": {}, "there": {},
	"these": {}, "they": {}, "this": {}, "to": {}, "was": {}, "we": {}, "were": {}, "what": {},
	"when": {}, "where": {}, "which": {}, "who": {}, "why": {}, "will": {}, "with": {},
	"all": {}, "any": {}, "but": {}, "if": {}, "not": {}, "no": {}, "so": {}, "than": {},
	"then": {}, "up": {}, "out": {}, "about": {}, "over": {}, "under": {}, "more": {},
	"most": {}, "such": {}, "only": {}, "also": {}, "just": {}, "should": {}, "would": {},
	"could": {}, "may": {}, "might": {}, "must": {}, "func": {}, "return": {}, "var": {},
	"const": {}, "type": {}, "struct": {}, "interface": {}, "nil": {}, "err": {}, "true": {},
	"false": {}, "else": {}, "range": {}, "switch": {}, "case": {}, "default": {},
}

// EstimateTokens computes a deterministic token count approximation (~4 chars
// per token plus word/punctuation boundaries) for repeatable cost accounting.
func EstimateTokens(text string) int {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return 0
	}
	words := len(strings.Fields(trimmed))
	charEst := (len(trimmed) + 3) / 4
	if words > charEst {
		return words
	}
	return charEst
}

// Tokenize splits text into lowercase alphanumeric tokens, decomposing
// camelCase and snake_case identifiers while preserving whole compound tokens.
func Tokenize(text string) []string {
	if text == "" {
		return nil
	}
	raw := splitRawWords(text)
	out := make([]string, 0, len(raw)*2)
	for _, w := range raw {
		lower := strings.ToLower(w)
		if len(lower) >= 2 {
			if _, stop := stopWords[lower]; !stop {
				out = append(out, lower)
			}
		}
		// Also emit camelCase/snake_case sub-tokens for better cross-document recall.
		parts := splitIdentifierParts(w)
		if len(parts) > 1 {
			for _, p := range parts {
				lp := strings.ToLower(p)
				if len(lp) < 2 {
					continue
				}
				if _, stop := stopWords[lp]; stop {
					continue
				}
				out = append(out, lp)
			}
		}
	}
	return out
}

func splitRawWords(text string) []string {
	var words []string
	var buf strings.Builder
	flush := func() {
		if buf.Len() > 0 {
			words = append(words, buf.String())
			buf.Reset()
		}
	}
	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' || r == '-' || r == '.' || r == '/' {
			buf.WriteRune(r)
		} else {
			flush()
		}
	}
	flush()
	return words
}

func splitIdentifierParts(ident string) []string {
	var parts []string
	var buf strings.Builder
	runes := []rune(ident)
	flush := func() {
		if buf.Len() > 0 {
			parts = append(parts, buf.String())
			buf.Reset()
		}
	}
	for i, r := range runes {
		if r == '_' || r == '-' || r == '.' || r == '/' {
			flush()
			continue
		}
		if i > 0 && unicode.IsUpper(r) && (unicode.IsLower(runes[i-1]) || (i+1 < len(runes) && unicode.IsLower(runes[i+1]))) {
			flush()
		}
		buf.WriteRune(r)
	}
	flush()
	return parts
}

// EmbedText computes a deterministic 128-dim L2-normalized signed FNV-1a
// unigram + bigram + char-4gram feature hashing embedding.
func EmbedText(text string) [embeddingDim]float64 {
	var vec [embeddingDim]float64
	tokens := Tokenize(text)
	if len(tokens) == 0 {
		return vec
	}
	addFeature := func(feat string, weight float64) {
		h := fnv.New64a()
		_, _ = h.Write([]byte(feat))
		sum := h.Sum64()
		idx := sum % embeddingDim
		sign := 1.0
		if (sum>>8)&1 == 1 {
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
				addFeature("c:"+tok[j:j+4], 0.25)
			}
		}
	}
	var norm float64
	for i := range vec {
		norm += vec[i] * vec[i]
	}
	if norm > 0 {
		inv := 1.0 / math.Sqrt(norm)
		for i := range vec {
			vec[i] *= inv
		}
	}
	return vec
}

// CosineSimilarity computes cosine similarity between two L2-normalized vectors.
func CosineSimilarity(a, b [embeddingDim]float64) float64 {
	var dot float64
	for i := 0; i < embeddingDim; i++ {
		dot += a[i] * b[i]
	}
	if dot < 0 {
		return 0
	}
	return dot
}

// TermFreq computes deterministic term frequencies for a token slice.
func TermFreq(tokens []string) map[string]int {
	tf := make(map[string]int, len(tokens))
	for _, t := range tokens {
		tf[t]++
	}
	return tf
}

// BM25Score computes standard Okapi BM25 score for query tokens against a
// document's term frequency map and corpus document frequencies.
func BM25Score(queryTokens []string, docTF map[string]int, docLen int, avgDocLen float64, df map[string]int, totalDocs int) float64 {
	if len(queryTokens) == 0 || docLen == 0 || totalDocs == 0 {
		return 0
	}
	if avgDocLen <= 0 {
		avgDocLen = 1.0
	}
	const k1 = 1.5
	const b = 0.75

	seen := make(map[string]struct{}, len(queryTokens))
	var score float64
	for _, qt := range queryTokens {
		if _, dup := seen[qt]; dup {
			continue
		}
		seen[qt] = struct{}{}
		f := float64(docTF[qt])
		if f == 0 {
			continue
		}
		n := float64(df[qt])
		idf := math.Log(1.0 + (float64(totalDocs)-n+0.5)/(n+0.5))
		denom := f + k1*(1.0-b+b*(float64(docLen)/avgDocLen))
		score += idf * (f * (k1 + 1.0)) / denom
	}
	return score
}

// NormalizeEntityID canonicalizes an entity identifier deterministically.
func NormalizeEntityID(kind, name string) string {
	k := strings.ToLower(strings.TrimSpace(kind))
	if k == "" {
		k = "concept"
	}
	n := strings.ToLower(strings.TrimSpace(name))
	n = strings.ReplaceAll(n, " ", "_")
	n = strings.Trim(n, ".,:;()[]{}\"'`")
	if n == "" {
		return ""
	}
	return k + ":" + n
}

func sortedKeysInt(m map[string]int) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func sortedKeysFloat(m map[string]float64) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func dedupStringsSorted(in []string) []string {
	if len(in) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(in))
	out := make([]string, 0, len(in))
	for _, s := range in {
		s = strings.TrimSpace(s)
		if s == "" {
			continue
		}
		if _, ok := seen[s]; ok {
			continue
		}
		seen[s] = struct{}{}
		out = append(out, s)
	}
	sort.Strings(out)
	return out
}
