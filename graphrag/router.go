package graphrag

import (
	"fmt"
	"regexp"
	"strings"
)

// QueryRouter implements an explicit, auditable routing gate deciding whether
// a query requires GraphRAG hierarchical map-reduce (corpus-wide synthesis) or
// should bypass GraphRAG in favor of fast ordinary hybrid retrieval (exact
// symbol, file, or record lookups).
type QueryRouter struct {
	DefaultGlobalLevel int
}

// NewQueryRouter returns a QueryRouter with default global hierarchy level L1.
func NewQueryRouter(defaultGlobalLevel int) *QueryRouter {
	if defaultGlobalLevel < 0 || defaultGlobalLevel > 2 {
		defaultGlobalLevel = 1
	}
	return &QueryRouter{DefaultGlobalLevel: defaultGlobalLevel}
}

var (
	reExactFilePath = regexp.MustCompile(`\b[a-zA-Z0-9_/-]+\.(go|md|yml|yaml|json|txt)\b`)
	reExactRecordID = regexp.MustCompile(`\b(INC|REG|ADR|BUG|PR|HIP)-[0-9]{1,6}\b`)
	reExactSymbol   = regexp.MustCompile(`\b([a-z0-9_]+\.[A-Z][a-zA-Z0-9_]+|[A-Z][a-z0-9]+[A-Z][a-zA-Z0-9]*)\b`)
)

var globalPhraseCues = []struct {
	phrase string
	weight float64
}{
	{"across our repositories", 2.5},
	{"across all packages", 2.5},
	{"across the codebase", 2.4},
	{"across repositories", 2.4},
	{"corpus-wide", 2.5},
	{"largest sources", 2.2},
	{"biggest sources", 2.0},
	{"main sources", 1.8},
	{"top themes", 2.0},
	{"major themes", 2.0},
	{"recurring", 1.6},
	{"systemic", 1.8},
	{"overall", 1.5},
	{"broadly", 1.4},
	{"summarize", 1.5},
	{"highest concentration", 1.8},
	{"dominant patterns", 2.0},
	{"cross-package", 2.0},
	{"cross-cutting", 2.0},
	{"compare patterns", 1.8},
	{"all subsystems", 2.0},
	{"portfolio", 1.5},
	{"across", 1.2},
	{"most common", 1.6},
	{"what are the main", 1.8},
	{"what are the top", 1.8},
	{"engineering regressions", 1.2},
}

var localPhraseCues = []struct {
	phrase string
	weight float64
}{
	{"where is", 2.2},
	{"defined in", 2.2},
	{"which file", 2.0},
	{"exact", 2.0},
	{"signature of", 2.2},
	{"line number", 2.2},
	{"what does", 1.5},
	{"specific", 1.4},
	{"single", 1.2},
	{"return type", 2.0},
	{"implemented in", 1.8},
	{"lookup", 2.0},
	{"find the function", 2.2},
	{"find the struct", 2.2},
	{"root cause of inc-", 2.4},
	{"root cause in inc-", 2.4},
	{"details of reg-", 2.4},
}

// Route classifies a query as RouteGlobalGraphRAG vs RouteLocalHybrid and
// returns transparent cue scores and rationale.
func (r *QueryRouter) Route(query string) RoutingDecision {
	lower := strings.ToLower(strings.TrimSpace(query))

	var globalScore, localScore float64
	var globalCues, localCues []string

	for _, cue := range globalPhraseCues {
		if strings.Contains(lower, cue.phrase) {
			globalScore += cue.weight
			globalCues = append(globalCues, cue.phrase)
		}
	}
	for _, cue := range localPhraseCues {
		if strings.Contains(lower, cue.phrase) {
			localScore += cue.weight
			localCues = append(localCues, cue.phrase)
		}
	}

	if m := reExactFilePath.FindString(query); m != "" {
		localScore += 3.0
		localCues = append(localCues, "file:"+m)
	}
	if m := reExactRecordID.FindString(query); m != "" {
		localScore += 3.2
		localCues = append(localCues, "id:"+m)
	}
	if m := reExactSymbol.FindString(query); m != "" {
		// Ignore generic GraphRAG / DQL / MVCC / HNSW / ACL acronyms unless paired with exact lookup cue.
		if !isCommonAcronym(m) {
			localScore += 2.2
			localCues = append(localCues, "symbol:"+m)
		}
	}

	// Default posture per hanzoai/graph#3: GraphRAG is NOT the default path.
	// Only route to Global GraphRAG when global synthesis cues strictly outweigh
	// local/exact cues AND globalScore exceeds the activation threshold.
	if globalScore >= 2.0 && globalScore > localScore+0.5 {
		level := r.DefaultGlobalLevel
		if strings.Contains(lower, "executive") || strings.Contains(lower, "macro") || strings.Contains(lower, "portfolio") {
			level = 2
		}
		mode := modeForLevel(level)
		return RoutingDecision{
			Target:            RouteGlobalGraphRAG,
			RecommendedMode:   mode,
			RecommendedLevel:  level,
			GlobalScore:       globalScore,
			LocalScore:        localScore,
			Reason:            fmt.Sprintf("corpus-wide synthesis detected (global=%.2f > local=%.2f); dispatching to GraphRAG L%d map-reduce", globalScore, localScore, level),
			MatchedGlobalCues: dedupStringsSorted(globalCues),
			MatchedLocalCues:  dedupStringsSorted(localCues),
		}
	}

	return RoutingDecision{
		Target:            RouteLocalHybrid,
		RecommendedMode:   ModeLocalHybrid,
		RecommendedLevel:  -1,
		GlobalScore:       globalScore,
		LocalScore:        localScore,
		Reason:            fmt.Sprintf("exact/local lookup or insufficient global signal (local=%.2f, global=%.2f); bypassing GraphRAG via local hybrid baseline", localScore, globalScore),
		MatchedGlobalCues: dedupStringsSorted(globalCues),
		MatchedLocalCues:  dedupStringsSorted(localCues),
	}
}

func modeForLevel(level int) QueryMode {
	switch level {
	case 0:
		return ModeGlobalL0
	case 2:
		return ModeGlobalL2
	default:
		return ModeGlobalL1
	}
}

func isCommonAcronym(s string) bool {
	switch strings.ToUpper(s) {
	case "GRAPHRAG", "LIGHTRAG", "HIPPORAG", "MVCC", "HNSW", "DQL", "ACL", "TLS", "FIPS", "AST", "JSON", "HTTP", "GRPC":
		return true
	default:
		return false
	}
}
