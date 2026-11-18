package planner

import (
	"github.com/hanzoai/graph/algo"
)

// Strategy defines the chosen execution path for the query.
type Strategy int

const (
	VectorFirst Strategy = iota
	GraphFirst
	PureTraversal
	CachedNeighborhood
	PlainRAG // Fallback when graph adds cost without evidence
)

// QueryParams represents the input constraints for the planner.
type QueryParams struct {
	LatencyBudget  float64 // in milliseconds
	TargetRecall   float64 // 0.0 to 1.0
	VectorK        int     // Number of vector candidates
	Hops           int     // Graph traversal depth
}

// StatsProvider defines the interface for retrieving graph statistics.
type StatsProvider interface {
	GetDegree(uid uint64) int
	GetSelectivity(predicate string) float64
	GetEstimateCount(predicate string) int
}

// Plan represents the decision made by the planner.
type Plan struct {
	SelectedStrategy Strategy
	EstimatedCost    float64
	Explanation      string
}
