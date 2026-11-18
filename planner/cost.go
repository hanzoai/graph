package planner

import (
	"math"
)

const (
	VectorSearchLatencyBase = 5.0  // ms
	GraphHopLatencyBase     = 1.5  // ms per hop per node
	DiskFetchLatency        = 0.5  // ms
)

// CostModel calculates the estimated latency for a given strategy.
type CostModel struct {
	Stats StatsProvider
}

func (c *CostModel) EstimateVectorFirst(params QueryParams) float64 {
	// Cost = Vector Search + Fetching Metadata for K nodes + Small Traversal
	return VectorSearchLatencyBase + (float64(params.VectorK) * 0.1)
}

func (c *CostModel) EstimateGraphFirst(params QueryParams, startNodes int) float64 {
	// Cost = Start Nodes * Avg Degree ^ Hops * Hop Latency
	avgDegree := float64(c.Stats.GetDegree(0))
	nodesToVisit := float64(startNodes) * math.Pow(avgDegree, float64(params.Hops))
	return nodesToVisit * GraphHopLatencyBase
}

func (c *CostModel) EstimateTraversal(params QueryParams) float64 {
	return c.EstimateGraphFirst(params, 1) // Assuming single entry point
}
