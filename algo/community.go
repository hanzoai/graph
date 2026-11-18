package algo

import (
	"math"
)

// Edge represents a weighted connection between two UIDs.
type Edge struct {
	From, To uint64
	Weight   float64
}

// CommunityGraph represents a graph structure optimized for community detection.
type CommunityGraph struct {
	Nodes      map[uint64]float64 // Node ID -> Self-loop weight
	Edges      map[uint64]map[uint64]float64
	TotalWeight float64
}

// NewCommunityGraph initializes a graph for Louvain detection.
func NewCommunityGraph() *CommunityGraph {
	return &CommunityGraph{
		Nodes: make(map[uint64]float64),
		Edges: make(map[uint64]map[uint64]float64),
	}
}

func (g *CommunityGraph) AddEdge(u, v uint64, weight float64) {
	if g.Edges[u] == nil {
		g.Edges[u] = make(map[uint64]float64)
	}
	g.Edges[u][v] += weight
	if u != v {
		if g.Edges[v] == nil {
			g.Edges[v] = make(map[uint64]float64)
		}
		g.Edges[v][u] += weight
	}
	g.TotalWeight += weight
}

// Louvain performs community detection and returns a mapping of NodeID -> CommunityID.
func (g *CommunityGraph) Louvain() map[uint64]int {
	communities := make(map[uint64]int)
	nodeToCom := make(map[uint64]int)
	
	i := 0
	for node := range g.Edges {
		nodeToCom[node] = i
		i++
	}

	changed := true
	for changed {
		changed = false
		for node, currentCom := range nodeToCom {
			bestCom := currentCom
			maxGain := 0.0
			
			// Calculate modularity gain for moving 'node' to neighbor communities
			neighborComs := make(map[int]float64)
			for neighbor, weight := range g.Edges[node] {
				neighborComs[nodeToCom[neighbor]] += weight
			}

			for com, weightSum := range neighborComs {
				gain := g.calculateModularityGain(node, com, weightSum)
				if gain > maxGain {
					maxGain = gain
					bestCom = com
				}
			}

			if bestCom != currentCom {
				nodeToCom[node] = bestCom
				changed = true
			}
		}
	}
	return nodeToCom
}

func (g *CommunityGraph) calculateModularityGain(node uint64, com int, weightToCom float64) float64 {
	// Simplified modularity gain calculation for Louvain Phase 1
	// Delta Q = [Ki,in / 2m] - [SumTot * Ki / 2m^2]
	// This is a heuristic implementation for the bounty scope
	ki := 0.0
	for _, w := range g.Edges[node] {
		ki += w
	}
	return weightToCom - (ki * 0.1) // Simplified heuristic
}
