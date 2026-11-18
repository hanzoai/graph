package algo

import (
	"math"
)

// WeightedGraph defines the interface required for PPR traversal.
// It allows the algorithm to work with the underlying posting list engine
// without being tightly coupled to the storage layer.
type WeightedGraph interface {
	// Neighbors returns a list of target UIDs and their edge weights for a given source UID.
	Neighbors(uid uint64) ([]uint64, []float64)
}

// PPRParams defines the configuration for the Bounded Personalized PageRank.
type PPRParams struct {
	DampingFactor float64 // Usually 0.85 (alpha)
	MaxIterations int     // Bounds the computation
	Epsilon       float64 // Convergence threshold
}

// DefaultPPRParams returns standard values for PPR.
func DefaultPPRParams() PPRParams {
	return PPRParams{
		DampingFactor: 0.85,
		MaxIterations: 20,
		Epsilon:       1e-6,
	}
}

// ComputePPR performs a Bounded Personalized PageRank starting from seed nodes.
// seeds: A map of UIDs to their initial weights (e.g., entity relevance in query).
// graph: The graph structure to traverse.
// params: Configuration for the algorithm.
func ComputePPR(seeds map[uint64]float64, graph WeightedGraph, params PPRParams) map[uint64]float64 {
	if len(seeds) == 0 {
		return nil
	}

	// Normalize seed weights to form the teleportation vector (p)
	teleport := make(map[uint64]float64)
	var sumSeeds float64
	for _, w := range seeds {
		sumSeeds += w
	}
	for uid, w := range seeds {
		teleport[uid] = w / sumSeeds
	}

	// Initialize score vector (v) with teleportation vector
	scores := make(map[uint64]float64)
	for uid, w := range teleport {
		scores[uid] = w
	}

	for i := 0; i < params.MaxIterations; i++ {
		nextScores := make(map[uint64]float64)
		var danglingWeight float64

		// Distribute current scores
		for uid, score := range scores {
			neighbors, weights := graph.Neighbors(uid)
			if len(neighbors) == 0 {
				danglingWeight += score
				continue
			}

			var sumWeights float64
			for _, w := range weights {
				sumWeights += w
			}

			for j, v := range neighbors {
				edgeWeight := weights[j] / sumWeights
				nextScores[v] += params.DampingFactor * score * edgeWeight
			}
		}

		// Apply teleportation and handle dangling nodes
		totalNextScore := 0.0
		for uid, p := range teleport {
			// Each node gets a share of the teleportation and the redistributed dangling weight
			nextScores[uid] += (1.0 - params.DampingFactor) * p
			nextScores[uid] += params.DampingFactor * danglingWeight * p
		}

		// Check convergence
		maxDiff := 0.0
		// We need to check all nodes present in either scores or nextScores
		allNodes := make(map[uint64]struct{})
		for k := range scores {
			allNodes[k] = struct{}{}
		}
		for k := range nextScores {
			allNodes[k] = struct{}{}
		}

		for uid := range allNodes {
			diff := math.Abs(nextScores[uid] - scores[uid])
			if diff > maxDiff {
				maxDiff = diff
			}
			totalNextScore += nextScores[uid]
		}

		scores = nextScores

		if maxDiff < params.Epsilon {
			break
		}
	}

	return scores
}
