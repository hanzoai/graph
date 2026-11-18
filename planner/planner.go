package planner

import (
	"fmt"
)

// Planner is the core engine for HIP-1347 cost-based decisions.
type Planner struct {
	Stats     StatsProvider
	CostModel *CostModel
}

func NewPlanner(stats StatsProvider) *Planner {
	return &Planner{
		Stats:     stats,
		CostModel: &CostModel{Stats: stats},
	}
}

// Plan decides the best retrieval strategy based on constraints and stats.
func (p *Planner) Plan(params QueryParams, predicate string) Plan {
	vCost := p.CostModel.EstimateVectorFirst(params)
	gCost := p.CostModel.EstimateTraversal(params)
	
	selectivity := p.Stats.GetSelectivity(predicate)

	// Rule 1: Latency Budget Enforcement
	if vCost > params.LatencyBudget && gCost > params.LatencyBudget {
		return Plan{
			SelectedStrategy: PlainRAG,
			EstimatedCost:    vCost,
			Explanation:      "Both strategies exceed budget; falling back to minimal RAG.",
		}
	}

	// Rule 2: Factual Retrieval - Graph retrieval can lose to plain RAG 
	// when the graph adds cost without evidence (low selectivity).
	if selectivity < 0.01 {
		return Plan{
			SelectedStrategy: PlainRAG,
			EstimatedCost:    vCost,
			Explanation:      "Low predicate selectivity; graph traversal unlikely to improve results.",
		}
	}

	// Rule 3: Strategy Selection
	if vCost < gCost {
		return Plan{
			SelectedStrategy: VectorFirst,
			EstimatedCost:    vCost,
			Explanation:      fmt.Sprintf("Vector search is cheaper (%.2fms) than graph traversal (%.2fms).", vCost, gCost),
		}
	}

	return Plan{
		SelectedStrategy: GraphFirst,
		EstimatedCost:    gCost,
		Explanation:      "Graph traversal preferred for dense local connectivity.",
	}
}
