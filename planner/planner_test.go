package planner

import (
	"testing"
)

func TestPlannerDecisions(t *testing.T) {
	stats := &MockStatsProvider{
		DefaultDegree: 5,
		PredicateSelectivity: map[string]float64{
			"knows": 0.5,
			"rare":  0.001,
		},
	}
	p := NewPlanner(stats)

	t.Run("Prefer Vector for High Latency Graph", func(t *testing.T) {
		params := QueryParams{
			LatencyBudget: 100.0,
			VectorK:       10,
			Hops:          3, // 5^3 = 125 nodes, will be expensive
		}
		plan := p.Plan(params, "knows")
		if plan.SelectedStrategy != VectorFirst {
			t.Errorf("Expected VectorFirst, got %v. Reason: %s", plan.SelectedStrategy, plan.Explanation)
		}
	})

	t.Run("Fallback to PlainRAG on Low Selectivity", func(t *testing.T) {
		params := QueryParams{
			LatencyBudget: 50.0,
			VectorK:       5,
			Hops:          1,
		}
		plan := p.Plan(params, "rare")
		if plan.SelectedStrategy != PlainRAG {
			t.Errorf("Expected PlainRAG for rare predicate, got %v", plan.SelectedStrategy)
		}
	})

	t.Run("Enforce Latency Budget", func(t *testing.T) {
		params := QueryParams{
			LatencyBudget: 1.0, // Impossible budget
			VectorK:       100,
			Hops:          5,
		}
		plan := p.Plan(params, "knows")
		if plan.SelectedStrategy != PlainRAG {
			t.Error("Should have fallen back to PlainRAG due to budget constraints")
		}
	})
}
