package algo

import (
	"testing"
)

// MockGraph implements WeightedGraph for testing.
type MockGraph struct {
	adj map[uint64][]uint64
	w   map[uint64][]float64
}

func (m *MockGraph) Neighbors(uid uint64) ([]uint64, []float64) {
	return m.adj[uid], m.w[uid]
}

func TestPPR(t *testing.T) {
	// Create a simple graph: 1 -> 2 -> 3
	//                    1 -> 3
	g := &MockGraph{
		adj: map[uint64][]uint64{
			1: {2, 3},
			2: {3},
		},
		w: map[uint64][]float64{
			1: {1.0, 1.0},
			2: {1.0},
		},
	}

	seeds := map[uint64]float64{1: 1.0}
	params := DefaultPPRParams()
	params.MaxIterations = 50

	scores := ComputePPR(seeds, g, params)

	if scores[1] == 0 {
		t.Errorf("Seed node 1 should have a score")
	}

	if scores[3] <= scores[2] {
		t.Errorf("Node 3 should have a higher score than Node 2 due to multiple paths. Got 3:%f, 2:%f", scores[3], scores[2])
	}

	t.Logf("Scores: 1:%f, 2:%f, 3:%f", scores[1], scores[2], scores[3])
}

func TestPPRConvergence(t *testing.T) {
	// Cyclic graph: 1 <-> 2
	g := &MockGraph{
		adj: map[uint64][]uint64{
			1: {2},
			2: {1},
		},
		w: map[uint64][]float64{
			1: {1.0},
			2: {1.0},
		},
	}

	seeds := map[uint64]float64{1: 1.0}
	params := DefaultPPRParams()
	
	scores := ComputePPR(seeds, g, params)

	sum := 0.0
	for _, s := range scores {
		sum += s
	}

	if math.Abs(sum-1.0) > 1e-7 {
		t.Errorf("Total probability mass should be 1.0, got %f", sum)
	}
}

func TestPPRWeighted(t *testing.T) {
	// Node 1 points to 2 and 3. Edge to 2 is much stronger.
	g := &MockGraph{
		adj: map[uint64][]uint64{
			1: {2, 3},
		},
		w: map[uint64][]float64{
			1: {10.0, 1.0},
		},
	}

	seeds := map[uint64]float64{1: 1.0}
	scores := ComputePPR(seeds, g, DefaultPPRParams())

	if scores[2] <= scores[3] {
		t.Errorf("Node 2 should have higher score than 3 due to edge weight. Got 2:%f, 3:%f", scores[2], scores[3])
	}
}
