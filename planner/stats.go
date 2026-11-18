package planner

// MockStatsProvider implements StatsProvider for testing and initial integration.
type MockStatsProvider struct {
	DefaultDegree      int
	PredicateSelectivity map[string]float64
}

func (m *MockStatsProvider) GetDegree(uid uint64) int {
	if m.DefaultDegree == 0 {
		return 10 // Default heuristic
	}
	return m.DefaultDegree
}

func (m *MockStatsProvider) GetSelectivity(predicate string) float64 {
	if val, ok := m.PredicateSelectivity[predicate]; ok {
		return val
	}
	return 0.1 // Default 10% selectivity
}

func (m *MockStatsProvider) GetEstimateCount(predicate string) int {
	return 1000 // Placeholder for total count * selectivity
}
