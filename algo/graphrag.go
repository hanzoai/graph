package algo

import (
	"sync"
)

// SummaryStatus defines the state of a community summary.
type SummaryStatus int

const (
	Valid SummaryStatus = iota
	Invalidated
	Pending
)

// CommunityNode represents a node in the hierarchical summary tree.
type CommunityNode struct {
	ID           int
	Level        int
	ChildNodes   []int
	MemberUIDs   []uint64
	Summary      string
	Status       SummaryStatus
	SourceHashes map[string]string // SourceID -> VersionHash
}

// GraphRAGManager handles the hierarchical summaries and incremental updates.
type GraphRAGManager struct {
	mu         sync.RWMutex
	Hierarchy  map[int]*CommunityNode
	SourceToCom map[string][]int // SourceID -> List of Community IDs it influences
}

func NewGraphRAGManager() *GraphRAGManager {
	return &GraphRAGManager{
		Hierarchy:   make(map[int]*CommunityNode),
		SourceToCom: make(map[string][]int),
	}
}

// InvalidateSource marks all summaries derived from a specific source as invalid.
func (m *GraphRAGManager) InvalidateSource(sourceID string, newHash string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	comIDs, exists := m.SourceToCom[sourceID]
	if !exists {
		return
	}

	for _, id := range comIDs {
		if node, ok := m.Hierarchy[id]; ok {
			node.Status = Invalidated
			node.SourceHashes[sourceID] = newHash
			// Invalidate parents recursively
			m.invalidateParents(id)
		}
	}
}

func (m *GraphRAGManager) invalidateParents(childID int) {
	// In a real implementation, we would traverse the tree upwards.
	for _, node := range m.Hierarchy {
		for _, cid := range node.ChildNodes {
			if cid == childID {
				node.Status = Invalidated
				m.invalidateParents(node.ID)
			}
		}
	}
}

// RegisterCommunity adds a detected community to the RAG manager.
func (m *GraphRAGManager) RegisterCommunity(id int, level int, uids []uint64, sources map[string]string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	node := &CommunityNode{
		ID:           id,
		Level:        level,
		MemberUIDs:   uids,
		Status:       Pending,
		SourceHashes: sources,
	}

	m.Hierarchy[id] = node

	for sID := range sources {
		m.SourceToCom[sID] = append(m.SourceToCom[sID], id)
	}
}
