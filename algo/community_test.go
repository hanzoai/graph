package algo

import (
	"testing"
)

func TestLouvainBasic(t *testing.T) {
	g := NewCommunityGraph()
	
	// Cluster 1: 1, 2, 3
	g.AddEdge(1, 2, 1.0)
	g.AddEdge(2, 3, 1.0)
	g.AddEdge(3, 1, 1.0)
	
	// Bridge
	g.AddEdge(3, 4, 0.1)
	
	// Cluster 2: 4, 5, 6
	g.AddEdge(4, 5, 1.0)
	g.AddEdge(5, 6, 1.0)
	g.AddEdge(6, 4, 1.0)

	communities := g.Louvain()

	if communities[1] != communities[2] {
		t.Errorf("Nodes 1 and 2 should be in the same community")
	}
	if communities[1] == communities[4] {
		t.Errorf("Nodes 1 and 4 should be in different communities")
	}
}

func TestIncrementalInvalidation(t *testing.T) {
	mgr := NewGraphRAGManager()
	
	sources := map[string]string{"doc1": "v1", "doc2": "v1"}
	mgr.RegisterCommunity(101, 0, []uint64{1, 2, 3}, sources)
	
	node := mgr.Hierarchy[101]
	if node.Status != Pending {
		t.Errorf("Initial status should be Pending")
	}
	
	mgr.InvalidateSource("doc1", "v2")
	
	if node.Status != Invalidated {
		t.Errorf("Status should be Invalidated after source update")
	}
	
	if node.SourceHashes["doc1"] != "v2" {
		t.Errorf("Source hash was not updated")
	}
}
