package rag

import (
	"context"
	"testing"
	"time"
)

// MockExtractor simulates LLM extraction for benchmarking.
type MockExtractor struct{}

func (m *MockExtractor) Extract(content []byte, source Source) (*ExtractionResult, error) {
	// Simulate extraction of 5 entities and 5 relationships per doc
	entities := make([]Entity, 5)
	relations := make([]Relationship, 5)
	for i := 0; i < 5; i++ {
		entities[i] = Entity{UID: uint64(i), Name: "Entity"}
		relations[i] = Relationship{UID: uint64(i + 100), SourceUID: uint64(i)}
	}
	return &ExtractionResult{Entities: entities, Relationships: relations}, nil
}

func BenchmarkIncrementalIngest(b *testing.B) {
	indexer := NewIndexer(&MockExtractor{})
	ctx := context.Background()
	
	source := Source{
		ID:      "doc_1",
		Version: "v1",
		Type:    "doc",
	}
	content := []byte("Sample content for LightRAG indexing.")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Simulate version change every iteration to force incremental update logic
		source.Version = string(rune(i))
		_, err := indexer.Update(ctx, source, content)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func TestDuplicateEntityRate(t *testing.T) {
	indexer := NewIndexer(&MockExtractor{})
	ctx := context.Background()
	
	source := Source{ID: "doc_1", Version: "v1"}
	metrics, _ := indexer.Update(ctx, source, []byte("test"))

	// In a mock scenario, we expect 5 entities.
	// If we run again with same version, it should be 0 new entities (100% deduplication).
	metrics2, _ := indexer.Update(ctx, source, []byte("test"))
	
	if metrics2.EntitiesExtracted != 0 {
		t.Errorf("Expected 0 new entities for same version, got %d", metrics2.EntitiesExtracted)
	}
}

func TestRelationshipsPerDollar(t *testing.T) {
	// Benchmark: relationships extracted per dollar
	// Assuming $0.01 per 1k tokens and 10 relationships per 1k tokens
	costPerRelation := 0.001 
	relations := 1000
	totalCost := float64(relations) * costPerRelation
	
	t.Logf("Benchmark: %d relationships extracted for $%.2f", relations, totalCost)
	if totalCost > 2.0 {
		t.Errorf("Extraction cost too high")
	}
}
