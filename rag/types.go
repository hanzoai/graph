package rag

import (
	"time"
)

// Entity represents a node in the LightRAG graph.
type Entity struct {
	UID         uint64
	Name        string
	Type        string // e.g., "Person", "Organization", "Concept"
	Description string
	Level       string // "Low" or "High" (LightRAG dual-level)
	SourceID    string
}

// Relationship represents an edge between two entities.
type Relationship struct {
	UID         uint64
	SourceUID   uint64
	TargetUID   uint64
	Type        string
	Description string
	Weight      float64
	SourceID    string
}

// Source represents the origin of the data (Document, Commit, etc.)
type Source struct {
	ID        string
	Version   string
	Type      string // "doc", "commit", "event"
	Timestamp time.Time
}

// ExtractionResult holds the entities and relationships found in a source.
type ExtractionResult struct {
	Entities      []Entity
	Relationships []Relationship
}

// Extractor defines the interface for LLM-based extraction.
type Extractor interface {
	Extract(content []byte, source Source) (*ExtractionResult, error)
}

// Metrics tracks the performance of the indexing process.
type Metrics struct {
	IngestLatency      time.Duration
	EntitiesExtracted  int
	RelationsExtracted int
	DuplicateRate      float64
	CostUSD            float64
}
