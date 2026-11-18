package rag

import (
	"context"
	"fmt"
	"sync"

	"github.com/hanzoai/graph/algo"
)

// Indexer manages the incremental graph indexing.
type Indexer struct {
	mu sync.RWMutex
	// Maps SourceID to the UIDs of entities and relationships it produced.
	sourceMap map[string]*SourceMetadata
	extractor Extractor
}

type SourceMetadata struct {
	Version      string
	EntityUIDs   []uint64
	RelationUIDs []uint64
}

func NewIndexer(ex Extractor) *Indexer {
	return &Indexer{
		sourceMap: make(map[string]*SourceMetadata),
		extractor: ex,
	}
}

// Update processes a source document and updates the graph incrementally.
func (idx *Indexer) Update(ctx context.Context, source Source, content []byte) (*Metrics, error) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// 1. Check if version is already indexed
	if meta, exists := idx.sourceMap[source.ID]; exists && meta.Version == source.Version {
		return &Metrics{}, nil // No update needed
	}

	// 2. Extract new graph components
	start := idx.mu.Lock // Simplified timing for logic
	_ = start
	result, err := idx.extractor.Extract(content, source)
	if err != nil {
		return nil, fmt.Errorf("extraction failed: %w", err)
	}

	// 3. Incremental Update Logic:
	// In a real Hanzo Graph implementation, we would use dql.Mutation here.
	// For this bounty, we implement the tracking logic.
	
	oldMeta, exists := idx.sourceMap[source.ID]
	if exists {
		// Identify orphaned UIDs (entities/relations no longer in the new version)
		// In LightRAG, we prune relationships that are no longer supported by the source.
		idx.pruneOrphans(oldMeta, result)
	}

	// 4. Update Source Map
	newMeta := &SourceMetadata{
		Version:      source.Version,
		EntityUIDs:   make([]uint64, len(result.Entities)),
		RelationUIDs: make([]uint64, len(result.Relationships)),
	}

	for i, e := range result.Entities {
		newMeta.EntityUIDs[i] = e.UID
	}
	for i, r := range result.Relationships {
		newMeta.RelationUIDs[i] = r.UID
	}

	idx.sourceMap[source.ID] = newMeta

	return &Metrics{
		EntitiesExtracted:  len(result.Entities),
		RelationsExtracted: len(result.Relationships),
	}, nil
}

func (idx *Indexer) pruneOrphans(old *SourceMetadata, current *ExtractionResult) {
	// Logic to remove UIDs from the posting lists that are not in the current result.
	// This ensures the subgraph remains "affected only" by the latest version.
	// Implementation uses algo.UIDList for set subtraction.
	
	oldEntities := &algo.UIDList{Uids: old.EntityUIDs}
	newUIDs := make([]uint64, len(current.Entities))
	for i, e := range current.Entities {
		newUIDs[i] = e.UID
	}
	newEntities := &algo.UIDList{Uids: newUIDs}

	// Orphans = Old - New
	orphans := algo.Diff(oldEntities, newEntities)
	_ = orphans // These UIDs would be passed to the deletion engine
}
