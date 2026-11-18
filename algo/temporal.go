package algo

import (
	"github.com/hanzoai/graph/codec"
)

// Fact represents a bitemporal edge result.
type Fact struct {
	UID      uint64
	Metadata *codec.TemporalMeta
}

// FilterBitemporal selects facts that were valid at 'worldTime' 
// as known by the system at 'systemTime'.
// This solves the "superseded facts" problem by allowing historical reconstruction.
func FilterBitemporal(facts []Fact, worldTime uint64, systemTime uint64) []uint64 {
	var result []uint64
	
	// Map to keep track of the most recent transaction for a given valid time window
	// In a full implementation, this would handle complex overlaps.
	for _, fact := range facts {
		if fact.Metadata.WasKnownAt(systemTime) && fact.Metadata.IsValidAt(worldTime) {
			result = append(result, fact.UID)
		}
	}
	return result
}
