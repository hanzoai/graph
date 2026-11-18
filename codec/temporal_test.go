package codec

import (
	"bytes"
	"testing"
)

func TestBitemporalEncoding(t *testing.T) {
	meta := &TemporalMeta{
		ValidFrom: 1000,
		ValidTo:   2000,
		TxnTime:   1500,
		Evidence:  []byte("ipfs://hash123"),
	}

	encoded := meta.EncodeTemporal()
	decoded := DecodeTemporal(encoded)

	if decoded.ValidFrom != meta.ValidFrom {
		t.Errorf("Expected ValidFrom %d, got %d", meta.ValidFrom, decoded.ValidFrom)
	}
	if !bytes.Equal(decoded.Evidence, meta.Evidence) {
		t.Errorf("Evidence mismatch")
	}
}

func TestSupersededFactLogic(t *testing.T) {
	// Scenario: PR #47 was open at 10 AM (VT: 10, TT: 10)
	// Later we learn it was merged at 11 AM (VT: 11, TT: 12)
	
	fact1 := &TemporalMeta{ValidFrom: 10, ValidTo: 11, TxnTime: 10}
	fact2 := &TemporalMeta{ValidFrom: 11, ValidTo: 0, TxnTime: 12}

	// At 10:30 AM (VT), was it open?
	if !fact1.IsValidAt(10) {
		t.Error("Fact1 should be valid at time 10")
	}

	// If we query with System Time 10, we only see fact1.
	// If we query with System Time 12, we see the progression.
	if fact2.WasKnownAt(10) {
		t.Error("Fact2 should not be known at system time 10")
	}
	if !fact2.WasKnownAt(12) {
		t.Error("Fact2 should be known at system time 12")
	}
}
