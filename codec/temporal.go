package codec

import (
	"encoding/binary"
	"io"
)

// TemporalMeta represents the bitemporal and provenance metadata for an edge.
// ValidTime: When the fact was true in the real world.
// TxnTime: When the system recorded the fact.
// Evidence: Pointer to the source identity (e.g., a CID or internal UID).
type TemporalMeta struct {
	ValidFrom uint64
	ValidTo   uint64
	TxnTime   uint64
	Evidence  []byte
}

// EncodeTemporal serializes the temporal metadata into a byte slice.
func (m *TemporalMeta) EncodeTemporal() []byte {
	evidenceLen := len(m.Evidence)
	buf := make([]byte, 24+evidenceLen)
	binary.BigEndian.PutUint64(buf[0:8], m.ValidFrom)
	binary.BigEndian.PutUint64(buf[8:16], m.ValidTo)
	binary.BigEndian.PutUint64(buf[16:24], m.TxnTime)
	copy(buf[24:], m.Evidence)
	return buf
}

// DecodeTemporal deserializes the temporal metadata from a byte slice.
func DecodeTemporal(data []byte) *TemporalMeta {
	if len(data) < 24 {
		return nil
	}
	return &TemporalMeta{
		ValidFrom: binary.BigEndian.Uint64(data[0:8]),
		ValidTo:   binary.BigEndian.Uint64(data[8:16]),
		TxnTime:   binary.BigEndian.Uint64(data[16:24]),
		Evidence:  data[24:],
	}
}

// IsValidAt checks if the fact was valid at a specific point in world time.
func (m *TemporalMeta) IsValidAt(ts uint64) bool {
	return ts >= m.ValidFrom && (m.ValidTo == 0 || ts < m.ValidTo)
}

// WasKnownAt checks if the system knew this fact at a specific transaction time.
func (m *TemporalMeta) WasKnownAt(ts uint64) bool {
	return ts >= m.TxnTime
}
