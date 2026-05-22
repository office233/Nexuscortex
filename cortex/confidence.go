package cortex

import (
	"encoding/binary"
	"fmt"
)

// confidence.go — Adaptive Confidence Channel for Ternary Weights
//
// Adds a parallel "confidence tile" alongside each weight tile.
// Each weight gets 2 bits of confidence (4 levels):
//
//   00 = 25% confident  (weight is unreliable, skip in smart forward)
//   01 = 50% confident  (weak evidence)
//   10 = 75% confident  (moderate evidence)
//   11 = 100% confident (strongly confirmed by experience)
//
// Packing: 16 weights × 2 bits = 32 bits = 1 uint32 per tile.
// Same alignment as TernaryTile — one ConfidenceTile per TernaryTile.
//
// Key insight: this enables ADAPTIVE SPARSITY. During forward pass,
// weights with confidence below threshold are treated as zero → skipped.
// This means the network automatically becomes sparser on uncertain
// weights and denser on well-learned weights.

// ConfidenceTile stores 2-bit confidence levels for 16 weights.
// Packed identically to TernaryTile: one uint32 = 16 weights × 2 bits.
//
// Layout:
//   bits  0-15: confidence level low bit per weight (weight 0 = bit 0)
//   bits 16-31: confidence level high bit per weight (weight 0 = bit 16)
//
// Confidence level = (highBit << 1) | lowBit → 0,1,2,3 → maps to 25%,50%,75%,100%
type ConfidenceTile uint32

// ConfidenceLevel constants
const (
	ConfLow      = 0 // 25% — unreliable, skip in smart forward
	ConfWeak     = 1 // 50% — weak evidence
	ConfModerate = 2 // 75% — moderate evidence
	ConfHigh     = 3 // 100% — strongly confirmed
)

// NewConfidenceTile creates a tile with all weights at the given confidence level.
func NewConfidenceTile(level uint8) ConfidenceTile {
	if level > 3 {
		level = 3
	}
	var lo, hi uint16
	if level&1 != 0 {
		lo = 0xFFFF // all 16 low bits set
	}
	if level&2 != 0 {
		hi = 0xFFFF // all 16 high bits set
	}
	return ConfidenceTile(uint32(lo) | uint32(hi)<<16)
}

// GetConfidence returns the confidence level (0-3) for weight index 0-15.
func (c ConfidenceTile) GetConfidence(idx int) uint8 {
	if idx < 0 || idx >= 16 {
		return 0
	}
	v := uint32(c)
	lo := (v >> uint(idx)) & 1
	hi := (v >> uint(16+idx)) & 1
	return uint8(hi<<1 | lo)
}

// SetConfidence sets the confidence level (0-3) for weight index 0-15.
func (c ConfidenceTile) SetConfidence(idx int, level uint8) ConfidenceTile {
	if idx < 0 || idx >= 16 || level > 3 {
		return c
	}
	v := uint32(c)
	// Clear existing bits
	v &^= (1 << uint(idx)) | (1 << uint(16+idx))
	// Set new bits
	if level&1 != 0 {
		v |= 1 << uint(idx)
	}
	if level&2 != 0 {
		v |= 1 << uint(16+idx)
	}
	return ConfidenceTile(v)
}

// ConfidenceMask returns a 16-bit mask where bit i is set if weight i
// has confidence >= threshold (0-3). Used in ForwardWithConfidence to
// skip uncertain weights.
func (c ConfidenceTile) ConfidenceMask(threshold uint8) uint16 {
	v := uint32(c)
	lo := uint16(v)
	hi := uint16(v >> 16)

	switch threshold {
	case 0:
		// All weights pass (even confidence=0)
		return 0xFFFF
	case 1:
		// confidence >= 1: at least one bit set
		return lo | hi
	case 2:
		// confidence >= 2: high bit must be set
		return hi
	case 3:
		// confidence == 3: both bits must be set
		return lo & hi
	default:
		return 0
	}
}

// Increment increases confidence for weight idx by 1 (capped at 3).
func (c ConfidenceTile) Increment(idx int) ConfidenceTile {
	level := c.GetConfidence(idx)
	if level < 3 {
		return c.SetConfidence(idx, level+1)
	}
	return c
}

// Decrement decreases confidence for weight idx by 1 (capped at 0).
func (c ConfidenceTile) Decrement(idx int) ConfidenceTile {
	level := c.GetConfidence(idx)
	if level > 0 {
		return c.SetConfidence(idx, level-1)
	}
	return c
}

// AverageConfidence returns the mean confidence level (0-3) across all 16 weights.
// Returns fixed-point 0-255 where 255 = all weights at confidence 3.
func (c ConfidenceTile) AverageConfidence() uint8 {
	v := uint32(c)
	lo := uint16(v)
	hi := uint16(v >> 16)

	// Sum of all confidence levels:
	// level = 2*highBit + lowBit
	// total = 2 * popcount(hi) + popcount(lo)
	total := 2*pop16[hi] + pop16[lo]

	// Max possible: 16 weights × 3 = 48
	// Scale to 0-255: (total * 255) / 48
	return uint8((uint16(total) * 255) / 48)
}

// ─────────────────────────────────────────────────────────────────────
// ConfidenceLayer — Parallel confidence for an entire TernaryLayer
// ─────────────────────────────────────────────────────────────────────

// ConfidenceLayer stores per-weight confidence for an entire TernaryLayer.
// It has exactly the same tile layout: one ConfidenceTile per TernaryTile.
type ConfidenceLayer struct {
	Tiles       []ConfidenceTile
	TilesPerRow int
	OutputSize  int
}

// NewConfidenceLayer creates a confidence layer matching a TernaryLayer's dimensions.
// All weights start at the given initial confidence level.
func NewConfidenceLayer(layer *TernaryLayer, initialLevel uint8) *ConfidenceLayer {
	tile := NewConfidenceTile(initialLevel)
	tiles := make([]ConfidenceTile, len(layer.Tiles))
	for i := range tiles {
		tiles[i] = tile
	}
	return &ConfidenceLayer{
		Tiles:       tiles,
		TilesPerRow: layer.TilesPerRow,
		OutputSize:  layer.OutputSize,
	}
}

// ForwardWithConfidence performs a forward pass that skips low-confidence weights.
// Weights with confidence below threshold are treated as zero (skipped).
// This creates adaptive sparsity: uncertain weights don't contribute.
//
// Returns the output activations (same format as TernaryLayer.Forward).
func ForwardWithConfidence(layer *TernaryLayer, conf *ConfidenceLayer, activeMask []uint16, threshold uint8) []int16 {
	output := make([]int16, layer.OutputSize)
	copy(output, layer.Bias)

	for j := 0; j < layer.OutputSize; j++ {
		var acc int32
		rowOffset := j * layer.TilesPerRow

		for t := 0; t < layer.TilesPerRow; t++ {
			tileIdx := rowOffset + t
			tile := uint32(layer.Tiles[tileIdx])
			if tile == 0 {
				continue // All-zero tile — skip
			}

			// Get confidence mask: only weights above threshold contribute
			confMask := conf.Tiles[tileIdx].ConfidenceMask(threshold)

			// Extract sign and active masks
			signLo := uint8(tile)
			maskLo := uint8(tile >> 8)
			signHi := uint8(tile >> 16)
			maskHi := uint8(tile >> 24)

			// Combine into 16-bit masks
			posMask := uint16(maskLo&^signLo) | uint16(maskHi&^signHi)<<8
			negMask := uint16(maskLo&signLo) | uint16(maskHi&signHi)<<8

			// Apply confidence filter: only confident weights contribute
			posMask &= confMask
			negMask &= confMask

			// AND with input
			if t < len(activeMask) {
				active := activeMask[t]
				acc += int32(pop16[posMask&active]) - int32(pop16[negMask&active])
			}
		}

		if acc > 32767 {
			acc = 32767
		} else if acc < -32768 {
			acc = -32768
		}
		output[j] += int16(acc)
	}

	return output
}

// ─────────────────────────────────────────────────────────────────────
// Confidence Plasticity — Update confidence based on learning signals
// ─────────────────────────────────────────────────────────────────────

// ConfirmWeights increases confidence for weights that contributed
// to a correct prediction. Takes the input SDR and the tile positions
// that were active in the correct output.
func (cl *ConfidenceLayer) ConfirmWeights(rowIdx int, activeMask []uint16) {
	if rowIdx < 0 || rowIdx >= cl.OutputSize {
		return
	}
	rowOffset := rowIdx * cl.TilesPerRow

	for t := 0; t < cl.TilesPerRow && t < len(activeMask); t++ {
		if activeMask[t] == 0 {
			continue
		}
		tileIdx := rowOffset + t
		if tileIdx >= len(cl.Tiles) {
			break
		}
		// Increment confidence for each active input weight
		active := activeMask[t]
		for active != 0 {
			bit := active & (-active) // lowest set bit
			weightIdx := 0
			tmp := bit
			for tmp > 1 {
				tmp >>= 1
				weightIdx++
			}
			cl.Tiles[tileIdx] = cl.Tiles[tileIdx].Increment(weightIdx)
			active &^= bit
		}
	}
}

// ContradicWeights decreases confidence for weights that contributed
// to an incorrect prediction.
func (cl *ConfidenceLayer) ContradictWeights(rowIdx int, activeMask []uint16) {
	if rowIdx < 0 || rowIdx >= cl.OutputSize {
		return
	}
	rowOffset := rowIdx * cl.TilesPerRow

	for t := 0; t < cl.TilesPerRow && t < len(activeMask); t++ {
		if activeMask[t] == 0 {
			continue
		}
		tileIdx := rowOffset + t
		if tileIdx >= len(cl.Tiles) {
			break
		}
		active := activeMask[t]
		for active != 0 {
			bit := active & (-active)
			weightIdx := 0
			tmp := bit
			for tmp > 1 {
				tmp >>= 1
				weightIdx++
			}
			cl.Tiles[tileIdx] = cl.Tiles[tileIdx].Decrement(weightIdx)
			active &^= bit
		}
	}
}

// SparsityAtThreshold returns what fraction of weights would be skipped
// at the given confidence threshold. Returns 0.0-1.0.
func (cl *ConfidenceLayer) SparsityAtThreshold(threshold uint8) float64 {
	totalActive := 0
	totalWeights := len(cl.Tiles) * 16

	for _, tile := range cl.Tiles {
		mask := tile.ConfidenceMask(threshold)
		totalActive += int(pop16[mask])
	}

	if totalWeights == 0 {
		return 0
	}
	return 1.0 - float64(totalActive)/float64(totalWeights)
}

// MarshalBinary serializes the confidence layer to bytes.
func (cl *ConfidenceLayer) MarshalBinary() []byte {
	data := make([]byte, 8+len(cl.Tiles)*4)
	binary.LittleEndian.PutUint32(data[0:], uint32(cl.TilesPerRow))
	binary.LittleEndian.PutUint32(data[4:], uint32(cl.OutputSize))
	for i, t := range cl.Tiles {
		binary.LittleEndian.PutUint32(data[8+i*4:], uint32(t))
	}
	return data
}

// UnmarshalConfidenceLayer deserializes a confidence layer from bytes.
func UnmarshalConfidenceLayer(data []byte) (*ConfidenceLayer, error) {
	if len(data) < 8 {
		return nil, fmt.Errorf("confidence layer too short: %d bytes", len(data))
	}
	tilesPerRow := int(binary.LittleEndian.Uint32(data[0:]))
	outputSize := int(binary.LittleEndian.Uint32(data[4:]))

	if tilesPerRow <= 0 || tilesPerRow > 100000 || outputSize <= 0 || outputSize > 100000 {
		return nil, fmt.Errorf("confidence layer invalid dims: %dx%d", tilesPerRow, outputSize)
	}

	expectedTiles := tilesPerRow * outputSize
	expectedSize := 8 + expectedTiles*4
	if len(data) < expectedSize {
		return nil, fmt.Errorf("confidence layer data too short: got %d, need %d", len(data), expectedSize)
	}

	tiles := make([]ConfidenceTile, expectedTiles)
	for i := range tiles {
		tiles[i] = ConfidenceTile(binary.LittleEndian.Uint32(data[8+i*4:]))
	}

	return &ConfidenceLayer{
		Tiles:       tiles,
		TilesPerRow: tilesPerRow,
		OutputSize:  outputSize,
	}, nil
}
