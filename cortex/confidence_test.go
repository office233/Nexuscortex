package cortex

import (
	"math/bits"
	"testing"
)

func TestConfidenceTileGetSet(t *testing.T) {
	tile := NewConfidenceTile(0) // all at 0

	// Set weight 0 to confidence 3
	tile = tile.SetConfidence(0, 3)
	if got := tile.GetConfidence(0); got != 3 {
		t.Errorf("weight 0: expected 3, got %d", got)
	}

	// Set weight 5 to confidence 2
	tile = tile.SetConfidence(5, 2)
	if got := tile.GetConfidence(5); got != 2 {
		t.Errorf("weight 5: expected 2, got %d", got)
	}

	// Set weight 15 to confidence 1
	tile = tile.SetConfidence(15, 1)
	if got := tile.GetConfidence(15); got != 1 {
		t.Errorf("weight 15: expected 1, got %d", got)
	}

	// Others should still be 0
	if got := tile.GetConfidence(1); got != 0 {
		t.Errorf("weight 1: expected 0, got %d", got)
	}
}

func TestConfidenceTileNewWithLevel(t *testing.T) {
	for level := uint8(0); level <= 3; level++ {
		tile := NewConfidenceTile(level)
		for i := 0; i < 16; i++ {
			if got := tile.GetConfidence(i); got != level {
				t.Errorf("level %d, weight %d: expected %d, got %d", level, i, level, got)
			}
		}
	}
}

func TestConfidenceMask(t *testing.T) {
	// Create tile: weights 0-3 at conf=3, weights 4-7 at conf=1, rest at conf=0
	tile := NewConfidenceTile(0)
	for i := 0; i < 4; i++ {
		tile = tile.SetConfidence(i, 3)
	}
	for i := 4; i < 8; i++ {
		tile = tile.SetConfidence(i, 1)
	}

	// Threshold 0: all pass
	mask := tile.ConfidenceMask(0)
	if mask != 0xFFFF {
		t.Errorf("threshold 0: expected 0xFFFF, got 0x%04X", mask)
	}

	// Threshold 1: weights 0-7 pass (conf >= 1)
	mask = tile.ConfidenceMask(1)
	active := bits.OnesCount16(mask)
	if active != 8 {
		t.Errorf("threshold 1: expected 8 active, got %d (mask=0x%04X)", active, mask)
	}

	// Threshold 2: only weights 0-3 pass (conf >= 2)
	mask = tile.ConfidenceMask(2)
	active = bits.OnesCount16(mask)
	if active != 4 {
		t.Errorf("threshold 2: expected 4 active, got %d (mask=0x%04X)", active, mask)
	}

	// Threshold 3: only weights 0-3 pass (conf == 3)
	mask = tile.ConfidenceMask(3)
	active = bits.OnesCount16(mask)
	if active != 4 {
		t.Errorf("threshold 3: expected 4 active, got %d (mask=0x%04X)", active, mask)
	}
}

func TestConfidenceIncrementDecrement(t *testing.T) {
	tile := NewConfidenceTile(1) // all at 1

	// Increment weight 0: 1 → 2
	tile = tile.Increment(0)
	if got := tile.GetConfidence(0); got != 2 {
		t.Errorf("after increment: expected 2, got %d", got)
	}

	// Increment again: 2 → 3
	tile = tile.Increment(0)
	if got := tile.GetConfidence(0); got != 3 {
		t.Errorf("after 2nd increment: expected 3, got %d", got)
	}

	// Increment again: 3 → 3 (capped)
	tile = tile.Increment(0)
	if got := tile.GetConfidence(0); got != 3 {
		t.Errorf("after 3rd increment: expected 3 (capped), got %d", got)
	}

	// Decrement weight 5: 1 → 0
	tile = tile.Decrement(5)
	if got := tile.GetConfidence(5); got != 0 {
		t.Errorf("after decrement: expected 0, got %d", got)
	}

	// Decrement again: 0 → 0 (capped)
	tile = tile.Decrement(5)
	if got := tile.GetConfidence(5); got != 0 {
		t.Errorf("after 2nd decrement: expected 0 (capped), got %d", got)
	}
}

func TestForwardWithConfidence(t *testing.T) {
	// Create a small layer: 16 inputs → 2 outputs
	layer := NewTernaryLayer(16, 2)

	// Set all weights to +1 in first row
	weights := [16]int8{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
	layer.Tiles[0] = PackTernaryTile(weights)

	// Set all weights to -1 in second row
	negWeights := [16]int8{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
	layer.Tiles[1] = PackTernaryTile(negWeights)

	// Create confidence: first 8 weights at conf=3, last 8 at conf=0
	conf := NewConfidenceLayer(layer, 0)
	for i := 0; i < 8; i++ {
		conf.Tiles[0] = conf.Tiles[0].SetConfidence(i, 3)
		conf.Tiles[1] = conf.Tiles[1].SetConfidence(i, 3)
	}

	// Input: all 16 bits active
	activeMask := []uint16{0xFFFF}

	// Without confidence (threshold=0): all 16 weights contribute
	outAll := ForwardWithConfidence(layer, conf, activeMask, 0)
	t.Logf("Threshold 0 (all): output[0]=%d, output[1]=%d", outAll[0], outAll[1])
	if outAll[0] != 16 {
		t.Errorf("threshold 0, row 0: expected 16, got %d", outAll[0])
	}
	if outAll[1] != -16 {
		t.Errorf("threshold 0, row 1: expected -16, got %d", outAll[1])
	}

	// With confidence threshold=2: only first 8 weights (conf=3) contribute
	outConf := ForwardWithConfidence(layer, conf, activeMask, 2)
	t.Logf("Threshold 2 (confident): output[0]=%d, output[1]=%d", outConf[0], outConf[1])
	if outConf[0] != 8 {
		t.Errorf("threshold 2, row 0: expected 8, got %d", outConf[0])
	}
	if outConf[1] != -8 {
		t.Errorf("threshold 2, row 1: expected -8, got %d", outConf[1])
	}
}

func TestConfidenceAdaptiveSparsity(t *testing.T) {
	layer := NewTernaryLayer(1024, 512)
	// Fill with random-ish weights (all +1 for simplicity)
	for i := range layer.Tiles {
		layer.Tiles[i] = PackTernaryTile([16]int8{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
	}

	// All start at confidence 2 (75%)
	conf := NewConfidenceLayer(layer, 2)

	// Sparsity at threshold 0 should be 0% (all pass)
	s0 := conf.SparsityAtThreshold(0)
	if s0 != 0 {
		t.Errorf("threshold 0: expected 0%% sparsity, got %.1f%%", s0*100)
	}

	// Sparsity at threshold 3 should be 100% (none at conf=3)
	s3 := conf.SparsityAtThreshold(3)
	if s3 != 1.0 {
		t.Errorf("threshold 3: expected 100%% sparsity, got %.1f%%", s3*100)
	}

	// Sparsity at threshold 2 should be 0% (all at conf=2)
	s2 := conf.SparsityAtThreshold(2)
	if s2 != 0 {
		t.Errorf("threshold 2: expected 0%% sparsity, got %.1f%%", s2*100)
	}

	t.Logf("Sparsity: threshold 0=%.0f%%, threshold 2=%.0f%%, threshold 3=%.0f%%",
		s0*100, s2*100, s3*100)
}

func TestConfidenceLayerSerialize(t *testing.T) {
	layer := NewTernaryLayer(32, 4)
	conf := NewConfidenceLayer(layer, 2)

	// Modify some tiles
	conf.Tiles[0] = conf.Tiles[0].SetConfidence(0, 3)
	conf.Tiles[1] = conf.Tiles[1].SetConfidence(5, 0)

	// Marshal
	data := conf.MarshalBinary()

	// Unmarshal
	restored, err := UnmarshalConfidenceLayer(data)
	if err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	// Verify
	if restored.TilesPerRow != conf.TilesPerRow {
		t.Errorf("TilesPerRow: expected %d, got %d", conf.TilesPerRow, restored.TilesPerRow)
	}
	if restored.OutputSize != conf.OutputSize {
		t.Errorf("OutputSize: expected %d, got %d", conf.OutputSize, restored.OutputSize)
	}
	if len(restored.Tiles) != len(conf.Tiles) {
		t.Fatalf("tile count: expected %d, got %d", len(conf.Tiles), len(restored.Tiles))
	}
	for i, tile := range conf.Tiles {
		if restored.Tiles[i] != tile {
			t.Errorf("tile %d: expected %08X, got %08X", i, uint32(tile), uint32(restored.Tiles[i]))
		}
	}
}

func TestConfidenceAverageConfidence(t *testing.T) {
	// All at max confidence
	tileMax := NewConfidenceTile(3)
	avg := tileMax.AverageConfidence()
	if avg != 255 {
		t.Errorf("all conf=3: expected avg=255, got %d", avg)
	}

	// All at zero confidence
	tileZero := NewConfidenceTile(0)
	avg = tileZero.AverageConfidence()
	if avg != 0 {
		t.Errorf("all conf=0: expected avg=0, got %d", avg)
	}

	// Mixed: half at 3, half at 0
	tileMixed := NewConfidenceTile(0)
	for i := 0; i < 8; i++ {
		tileMixed = tileMixed.SetConfidence(i, 3)
	}
	avg = tileMixed.AverageConfidence()
	// 8*3 = 24 out of 48 max → 24/48 * 255 = 127
	if avg < 125 || avg > 129 {
		t.Errorf("half conf=3: expected avg≈127, got %d", avg)
	}
	t.Logf("Mixed confidence avg: %d/255", avg)
}

func TestConfirmContradictWeights(t *testing.T) {
	layer := NewTernaryLayer(16, 2)
	conf := NewConfidenceLayer(layer, 1) // all start at 1

	// Confirm row 0 with first 4 bits active
	activeMask := []uint16{0x000F} // bits 0-3
	conf.ConfirmWeights(0, activeMask)

	// Weights 0-3 should be at conf=2, rest at conf=1
	for i := 0; i < 4; i++ {
		if got := conf.Tiles[0].GetConfidence(i); got != 2 {
			t.Errorf("weight %d after confirm: expected 2, got %d", i, got)
		}
	}
	for i := 4; i < 16; i++ {
		if got := conf.Tiles[0].GetConfidence(i); got != 1 {
			t.Errorf("weight %d after confirm: expected 1 (unchanged), got %d", i, got)
		}
	}

	// Contradict row 0 with bits 0-3
	conf.ContradictWeights(0, activeMask)

	// Weights 0-3 should be back to conf=1
	for i := 0; i < 4; i++ {
		if got := conf.Tiles[0].GetConfidence(i); got != 1 {
			t.Errorf("weight %d after contradict: expected 1, got %d", i, got)
		}
	}
}

func BenchmarkForwardWithConfidence(b *testing.B) {
	layer := NewTernaryLayer(1024, 512)
	for i := range layer.Tiles {
		layer.Tiles[i] = 0xFFFFFFFF // all active
	}
	conf := NewConfidenceLayer(layer, 2)

	// Create input
	activeMask := make([]uint16, layer.TilesPerRow)
	for i := range activeMask {
		activeMask[i] = 0xFFFF
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ForwardWithConfidence(layer, conf, activeMask, 2)
	}
}

func BenchmarkForwardPopcountBaseline(b *testing.B) {
	layer := NewTernaryLayer(1024, 512)
	for i := range layer.Tiles {
		layer.Tiles[i] = 0xFFFFFFFF
	}

	activeMask := make([]uint16, layer.TilesPerRow)
	for i := range activeMask {
		activeMask[i] = 0xFFFF
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		layer.ForwardPopcount(activeMask)
	}
}
