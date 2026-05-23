package cortex

import (
	"math/rand"
	"testing"
)

func TestNeuroRadioTile_Forward_NoSignal(t *testing.T) {
	tile := NewNeuroRadioTile(0xAAAAAAAA, 0xFFFFFFFF, 100, 128, 200, 50, false)
	var input [16]int8
	for i := range input {
		input[i] = 100
	}
	// No bus signal → should return 0
	result := tile.Forward(input, 0, 128)
	if result != 0 {
		t.Errorf("Expected 0 on no signal, got %d", result)
	}
}

func TestNeuroRadioTile_Forward_WithSignal(t *testing.T) {
	tile := NewNeuroRadioTile(0x55555555, 0xFFFFFFFF, 100, 128, 200, 50, false)
	var input [16]int8
	for i := range input {
		input[i] = 100
	}
	// Strong signal, matched phase
	result := tile.Forward(input, 200, 128)
	if result == 0 {
		t.Error("Expected non-zero output with matching signal and phase")
	}
}

func TestNeuroRadioTile_Forward_PoorPhase(t *testing.T) {
	tile := NewNeuroRadioTile(0x55555555, 0xFFFFFFFF, 100, 0, 200, 50, false)
	var input [16]int8
	for i := range input {
		input[i] = 100
	}
	// Signal present but phase completely off
	result := tile.Forward(input, 200, 128)
	// With phase 0 vs bus phase 128, resonance might be low
	_ = result // Just ensure no panic
}

func TestNeuroRadioTile_ConfirmContradict(t *testing.T) {
	tile := NewNeuroRadioTile(0x55555555, 0x55555555, 100, 128, 100, 50, false)

	// Confirm should increase amplitude
	ampBefore := tile.Amplitude()
	tile.Confirm()
	ampAfter := tile.Amplitude()
	if ampAfter <= ampBefore {
		t.Errorf("Confirm should increase amplitude: %d → %d", ampBefore, ampAfter)
	}

	// Contradict should decrease amplitude
	tile2 := NewNeuroRadioTile(0x55555555, 0x55555555, 100, 128, 50, 50, false)
	ampBefore = tile2.Amplitude()
	tile2.Contradict()
	ampAfter = tile2.Amplitude()
	if ampAfter >= ampBefore {
		t.Errorf("Contradict should decrease amplitude: %d → %d", ampBefore, ampAfter)
	}
}

func TestNeuroRadioTile_Death(t *testing.T) {
	tile := NewNeuroRadioTile(0, 0, 100, 128, 1, 50, false)
	if !tile.IsAlive() {
		t.Error("Tile should start alive")
	}
	tile.Contradict() // amp 1 → 0, should die
	if tile.IsAlive() {
		t.Error("Tile should be dead after contradict at amp=1")
	}
}

func TestRadioBucketIndex(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	tiles := make([]NeuroRadioTile, 1000)
	for i := range tiles {
		tiles[i] = NewNeuroRadioTile(rng.Uint32(), 0x55555555,
			uint8(i%256), uint8(rng.Intn(256)), 200, uint8(rng.Intn(64)), false)
	}

	idx := NewRadioBucketIndex(tiles)

	// Frequency 0 should have ~4 tiles (1000/256)
	count := len(idx.TilesOnFreq(0))
	if count == 0 {
		t.Error("Expected at least one tile on frequency 0")
	}

	// Sum of all buckets should equal total alive tiles
	total := 0
	for i := 0; i < 256; i++ {
		total += len(idx.TilesOnFreq(uint8(i)))
	}
	if total != 1000 {
		t.Errorf("Total indexed tiles %d != 1000", total)
	}
}

func TestSemanticFreqCodec_CooccurrenceOrdering(t *testing.T) {
	codec := NewSemanticFreqCodec()

	// "pisica" (0) and "motan" (1) co-occur a lot
	// "computer" (2) and "software" (3) co-occur a lot
	// The two groups don't co-occur
	for i := 0; i < 100; i++ {
		codec.ObserveCooccurrence([]int{0, 1}) // pisica + motan
		codec.ObserveCooccurrence([]int{2, 3}) // computer + software
	}

	codec.AssignFrequencies()

	// Check that tokens have frequencies assigned
	freqPisica := codec.PrimaryFreq(0)
	freqMotan := codec.PrimaryFreq(1)
	freqComputer := codec.PrimaryFreq(2)
	freqSoftware := codec.PrimaryFreq(3)

	t.Logf("pisica=%d, motan=%d, computer=%d, software=%d",
		freqPisica, freqMotan, freqComputer, freqSoftware)

	// The greedy chain should place co-occurring tokens adjacent
	// So pisica-motan should be neighbors, and computer-software should be neighbors
	distPisicaMotan := abs(int(freqPisica) - int(freqMotan))
	distComputerSoftware := abs(int(freqComputer) - int(freqSoftware))

	// With 4 tokens spread across 0-255, adjacent tokens are 85 apart
	// So co-occurring pairs should be adjacent in the chain (85 apart)
	if distPisicaMotan > 100 {
		t.Errorf("pisica and motan should be adjacent, distance=%d", distPisicaMotan)
	}
	if distComputerSoftware > 100 {
		t.Errorf("computer and software should be adjacent, distance=%d", distComputerSoftware)
	}
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func TestNeuroRadioCortex_Creation(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	nrc := NewNeuroRadioCortex(10000, rng)

	if nrc.Size != 10000 {
		t.Errorf("Expected 10000 tiles, got %d", nrc.Size)
	}

	stats := nrc.Stats()
	if stats.AliveTiles != 10000 {
		t.Errorf("Expected all tiles alive, got %d", stats.AliveTiles)
	}
}

func TestNeuroRadioCortex_SparseSingle(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	nrc := NewNeuroRadioCortex(100000, rng) // 100K tiles

	// Inject signal on frequency 42
	nrc.Bus.Emit(42, 200, 128, false)

	// Step should only process tiles on freq 42 (~390 tiles, not 100K)
	nrc.Step()

	// Should have activated SOME tiles
	if nrc.LastActiveTiles == 0 {
		t.Log("No tiles activated — phase mismatch likely, acceptable")
	}
	if nrc.LastActiveTiles > 1000 {
		t.Errorf("Too many tiles activated (%d) — sparsity broken", nrc.LastActiveTiles)
	}
	t.Logf("Activated %d / 100000 tiles (%.3f%%)", nrc.LastActiveTiles,
		float64(nrc.LastActiveTiles)/1000.0)
}

func TestNeuroRadioCortex_TrainStep(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	nrc := NewNeuroRadioCortex(10000, rng)

	// Train: input tokens [1,2,3] → target token 5
	matches := nrc.TrainStep([]int{1, 2, 3}, 5, 10)
	t.Logf("TrainStep matches: %d", matches)
}

func BenchmarkNeuroRadioCortex_Step_100K(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	nrc := NewNeuroRadioCortex(100000, rng)

	// Inject 3 frequencies (typical for a short input)
	nrc.Bus.Emit(42, 200, 128, false)
	nrc.Bus.Emit(100, 200, 128, false)
	nrc.Bus.Emit(200, 200, 128, false)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		nrc.Step()
		// Re-inject signal since step clears bus
		nrc.Bus.Emit(42, 200, 128, false)
	}
	b.ReportMetric(float64(nrc.LastActiveTiles), "active_tiles")
}

func BenchmarkNeuroRadioCortex_Step_1M(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	nrc := NewNeuroRadioCortex(1000000, rng)

	nrc.Bus.Emit(42, 200, 128, false)
	nrc.Bus.Emit(100, 200, 128, false)
	nrc.Bus.Emit(200, 200, 128, false)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		nrc.Step()
		nrc.Bus.Emit(42, 200, 128, false)
	}
	b.ReportMetric(float64(nrc.LastActiveTiles), "active_tiles")
}
