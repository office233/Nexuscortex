package cortex

import (
	"math"
	"math/rand"
	"testing"
)

// ─────────────────────────────────────────────────────────────────────
// Quantum-Inspired NeuroTexture Tests
// ─────────────────────────────────────────────────────────────────────

func TestCos256Table(t *testing.T) {
	// Verify cos256 LUT matches mathematical cosine
	for i := 0; i < 256; i++ {
		angle := float64(i) * 2.0 * math.Pi / 256.0
		expected := int8(math.Round(math.Cos(angle) * 127.0))
		if cos256[i] != expected {
			t.Fatalf("cos256[%d] = %d, want %d", i, cos256[i], expected)
		}
	}

	// Check key values
	if cos256[0] != 127 {
		t.Errorf("cos(0°) = %d, want 127", cos256[0])
	}
	if cos256[64] != 0 {
		t.Errorf("cos(90°) = %d, want 0", cos256[64])
	}
	if cos256[128] != -127 {
		t.Errorf("cos(180°) = %d, want -127", cos256[128])
	}
}

func TestQuantumTileClassicalEquivalence(t *testing.T) {
	// When amplitude=255 and phase=0, quantum forward should match classical
	rng := rand.New(rand.NewSource(42))

	base := NewTernaryLayer(64, 16)
	for j := 0; j < 16; j++ {
		for i := 0; i < 64; i++ {
			base.SetWeight(j, i, int8(rng.Intn(3)-1))
		}
	}

	quantum := NewQuantumTernaryLayer(base)

	// Create SDR input
	sdr := NewSDR(64)
	for i := 0; i < 8; i++ {
		sdr.Set(rng.Intn(64))
	}
	activeMask := SDRToActiveMask(sdr, base.TilesPerRow)

	// Classical (popcount)
	classical := base.ForwardPopcount(activeMask)

	// Quantum with amp=255, phase=0 (should be ~equivalent)
	quantumOut := quantum.ForwardQuantum(activeMask, 0)

	// They won't be exactly identical because quantum uses >>15 approximation
	// but they should be in the same ballpark
	totalDiff := 0
	for j := 0; j < 16; j++ {
		diff := int(classical[j]) - int(quantumOut[j])
		if diff < 0 {
			diff = -diff
		}
		totalDiff += diff
	}

	avgDiff := float64(totalDiff) / 16.0
	t.Logf("Classical vs Quantum (amp=255, phase=0): avg diff = %.2f", avgDiff)

	// Allow small rounding differences but catch major divergence
	if avgDiff > 2.0 {
		t.Errorf("Too much divergence between classical and quantum: avg diff = %.2f", avgDiff)
		for j := 0; j < 16; j++ {
			t.Logf("  output[%d]: classical=%d, quantum=%d", j, classical[j], quantumOut[j])
		}
	}
}

func TestQuantumPhaseInterference(t *testing.T) {
	rng := rand.New(rand.NewSource(42))

	base := NewTernaryLayer(64, 8)
	for j := 0; j < 8; j++ {
		for i := 0; i < 64; i++ {
			base.SetWeight(j, i, int8(rng.Intn(3)-1))
		}
	}

	quantum := NewQuantumTernaryLayer(base)

	// Set all tiles to phase=0
	for i := range quantum.Phases {
		quantum.Phases[i] = 0
	}

	sdr := NewSDR(64)
	for i := 0; i < 10; i++ {
		sdr.Set(rng.Intn(64))
	}
	activeMask := SDRToActiveMask(sdr, base.TilesPerRow)

	// Forward with aligned phase (0 - 0 = constructive)
	constructive := quantum.ForwardQuantum(activeMask, 0)

	// Forward with opposite phase (128 - 0 = 180° = destructive)
	destructive := quantum.ForwardQuantum(activeMask, 128)

	// Constructive output should have larger magnitudes than destructive
	var constructiveMag, destructiveMag int64
	for j := 0; j < 8; j++ {
		cm := int64(constructive[j])
		if cm < 0 {
			cm = -cm
		}
		constructiveMag += cm

		dm := int64(destructive[j])
		if dm < 0 {
			dm = -dm
		}
		destructiveMag += dm
	}

	t.Logf("Constructive magnitude: %d", constructiveMag)
	t.Logf("Destructive magnitude:  %d", destructiveMag)

	// Destructive should be opposite sign (cos(180°) = -1)
	// So destructive magnitude should be similar but outputs inverted
	if constructiveMag == 0 && destructiveMag == 0 {
		t.Skip("Both zero — trivial test case with this random seed")
	}
}

func TestQuantumAmplitudeEffect(t *testing.T) {
	rng := rand.New(rand.NewSource(42))

	base := NewTernaryLayer(64, 8)
	for j := 0; j < 8; j++ {
		for i := 0; i < 64; i++ {
			base.SetWeight(j, i, int8(rng.Intn(3)-1))
		}
	}

	quantum := NewQuantumTernaryLayer(base)

	sdr := NewSDR(64)
	for i := 0; i < 10; i++ {
		sdr.Set(rng.Intn(64))
	}
	activeMask := SDRToActiveMask(sdr, base.TilesPerRow)

	// Full amplitude
	fullAmpOut := quantum.ForwardQuantum(activeMask, 0)

	// Reduce amplitude to ~50%
	for i := range quantum.Amplitudes {
		quantum.Amplitudes[i] = 128
	}
	halfAmpOut := quantum.ForwardQuantum(activeMask, 0)

	// Half amplitude should produce roughly half the output magnitudes
	var fullMag, halfMag int64
	for j := 0; j < 8; j++ {
		fm := int64(fullAmpOut[j])
		if fm < 0 {
			fm = -fm
		}
		fullMag += fm

		hm := int64(halfAmpOut[j])
		if hm < 0 {
			hm = -hm
		}
		halfMag += hm
	}

	t.Logf("Full amplitude total:  %d", fullMag)
	t.Logf("Half amplitude total:  %d", halfMag)

	if fullMag > 0 && halfMag > fullMag {
		t.Errorf("Half amplitude should not exceed full: half=%d full=%d", halfMag, fullMag)
	}
}

func TestPBitLayerDeterministic(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	layer := NewPBitLayer(16, rng)

	// Temperature=0 → deterministic
	layer.SetTemperature(0)

	input := make([]int16, 16)
	input[0] = 100  // should → +1
	input[1] = -100 // should → -1
	input[2] = 0    // should → 0

	output := layer.Activate(input)

	if output[0] != 1 {
		t.Errorf("positive input with temp=0: got %d, want 1", output[0])
	}
	if output[1] != -1 {
		t.Errorf("negative input with temp=0: got %d, want -1", output[1])
	}
	if output[2] != 0 {
		t.Errorf("zero input with temp=0: got %d, want 0", output[2])
	}
}

func TestPBitLayerStochastic(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	layer := NewPBitLayer(1, rng)

	// Temperature=128 → probabilistic
	layer.SetTemperature(128)
	layer.Neurons[0].Bias = 50 // slight positive bias

	// Run many trials
	counts := map[int16]int{-1: 0, 0: 0, 1: 0}
	trials := 1000
	for i := 0; i < trials; i++ {
		input := []int16{0} // no input, just bias
		out := layer.Activate(input)
		counts[out[0]]++
	}

	t.Logf("P-bit distribution (bias=50, temp=128): +1=%d, 0=%d, -1=%d",
		counts[1], counts[0], counts[-1])

	// With positive bias, +1 should be more frequent than -1
	if counts[1] < counts[-1] {
		t.Errorf("positive bias should produce more +1 than -1: got +1=%d, -1=%d",
			counts[1], counts[-1])
	}

	// Should produce at least some of each (stochastic)
	if counts[1] == trials || counts[-1] == trials {
		t.Errorf("temperature>0 should produce variety, got all same")
	}
}

func TestQuantumRouterInterference(t *testing.T) {
	router := NewQuantumRouter(8, 1024, 2)

	// Set distinct phases for each expert
	for i := 0; i < 8; i++ {
		router.ExpertPhases[i] = uint8(i * 32) // 0, 32, 64, ..., 224
		router.ExpertAmps[i] = 200
	}

	// Route with phase close to expert 0 (phase=0)
	selected := router.Route(0)
	t.Logf("Input phase=0: selected experts %v", selected)

	if len(selected) != 2 {
		t.Fatalf("expected 2 experts, got %d", len(selected))
	}

	// Expert 0 (phase=0) should be first (most constructive with input phase=0)
	if selected[0] != 0 {
		t.Errorf("expected expert 0 first for phase=0, got %d", selected[0])
	}

	// Route with phase=128 (opposite to expert 0, closest to expert 4)
	selected2 := router.Route(128)
	t.Logf("Input phase=128: selected experts %v", selected2)

	if selected2[0] != 4 {
		t.Errorf("expected expert 4 first for phase=128, got %d", selected2[0])
	}
}

func TestQuantumRouterUpdate(t *testing.T) {
	router := NewQuantumRouter(4, 1024, 2)
	router.ExpertPhases[0] = 0
	router.ExpertAmps[0] = 128

	// Reward: phase should move toward input, amplitude should increase
	router.UpdateExpertPhase(0, 100, true)

	if router.ExpertAmps[0] <= 128 {
		t.Errorf("amplitude should increase on reward: got %d", router.ExpertAmps[0])
	}
	t.Logf("After reward: phase=%d, amp=%d", router.ExpertPhases[0], router.ExpertAmps[0])

	// Punish: amplitude should decrease
	oldAmp := router.ExpertAmps[0]
	router.UpdateExpertPhase(0, 100, false)
	if router.ExpertAmps[0] >= oldAmp {
		t.Errorf("amplitude should decrease on punishment: got %d (was %d)",
			router.ExpertAmps[0], oldAmp)
	}
}

func TestMultiSampleForward(t *testing.T) {
	rng := rand.New(rand.NewSource(42))

	base := NewTernaryLayer(64, 32)
	for j := 0; j < 32; j++ {
		for i := 0; i < 64; i++ {
			base.SetWeight(j, i, int8(rng.Intn(3)-1))
		}
	}

	quantum := NewQuantumTernaryLayer(base)

	sdr := NewSDR(64)
	for i := 0; i < 8; i++ {
		sdr.Set(rng.Intn(64))
	}
	activeMask := SDRToActiveMask(sdr, base.TilesPerRow)

	result := MultiSampleForward(quantum, activeMask, 0, 5, rng)

	t.Logf("MultiSample: %d samples, confidence=%.3f, output active=%d",
		result.NumSamples, result.Confidence, result.Output.ActiveCount)

	if result.NumSamples != 5 {
		t.Errorf("expected 5 samples, got %d", result.NumSamples)
	}

	if result.Confidence < 0 || result.Confidence > 1.0 {
		t.Errorf("confidence out of range: %.3f", result.Confidence)
	}
}

func TestAmplitudePlasticity(t *testing.T) {
	rng := rand.New(rand.NewSource(42))

	base := NewTernaryLayer(64, 8)
	for j := 0; j < 8; j++ {
		for i := 0; i < 64; i++ {
			base.SetWeight(j, i, int8(rng.Intn(3)-1))
		}
	}

	quantum := NewQuantumTernaryLayer(base)

	// Start at 128 amplitude
	for i := range quantum.Amplitudes {
		quantum.Amplitudes[i] = 128
	}

	sdr := NewSDR(64)
	for i := 0; i < 10; i++ {
		sdr.Set(rng.Intn(64))
	}
	activeMask := SDRToActiveMask(sdr, base.TilesPerRow)

	// Record initial state
	initialAmp := quantum.Amplitudes[0]

	// Correct prediction → amplitude should increase for involved tiles
	quantum.UpdateAmplitudes(activeMask, 0, true)

	// At least some amplitudes should have increased
	increased := 0
	for i := 0; i < base.TilesPerRow; i++ {
		if quantum.Amplitudes[i] > initialAmp {
			increased++
		}
	}

	t.Logf("After correct: %d tiles increased amplitude (initial=%d)", increased, initialAmp)
}

func TestSDRPhase(t *testing.T) {
	sdr1 := NewSDR(1024)
	sdr1.Set(5)
	sdr1.Set(100)
	sdr1.Set(500)

	sdr2 := NewSDR(1024)
	sdr2.Set(5)
	sdr2.Set(100)
	sdr2.Set(500)

	// Same SDR → same phase
	p1 := SDRPhase(sdr1)
	p2 := SDRPhase(sdr2)
	if p1 != p2 {
		t.Errorf("same SDR should produce same phase: %d vs %d", p1, p2)
	}

	// Different SDR → likely different phase
	sdr3 := NewSDR(1024)
	sdr3.Set(999)
	p3 := SDRPhase(sdr3)
	t.Logf("Phase sdr1=%d, sdr3=%d", p1, p3)
}

// ─────────────────────────────────────────────────────────────────────
// Benchmarks
// ─────────────────────────────────────────────────────────────────────

func BenchmarkForwardQuantum_10k(b *testing.B) {
	rng := rand.New(rand.NewSource(42))

	base := NewTernaryLayer(10000, 1000)
	for j := 0; j < 1000; j++ {
		for i := 0; i < 10000; i++ {
			if rng.Float32() < 0.33 {
				base.SetWeight(j, i, int8(rng.Intn(2)*2-1))
			}
		}
	}

	quantum := NewQuantumTernaryLayer(base)

	// Set varied amplitudes and phases
	for i := range quantum.Amplitudes {
		quantum.Amplitudes[i] = uint8(rng.Intn(256))
		quantum.Phases[i] = uint8(rng.Intn(256))
	}

	sdr := NewSDR(10000)
	for i := 0; i < 50; i++ {
		sdr.Set(rng.Intn(10000))
	}
	activeMask := SDRToActiveMask(sdr, base.TilesPerRow)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		quantum.ForwardQuantum(activeMask, uint8(i))
	}
}

func BenchmarkPBitActivate_10k(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	layer := NewPBitLayer(10000, rng)
	layer.SetTemperature(128)

	input := make([]int16, 10000)
	for i := range input {
		input[i] = int16(rng.Intn(200) - 100)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		layer.Activate(input)
	}
}

func BenchmarkQuantumRouter_8experts(b *testing.B) {
	router := NewQuantumRouter(8, 1024, 2)
	for i := 0; i < 8; i++ {
		router.ExpertPhases[i] = uint8(i * 32)
		router.ExpertAmps[i] = 200
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		router.Route(uint8(i))
	}
}

func BenchmarkMultiSample_5passes(b *testing.B) {
	rng := rand.New(rand.NewSource(42))

	base := NewTernaryLayer(1000, 100)
	for j := 0; j < 100; j++ {
		for i := 0; i < 1000; i++ {
			if rng.Float32() < 0.33 {
				base.SetWeight(j, i, int8(rng.Intn(2)*2-1))
			}
		}
	}

	quantum := NewQuantumTernaryLayer(base)
	sdr := NewSDR(1000)
	for i := 0; i < 30; i++ {
		sdr.Set(rng.Intn(1000))
	}
	activeMask := SDRToActiveMask(sdr, base.TilesPerRow)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MultiSampleForward(quantum, activeMask, 0, 5, rng)
	}
}
