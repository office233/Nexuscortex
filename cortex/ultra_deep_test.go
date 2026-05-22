package cortex

import (
	"testing"
)

func TestUltraDeepStack5T(t *testing.T) {
	// Config for 5T effective parameters:
	// dim=10000, hidden=10000 → stored = 10000*10000*2 + 3*10000*10000 = 500M
	// × 1000 layers = 500B effective... need bigger dims for 5T
	//
	// For 5T: stored * layers = 5T
	// If layers=1000: stored = 5B → dim ~= sqrt(5B/5) ≈ 31623
	// But that's too big for a test. Let's test at small scale.
	
	// Small test: 100 layers × 256 dim × 256 hidden
	stack := NewUltraDeepStack(100, 256, 256, 32)

	input := NewSDR(256)
	for i := 0; i < 13; i++ { // ~5% sparsity
		input.Set(i * 19)
	}

	// Process a token
	output := stack.ProcessToken(input)
	if output.ActiveCount == 0 {
		t.Error("output should not be empty")
	}

	stored := stack.StoredParameters()
	effective := stack.EffectiveParameters()

	t.Logf("Small test: %s", stack.Stats())
	t.Logf("  Stored:    %d (%.2fM)", stored, float64(stored)/1e6)
	t.Logf("  Effective: %d (%.2fM)", effective, float64(effective)/1e6)
	t.Logf("  Ratio:     %d×", stack.NumLayers)
}

func TestUltraDeepStackScaling(t *testing.T) {
	// Test memory calculations at various scales
	configs := []struct {
		layers    int
		dim       int
		hiddenDim int
		name      string
	}{
		{100, 256, 256, "Test"},
		{100, 1024, 1024, "Small"},
		{500, 2048, 2048, "Medium"},
		{1000, 4096, 4096, "Large"},
	}

	for _, cfg := range configs {
		// Don't actually create the stack for huge configs — just calculate
		storedParams := cfg.dim*cfg.hiddenDim + cfg.hiddenDim*cfg.dim + 3*cfg.dim*cfg.dim
		effectiveParams := int64(storedParams) * int64(cfg.layers)
		memBytes := storedParams / 16 * 4 // ternary RGBA32

		t.Logf("[%s] %d layers × %d dim × %d hidden:", cfg.name, cfg.layers, cfg.dim, cfg.hiddenDim)
		t.Logf("  Stored:    %.2fM params, %.1f MB", float64(storedParams)/1e6, float64(memBytes)/(1024*1024))
		t.Logf("  Effective: %.2fB params (%.2fT)", float64(effectiveParams)/1e9, float64(effectiveParams)/1e12)
		t.Logf("  Compression: %d×", cfg.layers)
	}

	// Show the EXACT config needed for 5T
	// 5T / 1000 layers = 5B stored params per layer set
	// storedParams = 2*dim*hiddenDim + 3*dim*dim
	// For dim=hiddenDim: storedParams = 5*dim²
	// 5*dim² = 5B → dim² = 1B → dim = 31623
	targetDim := 31623
	storedParams := 5 * targetDim * targetDim
	effectiveParams := int64(storedParams) * 1000
	memBytes := storedParams / 16 * 4

	t.Logf("\n[5T CONFIG] 1000 layers × %d dim:", targetDim)
	t.Logf("  Stored:    %.2fB params, %.1f MB", float64(storedParams)/1e9, float64(memBytes)/(1024*1024))
	t.Logf("  Effective: %.2fT params", float64(effectiveParams)/1e12)
	t.Logf("  VRAM needed: %.1f MB ← FITS IN 6 GB!", float64(memBytes)/(1024*1024))
}

func BenchmarkUltraDeepStack_100layers(b *testing.B) {
	stack := NewUltraDeepStack(100, 512, 512, 32)

	input := NewSDR(512)
	for i := 0; i < 25; i++ { // 5% sparsity
		input.Set(i * 20)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		stack.ProcessToken(input)
	}
}

func BenchmarkUltraDeepStack_1000layers(b *testing.B) {
	stack := NewUltraDeepStack(1000, 512, 512, 32)

	input := NewSDR(512)
	for i := 0; i < 25; i++ { // 5% sparsity
		input.Set(i * 20)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		stack.ProcessToken(input)
	}
}
