package cortex

import (
	"fmt"
	"testing"
)

func TestLinearScanStep(t *testing.T) {
	layer := NewLinearScanLayer(256, 256, 128, 64) // 25% decay

	input := NewSDR(256)
	for i := 0; i < 10; i++ {
		input.Set(i * 25)
	}

	// Initialize gate and projection weights so the layer produces output
	for i := 0; i < 128; i++ {
		layer.InputGate.SetWeight(i, i, 1)
		layer.InputProjection.SetWeight(i, i, 1)
		layer.OutputProjection.SetWeight(i, i, 1)
	}

	output := layer.Step(input)
	if output.ActiveCount == 0 {
		t.Error("output should not be empty after step with initialized weights")
	}
}


func TestLinearScanStepFast(t *testing.T) {
	layer := NewLinearScanLayer(256, 256, 128, 32) // Low decay

	input := NewSDR(256)
	for i := 0; i < 10; i++ {
		input.Set(i)
	}

	output := layer.StepFast(input)
	_ = output // Just verify no panic

	// State should have accumulated bits from input
	if layer.StateActiveCount() == 0 {
		t.Error("state should contain input bits after StepFast")
	}
}

func TestLinearScanDecay(t *testing.T) {
	layer := NewLinearScanLayer(256, 256, 128, 255) // Maximum decay

	// Set some state bits manually
	for i := 0; i < 100; i++ {
		layer.State.Set(i)
	}
	layer.State.ActiveCount = 100

	initialActive := layer.StateActiveCount()

	// Apply decay multiple times
	for i := 0; i < 10; i++ {
		layer.decay()
	}

	// With max decay, most bits should be cleared
	if layer.StateActiveCount() >= initialActive {
		t.Errorf("decay should reduce active bits: was %d, now %d",
			initialActive, layer.StateActiveCount())
	}
}

func TestLinearScanReset(t *testing.T) {
	layer := NewLinearScanLayer(256, 256, 128, 64)

	input := NewSDR(256)
	input.Set(0)
	input.Set(1)

	layer.StepFast(input)
	if layer.StateActiveCount() == 0 {
		t.Error("state should be non-empty before reset")
	}

	layer.Reset()
	if layer.StateActiveCount() != 0 {
		t.Errorf("state should be empty after reset, got %d", layer.StateActiveCount())
	}
}

func TestSDRAnd(t *testing.T) {
	a := NewSDR(128)
	b := NewSDR(128)

	a.Set(0)
	a.Set(1)
	a.Set(2)

	b.Set(1)
	b.Set(2)
	b.Set(3)

	result := sdrAnd(a, b)
	if result.ActiveCount != 2 { // bits 1 and 2
		t.Errorf("AND should have 2 active bits, got %d", result.ActiveCount)
	}
	if !result.IsActive(1) || !result.IsActive(2) {
		t.Error("AND should have bits 1 and 2 active")
	}
}

func TestSDRAndNot(t *testing.T) {
	a := NewSDR(128)
	b := NewSDR(128)

	a.Set(0)
	a.Set(1)
	a.Set(2)

	b.Set(1)
	b.Set(2)
	b.Set(3)

	result := sdrAndNot(a, b)
	if result.ActiveCount != 1 { // only bit 0
		t.Errorf("AND-NOT should have 1 active bit, got %d", result.ActiveCount)
	}
	if !result.IsActive(0) {
		t.Error("AND-NOT should have bit 0 active")
	}
}

func TestCortexBlock(t *testing.T) {
	block := NewCortexBlock(256, 100, 3, 64)

	input := NewSDR(256)
	for i := 0; i < 10; i++ {
		input.Set(i * 20)
	}

	// Process 5 tokens
	for i := 0; i < 5; i++ {
		output := block.ProcessToken(input)
		if output.ActiveCount == 0 {
			t.Errorf("token %d: output should not be empty", i)
		}
	}

	params := block.ParameterCount()
	mem := block.MemoryBytes()
	t.Logf("CortexBlock: %d params, %d bytes (%.1f KB)", params, mem, float64(mem)/1024)
}

func TestCortexStack(t *testing.T) {
	stack := NewCortexStack(4, 256, 50, 3, 64) // 4 layers

	input := NewSDR(256)
	for i := 0; i < 10; i++ {
		input.Set(i * 25)
	}

	// Process 10 tokens
	for i := 0; i < 10; i++ {
		output := stack.ProcessToken(input)
		if output.ActiveCount == 0 {
			t.Errorf("token %d: output should not be empty", i)
		}
	}

	t.Logf("CortexStack: %d layers, %d total params, %.1f KB",
		len(stack.Blocks), stack.TotalParameters(),
		float64(stack.TotalMemoryBytes())/1024)
}

func TestCortexStackScaling(t *testing.T) {
	// Test different scales
	configs := []struct {
		layers     int
		dim        int
		contextLen int
	}{
		{4, 256, 100},
		{8, 512, 200},
		{12, 1024, 500},
	}

	for _, cfg := range configs {
		stack := NewCortexStack(cfg.layers, cfg.dim, cfg.contextLen, 5, 64)
		params := stack.TotalParameters()
		memMB := float64(stack.TotalMemoryBytes()) / (1024 * 1024)

		t.Logf("Stack[%d layers × %d dim × %d ctx]: %.2fM params, %.1f MB, compression %.1fx vs float32",
			cfg.layers, cfg.dim, cfg.contextLen,
			float64(params)/1e6, memMB,
			float64(params*4)/(float64(stack.TotalMemoryBytes())))
	}
}

func TestSharedCortexStack(t *testing.T) {
	shared := NewSharedCortexStack(24, 256, 50, 3, 64, nil)

	input := NewSDR(256)
	for i := 0; i < 10; i++ {
		input.Set(i * 25)
	}

	// Process 10 tokens
	for i := 0; i < 10; i++ {
		output := shared.ProcessToken(input)
		if output.ActiveCount == 0 {
			t.Errorf("token %d: output should not be empty", i)
		}
	}

	stored := shared.StoredParameters()
	effective := shared.EffectiveParameters()
	ratio := shared.CompressionRatio()
	storedMB := float64(shared.StoredMemoryBytes()) / (1024 * 1024)

	t.Logf("SharedCortexStack: 24 layers × 256 dim")
	t.Logf("  Stored params:    %d (%.2f MB)", stored, storedMB)
	t.Logf("  Effective params: %d (%.2fM)", effective, float64(effective)/1e6)
	t.Logf("  Compression:      %.1f×", ratio)
}

func TestSharedVsNonSharedComparison(t *testing.T) {
	configs := []struct {
		layers int
		dim    int
	}{
		{12, 256},
		{24, 512},
		{24, 1024},
	}

	for _, cfg := range configs {
		regular := NewCortexStack(cfg.layers, cfg.dim, 100, 3, 64)
		shared := NewSharedCortexStack(cfg.layers, cfg.dim, 100, 3, 64, nil)

		regMem := float64(regular.TotalMemoryBytes()) / (1024 * 1024)
		sharedMem := float64(shared.StoredMemoryBytes()) / (1024 * 1024)
		savings := regMem / sharedMem

		t.Logf("%d layers × %d dim:", cfg.layers, cfg.dim)
		t.Logf("  Regular:  %.2fM params, %.1f MB",
			float64(regular.TotalParameters())/1e6, regMem)
		t.Logf("  Shared:   %.2fM stored, %.2fM effective, %.1f MB (%.1f× smaller)",
			float64(shared.StoredParameters())/1e6,
			float64(shared.EffectiveParameters())/1e6,
			sharedMem, savings)
	}
}

// Benchmarks

func BenchmarkLinearScanStepFast(b *testing.B) {
	layer := NewLinearScanLayer(1024, 1024, 512, 64)

	input := NewSDR(1024)
	for i := 0; i < 50; i++ {
		input.Set(i * 20)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		layer.StepFast(input)
	}
}

func BenchmarkCortexBlock_256(b *testing.B) {
	block := NewCortexBlock(256, 100, 3, 64)

	input := NewSDR(256)
	for i := 0; i < 10; i++ {
		input.Set(i * 25)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		block.ProcessToken(input)
	}
}

func BenchmarkCortexStack_4x256(b *testing.B) {
	stack := NewCortexStack(4, 256, 50, 3, 64)

	input := NewSDR(256)
	for i := 0; i < 10; i++ {
		input.Set(i * 25)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		stack.ProcessToken(input)
	}
}

func BenchmarkCortexStack_12x1024(b *testing.B) {
	stack := NewCortexStack(12, 1024, 100, 5, 64)

	input := NewSDR(1024)
	for i := 0; i < 50; i++ {
		input.Set(i * 20)
	}

	// Pre-fill with some context
	for i := 0; i < 10; i++ {
		stack.ProcessToken(input)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		stack.ProcessToken(input)
	}
}

func BenchmarkCortexStackMemory(b *testing.B) {
	// Measure memory for a production-size stack
	stack := NewCortexStack(12, 1024, 100, 5, 64)

	params := stack.TotalParameters()
	mem := stack.TotalMemoryBytes()
	fmt.Printf("12-layer 1024-dim stack: %dM params, %.1f MB\n",
		params/1_000_000, float64(mem)/(1024*1024))

	input := NewSDR(1024)
	for i := 0; i < 50; i++ {
		input.Set(i * 20)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		stack.ProcessToken(input)
	}
}
