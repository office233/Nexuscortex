package cortex

import (
	"testing"
)

func TestPackUnpackTernaryTile(t *testing.T) {
	// Test all combinations of -1, 0, +1
	weights := [16]int8{1, -1, 0, 1, -1, -1, 0, 1, 0, 0, 1, -1, 1, 0, -1, 1}
	tile := PackTernaryTile(weights)
	got := tile.Unpack()

	for i := 0; i < 16; i++ {
		if got[i] != weights[i] {
			t.Errorf("position %d: want %d, got %d", i, weights[i], got[i])
		}
	}
}

func TestTernaryTileAllZero(t *testing.T) {
	weights := [16]int8{}
	tile := PackTernaryTile(weights)

	if tile.ActiveCount() != 0 {
		t.Errorf("all-zero tile should have 0 active, got %d", tile.ActiveCount())
	}
	if tile.Sparsity() != 255 {
		t.Errorf("all-zero tile should have max sparsity 255, got %d", tile.Sparsity())
	}
}

func TestTernaryTileAllPositive(t *testing.T) {
	var weights [16]int8
	for i := range weights {
		weights[i] = 1
	}
	tile := PackTernaryTile(weights)

	if tile.ActiveCount() != 16 {
		t.Errorf("all-positive tile should have 16 active, got %d", tile.ActiveCount())
	}
	if tile.Sparsity() != 0 {
		t.Errorf("all-positive tile should have sparsity 0, got %d", tile.Sparsity())
	}
}

func TestTernaryTileRGBA32Roundtrip(t *testing.T) {
	weights := [16]int8{1, -1, 0, 1, -1, -1, 0, 1, 0, 0, 1, -1, 1, 0, -1, 1}
	tile := PackTernaryTile(weights)

	bytes := tile.RGBA32Bytes()
	restored := TernaryTileFromRGBA32(bytes)

	if tile != restored {
		t.Errorf("RGBA32 roundtrip failed: original %08x, restored %08x", uint32(tile), uint32(restored))
	}
}

func TestTernaryLayerForward(t *testing.T) {
	// 4 inputs → 2 outputs
	layer := NewTernaryLayer(4, 2)

	// Output 0: weights = [+1, -1, +1, 0]
	layer.SetWeight(0, 0, 1)
	layer.SetWeight(0, 1, -1)
	layer.SetWeight(0, 2, 1)
	layer.SetWeight(0, 3, 0)

	// Output 1: weights = [0, +1, -1, +1]
	layer.SetWeight(1, 0, 0)
	layer.SetWeight(1, 1, 1)
	layer.SetWeight(1, 2, -1)
	layer.SetWeight(1, 3, 1)

	input := []int16{10, 20, 30, 40}
	output := layer.Forward(input)

	// Output 0 = 10 - 20 + 30 + 0 = 20
	if output[0] != 20 {
		t.Errorf("output[0]: want 20, got %d", output[0])
	}

	// Output 1 = 0 + 20 - 30 + 40 = 30
	if output[1] != 30 {
		t.Errorf("output[1]: want 30, got %d", output[1])
	}
}

func TestTernaryLayerForwardSparse(t *testing.T) {
	// Same setup as dense test but using sparse interface
	layer := NewTernaryLayer(4, 2)

	layer.SetWeight(0, 0, 1)
	layer.SetWeight(0, 1, -1)
	layer.SetWeight(0, 2, 1)
	layer.SetWeight(0, 3, 0)

	layer.SetWeight(1, 0, 0)
	layer.SetWeight(1, 1, 1)
	layer.SetWeight(1, 2, -1)
	layer.SetWeight(1, 3, 1)

	// Only indices 0 and 2 are active
	indices := []int{0, 2}
	values := []int16{10, 30}
	output := layer.ForwardSparse(indices, values)

	// Output 0 = 10*1 + 30*1 = 40
	if output[0] != 40 {
		t.Errorf("sparse output[0]: want 40, got %d", output[0])
	}

	// Output 1 = 10*0 + 30*(-1) = -30
	if output[1] != -30 {
		t.Errorf("sparse output[1]: want -30, got %d", output[1])
	}
}

func TestTernaryLayerMarshalRoundtrip(t *testing.T) {
	layer := NewTernaryLayer(32, 4)

	// Set some weights
	layer.SetWeight(0, 0, 1)
	layer.SetWeight(0, 15, -1)
	layer.SetWeight(1, 16, 1)
	layer.SetWeight(2, 31, -1)
	layer.SetWeight(3, 0, 1)
	layer.SetWeight(3, 31, 1)
	layer.Bias[0] = 5
	layer.Bias[3] = -10

	data := layer.MarshalRGBA32()
	restored, err := UnmarshalTernaryLayer(data)
	if err != nil {
		t.Fatalf("unmarshal failed: %v", err)
	}

	if restored.InputSize != layer.InputSize || restored.OutputSize != layer.OutputSize {
		t.Errorf("dimensions mismatch: %dx%d vs %dx%d",
			layer.InputSize, layer.OutputSize, restored.InputSize, restored.OutputSize)
	}

	// Verify all weights match
	for o := 0; o < layer.OutputSize; o++ {
		for i := 0; i < layer.InputSize; i++ {
			if layer.GetWeight(o, i) != restored.GetWeight(o, i) {
				t.Errorf("weight[%d][%d] mismatch: %d vs %d",
					o, i, layer.GetWeight(o, i), restored.GetWeight(o, i))
			}
		}
	}

	// Verify biases
	for i := range layer.Bias {
		if layer.Bias[i] != restored.Bias[i] {
			t.Errorf("bias[%d] mismatch: %d vs %d", i, layer.Bias[i], restored.Bias[i])
		}
	}
}

func TestTernaryLayerMemoryEfficiency(t *testing.T) {
	// 1024×512 layer
	layer := NewTernaryLayer(1024, 512)

	params := layer.ParameterCount()
	mem := layer.MemoryBytes()
	float32Equivalent := params * 4

	t.Logf("Layer: %s", layer.Stats())
	t.Logf("Params: %d, Memory: %d bytes, Float32 equivalent: %d bytes", params, mem, float32Equivalent)
	t.Logf("Compression ratio: %.1fx", float64(float32Equivalent)/float64(mem))

	// Ternary should be at least 10x smaller than float32
	ratio := float64(float32Equivalent) / float64(mem)
	if ratio < 10 {
		t.Errorf("compression ratio %.1fx is less than expected 10x", ratio)
	}
}

func TestTernaryLayerOverflowClamp(t *testing.T) {
	// Test that accumulation doesn't overflow int16
	layer := NewTernaryLayer(1024, 1)

	// Set all weights to +1
	for i := 0; i < 1024; i++ {
		layer.SetWeight(0, i, 1)
	}

	// Input all max positive
	input := make([]int16, 1024)
	for i := range input {
		input[i] = 32
	}

	output := layer.Forward(input)

	// Should clamp to 32767 (max int16)
	if output[0] != 32767 {
		t.Errorf("overflow clamp: want 32767, got %d", output[0])
	}
}

// Benchmarks

func BenchmarkTernaryForward_256x256(b *testing.B) {
	layer := NewTernaryLayer(256, 256)
	// Set ~33% of weights to non-zero (realistic sparsity)
	for o := 0; o < 256; o++ {
		for i := 0; i < 256; i += 3 {
			if i%2 == 0 {
				layer.SetWeight(o, i, 1)
			} else {
				layer.SetWeight(o, i, -1)
			}
		}
	}
	input := make([]int16, 256)
	for i := range input {
		input[i] = int16(i)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		layer.Forward(input)
	}
}

func BenchmarkTernaryForward_1024x1024(b *testing.B) {
	layer := NewTernaryLayer(1024, 1024)
	for o := 0; o < 1024; o++ {
		for i := 0; i < 1024; i += 3 {
			layer.SetWeight(o, i, 1)
		}
	}
	input := make([]int16, 1024)
	for i := range input {
		input[i] = int16(i % 100)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		layer.Forward(input)
	}
}

func BenchmarkTernarySparseForward_1024x1024(b *testing.B) {
	layer := NewTernaryLayer(1024, 1024)
	for o := 0; o < 1024; o++ {
		for i := 0; i < 1024; i += 3 {
			layer.SetWeight(o, i, 1)
		}
	}

	// Only 50 active inputs (5% sparsity — like SDR)
	indices := make([]int, 50)
	values := make([]int16, 50)
	for i := range indices {
		indices[i] = i * 20
		values[i] = int16(i + 1)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		layer.ForwardSparse(indices, values)
	}
}
