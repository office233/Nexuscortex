package compute

import (
	"math/bits"
	"math/rand"
	"testing"
)

// TestForwardSparseConsistency verifies that CUDAEngine (when available)
// produces identical results to CPUEngine for ForwardSparse.
// This is the CRITICAL correctness test — if this fails, CUDA training
// is producing wrong gradients silently.
func TestForwardSparseConsistency(t *testing.T) {
	cpu := NewCPUEngine()

	// Build a small ternary layer: 32 inputs, 16 outputs
	inputSize := 32
	outputSize := 16
	tilesPerRow := (inputSize + 15) / 16 // = 2

	rng := rand.New(rand.NewSource(42))

	// Random tiles (ternary weights)
	tiles := make([]uint32, outputSize*tilesPerRow)
	for i := range tiles {
		tiles[i] = rng.Uint32()
	}

	// Random bias
	bias := make([]int16, outputSize)
	for i := range bias {
		bias[i] = int16(rng.Intn(200) - 100)
	}

	// Sparse active inputs: 8 out of 32
	activeIndices := []uint32{2, 5, 8, 13, 17, 21, 26, 30}
	activeValues := []int16{100, -50, 200, 75, -120, 90, 60, -80}

	// CPU result
	cpuOut, err := cpu.ForwardSparse(activeIndices, activeValues, tiles, bias, tilesPerRow, outputSize)
	if err != nil {
		t.Fatalf("CPU ForwardSparse failed: %v", err)
	}

	// Try CUDA
	cuda := NewCUDAEngine()
	if err := cuda.Init(); err != nil {
		t.Skipf("CUDA not available, skipping GPU comparison: %v", err)
		return
	}
	defer cuda.Close()

	cudaOut, err := cuda.ForwardSparse(activeIndices, activeValues, tiles, bias, tilesPerRow, outputSize)
	if err != nil {
		t.Fatalf("CUDA ForwardSparse failed: %v", err)
	}

	// Compare outputs
	if len(cpuOut) != len(cudaOut) {
		t.Fatalf("output length mismatch: CPU=%d CUDA=%d", len(cpuOut), len(cudaOut))
	}

	mismatches := 0
	for i := range cpuOut {
		if cpuOut[i] != cudaOut[i] {
			t.Errorf("output[%d] mismatch: CPU=%d CUDA=%d", i, cpuOut[i], cudaOut[i])
			mismatches++
		}
	}

	if mismatches == 0 {
		t.Logf("✅ CUDA matches CPU perfectly (%d outputs verified)", outputSize)
	} else {
		t.Errorf("❌ %d/%d outputs differ between CUDA and CPU", mismatches, outputSize)
	}
}

// TestForwardSparseLarger tests with a larger layer to stress-test the kernel.
func TestForwardSparseLarger(t *testing.T) {
	cpu := NewCPUEngine()

	inputSize := 256
	outputSize := 128
	tilesPerRow := (inputSize + 15) / 16

	rng := rand.New(rand.NewSource(99))
	tiles := make([]uint32, outputSize*tilesPerRow)
	for i := range tiles {
		tiles[i] = rng.Uint32()
	}
	bias := make([]int16, outputSize)
	for i := range bias {
		bias[i] = int16(rng.Intn(100) - 50)
	}

	// 32 active inputs
	activeIndices := make([]uint32, 32)
	activeValues := make([]int16, 32)
	for i := range activeIndices {
		activeIndices[i] = uint32(rng.Intn(inputSize))
		activeValues[i] = int16(rng.Intn(400) - 200)
	}

	cpuOut, err := cpu.ForwardSparse(activeIndices, activeValues, tiles, bias, tilesPerRow, outputSize)
	if err != nil {
		t.Fatalf("CPU failed: %v", err)
	}

	cuda := NewCUDAEngine()
	if err := cuda.Init(); err != nil {
		t.Skipf("CUDA not available: %v", err)
		return
	}
	defer cuda.Close()

	cudaOut, err := cuda.ForwardSparse(activeIndices, activeValues, tiles, bias, tilesPerRow, outputSize)
	if err != nil {
		t.Fatalf("CUDA failed: %v", err)
	}

	mismatches := 0
	for i := range cpuOut {
		if cpuOut[i] != cudaOut[i] {
			t.Errorf("output[%d] mismatch: CPU=%d CUDA=%d", i, cpuOut[i], cudaOut[i])
			mismatches++
		}
	}
	if mismatches == 0 {
		t.Logf("✅ CUDA matches CPU: 128 outputs, 32 active inputs, 256 input dims")
	}
}

// TestBatchSDRSimilarityCUDA tests the CUDA bitmask-based popcount SDR similarity.
// NOTE: CPUEngine.BatchSDRSimilarity uses a SET-based overlap (different semantics).
// The CUDA kernel uses BITWISE AND + popcount, which is correct for SDR bitmasks.
// This test verifies CUDA against a manual Go popcount reference.
func TestBatchSDRSimilarityCUDA(t *testing.T) {
	rng := rand.New(rand.NewSource(123))

	queryWords := 8 // 256 bits
	query := make([]uint32, queryWords)
	for i := range query {
		query[i] = rng.Uint32()
	}

	numMem := 50
	memories := make([][]uint32, numMem)
	for i := range memories {
		mem := make([]uint32, queryWords)
		for j := range mem {
			mem[j] = rng.Uint32()
		}
		memories[i] = mem
	}

	// Reference: manual popcount(query & memory)
	expected := make([]uint8, numMem)
	for i, mem := range memories {
		overlap := 0
		for w := 0; w < queryWords; w++ {
			overlap += bits.OnesCount32(query[w] & mem[w])
		}
		if overlap > 255 {
			overlap = 255
		}
		expected[i] = uint8(overlap)
	}

	cuda := NewCUDAEngine()
	if err := cuda.Init(); err != nil {
		t.Skipf("CUDA not available: %v", err)
		return
	}
	defer cuda.Close()

	cudaOut, err := cuda.BatchSDRSimilarity(query, memories)
	if err != nil {
		t.Fatalf("CUDA BatchSDRSimilarity failed: %v", err)
	}

	mismatches := 0
	for i := range expected {
		if cudaOut[i] != expected[i] {
			t.Errorf("memory[%d] mismatch: expected=%d CUDA=%d", i, expected[i], cudaOut[i])
			mismatches++
		}
	}
	if mismatches == 0 {
		t.Logf("✅ CUDA BatchSDR matches Go popcount reference (%d memories verified)", numMem)
	}
}

// TestForwardSparseEmpty verifies edge case with no active inputs.
func TestForwardSparseEmpty(t *testing.T) {
	cpu := NewCPUEngine()

	bias := []int16{10, -20, 30}
	out, err := cpu.ForwardSparse(nil, nil, nil, bias, 1, 3)
	if err != nil {
		t.Fatalf("empty ForwardSparse failed: %v", err)
	}

	for i, v := range out {
		if v != bias[i] {
			t.Errorf("output[%d]=%d, expected bias=%d", i, v, bias[i])
		}
	}
	t.Logf("✅ Empty inputs correctly return bias values")
}
