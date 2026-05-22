package cortex

import (
	"math/bits"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

// ─────────────────────────────────────────────────────────────────────
// NeuroTexture Engine Tests + Benchmarks
// ─────────────────────────────────────────────────────────────────────

func TestPop16Table(t *testing.T) {
	// Verify the precomputed popcount table
	for i := 0; i < 65536; i++ {
		expected := uint8(bits.OnesCount16(uint16(i)))
		if pop16[i] != expected {
			t.Fatalf("pop16[%d] = %d, want %d", i, pop16[i], expected)
		}
	}
}

func TestForwardPopcountMatchesForward(t *testing.T) {
	rng := rand.New(rand.NewSource(42))

	// Create a small layer
	layer := NewTernaryLayer(64, 16)

	// Set random ternary weights
	for j := 0; j < 16; j++ {
		for i := 0; i < 64; i++ {
			v := int8(rng.Intn(3) - 1) // -1, 0, +1
			layer.SetWeight(j, i, v)
		}
	}

	// Create a dense int16 input (binary: 0 or 1)
	input := make([]int16, 64)
	for i := range input {
		if rng.Float32() < 0.1 { // ~10% active
			input[i] = 1
		}
	}

	// Compute with Forward (original)
	expected := layer.Forward(input)

	// Create activeMask from binary input
	activeMask := make([]uint16, layer.TilesPerRow)
	for i, v := range input {
		if v != 0 {
			tileIdx := i / 16
			bitPos := i % 16
			activeMask[tileIdx] |= 1 << uint(bitPos)
		}
	}

	// Compute with ForwardPopcount
	actual := layer.ForwardPopcount(activeMask)

	// Compare
	for j := 0; j < 16; j++ {
		if expected[j] != actual[j] {
			t.Errorf("output[%d]: Forward=%d, ForwardPopcount=%d", j, expected[j], actual[j])
		}
	}
}

func TestNeuroTextureSaveLoad(t *testing.T) {
	rng := rand.New(rand.NewSource(42))

	// Create layer with random weights
	layer := NewTernaryLayer(128, 32)
	for j := 0; j < 32; j++ {
		layer.Bias[j] = int16(rng.Intn(200) - 100)
		for i := 0; i < 128; i++ {
			layer.SetWeight(j, i, int8(rng.Intn(3)-1))
		}
	}

	// Save
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "test.ntx")

	if err := SaveNeuroTexture(path, layer); err != nil {
		t.Fatalf("save: %v", err)
	}

	// Check file exists
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	t.Logf("File size: %d bytes", info.Size())

	// Load
	loaded, err := LoadNeuroTexture(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	// Verify
	if loaded.InputSize != layer.InputSize {
		t.Errorf("InputSize: got %d, want %d", loaded.InputSize, layer.InputSize)
	}
	if loaded.OutputSize != layer.OutputSize {
		t.Errorf("OutputSize: got %d, want %d", loaded.OutputSize, layer.OutputSize)
	}

	// Verify all tiles match
	for i := range layer.Tiles {
		if layer.Tiles[i] != loaded.Tiles[i] {
			t.Errorf("tile[%d]: got %d, want %d", i, loaded.Tiles[i], layer.Tiles[i])
			break
		}
	}

	// Verify biases
	for i := range layer.Bias {
		if layer.Bias[i] != loaded.Bias[i] {
			t.Errorf("bias[%d]: got %d, want %d", i, layer.Bias[i], loaded.Bias[i])
			break
		}
	}
}

func TestNeuroTextureCorruption(t *testing.T) {
	layer := NewTernaryLayer(64, 16)
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "corrupt.ntx")

	if err := SaveNeuroTexture(path, layer); err != nil {
		t.Fatalf("save: %v", err)
	}

	// Corrupt a byte in the tile data
	data, _ := os.ReadFile(path)
	data[ntxHeaderSize+5] ^= 0xFF // flip bits in tile data
	os.WriteFile(path, data, 0600)

	_, err := LoadNeuroTexture(path)
	if err == nil {
		t.Error("expected checksum error, got nil")
	}
	t.Logf("Got expected error: %v", err)
}

func TestSDRCache(t *testing.T) {
	cache := NewSDRCache(100)

	sdr1 := NewSDR(1024)
	sdr1.Set(5)
	sdr1.Set(100)
	sdr1.Set(500)

	output1 := NewSDR(1024)
	output1.Set(10)
	output1.Set(200)

	// Miss
	_, ok := cache.Lookup(sdr1)
	if ok {
		t.Error("expected miss")
	}

	// Store
	cache.Store(sdr1, output1)

	// Hit
	result, ok := cache.Lookup(sdr1)
	if !ok {
		t.Fatal("expected hit")
	}

	// Verify
	if result.ActiveCount != output1.ActiveCount {
		t.Errorf("active count: got %d, want %d", result.ActiveCount, output1.ActiveCount)
	}

	// Stats
	size, hits, misses := cache.Stats()
	if size != 1 || hits != 1 || misses != 1 {
		t.Errorf("stats: size=%d hits=%d misses=%d", size, hits, misses)
	}
}

func TestExpertRouter(t *testing.T) {
	router := NewExpertRouter(8, 1024, 2) // 8 experts, top-2

	// Set distinct embeddings
	for i := 0; i < 8; i++ {
		for j := 0; j < 50; j++ {
			router.ExpertEmbeddings[i].Set(i*100 + j)
		}
	}

	// Input similar to expert 3
	input := NewSDR(1024)
	for j := 0; j < 40; j++ {
		input.Set(300 + j)
	}

	selected := router.Route(input)

	if len(selected) != 2 {
		t.Fatalf("expected 2 experts, got %d", len(selected))
	}

	// Expert 3 should be first
	if selected[0] != 3 {
		t.Errorf("expected expert 3 first, got %d", selected[0])
	}
	t.Logf("Selected experts: %v", selected)
}

func TestPlasticityJournal(t *testing.T) {
	journal := NewPlasticityJournal()

	layer := NewTernaryLayer(64, 8)
	oldTile := layer.Tiles[0]
	newTile := TernaryTile(0xDEADBEEF)

	journal.Record(PlasticityEntry{
		ExpertID:  0,
		LayerName: "feature",
		TileIndex: 0,
		OldTile:   oldTile,
		NewTile:   newTile,
		Reason:    "test",
	})

	if journal.Size() != 1 {
		t.Errorf("expected 1 entry, got %d", journal.Size())
	}

	applied := journal.Merge(layer, 0, "feature")
	if applied != 1 {
		t.Errorf("expected 1 applied, got %d", applied)
	}

	if layer.Tiles[0] != newTile {
		t.Errorf("tile not updated: got %x, want %x", layer.Tiles[0], newTile)
	}

	if journal.Size() != 0 {
		t.Errorf("expected 0 entries after merge, got %d", journal.Size())
	}
}

func TestSDRToActiveMask(t *testing.T) {
	sdr := NewSDR(64)
	sdr.Set(0)  // tile 0, bit 0
	sdr.Set(15) // tile 0, bit 15
	sdr.Set(16) // tile 1, bit 0
	sdr.Set(63) // tile 3, bit 15

	masks := SDRToActiveMask(sdr, 4)

	if masks[0] != (1 | (1 << 15)) {
		t.Errorf("mask[0]: got %016b, want %016b", masks[0], uint16(1|(1<<15)))
	}
	if masks[1] != 1 {
		t.Errorf("mask[1]: got %016b, want %016b", masks[1], uint16(1))
	}
	if masks[3] != (1 << 15) {
		t.Errorf("mask[3]: got %016b, want %016b", masks[3], uint16(1<<15))
	}
}

func TestLayerMemoryReport(t *testing.T) {
	layer := NewTernaryLayer(10000, 10000)
	report := LayerMemoryReport(layer)
	t.Log(report)

	// Verify compression ratio
	totalBytes := layer.MemoryBytes()
	equivFloat32 := 10000 * 10000 * 4
	compression := float64(equivFloat32) / float64(totalBytes)

	if compression < 15 {
		t.Errorf("compression ratio too low: %.1fx (expected ~16x)", compression)
	}
	t.Logf("Compression: %.1fx vs float32", compression)
}

// ─────────────────────────────────────────────────────────────────────
// Benchmarks — RGBA32 vs Dense
// ─────────────────────────────────────────────────────────────────────

func BenchmarkForwardSparse_SDR10k(b *testing.B) {
	// Real-world dimensions: 10k SDR, 50 active bits (0.5%)
	layer := NewTernaryLayer(10000, 10000)
	rng := rand.New(rand.NewSource(42))

	// Set ~33% non-zero weights
	for j := 0; j < 10000; j++ {
		for i := 0; i < 10000; i++ {
			if rng.Float32() < 0.33 {
				layer.SetWeight(j, i, int8(rng.Intn(2)*2-1))
			}
		}
	}

	// Create SDR input
	indices := make([]int, 50)
	values := make([]int16, 50)
	for i := 0; i < 50; i++ {
		indices[i] = rng.Intn(10000)
		values[i] = 1
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		layer.ForwardSparse(indices, values)
	}
}

func BenchmarkForwardPopcount_SDR10k(b *testing.B) {
	// Same dimensions as above, using popcount
	layer := NewTernaryLayer(10000, 10000)
	rng := rand.New(rand.NewSource(42))

	for j := 0; j < 10000; j++ {
		for i := 0; i < 10000; i++ {
			if rng.Float32() < 0.33 {
				layer.SetWeight(j, i, int8(rng.Intn(2)*2-1))
			}
		}
	}

	// Create active mask
	sdr := NewSDR(10000)
	for i := 0; i < 50; i++ {
		sdr.Set(rng.Intn(10000))
	}
	activeMask := SDRToActiveMask(sdr, layer.TilesPerRow)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		layer.ForwardPopcount(activeMask)
	}
}

func BenchmarkForwardDense_int16_10k(b *testing.B) {
	// Dense int16 baseline for comparison
	layer := NewTernaryLayer(10000, 10000)
	rng := rand.New(rand.NewSource(42))

	for j := 0; j < 10000; j++ {
		for i := 0; i < 10000; i++ {
			if rng.Float32() < 0.33 {
				layer.SetWeight(j, i, int8(rng.Intn(2)*2-1))
			}
		}
	}

	input := make([]int16, 10000)
	for i := range input {
		if rng.Float32() < 0.005 {
			input[i] = 1
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		layer.Forward(input)
	}
}

func BenchmarkCacheHit(b *testing.B) {
	cache := NewSDRCache(10000)

	sdr := NewSDR(10000)
	for i := 0; i < 50; i++ {
		sdr.Set(i * 200)
	}
	output := NewSDR(10000)
	for i := 0; i < 50; i++ {
		output.Set(i * 100)
	}
	cache.Store(sdr, output)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache.Lookup(sdr)
	}
}

func BenchmarkExpertRouting_8experts(b *testing.B) {
	router := NewExpertRouter(8, 10000, 2)
	rng := rand.New(rand.NewSource(42))

	for i := 0; i < 8; i++ {
		for j := 0; j < 200; j++ {
			router.ExpertEmbeddings[i].Set(rng.Intn(10000))
		}
	}

	input := NewSDR(10000)
	for i := 0; i < 50; i++ {
		input.Set(rng.Intn(10000))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		router.Route(input)
	}
}
