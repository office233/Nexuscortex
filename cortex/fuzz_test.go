package cortex

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// ---------------------------------------------------------------------------
// FuzzUnmarshalTernaryLayer — fuzz the binary .nxt1 deserializer
// ---------------------------------------------------------------------------

func FuzzUnmarshalTernaryLayer(f *testing.F) {
	// Seed 1: valid small layer (inputSize=16, outputSize=1)
	// Format: [magic:4][inputSize:4][outputSize:4][biases:2*outputSize][tiles:4*tilesPerRow*outputSize]
	// tilesPerRow = ceil(16/16) = 1, so total = 12 + 2 + 4 = 18 bytes
	valid := make([]byte, 18)
	copy(valid[0:4], []byte("NXT1"))
	binary.LittleEndian.PutUint32(valid[4:8], 16)
	binary.LittleEndian.PutUint32(valid[8:12], 1)
	binary.LittleEndian.PutUint16(valid[12:14], 0) // bias = 0
	binary.LittleEndian.PutUint32(valid[14:18], 0) // one tile, all zeros
	f.Add(valid)

	// Seed 2: valid 32×2 layer
	// tilesPerRow = 2, biases = 2*2 = 4, tiles = 2*2*4 = 16
	// total = 12 + 4 + 16 = 32
	valid2 := make([]byte, 32)
	copy(valid2[0:4], []byte("NXT1"))
	binary.LittleEndian.PutUint32(valid2[4:8], 32)
	binary.LittleEndian.PutUint32(valid2[8:12], 2)
	// biases and tiles left as zeroes
	f.Add(valid2)

	// Seed 3: too short — only header
	f.Add([]byte("NXT1\x10\x00\x00\x00\x01\x00\x00\x00"))

	// Seed 4: wrong magic
	bad := make([]byte, 18)
	copy(bad[0:4], []byte("BAD!"))
	binary.LittleEndian.PutUint32(bad[4:8], 16)
	binary.LittleEndian.PutUint32(bad[8:12], 1)
	f.Add(bad)

	// Seed 5: empty
	f.Add([]byte{})

	// Seed 6: single byte
	f.Add([]byte{0x42})

	// Seed 7: huge dimensions that would overflow
	overflow := make([]byte, 12)
	copy(overflow[0:4], []byte("NXT1"))
	binary.LittleEndian.PutUint32(overflow[4:8], 0xFFFFFFFF)
	binary.LittleEndian.PutUint32(overflow[8:12], 0xFFFFFFFF)
	f.Add(overflow)

	// Seed 8: zero dimensions
	zeroDim := make([]byte, 12)
	copy(zeroDim[0:4], []byte("NXT1"))
	binary.LittleEndian.PutUint32(zeroDim[4:8], 0)
	binary.LittleEndian.PutUint32(zeroDim[8:12], 0)
	f.Add(zeroDim)

	// Seed 9: valid header but truncated body
	trunc := make([]byte, 14)
	copy(trunc[0:4], []byte("NXT1"))
	binary.LittleEndian.PutUint32(trunc[4:8], 16)
	binary.LittleEndian.PutUint32(trunc[8:12], 1)
	trunc[12] = 0xFF
	trunc[13] = 0xFF
	f.Add(trunc)

	f.Fuzz(func(t *testing.T, data []byte) {
		// Must not panic — errors are expected and acceptable.
		layer, err := UnmarshalTernaryLayer(data)
		if err != nil {
			return // error is fine
		}
		// If parsing succeeds, do a basic sanity check.
		if layer == nil {
			t.Fatal("UnmarshalTernaryLayer returned nil layer with nil error")
		}
		if layer.InputSize <= 0 || layer.OutputSize <= 0 {
			t.Fatalf("parsed layer has invalid dimensions: input=%d output=%d",
				layer.InputSize, layer.OutputSize)
		}
		// Exercise the layer to catch latent panics in dependent code.
		_ = layer.Stats()
		_ = layer.SparsityRatio()
		_ = layer.MemoryBytes()
	})
}

// ---------------------------------------------------------------------------
// FuzzLoadSemanticMemory — fuzz the semantic.json loader
// ---------------------------------------------------------------------------

func FuzzLoadSemanticMemory(f *testing.F) {
	// Seed 1: valid minimal JSON
	validJSON := `{"sdr_size":256,"concepts":[]}`
	f.Add([]byte(validJSON))

	// Seed 2: valid JSON with one concept
	withConcept := `{
		"sdr_size": 128,
		"concepts": [{
			"active_indices": [0, 5, 10, 50],
			"count": 3,
			"contexts": ["hello", "world"]
		}]
	}`
	f.Add([]byte(withConcept))

	// Seed 3: empty JSON object
	f.Add([]byte(`{}`))

	// Seed 4: empty
	f.Add([]byte{})

	// Seed 5: not JSON at all
	f.Add([]byte("this is not json!!!"))

	// Seed 6: JSON array instead of object
	f.Add([]byte(`[1,2,3]`))

	// Seed 7: negative sdr_size
	f.Add([]byte(`{"sdr_size":-1,"concepts":[]}`))

	// Seed 8: huge sdr_size
	f.Add([]byte(`{"sdr_size":999999999,"concepts":[]}`))

	// Seed 9: concept with out-of-range indices
	f.Add([]byte(`{"sdr_size":10,"concepts":[{"active_indices":[999999],"count":1,"contexts":[]}]}`))

	// Seed 10: concept with negative index
	f.Add([]byte(`{"sdr_size":100,"concepts":[{"active_indices":[-5],"count":1,"contexts":[]}]}`))

	// Seed 11: null values
	f.Add([]byte(`{"sdr_size":null,"concepts":null}`))

	// Seed 12: string instead of number
	f.Add([]byte(`{"sdr_size":"not_a_number","concepts":[]}`))

	// Seed 13: deeply nested/complex
	f.Add([]byte(`{"sdr_size":64,"concepts":[{"active_indices":[],"count":0,"contexts":["a","b","c","d","e"]}]}`))

	f.Fuzz(func(t *testing.T, data []byte) {
		// Write the fuzzed data to a temp file, then try to load it.
		tmpDir := t.TempDir()
		path := filepath.Join(tmpDir, "semantic.json")
		if err := os.WriteFile(path, data, 0600); err != nil {
			t.Fatalf("failed to write temp file: %v", err)
		}

		// Must not panic — errors are expected and acceptable.
		sm, err := LoadSemanticMemory(path, 256)
		if err != nil {
			return // error is fine
		}
		if sm == nil {
			t.Fatal("LoadSemanticMemory returned nil with nil error")
		}
		// Exercise returned struct to catch latent panics.
		_ = len(sm.Concepts)
		_ = sm.SDRSize
	})
}

// ---------------------------------------------------------------------------
// FuzzFractalCortexLoadMetadata — fuzz the metadata.json parser in Load()
// ---------------------------------------------------------------------------

func FuzzFractalCortexLoadMetadata(f *testing.F) {
	// Seed 1: valid metadata with 0 blocks (no block dirs needed)
	validMeta, _ := json.Marshal(map[string]interface{}{
		"blocks_count": 0,
		"num_layers":   24,
		"dim":          256,
		"top_k":        3,
	})
	f.Add(validMeta)

	// Seed 2: empty JSON object
	f.Add([]byte(`{}`))

	// Seed 3: empty bytes
	f.Add([]byte{})

	// Seed 4: not JSON
	f.Add([]byte("absolutely not json {{{"))

	// Seed 5: negative values
	f.Add([]byte(`{"blocks_count":-1,"num_layers":-1,"dim":-1,"top_k":-1}`))

	// Seed 6: huge values
	f.Add([]byte(`{"blocks_count":99999,"num_layers":99999,"dim":99999,"top_k":99999}`))

	// Seed 7: zero values
	f.Add([]byte(`{"blocks_count":0,"num_layers":0,"dim":0,"top_k":0}`))

	// Seed 8: null fields
	f.Add([]byte(`{"blocks_count":null,"num_layers":null,"dim":null,"top_k":null}`))

	// Seed 9: string types
	f.Add([]byte(`{"blocks_count":"many","num_layers":"deep","dim":"big","top_k":"few"}`))

	// Seed 10: blocks_count=1 but no block dir exists (tests file-not-found path)
	f.Add([]byte(`{"blocks_count":1,"num_layers":2,"dim":32,"top_k":1}`))

	// Seed 11: boundary at MaxFractalBlocks
	f.Add([]byte(`{"blocks_count":8,"num_layers":1,"dim":16,"top_k":1}`))

	// Seed 12: just over MaxFractalBlocks
	f.Add([]byte(`{"blocks_count":9,"num_layers":1,"dim":16,"top_k":1}`))

	f.Fuzz(func(t *testing.T, data []byte) {
		// Create a temporary directory structure that FractalCortex.Load expects.
		// It reads <dataDir>/fractal_cortex/metadata.json
		tmpDir := t.TempDir()
		fcDir := filepath.Join(tmpDir, "fractal_cortex")
		if err := os.MkdirAll(fcDir, 0700); err != nil {
			t.Fatalf("failed to create fractal_cortex dir: %v", err)
		}
		metaPath := filepath.Join(fcDir, "metadata.json")
		if err := os.WriteFile(metaPath, data, 0600); err != nil {
			t.Fatalf("failed to write metadata.json: %v", err)
		}

		fc := &FractalCortex{
			Config: Config{SDRSize: 256},
		}

		// Must not panic — errors are expected and acceptable.
		err := fc.Load(tmpDir)
		if err != nil {
			return // error is fine
		}
		// Exercise returned state.
		_ = len(fc.Blocks)
	})
}
