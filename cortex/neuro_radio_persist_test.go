package cortex

import (
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

func TestNeuroRadioPersistRoundtrip(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	nrc := NewNeuroRadioCortex(1000, rng)

	// Modify some tiles to ensure state is preserved
	nrc.Tiles[0].Radio.SetAmplitude(200)
	nrc.Tiles[0].Weights = 0xDEADBEEF
	nrc.Tiles[0].Confidence = 0xCAFEBABE
	nrc.Tiles[100].Radio.SetAmplitude(0) // Kill a tile
	nrc.TickCount = 42

	// Save
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "test.nxnr")
	if err := SaveNeuroRadioCortex(nrc, path); err != nil {
		t.Fatalf("SaveNeuroRadioCortex failed: %v", err)
	}

	// Verify file exists and has expected size
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("File not found: %v", err)
	}
	expectedSize := int64(neuroRadioHeaderSize + 1000*12)
	if info.Size() != expectedSize {
		t.Errorf("File size: got %d, want %d", info.Size(), expectedSize)
	}

	// Load
	loaded, err := LoadNeuroRadioCortex(path)
	if err != nil {
		t.Fatalf("LoadNeuroRadioCortex failed: %v", err)
	}

	// Verify
	if loaded.Size != 1000 {
		t.Errorf("Size: got %d, want 1000", loaded.Size)
	}
	if loaded.TickCount != 42 {
		t.Errorf("TickCount: got %d, want 42", loaded.TickCount)
	}

	// Check tile 0
	if loaded.Tiles[0].Weights != 0xDEADBEEF {
		t.Errorf("Tile 0 weights: got 0x%X, want 0xDEADBEEF", loaded.Tiles[0].Weights)
	}
	if loaded.Tiles[0].Confidence != 0xCAFEBABE {
		t.Errorf("Tile 0 confidence: got 0x%X, want 0xCAFEBABE", loaded.Tiles[0].Confidence)
	}
	if loaded.Tiles[0].Radio.Amplitude() != 200 {
		t.Errorf("Tile 0 amplitude: got %d, want 200", loaded.Tiles[0].Radio.Amplitude())
	}

	// Check dead tile 100
	if loaded.Tiles[100].Radio.IsAlive() {
		t.Error("Tile 100 should be dead")
	}

	// Verify bucket index was rebuilt
	if loaded.Index == nil {
		t.Fatal("BucketIndex not rebuilt")
	}

	// Check all tiles match
	for i := 0; i < 1000; i++ {
		if loaded.Tiles[i].Weights != nrc.Tiles[i].Weights {
			t.Errorf("Tile %d weights mismatch", i)
			break
		}
		if loaded.Tiles[i].Confidence != nrc.Tiles[i].Confidence {
			t.Errorf("Tile %d confidence mismatch", i)
			break
		}
		if uint32(loaded.Tiles[i].Radio) != uint32(nrc.Tiles[i].Radio) {
			t.Errorf("Tile %d radio mismatch", i)
			break
		}
	}
}

func TestNeuroRadioPersistNil(t *testing.T) {
	err := SaveNeuroRadioCortex(nil, "test.nxnr")
	if err == nil {
		t.Error("Expected error saving nil cortex")
	}
}

func TestNeuroRadioPersistBadMagic(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "bad.nxnr")
	os.WriteFile(path, []byte("BADD00000000000000"), 0600)

	_, err := LoadNeuroRadioCortex(path)
	if err == nil {
		t.Error("Expected error loading bad magic")
	}
}

func TestNeuroRadioPersistTruncated(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "trunc.nxnr")
	os.WriteFile(path, []byte("NXNR"), 0600)

	_, err := LoadNeuroRadioCortex(path)
	if err == nil {
		t.Error("Expected error loading truncated file")
	}
}
