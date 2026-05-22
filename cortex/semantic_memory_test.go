package cortex

import (
	"os"
	"path/filepath"
	"testing"
)

// TestSemanticMemoryGeneralize verifies that semantic generalization correctly
// extracts invariant overlapping features from episodic memories.
func TestSemanticMemoryGeneralize(t *testing.T) {
	sdrSize := 1000
	sm := NewSemanticMemory(sdrSize)
	hip := NewHippocampus(DefaultConfig())

	// Create two overlapping input SDRs:
	// They share bit 10, 20, 30, and 40 (4 overlapping bits),
	// but differ on other bits.
	s1 := NewSDR(sdrSize)
	s1.Set(10)
	s1.Set(20)
	s1.Set(30)
	s1.Set(40)
	s1.Set(50)
	s1.Set(60)

	s2 := NewSDR(sdrSize)
	s2.Set(10)
	s2.Set(20)
	s2.Set(30)
	s2.Set(40)
	s2.Set(70)
	s2.Set(80)

	// Verify that they have similarity high enough to merge.
	// MaxActive = 6. Overlap = 4. Similarity = 4 * 255 / 6 = 170.
	// Since 170 >= SimThreshold (80), they will be generalized!
	sim := s1.Similarity(s2)
	if sim < 80 {
		t.Fatalf("expected similarity >= 80, got %d", sim)
	}

	hip.Store(s1, NewSDR(sdrSize), "prompt one")
	hip.Store(s2, NewSDR(sdrSize), "prompt two")

	// Trigger concept generalization
	sm.Generalize(hip)

	// Verify concept was extracted
	if len(sm.Concepts) != 1 {
		t.Fatalf("expected exactly 1 concept, got %d", len(sm.Concepts))
	}

	concept := sm.Concepts[0]
	if concept.Count != 2 {
		t.Errorf("expected count 2, got %d", concept.Count)
	}

	// Prototype must only contain the overlapping bits: 10, 20, 30, 40
	indices := concept.Prototype.ActiveIndices()
	expected := []int{10, 20, 30, 40}

	if len(indices) != len(expected) {
		t.Fatalf("expected active indices length %d, got %v", len(expected), indices)
	}

	for i, idx := range indices {
		if idx != expected[i] {
			t.Errorf("expected bit %d at position %d, got %d", expected[i], i, idx)
		}
	}

	// Verify context prompts were saved
	if len(concept.Contexts) != 2 {
		t.Errorf("expected 2 contexts, got %d", len(concept.Contexts))
	}
}

// TestSemanticMemoryPersistence verifies that semantic memory correctly persists to JSON
// and loads back, recreating the correct SDR representations.
func TestSemanticMemoryPersistence(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "semantic.json")

	sdrSize := 500
	sm := NewSemanticMemory(sdrSize)

	// Create a mock concept
	sdr := NewSDR(sdrSize)
	sdr.Set(100)
	sdr.Set(200)
	sdr.Set(300)

	sm.Concepts = append(sm.Concepts, Concept{
		Prototype: sdr,
		Count:     5,
		Contexts:  []string{"cat", "feline"},
	})

	// Save to JSON
	if err := sm.Save(path); err != nil {
		t.Fatalf("expected successful save, got: %v", err)
	}

	// Verify file was written
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Fatalf("expected file to exist at %s, but it does not", path)
	}

	// Load back
	sm2, err := LoadSemanticMemory(path, sdrSize)
	if err != nil {
		t.Fatalf("expected successful load, got: %v", err)
	}

	// Check loaded properties
	if sm2.SDRSize != sdrSize {
		t.Errorf("expected SDR size %d, got %d", sdrSize, sm2.SDRSize)
	}

	if len(sm2.Concepts) != 1 {
		t.Fatalf("expected 1 concept, got %d", len(sm2.Concepts))
	}

	c := sm2.Concepts[0]
	if c.Count != 5 {
		t.Errorf("expected count 5, got %d", c.Count)
	}

	if len(c.Contexts) != 2 || c.Contexts[0] != "cat" || c.Contexts[1] != "feline" {
		t.Errorf("unexpected contexts: %v", c.Contexts)
	}

	// Check prototype SDR bits
	indices := c.Prototype.ActiveIndices()
	if len(indices) != 3 || indices[0] != 100 || indices[1] != 200 || indices[2] != 300 {
		t.Errorf("unexpected prototype indices: %v", indices)
	}
}
