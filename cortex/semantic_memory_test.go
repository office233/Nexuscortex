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

// TestSemanticMemoryConsolidation_NoIdentityBug verifică reparația
// pentru bug-ul matematic descris în HARDCODING_AND_LIMITATIONS.md §8.1:
//
//	v1: Prototype = Prototype ∩ Input → conceptele se atrofiază monoton.
//	v2: Prototype = Prototype ∪ Input → conceptele cresc necontrolat (zgomot).
//	v3 (corect): merge confidence-weighted.
//
// Acest test scenarizează o secvență tipică:
//  1. Două episode foarte similare formează un concept (Phase 2).
//  2. Un al treilea episod *high-confidence* aduce bit-uri NOI invariante
//     → conceptul trebuie să le absoarbă (fix v1).
//  3. Un al patrulea episod *moderate-confidence* aduce bit-uri zgomotoase
//     → conceptul NU trebuie să le absoarbă (fix v2).
func TestSemanticMemoryConsolidation_NoIdentityBug(t *testing.T) {
	sdrSize := 1000
	cfg := DefaultConfig()
	cfg.SemanticMemorySimThreshold = 80 // explicit, ca să fie clar
	sm := NewSemanticMemory(sdrSize, cfg)
	hip := NewHippocampus(cfg)

	mkSDR := func(bits ...int) SDR {
		s := NewSDR(sdrSize)
		for _, b := range bits {
			s.Set(b)
		}
		return s
	}

	// Pas 1: formăm un concept din două episode care overlap puternic.
	// Bit-uri comune: 10, 20, 30, 40. Distincte: 50/60 vs 70/80.
	e1 := mkSDR(10, 20, 30, 40, 50, 60)
	e2 := mkSDR(10, 20, 30, 40, 70, 80)
	hip.Store(e1, NewSDR(sdrSize), "ep1")
	hip.Store(e2, NewSDR(sdrSize), "ep2")
	sm.Generalize(hip)

	if len(sm.Concepts) != 1 {
		t.Fatalf("Pas 1: așteptat 1 concept, primit %d", len(sm.Concepts))
	}
	initialBits := sm.Concepts[0].Prototype.ActiveCount
	if initialBits != 4 {
		t.Fatalf("Pas 1: prototip inițial trebuie să aibă 4 bit-uri (intersect), are %d", initialBits)
	}
	initialCount := sm.Concepts[0].Count
	if initialCount != 2 {
		t.Fatalf("Pas 1: Count așteptat 2, primit %d", initialCount)
	}

	// Pas 2: episod high-confidence (overlap 4/5 = sim foarte mare) cu
	// un bit NOU invariant (100) → trebuie absorbit (fix v1, anti-identity).
	hip.Memories = hip.Memories[:0] // reset hip ca să nu reproceseze e1/e2
	e3 := mkSDR(10, 20, 30, 40, 100)
	hip.Store(e3, NewSDR(sdrSize), "ep3")
	sm.Generalize(hip)

	bitsAfterHighConf := sm.Concepts[0].Prototype.ActiveCount
	if bitsAfterHighConf <= initialBits {
		t.Errorf("Pas 2 (fix v1): conceptul trebuia să CREASCĂ după high-confidence merge, "+
			"a rămas la %d bit-uri (inițial %d). Bug-ul de identitate matematică NU e reparat.",
			bitsAfterHighConf, initialBits)
	}
	// Verificăm că bit-ul 100 a fost absorbit explicit.
	bit100Active := false
	for _, idx := range sm.Concepts[0].Prototype.ActiveIndices() {
		if idx == 100 {
			bit100Active = true
			break
		}
	}
	if !bit100Active {
		t.Errorf("Pas 2: bit-ul 100 (invariant nou, high-confidence) trebuia absorbit")
	}

	bitsAfterStep2 := sm.Concepts[0].Prototype.ActiveCount

	// Pas 3: episod moderate-confidence (sim deasupra pragului dar nu mult)
	// cu MULTE bit-uri noi zgomotoase → NU trebuie absorbit (fix v2).
	// Construim un episod care are doar 4 bit-uri din prototip + 4 bit-uri
	// total străine. Similaritatea va fi peste 80 dar sub high-confidence.
	hip.Memories = hip.Memories[:0]
	e4 := mkSDR(10, 20, 30, 40, 500, 501, 502, 503)
	hip.Store(e4, NewSDR(sdrSize), "ep4")
	simModerate := e4.Similarity(sm.Concepts[0].Prototype)
	if simModerate < 80 {
		t.Fatalf("setup pas 3: sim trebuie să fie ≥80 ca să match-uiască, e %d", simModerate)
	}
	sm.Generalize(hip)

	bitsAfterModerate := sm.Concepts[0].Prototype.ActiveCount
	// Numărăm câte bit-uri zgomotoase (500-503) au pătruns.
	noiseAbsorbed := 0
	for _, idx := range sm.Concepts[0].Prototype.ActiveIndices() {
		if idx >= 500 && idx <= 503 {
			noiseAbsorbed++
		}
	}

	// Decision: dacă sim a fost moderate (sub high-conf prag), NU absorbim.
	// highConf = 80 + (255-80)/2 = 80 + 87 = 167.
	// simModerate va fi sub 167 → nu trebuie să absorbim bit-urile 500-503.
	if simModerate < 167 && noiseAbsorbed > 0 {
		t.Errorf("Pas 3 (fix v2): cu sim=%d (sub high-confidence 167), "+
			"prototype-ul a absorbit %d bit-uri zgomotoase (500-503). "+
			"Conceptul crește necontrolat — bug-ul de overcorrection NU e reparat.",
			simModerate, noiseAbsorbed)
	}
	if simModerate < 167 && bitsAfterModerate != bitsAfterStep2 {
		t.Errorf("Pas 3: bit-count trebuia să rămână %d (moderate-conf no-op), e %d",
			bitsAfterStep2, bitsAfterModerate)
	}
	// În toate cazurile Count trebuie să crească (validare statistică).
	if sm.Concepts[0].Count != initialCount+2 {
		t.Errorf("Count: așteptat %d, primit %d", initialCount+2, sm.Concepts[0].Count)
	}
}

// TestSemanticMemoryMatureConceptImmutable: după ce un concept devine
// matur (Count >= maturityThresh), prototype-ul nu se mai modifică.
func TestSemanticMemoryMatureConceptImmutable(t *testing.T) {
	sdrSize := 1000
	cfg := DefaultConfig()
	cfg.SemanticMemoryConceptMaturity = 3 // matur după 3 episode
	cfg.SemanticMemoryMinViableBits = 3   // permite prototip cu 3+ bit-uri
	sm := NewSemanticMemory(sdrSize, cfg)

	// Concept pre-existent matur cu 6 bit-uri (peste minViableBits).
	proto := NewSDR(sdrSize)
	for _, b := range []int{10, 20, 30, 40, 50, 60} {
		proto.Set(b)
	}
	sm.Concepts = append(sm.Concepts, Concept{
		Prototype: proto,
		Count:     5, // mature (>=3)
		Contexts:  []string{"anterior"},
	})

	// Episod high-confidence cu bit-uri noi — NU trebuie absorbit.
	hip := NewHippocampus(cfg)
	ep := NewSDR(sdrSize)
	for _, b := range []int{10, 20, 30, 40, 50, 60, 999} {
		ep.Set(b)
	}
	hip.Store(ep, NewSDR(sdrSize), "ep")
	sm.Generalize(hip)

	if len(sm.Concepts) == 0 {
		t.Fatal("conceptul matur a fost eliminat — probabil prune-uit incorect")
	}
	for _, idx := range sm.Concepts[0].Prototype.ActiveIndices() {
		if idx == 999 {
			t.Errorf("Concept matur a absorbit bit nou (999) — trebuia imutabil")
		}
	}
	if sm.Concepts[0].Count != 6 {
		t.Errorf("Count matur trebuie să crească (validare): vrut 6, primit %d", sm.Concepts[0].Count)
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
