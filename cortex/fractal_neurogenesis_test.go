package cortex

import "testing"

// TestPerturbTernaryLayer_BreaksSymmetry: regression test pentru bug-ul
// documentat în HARDCODING_AND_LIMITATIONS.md §8.4.
//
// Bug-ul original: neurogeneza copia weights din block-părinte fără
// perturbare → noul block aplică exact aceleași transformări ternare,
// "neurogeneza" devine iluzorie (zero parametri independenți noi).
//
// Fix-ul: după copyTernaryWeights, perturbTernaryLayer flippează un
// procent configurabil de bit-uri cu seed per-block (uniqueness garantată).
//
// Acest test verifică două invariante:
//  1. După perturbare, layer-ul perturbat DIFERĂ de original (break-symmetry).
//  2. Două perturbări cu seed-uri diferite produc layere DIFERITE între ele
//     (fiecare block ulterior are signature unic).
func TestPerturbTernaryLayer_BreaksSymmetry(t *testing.T) {
	original := NewTernaryLayer(64, 32)
	// Populăm cu pattern non-zero ca să avem ce flippa.
	for i := range original.Tiles {
		original.Tiles[i] = TernaryTile(uint32(i)*0x9E37 + 0x12345)
	}

	clone1 := NewTernaryLayer(64, 32)
	copy(clone1.Tiles, original.Tiles)
	perturbTernaryLayer(clone1, 64, 1) // rate=64 (~25%), seed=1

	clone2 := NewTernaryLayer(64, 32)
	copy(clone2.Tiles, original.Tiles)
	perturbTernaryLayer(clone2, 64, 2) // rate=64, seed=2

	// Invariant 1: clone1 != original.
	differentFromOrig := 0
	for i := range original.Tiles {
		if clone1.Tiles[i] != original.Tiles[i] {
			differentFromOrig++
		}
	}
	if differentFromOrig == 0 {
		t.Fatalf("perturbare cu rate=64 a lăsat layer-ul neschimbat — break-symmetry NU funcționează")
	}
	// Rate 64/255 ≈ 25% — așteptăm între 10% și 45% tile-uri schimbate
	// (variație stochastică pe samples mici).
	pct := float64(differentFromOrig) / float64(len(original.Tiles)) * 100
	if pct < 10 || pct > 45 {
		t.Errorf("rate de schimbare așteptat ~25%%, primit %.1f%% (%d/%d)",
			pct, differentFromOrig, len(original.Tiles))
	}

	// Invariant 2: clone1 != clone2 (seed-uri diferite → perturbări diferite).
	differentBetweenClones := 0
	for i := range clone1.Tiles {
		if clone1.Tiles[i] != clone2.Tiles[i] {
			differentBetweenClones++
		}
	}
	if differentBetweenClones == 0 {
		t.Fatal("seed-uri diferite produc layere identice — uniqueness per-block lipsește")
	}
}

// TestPerturbTernaryLayer_ZeroRateNoOp: rate=0 = no-op (block-urile primare
// folosesc fresh init, nu perturbare; configurația implicită trebuie să fie
// safe).
func TestPerturbTernaryLayer_ZeroRateNoOp(t *testing.T) {
	original := NewTernaryLayer(64, 32)
	for i := range original.Tiles {
		original.Tiles[i] = TernaryTile(uint32(i) * 7)
	}
	clone := NewTernaryLayer(64, 32)
	copy(clone.Tiles, original.Tiles)
	perturbTernaryLayer(clone, 0, 42) // rate=0
	for i := range original.Tiles {
		if clone.Tiles[i] != original.Tiles[i] {
			t.Fatalf("rate=0 a modificat tile %d: vrut %v, primit %v",
				i, original.Tiles[i], clone.Tiles[i])
		}
	}
}

// TestPerturbTernaryLayer_NilSafe: nil layer = no panic.
func TestPerturbTernaryLayer_NilSafe(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("perturbTernaryLayer(nil) panic-uit: %v", r)
		}
	}()
	perturbTernaryLayer(nil, 64, 1)
}
