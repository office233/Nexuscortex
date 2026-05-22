package cortex

import (
	"math/bits"
	"math/rand"
	"testing"
)

// ─────────────────────────────────────────────────────────────────────
// Helper: original (non-optimized) Overlap for reference comparison.
// ─────────────────────────────────────────────────────────────────────

func origOverlap(a, b SDR) int {
	n := len(a.Bits)
	if len(b.Bits) < n {
		n = len(b.Bits)
	}
	count := 0
	for i := 0; i < n; i++ {
		count += bits.OnesCount64(a.Bits[i] & b.Bits[i])
	}
	return count
}

func origSimilarity(a, b SDR) uint8 {
	maxActive := a.ActiveCount
	if b.ActiveCount > maxActive {
		maxActive = b.ActiveCount
	}
	if maxActive == 0 {
		return 0
	}
	result := origOverlap(a, b) * 255 / maxActive
	if result > 255 {
		result = 255
	}
	return uint8(result)
}

func origUnion(a, b SDR) SDR {
	size := a.Size
	if b.Size > size {
		size = b.Size
	}
	result := NewSDR(size)
	na, nb := len(a.Bits), len(b.Bits)
	for i := 0; i < len(result.Bits); i++ {
		var va, vb uint64
		if i < na {
			va = a.Bits[i]
		}
		if i < nb {
			vb = b.Bits[i]
		}
		result.Bits[i] = va | vb
	}
	result.ActiveCount = result.recount()
	return result
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

func TestFastOverlapMatchesOriginal(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	sizes := []int{64, 128, 256, 500, 1000, 10000}
	for _, size := range sizes {
		for trial := 0; trial < 20; trial++ {
			active := 1 + rng.Intn(size/10+1)
			a := RandomSDR(size, active, rng)
			b := RandomSDR(size, active, rng)

			got := FastOverlap(a, b)
			want := origOverlap(a, b)
			if got != want {
				t.Errorf("size=%d trial=%d: FastOverlap=%d, origOverlap=%d", size, trial, got, want)
			}
		}
	}
}

func TestFastSimilarityMatchesOriginal(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	sizes := []int{64, 200, 1000, 10000}
	for _, size := range sizes {
		for trial := 0; trial < 20; trial++ {
			active := 1 + rng.Intn(size/10+1)
			a := RandomSDR(size, active, rng)
			b := RandomSDR(size, active, rng)

			got := FastSimilarity(a, b)
			want := origSimilarity(a, b)
			if got != want {
				t.Errorf("size=%d trial=%d: FastSimilarity=%d, origSimilarity=%d", size, trial, got, want)
			}
		}
	}
}

func TestFastUnionMatchesOriginal(t *testing.T) {
	rng := rand.New(rand.NewSource(77))
	sizes := []int{64, 128, 300, 1000, 10000}
	for _, size := range sizes {
		for trial := 0; trial < 10; trial++ {
			active := 1 + rng.Intn(size/10+1)
			a := RandomSDR(size, active, rng)
			b := RandomSDR(size, active, rng)

			got := FastUnion(a, b)
			want := origUnion(a, b)

			if got.Size != want.Size {
				t.Errorf("size=%d trial=%d: Size mismatch %d vs %d", size, trial, got.Size, want.Size)
				continue
			}
			if got.ActiveCount != want.ActiveCount {
				t.Errorf("size=%d trial=%d: ActiveCount mismatch %d vs %d", size, trial, got.ActiveCount, want.ActiveCount)
				continue
			}
			if len(got.Bits) != len(want.Bits) {
				t.Errorf("size=%d trial=%d: Bits length mismatch %d vs %d", size, trial, len(got.Bits), len(want.Bits))
				continue
			}
			for i := range got.Bits {
				if got.Bits[i] != want.Bits[i] {
					t.Errorf("size=%d trial=%d: Bits[%d] mismatch %x vs %x", size, trial, i, got.Bits[i], want.Bits[i])
					break
				}
			}
		}
	}
}

func TestFastUnionMismatchedSizes(t *testing.T) {
	rng := rand.New(rand.NewSource(55))
	a := RandomSDR(500, 25, rng)
	b := RandomSDR(10000, 50, rng)

	got := FastUnion(a, b)
	want := origUnion(a, b)

	if got.Size != want.Size {
		t.Errorf("Size mismatch %d vs %d", got.Size, want.Size)
	}
	if got.ActiveCount != want.ActiveCount {
		t.Errorf("ActiveCount mismatch %d vs %d", got.ActiveCount, want.ActiveCount)
	}
	for i := range got.Bits {
		if got.Bits[i] != want.Bits[i] {
			t.Errorf("Bits[%d] mismatch %x vs %x", i, got.Bits[i], want.Bits[i])
			break
		}
	}
}

func TestBatchSimilarity(t *testing.T) {
	rng := rand.New(rand.NewSource(123))
	target := RandomSDR(10000, 50, rng)
	candidates := make([]SDR, 100)
	for i := range candidates {
		candidates[i] = RandomSDR(10000, 50, rng)
	}

	scores := BatchSimilarity(target, candidates)
	if len(scores) != len(candidates) {
		t.Fatalf("expected %d scores, got %d", len(candidates), len(scores))
	}

	for i, cand := range candidates {
		want := origSimilarity(target, cand)
		if scores[i] != want {
			t.Errorf("candidate %d: BatchSimilarity=%d, origSimilarity=%d", i, scores[i], want)
		}
	}
}

func TestBatchSimilarityEmpty(t *testing.T) {
	target := RandomSDR(1000, 10, rand.New(rand.NewSource(1)))
	scores := BatchSimilarity(target, nil)
	if scores != nil {
		t.Errorf("expected nil for empty candidates, got %v", scores)
	}
	scores = BatchSimilarity(target, []SDR{})
	if scores != nil {
		t.Errorf("expected nil for zero-length candidates, got %v", scores)
	}
}

func TestFastEdgeCases(t *testing.T) {
	t.Run("EmptySDRs", func(t *testing.T) {
		a := NewSDR(1000)
		b := NewSDR(1000)
		if FastOverlap(a, b) != 0 {
			t.Error("expected 0 overlap for empty SDRs")
		}
		if FastSimilarity(a, b) != 0 {
			t.Error("expected 0 similarity for empty SDRs")
		}
		u := FastUnion(a, b)
		if u.ActiveCount != 0 {
			t.Error("expected 0 active for union of empty SDRs")
		}
	})

	t.Run("SelfOverlap", func(t *testing.T) {
		rng := rand.New(rand.NewSource(7))
		s := RandomSDR(10000, 50, rng)
		if FastOverlap(s, s) != s.ActiveCount {
			t.Errorf("self-overlap should equal ActiveCount, got %d want %d",
				FastOverlap(s, s), s.ActiveCount)
		}
		if FastSimilarity(s, s) != 255 {
			t.Errorf("self-similarity should be 255, got %d", FastSimilarity(s, s))
		}
	})

	t.Run("SingleWord", func(t *testing.T) {
		a := RandomSDR(32, 5, rand.New(rand.NewSource(1)))
		b := RandomSDR(32, 5, rand.New(rand.NewSource(2)))
		got := FastOverlap(a, b)
		want := origOverlap(a, b)
		if got != want {
			t.Errorf("single-word: FastOverlap=%d, origOverlap=%d", got, want)
		}
	})

	t.Run("MismatchedSizes", func(t *testing.T) {
		rng := rand.New(rand.NewSource(33))
		a := RandomSDR(100, 10, rng)
		b := RandomSDR(5000, 30, rng)
		got := FastOverlap(a, b)
		want := origOverlap(a, b)
		if got != want {
			t.Errorf("mismatched: FastOverlap=%d, origOverlap=%d", got, want)
		}
		gotSim := FastSimilarity(a, b)
		wantSim := origSimilarity(a, b)
		if gotSim != wantSim {
			t.Errorf("mismatched: FastSimilarity=%d, origSimilarity=%d", gotSim, wantSim)
		}
	})

	t.Run("ZeroSizeSDR", func(t *testing.T) {
		a := NewSDR(0)
		b := NewSDR(0)
		if FastOverlap(a, b) != 0 {
			t.Error("expected 0 overlap for zero-size SDRs")
		}
		if FastSimilarity(a, b) != 0 {
			t.Error("expected 0 similarity for zero-size SDRs")
		}
	})
}

// ─────────────────────────────────────────────────────────────────────
// Method delegation tests: SDR.Overlap and SDR.Similarity should now
// delegate to FastOverlap / FastSimilarity and still produce the same
// results as the original implementation.
// ─────────────────────────────────────────────────────────────────────

func TestMethodDelegation(t *testing.T) {
	rng := rand.New(rand.NewSource(888))
	for trial := 0; trial < 50; trial++ {
		size := 100 + rng.Intn(10000)
		active := 1 + rng.Intn(size/10+1)
		a := RandomSDR(size, active, rng)
		b := RandomSDR(size, active, rng)

		// Overlap method should still return correct result.
		methodOverlap := a.Overlap(b)
		origOv := origOverlap(a, b)
		if methodOverlap != origOv {
			t.Errorf("trial=%d: Overlap()=%d, want=%d", trial, methodOverlap, origOv)
		}

		// Similarity method should still return correct result.
		methodSim := a.Similarity(b)
		origSim := origSimilarity(a, b)
		if methodSim != origSim {
			t.Errorf("trial=%d: Similarity()=%d, want=%d", trial, methodSim, origSim)
		}
	}
}

// ─────────────────────────────────────────────────────────────────────
// Benchmarks
// ─────────────────────────────────────────────────────────────────────

func BenchmarkFastOverlap(b *testing.B) {
	a := RandomSDR(10000, 50, rand.New(rand.NewSource(42)))
	c := RandomSDR(10000, 50, rand.New(rand.NewSource(43)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FastOverlap(a, c)
	}
}

func BenchmarkFastSimilarity(b *testing.B) {
	a := RandomSDR(10000, 50, rand.New(rand.NewSource(42)))
	c := RandomSDR(10000, 50, rand.New(rand.NewSource(43)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FastSimilarity(a, c)
	}
}

func BenchmarkFastUnion(b *testing.B) {
	a := RandomSDR(10000, 50, rand.New(rand.NewSource(42)))
	c := RandomSDR(10000, 50, rand.New(rand.NewSource(43)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FastUnion(a, c)
	}
}

func BenchmarkBatchSimilarity(b *testing.B) {
	target := RandomSDR(10000, 50, rand.New(rand.NewSource(42)))
	candidates := make([]SDR, 100)
	rng := rand.New(rand.NewSource(43))
	for i := range candidates {
		candidates[i] = RandomSDR(10000, 50, rng)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		BatchSimilarity(target, candidates)
	}
}
