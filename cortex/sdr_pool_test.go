package cortex

import (
	"math/rand"
	"sync"
	"testing"
	"unsafe"
)

// ─── Pool basics ────────────────────────────────────────────────────

func TestSDRPoolAcquireRelease(t *testing.T) {
	sdr := AcquireSDR(10000)

	// Must have correct geometry.
	if sdr.Size != 10000 {
		t.Fatalf("Size = %d, want 10000", sdr.Size)
	}
	wantWords := wordsNeeded(10000)
	if len(sdr.Bits) != wantWords {
		t.Fatalf("len(Bits) = %d, want %d", len(sdr.Bits), wantWords)
	}
	if sdr.ActiveCount != 0 {
		t.Fatalf("ActiveCount = %d, want 0", sdr.ActiveCount)
	}

	// All words must be zero.
	for i, w := range sdr.Bits {
		if w != 0 {
			t.Fatalf("Bits[%d] = %d, want 0", i, w)
		}
	}

	// Release must not panic.
	ReleaseSDR(sdr)

	// Release on nil Bits must not panic.
	ReleaseSDR(SDR{})
}

func TestSDRPoolReuse(t *testing.T) {
	// Acquire, dirty, release, acquire again — should reuse backing.
	sdr1 := AcquireSDR(640) // 10 words exactly
	ptr1 := unsafe.Pointer(&sdr1.Bits[0])

	// Dirty it so we can verify zeroing.
	sdr1.Bits[0] = 0xDEADBEEF
	sdr1.Bits[9] = 0xCAFEBABE
	ReleaseSDR(sdr1)

	// Acquire same size — high probability of getting the same slice
	// (sync.Pool does not guarantee reuse, so we retry a few times).
	reused := false
	for attempt := 0; attempt < 5; attempt++ {
		sdr2 := AcquireSDR(640)
		ptr2 := unsafe.Pointer(&sdr2.Bits[0])

		if ptr1 == ptr2 {
			reused = true
			// Verify it was zeroed.
			for i, w := range sdr2.Bits {
				if w != 0 {
					t.Fatalf("reused Bits[%d] = %d, want 0 (not zeroed)", i, w)
				}
			}
			ReleaseSDR(sdr2)
			break
		}
		ReleaseSDR(sdr2)
	}
	if !reused {
		t.Log("pool did not reuse backing array in 5 attempts (non-deterministic, acceptable)")
	}
}

func TestSDRPoolConcurrency(t *testing.T) {
	const goroutines = 16
	const opsPerGoroutine = 200

	var wg sync.WaitGroup
	wg.Add(goroutines)

	for g := 0; g < goroutines; g++ {
		go func() {
			defer wg.Done()
			for i := 0; i < opsPerGoroutine; i++ {
				sdr := AcquireSDR(2048)
				// Light work.
				sdr.Set(i % sdr.Size)
				if !sdr.IsActive(i % sdr.Size) {
					t.Errorf("bit %d not active after Set", i%sdr.Size)
				}
				ReleaseSDR(sdr)
			}
		}()
	}
	wg.Wait()
}

// ─── Union / Intersect correctness with pooled SDRs ────────────────

func TestSDRUnionStillCorrect(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	a := RandomSDR(10000, 50, rng)
	b := RandomSDR(10000, 50, rng)

	result := a.Union(b)

	// Every bit in a must be in result.
	for _, idx := range a.ActiveIndices() {
		if !result.IsActive(idx) {
			t.Fatalf("Union missing bit %d from a", idx)
		}
	}
	// Every bit in b must be in result.
	for _, idx := range b.ActiveIndices() {
		if !result.IsActive(idx) {
			t.Fatalf("Union missing bit %d from b", idx)
		}
	}
	// No extra bits — result active count must equal unique union count.
	expected := make(map[int]bool)
	for _, idx := range a.ActiveIndices() {
		expected[idx] = true
	}
	for _, idx := range b.ActiveIndices() {
		expected[idx] = true
	}
	if result.ActiveCount != len(expected) {
		t.Fatalf("ActiveCount = %d, want %d", result.ActiveCount, len(expected))
	}

	ReleaseSDR(result)
}

func TestSDRIntersectStillCorrect(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	a := RandomSDR(10000, 200, rng)
	b := RandomSDR(10000, 200, rng)

	result := a.Intersect(b)

	// Every bit in result must be in both a and b.
	for _, idx := range result.ActiveIndices() {
		if !a.IsActive(idx) {
			t.Fatalf("Intersect result bit %d not in a", idx)
		}
		if !b.IsActive(idx) {
			t.Fatalf("Intersect result bit %d not in b", idx)
		}
	}

	// Count overlap manually.
	overlap := a.Overlap(b)
	if result.ActiveCount != overlap {
		t.Fatalf("ActiveCount = %d, want overlap %d", result.ActiveCount, overlap)
	}

	ReleaseSDR(result)
}

func TestSDRUnionDifferentSizes(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	a := RandomSDR(5000, 30, rng)
	b := RandomSDR(10000, 30, rng)

	result := a.Union(b)
	if result.Size != 10000 {
		t.Fatalf("Union size = %d, want 10000", result.Size)
	}
	for _, idx := range a.ActiveIndices() {
		if !result.IsActive(idx) {
			t.Fatalf("Union missing bit %d from smaller SDR", idx)
		}
	}
	for _, idx := range b.ActiveIndices() {
		if !result.IsActive(idx) {
			t.Fatalf("Union missing bit %d from larger SDR", idx)
		}
	}
	ReleaseSDR(result)
}

func TestSDRIntersectDifferentSizes(t *testing.T) {
	rng := rand.New(rand.NewSource(8))
	a := RandomSDR(10000, 100, rng)
	b := RandomSDR(5000, 100, rng)

	result := a.Intersect(b)
	if result.Size != 5000 {
		t.Fatalf("Intersect size = %d, want 5000", result.Size)
	}
	for _, idx := range result.ActiveIndices() {
		if !a.IsActive(idx) || !b.IsActive(idx) {
			t.Fatalf("Intersect bit %d not in both inputs", idx)
		}
	}
	ReleaseSDR(result)
}

// ─── Benchmarks ─────────────────────────────────────────────────────

func BenchmarkSDRUnionPooled(b *testing.B) {
	a := RandomSDR(10000, 50, rand.New(rand.NewSource(42)))
	c := RandomSDR(10000, 50, rand.New(rand.NewSource(43)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result := a.Union(c)
		ReleaseSDR(result)
	}
}

func BenchmarkSDRIntersectPooled(b *testing.B) {
	a := RandomSDR(10000, 50, rand.New(rand.NewSource(42)))
	c := RandomSDR(10000, 50, rand.New(rand.NewSource(43)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result := a.Intersect(c)
		ReleaseSDR(result)
	}
}

func BenchmarkSDRUnionNoRelease(b *testing.B) {
	// Baseline: caller doesn't release — GC collects like before.
	a := RandomSDR(10000, 50, rand.New(rand.NewSource(42)))
	c := RandomSDR(10000, 50, rand.New(rand.NewSource(43)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.Union(c)
	}
}

func BenchmarkSDRIntersectNoRelease(b *testing.B) {
	a := RandomSDR(10000, 50, rand.New(rand.NewSource(42)))
	c := RandomSDR(10000, 50, rand.New(rand.NewSource(43)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.Intersect(c)
	}
}
