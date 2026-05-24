package cortex

import (
	"math/rand"
	"testing"
	"time"
)

// TestGenerateFastEquivalence verifies that the KV-cached fast path
// produces the same tokens as the un-cached training path for the same
// prompt and the same RNG state. With greedy sampling (low temperature,
// top-k = 1) the two paths must agree deterministically; this is the
// strictest possible correctness check for the cache.
func TestGenerateFastEquivalence(t *testing.T) {
	cfg := TransformerConfig{
		VocabSize:  64,
		EmbedDim:   16,
		NumHeads:   2,
		NumLayers:  2,
		FFNDim:     32,
		MaxSeqLen:  32,
		EOSTokenID: -1,
	}

	rng1 := rand.New(rand.NewSource(7))
	m := NewMiniTransformer(cfg, rng1)

	prompt := []int{1, 5, 9, 13}
	maxNew := 8

	const temp = 0.001
	const k = 1

	m.Rng = rand.New(rand.NewSource(42))
	slow := m.Generate(prompt, maxNew, temp, k)

	m.Rng = rand.New(rand.NewSource(42))
	fast := m.GenerateFast(prompt, maxNew, temp, k)

	if len(slow) != len(fast) {
		t.Fatalf("length mismatch: slow=%d fast=%d", len(slow), len(fast))
	}
	for i := range slow {
		if slow[i] != fast[i] {
			t.Fatalf("token mismatch at %d: slow=%d fast=%d (slow=%v fast=%v)",
				i, slow[i], fast[i], slow, fast)
		}
	}
}

// TestGenerateFastNotSlower exercises a basic timing sanity check: the
// cached path must not regress catastrophically. We do not assert a
// specific speedup because that depends on the host, but a bug that
// makes cached path slower than scratch needs to be caught.
func TestGenerateFastNotSlower(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping perf check in -short mode")
	}

	cfg := TransformerConfig{
		VocabSize:  256,
		EmbedDim:   32,
		NumHeads:   4,
		NumLayers:  2,
		FFNDim:     64,
		MaxSeqLen:  64,
		EOSTokenID: -1,
	}

	rng := rand.New(rand.NewSource(1))
	m := NewMiniTransformer(cfg, rng)

	prompt := make([]int, 20)
	for i := range prompt {
		prompt[i] = i + 1
	}
	maxNew := 16

	m.Rng = rand.New(rand.NewSource(99))
	t0 := time.Now()
	_ = m.Generate(prompt, maxNew, 1.0, 8)
	slow := time.Since(t0)

	m.Rng = rand.New(rand.NewSource(99))
	t0 = time.Now()
	_ = m.GenerateFast(prompt, maxNew, 1.0, 8)
	fast := time.Since(t0)

	t.Logf("Generate (slow)     = %v", slow)
	t.Logf("GenerateFast (cache)= %v", fast)

	if fast > 2*slow+10*time.Millisecond {
		t.Fatalf("GenerateFast unexpectedly slow: %v vs %v", fast, slow)
	}
}
