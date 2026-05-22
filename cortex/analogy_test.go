package cortex

import (
	"math/rand"
	"testing"
)

// ─────────────────────────────────────────────────────────────────────
// TestSDRDifference — Verify the set-difference operation on SDRs
// ─────────────────────────────────────────────────────────────────────

func TestSDRDifference(t *testing.T) {
	// Create two SDRs with known bit patterns.
	a := NewSDR(256)
	b := NewSDR(256)

	// a = {0, 1, 2, 3, 4}
	for i := 0; i < 5; i++ {
		a.Set(i)
	}
	// b = {3, 4, 5, 6}
	for i := 3; i < 7; i++ {
		b.Set(i)
	}

	// a.Difference(b) should be {0, 1, 2} — bits in a but not in b.
	diff := a.Difference(b)

	if diff.ActiveCount != 3 {
		t.Errorf("expected 3 active bits in difference, got %d", diff.ActiveCount)
	}
	for i := 0; i < 3; i++ {
		if !diff.IsActive(i) {
			t.Errorf("expected bit %d to be active in difference", i)
		}
	}
	for i := 3; i < 7; i++ {
		if diff.IsActive(i) {
			t.Errorf("expected bit %d to be inactive in difference", i)
		}
	}

	// b.Difference(a) should be {5, 6} — bits in b but not in a.
	diff2 := b.Difference(a)
	if diff2.ActiveCount != 2 {
		t.Errorf("expected 2 active bits in b.Difference(a), got %d", diff2.ActiveCount)
	}
	if !diff2.IsActive(5) || !diff2.IsActive(6) {
		t.Error("expected bits 5 and 6 to be active in b.Difference(a)")
	}

	// Difference with self should be empty.
	self := a.Difference(a)
	if self.ActiveCount != 0 {
		t.Errorf("expected 0 active bits in a.Difference(a), got %d", self.ActiveCount)
	}

	// Difference with empty should be the same as original.
	empty := NewSDR(256)
	diffEmpty := a.Difference(empty)
	if diffEmpty.ActiveCount != a.ActiveCount {
		t.Errorf("expected %d active bits in a.Difference(empty), got %d", a.ActiveCount, diffEmpty.ActiveCount)
	}
}

// ─────────────────────────────────────────────────────────────────────
// TestAnalogyFindSimilar — Train brain with related words, verify
// similar words are found
// ─────────────────────────────────────────────────────────────────────

func TestAnalogyFindSimilar(t *testing.T) {
	cfg := DefaultConfig()
	cfg.DataDir = t.TempDir()
	cfg.AnalogyMaxCandidates = 1000

	rng := rand.New(rand.NewSource(42))
	vocab := NewVocab()
	encoder := NewEncoder(vocab, cfg.SDRSize, cfg.ActiveCount, rng, cfg)
	brain := NewBrain(
		cfg.DataDir+"/brain.nxbrain",
		cfg.DataDir+"/vocab.json",
		rng, cfg,
	)
	brain.Vocab = vocab

	// Train with related sentences so words get overlapping SDRs.
	sentences := []string{
		"the cat sat on the mat",
		"the dog sat on the rug",
		"the cat and the dog played",
		"a bird flew over the mat",
	}
	for _, s := range sentences {
		// Encode words to build SDRs (context-aware overlap).
		encoder.EncodeText(s)
		brain.Learn(s)
	}

	ae := NewAnalogyEngine(cfg, brain, encoder)

	results := ae.FindSimilar("cat", 5)

	// We should get some results (words that share SDR overlap with "cat").
	if len(results) == 0 {
		t.Fatal("expected FindSimilar to return at least one result")
	}

	// Results should be sorted by score descending.
	for i := 1; i < len(results); i++ {
		if results[i].Score > results[i-1].Score {
			t.Errorf("results not sorted: score[%d]=%d > score[%d]=%d",
				i, results[i].Score, i-1, results[i-1].Score)
		}
	}

	// "cat" itself should not be in the results.
	for _, r := range results {
		if r.Word == "cat" {
			t.Error("FindSimilar should exclude the query word itself")
		}
	}

	// Each result should have a non-zero score.
	for _, r := range results {
		if r.Score == 0 {
			t.Errorf("result word %q has zero score", r.Word)
		}
	}
}

// ─────────────────────────────────────────────────────────────────────
// TestAnalogyFindAnalogy — Train brain with word pairs, verify
// analogy completion works
// ─────────────────────────────────────────────────────────────────────

func TestAnalogyFindAnalogy(t *testing.T) {
	cfg := DefaultConfig()
	cfg.DataDir = t.TempDir()
	cfg.AnalogyMaxCandidates = 1000

	rng := rand.New(rand.NewSource(42))
	vocab := NewVocab()
	encoder := NewEncoder(vocab, cfg.SDRSize, cfg.ActiveCount, rng, cfg)
	brain := NewBrain(
		cfg.DataDir+"/brain.nxbrain",
		cfg.DataDir+"/vocab.json",
		rng, cfg,
	)
	brain.Vocab = vocab

	// Train with pairs that create structure.
	sentences := []string{
		"the king rules the kingdom",
		"the queen rules the kingdom",
		"the man works in the field",
		"the woman works in the field",
		"a king and a queen sit on thrones",
		"a man and a woman walk together",
	}
	for _, s := range sentences {
		encoder.EncodeText(s)
		brain.Learn(s)
	}

	ae := NewAnalogyEngine(cfg, brain, encoder)

	// FindAnalogy should return something (not empty) for valid inputs.
	result, confidence := ae.FindAnalogy("king", "queen", "man")

	// We can't guarantee "woman" in a small corpus, but we should get
	// a non-empty result with some confidence.
	if result == "" {
		t.Fatal("expected FindAnalogy to return a non-empty result")
	}

	// The result should not be one of the input words.
	if result == "king" || result == "queen" || result == "man" {
		t.Errorf("FindAnalogy returned an input word: %q", result)
	}

	// Confidence should be > 0.
	if confidence == 0 {
		t.Error("expected non-zero confidence for analogy result")
	}

	t.Logf("king:queen :: man:%s (confidence=%d)", result, confidence)
}

// ─────────────────────────────────────────────────────────────────────
// TestAnalogyEmptyInputs — Edge cases
// ─────────────────────────────────────────────────────────────────────

func TestAnalogyEmptyInputs(t *testing.T) {
	cfg := DefaultConfig()
	cfg.DataDir = t.TempDir()

	rng := rand.New(rand.NewSource(42))
	vocab := NewVocab()
	encoder := NewEncoder(vocab, cfg.SDRSize, cfg.ActiveCount, rng, cfg)
	brain := NewBrain(
		cfg.DataDir+"/brain.nxbrain",
		cfg.DataDir+"/vocab.json",
		rng, cfg,
	)
	brain.Vocab = vocab

	ae := NewAnalogyEngine(cfg, brain, encoder)

	// Empty inputs should return empty results.
	result, conf := ae.FindAnalogy("", "b", "c")
	if result != "" || conf != 0 {
		t.Errorf("expected empty result for empty 'a', got %q %d", result, conf)
	}

	result, conf = ae.FindAnalogy("a", "", "c")
	if result != "" || conf != 0 {
		t.Errorf("expected empty result for empty 'b', got %q %d", result, conf)
	}

	result, conf = ae.FindAnalogy("a", "b", "")
	if result != "" || conf != 0 {
		t.Errorf("expected empty result for empty 'c', got %q %d", result, conf)
	}

	// FindSimilar with empty word should return nil.
	similar := ae.FindSimilar("", 5)
	if similar != nil {
		t.Errorf("expected nil for empty word FindSimilar, got %v", similar)
	}

	// FindSimilar with topN=0 should return nil.
	similar = ae.FindSimilar("test", 0)
	if similar != nil {
		t.Errorf("expected nil for topN=0 FindSimilar, got %v", similar)
	}

	// FindSimilar on a word with no vocabulary should return empty.
	// (no words have been trained)
	similar = ae.FindSimilar("unknown", 5)
	if len(similar) != 0 {
		t.Errorf("expected empty results for unknown word with empty vocab, got %d results", len(similar))
	}
}
