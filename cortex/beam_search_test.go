package cortex

import (
	"math/rand"
	"strings"
	"testing"
)

// TestBeamSearchBasic trains the brain with known text and verifies
// that beam search produces coherent output from a prompt.
func TestBeamSearchBasic(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	cfg := DefaultConfig()
	cfg.BeamSearchWidth = 3
	cfg.BeamSearchMaxCandidatesPerStep = 5
	b := NewBrain("beam_test.nxbrain", "beam_vocab.json", rng, cfg)

	// Train with repetitive text to build strong associations.
	for i := 0; i < 10; i++ {
		b.Learn("the cat sat on the mat")
		b.Learn("the dog ran in the park")
	}

	result := b.GenerateBeam("the cat", 5)
	if result == "" {
		t.Fatal("beam search returned empty result for trained brain")
	}

	// The prompt words should NOT appear in the output (same contract as Generate).
	if strings.Contains(result, "the cat") {
		t.Errorf("output should not echo prompt, got %q", result)
	}

	// Should produce at least one word.
	words := strings.Fields(result)
	if len(words) == 0 {
		t.Error("beam search produced no words")
	}
	t.Logf("BeamSearchBasic output: %q", result)
}

// TestBeamSearchBeatsGreedy verifies that beam search can produce
// a higher-scoring (or equally coherent) output compared to greedy.
// We set up a deterministic scenario where beam search explores a
// longer-scoring path.
func TestBeamSearchBeatsGreedy(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	cfg := DefaultConfig()
	cfg.BeamSearchWidth = 5
	cfg.BeamSearchMaxCandidatesPerStep = 10
	b := NewBrain("beam_greedy.nxbrain", "beam_greedy_vocab.json", rng, cfg)

	// Train a sequence where the greedy first choice leads to a dead end
	// but a slightly weaker first choice leads to a longer chain.
	// "alpha beta gamma delta" — strong full chain (10 repetitions).
	for i := 0; i < 10; i++ {
		b.Learn("alpha beta gamma delta epsilon")
	}

	// "alpha zeta" — slightly stronger bigram but dead end.
	for i := 0; i < 12; i++ {
		b.Learn("alpha zeta")
	}

	beamResult := b.GenerateBeam("alpha", 5)
	greedyResult := b.Generate("alpha", 5)

	beamWords := strings.Fields(beamResult)
	greedyWords := strings.Fields(greedyResult)

	t.Logf("Greedy: %q (%d words)", greedyResult, len(greedyWords))
	t.Logf("Beam:   %q (%d words)", beamResult, len(beamWords))

	// Beam search should produce at least as many words as greedy.
	// With the dead-end setup, beam search should find the longer chain.
	if len(beamWords) < len(greedyWords) {
		t.Errorf("beam search produced fewer words (%d) than greedy (%d); "+
			"beam=%q, greedy=%q", len(beamWords), len(greedyWords), beamResult, greedyResult)
	}
}

// TestBeamSearchEmptyBrain verifies graceful handling when the brain
// has no learned data at all.
func TestBeamSearchEmptyBrain(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	cfg := DefaultConfig()
	b := NewBrain("beam_empty.nxbrain", "beam_empty_vocab.json", rng, cfg)

	// Empty brain — no learned text.
	result := b.GenerateBeam("hello world", 10)
	if result != "" {
		t.Errorf("expected empty result from empty brain, got %q", result)
	}

	// Empty prompt.
	result = b.GenerateBeam("", 10)
	if result != "" {
		t.Errorf("expected empty result from empty prompt, got %q", result)
	}
}
