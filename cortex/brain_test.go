package cortex

import (
	"math/rand"
	"strings"
	"testing"
)

func TestBrainContextIsolationAndEcho(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	b := NewBrain("brain_test.nxbrain", "vocab_test.json", rng, DefaultConfig())

	// Teach the brain a simple pattern.
	b.Learn("where do neurons fire?")

	// Prompting with the last word.
	// Since "where do neurons fire?" has words: "where", "do", "neurons", "fire", "?"
	// Let's generate from "where do neurons"
	output := b.Generate("where do neurons", 3)

	// Verify that the prompt "where do neurons" is NOT prepended as output.
	// The output should be the next predicted word(s) like "fire ?" or similar,
	// but must NOT be "where do neurons".
	if strings.Contains(output, "where do neurons") {
		t.Errorf("output contains the prompt words! Prompt must be context, not output. Got: %q", output)
	}

	// Clean up files if created.
	// These are in memory unless b.Save() was called, which we didn't call.
}
