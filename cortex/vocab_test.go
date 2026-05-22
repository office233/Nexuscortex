package cortex

import (
	"testing"
)

func TestTokenizeUnicode(t *testing.T) {
	text := "ce face câinele? O fată mănâncă o prăjitură șmecheră."
	expected := []string{
		"ce", "face", "câinele", "?",
		"o", "fată", "mănâncă", "o", "prăjitură", "șmecheră", ".",
	}

	tokens := Tokenize(text)
	if len(tokens) != len(expected) {
		t.Fatalf("expected %d tokens, got %d: %v", len(expected), len(tokens), tokens)
	}

	for i, tok := range tokens {
		if tok != expected[i] {
			t.Errorf("token %d: expected %q, got %q", i, expected[i], tok)
		}
	}
}

func TestVocabBasic(t *testing.T) {
	v := NewVocab()

	id1 := v.GetOrCreate("câine")
	if id1 == 0 {
		t.Fatal("expected non-zero ID for new word")
	}

	id2 := v.GetOrCreate("câine")
	if id1 != id2 {
		t.Fatalf("expected identical IDs for same word, got %d and %d", id1, id2)
	}

	decoded := v.Decode(id1)
	if decoded != "câine" {
		t.Errorf("expected decoded word to be 'câine', got %q", decoded)
	}

	if v.Get("câine") != id1 {
		t.Errorf("expected Get('câine') to return %d, got %d", id1, v.Get("câine"))
	}

	if v.Get("pisică") != 0 {
		t.Errorf("expected Get for unknown word to return 0, got %d", v.Get("pisică"))
	}
}

func TestVocabOverflowFails(t *testing.T) {
	v := NewVocab()

	// Pretend vocab is full (NextID reaches 4294967295)
	v.NextID = 4294967295

	// Attempting to register a new word under the old method returns 0 (silent <UNK>).
	idOld := v.GetOrCreate("newword")
	if idOld != 0 {
		t.Errorf("expected old method to return 0, got %d", idOld)
	}

	// We want to verify that using a checked method returns an explicit error.
	_, err := v.GetOrCreateChecked("newword")
	if err == nil {
		t.Error("expected error when vocabulary is full, but got nil")
	} else {
		expectedErr := "vocabulary full (limit 4294967295)"
		if err.Error() != expectedErr {
			t.Errorf("expected error message %q, got %q", expectedErr, err.Error())
		}
	}
}
