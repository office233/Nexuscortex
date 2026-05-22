package cortex

import (
	"testing"
)

func TestReasoningArithmetic(t *testing.T) {
	r := NewReasoningEngine()

	tests := []struct {
		input    string
		expected string
	}{
		{"What is 15 + 27?", "42"},
		{"what is 10 - 3?", "7"},
		{"what is 6 * 7?", "42"},
		{"what is 100 / 4?", "25"},
		{"what is 5 plus 3?", "8"},
		{"what is 10 minus 4?", "6"},
		{"what is 3 times 9?", "27"},
		{"what is 20 divided by 5?", "4"},
		{"Calculate 123 + 456", "579"},
	}

	for _, tc := range tests {
		answer, ok := r.TryReason(tc.input)
		if !ok {
			t.Errorf("TryReason(%q): expected reasoning to handle this, got ok=false", tc.input)
			continue
		}
		if answer != tc.expected {
			t.Errorf("TryReason(%q) = %q, want %q", tc.input, answer, tc.expected)
		}
	}
}

func TestReasoningWordProblems(t *testing.T) {
	r := NewReasoningEngine()

	tests := []struct {
		input    string
		expected string
	}{
		{"If I have 3 apples and give away 1, how many do I have?", "2"},
		{"There are 5 cats in a room. 2 leave. Then 3 more come in. How many cats?", "6"},
	}

	for _, tc := range tests {
		answer, ok := r.TryReason(tc.input)
		if !ok {
			t.Errorf("TryReason(%q): expected reasoning to handle this, got ok=false", tc.input)
			continue
		}
		if answer != tc.expected {
			t.Errorf("TryReason(%q) = %q, want %q", tc.input, answer, tc.expected)
		}
	}
}

func TestReasoningSequences(t *testing.T) {
	r := NewReasoningEngine()

	tests := []struct {
		input    string
		expected string
	}{
		{"What comes next: 2, 4, 8, 16, ?", "32"},
		{"What comes next: 3, 6, 9, 12, ?", "15"},
		{"What comes next: 1, 3, 5, 7, ?", "9"},
		{"What comes next: 1, 2, 4, 8, ?", "16"},
	}

	for _, tc := range tests {
		answer, ok := r.TryReason(tc.input)
		if !ok {
			t.Errorf("TryReason(%q): expected reasoning to handle this, got ok=false", tc.input)
			continue
		}
		if answer != tc.expected {
			t.Errorf("TryReason(%q) = %q, want %q", tc.input, answer, tc.expected)
		}
	}
}

func TestReasoningSyllogism(t *testing.T) {
	r := NewReasoningEngine()

	tests := []struct {
		input    string
		expected string
	}{
		{"All dogs are animals. Rex is a dog. Is Rex an animal?", "yes, Rex is an animal"},
		{"All cats are mammals. Luna is a cat. Is Luna a mammal?", "yes, Luna is a mammal"},
	}

	for _, tc := range tests {
		answer, ok := r.TryReason(tc.input)
		if !ok {
			t.Errorf("TryReason(%q): expected reasoning to handle this, got ok=false", tc.input)
			continue
		}
		if answer != tc.expected {
			t.Errorf("TryReason(%q) = %q, want %q", tc.input, answer, tc.expected)
		}
	}
}

func TestReasoningSorting(t *testing.T) {
	r := NewReasoningEngine()

	tests := []struct {
		input    string
		expected string
	}{
		{"Sort these numbers from smallest to largest: 7, 2, 9, 1, 5", "1 2 5 7 9"},
		{"Sort these numbers: 10, 3, 8, 1", "1 3 8 10"},
		{"Sort from largest to smallest: 1, 5, 3", "5 3 1"},
	}

	for _, tc := range tests {
		answer, ok := r.TryReason(tc.input)
		if !ok {
			t.Errorf("TryReason(%q): expected reasoning to handle this, got ok=false", tc.input)
			continue
		}
		if answer != tc.expected {
			t.Errorf("TryReason(%q) = %q, want %q", tc.input, answer, tc.expected)
		}
	}
}

func TestReasoningNotHandled(t *testing.T) {
	r := NewReasoningEngine()

	// These should NOT be handled by the reasoning engine.
	inputs := []string{
		"Hello, how are you?",
		"Tell me about the weather",
		"What is DNA?",
		"",
	}

	for _, input := range inputs {
		_, ok := r.TryReason(input)
		if ok {
			t.Errorf("TryReason(%q): expected ok=false for non-reasoning input", input)
		}
	}
}

func TestFormatNumber(t *testing.T) {
	tests := []struct {
		input    float64
		expected string
	}{
		{42, "42"},
		{3.14, "3.14"},
		{0, "0"},
		{-7, "-7"},
		{100.5, "100.5"},
	}

	for _, tc := range tests {
		result := formatNumber(tc.input)
		if result != tc.expected {
			t.Errorf("formatNumber(%v) = %q, want %q", tc.input, result, tc.expected)
		}
	}
}

func TestExtractNumbers(t *testing.T) {
	tests := []struct {
		input    string
		expected []float64
	}{
		{"I have 3 apples and 5 oranges", []float64{3, 5}},
		{"2, 4, 8, 16", []float64{2, 4, 8, 16}},
		{"no numbers here", nil},
		{"price is 10.5 dollars", []float64{10.5}},
	}

	for _, tc := range tests {
		result := extractNumbers(tc.input)
		if len(result) != len(tc.expected) {
			t.Errorf("extractNumbers(%q) returned %d numbers, want %d", tc.input, len(result), len(tc.expected))
			continue
		}
		for i, n := range result {
			if n != tc.expected[i] {
				t.Errorf("extractNumbers(%q)[%d] = %v, want %v", tc.input, i, n, tc.expected[i])
			}
		}
	}
}
