package cortex

import (
	"strings"
	"testing"
)

func TestBuildCoTPrompt_WithMemoryAndPrimer(t *testing.T) {
	cfg := DefaultCoTConfig()
	got := BuildCoTPrompt("paris is the capital of france", []string{"what", "is", "the", "capital"}, cfg)
	if !strings.Contains(got, "paris is the capital of france | ") {
		t.Errorf("memory not injected as prefix: %q", got)
	}
	if !strings.Contains(strings.ToLower(got), "step by step") {
		t.Errorf("CoT primer missing: %q", got)
	}
	if !strings.HasSuffix(got, "what is the capital") {
		t.Errorf("context words not at tail: %q", got)
	}
}

func TestBuildCoTPrompt_NoMemoryNoPrimer(t *testing.T) {
	cfg := CoTConfig{Samples: 1, UseCoTPrimer: false}
	got := BuildCoTPrompt("", []string{"hello", "world"}, cfg)
	if got != "hello world" {
		t.Errorf("expected 'hello world', got %q", got)
	}
}

func TestBuildCoTPrompt_CustomPrimer(t *testing.T) {
	cfg := CoTConfig{Samples: 1, UseCoTPrimer: true, CoTPrimer: "REASON: "}
	got := BuildCoTPrompt("", []string{"q"}, cfg)
	if !strings.HasPrefix(got, "REASON: ") {
		t.Errorf("custom primer not honoured: %q", got)
	}
}

func TestMakeTemperatureSpread(t *testing.T) {
	// n=1 returns just base
	out := makeTemperatureSpread(0.8, 1)
	if len(out) != 1 || out[0] != 0.8 {
		t.Errorf("n=1 unexpected: %v", out)
	}

	// n=3 fans out around base
	out = makeTemperatureSpread(0.8, 3)
	if len(out) != 3 {
		t.Fatalf("expected 3 temps, got %d", len(out))
	}
	if out[0] != 0.8 {
		t.Errorf("first temp should be base, got %f", out[0])
	}
	// All temps within sane bounds
	for i, tv := range out {
		if tv < 0.3 || tv > 1.5 {
			t.Errorf("temp %d=%f out of bounds", i, tv)
		}
	}

	// n=5 produces unique spread
	out = makeTemperatureSpread(0.9, 5)
	if len(out) != 5 {
		t.Fatalf("expected 5 temps, got %d", len(out))
	}
	seen := make(map[float32]bool)
	for _, tv := range out {
		seen[tv] = true
	}
	if len(seen) < 3 {
		t.Errorf("expected diverse temperatures, got %v", out)
	}
}

func TestVoteMajority_MajorityWins(t *testing.T) {
	candidates := []string{
		"the answer is 42",
		"forty two",
		"the answer is 42",
		"42",
		"the answer is 42",
	}
	winner := voteMajority(candidates)
	if winner != "the answer is 42" {
		t.Errorf("expected majority winner, got %q", winner)
	}
}

func TestVoteMajority_TieGoesToFirst(t *testing.T) {
	candidates := []string{
		"answer A",
		"answer B",
		"answer C",
	}
	// All unique → all tied at count 1 → first wins
	winner := voteMajority(candidates)
	if winner != "answer A" {
		t.Errorf("expected first on tie, got %q", winner)
	}
}

func TestVoteMajority_Empty(t *testing.T) {
	if got := voteMajority(nil); got != "" {
		t.Errorf("expected empty for nil, got %q", got)
	}
	if got := voteMajority([]string{}); got != "" {
		t.Errorf("expected empty for empty slice, got %q", got)
	}
}

func TestVoteMajority_NormaliseGrouping(t *testing.T) {
	// "Hello" / "hello" / "  hello  " should group together by lowercase + trim.
	candidates := []string{
		"Hello",
		"different",
		"hello",
		"  hello  ",
	}
	winner := voteMajority(candidates)
	// "Hello" is the first form of the winning group → returned as-is.
	if winner != "Hello" {
		t.Errorf("expected 'Hello' as group representative, got %q", winner)
	}
}

func TestStripCoTPrimerEcho(t *testing.T) {
	cases := []struct {
		in     string
		expect string
	}{
		{"Let's think step by step. The answer is 42.", "The answer is 42."},
		{"lets think step by step. yes", "yes"},
		{"Sa gandim pas cu pas. raspunsul e da", "raspunsul e da"},
		{"no primer here", "no primer here"},
		{"", ""},
	}
	for _, c := range cases {
		got := stripCoTPrimerEcho(c.in)
		if got != c.expect {
			t.Errorf("strip(%q) = %q, want %q", c.in, got, c.expect)
		}
	}
}

func TestGenerateWithSelfConsistency_NilGuards(t *testing.T) {
	// nil transformer → clean miss
	_, ok := GenerateWithSelfConsistency(nil, nil, "", []string{"x"}, DefaultCoTConfig())
	if ok {
		t.Error("expected false on nil transformer")
	}
}

func TestDefaultCoTConfig_SaneValues(t *testing.T) {
	cfg := DefaultCoTConfig()
	if cfg.Samples < 1 || cfg.Samples > 10 {
		t.Errorf("Samples out of sane range: %d", cfg.Samples)
	}
	if cfg.BaseTemperature < 0.3 || cfg.BaseTemperature > 1.5 {
		t.Errorf("BaseTemperature out of sane range: %f", cfg.BaseTemperature)
	}
	if !cfg.UseCoTPrimer {
		t.Error("default should enable CoT primer")
	}
}
