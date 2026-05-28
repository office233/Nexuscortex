package cortex

import (
	"sort"
	"strings"
)

// ─────────────────────────────────────────────────────────────────────
// Chain-of-Thought + Self-Consistency
// ─────────────────────────────────────────────────────────────────────
//
// Two cheap quality boosters that sit on top of Broca 2.0 without any
// retraining:
//
//   • CoT prompt prefix: we prepend a "Let's think step by step." style
//     primer in the prompt so the model is biased toward producing
//     intermediate reasoning tokens before its final answer.
//
//   • Self-Consistency: we sample N independent completions at slightly
//     varied temperatures and vote on the answer that the majority of
//     samples agree on. Even a 5M-param model gains measurable accuracy
//     on factual / numerical questions from this, because the modes of
//     the output distribution tend to be correct more often than any
//     single high-temperature sample.
//
// Both are PURE additions — the existing single-shot
// GenerateWithTransformer path remains untouched. Callers opt in via
// GenerateWithSelfConsistency / BuildCoTPrompt.

// CoTConfig controls Chain-of-Thought + Self-Consistency behaviour.
// Zero value means "disabled" — callers must pass a populated config.
type CoTConfig struct {
	// Samples is the number of independent generations to draw before
	// voting. 1 = effectively disabled (single-shot, no voting).
	// 3-5 is a sweet spot: meaningful quality gain, modest cost.
	Samples int

	// MaxTokens is the per-sample token budget passed to the transformer.
	MaxTokens int

	// BaseTemperature is the starting temperature for sample 0; subsequent
	// samples spread linearly around it (±0.2) to encourage diversity
	// without going completely off-distribution.
	BaseTemperature float32

	// TopK passed through to the transformer's top-K sampling.
	TopK int

	// MinTokens, if > 0, suppresses the EOS token for the first
	// MinTokens emitted tokens. Useful when the model has learned to
	// emit EOS too early after short prompts. Default 0 = no
	// suppression (backward-compatible).
	MinTokens int

	// UseCoTPrimer toggles the "let's think step by step" prefix.
	UseCoTPrimer bool

	// CoTPrimer overrides the default primer text. Leave empty to use
	// the built-in bilingual default.
	CoTPrimer string
}

// DefaultCoTConfig returns sensible defaults for a model in the 5-50M
// range. Larger models can afford more samples; smaller models still
// benefit but should keep Samples low to stay responsive.
func DefaultCoTConfig() CoTConfig {
	return CoTConfig{
		Samples:         3,
		MaxTokens:       40,
		BaseTemperature: 0.8,
		TopK:            40,
		UseCoTPrimer:    true,
	}
}

// defaultCoTPrimer is bilingual (RO + EN) on purpose — the model was
// trained on mixed Dolly/Alpaca instruction data and the user writes
// Romanian frequently. Either keyword should bias generation toward
// step-by-step reasoning without dominating short answers.
const defaultCoTPrimer = "Let's think step by step. Sa gandim pas cu pas. "

// BuildCoTPrompt assembles the final prompt string from optional memory
// context, CoT primer, and the actual context words. Mirrors the format
// used by GenerateWithTransformer so swapping in CoT is transparent for
// the downstream tokenizer.
func BuildCoTPrompt(memoryContext string, contextWords []string, cfg CoTConfig) string {
	var b strings.Builder
	if memoryContext != "" {
		b.WriteString(memoryContext)
		b.WriteString(" | ")
	}
	if cfg.UseCoTPrimer {
		if cfg.CoTPrimer != "" {
			b.WriteString(cfg.CoTPrimer)
		} else {
			b.WriteString(defaultCoTPrimer)
		}
	}
	b.WriteString(strings.Join(contextWords, " "))
	return b.String()
}

// GenerateWithSelfConsistency draws cfg.Samples completions from the
// transformer, normalises each result, and returns the one with the
// highest vote count (ties broken by occurrence order). When
// cfg.Samples <= 1, falls back to a single-shot generation equivalent
// to the existing GenerateWithTransformer path.
//
// Returns ("", false) only when the transformer / tokenizer is nil or
// every sample produced empty output — callers should then fall back
// to the regular Broca path.
func GenerateWithSelfConsistency(
	transformer *MiniTransformer,
	tokenizer *BPETokenizer,
	memoryContext string,
	contextWords []string,
	cfg CoTConfig,
) (string, bool) {
	if transformer == nil || tokenizer == nil || len(contextWords) == 0 {
		return "", false
	}
	if cfg.Samples < 1 {
		cfg.Samples = 1
	}
	if cfg.MaxTokens <= 0 {
		cfg.MaxTokens = 40
	}
	if cfg.TopK <= 0 {
		cfg.TopK = 40
	}
	if cfg.BaseTemperature <= 0 {
		cfg.BaseTemperature = 0.8
	}

	prompt := BuildCoTPrompt(memoryContext, contextWords, cfg)
	promptIDs := tokenizer.Encode(prompt)
	if len(promptIDs) == 0 {
		return "", false
	}
	input := make([]int, 0, len(promptIDs)+1)
	input = append(input, tokenizer.BosID())
	input = append(input, promptIDs...)

	// Spread temperatures evenly around BaseTemperature for diversity.
	// Sample 0 gets the base; subsequent samples alternate above/below.
	temps := makeTemperatureSpread(cfg.BaseTemperature, cfg.Samples)

	candidates := make([]string, 0, cfg.Samples)
	for i := 0; i < cfg.Samples; i++ {
		out := transformer.GenerateFastMin(input, cfg.MaxTokens, cfg.MinTokens, temps[i], cfg.TopK)
		if len(out) <= len(input) {
			continue
		}
		gen := out[len(input):]
		text := strings.TrimSpace(tokenizer.Decode(gen))
		text = stripCoTPrimerEcho(text)
		if text == "" || text == "<UNK>" {
			continue
		}
		candidates = append(candidates, text)
	}
	if len(candidates) == 0 {
		return "", false
	}

	winner := voteMajority(candidates)
	return winner, true
}

// makeTemperatureSpread returns n temperatures centred on base, fanning
// out in ±0.2 increments. n=1 returns just [base]. Values are clamped
// to a sane (0.3, 1.5) range so we don't go fully greedy or fully chaotic.
func makeTemperatureSpread(base float32, n int) []float32 {
	out := make([]float32, n)
	if n == 1 {
		out[0] = base
		return out
	}
	step := float32(0.2)
	for i := 0; i < n; i++ {
		var t float32
		switch {
		case i == 0:
			t = base
		case i%2 == 1:
			t = base + step*float32((i+1)/2)
		default:
			t = base - step*float32(i/2)
		}
		if t < 0.3 {
			t = 0.3
		}
		if t > 1.5 {
			t = 1.5
		}
		out[i] = t
	}
	return out
}

// stripCoTPrimerEcho removes a leading echo of the CoT primer that the
// model sometimes regurgitates verbatim before its actual answer.
// Conservative: only strips when the primer appears as a clean prefix.
func stripCoTPrimerEcho(text string) string {
	primers := []string{
		"let's think step by step.",
		"lets think step by step.",
		"sa gandim pas cu pas.",
		"să gândim pas cu pas.",
	}
	lower := strings.ToLower(text)
	for _, p := range primers {
		if strings.HasPrefix(lower, p) {
			text = strings.TrimSpace(text[len(p):])
			lower = strings.ToLower(text)
		}
	}
	return text
}

// voteMajority returns the candidate with the highest occurrence count,
// using normalised (lowercased, whitespace-collapsed) keys for grouping
// but returning the FIRST original-form occurrence of the winning group.
// Single-occurrence ties are resolved by insertion order (i.e. the
// first sample wins), which matches the bias humans expect from a
// "first answer is usually fine" heuristic.
func voteMajority(candidates []string) string {
	if len(candidates) == 0 {
		return ""
	}
	type bucket struct {
		count    int
		firstIdx int
		display  string
	}
	groups := make(map[string]*bucket, len(candidates))
	for i, c := range candidates {
		key := normaliseAnswer(c)
		if b, ok := groups[key]; ok {
			b.count++
			continue
		}
		groups[key] = &bucket{count: 1, firstIdx: i, display: c}
	}
	// Sort by (count desc, firstIdx asc) for stable winner selection.
	keys := make([]string, 0, len(groups))
	for k := range groups {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		a, b := groups[keys[i]], groups[keys[j]]
		if a.count != b.count {
			return a.count > b.count
		}
		return a.firstIdx < b.firstIdx
	})
	return groups[keys[0]].display
}

// normaliseAnswer collapses whitespace and lowercases for vote-grouping.
// Punctuation is intentionally kept so "42" and "42." don't merge —
// callers can pre-clean if they want looser grouping.
func normaliseAnswer(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	return strings.Join(strings.Fields(s), " ")
}
