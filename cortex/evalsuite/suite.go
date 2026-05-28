// Package evalsuite provides the curated correctness-grading suite used
// by both the broca-eval CLI and the in-training auto-eval hook.
//
// Keep this package free of UI / IO concerns: the suite + scoring are
// pure data and pure functions, so they can be linked into the trainer
// without dragging in compute / organism setup.
package evalsuite

// ScoreMode determines how a generated answer is compared to expectations.
type ScoreMode string

const (
	// ModeContains: answer is correct iff the generation (lowercased)
	// contains ALL of the expected strings (also lowercased).
	ModeContains ScoreMode = "contains"

	// ModeContainsAny: correct iff ANY of the expected strings is found.
	ModeContainsAny ScoreMode = "contains_any"

	// ModeNumeric: extract the first number from the generation and
	// compare to ExpectedNumber within Tolerance.
	ModeNumeric ScoreMode = "numeric"

	// ModeRegex: correct iff Expected[0] (compiled as a regex) matches.
	ModeRegex ScoreMode = "regex"
)

// Task is a single eval item. Each task is graded independently; the
// suite-level score is the unweighted accuracy across all tasks.
type Task struct {
	ID             string    `json:"id"`
	Category       string    `json:"category"`
	Prompt         string    `json:"prompt"`
	Expected       []string  `json:"expected"`                  // strings for contains*/regex modes
	ExpectedNumber float64   `json:"expected_number,omitempty"` // for numeric mode
	Tolerance      float64   `json:"tolerance,omitempty"`       // absolute, for numeric
	Mode           ScoreMode `json:"mode"`
}

// Standard is the canonical eval set used to compare checkpoints.
// Keep small (<50 items) so a full run completes in seconds even on
// CPU; we care about trends across runs, not absolute SOTA numbers.
var Standard = []Task{
	// ── Factual recall (single-shot knowledge) ───────────────────────
	{ID: "fact_capital_france", Category: "factual",
		Prompt: "What is the capital of France?",
		Expected: []string{"paris"}, Mode: ModeContains},
	{ID: "fact_capital_japan", Category: "factual",
		Prompt: "What is the capital of Japan?",
		Expected: []string{"tokyo"}, Mode: ModeContains},
	{ID: "fact_capital_germany", Category: "factual",
		Prompt: "What is the capital of Germany?",
		Expected: []string{"berlin"}, Mode: ModeContains},
	{ID: "fact_romeo_juliet", Category: "factual",
		Prompt: "Who wrote Romeo and Juliet?",
		Expected: []string{"shakespeare"}, Mode: ModeContains},
	{ID: "fact_relativity", Category: "factual",
		Prompt: "Who developed the theory of relativity?",
		Expected: []string{"einstein"}, Mode: ModeContains},
	{ID: "fact_largest_planet", Category: "factual",
		Prompt: "What is the largest planet in our solar system?",
		Expected: []string{"jupiter"}, Mode: ModeContains},
	{ID: "fact_h2o", Category: "factual",
		Prompt: "What is the chemical formula for water?",
		Expected: []string{"h2o", "h₂o"}, Mode: ModeContainsAny},
	{ID: "fact_continents", Category: "factual",
		Prompt: "How many continents are there?",
		ExpectedNumber: 7, Tolerance: 0, Mode: ModeNumeric},

	// ── Math (deterministic — also covered by Reasoning tools) ──────
	{ID: "math_add_basic", Category: "math",
		Prompt: "What is 15 plus 27?",
		ExpectedNumber: 42, Tolerance: 0, Mode: ModeNumeric},
	{ID: "math_mul", Category: "math",
		Prompt: "What is 12 times 7?",
		ExpectedNumber: 84, Tolerance: 0, Mode: ModeNumeric},
	{ID: "math_sub", Category: "math",
		Prompt: "What is 100 minus 37?",
		ExpectedNumber: 63, Tolerance: 0, Mode: ModeNumeric},
	{ID: "math_div", Category: "math",
		Prompt: "What is 144 divided by 12?",
		ExpectedNumber: 12, Tolerance: 0, Mode: ModeNumeric},
	{ID: "math_sqrt", Category: "math",
		Prompt: "What is the square root of 81?",
		ExpectedNumber: 9, Tolerance: 0, Mode: ModeNumeric},
	{ID: "math_word_apples", Category: "math",
		Prompt: "Alice has 5 apples and gives 2 to Bob. How many does she have left?",
		ExpectedNumber: 3, Tolerance: 0, Mode: ModeNumeric},

	// ── Instruction following (format obedience) ────────────────────
	{ID: "instr_primary_colors", Category: "instruct",
		Prompt: "List the three primary colors.",
		Expected: []string{"red", "blue", "yellow"}, Mode: ModeContains},
	{ID: "instr_translate_hello_es", Category: "instruct",
		Prompt: "Translate 'hello' to Spanish.",
		Expected: []string{"hola"}, Mode: ModeContains},
	{ID: "instr_translate_good_morning", Category: "instruct",
		Prompt: "Translate 'good morning' to Spanish.",
		Expected: []string{"buenos d"}, Mode: ModeContains},
	{ID: "instr_define_photo", Category: "instruct",
		Prompt: "What is photosynthesis?",
		Expected: []string{"plant", "light"}, Mode: ModeContains},
	{ID: "instr_yesno", Category: "instruct",
		Prompt: "Is the Earth round? Answer yes or no.",
		Expected: []string{"yes"}, Mode: ModeContainsAny},

	// ── Reasoning (multi-step, benefits from CoT) ────────────────────
	{ID: "reason_seq_arith", Category: "reasoning",
		Prompt: "What number comes next in this sequence: 2, 4, 6, 8?",
		ExpectedNumber: 10, Tolerance: 0, Mode: ModeNumeric},
	{ID: "reason_seq_geom", Category: "reasoning",
		Prompt: "What number comes next: 1, 2, 4, 8, 16?",
		ExpectedNumber: 32, Tolerance: 0, Mode: ModeNumeric},
	{ID: "reason_syllogism", Category: "reasoning",
		Prompt: "All birds can fly. A penguin is a bird. Can a penguin fly? Answer yes or no.",
		Expected: []string{"yes", "no"}, Mode: ModeContainsAny},
	{ID: "reason_age", Category: "reasoning",
		Prompt: "If Anna is 10 years old and Mark is 5 years older than Anna, how old is Mark?",
		ExpectedNumber: 15, Tolerance: 0, Mode: ModeNumeric},
	{ID: "reason_compare", Category: "reasoning",
		Prompt: "Which is larger: 0.9 or 0.11?",
		Expected: []string{"0.9"}, Mode: ModeContains},
}

// Categories returns the distinct category labels present in tasks.
func Categories(tasks []Task) []string {
	seen := map[string]struct{}{}
	out := []string{}
	for _, t := range tasks {
		if _, ok := seen[t.Category]; ok {
			continue
		}
		seen[t.Category] = struct{}{}
		out = append(out, t.Category)
	}
	return out
}
