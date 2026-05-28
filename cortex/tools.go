package cortex

import (
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ─────────────────────────────────────────────────────────────────────
// Tool Registry — extensible deterministic skills
// ─────────────────────────────────────────────────────────────────────
//
// Tool is a pluggable deterministic capability that ReasoningEngine can
// dispatch to before falling back to the neural pipeline. Each Tool
// decides for itself whether an input matches its trigger pattern via
// Match(); the registry walks tools in priority order and returns the
// first successful Execute().
//
// Adding a new capability is now O(1): write a struct that satisfies
// Tool, register it once, done. Existing hardcoded skills in
// ReasoningEngine remain as the fallback path and are unaffected.

// Tool is the contract for a registered deterministic skill.
type Tool interface {
	// Name returns a short stable identifier (used for logging/debug).
	Name() string

	// Match reports whether the (already lowercased) input looks like
	// something this tool can handle. Should be cheap — regex or
	// substring tests only, no heavy computation.
	Match(lowerInput string) bool

	// Execute produces the deterministic answer. Receives the ORIGINAL
	// (case-preserved) input so it can extract precise tokens.
	// Returns ("", false) if execution fails mid-way (e.g. unparseable
	// numbers) so the dispatcher can try the next tool.
	Execute(originalInput string) (string, bool)
}

// ToolRegistry holds an ordered list of tools and dispatches incoming
// inputs to the first matching one. Safe for concurrent reads after
// initial registration; Register() should only run at startup.
type ToolRegistry struct {
	mu    sync.RWMutex
	tools []Tool
}

// NewToolRegistry creates an empty registry.
func NewToolRegistry() *ToolRegistry {
	return &ToolRegistry{}
}

// Register appends a tool to the dispatch list. Order matters: tools
// registered earlier are matched first. Register more specific tools
// before more general ones.
func (r *ToolRegistry) Register(t Tool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.tools = append(r.tools, t)
}

// Dispatch finds the first tool whose Match() succeeds and returns its
// Execute() result. Returns ("", "", false) when no tool fires.
// The returned string-1 is the answer; string-2 is the tool name (for
// logging / confidence-weighting upstream).
func (r *ToolRegistry) Dispatch(input string) (answer string, toolName string, ok bool) {
	input = strings.TrimSpace(input)
	if input == "" {
		return "", "", false
	}
	lower := strings.ToLower(input)

	r.mu.RLock()
	defer r.mu.RUnlock()
	for _, t := range r.tools {
		if !t.Match(lower) {
			continue
		}
		if ans, ok := t.Execute(input); ok {
			return ans, t.Name(), true
		}
	}
	return "", "", false
}

// Names returns the registered tool names in dispatch order. Mainly for
// debugging / introspection from the dashboard.
func (r *ToolRegistry) Names() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]string, len(r.tools))
	for i, t := range r.tools {
		out[i] = t.Name()
	}
	return out
}

// ─────────────────────────────────────────────────────────────────────
// Built-in tools
// ─────────────────────────────────────────────────────────────────────

// DateTimeTool answers "what time is it", "what is the date", "what day
// is today" — anything the user could plausibly ask about the wall clock.
type DateTimeTool struct{}

func (DateTimeTool) Name() string { return "datetime" }

var reDateTimeTrigger = regexp.MustCompile(
	`\b(time|hour|clock|date|day|today|tomorrow|yesterday|year|month|now|ora|ce ora|ce zi|data|azi|maine|ieri|anul|luna)\b`,
)

func (DateTimeTool) Match(lower string) bool {
	return reDateTimeTrigger.MatchString(lower)
}

func (DateTimeTool) Execute(input string) (string, bool) {
	lower := strings.ToLower(input)
	now := time.Now()

	// Romanian-friendly answers first (since the user often writes RO).
	switch {
	case strings.Contains(lower, "ora") || strings.Contains(lower, "ce ora") ||
		strings.Contains(lower, "what time") || strings.Contains(lower, "hour") ||
		strings.Contains(lower, "clock"):
		return now.Format("15:04:05"), true
	case strings.Contains(lower, "ce zi") || strings.Contains(lower, "what day") ||
		strings.Contains(lower, "today"):
		return now.Format("Monday, 2 January 2006"), true
	case strings.Contains(lower, "tomorrow") || strings.Contains(lower, "maine"):
		return now.AddDate(0, 0, 1).Format("Monday, 2 January 2006"), true
	case strings.Contains(lower, "yesterday") || strings.Contains(lower, "ieri"):
		return now.AddDate(0, 0, -1).Format("Monday, 2 January 2006"), true
	case strings.Contains(lower, "year") || strings.Contains(lower, "anul"):
		return strconv.Itoa(now.Year()), true
	case strings.Contains(lower, "month") || strings.Contains(lower, "luna"):
		return now.Format("January"), true
	case strings.Contains(lower, "date") || strings.Contains(lower, "data"):
		return now.Format("2 January 2006"), true
	case strings.Contains(lower, "now"):
		return now.Format("2006-01-02 15:04:05"), true
	}
	// Triggered but unrecognised subtype — let neural pipeline try.
	return "", false
}

// UnitConvertTool handles simple unit conversions: km↔mi, kg↔lb, C↔F,
// m↔ft. Pattern: "convert N <from> to <to>" or "N <from> in <to>".
type UnitConvertTool struct{}

func (UnitConvertTool) Name() string { return "unitconvert" }

var reUnitConvert = regexp.MustCompile(
	`(?i)([-+]?\d+(?:\.\d+)?)\s*(km|mi|miles?|kilometers?|kg|lbs?|pounds?|kilograms?|c|f|celsius|fahrenheit|m|ft|feet|meters?)\b.*?\b(?:to|in|en|spre)\b\s*(km|mi|miles?|kilometers?|kg|lbs?|pounds?|kilograms?|c|f|celsius|fahrenheit|m|ft|feet|meters?)\b`,
)

func (UnitConvertTool) Match(lower string) bool {
	return reUnitConvert.MatchString(lower)
}

func (UnitConvertTool) Execute(input string) (string, bool) {
	m := reUnitConvert.FindStringSubmatch(input)
	if len(m) != 4 {
		return "", false
	}
	val, err := strconv.ParseFloat(m[1], 64)
	if err != nil {
		return "", false
	}
	from := normUnit(m[2])
	to := normUnit(m[3])
	if from == to {
		return fmt.Sprintf("%.4g %s", val, to), true
	}
	out, ok := convertUnit(val, from, to)
	if !ok {
		return "", false
	}
	return fmt.Sprintf("%.4g %s", out, to), true
}

func normUnit(u string) string {
	u = strings.ToLower(strings.TrimSpace(u))
	switch u {
	case "kilometer", "kilometers":
		return "km"
	case "mile", "miles":
		return "mi"
	case "kilogram", "kilograms":
		return "kg"
	case "lb", "lbs", "pound", "pounds":
		return "lb"
	case "celsius":
		return "c"
	case "fahrenheit":
		return "f"
	case "meter", "meters":
		return "m"
	case "ft", "feet":
		return "ft"
	}
	return u
}

func convertUnit(v float64, from, to string) (float64, bool) {
	type pair struct{ from, to string }
	switch (pair{from, to}) {
	// Distance
	case pair{"km", "mi"}:
		return v * 0.621371, true
	case pair{"mi", "km"}:
		return v * 1.609344, true
	case pair{"m", "ft"}:
		return v * 3.28084, true
	case pair{"ft", "m"}:
		return v * 0.3048, true
	case pair{"km", "m"}:
		return v * 1000, true
	case pair{"m", "km"}:
		return v / 1000, true
	// Mass
	case pair{"kg", "lb"}:
		return v * 2.20462, true
	case pair{"lb", "kg"}:
		return v * 0.453592, true
	// Temperature
	case pair{"c", "f"}:
		return v*9/5 + 32, true
	case pair{"f", "c"}:
		return (v - 32) * 5 / 9, true
	}
	return 0, false
}

// HippoSearchTool exposes a "search your memory for X" style query so
// the user can explicitly probe the Hippocampus without relying on the
// implicit recall step. It needs a back-reference to the Organism, so
// it's constructed with one rather than being a zero-value struct.
type HippoSearchTool struct {
	hippocampus *Hippocampus
	wernicke    *Wernicke
	brain       *Brain
}

// NewHippoSearchTool wires the tool to a live Organism's components.
// All three references must be non-nil.
func NewHippoSearchTool(h *Hippocampus, w *Wernicke, b *Brain) *HippoSearchTool {
	return &HippoSearchTool{hippocampus: h, wernicke: w, brain: b}
}

func (*HippoSearchTool) Name() string { return "hippo_search" }

var reHippoTrigger = regexp.MustCompile(
	`(?i)\b(remember|recall|search memory|look up|cauta in memorie|tine minte|iti amintesti)\b`,
)

func (t *HippoSearchTool) Match(lower string) bool {
	return reHippoTrigger.MatchString(lower)
}

func (t *HippoSearchTool) Execute(input string) (string, bool) {
	if t.hippocampus == nil || t.wernicke == nil {
		return "", false
	}
	// Strip trigger words so the remainder is just the query body.
	q := reHippoTrigger.ReplaceAllString(input, "")
	q = strings.TrimSpace(strings.Trim(q, "?:,.\""))
	if q == "" {
		return "", false
	}
	understanding := t.wernicke.Understand(q)
	tokens := Tokenize(q)
	threshold := 1
	if len(tokens) >= 2 {
		threshold = 2
	}
	// Try expanded (Brain-association) recall first, then plain.
	if t.brain != nil {
		if mem, score, ok := t.hippocampus.RecallByKeywordsExpanded(
			tokens, threshold, understanding.Combined, t.brain,
		); ok && score > 0 {
			if ans := extractAnswerFromContext(mem.Context); ans != "" {
				return ans, true
			}
			return mem.Context, true
		}
	}
	if mem, score, ok := t.hippocampus.RecallByKeywords(
		tokens, threshold, understanding.Combined,
	); ok && score > 0 {
		if ans := extractAnswerFromContext(mem.Context); ans != "" {
			return ans, true
		}
		return mem.Context, true
	}
	return "", false
}

// MathSquareRootTool handles "square root of N" / "sqrt(N)" — the
// existing arithmetic skill in ReasoningEngine can't parse sqrt because
// govaluate has no built-in for it. This fills the gap without forking
// govaluate.
type MathSquareRootTool struct{}

func (MathSquareRootTool) Name() string { return "math_sqrt" }

var reSqrt = regexp.MustCompile(`(?i)\b(?:sqrt|square root of|radacina patrata din)\s*[\(]?\s*([-+]?\d+(?:\.\d+)?)\s*[\)]?`)

func (MathSquareRootTool) Match(lower string) bool {
	return reSqrt.MatchString(lower)
}

func (MathSquareRootTool) Execute(input string) (string, bool) {
	m := reSqrt.FindStringSubmatch(input)
	if len(m) != 2 {
		return "", false
	}
	v, err := strconv.ParseFloat(m[1], 64)
	if err != nil || v < 0 {
		return "", false
	}
	out := math.Sqrt(v)
	if out == math.Trunc(out) {
		return strconv.FormatFloat(out, 'f', 0, 64), true
	}
	return strconv.FormatFloat(out, 'f', 4, 64), true
}

// MathPowerTool handles "N to the power of M" / "N^M" / "pow(N,M)".
type MathPowerTool struct{}

func (MathPowerTool) Name() string { return "math_pow" }

var rePow = regexp.MustCompile(`(?i)([-+]?\d+(?:\.\d+)?)\s*(?:\^|\*\*|to the power of|la puterea)\s*([-+]?\d+(?:\.\d+)?)`)

func (MathPowerTool) Match(lower string) bool {
	return rePow.MatchString(lower)
}

func (MathPowerTool) Execute(input string) (string, bool) {
	m := rePow.FindStringSubmatch(input)
	if len(m) != 3 {
		return "", false
	}
	base, err1 := strconv.ParseFloat(m[1], 64)
	exp, err2 := strconv.ParseFloat(m[2], 64)
	if err1 != nil || err2 != nil {
		return "", false
	}
	out := math.Pow(base, exp)
	if math.IsInf(out, 0) || math.IsNaN(out) {
		return "", false
	}
	if out == math.Trunc(out) && math.Abs(out) < 1e15 {
		return strconv.FormatFloat(out, 'f', 0, 64), true
	}
	return strconv.FormatFloat(out, 'f', 6, 64), true
}
