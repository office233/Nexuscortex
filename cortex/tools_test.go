package cortex

import (
	"strings"
	"testing"
)

func TestDateTimeTool_Match(t *testing.T) {
	tool := DateTimeTool{}
	yes := []string{
		"what time is it",
		"ce ora este",
		"what is the date today",
		"what year is it",
		"ce zi e azi",
	}
	no := []string{
		"how are you",
		"calculate 5+5",
		"sort 3 1 2",
	}
	for _, s := range yes {
		if !tool.Match(strings.ToLower(s)) {
			t.Errorf("expected match for %q", s)
		}
	}
	for _, s := range no {
		if tool.Match(strings.ToLower(s)) {
			t.Errorf("did not expect match for %q", s)
		}
	}
}

func TestDateTimeTool_Execute_Hour(t *testing.T) {
	tool := DateTimeTool{}
	ans, ok := tool.Execute("what time is it?")
	if !ok {
		t.Fatal("expected hour answer")
	}
	if len(ans) < 5 || !strings.Contains(ans, ":") {
		t.Errorf("hour format unexpected: %q", ans)
	}
}

func TestUnitConvertTool_Distance(t *testing.T) {
	tool := UnitConvertTool{}
	cases := []struct {
		in     string
		expect string // substring check
	}{
		{"convert 10 km to mi", "mi"},
		{"5 miles to km", "km"},
		{"convert 100 m to ft", "ft"},
	}
	for _, c := range cases {
		if !tool.Match(strings.ToLower(c.in)) {
			t.Errorf("no match for %q", c.in)
			continue
		}
		ans, ok := tool.Execute(c.in)
		if !ok {
			t.Errorf("execute failed for %q", c.in)
			continue
		}
		if !strings.Contains(ans, c.expect) {
			t.Errorf("answer %q missing %q (input: %q)", ans, c.expect, c.in)
		}
	}
}

func TestUnitConvertTool_Temperature(t *testing.T) {
	tool := UnitConvertTool{}
	ans, ok := tool.Execute("convert 100 c to f")
	if !ok {
		t.Fatal("c→f failed")
	}
	// 100C = 212F
	if !strings.HasPrefix(ans, "212") {
		t.Errorf("expected ~212 f, got %q", ans)
	}
}

func TestMathSquareRootTool(t *testing.T) {
	tool := MathSquareRootTool{}
	cases := []struct {
		in     string
		expect string
	}{
		{"sqrt(16)", "4"},
		{"square root of 25", "5"},
		{"radacina patrata din 144", "12"},
	}
	for _, c := range cases {
		if !tool.Match(strings.ToLower(c.in)) {
			t.Errorf("no match for %q", c.in)
			continue
		}
		ans, ok := tool.Execute(c.in)
		if !ok {
			t.Errorf("execute failed for %q", c.in)
			continue
		}
		if ans != c.expect {
			t.Errorf("expected %q for %q, got %q", c.expect, c.in, ans)
		}
	}
}

func TestMathPowerTool(t *testing.T) {
	tool := MathPowerTool{}
	cases := []struct {
		in     string
		expect string
	}{
		{"2^10", "1024"},
		{"3 to the power of 4", "81"},
		{"5 la puterea 3", "125"},
	}
	for _, c := range cases {
		ans, ok := tool.Execute(c.in)
		if !ok {
			t.Errorf("execute failed for %q", c.in)
			continue
		}
		if ans != c.expect {
			t.Errorf("expected %q for %q, got %q", c.expect, c.in, ans)
		}
	}
}

func TestToolRegistry_DispatchOrder(t *testing.T) {
	reg := NewToolRegistry()
	reg.Register(DateTimeTool{})
	reg.Register(MathSquareRootTool{})
	reg.Register(UnitConvertTool{})

	// sqrt should win over the others
	ans, name, ok := reg.Dispatch("sqrt(81)")
	if !ok {
		t.Fatal("expected dispatch hit on sqrt")
	}
	if name != "math_sqrt" {
		t.Errorf("expected math_sqrt, got %q", name)
	}
	if ans != "9" {
		t.Errorf("expected 9, got %q", ans)
	}

	// no match should miss cleanly
	_, _, ok = reg.Dispatch("hello world")
	if ok {
		t.Error("did not expect dispatch on 'hello world'")
	}

	// names exposed
	got := reg.Names()
	if len(got) != 3 {
		t.Errorf("expected 3 tools, got %d", len(got))
	}
}

func TestReasoningEngine_ToolsTakePriority(t *testing.T) {
	r := NewReasoningEngine()
	reg := NewToolRegistry()
	reg.Register(MathSquareRootTool{})
	r.AttachTools(reg)

	// "sqrt" is handled by the tool, not by the hardcoded arithmetic skill.
	ans, ok := r.TryReason("sqrt(49)")
	if !ok {
		t.Fatal("expected tool to answer sqrt")
	}
	if ans != "7" {
		t.Errorf("expected 7, got %q", ans)
	}

	// Hardcoded arithmetic still works when no tool matches.
	ans, ok = r.TryReason("what is 15 + 27")
	if !ok {
		t.Fatal("expected arithmetic fallback")
	}
	if !strings.Contains(ans, "42") {
		t.Errorf("expected 42, got %q", ans)
	}
}
