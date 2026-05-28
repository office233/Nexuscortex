// analyze-gen-length — citește toate fișierele JSON dintr-un director
// de evaluări auto-step*.json și raportează evoluția lungimii medii a
// generațiilor în timp, atât global cât și per categorie.
//
// Folosit pentru a investiga fenomenul de "generation length collapse"
// observat în cursa D (vezi docs/plans/2026-05-25-cursa-D-postmortem.md):
// lungimea medie a generațiilor factuale a scăzut de la 21.1 → 0.9
// caractere între step 10500 și step 30000.
//
// Întrebarea: e doar la factual sau e fenomen global? Empty rate crește
// monoton sau e oscilatoriu? Apare brusc sau gradual?
//
// Build & run:
//   go run scripts/analyze-gen-length.go
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

type evalResult struct {
	TaskID    string `json:"task_id"`
	Category  string `json:"category"`
	Generated string `json:"generated"`
}

type evalReport struct {
	Results []evalResult `json:"results"`
}

type stepStats struct {
	Step       int
	ByCategory map[string]*catStats
	Overall    *catStats
}

type catStats struct {
	N         int
	Empty     int
	TotalLen  int
	MaxLen    int
	NonEmpty  int
}

func (c *catStats) add(gen string) {
	c.N++
	l := len(gen)
	if l == 0 {
		c.Empty++
		return
	}
	c.NonEmpty++
	c.TotalLen += l
	if l > c.MaxLen {
		c.MaxLen = l
	}
}

func (c *catStats) avgAll() float64 {
	if c.N == 0 {
		return 0
	}
	return float64(c.TotalLen) / float64(c.N)
}

func (c *catStats) avgNonEmpty() float64 {
	if c.NonEmpty == 0 {
		return 0
	}
	return float64(c.TotalLen) / float64(c.NonEmpty)
}

func main() {
	dir := "data/cortex-auto/evals"
	if len(os.Args) > 1 {
		dir = os.Args[1]
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read dir: %v\n", err)
		os.Exit(2)
	}

	steps := []stepStats{}
	for _, e := range entries {
		if !strings.HasPrefix(e.Name(), "auto-step") || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		stepStr := strings.TrimSuffix(strings.TrimPrefix(e.Name(), "auto-step"), ".json")
		stepNum, err := strconv.Atoi(stepStr)
		if err != nil {
			continue
		}
		path := filepath.Join(dir, e.Name())
		raw, err := os.ReadFile(path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "read %s: %v\n", path, err)
			continue
		}
		var rep evalReport
		if err := json.Unmarshal(raw, &rep); err != nil {
			fmt.Fprintf(os.Stderr, "unmarshal %s: %v\n", path, err)
			continue
		}
		stat := stepStats{
			Step:       stepNum,
			ByCategory: map[string]*catStats{},
			Overall:    &catStats{},
		}
		for _, r := range rep.Results {
			if stat.ByCategory[r.Category] == nil {
				stat.ByCategory[r.Category] = &catStats{}
			}
			stat.ByCategory[r.Category].add(r.Generated)
			stat.Overall.add(r.Generated)
		}
		steps = append(steps, stat)
	}

	sort.Slice(steps, func(i, j int) bool { return steps[i].Step < steps[j].Step })

	// Colectează categoriile sortate alfabetic pentru consistență.
	catSet := map[string]bool{}
	for _, s := range steps {
		for c := range s.ByCategory {
			catSet[c] = true
		}
	}
	categories := []string{}
	for c := range catSet {
		categories = append(categories, c)
	}
	sort.Strings(categories)

	// Header: step + per-categorie (avg gen len ALL, empty rate) + overall
	fmt.Printf("%5s | %s | %14s\n", "STEP", header(categories), "OVERALL")
	fmt.Println(strings.Repeat("-", 120))
	for _, s := range steps {
		row := fmt.Sprintf("%5d |", s.Step)
		for _, c := range categories {
			cs := s.ByCategory[c]
			if cs == nil {
				row += fmt.Sprintf(" %5s/%3s |", "-", "-")
				continue
			}
			row += fmt.Sprintf(" %5.1f/%d.%d |",
				cs.avgAll(), cs.Empty, cs.N-cs.Empty)
		}
		o := s.Overall
		row += fmt.Sprintf(" %5.1f (%2d empty)", o.avgAll(), o.Empty)
		fmt.Println(row)
	}

	// Sumar trend: primii 5 vs ultimii 5
	if len(steps) >= 10 {
		fmt.Println()
		fmt.Println("=== TREND (first 5 vs last 5 steps) ===")
		fmt.Printf("%-14s %12s %12s %10s\n", "CATEGORY", "FIRST5_AVG", "LAST5_AVG", "DELTA%")
		categories = append(categories, "OVERALL")
		for _, c := range categories {
			f5, l5 := avgFirstLast(steps, c, 5)
			delta := 0.0
			if f5 > 0 {
				delta = (l5 - f5) / f5 * 100
			}
			fmt.Printf("%-14s %12.2f %12.2f %+9.1f%%\n", c, f5, l5, delta)
		}
	}
}

func header(cats []string) string {
	parts := []string{}
	for _, c := range cats {
		parts = append(parts, fmt.Sprintf("%5s/emp", trunc(c, 8)))
	}
	return strings.Join(parts, " | ")
}

func trunc(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}

func avgFirstLast(steps []stepStats, cat string, k int) (first, last float64) {
	if len(steps) < 2*k {
		return 0, 0
	}
	get := func(s stepStats) float64 {
		if cat == "OVERALL" {
			return s.Overall.avgAll()
		}
		cs := s.ByCategory[cat]
		if cs == nil {
			return 0
		}
		return cs.avgAll()
	}
	for i := 0; i < k; i++ {
		first += get(steps[i])
		last += get(steps[len(steps)-1-i])
	}
	return first / float64(k), last / float64(k)
}
