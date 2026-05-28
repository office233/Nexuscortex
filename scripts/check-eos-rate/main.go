// check-eos-rate — pentru fiecare eval result, decide dacă generația
// s-a oprit din cauza <EOS> (length < max_tokens) sau din cauza
// max_tokens limit (length == max_tokens). Token-level analysis e
// dificilă fără re-rulare; aproximăm cu lungimea caracter-string.
//
// Heuristic: dacă generated length în caractere < 30, foarte probabil
// EOS sample timpuriu (max_tokens=40 cu BPE produce de obicei 100+ chars).
// Dacă generated length > 100, foarte probabil hit max_tokens.
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
	Generated string `json:"generated"`
}
type evalReport struct {
	Results []evalResult `json:"results"`
	MaxTokens int `json:"max_tokens"`
}

func main() {
	dir := "data/cortex-auto/evals"
	entries, _ := os.ReadDir(dir)
	type stepData struct {
		step int
		eosEarly int // gen < 10 chars  (very likely EOS sampled quickly)
		shortEnd int // 10-50 chars     (likely EOS)
		midEnd   int // 50-150 chars
		longEnd  int // > 150 chars (likely maxTokens)
		empty    int
		total    int
	}
	steps := []stepData{}
	for _, e := range entries {
		if !strings.HasPrefix(e.Name(), "auto-step") || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		stepStr := strings.TrimSuffix(strings.TrimPrefix(e.Name(), "auto-step"), ".json")
		stepNum, err := strconv.Atoi(stepStr)
		if err != nil {
			continue
		}
		raw, _ := os.ReadFile(filepath.Join(dir, e.Name()))
		var rep evalReport
		json.Unmarshal(raw, &rep)
		sd := stepData{step: stepNum}
		for _, r := range rep.Results {
			sd.total++
			l := len(r.Generated)
			switch {
			case l == 0:
				sd.empty++
			case l < 10:
				sd.eosEarly++
			case l < 50:
				sd.shortEnd++
			case l < 150:
				sd.midEnd++
			default:
				sd.longEnd++
			}
		}
		steps = append(steps, sd)
	}
	sort.Slice(steps, func(i, j int) bool { return steps[i].step < steps[j].step })

	fmt.Printf("%5s | %5s %5s %5s %5s %5s\n", "STEP", "EMPTY", "<10", "<50", "<150", ">=150")
	fmt.Println(strings.Repeat("-", 50))
	for _, s := range steps {
		fmt.Printf("%5d | %5d %5d %5d %5d %5d\n",
			s.step, s.empty, s.eosEarly, s.shortEnd, s.midEnd, s.longEnd)
	}

	// Trend summary
	fmt.Println("\n=== TREND: first 5 vs last 5 ===")
	if len(steps) >= 10 {
		f5, l5 := stepData{}, stepData{}
		for i := 0; i < 5; i++ {
			f5.empty += steps[i].empty
			f5.eosEarly += steps[i].eosEarly
			f5.shortEnd += steps[i].shortEnd
			f5.midEnd += steps[i].midEnd
			f5.longEnd += steps[i].longEnd
			f5.total += steps[i].total

			j := len(steps) - 1 - i
			l5.empty += steps[j].empty
			l5.eosEarly += steps[j].eosEarly
			l5.shortEnd += steps[j].shortEnd
			l5.midEnd += steps[j].midEnd
			l5.longEnd += steps[j].longEnd
			l5.total += steps[j].total
		}
		fmt.Printf("Bucket        FIRST5      LAST5    Delta%%\n")
		bucketRow("empty", f5.empty, l5.empty, f5.total, l5.total)
		bucketRow("<10 chars", f5.eosEarly, l5.eosEarly, f5.total, l5.total)
		bucketRow("<50 chars", f5.shortEnd, l5.shortEnd, f5.total, l5.total)
		bucketRow("<150 chars", f5.midEnd, l5.midEnd, f5.total, l5.total)
		bucketRow(">=150 chars", f5.longEnd, l5.longEnd, f5.total, l5.total)
	}
}

func bucketRow(name string, fc, lc, ft, lt int) {
	fp := pct(fc, ft)
	lp := pct(lc, lt)
	delta := lp - fp
	fmt.Printf("%-13s %4d=%4.1f%% %4d=%4.1f%%  %+5.1fpp\n", name, fc, fp, lc, lp, delta)
}

func pct(num, denom int) float64 {
	if denom == 0 {
		return 0
	}
	return float64(num) * 100 / float64(denom)
}
