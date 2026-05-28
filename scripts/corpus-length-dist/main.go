// corpus-length-dist — analizează distribuția lungimii secvențelor
// tokenizate din corpus-urile training, ca să verificăm ipoteza că
// modelul învață "EOS după prompt scurt" pentru că majoritatea
// secvențelor sunt scurte în training data.
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
)

func extractText(line string) string {
	// Try JSONL with "instruction"+"response", "text", "content", "prompt"+"completion".
	var m map[string]interface{}
	if err := json.Unmarshal([]byte(line), &m); err != nil {
		return strings.TrimSpace(line)
	}
	parts := []string{}
	for _, k := range []string{"instruction", "prompt", "input", "question"} {
		if v, ok := m[k].(string); ok && strings.TrimSpace(v) != "" {
			parts = append(parts, v)
		}
	}
	for _, k := range []string{"response", "completion", "output", "answer", "text", "content"} {
		if v, ok := m[k].(string); ok && strings.TrimSpace(v) != "" {
			parts = append(parts, v)
		}
	}
	return strings.Join(parts, " ")
}

func main() {
	files := []string{
		"data/corpus/dolly.jsonl",
		"data/corpus/alpaca.jsonl",
	}
	// Heuristic: token count ~ word count * 1.3 for BPE on English.
	// We'll measure in words (simpler and good enough).
	allLens := []int{}
	perFile := map[string][]int{}

	for _, fpath := range files {
		f, err := os.Open(fpath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "skip %s: %v\n", fpath, err)
			continue
		}
		scanner := bufio.NewScanner(f)
		scanner.Buffer(make([]byte, 4*1024*1024), 4*1024*1024)
		lens := []int{}
		for scanner.Scan() {
			text := extractText(scanner.Text())
			if len(text) < 20 {
				continue
			}
			wc := len(strings.Fields(text))
			lens = append(lens, wc)
			allLens = append(allLens, wc)
		}
		f.Close()
		perFile[fpath] = lens
		fmt.Printf("%s: %d sequences\n", fpath, len(lens))
	}
	fmt.Printf("\nTotal: %d sequences\n", len(allLens))

	report := func(name string, ls []int) {
		if len(ls) == 0 {
			return
		}
		sort.Ints(ls)
		fmt.Printf("\n=== %s ===\n", name)
		// percentile helper
		p := func(q float64) int {
			idx := int(float64(len(ls)) * q)
			if idx >= len(ls) {
				idx = len(ls) - 1
			}
			return ls[idx]
		}
		fmt.Printf("  count=%d\n", len(ls))
		fmt.Printf("  min=%d, max=%d\n", ls[0], ls[len(ls)-1])
		fmt.Printf("  p10=%d, p25=%d, p50=%d, p75=%d, p90=%d, p95=%d, p99=%d\n",
			p(0.10), p(0.25), p(0.50), p(0.75), p(0.90), p(0.95), p(0.99))
		// bucket histogram
		buckets := map[string]int{
			"<10": 0, "10-25": 0, "25-50": 0, "50-100": 0,
			"100-200": 0, "200-500": 0, ">=500": 0,
		}
		for _, l := range ls {
			switch {
			case l < 10:
				buckets["<10"]++
			case l < 25:
				buckets["10-25"]++
			case l < 50:
				buckets["25-50"]++
			case l < 100:
				buckets["50-100"]++
			case l < 200:
				buckets["100-200"]++
			case l < 500:
				buckets["200-500"]++
			default:
				buckets[">=500"]++
			}
		}
		order := []string{"<10", "10-25", "25-50", "50-100", "100-200", "200-500", ">=500"}
		fmt.Println("  bucket   count    pct")
		for _, b := range order {
			c := buckets[b]
			pct := float64(c) * 100 / float64(len(ls))
			fmt.Printf("  %-8s %5d  %5.1f%%\n", b, c, pct)
		}
	}

	for f, ls := range perFile {
		report(f, ls)
	}
	report("ALL", allLens)
}
