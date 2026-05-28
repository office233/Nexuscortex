// compare-dual — alătură rezultatele a două eval-uri (typically baseline
// vs min-tokens=N) și afișează, per task, ce a generat fiecare. Scop:
// vizualizează direct dacă suppression-ul EOS scoate la suprafață
// "modelul știe răspunsul dar îl ascunde" vs. "modelul pur și simplu nu
// știe nimic".
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

type result struct {
	TaskID    string  `json:"task_id"`
	Category  string  `json:"category"`
	Prompt    string  `json:"prompt"`
	Generated string  `json:"generated"`
	Expected  string  `json:"expected"`
	Correct   bool    `json:"correct"`
	GenMs     int64   `json:"gen_ms"`
}

type report struct {
	MinTokens int      `json:"min_tokens"`
	Results   []result `json:"results"`
}

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "usage: compare-dual <baseline.json> <withmin.json>")
		os.Exit(2)
	}
	rA := mustLoad(os.Args[1])
	rB := mustLoad(os.Args[2])

	idxA := map[string]result{}
	for _, r := range rA.Results {
		idxA[r.TaskID] = r
	}

	fmt.Printf("BASELINE  min_tokens=%d\n", rA.MinTokens)
	fmt.Printf("VARIANT   min_tokens=%d\n\n", rB.MinTokens)

	fmt.Printf("%-30s | %s\n", "TASK", "A_gen → B_gen (expected)")
	fmt.Println(strings.Repeat("-", 130))

	for _, b := range rB.Results {
		a := idxA[b.TaskID]
		aGen := truncQuoted(a.Generated, 30)
		bGen := truncQuoted(b.Generated, 60)
		exp := truncQuoted(b.Expected, 30)
		marker := "  "
		if a.Correct {
			marker = "A✓"
		}
		if b.Correct {
			marker = marker + "B✓"
		}
		fmt.Printf("%2s %-27s | %s → %s  (want: %s)\n",
			marker, b.TaskID, aGen, bGen, exp)
	}

	// Stats on length
	totalA, totalB := 0, 0
	emptyA, emptyB := 0, 0
	for _, b := range rB.Results {
		a := idxA[b.TaskID]
		totalA += len(a.Generated)
		totalB += len(b.Generated)
		if a.Generated == "" {
			emptyA++
		}
		if b.Generated == "" {
			emptyB++
		}
	}
	n := len(rB.Results)
	fmt.Printf("\n=== AGGREGATE ===\n")
	fmt.Printf("Tasks: %d\n", n)
	fmt.Printf("Empty gens: A=%d/%d (%.0f%%), B=%d/%d (%.0f%%)\n",
		emptyA, n, float64(emptyA)*100/float64(n),
		emptyB, n, float64(emptyB)*100/float64(n))
	fmt.Printf("Avg gen length (chars): A=%.1f, B=%.1f (delta %+.1f)\n",
		float64(totalA)/float64(n), float64(totalB)/float64(n),
		float64(totalB-totalA)/float64(n))
}

func mustLoad(p string) report {
	raw, err := os.ReadFile(p)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load %s: %v\n", p, err)
		os.Exit(2)
	}
	var r report
	if err := json.Unmarshal(raw, &r); err != nil {
		fmt.Fprintf(os.Stderr, "parse %s: %v\n", p, err)
		os.Exit(2)
	}
	return r
}

func truncQuoted(s string, n int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.ReplaceAll(s, "\r", " ")
	if len(s) > n {
		s = s[:n-1] + "…"
	}
	return fmt.Sprintf("%q", s)
}
