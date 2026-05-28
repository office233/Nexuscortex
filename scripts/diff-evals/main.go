// diff-evals — compară două rapoarte broca-eval per task, evidențiind
// task-uri unde verdictul s-a schimbat (correct↔incorrect). Folosit ca
// "smoke test" după fix-uri la scorer: dacă fix-ul scoate la suprafață
// false-positive vechi, vom vedea taskuri care erau "correct" și acum
// devin "incorrect".
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

type result struct {
	TaskID    string `json:"task_id"`
	Generated string `json:"generated"`
	Expected  string `json:"expected"`
	Correct   bool   `json:"correct"`
	Reason    string `json:"reason"`
}

type report struct {
	Results []result `json:"results"`
}

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "usage: diff-evals <before.json> <after.json>")
		os.Exit(2)
	}
	a := load(os.Args[1])
	b := load(os.Args[2])

	idxA := map[string]result{}
	for _, r := range a.Results {
		idxA[r.TaskID] = r
	}

	flipped := 0
	for _, br := range b.Results {
		ar, ok := idxA[br.TaskID]
		if !ok || ar.Correct == br.Correct {
			continue
		}
		flipped++
		dir := "✓→✗"
		if br.Correct {
			dir = "✗→✓"
		}
		fmt.Printf("%s  %-30s  gen=%q\n", dir, br.TaskID, trunc(br.Generated, 60))
		fmt.Printf("        reason A: %s\n", ar.Reason)
		fmt.Printf("        reason B: %s\n", br.Reason)
	}
	if flipped == 0 {
		fmt.Println("No verdict changes — both runs agree on every task.")
	} else {
		fmt.Printf("\n%d task(s) flipped verdict.\n", flipped)
	}
}

func load(p string) report {
	raw, _ := os.ReadFile(p)
	var r report
	json.Unmarshal(raw, &r)
	return r
}

func trunc(s string, n int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) > n {
		s = s[:n-1] + "…"
	}
	return s
}
