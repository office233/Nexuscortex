package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strings"

	"nexus-cortex/cortex"
)

func main() {
	dataDir := flag.String("data-dir", "./data/cortex", "Path to organism data directory")
	seed := flag.Int64("seed", 42, "Random seed for deterministic runs")
	fresh := flag.Bool("fresh", false, "Start with a new organism (ignore saved state)")
	evalPath := flag.String("eval", "", "Optional path to a single JSONL eval suite")
	flag.Parse()

	cfg := cortex.DefaultConfig()
	cfg.DataDir = *dataDir
	cfg.Fresh = *fresh
	cfg.Seed = *seed
	cfg.Demo = false
	cfg.NoSave = true

	loadOrganism(cfg, rand.New(rand.NewSource(cfg.Seed)), true)
	newOrganism := func() *cortex.Organism {
		return loadOrganism(cfg, rand.New(rand.NewSource(cfg.Seed)), false)
	}

	var suites []cortex.SuiteResult
	if *evalPath != "" {
		cases := mustLoadSuite(*evalPath)
		fmt.Printf("Loaded %d test cases from %s.\n\n", len(cases), *evalPath)
		suites = append(suites, cortex.RunSuiteIsolated(filepath.Base(*evalPath), cases, newOrganism))
	} else {
		evalsDir := filepath.Join(".", "data", "evals")
		recallPath := filepath.Join(evalsDir, "basic_recall.jsonl")
		noEchoPath := filepath.Join(evalsDir, "no_echo.jsonl")
		generalizationPath := filepath.Join(evalsDir, "generalization.jsonl")

		recallCases := mustLoadSuite(recallPath)
		noEchoCases := mustLoadSuite(noEchoPath)
		generalizationCases := mustLoadSuite(generalizationPath)
		fmt.Printf(
			"Loaded %d seen-recall cases, %d anti-echo cases, and %d generalization cases.\n\n",
			len(recallCases), len(noEchoCases), len(generalizationCases),
		)

		suites = append(suites,
			cortex.RunSuiteIsolated("Seen Training Recall", recallCases, newOrganism),
			cortex.RunSuiteIsolated("Anti-Echo Boundaries", noEchoCases, newOrganism),
			cortex.RunSuiteIsolated("Held-out Generalization", generalizationCases, newOrganism),
		)
	}

	printReport(suites)
}

func loadOrganism(cfg cortex.Config, rng *rand.Rand, verbose bool) *cortex.Organism {
	var org *cortex.Organism
	var err error
	if !cfg.Fresh {
		org, err = cortex.LoadOrganism(cfg, rng)
	}
	if org != nil {
		if verbose {
			fmt.Println("Loaded persisted organism state.")
		}
		return org
	}
	if verbose {
		if err != nil {
			fmt.Printf("No saved organism found or load failed (%v); using fresh state.\n", err)
		} else {
			fmt.Println("Using fresh organism state.")
		}
	}
	return cortex.NewOrganism(cfg, rng)
}

func mustLoadSuite(path string) []cortex.TestCase {
	cases, err := cortex.LoadEvalSuite(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to load eval suite %s: %v\n", path, err)
		os.Exit(1)
	}
	return cases
}

func printReport(suites []cortex.SuiteResult) {
	fmt.Println("FAILED TEST CASES")
	fmt.Println(strings.Repeat("-", 80))
	hasFailures := false
	for _, suite := range suites {
		for _, tr := range suite.Results {
			if tr.Passed {
				continue
			}
			hasFailures = true
			fmt.Printf("Suite:    %s\n", suite.Name)
			fmt.Printf("Input:    %q\n", tr.Case.Input)
			fmt.Printf("Got:      %q (conf: %d)\n", tr.Response, tr.Confidence)
			if len(tr.Case.ExpectedContains) > 0 {
				fmt.Printf("Expected: %v\n", tr.Case.ExpectedContains)
			}
			fmt.Printf("Reason:   %s\n", tr.Reason)
			fmt.Println(strings.Repeat("-", 80))
		}
	}
	if !hasFailures {
		fmt.Println("None")
		fmt.Println(strings.Repeat("-", 80))
	}

	fmt.Println("\nCAPABILITY SCOREBOARD")
	fmt.Printf("%-28s %6s %7s %9s %8s\n", "Suite", "Total", "Passed", "Recall", "Echo")
	for _, suite := range suites {
		fmt.Printf("%-28s %6d %7d %8.1f%% %7.1f%%\n",
			suite.Name, suite.Total, suite.Passed, suite.RecallRate, suite.EchoRate)
	}

	fmt.Println("\nCONFIDENCE CALIBRATION")
	for _, suite := range suites {
		fmt.Printf("%-28s avg=%5.1f passed=%5.1f failed=%5.1f\n",
			suite.Name, suite.AvgConfidence, suite.AvgPassedConf, suite.AvgFailedConf)
	}

	fmt.Printf("\nNEXUS CAPABILITY SCORE: %.1f / 100.0\n", capabilityScore(suites))
}

func capabilityScore(suites []cortex.SuiteResult) float64 {
	var total, passed, echoed int
	for _, suite := range suites {
		total += suite.Total
		passed += suite.Passed
		echoed += suite.Echoed
	}
	if total == 0 {
		return 0
	}
	passRate := float64(passed) / float64(total) * 100.0
	antiEchoRate := 100.0 - (float64(echoed) / float64(total) * 100.0)
	return (passRate * 0.8) + (antiEchoRate * 0.2)
}
