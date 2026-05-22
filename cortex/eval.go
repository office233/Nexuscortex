package cortex

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// TestCase represents a single evaluation scenario.
type TestCase struct {
	Input             string   `json:"input"`
	MustNotEqualInput bool     `json:"must_not_equal_input"`
	ExpectedContains  []string `json:"expected_contains"`
}

// TestCaseResult holds the outcome of running a single TestCase.
type TestCaseResult struct {
	Case       TestCase `json:"case"`
	Response   string   `json:"response"`
	Passed     bool     `json:"passed"`
	Echoed     bool     `json:"echoed"`
	Confidence uint8    `json:"confidence"`
	Reason     string   `json:"reason,omitempty"`
}

// SuiteResult aggregates metrics across an entire evaluation suite.
type SuiteResult struct {
	Name          string           `json:"name"`
	Total         int              `json:"total"`
	Passed        int              `json:"passed"`
	Echoed        int              `json:"echoed"`
	AvgConfidence float64          `json:"avg_confidence"`
	AvgPassedConf float64          `json:"avg_passed_conf"`
	AvgFailedConf float64          `json:"avg_failed_conf"`
	RecallRate    float64          `json:"recall_rate"`
	EchoRate      float64          `json:"echo_rate"`
	Results       []TestCaseResult `json:"results"`
}

// LoadEvalSuite parses a JSONL file containing evaluation test cases.
func LoadEvalSuite(path string) ([]TestCase, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open eval suite: %w", err)
	}
	defer file.Close()

	var cases []TestCase
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") || strings.HasPrefix(line, "//") {
			continue
		}
		var tc TestCase
		if err := json.Unmarshal([]byte(line), &tc); err != nil {
			return nil, fmt.Errorf("unmarshal test case line %q: %w", line, err)
		}
		cases = append(cases, tc)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan eval suite: %w", err)
	}
	return cases, nil
}

// RunSuite runs a set of test cases against the digital organism and aggregates performance metrics.
// The supplied organism is reused across cases, so callers that need strict
// non-mutating evaluation should use RunSuiteIsolated.
func RunSuite(org *Organism, name string, cases []TestCase) SuiteResult {
	return runSuite(name, cases, func() *Organism { return org })
}

// RunSuiteIsolated runs each test case against a freshly loaded organism.
// This prevents Process() learning from one evaluation case and leaking that
// state into later cases.
func RunSuiteIsolated(name string, cases []TestCase, newOrganism func() *Organism) SuiteResult {
	return runSuite(name, cases, newOrganism)
}

func runSuite(name string, cases []TestCase, organismForCase func() *Organism) SuiteResult {
	res := SuiteResult{
		Name:    name,
		Total:   len(cases),
		Results: make([]TestCaseResult, 0, len(cases)),
	}

	if len(cases) == 0 {
		return res
	}

	var sumConf, sumPassedConf, sumFailedConf uint64
	var countPassed, countFailed int
	var countRecallable int // cases that specify non-empty expected_contains
	var countRecalled int   // cases where expected_contains were matched

	for _, tc := range cases {
		org := organismForCase()
		response := org.Process(tc.Input)
		conf := org.Prefrontal.GetConfidence()
		if response == "(no confident response)" {
			conf = 0
		}

		passed := true
		echoed := false
		var failureReasons []string

		trimmedInput := strings.ToLower(strings.TrimSpace(tc.Input))
		trimmedResponse := strings.ToLower(strings.TrimSpace(response))

		responseClean := strings.Trim(trimmedResponse, " \t\n\r.,;:!?\"'()[]{}")
		if responseClean == "" || responseClean == "?" || responseClean == "." {
			passed = false
			failureReasons = append(failureReasons, "degenerate response")
		}

		if trimmedInput == trimmedResponse {
			echoed = true
			res.Echoed++
			if tc.MustNotEqualInput {
				passed = false
				failureReasons = append(failureReasons, "echoed input")
			}
		}

		if len(tc.ExpectedContains) > 0 {
			countRecallable++
			hasAll := true
			for _, exp := range tc.ExpectedContains {
				expClean := strings.ToLower(strings.TrimSpace(exp))
				if !strings.Contains(trimmedResponse, expClean) {
					hasAll = false
					passed = false
					failureReasons = append(failureReasons, fmt.Sprintf("missing expected token %q", exp))
				}
			}
			if hasAll {
				countRecalled++
			}
		}

		if passed {
			res.Passed++
			sumPassedConf += uint64(conf)
			countPassed++
		} else {
			sumFailedConf += uint64(conf)
			countFailed++
		}
		sumConf += uint64(conf)

		res.Results = append(res.Results, TestCaseResult{
			Case:       tc,
			Response:   response,
			Passed:     passed,
			Echoed:     echoed,
			Confidence: conf,
			Reason:     strings.Join(failureReasons, "; "),
		})
	}

	res.AvgConfidence = float64(sumConf) / float64(res.Total)
	if countPassed > 0 {
		res.AvgPassedConf = float64(sumPassedConf) / float64(countPassed)
	}
	if countFailed > 0 {
		res.AvgFailedConf = float64(sumFailedConf) / float64(countFailed)
	}

	if countRecallable > 0 {
		res.RecallRate = float64(countRecalled) / float64(countRecallable) * 100.0
	} else {
		res.RecallRate = 100.0 // vacuously correct if nothing to recall
	}
	res.EchoRate = float64(res.Echoed) / float64(res.Total) * 100.0

	return res
}
