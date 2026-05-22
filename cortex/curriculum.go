package cortex

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"
)

// CorpusItem represents a single training unit within the curriculum.
type CorpusItem struct {
	Text        string `json:"text,omitempty"`
	Instruction string `json:"instruction,omitempty"`
	Response    string `json:"response,omitempty"`
}

// GetFullText returns the complete text of the item for complexity evaluation.
func (item CorpusItem) GetFullText() string {
	if item.Text != "" {
		return item.Text
	}
	return item.Instruction + " " + item.Response
}

// Curriculum manages a sequence of CorpusItems and handles curriculum ordering.
type Curriculum struct {
	Items []CorpusItem
}

// NewCurriculum creates a new Curriculum manager.
func NewCurriculum(items []CorpusItem) *Curriculum {
	return &Curriculum{Items: items}
}

// SortByComplexity orders items from shortest to longest (simple to complex) using tokens.
func (c *Curriculum) SortByComplexity() {
	sort.Slice(c.Items, func(i, j int) bool {
		lenI := len(Tokenize(c.Items[i].GetFullText()))
		lenJ := len(Tokenize(c.Items[j].GetFullText()))
		return lenI < lenJ
	})
}

// GenerateRevisitBatch selects items that produced a prediction error above the threshold.
func (c *Curriculum) GenerateRevisitBatch(errors map[int]uint8, errorThreshold uint8) []CorpusItem {
	var batch []CorpusItem
	for idx, errVal := range errors {
		if errVal >= errorThreshold && idx >= 0 && idx < len(c.Items) {
			batch = append(batch, c.Items[idx])
		}
	}
	return batch
}

// rawJSONItem is used for flexible parsing of multiple JSON formats.
type rawJSONItem struct {
	Text        string `json:"text"`
	Content     string `json:"content"`
	Instruction string `json:"instruction"`
	Prompt      string `json:"prompt"`
	Response    string `json:"response"`
	Completion  string `json:"completion"`
}

// ParseCorpusStream reads a JSONL corpus stream and extracts items.
func ParseCorpusStream(r io.Reader) ([]CorpusItem, error) {
	var items []CorpusItem
	scanner := bufio.NewScanner(r)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") || strings.HasPrefix(line, "//") {
			continue
		}

		var raw rawJSONItem
		if err := json.Unmarshal([]byte(line), &raw); err != nil {
			return nil, fmt.Errorf("unmarshal corpus line %q: %w", line, err)
		}

		item := CorpusItem{}
		// Map standard fields to our unified structure
		if raw.Text != "" {
			item.Text = raw.Text
		} else if raw.Content != "" {
			item.Text = raw.Content
		}

		if raw.Instruction != "" {
			item.Instruction = raw.Instruction
		} else if raw.Prompt != "" {
			item.Instruction = raw.Prompt
		}

		if raw.Response != "" {
			item.Response = raw.Response
		} else if raw.Completion != "" {
			item.Response = raw.Completion
		}

		// Skip completely empty items
		if item.Text == "" && item.Instruction == "" && item.Response == "" {
			continue
		}

		items = append(items, item)
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan corpus: %w", err)
	}

	return items, nil
}
