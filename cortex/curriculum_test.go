package cortex

import (
	"strings"
	"testing"
)

func TestCurriculumParsing(t *testing.T) {
	jsonlData := `{"text": "câinele aleargă în parc."}
{"instruction": "ce face câinele?", "response": "câinele aleargă în parc."}
{"content": "pisica doarme pe canapea."}
{"prompt": "where do neurons fire?", "completion": "neurons fire together, wire together."}`

	r := strings.NewReader(jsonlData)
	items, err := ParseCorpusStream(r)
	if err != nil {
		t.Fatalf("failed to parse corpus stream: %v", err)
	}

	if len(items) != 4 {
		t.Errorf("expected 4 items, got %d", len(items))
	}

	// Verify formats are correctly mapped to our internal struct
	if items[0].Text != "câinele aleargă în parc." {
		t.Errorf("expected item 0 text %q, got %q", "câinele aleargă în parc.", items[0].Text)
	}
	if items[1].Instruction != "ce face câinele?" || items[1].Response != "câinele aleargă în parc." {
		t.Errorf("expected item 1 Q&A pair, got %+v", items[1])
	}
	if items[2].Text != "pisica doarme pe canapea." {
		t.Errorf("expected item 2 text %q, got %q", "pisica doarme pe canapea.", items[2].Text)
	}
	if items[3].Instruction != "where do neurons fire?" || items[3].Response != "neurons fire together, wire together." {
		t.Errorf("expected item 3 prompt/completion mapped to instruction/response, got %+v", items[3])
	}
}

func TestCurriculumSorting(t *testing.T) {
	items := []CorpusItem{
		{Text: "very long sentence containing many words that are complex and hard to learn"},
		{Text: "short"},
		{Instruction: "medium query?", Response: "medium response answer"},
	}

	c := NewCurriculum(items)
	c.SortByComplexity()

	if len(c.Items) != 3 {
		t.Fatalf("expected 3 items, got %d", len(c.Items))
	}

	// The sorted order by token count should be: "short" (1 token), Q&A medium (5 tokens), very long sentence (11 tokens)
	firstLen := len(Tokenize(c.Items[0].GetFullText()))
	secondLen := len(Tokenize(c.Items[1].GetFullText()))
	thirdLen := len(Tokenize(c.Items[2].GetFullText()))

	if firstLen > secondLen || secondLen > thirdLen {
		t.Errorf("curriculum items not sorted correctly by token length: %d -> %d -> %d", firstLen, secondLen, thirdLen)
	}
}

func TestCurriculumRevisitScheduler(t *testing.T) {
	items := []CorpusItem{
		{Text: "simple memory block"},
		{Text: "another complex structure"},
	}

	c := NewCurriculum(items)
	
	// Create a revisit list based on simulated errors (e.g. error > 100)
	errors := map[int]uint8{
		0: 20,  // low error, no need to revisit
		1: 150, // high surprise/error, must revisit
	}

	revisitBatch := c.GenerateRevisitBatch(errors, 100)
	if len(revisitBatch) != 1 {
		t.Fatalf("expected 1 item for revisit, got %d", len(revisitBatch))
	}

	if revisitBatch[0].Text != "another complex structure" {
		t.Errorf("expected revisit item to be %q, got %q", "another complex structure", revisitBatch[0].Text)
	}
}
