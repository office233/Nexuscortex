package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadDemoSpecReadsExternalData(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "demo.json")
	raw := []byte(`{
  "learning_sentences": ["alpha learns beta"],
  "interactions": [{"label": "recall", "input": "what learns beta?"}],
  "post_sleep_queries": [{"label": "after sleep", "input": "beta"}]
}`)
	if err := os.WriteFile(path, raw, 0644); err != nil {
		t.Fatalf("failed to write demo fixture: %v", err)
	}

	spec, err := loadDemoSpec(path)
	if err != nil {
		t.Fatalf("loadDemoSpec returned error: %v", err)
	}

	if len(spec.LearningSentences) != 1 || spec.LearningSentences[0] != "alpha learns beta" {
		t.Fatalf("unexpected learning sentences: %#v", spec.LearningSentences)
	}
	if len(spec.Interactions) != 1 || spec.Interactions[0].Label != "recall" || spec.Interactions[0].Input != "what learns beta?" {
		t.Fatalf("unexpected interactions: %#v", spec.Interactions)
	}
	if len(spec.PostSleepQueries) != 1 || spec.PostSleepQueries[0].Label != "after sleep" || spec.PostSleepQueries[0].Input != "beta" {
		t.Fatalf("unexpected post-sleep queries: %#v", spec.PostSleepQueries)
	}
}

func TestLoadDemoSpecRejectsEmptyDemo(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "empty.json")
	if err := os.WriteFile(path, []byte(`{}`), 0644); err != nil {
		t.Fatalf("failed to write demo fixture: %v", err)
	}

	if _, err := loadDemoSpec(path); err == nil {
		t.Fatal("expected empty demo spec to be rejected")
	}
}
