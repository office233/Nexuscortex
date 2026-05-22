package main

import (
	"encoding/json"
	"fmt"
	"os"
)

type demoQuery struct {
	Label string `json:"label"`
	Input string `json:"input"`
}

type demoSpec struct {
	LearningSentences []string    `json:"learning_sentences"`
	Interactions      []demoQuery `json:"interactions"`
	PostSleepQueries  []demoQuery `json:"post_sleep_queries"`
}

func loadDemoSpec(path string) (demoSpec, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return demoSpec{}, fmt.Errorf("read demo spec: %w", err)
	}

	var spec demoSpec
	if err := json.Unmarshal(raw, &spec); err != nil {
		return demoSpec{}, fmt.Errorf("parse demo spec: %w", err)
	}

	if len(spec.LearningSentences) == 0 && len(spec.Interactions) == 0 && len(spec.PostSleepQueries) == 0 {
		return demoSpec{}, fmt.Errorf("demo spec is empty")
	}

	return spec, nil
}
