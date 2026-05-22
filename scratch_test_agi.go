//go:build ignore

package main

import (
	"fmt"
	"math/rand"
	"nexus-cortex/cortex"
	"time"
)

func main() {
	cfg := cortex.DefaultConfig()
	cfg.DataDir = "data/cortex"
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	org := cortex.NewOrganism(cfg, rng)

	// Learn
	org.LearnQA("Care este capitala romaniei?", "capitala României este municipiul București.")
	
	// Query
	ans := org.Process("Care este capitala Frantei?")
	
	fmt.Printf("\nGenerated Answer: %s\n", ans)
}
