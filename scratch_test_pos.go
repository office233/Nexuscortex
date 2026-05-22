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
	cfg.DataDir = "data/cortex_test_pos"
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	org := cortex.NewOrganism(cfg, rng)

	sdr1 := org.Encoder.EncodeSentence("Care este capitala Romaniei ?")
	sdr2 := org.Encoder.EncodeSentence("Care este capitala Frantei ?")
	
	overlap := 0
	for _, idx := range sdr1.ActiveIndices() {
		if sdr2.IsActive(idx) {
			overlap++
		}
	}
	
	fmt.Printf("SDR1 Active: %d\n", sdr1.ActiveCount)
	fmt.Printf("SDR2 Active: %d\n", sdr2.ActiveCount)
	fmt.Printf("Overlap: %d\n", overlap)
	fmt.Printf("Overlap Percentage: %.2f%%\n", float64(overlap)/float64(sdr1.ActiveCount)*100)
}
