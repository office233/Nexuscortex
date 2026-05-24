// cortex-broca-train — train Broca 2.0 (MiniTransformer) on a corpus file.
//
// Loads an existing organism (must already have a BPE tokenizer), runs
// TrainTransformerFromCorpus for the requested number of epochs, then
// persists the trained transformer weights via Organism.Save.
//
// Example:
//
//	cortex-broca-train -data-dir ./data/cortex-auto \
//	    -corpus ./data/corpus/general.jsonl -lr 0.0003 -epochs 1 -max-lines 2000
package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"nexus-cortex/cortex"
)

func main() {
	dataDir := flag.String("data-dir", "./data/cortex-auto", "Organism data directory")
	corpus := flag.String("corpus", "./data/corpus/general.jsonl", "Corpus JSONL file")
	lr := flag.Float64("lr", 0.0003, "Learning rate")
	epochs := flag.Int("epochs", 1, "Number of passes over the corpus")
	maxLines := flag.Int("max-lines", 2000, "Max lines per epoch (0 = all)")
	seed := flag.Int64("seed", 42, "RNG seed")
	flag.Parse()

	if _, err := os.Stat(*corpus); err != nil {
		log.Fatalf("corpus not found: %s (%v)", *corpus, err)
	}

	cfg := cortex.DefaultConfig()
	cfg.DataDir = *dataDir
	cfg.Seed = *seed
	cfg.Demo = false
	cfg.NoSave = false

	tokPath := filepath.Join(*dataDir, "tokenizer.json")
	if _, err := os.Stat(tokPath); err != nil {
		log.Fatalf("tokenizer not found at %s — run cortex-tokenizer first or copy data/tokenizer.json", tokPath)
	}

	rng := rand.New(rand.NewSource(cfg.Seed))
	fmt.Printf("Loading organism from %s ...\n", *dataDir)
	org, err := cortex.LoadOrganism(cfg, rng)
	if err != nil || org == nil {
		log.Fatalf("LoadOrganism failed: %v", err)
	}

	if org.Tokenizer == nil {
		log.Fatal("organism has no BPE tokenizer — train one first via cortex-tokenizer")
	}

	// LoadOrganism intentionally leaves Transformer nil when no
	// transformer.nxtf exists on disk, so production loads do not
	// activate an untrained Broca 2.0. This CLI is the one place where
	// bootstrapping a fresh transformer IS the intent: create it here
	// so the first training run on a clean data dir works.
	if org.Transformer == nil {
		tfCfg := cortex.TransformerConfigFromConfig(org.Tokenizer.ActualVocabSize(), cfg)
		org.Transformer = cortex.NewMiniTransformer(tfCfg, rng)
		fmt.Printf("[Broca 2.0] Bootstrapped fresh transformer (%d params)\n",
			org.Transformer.ParamCount())
	}

	fmt.Printf("Transformer params: %d (~%.2fM)\n",
		org.Transformer.ParamCount(),
		float64(org.Transformer.ParamCount())/1e6)
	fmt.Printf("Tokenizer vocab: %d, merges: %d\n",
		org.Tokenizer.ActualVocabSize(), len(org.Tokenizer.Merges))

	totalSteps := 0
	totalLossSum := float32(0)
	start := time.Now()

	for epoch := 1; epoch <= *epochs; epoch++ {
		epochStart := time.Now()
		fmt.Printf("\n=== Epoch %d/%d: %s ===\n", epoch, *epochs, *corpus)
		avgLoss, steps, terr := org.TrainTransformerFromCorpus(*corpus, *maxLines, float32(*lr))
		if terr != nil {
			log.Fatalf("training failed: %v", terr)
		}
		totalSteps += steps
		totalLossSum += avgLoss * float32(steps)
		fmt.Printf("Epoch %d: %d steps, avg loss %.4f, took %s\n",
			epoch, steps, avgLoss, time.Since(epochStart).Truncate(time.Millisecond))
	}

	fmt.Printf("\nTotal: %d steps in %s\n", totalSteps, time.Since(start).Truncate(time.Millisecond))
	if totalSteps > 0 {
		fmt.Printf("Overall avg loss: %.4f\n", totalLossSum/float32(totalSteps))
	}

	fmt.Println("\nSaving organism (including trained transformer)...")
	if err := org.Save(*dataDir); err != nil {
		log.Fatalf("save failed: %v", err)
	}
	fmt.Println("Done.")
}
