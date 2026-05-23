package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math/rand"
	"nexus-cortex/cortex"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// QAPair represents a question-answer training pair loaded from JSON.
type QAPair struct {
	Q string `json:"q"`
	A string `json:"a"`
}

// loadQA reads Q&A pairs from a JSON file.
func loadQA(path string) ([]QAPair, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	var pairs []QAPair
	if err := json.Unmarshal(data, &pairs); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	return pairs, nil
}

// loadTexts reads plain text lines from a file (one text per line).
func loadTexts(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	var texts []string
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024) // 1MB lines
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" && !strings.HasPrefix(line, "#") {
			texts = append(texts, line)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan %s: %w", path, err)
	}
	return texts, nil
}

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║  🧠 NEXUS CORTEX — Training Session                        ║")
	fmt.Println("║  External Data · CUDA GPU · Configurable Neurons            ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Println()

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	cfg := cortex.DefaultConfig()
	cfg.DataDir = "./data/cortex-training"

	// ══════════════════════════════════════════════════════════
	// LOAD EXTERNAL TRAINING DATA
	// ══════════════════════════════════════════════════════════
	dataDir := cfg.TrainingDataDir

	qaPath := filepath.Join(dataDir, "qa.json")
	textsPath := filepath.Join(dataDir, "texts.txt")

	fmt.Printf("  📂 Loading data from: %s\n", dataDir)

	qaCorpus, err := loadQA(qaPath)
	if err != nil {
		fmt.Printf("  ⚠️  No Q&A data: %v\n", err)
		qaCorpus = nil
	} else {
		fmt.Printf("  ✅ Loaded %d Q&A pairs from %s\n", len(qaCorpus), filepath.Base(qaPath))
	}

	textCorpus, err := loadTexts(textsPath)
	if err != nil {
		fmt.Printf("  ⚠️  No text data: %v\n", err)
		textCorpus = nil
	} else {
		fmt.Printf("  ✅ Loaded %d texts from %s\n", len(textCorpus), filepath.Base(textsPath))
	}

	if len(qaCorpus) == 0 && len(textCorpus) == 0 {
		fmt.Println("  ❌ No training data found! Create qa.json and/or texts.txt in", dataDir)
		os.Exit(1)
	}
	fmt.Println()

	// ══════════════════════════════════════════════════════════
	// CREATE ORGANISM
	// ══════════════════════════════════════════════════════════
	fmt.Println("  🔬 Creating organism...")
	startTime := time.Now()
	org := cortex.NewOrganism(cfg, rng)
	fmt.Printf("  ✅ Organism alive. RadioCortex: %d neurons (%.1f MB)\n",
		org.RadioCortex.Size, float64(org.RadioCortex.Size*4)/(1024*1024))

	// Try GPU
	if org.RadioCortex.InitGPU() {
		fmt.Println("  🚀 GPU CUDA acceleration: ENABLED")
	} else {
		fmt.Println("  💻 Using CPU (no CUDA DLL found)")
	}
	fmt.Printf("  ⏱  Init time: %v\n\n", time.Since(startTime))

	// ══════════════════════════════════════════════════════════
	// PHASE 1: Text absorption
	// ══════════════════════════════════════════════════════════
	if len(textCorpus) > 0 {
		fmt.Println("┌──────────────────────────────────────────────────────────────┐")
		fmt.Printf("│  📚 PHASE 1: TEXT ABSORPTION (%d texts)                      │\n", len(textCorpus))
		fmt.Println("└──────────────────────────────────────────────────────────────┘")

		phase1Start := time.Now()
		for i, text := range textCorpus {
			org.Learn(text)
			if (i+1)%20 == 0 || i+1 == len(textCorpus) {
				fmt.Printf("  📝 Absorbed %d / %d texts\n", i+1, len(textCorpus))
			}
		}
		fmt.Printf("  ✅ Absorbed %d texts in %v\n\n", len(textCorpus), time.Since(phase1Start))
	}

	// ══════════════════════════════════════════════════════════
	// PHASE 2: Q&A Training
	// ══════════════════════════════════════════════════════════
	if len(qaCorpus) > 0 {
		epochs := 3
		fmt.Println("┌──────────────────────────────────────────────────────────────┐")
		fmt.Printf("│  🎓 PHASE 2: Q&A TRAINING (%d epochs × %d pairs)             │\n", epochs, len(qaCorpus))
		fmt.Println("└──────────────────────────────────────────────────────────────┘")

		phase2Start := time.Now()
		for epoch := 1; epoch <= epochs; epoch++ {
			epochStart := time.Now()
			fmt.Printf("\n  📖 Epoch %d/%d:\n", epoch, epochs)

			perm := rng.Perm(len(qaCorpus))
			for i, idx := range perm {
				qa := qaCorpus[idx]
				org.LearnQA(qa.Q, qa.A)
				if (i+1)%30 == 0 {
					stats := org.RadioCortex.Stats()
					fmt.Printf("     [%3d/%d] Radio ticks: %d | alive: %d | avg amp: %d\n",
						i+1, len(qaCorpus), stats.TickCount, stats.AliveNeurons, stats.AvgAmplitude)
				}
			}

			fmt.Printf("  💤 Sleep after epoch %d (took %v)...\n", epoch, time.Since(epochStart))
			logs := org.Sleep()
			for _, log := range logs {
				if strings.Contains(log, "Radio") || strings.Contains(log, "Neurogenesis") {
					fmt.Printf("     %s\n", log)
				}
			}
		}

		fmt.Printf("\n  ✅ Training complete in %v\n", time.Since(phase2Start))
		radioStats := org.RadioCortex.Stats()
		fmt.Printf("     Radio ticks: %d | Alive: %d/%d | Avg amplitude: %d\n\n",
			radioStats.TickCount, radioStats.AliveNeurons, radioStats.TotalNeurons, radioStats.AvgAmplitude)
	}

	// ══════════════════════════════════════════════════════════
	// PHASE 3: Recall testing
	// ══════════════════════════════════════════════════════════
	fmt.Println("┌──────────────────────────────────────────────────────────────┐")
	fmt.Println("│  🧪 PHASE 3: RECALL TESTING                                 │")
	fmt.Println("└──────────────────────────────────────────────────────────────┘")

	testQuestions := []struct{ Q, Expected string }{
		{"ce face pisica", "pisica doarme"},
		{"ce face câinele", "câinele aleargă"},
		{"unde doarme pisica", "canapea"},
		{"ce culoare are cerul", "albastru"},
		{"ce mănâncă pisica", "pește"},
		{"cum face câinele", "ham"},
		{"what is a neuron", "cell"},
		{"what is the brain", "organ"},
		{"what is memory", "store"},
		{"how do neurons communicate", "electrical"},
		{"what color is the sky", "blue"},
		{"how many legs does a dog have", "four"},
		{"hello", "hello"},
		{"what is your name", "nexus"},
		{"is fire hot or cold", "hot"},
		{"ce este soarele", "stea"},
	}

	correct, partial := 0, 0
	for _, t := range testQuestions {
		response := org.Process(t.Q)
		matched := strings.Contains(strings.ToLower(response), strings.ToLower(t.Expected))
		icon := "❌"
		if matched {
			icon = "✅"
			correct++
		} else if response != "" && response != "(no confident response)" {
			icon = "🔶"
			partial++
		}
		display := response
		if len(display) > 60 {
			display = display[:60] + "..."
		}
		fmt.Printf("  %s Q: %-35s → %s\n", icon, t.Q, display)
	}

	fmt.Printf("\n  📊 Results: %d/%d correct, %d partial, %d missed\n",
		correct, len(testQuestions), partial, len(testQuestions)-correct-partial)

	// ══════════════════════════════════════════════════════════
	// PHASE 4: Final stats
	// ══════════════════════════════════════════════════════════
	fmt.Println()
	fmt.Println("┌──────────────────────────────────────────────────────────────┐")
	fmt.Println("│  📊 FINAL STATS                                             │")
	fmt.Println("└──────────────────────────────────────────────────────────────┘")

	stats := org.Stats()
	radioFinal := org.RadioCortex.Stats()

	fmt.Printf("  🧠 Hippocampus memories : %d\n", stats.HippocampusMemories)
	fmt.Printf("  🗣  Broca patterns       : %d\n", stats.BrocaPatterns)
	fmt.Printf("  👂 Wernicke rules        : %d\n", stats.WernickeNGrams)
	fmt.Printf("  🎯 Prefrontal synapses   : %d\n", stats.PrefrontalSynapses)
	fmt.Printf("  📡 Radio neurons         : %d (alive: %d)\n", radioFinal.TotalNeurons, radioFinal.AliveNeurons)
	fmt.Printf("  📡 Radio ticks           : %d\n", radioFinal.TickCount)
	fmt.Printf("  📡 Radio avg amplitude   : %d / 255\n", radioFinal.AvgAmplitude)
	fmt.Printf("  💚 Mood                  : %s\n", stats.EmotionalMood)
	fmt.Printf("  🪞 Self accuracy         : %d / 255\n", stats.SelfAccuracy)
	fmt.Printf("  💬 Total interactions     : %d\n", stats.InteractionCount)

	fmt.Println()
	fmt.Println("  ✅ Training session complete!")

	// Save
	if err := org.Save(cfg.DataDir); err != nil {
		fmt.Printf("  ⚠️  Save error: %v\n", err)
	} else {
		fmt.Println("  💾 Organism saved to", cfg.DataDir)
	}
}
