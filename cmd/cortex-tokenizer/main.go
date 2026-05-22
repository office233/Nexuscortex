package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"nexus-cortex/cortex"
)

// ─────────────────────────────────────────────────────────────────────
// cortex-tokenizer — CLI tool for training and testing BPE tokenizers
// ─────────────────────────────────────────────────────────────────────
//
// Usage:
//   Train a new tokenizer:
//     go run cmd/cortex-tokenizer/main.go -train -input corpus.txt -vocab-size 32768 -output tokenizer.json
//
//   Encode text with an existing tokenizer:
//     go run cmd/cortex-tokenizer/main.go -encode "Hello world" -tokenizer tokenizer.json
//
//   Interactive mode:
//     go run cmd/cortex-tokenizer/main.go -interactive -tokenizer tokenizer.json

func main() {
	// Flags
	trainMode := flag.Bool("train", false, "Train a new tokenizer from corpus")
	inputFile := flag.String("input", "", "Input corpus file (one sentence per line)")
	outputFile := flag.String("output", "tokenizer.json", "Output tokenizer file path")
	vocabSize := flag.Int("vocab-size", 32768, "Target vocabulary size")
	tokenizerFile := flag.String("tokenizer", "", "Path to trained tokenizer for encode/decode")
	encodeText := flag.String("encode", "", "Text to encode")
	interactive := flag.Bool("interactive", false, "Interactive encode/decode mode")

	flag.Parse()

	if *trainMode {
		runTrain(*inputFile, *outputFile, *vocabSize)
		return
	}

	if *encodeText != "" {
		if *tokenizerFile == "" {
			fmt.Fprintln(os.Stderr, "Error: -tokenizer required for encoding")
			os.Exit(1)
		}
		runEncode(*tokenizerFile, *encodeText)
		return
	}

	if *interactive {
		if *tokenizerFile == "" {
			fmt.Fprintln(os.Stderr, "Error: -tokenizer required for interactive mode")
			os.Exit(1)
		}
		runInteractive(*tokenizerFile)
		return
	}

	flag.Usage()
}

func runTrain(inputFile, outputFile string, vocabSize int) {
	if inputFile == "" {
		fmt.Fprintln(os.Stderr, "Error: -input corpus file required for training")
		os.Exit(1)
	}

	fmt.Printf("╔══════════════════════════════════════════════════════╗\n")
	fmt.Printf("║      Nexus Cortex — BPE Tokenizer Trainer          ║\n")
	fmt.Printf("╚══════════════════════════════════════════════════════╝\n\n")

	// Read corpus
	fmt.Printf("[1/3] Reading corpus from %s...\n", inputFile)
	lines, err := readLines(inputFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading corpus: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("      → %d lines loaded\n\n", len(lines))

	// Train
	fmt.Printf("[2/3] Training BPE tokenizer (target vocab: %d)...\n", vocabSize)
	start := time.Now()

	tok := cortex.NewBPETokenizer(vocabSize)
	tok.Train(lines)

	elapsed := time.Since(start)
	fmt.Printf("      → Training completed in %s\n", elapsed.Round(time.Millisecond))
	fmt.Printf("      → Final vocab: %d tokens, %d merges\n\n", tok.ActualVocabSize(), len(tok.Merges))

	// Save
	fmt.Printf("[3/3] Saving tokenizer to %s...\n", outputFile)
	if err := tok.Save(outputFile); err != nil {
		fmt.Fprintf(os.Stderr, "Error saving: %v\n", err)
		os.Exit(1)
	}

	info, _ := os.Stat(outputFile)
	fmt.Printf("      → Saved (%d bytes)\n\n", info.Size())

	// Quick test
	fmt.Println("── Quick Test ─────────────────────────────────────────")
	testTexts := []string{
		"Hello world",
		"the cat sat on the mat",
	}
	if len(lines) > 0 {
		// Use first line of corpus
		firstLine := strings.TrimSpace(lines[0])
		if len(firstLine) > 80 {
			firstLine = firstLine[:80]
		}
		testTexts = append(testTexts, firstLine)
	}

	for _, text := range testTexts {
		ids := tok.Encode(text)
		tokens := tok.DecodeTokens(ids)
		decoded := tok.Decode(ids)
		fmt.Printf("  Input:   %q\n", text)
		fmt.Printf("  Tokens:  %v\n", tokens)
		fmt.Printf("  IDs:     %v\n", ids)
		fmt.Printf("  Decoded: %q\n", decoded)
		if decoded == text {
			fmt.Println("  ✓ Roundtrip OK")
		} else {
			fmt.Println("  ✗ Roundtrip FAILED")
		}
		fmt.Println()
	}

	fmt.Println("Done.")
}

func runEncode(tokenizerFile, text string) {
	tok, err := cortex.LoadBPETokenizer(tokenizerFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading tokenizer: %v\n", err)
		os.Exit(1)
	}

	ids := tok.Encode(text)
	tokens := tok.DecodeTokens(ids)
	decoded := tok.Decode(ids)

	fmt.Printf("Input:     %q\n", text)
	fmt.Printf("Token IDs: %v\n", ids)
	fmt.Printf("Tokens:    %v\n", tokens)
	fmt.Printf("Decoded:   %q\n", decoded)
	fmt.Printf("Count:     %d tokens\n", len(ids))
}

func runInteractive(tokenizerFile string) {
	tok, err := cortex.LoadBPETokenizer(tokenizerFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading tokenizer: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Nexus BPE Tokenizer — Interactive Mode\n")
	fmt.Printf("Vocab: %d tokens, %d merges\n", tok.ActualVocabSize(), len(tok.Merges))
	fmt.Printf("Type text to tokenize (Ctrl+C to exit):\n\n")

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("> ")
		if !scanner.Scan() {
			break
		}
		text := scanner.Text()
		if text == "" {
			continue
		}

		ids := tok.Encode(text)
		tokens := tok.DecodeTokens(ids)
		decoded := tok.Decode(ids)

		fmt.Printf("  Tokens:  %v\n", tokens)
		fmt.Printf("  IDs:     %v\n", ids)
		fmt.Printf("  Count:   %d\n", len(ids))
		fmt.Printf("  Decoded: %q\n", decoded)
		if decoded == text {
			fmt.Println("  ✓ Roundtrip OK")
		} else {
			fmt.Println("  ✗ Roundtrip MISMATCH")
		}
		fmt.Println()
	}
}

func readLines(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var lines []string
	scanner := bufio.NewScanner(f)
	// Increase buffer for long lines
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" {
			lines = append(lines, line)
		}
	}
	return lines, scanner.Err()
}
