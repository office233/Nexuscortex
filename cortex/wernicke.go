package cortex

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// ─────────────────────────────────────────────────────────────────────
// Wernicke — Language Understanding
// ─────────────────────────────────────────────────────────────────────
//
// Named after Wernicke's area in the human brain, this module handles
// language comprehension. It converts raw text into a structured
// understanding result containing:
//
//   - Individual word SDRs for fine-grained pattern matching
//   - A combined SDR representing the entire input's meaning
//   - Extracted keywords based on n-gram frequency analysis
//   - Question detection via interrogative markers
//
// The Wernicke module learns contextual patterns through n-gram
// tracking: words that frequently appear together are recognized
// as significant phrases, and their constituent words are flagged
// as keywords.

// Wernicke performs language understanding by encoding text into
// SDR patterns and extracting semantic structure.
type Wernicke struct {
	Encoder          *Encoder
	Vocab            *Vocab
	NGrams           map[string]uint32
	KeywordThreshold uint32
}

// UnderstandingResult holds the output of language comprehension.
type UnderstandingResult struct {
	Words      []string // Tokenized input words
	WordSDRs   []SDR    // SDR for each word
	Combined   SDR      // Union SDR representing the full input
	KeyWords   []string // Detected keywords (high-frequency n-gram members)
	IsQuestion bool     // True if the input is a question
}

// NewWernicke creates a new language understanding module tied to the
// given vocabulary, encoder, and config.
func NewWernicke(vocab *Vocab, encoder *Encoder, cfg Config) *Wernicke {
	return &Wernicke{
		Encoder:          encoder,
		Vocab:            vocab,
		NGrams:           make(map[string]uint32),
		KeywordThreshold: cfg.WernickeKeywordThreshold,
	}
}

// Understand performs full language comprehension on the input text.
//
// The process:
//  1. Tokenize the input into words.
//  2. Encode each word into its SDR representation.
//  3. Build a combined SDR (union of all word SDRs).
//  4. Extract keywords using n-gram frequency analysis.
//  5. Detect whether the input is a question.
//
// The combined SDR can be fed directly into downstream modules
// (Broca for generation, Prefrontal for reasoning).
func (w *Wernicke) Understand(text string) UnderstandingResult {
	tokens := Tokenize(text)

	result := UnderstandingResult{
		Words:      tokens,
		WordSDRs:   make([]SDR, 0, len(tokens)),
		IsQuestion: detectQuestion(text, tokens),
	}

	if len(tokens) == 0 {
		result.Combined = NewSDR(w.Encoder.sdrSize)
		return result
	}

	// Encode each word and build the combined SDR.
	result.Combined = w.Encoder.EncodeWord(tokens[0])
	result.WordSDRs = append(result.WordSDRs, result.Combined)

	for i := 1; i < len(tokens); i++ {
		sdr := w.Encoder.EncodeWord(tokens[i])
		result.WordSDRs = append(result.WordSDRs, sdr)
		result.Combined = result.Combined.Union(sdr)
	}

	// Extract keywords using n-gram frequency data.
	result.KeyWords = w.extractKeywords(tokens)

	return result
}

// LearnContext updates n-gram frequency counts from the given word
// sequence. Tracks bigrams and trigrams to identify significant
// phrases over time. Higher-frequency n-grams indicate important
// contextual patterns.
func (w *Wernicke) LearnContext(words []string) {
	if len(words) < 2 {
		return
	}

	// Track bigrams.
	for i := 0; i < len(words)-1; i++ {
		key := words[i] + " " + words[i+1]
		w.NGrams[key]++
	}

	// Track trigrams.
	for i := 0; i < len(words)-2; i++ {
		key := words[i] + " " + words[i+1] + " " + words[i+2]
		w.NGrams[key]++
	}
}

// extractKeywords identifies important words based on n-gram frequency.
// A word is considered a keyword if it participates in any n-gram
// that has been observed at least keywordFreqThreshold times.
func (w *Wernicke) extractKeywords(tokens []string) []string {
	keywordFreqThreshold := w.KeywordThreshold
	if keywordFreqThreshold == 0 {
		keywordFreqThreshold = 2
	}

	if len(w.NGrams) == 0 {
		return nil
	}

	// Collect words that participate in frequent n-grams.
	seen := make(map[string]bool)
	var keywords []string

	for i := 0; i < len(tokens)-1; i++ {
		bigram := tokens[i] + " " + tokens[i+1]
		if count, ok := w.NGrams[bigram]; ok && count >= keywordFreqThreshold {
			if !seen[tokens[i]] && !isStopWord(tokens[i]) {
				seen[tokens[i]] = true
				keywords = append(keywords, tokens[i])
			}
			if !seen[tokens[i+1]] && !isStopWord(tokens[i+1]) {
				seen[tokens[i+1]] = true
				keywords = append(keywords, tokens[i+1])
			}
		}
	}

	for i := 0; i < len(tokens)-2; i++ {
		trigram := tokens[i] + " " + tokens[i+1] + " " + tokens[i+2]
		if count, ok := w.NGrams[trigram]; ok && count >= keywordFreqThreshold {
			for _, j := range []int{i, i + 1, i + 2} {
				if !seen[tokens[j]] && !isStopWord(tokens[j]) {
					seen[tokens[j]] = true
					keywords = append(keywords, tokens[j])
				}
			}
		}
	}

	return keywords
}

// detectQuestion returns true if the text appears to be a question.
// Detection is based ONLY on punctuation — no hardcoded word lists.
// The cortex must learn interrogative patterns through experience,
// not through pre-programmed rules.
func detectQuestion(text string, _ []string) bool {
	trimmed := strings.TrimSpace(text)
	if len(trimmed) > 0 && trimmed[len(trimmed)-1] == '?' {
		return true
	}
	return false
}

// isStopWord returns true if a word is too short to be meaningful.
// This is the ONLY non-learned heuristic: single-character tokens
// and punctuation. Everything else is determined by frequency.
// The Wernicke module's extractKeywords uses n-gram frequency to
// identify important words — high-frequency words are naturally
// de-prioritized without a hardcoded list.
func isStopWord(word string) bool {
	if len(word) <= 1 {
		return true
	}
	// Punctuation-only tokens.
	for _, ch := range word {
		if ch != '.' && ch != ',' && ch != '!' && ch != '?' && ch != ':' && ch != ';' {
			return false
		}
	}
	return true
}

// ─────────────────────────────────────────────────────────────────────
// Persistence — Save / Load
// ─────────────────────────────────────────────────────────────────────

// wernickeSaveData is the JSON-serializable snapshot of Wernicke state.
// Only NGrams are persisted — Encoder and Vocab are wired externally.
type wernickeSaveData struct {
	NGrams map[string]uint32 `json:"ngrams"`
}

// Save persists the Wernicke n-gram context to a JSON file.
// Uses atomic temp-file + sync + rename to prevent corruption on crash.
func (w *Wernicke) Save(path string) error {
	data := wernickeSaveData{
		NGrams: w.NGrams,
	}
	raw, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return fmt.Errorf("wernicke save marshal: %w", err)
	}

	tmpPath := path + ".tmp"
	f, err := os.OpenFile(tmpPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0600)
	if err != nil {
		return fmt.Errorf("wernicke create tmp: %w", err)
	}
	if _, err := f.Write(raw); err != nil {
		f.Close()
		os.Remove(tmpPath)
		return fmt.Errorf("wernicke write: %w", err)
	}
	if err := f.Sync(); err != nil {
		f.Close()
		os.Remove(tmpPath)
		return fmt.Errorf("wernicke sync: %w", err)
	}
	f.Close()
	return os.Rename(tmpPath, path)
}

// LoadWernicke restores Wernicke state from a JSON file and wires the
// provided Vocab, Encoder, and Config. Returns (nil, nil) if the file does not
// exist, allowing the caller to fall back to NewWernicke.
func LoadWernicke(path string, vocab *Vocab, encoder *Encoder, cfg Config) (*Wernicke, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("wernicke load read: %w", err)
	}
	var data wernickeSaveData
	if err := json.Unmarshal(raw, &data); err != nil {
		return nil, fmt.Errorf("wernicke load unmarshal: %w", err)
	}
	if data.NGrams == nil {
		data.NGrams = make(map[string]uint32)
	}
	return &Wernicke{
		Encoder:          encoder,
		Vocab:            vocab,
		NGrams:           data.NGrams,
		KeywordThreshold: cfg.WernickeKeywordThreshold,
	}, nil
}
