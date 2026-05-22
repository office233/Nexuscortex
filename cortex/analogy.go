package cortex

import "sort"

// ─────────────────────────────────────────────────────────────────────
// AnalogyEngine — Analogical Reasoning via SDR Transformations
// ─────────────────────────────────────────────────────────────────────
//
// Analogical reasoning finds structural parallels between concepts.
// If "king is to queen as man is to ???", the answer is "woman" because
// the SDR transformation (king→queen) is similar to (man→woman).
//
// We implement this using SDR difference patterns:
//   1. Compute the transformation: diff_ab = bits in B but not in A
//   2. Apply the transformation to C: candidate = C UNION diff_ab
//   3. Search the vocabulary for the word whose SDR is most similar
//      to the candidate.
//
// All arithmetic is integer-only (uint8 similarity scores).

// SimilarWord holds a word and its similarity score.
type SimilarWord struct {
	Word  string
	Score uint8
}

// AnalogyEngine provides analogical reasoning over the Brain's vocabulary.
type AnalogyEngine struct {
	Cfg     Config
	Brain   *Brain
	Encoder *Encoder
}

// NewAnalogyEngine creates an AnalogyEngine wired to the given Brain and Encoder.
func NewAnalogyEngine(cfg Config, brain *Brain, encoder *Encoder) *AnalogyEngine {
	return &AnalogyEngine{
		Cfg:     cfg,
		Brain:   brain,
		Encoder: encoder,
	}
}

// FindAnalogy solves "A is to B as C is to ???".
//
// It encodes A, B, C as SDRs, computes the transformation SDR
// (bits in B but not in A), applies it to C, and searches the
// Brain's vocabulary for the most similar word.
//
// Returns (best_word, confidence). If no match is found, returns ("", 0).
// Words a, b, c themselves are excluded from the result.
func (ae *AnalogyEngine) FindAnalogy(a, b, c string) (string, uint8) {
	if a == "" || b == "" || c == "" {
		return "", 0
	}

	sdrA := ae.Encoder.EncodeSentence(a)
	sdrB := ae.Encoder.EncodeSentence(b)
	sdrC := ae.Encoder.EncodeSentence(c)

	// Compute transformation: bits in B but not in A.
	diffAB := sdrB.Difference(sdrA)

	// Apply transformation to C: candidate = C UNION diff_ab.
	candidate := sdrC.Union(diffAB)

	// Determine which input words to exclude from results.
	excludeWords := make(map[string]bool, 3)
	for _, w := range Tokenize(a) {
		excludeWords[w] = true
	}
	for _, w := range Tokenize(b) {
		excludeWords[w] = true
	}
	for _, w := range Tokenize(c) {
		excludeWords[w] = true
	}

	// Scan vocab for best match.
	maxCandidates := ae.Cfg.AnalogyMaxCandidates
	if maxCandidates <= 0 {
		maxCandidates = 100
	}

	ae.Brain.mu.RLock()
	defer ae.Brain.mu.RUnlock()

	var bestWord string
	var bestScore uint8

	scanned := 0
	for word, id := range ae.Brain.Vocab.WordToID {
		if excludeWords[word] {
			continue
		}

		wordSDR, ok := ae.Encoder.wordSDRs[id]
		if !ok {
			continue
		}

		score := candidate.Similarity(wordSDR)
		if score > bestScore {
			bestScore = score
			bestWord = word
		}

		scanned++
		if scanned >= maxCandidates {
			break
		}
	}

	return bestWord, bestScore
}

// FindSimilar returns the topN most similar words to the given word,
// excluding the word itself. Uses a simple sorted slice.
func (ae *AnalogyEngine) FindSimilar(word string, topN int) []SimilarWord {
	if word == "" || topN <= 0 {
		return nil
	}

	wordSDR := ae.Encoder.EncodeSentence(word)
	if wordSDR.ActiveCount == 0 {
		return nil
	}

	// Tokenize the input word to exclude it from results.
	excludeWords := make(map[string]bool)
	for _, w := range Tokenize(word) {
		excludeWords[w] = true
	}

	ae.Brain.mu.RLock()
	defer ae.Brain.mu.RUnlock()

	var results []SimilarWord

	for w, id := range ae.Brain.Vocab.WordToID {
		if excludeWords[w] {
			continue
		}

		wSDR, ok := ae.Encoder.wordSDRs[id]
		if !ok {
			continue
		}

		score := wordSDR.Similarity(wSDR)
		if score == 0 {
			continue
		}

		results = append(results, SimilarWord{Word: w, Score: score})
	}

	// Sort by score descending, then by word ascending for stability.
	sort.Slice(results, func(i, j int) bool {
		if results[i].Score != results[j].Score {
			return results[i].Score > results[j].Score
		}
		return results[i].Word < results[j].Word
	})

	if len(results) > topN {
		results = results[:topN]
	}

	return results
}
