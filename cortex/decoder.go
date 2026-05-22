package cortex

import (
	"sort"
)

// ─────────────────────────────────────────────────────────────────────
// Decoder — Spike Patterns to Text
// ─────────────────────────────────────────────────────────────────────
//
// The Decoder converts network activation patterns (which neurons
// fired) back into words by comparing the firing pattern against all
// known word SDRs. The word whose SDR has the highest overlap with
// the observed pattern is the decoded result.
//
// This is the inverse of the Encoder: where the Encoder maps
// words → SDRs → neuron inputs, the Decoder maps
// neuron outputs → SDRs → words.

// DecodedWord represents a candidate word with its match confidence.
type DecodedWord struct {
	Word       string  // The decoded word
	Confidence uint8   // Match confidence (0 = no match, 255 = perfect)
}

// Decoder converts neural spike patterns back into text.
type Decoder struct {
	encoder *Encoder
	vocab   *Vocab
}

// NewDecoder creates a decoder linked to the given encoder and vocabulary.
func NewDecoder(encoder *Encoder, vocab *Vocab) *Decoder {
	return &Decoder{
		encoder: encoder,
		vocab:   vocab,
	}
}

// spikePatternToSDR converts a boolean spike pattern (true = neuron fired)
// into an SDR for comparison against known word representations.
func spikePatternToSDR(pattern []bool, sdrSize int) SDR {
	sdr := NewSDR(sdrSize)

	// Use the shorter of pattern length and sdrSize.
	n := len(pattern)
	if n > sdrSize {
		n = sdrSize
	}

	for i := 0; i < n; i++ {
		if pattern[i] {
			sdr.Set(i)
		}
	}
	return sdr
}

// DecodeSpikePattern converts a spike pattern into the best matching word.
//
// The pattern is a boolean slice where pattern[i] == true means neuron i
// fired. This is compared against every known word SDR using the Similarity
// metric (overlap / max active count). Returns the best matching word and
// its confidence score (0.0–1.0).
//
// Returns "<UNK>" with 0.0 confidence if no words are known or
// no overlap is found.
func (d *Decoder) DecodeSpikePattern(pattern []bool, sdrSize int) (string, uint8) {
	results := d.DecodeTopK(pattern, sdrSize, 1)
	if len(results) == 0 {
		return "<UNK>", 0
	}
	return results[0].Word, results[0].Confidence
}

// DecodeTopK returns the top-K matching words for a spike pattern,
// sorted by confidence (highest first).
//
// This performs an exhaustive comparison against all known word SDRs.
// For large vocabularies, consider prefiltering or indexing strategies.
func (d *Decoder) DecodeTopK(pattern []bool, sdrSize int, k int) []DecodedWord {
	if len(d.encoder.wordSDRs) == 0 || k <= 0 {
		return nil
	}

	// Convert the spike pattern to an SDR.
	observed := spikePatternToSDR(pattern, sdrSize)
	if observed.ActiveCount == 0 {
		return nil
	}

	// Compare against every known word SDR.
	candidates := make([]DecodedWord, 0, len(d.encoder.wordSDRs))
	for wordID, wordSDR := range d.encoder.wordSDRs {
		sim := observed.Similarity(wordSDR)
		if sim > 0 {
			word := d.vocab.Decode(wordID)
			candidates = append(candidates, DecodedWord{
				Word:       word,
				Confidence: sim,
			})
		}
	}

	if len(candidates) == 0 {
		return nil
	}

	// Sort by confidence descending, then alphabetically for ties.
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].Confidence != candidates[j].Confidence {
			return candidates[i].Confidence > candidates[j].Confidence
		}
		return candidates[i].Word < candidates[j].Word
	})

	// Return at most k results.
	if k > len(candidates) {
		k = len(candidates)
	}
	return candidates[:k]
}
