package cortex

// SignalCodec maps tokens to frequency "chords" and back.
//
// Each token is represented by a set of 13 frequencies (out of 256),
// like a musical chord. This is deterministic: the same token always
// produces the same frequencies.
//
// The reverse mapping allows decoding: given a set of active frequencies
// on the bus, find the token whose chord overlaps the most.
//
// This is the CRITICAL missing piece that connects words to frequencies.
// Without it, RadioCortex neurons oscillate randomly.
// With it, "pisica" always activates [23, 87, 142, ...] — and neurons
// can learn to associate those frequencies with meaning.
type SignalCodec struct {
	vocabSize    int
	freqsPerTok  int        // how many frequencies per token (default 13)
	tokenFreqs   [][]uint8  // tokenFreqs[tokenID] = sorted list of frequencies
	freqToTokens [256][]int // reverse index: which tokens use this frequency

	// Semantic: if set, delegate frequency lookups here first.
	// SemanticFreqCodec provides co-occurrence-based frequency assignment.
	// SignalCodec falls back to hash if Semantic has no mapping.
	Semantic *SemanticFreqCodec
}

// NewSignalCodec creates a codec with default 13 frequencies per token.
// 13/256 ≈ 5% density — sparse enough to distinguish tokens,
// dense enough for overlap between related concepts.
func NewSignalCodec(vocabSize int) *SignalCodec {
	return NewSignalCodecCustom(vocabSize, 13)
}

// NewSignalCodecCustom creates a codec with a custom number of frequencies per token.
func NewSignalCodecCustom(vocabSize, freqsPerToken int) *SignalCodec {
	if freqsPerToken < 1 {
		freqsPerToken = 1
	}
	if freqsPerToken > 64 {
		freqsPerToken = 64 // cap at 25% of spectrum
	}

	sc := &SignalCodec{
		vocabSize:   vocabSize,
		freqsPerTok: freqsPerToken,
		tokenFreqs:  make([][]uint8, vocabSize),
	}
	sc.buildSpectrum()
	return sc
}

// buildSpectrum computes the frequency chord for every token using
// a deterministic hash function (FNV-inspired mixing).
func (sc *SignalCodec) buildSpectrum() {
	for tokenID := 0; tokenID < sc.vocabSize; tokenID++ {
		seen := make(map[uint8]bool, sc.freqsPerTok)
		freqs := make([]uint8, 0, sc.freqsPerTok)

		// FNV-inspired mixing: deterministic, good distribution
		seed := uint32(tokenID)*2654435761 + 0x9E3779B9

		for len(freqs) < sc.freqsPerTok {
			seed = seed*1664525 + 1013904223 // LCG step
			freq := uint8(seed >> 24)        // top 8 bits

			if !seen[freq] {
				seen[freq] = true
				freqs = append(freqs, freq)
			}
		}

		sc.tokenFreqs[tokenID] = freqs

		// Build reverse index
		for _, f := range freqs {
			sc.freqToTokens[f] = append(sc.freqToTokens[f], tokenID)
		}
	}
}

// TokenFreqs returns the frequency chord for a token.
// If a SemanticFreqCodec is attached and has a mapping, use that.
// Otherwise fall back to hash-based frequencies.
func (sc *SignalCodec) TokenFreqs(tokenID int) []uint8 {
	// Try semantic codec first
	if sc.Semantic != nil {
		if freqs := sc.Semantic.Encode(tokenID); len(freqs) > 0 {
			// Check if it's a real mapping (not just hash fallback)
			if _, ok := sc.Semantic.tokenFreqSet[tokenID]; ok {
				return freqs
			}
		}
	}
	// Fall back to hash-based frequencies
	if tokenID < 0 || tokenID >= sc.vocabSize {
		return nil
	}
	return sc.tokenFreqs[tokenID]
}

// EncodeToken injects a token's frequencies onto a RadioBus.
func (sc *SignalCodec) EncodeToken(bus *RadioBus, tokenID int, amplitude uint8) {
	freqs := sc.TokenFreqs(tokenID)
	if freqs == nil {
		return
	}
	// Each frequency in the chord gets emitted with the given amplitude.
	// Phase is derived from token position (for temporal ordering).
	for i, f := range freqs {
		phase := uint8(i * 256 / len(freqs)) // spread phase across cycle
		bus.Emit(f, amplitude, phase, false)
	}
}

// EncodeTokens injects multiple tokens with sequential phase offsets
// to preserve word order (earlier tokens = earlier phase).
func (sc *SignalCodec) EncodeTokens(bus *RadioBus, tokenIDs []int, amplitude uint8) {
	for pos, tid := range tokenIDs {
		freqs := sc.TokenFreqs(tid)
		if freqs == nil {
			continue
		}
		// Phase offset based on position in sequence
		posPhase := uint8(pos * 256 / max(len(tokenIDs), 1))
		for _, f := range freqs {
			bus.Emit(f, amplitude, posPhase, false)
		}
	}
}

// DecodeToken reads the bus spectrum and returns the token whose
// frequency chord has the highest total signal energy.
func (sc *SignalCodec) DecodeToken(bus *RadioBus) (tokenID int, score int64) {
	spectrum := bus.Spectrum()

	bestToken := -1
	var bestScore int64

	for tid := 0; tid < sc.vocabSize; tid++ {
		freqs := sc.tokenFreqs[tid]
		var total int64
		for _, f := range freqs {
			sig := int64(spectrum[f])
			if sig > 0 {
				total += sig
			}
		}
		if total > bestScore {
			bestScore = total
			bestToken = tid
		}
	}

	return bestToken, bestScore
}

// DecodeTopK returns the top K tokens sorted by spectrum match score.
func (sc *SignalCodec) DecodeTopK(bus *RadioBus, k int) []TokenScore {
	spectrum := bus.Spectrum()

	scores := make([]TokenScore, 0, k)
	for tid := 0; tid < sc.vocabSize; tid++ {
		freqs := sc.tokenFreqs[tid]
		var total int64
		for _, f := range freqs {
			sig := int64(spectrum[f])
			if sig > 0 {
				total += sig
			}
		}
		if total <= 0 {
			continue
		}

		// Insert into sorted top-k
		ts := TokenScore{TokenID: tid, Score: total}
		inserted := false
		for i, s := range scores {
			if total > s.Score {
				scores = append(scores, TokenScore{})
				copy(scores[i+1:], scores[i:])
				scores[i] = ts
				inserted = true
				break
			}
		}
		if !inserted && len(scores) < k {
			scores = append(scores, ts)
		}
		if len(scores) > k {
			scores = scores[:k]
		}
	}

	return scores
}

// TokenScore pairs a token ID with its frequency match score.
type TokenScore struct {
	TokenID int
	Score   int64
}

// FrequencyOverlap returns the Jaccard similarity between two tokens' frequency sets.
// 0.0 = no overlap, 1.0 = identical. Used to measure semantic proximity.
func (sc *SignalCodec) FrequencyOverlap(tokenA, tokenB int) float32 {
	if tokenA < 0 || tokenA >= sc.vocabSize || tokenB < 0 || tokenB >= sc.vocabSize {
		return 0
	}

	setA := make(map[uint8]bool, sc.freqsPerTok)
	for _, f := range sc.tokenFreqs[tokenA] {
		setA[f] = true
	}

	intersection := 0
	for _, f := range sc.tokenFreqs[tokenB] {
		if setA[f] {
			intersection++
		}
	}

	union := len(sc.tokenFreqs[tokenA]) + len(sc.tokenFreqs[tokenB]) - intersection
	if union == 0 {
		return 0
	}
	return float32(intersection) / float32(union)
}

// GrowVocab expands the codec to accommodate new tokens.
// Called when the Vocab grows beyond the original size.
func (sc *SignalCodec) GrowVocab(newSize int) {
	if newSize <= sc.vocabSize {
		return
	}

	oldSize := sc.vocabSize
	sc.vocabSize = newSize
	sc.tokenFreqs = append(sc.tokenFreqs, make([][]uint8, newSize-oldSize)...)

	// Build spectrum for new tokens only
	for tokenID := oldSize; tokenID < newSize; tokenID++ {
		seen := make(map[uint8]bool, sc.freqsPerTok)
		freqs := make([]uint8, 0, sc.freqsPerTok)

		seed := uint32(tokenID)*2654435761 + 0x9E3779B9
		for len(freqs) < sc.freqsPerTok {
			seed = seed*1664525 + 1013904223
			freq := uint8(seed >> 24)
			if !seen[freq] {
				seen[freq] = true
				freqs = append(freqs, freq)
			}
		}

		sc.tokenFreqs[tokenID] = freqs
		for _, f := range freqs {
			sc.freqToTokens[f] = append(sc.freqToTokens[f], tokenID)
		}
	}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
