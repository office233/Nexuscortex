package cortex

import (
	"sort"
)

// ═══════════════════════════════════════════════════════════════════
// SemanticFreqCodec — Maps tokens to SEMANTIC frequencies
// ═══════════════════════════════════════════════════════════════════
//
// Unlike random hashing (where "pisica" and "computer" might land on the
// same frequency), this codec ensures:
//   - Similar words → similar frequencies (nearby on the bus)
//   - Different words → different frequencies (far apart)
//
// Method: co-occurrence based clustering.
//   1. Track which words appear together in training data
//   2. Words that co-occur frequently get nearby frequencies
//   3. Words that never co-occur get distant frequencies
//
// This is a lightweight alternative to word2vec — no matrix multiply needed.

// SemanticFreqCodec maps tokens to semantically meaningful frequencies.
type SemanticFreqCodec struct {
	// tokenFreqs[tokenID] = primary frequency for this token
	tokenFreqs map[int]uint8

	// tokenFreqSet[tokenID] = set of frequencies this token activates
	// (a token can activate multiple nearby frequencies for fuzziness)
	tokenFreqSet map[int][]uint8

	// cooccurrence[tokenA][tokenB] = how often A and B appear together
	cooccurrence map[int]map[int]int

	// freqToTokens[freq] = list of token IDs assigned to this frequency
	// This is the INVERSE INDEX for fast decode
	freqToTokens map[uint8][]int

	// vocabSize tracks how many tokens we know
	vocabSize int

	// dirty marks whether frequencies need recalculation
	dirty bool
}

// NewSemanticFreqCodec creates a new semantic frequency codec.
func NewSemanticFreqCodec() *SemanticFreqCodec {
	return &SemanticFreqCodec{
		tokenFreqs:   make(map[int]uint8),
		tokenFreqSet: make(map[int][]uint8),
		cooccurrence: make(map[int]map[int]int),
		freqToTokens: make(map[uint8][]int),
	}
}

// ObserveCooccurrence records that tokens appeared together in context.
// Call this during training to build the co-occurrence matrix.
// Complexity: O(n²) where n = len(tokenIDs). This is inherent to pairwise
// co-occurrence recording and is acceptable because it runs during training,
// not inference.
func (s *SemanticFreqCodec) ObserveCooccurrence(tokenIDs []int) {
	for i, a := range tokenIDs {
		if s.cooccurrence[a] == nil {
			s.cooccurrence[a] = make(map[int]int)
		}
		for j, b := range tokenIDs {
			if i != j {
				s.cooccurrence[a][b]++
			}
		}
	}
	s.dirty = true
}

// AssignFrequencies clusters tokens by co-occurrence and assigns frequencies.
// Uses spectral-like ordering: tokens that co-occur get nearby frequencies.
//
// Algorithm:
//  1. Start with the most connected token → frequency 128 (center)
//  2. For each remaining token, place it near its most co-occurring neighbor
//  3. Spread tokens evenly across the 0-255 range
func (s *SemanticFreqCodec) AssignFrequencies() {
	tokens := make([]int, 0, len(s.cooccurrence))
	for t := range s.cooccurrence {
		tokens = append(tokens, t)
	}
	if len(tokens) == 0 {
		return
	}

	// Pre-compute total co-occurrence count per token to avoid O(n²)
	// recomputation inside the sort comparator.
	sumMap := make(map[int]int, len(tokens))
	for _, t := range tokens {
		sum := 0
		for _, v := range s.cooccurrence[t] {
			sum += v
		}
		sumMap[t] = sum
	}

	// Sort by total co-occurrence count (most connected first)
	sort.Slice(tokens, func(i, j int) bool {
		if sumMap[tokens[i]] != sumMap[tokens[j]] {
			return sumMap[tokens[i]] > sumMap[tokens[j]]
		}
		return tokens[i] < tokens[j] // Deterministic tie-breaker
	})

	// Order tokens by similarity chain (greedy nearest-neighbor).
	// Complexity: O(V²) where V = vocabulary size. This is acceptable because:
	//   - V is typically < 50K tokens
	//   - This only runs when the dirty flag is set (after ObserveCooccurrence),
	//     not on every inference call
	//   - A more complex algorithm (e.g. TSP solver) is not worth the code
	//     complexity for this use case
	ordered := make([]int, 0, len(tokens))
	used := make(map[int]bool)

	// Start with most connected
	ordered = append(ordered, tokens[0])
	used[tokens[0]] = true

	for len(ordered) < len(tokens) {
		last := ordered[len(ordered)-1]
		bestToken := -1
		bestScore := -1

		for _, t := range tokens {
			if used[t] {
				continue
			}
			score := s.cooccurrence[last][t] + s.cooccurrence[t][last]
			if score > bestScore {
				bestScore = score
				bestToken = t
			}
		}

		if bestToken == -1 {
			// No more connected tokens, add remaining in order
			for _, t := range tokens {
				if !used[t] {
					ordered = append(ordered, t)
					used[t] = true
				}
			}
			break
		}

		ordered = append(ordered, bestToken)
		used[bestToken] = true
	}

	// Assign frequencies: spread ordered tokens across 0-255
	s.tokenFreqs = make(map[int]uint8, len(ordered))
	s.tokenFreqSet = make(map[int][]uint8, len(ordered))
	s.freqToTokens = make(map[uint8][]int)

	for i, tokenID := range ordered {
		// Map index to frequency: spread across 0-255 using clean integer math
		freq := uint8(i * 255 / max(1, len(ordered)-1))

		s.tokenFreqs[tokenID] = freq

		// Each token activates its primary freq + 2 neighbors (fuzziness)
		freqs := []uint8{freq}
		if freq > 0 {
			freqs = append(freqs, freq-1)
		}
		if freq < 255 {
			freqs = append(freqs, freq+1)
		}
		s.tokenFreqSet[tokenID] = freqs

		// Build inverse index
		for _, f := range freqs {
			s.freqToTokens[f] = append(s.freqToTokens[f], tokenID)
		}
	}

	s.vocabSize = len(ordered)
	s.dirty = false
}

// Encode returns the semantic frequencies for a token.
func (s *SemanticFreqCodec) Encode(tokenID int) []uint8 {
	if s.dirty {
		s.AssignFrequencies()
	}
	if freqs, ok := s.tokenFreqSet[tokenID]; ok {
		return freqs
	}
	// Unknown token: hash to a frequency
	return []uint8{uint8(tokenID % 256)}
}

// EncodeTokens returns all active frequencies for a set of tokens.
func (s *SemanticFreqCodec) EncodeTokens(tokenIDs []int) []uint8 {
	seen := make(map[uint8]bool)
	var result []uint8
	for _, tid := range tokenIDs {
		for _, f := range s.Encode(tid) {
			if !seen[f] {
				seen[f] = true
				result = append(result, f)
			}
		}
	}
	return result
}

// DecodeFreq returns candidate token IDs for a frequency (inverse index).
func (s *SemanticFreqCodec) DecodeFreq(freq uint8) []int {
	return s.freqToTokens[freq]
}

// PrimaryFreq returns the primary frequency for a token.
func (s *SemanticFreqCodec) PrimaryFreq(tokenID int) uint8 {
	if f, ok := s.tokenFreqs[tokenID]; ok {
		return f
	}
	return uint8(tokenID % 256)
}

// Stats returns codec statistics.
func (s *SemanticFreqCodec) Stats() (numTokens, numFreqsUsed, totalCooccurrences int) {
	numTokens = len(s.tokenFreqs)
	freqUsed := make(map[uint8]bool)
	for _, f := range s.tokenFreqs {
		freqUsed[f] = true
	}
	numFreqsUsed = len(freqUsed)
	for _, m := range s.cooccurrence {
		for _, v := range m {
			totalCooccurrences += v
		}
	}
	return
}
