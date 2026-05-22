package cortex

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
)

// ─────────────────────────────────────────────────────────────────────
// Encoder — Text to Spike Trains (SDR Patterns)
// ─────────────────────────────────────────────────────────────────────
//
// The Encoder converts text into Sparse Distributed Representations
// that can be fed into the spiking neural network. Each word gets a
// unique SDR, and semantically related words (those appearing in
// similar contexts) share overlapping bit patterns.
//
// Semantic overlap strategy:
//   A new word's SDR = 70% unique random bits + 30% borrowed from
//   context words that appeared nearby. This creates implicit
//   semantic similarity without requiring an external embedding model.

// Encoder maps words to SDR patterns for neural network input.
type Encoder struct {
	vocab       *Vocab
	sdrSize     int            // Total bit positions per SDR (e.g. 10000)
	activeCount int            // Target active bits per SDR (e.g. 50)
	wordSDRs    map[uint32]SDR // Cached word-ID → SDR mappings
	rng         *rand.Rand

	// contextWindow tracks recently encoded words for semantic overlap.
	// When a new word is encountered, its SDR borrows bits from these.
	contextWindow []uint32
	contextMax    int // Maximum context window size
}

// NewEncoder creates a new encoder tied to the given vocabulary.
// contextMax is read from Config.EncoderContextMaxSize, defaulting to 10.
func NewEncoder(vocab *Vocab, sdrSize, activeCount int, rng *rand.Rand, cfgs ...Config) *Encoder {
	contextMax := 10
	if len(cfgs) > 0 && cfgs[0].EncoderContextMaxSize > 0 {
		contextMax = cfgs[0].EncoderContextMaxSize
	}
	return &Encoder{
		vocab:       vocab,
		sdrSize:     sdrSize,
		activeCount: activeCount,
		wordSDRs:    make(map[uint32]SDR),
		rng:         rng,
		contextMax:  contextMax,
	}
}

// EncodeWord returns the SDR for a word, creating one if it doesn't exist.
//
// If the word is new, its SDR is built using the semantic overlap strategy:
//   - 70% of active bits are unique random positions
//   - 30% of active bits are borrowed from recent context words
//
// This ensures that words appearing in similar contexts will naturally
// have overlapping SDRs, enabling the network to generalize.
func (e *Encoder) EncodeWord(word string) SDR {
	wordID := e.vocab.GetOrCreate(word)

	// Return cached SDR if it already exists.
	if sdr, ok := e.wordSDRs[wordID]; ok {
		e.pushContext(wordID)
		return sdr
	}

	// Build a new SDR with semantic overlap.
	sdr := e.buildSemanticSDR(wordID)
	e.wordSDRs[wordID] = sdr
	e.pushContext(wordID)
	return sdr
}

// buildSemanticSDR creates an SDR for a new word using the 70/30 strategy.
// 70% of bits are randomly chosen, 30% are borrowed from context neighbors.
func (e *Encoder) buildSemanticSDR(wordID uint32) SDR {
	sdr := NewSDR(e.sdrSize)

	// Determine how many bits come from context vs random.
	contextBits := 0
	overlapTarget := e.activeCount * 30 / 100 // 30% from context

	// Borrow bits from context words (words seen recently nearby).
	if len(e.contextWindow) > 0 && overlapTarget > 0 {
		// Collect all active indices from context word SDRs.
		contextPool := make(map[int]bool)
		for _, ctxID := range e.contextWindow {
			if ctxSDR, ok := e.wordSDRs[ctxID]; ok {
				for _, idx := range ctxSDR.ActiveIndices() {
					contextPool[idx] = true
				}
			}
		}

		if len(contextPool) > 0 {
			// Shuffle context indices and pick up to overlapTarget.
			poolSlice := make([]int, 0, len(contextPool))
			for idx := range contextPool {
				poolSlice = append(poolSlice, idx)
			}
			e.rng.Shuffle(len(poolSlice), func(i, j int) {
				poolSlice[i], poolSlice[j] = poolSlice[j], poolSlice[i]
			})

			for _, idx := range poolSlice {
				if contextBits >= overlapTarget {
					break
				}
				if !sdr.IsActive(idx) {
					sdr.Set(idx)
					contextBits++
				}
			}
		}
	}

	// Fill remaining bits with unique random positions.
	remaining := e.activeCount - contextBits
	if remaining > 0 {
		// Collect candidate positions (not already set).
		for filled := 0; filled < remaining; {
			if sdr.ActiveCount >= e.sdrSize {
				break // Safety: cannot set more bits than exist
			}
			idx := e.rng.Intn(e.sdrSize)
			if !sdr.IsActive(idx) {
				sdr.Set(idx)
				filled++
			}
		}
	}

	return sdr
}

// pushContext adds a word ID to the context window (FIFO).
func (e *Encoder) pushContext(wordID uint32) {
	if len(e.contextWindow) >= e.contextMax {
		copy(e.contextWindow, e.contextWindow[1:])
		e.contextWindow[len(e.contextWindow)-1] = wordID
	} else {
		e.contextWindow = append(e.contextWindow, wordID)
	}
}

// EncodeText tokenizes the text and returns an SDR for each token.
func (e *Encoder) EncodeText(text string) []SDR {
	tokens := Tokenize(text)
	sdrs := make([]SDR, 0, len(tokens))
	for _, token := range tokens {
		sdrs = append(sdrs, e.EncodeWord(token))
	}
	return sdrs
}

// EncodeSentence creates a single combined SDR for an entire sentence.
// It uses a positional encoding strategy where each word's active bits
// are shifted by its position in the sentence. This solves the "bag of words"
// false-overlap problem, ensuring "Capitala Frantei" and "Capitala Romaniei"
// have distinct signatures while still allowing partial overlap.
func (e *Encoder) EncodeSentence(text string) SDR {
	tokens := Tokenize(text)
	if len(tokens) == 0 {
		return NewSDR(e.sdrSize)
	}

	combined := NewSDR(e.sdrSize)

	// Shift amount per token position (e.g., 10000 / 50 = 200 positions)
	shiftAmount := e.sdrSize / 50
	if shiftAmount == 0 {
		shiftAmount = 1
	}

	// Union in all word SDRs, shifted by their position index.
	for i, token := range tokens {
		wordSDR := e.EncodeWord(token)
		shift := (i * shiftAmount) % e.sdrSize
		
		for _, idx := range wordSDR.ActiveIndices() {
			shiftedIdx := (idx + shift) % e.sdrSize
			if !combined.IsActive(shiftedIdx) {
				combined.Set(shiftedIdx)
				combined.ActiveCount++
			}
		}
	}

	return combined
}

// ─────────────────────────────────────────────────────────────────────
// Persistence — Save and Load encoder state
// ─────────────────────────────────────────────────────────────────────

// encoderData is the JSON-serializable representation of an Encoder.
type encoderData struct {
	SDRSize     int                  `json:"sdr_size"`
	ActiveCount int                  `json:"active_count"`
	WordSDRs    map[uint32]sdrRecord `json:"word_sdrs"`
}

// sdrRecord is the JSON-serializable representation of a single SDR.
type sdrRecord struct {
	Size        int    `json:"size"`
	ActiveCount int    `json:"active_count"`
	Data        []byte `json:"data"` // PackBytes output
}

// Save writes the encoder's learned SDR mappings to a JSON file.
func (e *Encoder) Save(path string) error {
	ed := encoderData{
		SDRSize:     e.sdrSize,
		ActiveCount: e.activeCount,
		WordSDRs:    make(map[uint32]sdrRecord, len(e.wordSDRs)),
	}

	for id, sdr := range e.wordSDRs {
		ed.WordSDRs[id] = sdrRecord{
			Size:        sdr.Size,
			ActiveCount: sdr.ActiveCount,
			Data:        sdr.PackBytes(),
		}
	}

	data, err := json.MarshalIndent(ed, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal encoder: %w", err)
	}
	return os.WriteFile(path, data, 0644)
}

// LoadEncoder reads a saved encoder from disk and reconnects it to
// the provided vocabulary.
func LoadEncoder(path string, vocab *Vocab, rng *rand.Rand) (*Encoder, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read encoder: %w", err)
	}

	var ed encoderData
	if err := json.Unmarshal(raw, &ed); err != nil {
		return nil, fmt.Errorf("unmarshal encoder: %w", err)
	}

	enc := NewEncoder(vocab, ed.SDRSize, ed.ActiveCount, rng)
	for id, rec := range ed.WordSDRs {
		enc.wordSDRs[id] = UnpackBytes(rec.Data, rec.Size)
	}

	return enc, nil
}
