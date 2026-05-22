package cortex

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"strings"
)

// ─────────────────────────────────────────────────────────────────────
// hippocampus.go — Episodic Memory System
// ─────────────────────────────────────────────────────────────────────
//
// The hippocampus is the brain's primary structure for forming and
// retrieving episodic memories — specific experiences bound to a
// temporal context. In biology, the hippocampal formation (CA1, CA3,
// dentate gyrus) performs pattern separation on input and pattern
// completion on recall, enabling one-shot learning of novel associations.
//
// This implementation stores input→output SDR associations as discrete
// Memory records. Each memory has:
//   - Strength (0–255): reinforced on repeated encoding, decayed on
//     consolidation. Models synaptic long-term potentiation (LTP).
//   - Age: incremented every consolidation cycle. Old, weak memories
//     are pruned, mirroring synaptic homeostasis during sleep.
//   - Context: a string tag for associative cuing (e.g., conversation ID).
//
// Recall is performed by SDR overlap: the query is compared against
// every stored input pattern and the best match above threshold is
// returned. This is analogous to CA3 auto-associative recall.

// Memory represents a single episodic record stored in the hippocampus.
type Memory struct {
	Input    SDR    // Sparse pattern that was observed
	Output   SDR    // Associated response pattern
	Strength uint8  // Synaptic strength (0 = weakest, 255 = strongest)
	Age      uint32 // Number of consolidation cycles since encoding
	Context  string // Contextual tag for associative cuing
}

// Hippocampus is an episodic memory store with capacity limits and
// consolidation dynamics.
type Hippocampus struct {
	Memories              []Memory
	MaxMemories           int
	ReconsolidationThresh uint8
	InitialStrength       uint8
	LtpThreshold          uint8

	// keywordIndex maps lowercased content keywords to the indices of
	// memories whose Context contains that keyword. This provides
	// lexical recall that avoids the false-overlap problem of
	// union-based SDR encoding on long sentences.
	keywordIndex map[string][]int
}

// hippoStopWords contains common function words and punctuation that
// should be excluded from the keyword index. These carry little
// discriminative meaning for memory retrieval.
var hippoStopWords = map[string]bool{
	"what": true, "is": true, "the": true, "a": true, "an": true,
	"of": true, "in": true, "to": true, "and": true, "or": true,
	"for": true, "that": true, "this": true, "it": true, "are": true,
	"was": true, "were": true, "be": true, "been": true, "how": true,
	"who": true, "which": true, "?": true, ".": true,
}

// extractKeywords tokenises a context string and returns the unique,
// lowercased content words (stop words removed).
func extractKeywords(context string) []string {
	tokens := Tokenize(context)
	seen := make(map[string]bool, len(tokens))
	var kw []string
	for _, t := range tokens {
		t = strings.ToLower(t)
		if hippoStopWords[t] || t == "" {
			continue
		}
		if seen[t] {
			continue
		}
		seen[t] = true
		kw = append(kw, t)
	}
	return kw
}

// NewHippocampus creates a hippocampus with the given config.
func NewHippocampus(cfg Config) *Hippocampus {
	return &Hippocampus{
		Memories:              make([]Memory, 0, cfg.MaxMemories),
		MaxMemories:           cfg.MaxMemories,
		ReconsolidationThresh: cfg.HippoReconsolidationThresh,
		InitialStrength:       cfg.HippoInitialStrength,
		LtpThreshold:          cfg.HippoLtpThreshold,
		keywordIndex:          make(map[string][]int),
	}
}

// ─────────────────────────────────────────────────────────────────────
// Store — Encode a new episodic memory
// ─────────────────────────────────────────────────────────────────────

// Store encodes an input→output association into episodic memory.
//
// If a highly similar memory already exists (overlap ≥ ReconconsolidationThresh),
// the existing memory is strengthened rather than creating a duplicate. This
// models reconsolidation — the biological process where retrieving a memory
// makes it labile and allows it to be updated.
//
// When the store is full, the weakest (lowest-strength) memory is
// evicted to make room, modeling synaptic homeostasis.
//
// The similarity threshold, initial strength, and LTP threshold are dynamic.
func (h *Hippocampus) Store(input SDR, output SDR, context string) {
	// Ensure the keyword index map is initialised (may be nil after
	// deserialisation from an older file format).
	if h.keywordIndex == nil {
		h.keywordIndex = make(map[string][]int)
	}

	// Check for an existing similar memory to reconsolidate.
	for i := range h.Memories {
		sim := h.Memories[i].Input.Similarity(input)
		if sim >= h.ReconsolidationThresh {
			// Reconsolidate: strengthen and update.
			if h.Memories[i].Strength < 255 {
				h.Memories[i].Strength++
			}
			oldContext := h.Memories[i].Context
			h.Memories[i].Output = output.Clone()
			h.Memories[i].Context = context
			h.Memories[i].Age = 0

			// Update keyword index: remove old keywords, add new.
			if oldContext != context {
				h.removeKeywordsForIndex(i, oldContext)
				h.addKeywordsForIndex(i, context)
			}
			return
		}
	}

	// Evict weakest memory if at capacity.
	if len(h.Memories) >= h.MaxMemories && h.MaxMemories > 0 {
		weakest := 0
		for i := 1; i < len(h.Memories); i++ {
			if h.Memories[i].Strength < h.Memories[weakest].Strength {
				weakest = i
			}
		}
		// Remove evicted memory's keywords.
		h.removeKeywordsForIndex(weakest, h.Memories[weakest].Context)

		// Replace the weakest entry in-place.
		h.Memories[weakest] = Memory{
			Input:    input.Clone(),
			Output:   output.Clone(),
			Strength: h.InitialStrength,
			Age:      0,
			Context:  context,
		}
		// Add new memory's keywords.
		h.addKeywordsForIndex(weakest, context)
		return
	}

	idx := len(h.Memories)
	h.Memories = append(h.Memories, Memory{
		Input:    input.Clone(),
		Output:   output.Clone(),
		Strength: h.InitialStrength,
		Age:      0,
		Context:  context,
	})
	h.addKeywordsForIndex(idx, context)
}

// ─────────────────────────────────────────────────────────────────────
// Recall — Pattern-completion retrieval
// ─────────────────────────────────────────────────────────────────────

// Recall finds the single best-matching memory whose similarity to
// the query is at or above the given threshold. Returns the memory
// and true if found, or a zero Memory and false otherwise.
//
// This models CA3 auto-associative recall: a partial cue is completed
// by the stored attractor pattern with the highest overlap.
func (h *Hippocampus) Recall(query SDR, threshold uint8) (Memory, bool) {
	mem, _, ok := h.RecallScored(query, threshold)
	return mem, ok
}

// RecallScored returns the best matching memory together with its similarity
// score. Callers that expose confidence should use this score instead of
// treating any recall above threshold as perfect certainty.
func (h *Hippocampus) RecallScored(query SDR, threshold uint8) (Memory, uint8, bool) {
	var bestSim uint8
	bestIdx := -1

	for i := range h.Memories {
		sim := h.Memories[i].Input.Similarity(query)
		if sim >= threshold && sim > bestSim {
			bestSim = sim
			bestIdx = i
		}
	}

	if bestIdx < 0 {
		return Memory{}, 0, false
	}
	return h.Memories[bestIdx], bestSim, true
}

// RecallByKeywords performs keyword-based retrieval as a complement to
// SDR similarity matching. It finds memories whose Context shares the
// most keywords with the query, and among ties picks the one with the
// highest SDR similarity to the query.
//
// threshold is the minimum number of shared keywords required for a
// match. Returns the best memory, a combined score (0-255), and
// whether a match was found.
func (h *Hippocampus) RecallByKeywords(keywords []string, threshold int, query SDR) (Memory, uint8, bool) {
	if len(keywords) == 0 || len(h.Memories) == 0 {
		return Memory{}, 0, false
	}
	if h.keywordIndex == nil {
		return Memory{}, 0, false
	}

	// Count how many query keywords each memory matches.
	hits := make(map[int]int) // memoryIndex → keyword overlap count
	for _, kw := range keywords {
		kw = strings.ToLower(kw)
		if hippoStopWords[kw] {
			continue
		}
		for _, idx := range h.keywordIndex[kw] {
			hits[idx]++
		}
	}

	bestIdx := -1
	bestHits := 0
	var bestSim uint8

	for idx, count := range hits {
		if count < threshold {
			continue
		}
		sim := h.Memories[idx].Input.Similarity(query)
		if count > bestHits || (count == bestHits && sim > bestSim) {
			bestIdx = idx
			bestHits = count
			bestSim = sim
		}
	}

	if bestIdx < 0 {
		return Memory{}, 0, false
	}

	// Compute a combined score: keyword fraction (0-200) + SDR sim (0-55).
	// This biases towards keyword overlap while still using SDR as a
	// tiebreaker.
	keywordCount := len(keywords)
	if keywordCount == 0 {
		keywordCount = 1
	}
	kwScore := (bestHits * 200) / keywordCount
	if kwScore > 200 {
		kwScore = 200
	}
	sdrPart := int(bestSim) * 55 / 255
	combined := kwScore + sdrPart
	if combined > 255 {
		combined = 255
	}

	return h.Memories[bestIdx], uint8(combined), true
}

// addKeywordsForIndex adds the keywords extracted from context to the
// keyword index, pointing at the given memory index.
func (h *Hippocampus) addKeywordsForIndex(idx int, context string) {
	for _, kw := range extractKeywords(context) {
		h.keywordIndex[kw] = append(h.keywordIndex[kw], idx)
	}
}

// removeKeywordsForIndex removes entries for the given memory index
// from the keyword index based on the old context string.
func (h *Hippocampus) removeKeywordsForIndex(idx int, context string) {
	for _, kw := range extractKeywords(context) {
		indices := h.keywordIndex[kw]
		for j := 0; j < len(indices); j++ {
			if indices[j] == idx {
				indices[j] = indices[len(indices)-1]
				indices = indices[:len(indices)-1]
				break
			}
		}
		if len(indices) == 0 {
			delete(h.keywordIndex, kw)
		} else {
			h.keywordIndex[kw] = indices
		}
	}
}

// RebuildKeywordIndex reconstructs the keyword index from all stored
// memories. Called after loading from disk or after consolidation
// (which may delete and reorder memories).
func (h *Hippocampus) RebuildKeywordIndex() {
	h.keywordIndex = make(map[string][]int, len(h.Memories))
	for i := range h.Memories {
		h.addKeywordsForIndex(i, h.Memories[i].Context)
	}
}

// ─────────────────────────────────────────────────────────────────────
// Consolidate — Sleep-like memory maintenance
// ─────────────────────────────────────────────────────────────────────

// Consolidate performs one cycle of memory maintenance, analogous to
// the memory consolidation that occurs during slow-wave sleep:
//
//  1. Age: every memory's age counter is incremented.
//  2. Strengthen: memories with strength ≥ LtpThreshold are further reinforced
//     (capped at 255), modeling LTP stabilization of strong traces.
//  3. Prune: memories with strength ≤ 1 are removed, modeling synaptic
//     homeostatic downscaling of insignificant traces.
//  4. Decay: all remaining memories lose 1 point of strength, modeling
//     the natural forgetting curve.
func (h *Hippocampus) Consolidate() {
	// Track which memories were strengthened this cycle to avoid
	// canceling LTP with decay in the same tick (was net-zero before).
	strengthened := make([]bool, len(h.Memories))

	// Phase 1 & 2: age and strengthen strong memories.
	for i := range h.Memories {
		h.Memories[i].Age++

		// Reinforce strong memories (LTP stabilization).
		if h.Memories[i].Strength >= h.LtpThreshold && h.Memories[i].Strength < 255 {
			h.Memories[i].Strength++
			strengthened[i] = true
		}
	}

	// Phase 3: prune weak memories.
	// Use fresh allocation to avoid slice aliasing (old pattern
	// h.Memories[:0] reused the backing array, leaking SDR []uint64 fields).
	alive := make([]Memory, 0, len(h.Memories))
	aliveStrengthened := make([]bool, 0, len(h.Memories))
	for i := range h.Memories {
		if h.Memories[i].Strength > 1 {
			alive = append(alive, h.Memories[i])
			aliveStrengthened = append(aliveStrengthened, strengthened[i])
		}
	}
	h.Memories = alive

	// Phase 4: decay all remaining memories (skip those just strengthened).
	for i := range h.Memories {
		if !aliveStrengthened[i] && h.Memories[i].Strength > 0 {
			h.Memories[i].Strength--
		}
	}

	// Rebuild the keyword index because pruning changed memory indices.
	h.RebuildKeywordIndex()
}

// Size returns the number of memories currently stored.
func (h *Hippocampus) Size() int {
	return len(h.Memories)
}

// ─────────────────────────────────────────────────────────────────────
// Persistence — Save / Load
// ─────────────────────────────────────────────────────────────────────
//
// Binary format (little-endian):
//   [4B  MaxMemories]
//   [4B  MemoryCount]
//   For each memory:
//     [4B  Input.Size]
//     [4B  len(Input.Bits)*8  (byte count)]
//     [... Input packed bytes]
//     [4B  Output.Size]
//     [4B  len(Output.Bits)*8 (byte count)]
//     [... Output packed bytes]
//     [1B  Strength]
//     [4B  Age]
//     [4B  len(Context)]
//     [... Context bytes]

// Save serializes the hippocampus to a binary file at the given path.
func (h *Hippocampus) Save(path string) error {
	tmpPath := path + ".tmp"
	f, err := os.Create(tmpPath)
	if err != nil {
		return fmt.Errorf("hippocampus save: %w", err)
	}

	// Header.
	if err := binary.Write(f, binary.LittleEndian, int32(h.MaxMemories)); err != nil {
		f.Close()
		os.Remove(tmpPath)
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, int32(len(h.Memories))); err != nil {
		f.Close()
		os.Remove(tmpPath)
		return err
	}

	for i := range h.Memories {
		m := &h.Memories[i]

		// Input SDR.
		if err := writeSDR(f, m.Input); err != nil {
			f.Close()
			os.Remove(tmpPath)
			return err
		}
		// Output SDR.
		if err := writeSDR(f, m.Output); err != nil {
			f.Close()
			os.Remove(tmpPath)
			return err
		}

		// Strength.
		if err := binary.Write(f, binary.LittleEndian, m.Strength); err != nil {
			f.Close()
			os.Remove(tmpPath)
			return err
		}
		// Age.
		if err := binary.Write(f, binary.LittleEndian, m.Age); err != nil {
			f.Close()
			os.Remove(tmpPath)
			return err
		}
		// Context string.
		ctx := []byte(m.Context)
		if err := binary.Write(f, binary.LittleEndian, int32(len(ctx))); err != nil {
			f.Close()
			os.Remove(tmpPath)
			return err
		}
		if _, err := f.Write(ctx); err != nil {
			f.Close()
			os.Remove(tmpPath)
			return err
		}
	}

	if err := f.Sync(); err != nil {
		f.Close()
		os.Remove(tmpPath)
		return fmt.Errorf("hippocampus sync: %w", err)
	}
	f.Close()

	if err := os.Rename(tmpPath, path); err != nil {
		return fmt.Errorf("hippocampus rename: %w", err)
	}
	return nil
}

// LoadHippocampus deserializes a hippocampus from a binary file.
func LoadHippocampus(path string) (*Hippocampus, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("hippocampus load: %w", err)
	}
	defer f.Close()

	var maxMem, count int32
	if err := binary.Read(f, binary.LittleEndian, &maxMem); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.LittleEndian, &count); err != nil {
		return nil, err
	}

	// Safety: reject absurdly large counts to prevent OOM from corrupted files.
	const MaxMemoryCount int32 = 1_000_000
	if count > MaxMemoryCount || count < 0 {
		return nil, fmt.Errorf("memory count %d exceeds safety limit %d", count, MaxMemoryCount)
	}
	if maxMem < 0 {
		return nil, fmt.Errorf("invalid maxMemories: %d", maxMem)
	}

	h := &Hippocampus{
		Memories:    make([]Memory, count),
		MaxMemories: int(maxMem),
	}

	for i := int32(0); i < count; i++ {
		m := &h.Memories[i]

		var err error
		m.Input, err = readSDR(f)
		if err != nil {
			return nil, err
		}
		m.Output, err = readSDR(f)
		if err != nil {
			return nil, err
		}

		if err := binary.Read(f, binary.LittleEndian, &m.Strength); err != nil {
			return nil, err
		}
		if err := binary.Read(f, binary.LittleEndian, &m.Age); err != nil {
			return nil, err
		}

		var ctxLen int32
		if err := binary.Read(f, binary.LittleEndian, &ctxLen); err != nil {
			return nil, err
		}
		const MaxContextLen int32 = 1_000_000
		if ctxLen < 0 || ctxLen > MaxContextLen {
			return nil, fmt.Errorf("context length %d exceeds safety limit %d", ctxLen, MaxContextLen)
		}
		ctxBuf := make([]byte, ctxLen)
		if _, err := io.ReadFull(f, ctxBuf); err != nil {
			return nil, err
		}
		m.Context = string(ctxBuf)
	}

	// Rebuild the keyword index from the loaded memories.
	h.RebuildKeywordIndex()

	return h, nil
}

// ─────────────────────────────────────────────────────────────────────
// Internal helpers for SDR binary I/O
// ─────────────────────────────────────────────────────────────────────

func writeSDR(w io.Writer, sdr SDR) error {
	packed := sdr.PackBytes()
	if err := binary.Write(w, binary.LittleEndian, int32(sdr.Size)); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, int32(len(packed))); err != nil {
		return err
	}
	_, err := w.Write(packed)
	return err
}

func readSDR(r io.Reader) (SDR, error) {
	var size, byteLen int32
	if err := binary.Read(r, binary.LittleEndian, &size); err != nil {
		return SDR{}, err
	}
	if err := binary.Read(r, binary.LittleEndian, &byteLen); err != nil {
		return SDR{}, err
	}

	// Safety: reject absurdly large or negative dimensions to prevent OOM / panic.
	const MaxSDRSize int32 = 1_000_000
	const MaxSDRByteLen int32 = 200_000
	if size > MaxSDRSize || size < 0 || byteLen > MaxSDRByteLen || byteLen < 0 {
		return SDR{}, fmt.Errorf("SDR size %d or byte length %d exceeds safety limits", size, byteLen)
	}

	data := make([]byte, byteLen)
	if _, err := io.ReadFull(r, data); err != nil {
		return SDR{}, err
	}
	return UnpackBytes(data, int(size)), nil
}
