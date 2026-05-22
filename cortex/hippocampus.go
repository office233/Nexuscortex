package cortex

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
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
}

// NewHippocampus creates a hippocampus with the given config.
func NewHippocampus(cfg Config) *Hippocampus {
	return &Hippocampus{
		Memories:              make([]Memory, 0, cfg.MaxMemories),
		MaxMemories:           cfg.MaxMemories,
		ReconsolidationThresh: cfg.HippoReconsolidationThresh,
		InitialStrength:       cfg.HippoInitialStrength,
		LtpThreshold:          cfg.HippoLtpThreshold,
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
	// Check for an existing similar memory to reconsolidate.
	for i := range h.Memories {
		sim := h.Memories[i].Input.Similarity(input)
		if sim >= h.ReconsolidationThresh {
			// Reconsolidate: strengthen and update.
			if h.Memories[i].Strength < 255 {
				h.Memories[i].Strength++
			}
			h.Memories[i].Output = output.Clone()
			h.Memories[i].Context = context
			h.Memories[i].Age = 0
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
		// Replace the weakest entry in-place.
		h.Memories[weakest] = Memory{
			Input:    input.Clone(),
			Output:   output.Clone(),
			Strength: h.InitialStrength,
			Age:      0,
			Context:  context,
		}
		return
	}

	h.Memories = append(h.Memories, Memory{
		Input:    input.Clone(),
		Output:   output.Clone(),
		Strength: h.InitialStrength,
		Age:      0,
		Context:  context,
	})
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
