package cortex

import (
	"encoding/json"
	"fmt"
	"os"
)

// ─────────────────────────────────────────────────────────────────────
// semantic_memory.go — Concept Generalization and Neocortical Abstraction
// ─────────────────────────────────────────────────────────────────────
//
// In biological brains, episodic memories are temporarily stored in the
// hippocampus. During sleep/consolidation, these memories are transferred
// to the neocortex. In this process, specific temporal details are decayed,
// and overlapping features are extracted to form invariant, generalized
// semantic concepts (e.g. abstract categories rather than specific events).
//
// This module implements this abstraction layer. It scans the Hippocampus
// episodic records, identifies memories with highly similar input SDRs,
// and performs bitwise intersections. The common active bits that remain
// represent the core invariant features — the generalized "Concept".
//
// Concepts are persistent and grow stronger/more generalized as the
// organism encounters more overlapping episodes.

// Concept represents an abstracted category or invariant prototype.
type Concept struct {
	Prototype   SDR      `json:"-"`        // Invariant active bits
	Count       int      `json:"count"`    // Number of episodic memories that formed this concept
	Contexts    []string `json:"contexts"` // List of unique contextual prompts/words associated with it
}

// SemanticMemory holds the collection of generalized neocortical concepts.
type SemanticMemory struct {
	Concepts     []Concept `json:"concepts"`
	SDRSize      int       `json:"sdr_size"`
	SimThreshold uint8     `json:"sim_threshold"` // Similarity threshold for concept match (default 80)
}

// NewSemanticMemory instantiates a new empty semantic memory.
// Accepts an optional Config to set the similarity threshold.
func NewSemanticMemory(sdrSize int, cfgs ...Config) *SemanticMemory {
	var simThresh uint8 = 80
	if len(cfgs) > 0 && cfgs[0].SemanticMemorySimThreshold > 0 {
		simThresh = cfgs[0].SemanticMemorySimThreshold
	}
	return &SemanticMemory{
		Concepts:     make([]Concept, 0),
		SDRSize:      sdrSize,
		SimThreshold: simThresh,
	}
}

// ─────────────────────────────────────────────────────────────────────
// Generalize — Extract semantic concepts from episodic records
// ─────────────────────────────────────────────────────────────────────

// Generalize scans the episodic memories currently stored in the hippocampus
// and abstracts concepts from highly similar patterns.
//
// 1. Matches episodes with existing concepts (similarity >= SimThreshold).
//    If matched, it intersects the existing prototype with the episode to
//    refine the invariant features.
// 2. Finds remaining unmatched episodes that are highly similar to each other
//    and creates new concepts from their intersection.
func (sm *SemanticMemory) Generalize(hip *Hippocampus) {
	if hip == nil || len(hip.Memories) == 0 {
		return
	}

	// Use configurable similarity threshold
	simThreshold := sm.SimThreshold
	if simThreshold == 0 {
		simThreshold = 80
	}

	mergedMemories := make(map[int]bool)

	// Phase 1: Try to merge episodic memories into existing concepts.
	for mIdx, m := range hip.Memories {
		bestConceptIdx := -1
		var bestSim uint8 = 0

		for cIdx, c := range sm.Concepts {
			sim := m.Input.Similarity(c.Prototype)
			if sim >= simThreshold && sim > bestSim {
				bestSim = sim
				bestConceptIdx = cIdx
			}
		}

		if bestConceptIdx != -1 {
			c := &sm.Concepts[bestConceptIdx]
			// WEIGHTED MERGE instead of pure intersection:
			// 1. Keep all existing prototype bits (don't shrink!)
			// 2. Selectively reinforce bits that are in BOTH prototype AND episode
			//    by leaving them as-is (they're already set)
			// 3. Add NEW bits from the episode that aren't in the prototype yet,
			//    but only if the concept is young (Count < 10) — this allows
			//    concepts to grow during early formation.
			//
			// OLD (broken): c.Prototype = c.Prototype.Intersect(m.Input)
			//   → This caused concepts to lose bits monotonically until they
			//     became too sparse to match anything.
			//
			// NEW: Merge new evidence into the prototype (union of episode bits
			// for young concepts to absorb novel information).
			if c.Count < 10 {
				// Young concept: absorb novel bits from episode to grow the prototype
				c.Prototype = c.Prototype.Union(m.Input)
			}
			// For mature concepts (Count >= 10), the prototype stays as-is.
			// The episode validates the concept but doesn't modify it.
			c.Count++

			// Append context if it's unique.
			foundCtx := false
			for _, ctx := range c.Contexts {
				if ctx == m.Context {
					foundCtx = true
					break
				}
			}
			if !foundCtx && m.Context != "" {
				c.Contexts = append(c.Contexts, m.Context)
			}
			mergedMemories[mIdx] = true
		}
	}

	// Phase 1b: Prune degenerate concepts whose prototypes have shrunk too much.
	const MinViableBits = 5
	alive := make([]Concept, 0, len(sm.Concepts))
	for _, c := range sm.Concepts {
		if c.Prototype.ActiveCount >= MinViableBits {
			alive = append(alive, c)
		}
	}
	sm.Concepts = alive

	// Phase 2: Form new concepts from overlapping, unmerged episodic memories.
	for i := 0; i < len(hip.Memories); i++ {
		if mergedMemories[i] {
			continue
		}
		m1 := hip.Memories[i]

		for j := i + 1; j < len(hip.Memories); j++ {
			if mergedMemories[j] {
				continue
			}
			m2 := hip.Memories[j]

			sim := m1.Input.Similarity(m2.Input)
			if sim >= simThreshold {
				// Overlapping memory traces found. Generalize via intersection!
				proto := m1.Input.Intersect(m2.Input)

				// Ensure the intersection actually yields a coherent pattern.
				if proto.ActiveCount >= 2 {
					var contexts []string
					if m1.Context != "" {
						contexts = append(contexts, m1.Context)
					}
					if m2.Context != "" && m2.Context != m1.Context {
						contexts = append(contexts, m2.Context)
					}

					sm.Concepts = append(sm.Concepts, Concept{
						Prototype: proto,
						Count:     2,
						Contexts:  contexts,
					})

					// Mark both as merged so they aren't processed repeatedly in this sleep cycle.
					mergedMemories[i] = true
					mergedMemories[j] = true
					break
				}
			}
		}
	}
}

// ─────────────────────────────────────────────────────────────────────
// Persistence — JSON Serialization (SDR packed as sparse index list)
// ─────────────────────────────────────────────────────────────────────

type conceptJSON struct {
	ActiveIndices []int    `json:"active_indices"`
	Count         int      `json:"count"`
	Contexts      []string `json:"contexts"`
}

type semanticMemoryJSON struct {
	Concepts     []conceptJSON `json:"concepts"`
	SDRSize      int           `json:"sdr_size"`
	SimThreshold uint8         `json:"sim_threshold,omitempty"` // Persisted since v2
}

// Save serializes the semantic memory concepts to a JSON file.
// Prototype SDRs are stored as space-efficient active indices arrays.
func (sm *SemanticMemory) Save(path string) error {
	aux := semanticMemoryJSON{
		Concepts:     make([]conceptJSON, len(sm.Concepts)),
		SDRSize:      sm.SDRSize,
		SimThreshold: sm.SimThreshold,
	}

	for i, c := range sm.Concepts {
		aux.Concepts[i] = conceptJSON{
			ActiveIndices: c.Prototype.ActiveIndices(),
			Count:         c.Count,
			Contexts:      c.Contexts,
		}
	}

	data, err := json.MarshalIndent(aux, "", "  ")
	if err != nil {
		return fmt.Errorf("semantic memory marshal: %w", err)
	}

	if err := os.WriteFile(path, data, 0600); err != nil {
		return fmt.Errorf("semantic memory write: %w", err)
	}

	return nil
}

// LoadSemanticMemory restores a semantic memory instance from a JSON file.
// Accepts optional Config to restore SimThreshold from configuration
// (matching the NewSemanticMemory pattern).
func LoadSemanticMemory(path string, sdrSize int, cfgs ...Config) (*SemanticMemory, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("semantic memory read: %w", err)
	}

	var aux semanticMemoryJSON
	if err := json.Unmarshal(data, &aux); err != nil {
		return nil, fmt.Errorf("semantic memory unmarshal: %w", err)
	}

	// Validate limits to prevent OOM from corrupted JSON
	const maxSemanticSDRSize = 100_000
	const maxSemanticConcepts = 100_000
	const maxConceptActiveIndices = 10_000
	const maxContextLen = 10_000

	if aux.SDRSize < 0 || aux.SDRSize > maxSemanticSDRSize {
		return nil, fmt.Errorf("semantic memory invalid SDRSize: %d", aux.SDRSize)
	}
	if len(aux.Concepts) > maxSemanticConcepts {
		return nil, fmt.Errorf("semantic memory too many concepts: %d (max %d)", len(aux.Concepts), maxSemanticConcepts)
	}

	// Restore SimThreshold: prefer persisted value, fall back to Config, then default 80.
	simThresh := aux.SimThreshold
	if simThresh == 0 && len(cfgs) > 0 && cfgs[0].SemanticMemorySimThreshold > 0 {
		simThresh = cfgs[0].SemanticMemorySimThreshold
	}

	sm := &SemanticMemory{
		Concepts:     make([]Concept, len(aux.Concepts)),
		SDRSize:      aux.SDRSize,
		SimThreshold: simThresh,
	}

	if sm.SDRSize <= 0 {
		sm.SDRSize = sdrSize
	}

	for i, cJSON := range aux.Concepts {
		if len(cJSON.ActiveIndices) > maxConceptActiveIndices {
			return nil, fmt.Errorf("semantic memory concept %d has too many active indices: %d", i, len(cJSON.ActiveIndices))
		}
		if len(cJSON.Contexts) > maxContextLen {
			return nil, fmt.Errorf("semantic memory concept %d has too many contexts: %d", i, len(cJSON.Contexts))
		}
		sdr := NewSDR(sm.SDRSize)
		for _, idx := range cJSON.ActiveIndices {
			sdr.Set(idx)
		}
		sm.Concepts[i] = Concept{
			Prototype: sdr,
			Count:     cJSON.Count,
			Contexts:  cJSON.Contexts,
		}
	}

	return sm, nil
}
