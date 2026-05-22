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
	Concepts []Concept `json:"concepts"`
	SDRSize  int       `json:"sdr_size"`
}

// NewSemanticMemory instantiates a new empty semantic memory.
func NewSemanticMemory(sdrSize int) *SemanticMemory {
	return &SemanticMemory{
		Concepts: make([]Concept, 0),
		SDRSize:  sdrSize,
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

	// Similarity threshold (80 out of 255 represents approx 31% similarity)
	const SimThreshold uint8 = 80

	mergedMemories := make(map[int]bool)

	// Phase 1: Try to merge episodic memories into existing concepts.
	for mIdx, m := range hip.Memories {
		bestConceptIdx := -1
		var bestSim uint8 = 0

		for cIdx, c := range sm.Concepts {
			sim := m.Input.Similarity(c.Prototype)
			if sim >= SimThreshold && sim > bestSim {
				bestSim = sim
				bestConceptIdx = cIdx
			}
		}

		if bestConceptIdx != -1 {
			c := &sm.Concepts[bestConceptIdx]
			// Retain only the overlapping invariant bits (LTP pattern separation).
			c.Prototype = c.Prototype.Intersect(m.Input)
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
			if sim >= SimThreshold {
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
	Concepts []conceptJSON `json:"concepts"`
	SDRSize  int           `json:"sdr_size"`
}

// Save serializes the semantic memory concepts to a JSON file.
// Prototype SDRs are stored as space-efficient active indices arrays.
func (sm *SemanticMemory) Save(path string) error {
	aux := semanticMemoryJSON{
		Concepts: make([]conceptJSON, len(sm.Concepts)),
		SDRSize:  sm.SDRSize,
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

	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("semantic memory write: %w", err)
	}

	return nil
}

// LoadSemanticMemory restores a semantic memory instance from a JSON file.
func LoadSemanticMemory(path string, sdrSize int) (*SemanticMemory, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("semantic memory read: %w", err)
	}

	var aux semanticMemoryJSON
	if err := json.Unmarshal(data, &aux); err != nil {
		return nil, fmt.Errorf("semantic memory unmarshal: %w", err)
	}

	sm := &SemanticMemory{
		Concepts: make([]Concept, len(aux.Concepts)),
		SDRSize:  aux.SDRSize,
	}

	if sm.SDRSize <= 0 {
		sm.SDRSize = sdrSize
	}

	for i, cJSON := range aux.Concepts {
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
