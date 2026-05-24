package cortex

import (
	"encoding/json"
	"fmt"
	"hash/fnv"
	"os"
	"sort"
	"sync"
)

// ─────────────────────────────────────────────────────────────────────
// SequenceMemory — Multi-Context Window Sequence Prediction
// ─────────────────────────────────────────────────────────────────────
//
// Unlike the Brain's simple bigram (word[i] → word[i+1]) synapses,
// SequenceMemory stores transitions over multiple context windows
// (1, 2, 4, 8, 16 tokens). Longer context matches produce higher-
// confidence predictions, enabling the system to disambiguate
// sequences that share short prefixes but differ at longer ranges.
//
// All math is integer-only (uint16/uint32/uint64). No float64.

// WeightedTransition represents a single target word with its
// accumulated weight and observation count.
type WeightedTransition struct {
	TargetID uint32 `json:"target_id"`
	Weight   uint16 `json:"weight"`
	Count    uint16 `json:"count"`
}

// ContextWindow stores transitions for a specific context length.
type ContextWindow struct {
	Size        int                              `json:"size"`
	Transitions map[uint64][]WeightedTransition  `json:"transitions"`
}

// SequenceMemory stores transitions over multiple context windows
// for richer next-word prediction.
type SequenceMemory struct {
	Windows         map[int]*ContextWindow `json:"windows"`
	MaxTargets      int                    `json:"max_targets"`
	IncrementWeight uint16                 `json:"increment_weight"`
	LTPAmount       uint16                 `json:"ltp_amount"`  // LTP reinforcement delta (default 50)
	LTDAmount       uint16                 `json:"ltd_amount"`  // LTD weakening delta (default 40)
	mu              sync.RWMutex
}

// Standard window sizes for multi-context prediction.
var defaultWindowSizes = []int{1, 2, 4, 8, 16}

// NewSequenceMemory creates a new sequence memory with windows
// for context sizes [1, 2, 4, 8, 16] using the given config.
func NewSequenceMemory(cfg Config) *SequenceMemory {
	maxTargets := cfg.SequenceMemoryMaxTargets
	if maxTargets <= 0 {
		maxTargets = 64
	}
	windows := make(map[int]*ContextWindow)
	for _, size := range defaultWindowSizes {
		windows[size] = &ContextWindow{
			Size:        size,
			Transitions: make(map[uint64][]WeightedTransition),
		}
	}
	ltpAmount := cfg.SequenceMemoryLTPAmount
	if ltpAmount == 0 {
		ltpAmount = 50
	}
	ltdAmount := cfg.SequenceMemoryLTDAmount
	if ltdAmount == 0 {
		ltdAmount = 40
	}
	incWeight := cfg.SequenceMemoryIncrementWeight
	if incWeight == 0 {
		incWeight = 10
	}
	return &SequenceMemory{
		Windows:         windows,
		MaxTargets:      maxTargets,
		IncrementWeight: incWeight,
		LTPAmount:       ltpAmount,
		LTDAmount:       ltdAmount,
	}
}

// contextHash computes an FNV-1a hash of a word ID sequence.
// This maps a context window (e.g., the last 4 word IDs) to a
// single uint64 key for fast lookup.
func contextHash(ids []uint32) uint64 {
	h := fnv.New64a()
	for _, id := range ids {
		h.Write([]byte{byte(id >> 24), byte(id >> 16), byte(id >> 8), byte(id)})
	}
	return h.Sum64()
}

// Learn records transitions at all window sizes from a sequence
// of word IDs. For each position in the sequence and each window
// size, it hashes the preceding N words and records the next word
// as a transition target.
func (sm *SequenceMemory) Learn(wordIDs []uint32) {
	if len(wordIDs) < 2 {
		return
	}

	sm.mu.Lock()
	defer sm.mu.Unlock()

	for _, winSize := range defaultWindowSizes {
		cw := sm.Windows[winSize]
		if cw == nil {
			continue
		}

		// For each position where we have enough context.
		for pos := winSize; pos < len(wordIDs); pos++ {
			// Context = the preceding winSize words.
			context := wordIDs[pos-winSize : pos]
			target := wordIDs[pos]
			hash := contextHash(context)

			sm.addTransition(cw, hash, target)
		}
	}
}

// addTransition adds or strengthens a transition in the given window.
func (sm *SequenceMemory) addTransition(cw *ContextWindow, hash uint64, targetID uint32) {
	transitions := cw.Transitions[hash]
	incWeight := sm.IncrementWeight

	// Look for existing transition.
	for i := range transitions {
		if transitions[i].TargetID == targetID {
			// Strengthen existing.
			if transitions[i].Weight < 65535 {
				if 65535-transitions[i].Weight < incWeight {
					transitions[i].Weight = 65535
				} else {
					transitions[i].Weight += incWeight
				}
			}
			if transitions[i].Count < 65535 {
				transitions[i].Count++
			}
			cw.Transitions[hash] = transitions
			return
		}
	}

	// Add new transition (prune if at capacity).
	if len(transitions) >= sm.MaxTargets {
		// Remove the weakest transition.
		minIdx := 0
		minWeight := transitions[0].Weight
		for i := 1; i < len(transitions); i++ {
			if transitions[i].Weight < minWeight {
				minWeight = transitions[i].Weight
				minIdx = i
			}
		}
		transitions[minIdx] = WeightedTransition{
			TargetID: targetID,
			Weight:   incWeight,
			Count:    1,
		}
	} else {
		transitions = append(transitions, WeightedTransition{
			TargetID: targetID,
			Weight:   incWeight,
			Count:    1,
		})
	}
	cw.Transitions[hash] = transitions
}

// Predict finds the most likely next word given a context of recent
// word IDs. It checks all window sizes and weights longer context
// matches more heavily (windowSize × weight).
//
// Returns the predicted word ID and true, or 0 and false if no
// prediction can be made.
func (sm *SequenceMemory) Predict(context []uint32, recentlyUsed map[uint32]int) (uint32, bool) {
	if len(context) == 0 {
		return 0, false
	}

	sm.mu.RLock()
	defer sm.mu.RUnlock()

	// Aggregate scores across all matching windows.
	scores := make(map[uint32]uint32)

	for _, winSize := range defaultWindowSizes {
		if winSize > len(context) {
			continue
		}
		cw := sm.Windows[winSize]
		if cw == nil {
			continue
		}

		// Extract the relevant context slice.
		ctxSlice := context[len(context)-winSize:]
		hash := contextHash(ctxSlice)

		transitions := cw.Transitions[hash]
		for _, t := range transitions {
			// Longer context = higher multiplier.
			// Window 1 → ×1, Window 2 → ×2, Window 4 → ×4, etc.
			score := uint32(t.Weight) * uint32(winSize)

			// Apply anti-repetition penalty.
			if count, ok := recentlyUsed[t.TargetID]; ok && count > 0 {
				score /= uint32(count + 1)
			}

			scores[t.TargetID] += score
		}
	}

	if len(scores) == 0 {
		return 0, false
	}

	// Find the candidate with the highest aggregate score.
	type candidate struct {
		id    uint32
		score uint32
	}
	candidates := make([]candidate, 0, len(scores))
	for id, score := range scores {
		if score > 0 {
			candidates = append(candidates, candidate{id, score})
		}
	}

	if len(candidates) == 0 {
		return 0, false
	}

	// Sort by score descending, with deterministic tie-breaking by id.
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].score != candidates[j].score {
			return candidates[i].score > candidates[j].score
		}
		return candidates[i].id < candidates[j].id
	})

	return candidates[0].id, true
}

// Stats returns basic statistics about the sequence memory.
type SequenceMemoryStats struct {
	WindowSizes     []int `json:"window_sizes"`
	TotalTransitions int  `json:"total_transitions"`
	TotalContexts   int  `json:"total_contexts"`
}

// Stats returns statistics about the sequence memory.
func (sm *SequenceMemory) Stats() SequenceMemoryStats {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	stats := SequenceMemoryStats{}
	for _, winSize := range defaultWindowSizes {
		cw := sm.Windows[winSize]
		if cw == nil {
			continue
		}
		stats.WindowSizes = append(stats.WindowSizes, winSize)
		stats.TotalContexts += len(cw.Transitions)
		for _, targets := range cw.Transitions {
			stats.TotalTransitions += len(targets)
		}
	}
	return stats
}

// ─────────────────────────────────────────────────────────────────────
// Persistence
// ─────────────────────────────────────────────────────────────────────

// Save writes the sequence memory to a JSON file using atomic save
// (temp file + sync + rename) to prevent corruption on crash.
func (sm *SequenceMemory) Save(path string) error {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	data, err := json.MarshalIndent(sm, "", "  ")
	if err != nil {
		return fmt.Errorf("sequence memory marshal: %w", err)
	}

	tmpPath := path + ".tmp"
	f, err := os.OpenFile(tmpPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0600)
	if err != nil {
		return fmt.Errorf("sequence memory create tmp: %w", err)
	}
	if _, err := f.Write(data); err != nil {
		f.Close()
		os.Remove(tmpPath)
		return fmt.Errorf("sequence memory write: %w", err)
	}
	if err := f.Sync(); err != nil {
		f.Close()
		os.Remove(tmpPath)
		return fmt.Errorf("sequence memory sync: %w", err)
	}
	if err := f.Close(); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("sequence memory close: %w", err)
	}
	return os.Rename(tmpPath, path)
}

// LoadSequenceMemory reads a sequence memory from a JSON file using the given config.
// Returns nil, nil if the file does not exist (graceful fallback).
func LoadSequenceMemory(path string, cfg Config) (*SequenceMemory, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("sequence memory read: %w", err)
	}

	sm := &SequenceMemory{}
	if err := json.Unmarshal(data, sm); err != nil {
		return nil, fmt.Errorf("sequence memory unmarshal: %w", err)
	}

	// Ensure all maps are initialized (JSON null → nil map).
	if sm.Windows == nil {
		sm.Windows = make(map[int]*ContextWindow)
	}
	for _, winSize := range defaultWindowSizes {
		if sm.Windows[winSize] == nil {
			sm.Windows[winSize] = &ContextWindow{
				Size:        winSize,
				Transitions: make(map[uint64][]WeightedTransition),
			}
		}
		if sm.Windows[winSize].Transitions == nil {
			sm.Windows[winSize].Transitions = make(map[uint64][]WeightedTransition)
		}
	}
	maxTargets := cfg.SequenceMemoryMaxTargets
	if maxTargets <= 0 {
		maxTargets = 64
	}
	sm.MaxTargets = maxTargets
	incWeight := cfg.SequenceMemoryIncrementWeight
	if incWeight == 0 {
		incWeight = 10
	}
	sm.IncrementWeight = incWeight
	ltpAmount := cfg.SequenceMemoryLTPAmount
	if ltpAmount == 0 {
		ltpAmount = 50
	}
	ltdAmount := cfg.SequenceMemoryLTDAmount
	if ltdAmount == 0 {
		ltdAmount = 40
	}
	sm.LTPAmount = ltpAmount
	sm.LTDAmount = ltdAmount

	return sm, nil
}

// ReinforceSequence strengthens or weakens a sequence of word IDs across all context windows.
func (sm *SequenceMemory) ReinforceSequence(wordIDs []uint32, positive bool) {
	if len(wordIDs) < 2 {
		return
	}

	sm.mu.Lock()
	defer sm.mu.Unlock()

	for _, winSize := range defaultWindowSizes {
		cw := sm.Windows[winSize]
		if cw == nil {
			continue
		}

		for pos := winSize; pos < len(wordIDs); pos++ {
			context := wordIDs[pos-winSize : pos]
			target := wordIDs[pos]
			hash := contextHash(context)

			transitions := cw.Transitions[hash]
			found := false
			for i := range transitions {
				if transitions[i].TargetID == target {
					found = true
					if positive {
						// Strengthen (LTP)
						newW := uint32(transitions[i].Weight) + uint32(sm.LTPAmount)
						if newW > 65535 {
							newW = 65535
						}
						transitions[i].Weight = uint16(newW)
					} else {
						// Weaken (LTD)
						if transitions[i].Weight > sm.LTDAmount {
							transitions[i].Weight -= sm.LTDAmount
						} else {
							transitions[i].Weight = 0
						}
					}
					break
				}
			}

			// Clean up pruned transitions (weight == 0)
			if !positive {
				alive := make([]WeightedTransition, 0, len(transitions))
				for _, t := range transitions {
					if t.Weight > 0 {
						alive = append(alive, t)
					}
				}
				cw.Transitions[hash] = alive
			} else if !found {
				// If positive and not found, add it if we have space
				if len(transitions) < sm.MaxTargets {
					cw.Transitions[hash] = append(transitions, WeightedTransition{
						TargetID: target,
						Weight:   sm.LTPAmount,
						Count:    1,
					})
				}
			}
		}
	}
}

