package cortex

// ─────────────────────────────────────────────────────────────────────
// working_memory.go — Short-Term Working Memory (Cognitive Scratchpad)
// ─────────────────────────────────────────────────────────────────────
//
// Biological working memory (prefrontal cortex + parietal regions)
// maintains 4–7 "chunks" of information in active neural firing.
// This module provides a fixed-capacity scratchpad that persists
// across Process() calls, giving the organism:
//
//   - CONTEXT: remembering what was just said/asked
//   - BINDING: linking current input to recent information
//   - REASONING: multi-step problem solving by keeping intermediates
//
// Architecture:
//   - Fixed array of WMSlot (default 8 slots)
//   - Each slot holds: SDR pattern, text label, relevance, age
//   - Store: insert with LRU eviction (oldest/weakest slot replaced)
//   - Recall: find best matching slot by SDR similarity
//   - Decay: all slots age each tick; very old ones auto-clear
//   - Blend: union of all active slots → context SDR for reasoning
//
// Integer-only, zero-gradient, biologically plausible.
// ─────────────────────────────────────────────────────────────────────

// WMSlot represents a single working memory slot.
type WMSlot struct {
	Pattern   SDR    // Neural activation pattern
	Text      string // Human-readable label
	Relevance uint8  // How relevant this slot is (0 = empty, 255 = critical)
	Age       uint16 // How many ticks since this was stored/refreshed
	Active    bool   // Whether this slot is occupied
}

// WorkingMemory is a fixed-capacity short-term memory buffer.
type WorkingMemory struct {
	Slots         []WMSlot
	Capacity      int   // Max number of slots (from Config)
	DecayRate     uint8 // How fast relevance decays per tick
	MinRelevance  uint8 // Below this, slot is auto-cleared
	TotalStores   uint64
	TotalRecalls  uint64
	TotalEvicts   uint64
}

// WorkingMemoryStats exposes metrics for monitoring.
type WorkingMemoryStats struct {
	Capacity     int
	ActiveSlots  int
	AvgRelevance uint8
	OldestAge    uint16
	TotalStores  uint64
	TotalRecalls uint64
	TotalEvicts  uint64
}

// NewWorkingMemory creates a working memory with Config parameters.
func NewWorkingMemory(cfg Config) *WorkingMemory {
	cap := cfg.WorkingMemoryCapacity
	if cap <= 0 {
		cap = 8 // Default: 8 slots (biological working memory ~7±2)
	}
	decayRate := cfg.WorkingMemoryDecayRate
	if decayRate == 0 {
		decayRate = 3 // Default: lose 3 relevance per tick
	}
	minRel := cfg.WorkingMemoryMinRelevance
	if minRel == 0 {
		minRel = 10 // Default: clear slots below 10 relevance
	}

	return &WorkingMemory{
		Slots:        make([]WMSlot, cap),
		Capacity:     cap,
		DecayRate:    decayRate,
		MinRelevance: minRel,
	}
}

// ─────────────────────────────────────────────────────────────────────
// Store — Write a pattern into working memory
// ─────────────────────────────────────────────────────────────────────

// Store places an SDR into working memory with the given relevance.
// If all slots are occupied, the least-relevant (or oldest if tied)
// slot is evicted.
//
// If a similar pattern already exists (similarity > 200), the existing
// slot is REFRESHED instead of creating a duplicate: its relevance is
// boosted and age is reset.
func (wm *WorkingMemory) Store(pattern SDR, text string, relevance uint8) {
	if pattern.ActiveCount == 0 {
		return // Don't store empty patterns
	}

	// Check for existing similar pattern → refresh instead of duplicate.
	for i := range wm.Slots {
		if wm.Slots[i].Active && wm.Slots[i].Pattern.Similarity(pattern) > 200 {
			// Refresh: boost relevance, reset age, update text.
			newRel := uint16(wm.Slots[i].Relevance) + uint16(relevance)/2
			if newRel > 255 {
				newRel = 255
			}
			wm.Slots[i].Relevance = uint8(newRel)
			wm.Slots[i].Age = 0
			wm.Slots[i].Text = text
			wm.Slots[i].Pattern = pattern.Clone()
			wm.TotalStores++
			return
		}
	}

	// Find an empty slot first.
	for i := range wm.Slots {
		if !wm.Slots[i].Active {
			wm.Slots[i] = WMSlot{
				Pattern:   pattern.Clone(),
				Text:      text,
				Relevance: relevance,
				Age:       0,
				Active:    true,
			}
			wm.TotalStores++
			return
		}
	}

	// All slots full — evict the weakest.
	evictIdx := 0
	evictScore := wm.evictionScore(0)
	for i := 1; i < len(wm.Slots); i++ {
		score := wm.evictionScore(i)
		if score < evictScore {
			evictScore = score
			evictIdx = i
		}
	}

	wm.Slots[evictIdx] = WMSlot{
		Pattern:   pattern.Clone(),
		Text:      text,
		Relevance: relevance,
		Age:       0,
		Active:    true,
	}
	wm.TotalStores++
	wm.TotalEvicts++
}

// evictionScore returns a combined score for eviction priority.
// Lower score = more likely to be evicted.
// Score = relevance × 256 - age (relevance dominates, age breaks ties).
func (wm *WorkingMemory) evictionScore(idx int) int {
	s := &wm.Slots[idx]
	return int(s.Relevance)*256 - int(s.Age)
}

// ─────────────────────────────────────────────────────────────────────
// Recall — Pattern-completion retrieval from working memory
// ─────────────────────────────────────────────────────────────────────

// Recall finds the best matching slot by SDR similarity.
// Returns the slot's text and similarity score.
// Returns ("", 0) if no active slots exist or no match exceeds
// the minimum threshold (similarity > 50).
func (wm *WorkingMemory) Recall(query SDR) (string, uint8) {
	if query.ActiveCount == 0 {
		return "", 0
	}

	bestIdx := -1
	var bestSim uint8

	for i := range wm.Slots {
		if !wm.Slots[i].Active {
			continue
		}
		sim := wm.Slots[i].Pattern.Similarity(query)
		if sim > bestSim {
			bestSim = sim
			bestIdx = i
		}
	}

	if bestIdx < 0 || bestSim < 50 {
		return "", 0
	}

	// Boost relevance of recalled slot (retrieval strengthens memory).
	if wm.Slots[bestIdx].Relevance < 245 {
		wm.Slots[bestIdx].Relevance += 10
	} else {
		wm.Slots[bestIdx].Relevance = 255
	}
	wm.TotalRecalls++

	return wm.Slots[bestIdx].Text, bestSim
}

// ─────────────────────────────────────────────────────────────────────
// Tick — Age and decay all slots
// ─────────────────────────────────────────────────────────────────────

// Tick ages all active slots and decays their relevance.
// Slots that drop below MinRelevance are automatically cleared.
func (wm *WorkingMemory) Tick() {
	for i := range wm.Slots {
		if !wm.Slots[i].Active {
			continue
		}

		// Age the slot.
		if wm.Slots[i].Age < 65535 {
			wm.Slots[i].Age++
		}

		// Decay relevance.
		if wm.Slots[i].Relevance > wm.DecayRate {
			wm.Slots[i].Relevance -= wm.DecayRate
		} else {
			wm.Slots[i].Relevance = 0
		}

		// Auto-clear very weak slots (fading from consciousness).
		if wm.Slots[i].Relevance < wm.MinRelevance {
			wm.Slots[i] = WMSlot{} // Zero out entirely
		}
	}
}

// ─────────────────────────────────────────────────────────────────────
// Blend — Get combined context from all active slots
// ─────────────────────────────────────────────────────────────────────

// BlendContext returns the union of all active slot patterns,
// weighted by relevance. This gives a "context cloud" that represents
// everything currently in working memory — ideal for feeding into
// the Prefrontal reasoning network as background context.
func (wm *WorkingMemory) BlendContext(sdrSize int) SDR {
	context := NewSDR(sdrSize)

	for i := range wm.Slots {
		if !wm.Slots[i].Active || wm.Slots[i].Pattern.Size == 0 {
			continue
		}
		// Only blend slots with meaningful relevance.
		if wm.Slots[i].Relevance > 30 {
			context = context.Union(wm.Slots[i].Pattern)
		}
	}

	return context
}

// ─────────────────────────────────────────────────────────────────────
// ActiveTexts — Human-readable dump of working memory contents
// ─────────────────────────────────────────────────────────────────────

// ActiveTexts returns the text labels of all active slots, ordered
// by relevance (highest first). Useful for debugging and for the
// Broca module to incorporate recent context into generation.
func (wm *WorkingMemory) ActiveTexts() []string {
	type ranked struct {
		text string
		rel  uint8
	}

	var items []ranked
	for i := range wm.Slots {
		if wm.Slots[i].Active && wm.Slots[i].Text != "" {
			items = append(items, ranked{wm.Slots[i].Text, wm.Slots[i].Relevance})
		}
	}

	// Simple insertion sort (max 8 items).
	for i := 1; i < len(items); i++ {
		for j := i; j > 0 && items[j].rel > items[j-1].rel; j-- {
			items[j], items[j-1] = items[j-1], items[j]
		}
	}

	result := make([]string, len(items))
	for i, item := range items {
		result[i] = item.text
	}
	return result
}

// ActiveCount returns the number of occupied slots.
func (wm *WorkingMemory) ActiveCount() int {
	count := 0
	for i := range wm.Slots {
		if wm.Slots[i].Active {
			count++
		}
	}
	return count
}

// Clear resets all slots to empty.
func (wm *WorkingMemory) Clear() {
	for i := range wm.Slots {
		wm.Slots[i] = WMSlot{}
	}
}

// Stats returns metrics about working memory state.
func (wm *WorkingMemory) Stats() WorkingMemoryStats {
	active := 0
	var sumRel uint16
	var maxAge uint16
	for i := range wm.Slots {
		if wm.Slots[i].Active {
			active++
			sumRel += uint16(wm.Slots[i].Relevance)
			if wm.Slots[i].Age > maxAge {
				maxAge = wm.Slots[i].Age
			}
		}
	}
	var avgRel uint8
	if active > 0 {
		avgRel = uint8(sumRel / uint16(active))
	}
	return WorkingMemoryStats{
		Capacity:     wm.Capacity,
		ActiveSlots:  active,
		AvgRelevance: avgRel,
		OldestAge:    maxAge,
		TotalStores:  wm.TotalStores,
		TotalRecalls: wm.TotalRecalls,
		TotalEvicts:  wm.TotalEvicts,
	}
}
