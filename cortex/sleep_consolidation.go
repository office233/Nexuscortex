package cortex

import "fmt"

// ─────────────────────────────────────────────────────────────────────
// sleep_consolidation.go — Sleep-Dependent Memory Consolidation
// ─────────────────────────────────────────────────────────────────────
//
// During slow-wave sleep the brain replays recent hippocampal memories
// through neocortical circuits, strengthening important patterns and
// weakening noisy ones. This module implements that process:
//
//   1. Gather recent (new) and older (sampled) memories from the
//      Hippocampus.
//   2. Interleave them at a configurable ratio so that old memories
//      are rehearsed alongside new ones — just as the brain does
//      during memory consolidation.
//   3. Replay each memory through the Prefrontal spiking reservoir
//      to measure its stability (confidence).
//   4. If the replayed pattern is stable (confidence ≥ threshold),
//      reinforce the word-level synapses via Brain.ReinforceSequence.
//   5. If unstable, weaken the synapses.
//   6. Record success/failure in the SelfModel for metacognitive
//      awareness.
//
// All arithmetic is integer-only; no float64 is used.

// ConsolidationReport summarises what happened during one sleep
// consolidation cycle.
type ConsolidationReport struct {
	Replayed     int      // Total memories replayed
	Strengthened int      // Memories whose synapses were reinforced
	Weakened     int      // Memories whose synapses were weakened
	Logs         []string // Human-readable log lines
}

// SleepConsolidator orchestrates the replay of hippocampal memories
// through the prefrontal reservoir during sleep.
type SleepConsolidator struct {
	Cfg             Config
	ReplayCount     int // How many memories to replay per sleep cycle
	InterleaveRatio int // Old memories per new memory (e.g. 2 = 2 old per 1 new)
}

// NewSleepConsolidator creates a consolidator with defaults drawn from
// the given config.
func NewSleepConsolidator(cfg Config) *SleepConsolidator {
	replayCount := cfg.SleepReplayCount
	if replayCount <= 0 {
		replayCount = 10
	}
	interleaveRatio := cfg.SleepInterleaveRatio
	if interleaveRatio <= 0 {
		interleaveRatio = 2
	}
	return &SleepConsolidator{
		Cfg:             cfg,
		ReplayCount:     replayCount,
		InterleaveRatio: interleaveRatio,
	}
}

// Consolidate replays hippocampal memories through the prefrontal
// reservoir. Stable patterns are reinforced in the Brain; unstable
// ones are weakened. Returns a report of actions taken.
func (sc *SleepConsolidator) Consolidate(
	hippo *Hippocampus,
	prefrontal *Prefrontal,
	brain *Brain,
	broca *Broca,
	self *SelfModel,
	cfg Config,
) ConsolidationReport {
	report := ConsolidationReport{
		Logs: []string{"Sleep consolidation: starting memory replay..."},
	}

	if hippo == nil || prefrontal == nil || brain == nil || broca == nil {
		return report
	}

	nMem := len(hippo.Memories)
	if nMem == 0 {
		report.Logs = append(report.Logs, "Sleep consolidation: no memories to replay.")
		return report
	}

	// ── Partition memories into "recent" and "old". ─────────────
	// Recent = last ReplayCount memories (or fewer if not enough).
	// Old    = everything before that.
	recentStart := nMem - sc.ReplayCount
	if recentStart < 0 {
		recentStart = 0
	}

	recentMems := make([]Memory, 0, nMem-recentStart)
	for i := recentStart; i < nMem; i++ {
		recentMems = append(recentMems, hippo.Memories[i])
	}

	oldMems := make([]Memory, 0, recentStart)
	for i := 0; i < recentStart; i++ {
		oldMems = append(oldMems, hippo.Memories[i])
	}

	// ── Build interleaved replay playlist. ──────────────────────
	playlist := interleaveMemories(recentMems, oldMems, sc.InterleaveRatio)

	report.Logs = append(report.Logs, fmt.Sprintf(
		"Sleep consolidation: %d recent, %d old, playlist length %d.",
		len(recentMems), len(oldMems), len(playlist)))

	// ── Stability threshold from config. ────────────────────────
	stabilityThresh := cfg.SleepStabilityThresh
	if stabilityThresh == 0 {
		stabilityThresh = 180
	}

	// ── Replay each memory. ─────────────────────────────────────
	for _, mem := range playlist {
		report.Replayed++

		// Replay the input pattern through the prefrontal reservoir.
		_ = prefrontal.Think(mem.Input.Clone(), cfg.ThinkCycles)
		confidence := prefrontal.GetConfidence()

		// Generate a text representation for ReinforceSequence.
		phrase := broca.Generate(mem.Output.Clone(), cfg.MaxGenWords)
		if phrase == "" || phrase == NoConfidentResponse {
			report.Logs = append(report.Logs, fmt.Sprintf(
				"  Replay #%d: skipped (no decodable phrase, confidence=%d).",
				report.Replayed, confidence))
			continue
		}

		topic := mem.Context

		if confidence >= stabilityThresh {
			// Stable attractor — strengthen.
			brain.ReinforceSequence(phrase, true)
			if self != nil {
				self.RecordSuccess(topic, confidence)
			}
			report.Strengthened++
			report.Logs = append(report.Logs, fmt.Sprintf(
				"  Replay #%d: STABLE (confidence=%d ≥ %d). Strengthened '%s'.",
				report.Replayed, confidence, stabilityThresh, truncate(phrase, 40)))
		} else {
			// Unstable — weaken.
			brain.ReinforceSequence(phrase, false)
			if self != nil {
				self.RecordFailure(topic)
			}
			report.Weakened++
			report.Logs = append(report.Logs, fmt.Sprintf(
				"  Replay #%d: UNSTABLE (confidence=%d < %d). Weakened '%s'.",
				report.Replayed, confidence, stabilityThresh, truncate(phrase, 40)))
		}
	}

	report.Logs = append(report.Logs, fmt.Sprintf(
		"Sleep consolidation: done. Replayed=%d, Strengthened=%d, Weakened=%d.",
		report.Replayed, report.Strengthened, report.Weakened))

	return report
}

// interleaveMemories builds a playlist of [new, old, old, new, old, old, ...]
// based on the interleave ratio. If there are not enough old memories,
// extra new memories fill the gaps and vice-versa.
func interleaveMemories(recent, old []Memory, ratio int) []Memory {
	if ratio <= 0 {
		ratio = 1
	}

	totalCapacity := len(recent) + len(old)
	playlist := make([]Memory, 0, totalCapacity)

	ri, oi := 0, 0
	for ri < len(recent) || oi < len(old) {
		// One new memory.
		if ri < len(recent) {
			playlist = append(playlist, recent[ri])
			ri++
		}
		// 'ratio' old memories.
		for k := 0; k < ratio && oi < len(old); k++ {
			playlist = append(playlist, old[oi])
			oi++
		}
	}

	return playlist
}

// truncate shortens a string to maxLen runes, appending "…" if
// truncated. Operates on runes to avoid splitting multi-byte characters.
func truncate(s string, maxLen int) string {
	runes := []rune(s)
	if len(runes) <= maxLen {
		return s
	}
	if maxLen <= 1 {
		return "…"
	}
	return string(runes[:maxLen-1]) + "…"
}
