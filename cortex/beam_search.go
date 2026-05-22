package cortex

import "sort"

// ─────────────────────────────────────────────────────────────────────
// Beam Search — Multi-Path Sequence Generation
// ─────────────────────────────────────────────────────────────────────
//
// Instead of greedily picking the single best next word at each step,
// beam search maintains BeamWidth candidate sequences in parallel.
// At each step every candidate is expanded with its top-K next-word
// predictions (from both SequenceMemory and Brain synapses), and
// only the globally best BeamWidth expansions survive to the next step.
//
// All arithmetic is integer-only (uint16/uint32/uint64). No float64.

// BeamCandidate represents one candidate sequence being explored.
type BeamCandidate struct {
	Words    []string // Words generated so far (output only, not prompt).
	IDs      []uint32 // Full context word IDs (prompt + generated).
	Score    uint64   // Cumulative score (higher = better).
	Finished bool     // True when no further predictions are possible.
}

// BeamSearch holds the state for a single beam search run.
type BeamSearch struct {
	Brain                *Brain
	BeamWidth            int // Number of candidates to keep per step.
	MaxWords             int // Maximum words to generate.
	MaxCandidatesPerStep int // How many next-word expansions per beam.
}

// beamNextCandidate is an internal struct for next-word options.
type beamNextCandidate struct {
	id    uint32
	score uint32
}

// Generate runs beam search from the given prompt and returns the
// best sequence as a formatted string. The caller must hold b.mu.RLock.
func (bs *BeamSearch) Generate(prompt string) string {
	tokens := Tokenize(prompt)
	if len(tokens) == 0 {
		return ""
	}

	b := bs.Brain

	// Build initial context IDs from prompt tokens.
	promptIDs := make([]uint32, 0, len(tokens))
	for _, t := range tokens {
		id := b.Vocab.Get(t)
		if id != 0 {
			promptIDs = append(promptIDs, id)
		}
	}
	if len(promptIDs) == 0 {
		return ""
	}

	beamWidth := bs.BeamWidth
	if beamWidth <= 0 {
		beamWidth = 5
	}
	maxCands := bs.MaxCandidatesPerStep
	if maxCands <= 0 {
		maxCands = 10
	}

	// Initialise beams: one starting candidate with the prompt context.
	initialIDs := make([]uint32, len(promptIDs))
	copy(initialIDs, promptIDs)

	beams := []BeamCandidate{
		{
			Words:    nil,
			IDs:      initialIDs,
			Score:    0,
			Finished: false,
		},
	}

	for step := 0; step < bs.MaxWords; step++ {
		// Collect all expansions across every active beam.
		var allExpansions []BeamCandidate

		allFinished := true
		for bi := range beams {
			beam := &beams[bi]
			if beam.Finished {
				allExpansions = append(allExpansions, *beam)
				continue
			}
			allFinished = false

			// Build a recentlyUsed map for anti-repetition.
			recentlyUsed := make(map[uint32]int)
			for _, id := range beam.IDs {
				recentlyUsed[id]++
			}

			// Gather next-word candidates from BOTH sources.
			nextCands := bs.gatherCandidates(beam.IDs, recentlyUsed, maxCands)

			if len(nextCands) == 0 {
				// No more predictions — mark finished, keep beam.
				finished := BeamCandidate{
					Words:    cloneStrings(beam.Words),
					IDs:      cloneUint32s(beam.IDs),
					Score:    beam.Score,
					Finished: true,
				}
				allExpansions = append(allExpansions, finished)
				continue
			}

			// Expand beam with each next-word candidate.
			for _, nc := range nextCands {
				word := b.Vocab.Decode(nc.id)

				newWords := make([]string, len(beam.Words)+1)
				copy(newWords, beam.Words)
				newWords[len(beam.Words)] = word

				newIDs := make([]uint32, len(beam.IDs)+1)
				copy(newIDs, beam.IDs)
				newIDs[len(beam.IDs)] = nc.id

				newScore := beam.Score + uint64(nc.score)

				allExpansions = append(allExpansions, BeamCandidate{
					Words:    newWords,
					IDs:      newIDs,
					Score:    newScore,
					Finished: false,
				})
			}
		}

		if allFinished {
			break
		}

		// Sort by score descending and keep only top BeamWidth.
		sort.Slice(allExpansions, func(i, j int) bool {
			return allExpansions[i].Score > allExpansions[j].Score
		})
		if len(allExpansions) > beamWidth {
			allExpansions = allExpansions[:beamWidth]
		}

		beams = allExpansions
	}

	// Pick the highest-scoring beam.
	if len(beams) == 0 {
		return ""
	}

	best := &beams[0]
	for i := 1; i < len(beams); i++ {
		if beams[i].Score > best.Score {
			best = &beams[i]
		}
	}

	return bs.Brain.formatOutput(best.Words)
}

// gatherCandidates collects scored next-word candidates from both
// SequenceMemory (all context windows) and Brain synapses, merges
// them, and returns the top-K by score. All math is integer.
func (bs *BeamSearch) gatherCandidates(
	contextIDs []uint32,
	recentlyUsed map[uint32]int,
	topK int,
) []beamNextCandidate {
	b := bs.Brain
	scores := make(map[uint32]uint32)

	// ── Source 1: SequenceMemory (multi-context windows) ──
	if b.SeqMem != nil && len(contextIDs) > 0 {
		b.SeqMem.mu.RLock()
		for _, winSize := range defaultWindowSizes {
			if winSize > len(contextIDs) {
				continue
			}
			cw := b.SeqMem.Windows[winSize]
			if cw == nil {
				continue
			}
			ctxSlice := contextIDs[len(contextIDs)-winSize:]
			hash := contextHash(ctxSlice)
			transitions := cw.Transitions[hash]
			for _, t := range transitions {
				// Longer context = higher multiplier.
				score := uint32(t.Weight) * uint32(winSize)
				if count, ok := recentlyUsed[t.TargetID]; ok && count > 0 {
					score /= uint32(count + 1)
				}
				scores[t.TargetID] += score
			}
		}
		b.SeqMem.mu.RUnlock()
	}

	// ── Source 2: Brain synapses (bigram/skipgram/semantic) ──
	if len(contextIDs) > 0 {
		currentID := contextIDs[len(contextIDs)-1]
		seqMult := uint32(b.Config.BrocaSequentialMultiplier)
		if seqMult == 0 {
			seqMult = 3
		}
		syns := b.GetMergedSynapses(currentID)
		for _, syn := range syns {
			if syn.Flags&SynFlagActive == 0 || syn.Weight == 0 {
				continue
			}
			score := uint32(syn.Weight)
			if syn.Flags&SynFlagSequential != 0 {
				score *= seqMult
				if score > 65535 {
					score = 65535
				}
			}
			if count, ok := recentlyUsed[syn.Target]; ok && count > 0 {
				score /= uint32(count + 1)
			}
			if score > 0 {
				scores[syn.Target] += score
			}
		}
	}

	if len(scores) == 0 {
		return nil
	}

	// Collect and sort.
	candidates := make([]beamNextCandidate, 0, len(scores))
	for id, sc := range scores {
		if sc > 0 {
			candidates = append(candidates, beamNextCandidate{id: id, score: sc})
		}
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].score > candidates[j].score
	})

	if len(candidates) > topK {
		candidates = candidates[:topK]
	}

	return candidates
}

// GenerateBeam creates a BeamSearch and runs it against the brain.
// This is the public entry point wired onto Brain.
func (b *Brain) GenerateBeam(prompt string, maxWords int) string {
	b.mu.RLock()
	defer b.mu.RUnlock()

	beamWidth := b.Config.BeamSearchWidth
	if beamWidth <= 0 {
		beamWidth = 5
	}
	maxCands := b.Config.BeamSearchMaxCandidatesPerStep
	if maxCands <= 0 {
		maxCands = 10
	}

	bs := &BeamSearch{
		Brain:                b,
		BeamWidth:            beamWidth,
		MaxWords:             maxWords,
		MaxCandidatesPerStep: maxCands,
	}

	return bs.Generate(prompt)
}

// ─── helpers ──────────────────────────────────────────────────────────

func cloneStrings(s []string) []string {
	if s == nil {
		return nil
	}
	out := make([]string, len(s))
	copy(out, s)
	return out
}

func cloneUint32s(s []uint32) []uint32 {
	if s == nil {
		return nil
	}
	out := make([]uint32, len(s))
	copy(out, s)
	return out
}
