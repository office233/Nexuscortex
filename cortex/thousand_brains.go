package cortex

import (
	"math/bits"
	"math/rand"
	"sync"
)

// inputProjectionPrime is the prime multiplier used to distribute input
// bits across different column neuron indices. Each column c maps input
// bit idx to neuron (idx + c*inputProjectionPrime) % ColNeurons.
// This must be coprime with ColNeurons to avoid degenerate projections.
// 37 is a safe default for typical ColNeurons values (100, 200, etc.).
const inputProjectionPrime = 37

// ─────────────────────────────────────────────────────────────────────
// Thousand Brains — 1000 Brains Theory Implementation
// ─────────────────────────────────────────────────────────────────────
//
// Inspired by Numenta's Thousand Brains Theory of Intelligence, this
// module implements an ensemble of cortical mini-columns that each
// independently process sensory input and form local beliefs. A
// consensus mechanism aggregates across columns via majority voting.
//
// Architecture:
//   - Each MiniColumn wraps a spiking Network (100 neurons, 10%
//     connectivity) that processes input independently.
//   - After processing, each column's activation pattern becomes
//     its Belief SDR with an associated Confidence score.
//   - The Vote method produces a Consensus SDR where only bits
//     active in >50% of columns survive — a form of democratic
//     noise filtering inspired by cortical voting.
//   - Disagree measures the average pairwise similarity across
//     all columns (0 = perfect agreement, 1 = complete disagreement).
//
// All SDRs produced by this module share the same dimensionality
// as the underlying Network (100 neurons).

// ─────────────────────────────────────────────────────────────────────
// MiniColumn — Individual Cortical Column
// ─────────────────────────────────────────────────────────────────────

// MiniColumn represents a single cortical column in the ensemble.
// Each column has its own spiking network that independently forms
// beliefs about the input. Confidence reflects how strongly the
// column believes in its current representation (spike density).
type MiniColumn struct {
	Net            *Network // Spiking network for this column
	Belief         SDR      // Current belief state (activation pattern)
	Confidence     uint8    // Confidence level (0–255, derived from spike density)
	ExternalInputs []uint8  // Pre-allocated dense slice for projection inputs
}

// ─────────────────────────────────────────────────────────────────────
// ThousandBrains — Ensemble of Cortical Columns
// ─────────────────────────────────────────────────────────────────────

// ThousandBrains implements the Thousand Brains Theory by managing
// an ensemble of MiniColumn networks. Each column independently
// processes input, then a consensus mechanism aggregates beliefs
// through majority voting.
type ThousandBrains struct {
	Columns       []MiniColumn // All cortical columns in the ensemble
	ColumnCount   int          // Number of columns
	Consensus     SDR          // Current consensus SDR (majority vote)
	ConsensusConf uint8        // Consensus confidence (0–255)
	Rng           *rand.Rand   // Shared deterministic random source
	ColNeurons       int
	ColConnectivity  float64
	ProcessTicks     int
	InputCurrent     uint8

	votes []int // Pre-allocated transient slice for voting
}

// ThousandBrainsStats holds aggregate statistics about the ensemble.
type ThousandBrainsStats struct {
	ColumnCount   int     // Number of cortical columns
	AvgConfidence int     // Average column confidence (0–255)
	ConsensusConf int     // Current consensus confidence (0–255)
	TotalNeurons  int     // Total neurons across all columns
	TotalSynapses int     // Total synapses across all columns
	Disagreement  uint8   // Average pairwise disagreement (0–255)
}

// ─────────────────────────────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────────────────────────────

// NewThousandBrains creates a new Thousand Brains ensemble with cortical columns configured via Config.
func NewThousandBrains(cfg Config, rng *rand.Rand) *ThousandBrains {
	columnCount := cfg.ThousandBrainsColumns
	if columnCount <= 0 {
		columnCount = 1
	}

	tb := &ThousandBrains{
		Columns:         make([]MiniColumn, columnCount),
		ColumnCount:     columnCount,
		Consensus:       NewSDR(cfg.ThousandBrainsColNeurons),
		Rng:             rng,
		ColNeurons:      cfg.ThousandBrainsColNeurons,
		ColConnectivity: cfg.ThousandBrainsColConnectivity,
		ProcessTicks:    cfg.ThousandBrainsProcessTicks,
		InputCurrent:    cfg.ThousandBrainsInputCurrent,
		votes:           make([]int, cfg.ThousandBrainsColNeurons),
	}

	for i := 0; i < columnCount; i++ {
		// Each column gets its own rng derived from the shared source
		// so that columns have independent but reproducible dynamics.
		colSeed := rng.Int63()
		colRng := rand.New(rand.NewSource(colSeed))

		net := NewNetwork(tb.ColNeurons, tb.ColConnectivity, cfg, colRng)

		tb.Columns[i] = MiniColumn{
			Net:            net,
			Belief:         NewSDR(tb.ColNeurons),
			ExternalInputs: make([]uint8, tb.ColNeurons),
		}
	}

	return tb
}

// ─────────────────────────────────────────────────────────────────────
// Process — Feed input through all columns (parallel)
// ─────────────────────────────────────────────────────────────────────

// Process injects the input SDR into every column's network, runs
// each column for Config.ThousandBrainsProcessTicks ticks, extracts
// each column's belief (activation pattern), computes confidence,
// and then calls Vote to form the ensemble consensus. Returns the
// consensus SDR.
//
// Each column's Network has its own state (neurons, synapses, rng)
// so goroutines are safe: each goroutine writes only to its own
// column's Belief and Confidence fields within the pre-allocated
// Columns slice.
//
// Input SDR active bits are mapped to neuron indices (clamped to the
// column size). Each active bit injects a current of InputCurrent
// into the corresponding neuron on every tick.
func (tb *ThousandBrains) Process(input SDR) SDR {
	var wg sync.WaitGroup
	wg.Add(tb.ColumnCount)

	for c := range tb.Columns {
		c := c // capture loop variable for goroutine
		go func() {
			defer wg.Done()
			col := &tb.Columns[c]

			// Build external input slice by folding the full SDR into the
			// column's neuron range. Each column uses a different offset
			// so that columns see diverse projections of the same input.
			// Mapping: neuronIdx = (bitIndex + columnIndex * 37) % ColNeurons
			for i := range col.ExternalInputs {
				col.ExternalInputs[i] = 0
			}
			for w, word := range input.Bits {
				if word == 0 {
					continue
				}
				base := w * 64
				for word != 0 {
					tz := bits.TrailingZeros64(word)
					idx := base + tz
					neuronIdx := (idx + c*inputProjectionPrime) % tb.ColNeurons
					col.ExternalInputs[neuronIdx] = tb.InputCurrent
					word &= word - 1
				}
			}

			// Run the column's network for ProcessTicks ticks by calling Tick directly in a loop.
			// This completely avoids allocating results slices inside RunTicks.
			for t := 0; t < tb.ProcessTicks; t++ {
				col.Net.Tick(col.ExternalInputs)
			}

			// Extract the column's belief from the final activation pattern.
			pattern := col.Net.GetActivationPatternReadOnly()
			col.Belief.Reset()
			spikeCount := 0
			for i, spiked := range pattern {
				if spiked {
					col.Belief.Set(i)
					spikeCount++
				}
			}

			// Confidence = spike density scaled to [0, 255].
			// If all neurons spike, confidence = 255. If none, confidence = 0.
			if tb.ColNeurons > 0 {
				col.Confidence = uint8((spikeCount * 255) / tb.ColNeurons)
			}
		}()
	}

	wg.Wait()

	// Form consensus via majority vote across all columns.
	tb.Consensus = tb.Vote()

	// Consensus confidence = average of column confidences weighted
	// by how much each column agrees with the consensus.
	var totalConf uint64
	for c := range tb.Columns {
		totalConf += uint64(tb.Columns[c].Confidence)
	}
	if tb.ColumnCount > 0 {
		tb.ConsensusConf = uint8(totalConf / uint64(tb.ColumnCount))
	}

	return tb.Consensus
}

// ─────────────────────────────────────────────────────────────────────
// Vote — Majority rule across columns
// ─────────────────────────────────────────────────────────────────────

// Vote produces a consensus SDR using majority rule. A bit is set
// in the output if and only if it is active in more than 50% of the
// columns' Belief SDRs. This filters out noise: only strongly
// corroborated bits survive.
func (tb *ThousandBrains) Vote() SDR {
	result := NewSDR(tb.ColNeurons)
	threshold := tb.ColumnCount / 2 // >50% means strictly greater than half

	// Clear the pre-allocated votes slice
	for i := range tb.votes {
		tb.votes[i] = 0
	}

	// Count how many columns have each bit active.
	// We iterate the bits in col.Belief.Bits directly via TrailingZeros64,
	// which avoids calling ActiveIndices() and allocating index slices.
	for c := range tb.Columns {
		s := tb.Columns[c].Belief
		for w, word := range s.Bits {
			if word == 0 {
				continue
			}
			base := w * 64
			for word != 0 {
				// Find lowest set bit position.
				tz := bits.TrailingZeros64(word)
				idx := base + tz
				if idx < tb.ColNeurons {
					tb.votes[idx]++
				}
				word &= word - 1 // Clear lowest set bit.
			}
		}
	}

	// Set bits that exceed the majority threshold.
	for i := 0; i < tb.ColNeurons; i++ {
		if tb.votes[i] > threshold {
			result.Set(i)
		}
	}

	return result
}

// ─────────────────────────────────────────────────────────────────────
// Disagree — Inter-column disagreement measure
// ─────────────────────────────────────────────────────────────────────

// Disagree returns the average pairwise disagreement across all
// columns. Disagreement is defined as 1 - similarity for each pair.
//
// Returns 0.0 when all columns agree perfectly (identical beliefs),
// and approaches 1.0 when columns have no overlap at all.
//
// For a single column, disagreement is 0.0 by definition (no pairs).
func (tb *ThousandBrains) Disagree() uint8 {
	if tb.ColumnCount < 2 {
		return 0
	}

	var totalDisagreement int
	pairs := 0

	for i := 0; i < tb.ColumnCount; i++ {
		for j := i + 1; j < tb.ColumnCount; j++ {
			sim := tb.Columns[i].Belief.Similarity(tb.Columns[j].Belief)
			totalDisagreement += 255 - int(sim)
			pairs++
		}
	}

	if pairs == 0 {
		return 0
	}

	return uint8(totalDisagreement / pairs)
}

// ─────────────────────────────────────────────────────────────────────
// Stats — Ensemble introspection
// ─────────────────────────────────────────────────────────────────────

// Stats returns aggregate statistics about the Thousand Brains
// ensemble, including total neurons, total synapses, average
// confidence, and disagreement across all columns.
func (tb *ThousandBrains) Stats() ThousandBrainsStats {
	stats := ThousandBrainsStats{
		ColumnCount:   tb.ColumnCount,
		ConsensusConf: int(tb.ConsensusConf),
	}

	var totalConf uint64
	for c := range tb.Columns {
		col := &tb.Columns[c]
		totalConf += uint64(col.Confidence)

		netStats := col.Net.Stats()
		stats.TotalNeurons += netStats.TotalNeurons
		stats.TotalSynapses += netStats.TotalSynapses
	}

	if tb.ColumnCount > 0 {
		stats.AvgConfidence = int(totalConf / uint64(tb.ColumnCount))
	}

	stats.Disagreement = tb.Disagree()

	return stats
}
