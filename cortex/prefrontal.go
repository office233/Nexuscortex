package cortex

import (
	"math/bits"
	"math/rand"
)

// ─────────────────────────────────────────────────────────────────────
// Prefrontal — Recursive Reasoning
// ─────────────────────────────────────────────────────────────────────
//
// Named after the prefrontal cortex in the human brain, this module
// handles higher-order reasoning by recursively refining SDR patterns
// through a spiking neural network reservoir.
//
// Architecture:
//   - 3000-neuron spiking network (80/20 excitatory/inhibitory)
//   - Sparse connectivity (5%) for rich dynamic attractors
//   - Recursive refinement: input is injected, the network runs
//     for multiple cycles, and the resulting activation pattern
//     is read back as a refined SDR.
//
// The key insight is that the reservoir's attractor dynamics act
// as a form of "deliberation": the network settles into a stable
// pattern that represents the most coherent interpretation of the
// input, analogous to how the prefrontal cortex iterates on a
// thought before committing to a decision.
//
// Confidence is measured by activation stability: how consistent
// the network's firing pattern is across the final cycles.

// Prefrontal performs recursive reasoning by refining SDR patterns
// through a spiking neural network reservoir.
type Prefrontal struct {
	Net           *Network
	ThinkCycles   int
	Confidence    uint8
	InputCurrent  uint8
	Cfg           Config
	tickInputs    []uint8 // Pre-allocated transient slice for Tick inputs
	activeIndices []int   // Pre-allocated transient buffer for sorting/truncation
}

// NewPrefrontal creates a new reasoning module with a spiking network.
func NewPrefrontal(cfg Config, rng *rand.Rand) *Prefrontal {
	return NewPrefrontalWithSize(cfg, cfg.PrefrontalNetSize, cfg.PrefrontalConnectivity, rng)
}

// NewPrefrontalWithSize creates a prefrontal module with explicit
// network parameters.
func NewPrefrontalWithSize(cfg Config, netSize int, connectivity float64, rng *rand.Rand) *Prefrontal {
	thinkCycles := cfg.ThinkCycles
	if thinkCycles <= 0 {
		thinkCycles = 10
	}
	if netSize <= 0 {
		netSize = 3000
	}
	if connectivity <= 0 {
		connectivity = 0.05
	}
	return &Prefrontal{
		Net:           NewNetwork(netSize, connectivity, cfg, rng),
		ThinkCycles:   thinkCycles,
		Confidence:    0,
		InputCurrent:  cfg.PrefrontalInputCurrent,
		Cfg:           cfg,
		tickInputs:    make([]uint8, netSize),
		activeIndices: make([]int, 0, netSize),
	}
}

// Think recursively refines the input SDR by running it through
// the spiking network for the specified number of cycles.
//
// The process:
//  1. Convert the input SDR into neuron injection currents.
//  2. Run the network for 'cycles' ticks with the input applied.
//  3. Read the resulting activation pattern as a refined SDR.
//  4. Measure confidence from the stability of the final pattern.
//
// The output SDR has the same size as the input and represents the
// network's refined interpretation of the input pattern. Patterns
// that align with learned attractors will be sharpened; noisy or
// ambiguous patterns will converge toward the nearest attractor.
func (p *Prefrontal) Think(input SDR, cycles int) SDR {
	if input.Size == 0 {
		return input
	}
	if cycles <= 0 {
		cycles = p.ThinkCycles
	}

	// Build the external input slice by folding the input SDR.
	// We iterate the bits in input.Bits directly via TrailingZeros64
	// to completely avoid allocating a map or active index slice.
	netSize := p.Net.Neurons.Size()
	if len(p.tickInputs) != netSize {
		p.tickInputs = make([]uint8, netSize)
	}
	for i := range p.tickInputs {
		p.tickInputs[i] = 0
	}
	for w, word := range input.Bits {
		if word == 0 {
			continue
		}
		base := w * 64
		for word != 0 {
			tz := bits.TrailingZeros64(word)
			idx := base + tz
			mapped := idx % netSize
			p.tickInputs[mapped] = p.InputCurrent
			word &= word - 1
		}
	}

	// Run the network for the specified number of cycles.
	for tick := 0; tick < cycles; tick++ {
		p.Net.Tick(p.tickInputs)
	}

	// Determine stability window.
	stabilityWindow := cycles / 3
	if stabilityWindow < 2 {
		stabilityWindow = 2
	}

	// Directly sub-slice SpikeHistory instead of allocating and copying snapshots.
	history := p.Net.SpikeHistory
	if len(history) < stabilityWindow {
		stabilityWindow = len(history)
	}
	var recentPatterns [][]bool
	if stabilityWindow >= 2 {
		recentPatterns = history[len(history)-stabilityWindow:]
	}

	// Read the final activation pattern using a zero-allocation read-only reference
	// and convert it back to SDR space.
	finalPattern := p.Net.GetActivationPatternReadOnly()
	outputSDR := NewSDR(input.Size)

	for i, fired := range finalPattern {
		if fired && i < input.Size {
			outputSDR.Set(i)
		}
	}

	// If the input SDR is smaller than the network, wrap excess
	// neuron indices back into the SDR space.
	if input.Size < netSize {
		for i := input.Size; i < len(finalPattern); i++ {
			if finalPattern[i] {
				wrapped := i % input.Size
				if !outputSDR.IsActive(wrapped) {
					outputSDR.Set(wrapped)
				}
			}
		}
	}

	// Sparsity cap: deterministic truncation to preserve sparsity invariants.
	// Uses the pre-allocated p.activeIndices slice and popcounts raw bitfield words
	// to avoid intermediate index allocations.
	maxActive := input.ActiveCount * 2
	if maxActive > 0 && outputSDR.ActiveCount > maxActive {
		p.activeIndices = p.activeIndices[:0]
		for w, word := range outputSDR.Bits {
			if word == 0 {
				continue
			}
			base := w * 64
			for word != 0 {
				tz := bits.TrailingZeros64(word)
				idx := base + tz
				if idx < outputSDR.Size {
					p.activeIndices = append(p.activeIndices, idx)
				}
				word &= word - 1
			}
		}
		outputSDR.Reset()
		for i := 0; i < maxActive && i < len(p.activeIndices); i++ {
			outputSDR.Set(p.activeIndices[i])
		}
	}

	// Measure confidence from activation stability.
	p.Confidence = measureStability(recentPatterns)

	return outputSDR
}

// ThinkDeep performs multi-hop iterative reasoning by feeding the output
// of each Think pass back as input for the next hop. This enables
// recursive refinement where the network progressively converges on
// a stable attractor state.
//
// Early termination occurs when consecutive outputs achieve a similarity
// above the configured convergence threshold, indicating the network
// has settled into a stable state.
//
// Confidence is set to the inter-hop stability (similarity between the
// last two outputs), NOT the intra-think stability.
func (p *Prefrontal) ThinkDeep(input SDR, hops int) SDR {
	if input.Size == 0 || hops <= 0 {
		return input
	}

	convergenceThresh := p.Cfg.PrefrontalConvergenceThresh
	if convergenceThresh == 0 {
		convergenceThresh = 240
	}

	current := input
	var prev SDR
	var hopStability uint8

	for hop := 0; hop < hops; hop++ {
		result := p.Think(current, p.ThinkCycles)

		if hop > 0 {
			// Measure similarity between consecutive hop outputs.
			hopStability = result.Similarity(prev)
			if hopStability >= convergenceThresh {
				// Converged — the network has settled.
				p.Confidence = hopStability
				return result
			}
		}

		prev = current
		current = result
	}

	// Set confidence to inter-hop stability of the final pair.
	if hops > 1 {
		p.Confidence = hopStability
	}
	// If only 1 hop, confidence was already set by Think.

	return current
}

// GetConfidence returns the confidence score from the most recent
// Think operation. Ranges from 0.0 (chaotic, unstable firing) to
// 1.0 (perfectly stable attractor state).
func (p *Prefrontal) GetConfidence() uint8 {
	return p.Confidence
}

// Simulate evaluates multiple competing SDR options by running each
// through the reasoning network and comparing their resulting
// confidence scores. Returns the index of the best option and its
// confidence.
//
// This simulates deliberation: the prefrontal cortex "tries out"
// each option internally and selects the one that produces the
// most stable (confident) response.


// measureStability computes how consistent the ACTIVE firing patterns are
// across the given snapshots. Returns 0 for no agreement and 255 for
// perfectly identical firing patterns.
//
// IMPORTANT: We only count neurons that FIRED in at least one of the two
// patterns. Counting silent-silent matches is meaningless with sparse
// networks (~5% active) because it would always produce ~95% stability.
//
// Algorithm: for each pair of consecutive snapshots, count neurons that
// fired in BOTH (intersection) and divide by neurons that fired in EITHER
// (union). This is the Jaccard index of active neurons, scaled to 0–255.
func measureStability(patterns [][]bool) uint8 {
	if len(patterns) < 2 {
		return 0
	}

	totalScore := 0
	pairs := 0

	for i := 1; i < len(patterns); i++ {
		prev := patterns[i-1]
		curr := patterns[i]

		n := len(prev)
		if len(curr) < n {
			n = len(curr)
		}
		if n == 0 {
			continue
		}

		intersection := 0 // Fired in BOTH
		union := 0        // Fired in EITHER

		for j := 0; j < n; j++ {
			if prev[j] || curr[j] {
				union++
				if prev[j] && curr[j] {
					intersection++
				}
			}
		}

		if union > 0 {
			totalScore += intersection * 255 / union
		}
		// If union == 0, no neurons fired at all — skip this pair
		pairs++
	}

	if pairs == 0 {
		return 0
	}
	return uint8(totalScore / pairs)
}
