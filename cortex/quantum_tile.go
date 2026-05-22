package cortex

import (
	"math"
	"math/bits"
	"math/rand"
	"sync"
	"sync/atomic"
)

// ─────────────────────────────────────────────────────────────────────
// Quantum-Inspired NeuroTexture Engine
// ─────────────────────────────────────────────────────────────────────
//
// Extends the RGBA32 TernaryTile with quantum-inspired concepts:
//
//   1. QuantumTile   = TernaryTile + Amplitude + Phase (6 bytes/16 weights)
//   2. PBitNeuron    = probabilistic neuron with temperature-controlled sampling
//   3. QuantumRouter = phase-based expert routing with interference
//   4. MultiSample   = multiple forward passes with perturbation → confidence
//
// Mathematical foundation from quantum mechanics:
//   - Amplitude: |ψ|² = probability of being "correct" (0-255 → 0.0-1.0)
//   - Phase: φ = angle in [0, 2π) encoded as uint8 (0=0°, 128=180°, 255≈360°)
//   - Interference: contribution × cos(phase_diff) — constructive when aligned
//
// This is NOT a quantum computer. It's a "quantum-inspired probabilistic
// neural engine" that uses the same mathematical principles on classical hardware.

// ─────────────────────────────────────────────────────────────────────
// QuantumTile — TernaryTile + Amplitude + Phase
// ─────────────────────────────────────────────────────────────────────

// QuantumTile extends TernaryTile with quantum-inspired metadata.
// Total: 6 bytes per 16 weights (vs 4 bytes for plain TernaryTile).
//
//   Weights:   16 ternary values {-1, 0, +1} packed as RGBA32
//   Amplitude: confidence/importance of this tile (0=uncertain, 255=certain)
//   Phase:     interference angle (0=0°, 64=90°, 128=180°, 192=270°)
type QuantumTile struct {
	Weights   TernaryTile
	Amplitude uint8 // |ψ|² mapped to [0, 255]
	Phase     uint8 // φ mapped to [0, 255] → [0, 2π)
}

// NewQuantumTile wraps an existing TernaryTile with full amplitude and zero phase.
// NOTE: amplitude=255 + phase=0 produces APPROXIMATELY classical behavior.
// There is a small rounding error (~1%) due to integer division (>>15 vs /32385).
func NewQuantumTile(t TernaryTile) QuantumTile {
	return QuantumTile{
		Weights:   t,
		Amplitude: 255, // fully certain
		Phase:     0,   // no phase shift
	}
}

// ─────────────────────────────────────────────────────────────────────
// Cosine lookup table for fast phase computation
// ─────────────────────────────────────────────────────────────────────

// cos256 maps uint8 phase → cos(phase * 2π / 256) * 127, as int8.
// This avoids float math entirely in the hot path.
//
//   cos256[0]   = +127 (cos 0°)
//   cos256[64]  =    0 (cos 90°)
//   cos256[128] = -127 (cos 180°)
//   cos256[192] =    0 (cos 270°)
var cos256 [256]int8

func init() {
	for i := 0; i < 256; i++ {
		angle := float64(i) * 2.0 * math.Pi / 256.0
		cos256[i] = int8(math.Round(math.Cos(angle) * 127.0))
	}
}

// ─────────────────────────────────────────────────────────────────────
// QuantumTernaryLayer — Layer with amplitude + phase per tile
// ─────────────────────────────────────────────────────────────────────

// QuantumTernaryLayer extends TernaryLayer with per-tile amplitude and phase.
type QuantumTernaryLayer struct {
	Base       *TernaryLayer // underlying ternary weights
	Amplitudes []uint8       // per-tile amplitude (|ψ|²)
	Phases     []uint8       // per-tile phase (φ)
}

// NewQuantumTernaryLayer wraps a TernaryLayer with quantum metadata.
// All amplitudes start at 255 (fully certain), phases at 0 (no shift).
func NewQuantumTernaryLayer(base *TernaryLayer) *QuantumTernaryLayer {
	n := len(base.Tiles)
	amps := make([]uint8, n)
	phases := make([]uint8, n)
	for i := range amps {
		amps[i] = 255 // fully certain — classical behavior
	}
	return &QuantumTernaryLayer{
		Base:       base,
		Amplitudes: amps,
		Phases:     phases,
	}
}

// ForwardQuantum performs a forward pass with quantum-inspired interference.
//
// For each output neuron j, for each tile t in that row:
//
//	classical_contribution = popcount(pos & active) - popcount(neg & active)
//	amplitude_scale = Amplitudes[tile_idx] / 255.0
//	phase_factor = cos(Phases[tile_idx] - input_phase)
//	quantum_contribution = classical_contribution * amplitude_scale * phase_factor
//
// The result: tiles with high amplitude and aligned phase → amplified (constructive).
// Tiles with low amplitude or misaligned phase → suppressed (destructive).
//
// NOTE: When all amplitudes=255 and all phases=0, this produces APPROXIMATELY
// classical results. The integer division (>>15 ≈ /32768 vs exact /32385)
// introduces ~1% rounding error. Use ForwardClassical() for exact classical path.
func (q *QuantumTernaryLayer) ForwardQuantum(activeMask []uint16, inputPhase uint8) []int16 {
	base := q.Base
	output := make([]int16, base.OutputSize)
	copy(output, base.Bias)

	for j := 0; j < base.OutputSize; j++ {
		var acc int32
		rowOffset := j * base.TilesPerRow

		for t := 0; t < base.TilesPerRow; t++ {
			tileIdx := rowOffset + t
			tile := uint32(base.Tiles[tileIdx])
			if tile == 0 {
				continue
			}

			// Classical ternary contribution via popcount
			signLo := uint8(tile)
			maskLo := uint8(tile >> 8)
			signHi := uint8(tile >> 16)
			maskHi := uint8(tile >> 24)

			posMask := uint16(maskLo&^signLo) | uint16(maskHi&^signHi)<<8
			negMask := uint16(maskLo&signLo) | uint16(maskHi&signHi)<<8

			var contribution int32
			if t < len(activeMask) {
				active := activeMask[t]
				contribution = int32(pop16[posMask&active]) - int32(pop16[negMask&active])
			}

			if contribution == 0 {
				continue
			}

			// Quantum modulation: amplitude × cos(phase_diff)
			amp := q.Amplitudes[tileIdx]
			phaseDiff := q.Phases[tileIdx] - inputPhase
			cosFactor := cos256[phaseDiff] // int8: -127 to +127

			// Scale: contribution * (amp/255) * (cosFactor/127)
			// Using integer math: contribution * amp * cosFactor / (255 * 127)
			// Approximate: / 32385 ≈ >> 15
			modulated := int32(contribution) * int32(amp) * int32(cosFactor)
			acc += modulated >> 15 // fast approximate division by 32768
		}

		if acc > 32767 {
			acc = 32767
		} else if acc < -32768 {
			acc = -32768
		}
		output[j] += int16(acc)
	}

	return output
}

// ForwardClassical performs a forward pass ignoring amplitude/phase (backward compat).
// Equivalent to ForwardPopcount on the base layer.
func (q *QuantumTernaryLayer) ForwardClassical(activeMask []uint16) []int16 {
	return q.Base.ForwardPopcount(activeMask)
}

// AmplitudeStats returns min, max, mean amplitude across all tiles.
func (q *QuantumTernaryLayer) AmplitudeStats() (min, max uint8, mean float64) {
	if len(q.Amplitudes) == 0 {
		return 0, 0, 0
	}
	min = 255
	max = 0
	var sum uint64
	for _, a := range q.Amplitudes {
		if a < min {
			min = a
		}
		if a > max {
			max = a
		}
		sum += uint64(a)
	}
	mean = float64(sum) / float64(len(q.Amplitudes))
	return
}

// ─────────────────────────────────────────────────────────────────────
// PBitLayer — Probabilistic Neuron Layer
// ─────────────────────────────────────────────────────────────────────
//
// Inspired by p-bit research (UCSB/Tohoku 2025).
// Each neuron fluctuates stochastically based on temperature,
// simulating quantum superposition of states.
//
//   temperature = 0:   deterministic (classical)
//   temperature > 0:   probabilistic (quantum-fake superposition)
//   temperature = 255: maximally stochastic

// PBitNeuron represents a single probabilistic neuron.
type PBitNeuron struct {
	Bias        int16  // activation bias
	Temperature uint8  // stochastic temperature (0=deterministic, 255=max noise)
	Trace       uint8  // eligibility trace for plasticity
}

// PBitLayer is a layer of probabilistic neurons.
type PBitLayer struct {
	Neurons []PBitNeuron
	Size    int
	rng     *rand.Rand
	mu      sync.Mutex
}

// NewPBitLayer creates a layer of probabilistic neurons.
func NewPBitLayer(size int, rng *rand.Rand) *PBitLayer {
	neurons := make([]PBitNeuron, size)
	return &PBitLayer{
		Neurons: neurons,
		Size:    size,
		rng:     rng,
	}
}

// Activate computes probabilistic activations for the layer.
//
// For each neuron:
//
//	raw_activation = bias + input[i]
//	if temperature == 0: output = sign(raw_activation)
//	else:
//	  step 1: Bernoulli(sigmoid(|raw| / temperature)) → active or not
//	  step 2: if active, sign = sign(raw)
//
// Returns activations as int16 values (-1, 0, or +1).
func (p *PBitLayer) Activate(input []int16) []int16 {
	output := make([]int16, p.Size)

	p.mu.Lock()
	defer p.mu.Unlock()

	n := p.Size
	if len(input) < n {
		n = len(input)
	}

	for i := 0; i < n; i++ {
		neuron := &p.Neurons[i]
		raw := int32(neuron.Bias) + int32(input[i])

		if neuron.Temperature == 0 {
			// Deterministic: simple sign function
			if raw > 0 {
				output[i] = 1
			} else if raw < 0 {
				output[i] = -1
			}
			continue
		}

		// Probabilistic: two-step clean Bernoulli.
		// Step 1: compute activation probability from |raw| / temperature.
		// Higher |raw| → more likely to fire. Higher temperature → more noise.
		absRaw := raw
		if absRaw < 0 {
			absRaw = -absRaw
		}
		scaled := absRaw * 256 / (int32(neuron.Temperature) + 1)
		prob := fastSigmoid256(int16(clamp32(scaled, -2048, 2047)))

		// Step 2: Bernoulli sample — fire or not
		sample := uint8(p.rng.Intn(256))
		if sample < prob {
			// Fires. Sign determined by raw activation direction.
			if raw >= 0 {
				output[i] = 1
			} else {
				output[i] = -1
			}
		}
		// else: output[i] = 0 (did not fire — "undecided")
	}

	return output
}

// SetTemperature sets uniform temperature for all neurons.
func (p *PBitLayer) SetTemperature(temp uint8) {
	for i := range p.Neurons {
		p.Neurons[i].Temperature = temp
	}
}

// ─────────────────────────────────────────────────────────────────────
// QuantumRouter — Phase-based Expert Routing with Interference
// ─────────────────────────────────────────────────────────────────────

// QuantumRouter extends ExpertRouter with SDR similarity + phase interference.
// Score = semantic_similarity + phase_bonus + amplitude_confidence - usage_penalty
type QuantumRouter struct {
	ExpertPhases     []uint8  // phase per expert
	ExpertAmps       []uint8  // amplitude/confidence per expert
	ExpertEmbeddings []SDR    // SDR embedding per expert (semantic)
	TopK             int
	UsageCounts      []uint64 // accessed atomically
	SDRSize          int
}

// NewQuantumRouter creates a quantum-inspired router with SDR embeddings.
func NewQuantumRouter(numExperts, sdrSize, topK int) *QuantumRouter {
	phases := make([]uint8, numExperts)
	amps := make([]uint8, numExperts)
	embeddings := make([]SDR, numExperts)
	for i := range amps {
		amps[i] = 128 // initial neutral amplitude
		embeddings[i] = NewSDR(sdrSize)
	}
	return &QuantumRouter{
		ExpertPhases:     phases,
		ExpertAmps:       amps,
		ExpertEmbeddings: embeddings,
		TopK:             topK,
		UsageCounts:      make([]uint64, numExperts),
		SDRSize:          sdrSize,
	}
}

// RouteSDR selects top-K experts using BOTH SDR similarity AND phase interference.
//
// Score = semantic_similarity(input, expert_embedding) * 256
//       + amplitude * cos(phase_diff) / 128   (phase bonus, smaller weight)
//
// This ensures experts are relevant SEMANTICALLY first, with phase as a tiebreaker.
func (r *QuantumRouter) RouteSDR(input SDR) []int {
	type scored struct {
		idx   int
		score int32
	}

	inputPhase := SDRPhase(input)

	scores := make([]scored, len(r.ExpertPhases))
	for i := range r.ExpertPhases {
		// Primary: semantic SDR similarity (0-255 scaled to 0-65280)
		var semanticScore int32
		if r.ExpertEmbeddings[i].ActiveCount > 0 {
			semanticScore = int32(input.Similarity(r.ExpertEmbeddings[i])) * 256
		}

		// Secondary: phase interference bonus (-127 to +127 scaled by amplitude)
		phaseDiff := r.ExpertPhases[i] - inputPhase
		phaseBonus := int32(r.ExpertAmps[i]) * int32(cos256[phaseDiff]) / 128

		scores[i] = scored{i, semanticScore + phaseBonus}
	}

	// Select top-K by score
	k := r.TopK
	if k > len(scores) {
		k = len(scores)
	}

	topK := make([]scored, 0, k)
	for _, s := range scores {
		if len(topK) < k {
			topK = append(topK, s)
			for j := len(topK) - 1; j > 0; j-- {
				if topK[j].score > topK[j-1].score {
					topK[j], topK[j-1] = topK[j-1], topK[j]
				}
			}
		} else if s.score > topK[k-1].score {
			topK[k-1] = s
			for j := k - 1; j > 0; j-- {
				if topK[j].score > topK[j-1].score {
					topK[j], topK[j-1] = topK[j-1], topK[j]
				}
			}
		}
	}

	result := make([]int, len(topK))
	for i, s := range topK {
		result[i] = s.idx
		atomic.AddUint64(&r.UsageCounts[s.idx], 1)
	}
	return result
}

// Route selects top-K using phase-only scoring (simple path, no SDR needed).
func (r *QuantumRouter) Route(inputPhase uint8) []int {
	type scored struct {
		idx   int
		score int32
	}

	scores := make([]scored, len(r.ExpertPhases))
	for i := range r.ExpertPhases {
		phaseDiff := r.ExpertPhases[i] - inputPhase
		cosFactor := int32(cos256[phaseDiff])
		amp := int32(r.ExpertAmps[i])
		scores[i] = scored{i, amp * cosFactor}
	}

	k := r.TopK
	if k > len(scores) {
		k = len(scores)
	}

	topK := make([]scored, 0, k)
	for _, s := range scores {
		if len(topK) < k {
			topK = append(topK, s)
			for j := len(topK) - 1; j > 0; j-- {
				if topK[j].score > topK[j-1].score {
					topK[j], topK[j-1] = topK[j-1], topK[j]
				}
			}
		} else if s.score > topK[k-1].score {
			topK[k-1] = s
			for j := k - 1; j > 0; j-- {
				if topK[j].score > topK[j-1].score {
					topK[j], topK[j-1] = topK[j-1], topK[j]
				}
			}
		}
	}

	result := make([]int, len(topK))
	for i, s := range topK {
		result[i] = s.idx
		atomic.AddUint64(&r.UsageCounts[s.idx], 1)
	}
	return result
}

// UpdateExpertPhase adjusts an expert's phase toward the input phase.
// This is Hebbian-like: experts that succeed become more "in phase" with
// the inputs they're good at.
func (r *QuantumRouter) UpdateExpertPhase(expertIdx int, inputPhase uint8, reward bool) {
	if expertIdx < 0 || expertIdx >= len(r.ExpertPhases) {
		return
	}

	if reward {
		// Move phase toward input (constructive alignment)
		diff := int16(inputPhase) - int16(r.ExpertPhases[expertIdx])
		r.ExpertPhases[expertIdx] += uint8(diff / 4) // gradual alignment

		// Increase amplitude (more confident)
		if r.ExpertAmps[expertIdx] < 250 {
			r.ExpertAmps[expertIdx] += 5
		}
	} else {
		// Move phase away (destructive)
		diff := int16(inputPhase) - int16(r.ExpertPhases[expertIdx])
		r.ExpertPhases[expertIdx] -= uint8(diff / 8) // gradual misalignment

		// Decrease amplitude
		if r.ExpertAmps[expertIdx] > 5 {
			r.ExpertAmps[expertIdx] -= 5
		}
	}
}

// ─────────────────────────────────────────────────────────────────────
// MultiSampleInference — Quantum Superposition Fake
// ─────────────────────────────────────────────────────────────────────

// MultiSampleResult holds the result of multi-sample inference.
type MultiSampleResult struct {
	Output     SDR     // majority-voted output
	Confidence float32 // Jaccard agreement on active bits (0.0 = chaotic, 1.0 = all agree)
	NumSamples int     // how many samples were taken
}

// MultiSampleForward runs multiple forward passes with phase perturbations
// and returns the majority-voted result with a confidence score.
//
// Confidence is computed as Jaccard agreement over the UNION of all bits
// that were active in ANY sample. This avoids the bias where sparse SDRs
// get artificially high confidence from millions of always-inactive bits.
//
// Higher confidence = system is "certain" → classical-like
// Lower confidence = system is "uncertain" → needs more information
func MultiSampleForward(
	layer *QuantumTernaryLayer,
	activeMask []uint16,
	basePhase uint8,
	numSamples int,
	rng *rand.Rand,
) MultiSampleResult {
	if numSamples < 1 {
		numSamples = 1
	}
	if numSamples > 16 {
		numSamples = 16
	}

	dim := layer.Base.OutputSize
	voteCounts := make([]int, dim)

	for s := 0; s < numSamples; s++ {
		phase := basePhase
		if s > 0 {
			phase = basePhase + uint8(rng.Intn(64)-32)
		}

		output := layer.ForwardQuantum(activeMask, phase)

		for j, val := range output {
			if val > 0 {
				voteCounts[j]++
			}
		}
	}

	// Majority vote
	threshold := numSamples / 2
	result := NewSDR(dim)

	// Confidence: Jaccard over union of active bits.
	// Only count positions that were active in at least ONE sample.
	unionActive := 0  // bits active in any sample
	agreedActive := 0 // bits where all samples agreed (all active or all inactive WITHIN union)

	for j, count := range voteCounts {
		if count > threshold {
			result.Set(j)
		}
		// Only consider bits that appeared in at least one sample
		if count > 0 {
			unionActive++
			// Perfect agreement = active in ALL samples
			if count == numSamples {
				agreedActive++
			}
		}
	}

	var confidence float32
	if unionActive > 0 {
		confidence = float32(agreedActive) / float32(unionActive)
	} else {
		confidence = 0 // no activity at all = zero confidence
	}

	return MultiSampleResult{
		Output:     result,
		Confidence: confidence,
		NumSamples: numSamples,
	}
}

// ─────────────────────────────────────────────────────────────────────
// SDR Phase Extraction — Convert SDR to phase angle
// ─────────────────────────────────────────────────────────────────────

// SDRPhase extracts a phase angle from an SDR based on its active bit pattern.
// Uses a weighted combination of active bit positions for a richer phase signal
// than a simple hash truncation. Still 256 possible phases, but more spread.
func SDRPhase(sdr SDR) uint8 {
	if sdr.ActiveCount == 0 {
		return 0
	}

	// Combine hash (global pattern) with weighted position sum (local structure).
	// This gives different SDRs with similar hashes distinguishable phases.
	h := sdr.Hash()

	var posSum uint64
	for _, idx := range sdr.ActiveIndices() {
		posSum += uint64(idx) * 7919 // prime multiplier for spread
	}

	combined := h ^ posSum
	return uint8(combined & 0xFF)
}

// ─────────────────────────────────────────────────────────────────────
// Amplitude Plasticity — Update amplitudes based on prediction error
// ─────────────────────────────────────────────────────────────────────

// UpdateAmplitudes adjusts tile amplitudes based on their contribution
// to correct/incorrect predictions. This is the quantum-inspired analog
// of Elastic Weight Consolidation (EWC):
//
//   Tiles that contributed to CORRECT predictions → amplitude increases
//   Tiles that contributed to WRONG predictions → amplitude decreases
//
// This creates a natural "importance" measure: high-amplitude tiles are
// "important" and should be protected during learning (like EWC's Fisher
// Information Matrix, but computed incrementally and stored per-tile).
func (q *QuantumTernaryLayer) UpdateAmplitudes(
	activeMask []uint16,
	outputNeuron int,
	correct bool,
) {
	if outputNeuron < 0 || outputNeuron >= q.Base.OutputSize {
		return
	}

	rowOffset := outputNeuron * q.Base.TilesPerRow

	for t := 0; t < q.Base.TilesPerRow; t++ {
		tileIdx := rowOffset + t
		tile := uint32(q.Base.Tiles[tileIdx])
		if tile == 0 {
			continue
		}

		// Check if this tile was actually involved (active mask overlap)
		maskLo := uint8(tile >> 8)
		maskHi := uint8(tile >> 24)
		tileMask := uint16(maskLo) | uint16(maskHi)<<8

		var overlap int
		if t < len(activeMask) {
			overlap = bits.OnesCount16(tileMask & activeMask[t])
		}

		if overlap == 0 {
			continue // tile wasn't involved in this computation
		}

		if correct {
			// Increase amplitude (tile contributed to correct answer)
			if q.Amplitudes[tileIdx] < 252 {
				q.Amplitudes[tileIdx] += 3
			} else {
				q.Amplitudes[tileIdx] = 255
			}
		} else {
			// Decrease amplitude (tile contributed to wrong answer)
			if q.Amplitudes[tileIdx] > 3 {
				q.Amplitudes[tileIdx] -= 3
			} else {
				q.Amplitudes[tileIdx] = 0
			}
		}
	}
}

// ─────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────

// fastSigmoid256 approximates sigmoid(x/256) scaled to [0, 255].
// Uses piecewise linear approximation for speed.
func fastSigmoid256(x int16) uint8 {
	if x >= 512 {
		return 250
	}
	if x <= -512 {
		return 5
	}
	// Linear region: sigmoid ≈ 0.5 + x/4 in the middle
	return uint8(128 + int32(x)/4)
}

// clamp32 clamps an int32 to [lo, hi].
func clamp32(v, lo, hi int32) int32 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
