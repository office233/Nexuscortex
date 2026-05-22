package cortex

import (
	"encoding/binary"
	"fmt"
	"math/rand"
	"os"
)

// ─────────────────────────────────────────────────────────────────────
// Network — Spiking Neural Network (Reservoir Computing)
// ─────────────────────────────────────────────────────────────────────
//
// The Network is a reservoir computing engine that simulates a
// biologically-inspired spiking neural network. Neurons fire (spike)
// when their membrane potential exceeds a threshold, and connections
// between neurons are strengthened or weakened via Spike-Timing
// Dependent Plasticity (STDP).
//
// Architecture:
//   - 80% excitatory / 20% inhibitory neurons (like real cortex)
//   - Random sparse connectivity (configurable density)
//   - STDP learning on every tick
//   - Spike history for temporal pattern analysis
//
// All arithmetic is integer-only (uint8/uint16/uint32/uint64).
// Serialization uses the RGBA32 pixel format consistent with .cortex.
//
// Neuron types used from neuron.go:
//   - SpikingNeuron: the LIF neuron (RGBA32 packed)
//   - NeuronPool:    manages a population of SpikingNeurons
//
// Synapse types defined here (will migrate to synapse.go when it
// lands with StdpSynapse and SynapseBundle):
//   - ResSynapse: STDP-capable weighted connection

// ─────────────────────────────────────────────────────────────────────
// Reservoir Synapse — STDP-capable connection
// ─────────────────────────────────────────────────────────────────────
//
// Each synapse is encoded as two RGBA32 pixels (8 bytes total):
//
//   Pixel 1: [Source_Hi, Source_Lo, Target_Hi, Target_Lo]
//   Pixel 2: [Weight, PreTrace, PostTrace, Delay]
//
// This gives us:
//   - 65,535 source/target neuron IDs (uint16)
//   - 0-255 weight resolution
//   - 0-255 pre/post STDP traces
//   - 0-255 axonal delay
//
// When synapse.go lands with StdpSynapse and SynapseBundle, this
// type can be replaced or bridged. The RGBA32 encoding is compatible.

// ResSynapse is a synapse with STDP traces for the reservoir network.
type ResSynapse struct {
	Source    uint16 // Pre-synaptic neuron index
	Target    uint16 // Post-synaptic neuron index
	Weight    uint8  // Connection strength (0=pruned, 1-255=active)
	PreTrace  uint8  // STDP pre-synaptic eligibility trace
	PostTrace uint8  // STDP post-synaptic eligibility trace
	Delay     uint8  // Axonal propagation delay (ticks)
}

// STDP learning parameters (integer-scaled).
const (
	StdpPotentiate uint8 = 10  // Amount to strengthen on pre→post
	StdpDepress    uint8 = 6   // Amount to weaken on post→pre
	StdpTraceDecay uint8 = 20  // Trace decay per tick (out of 255)
	StdpMaxWeight  uint8 = 250 // Maximum weight cap
)

// PackResSynapseRGBA32 encodes this synapse as two uint32 pixels.
func (s *ResSynapse) PackResSynapseRGBA32() (uint32, uint32) {
	p1 := uint32(s.Source>>8)<<24 | uint32(s.Source&0xFF)<<16 |
		uint32(s.Target>>8)<<8 | uint32(s.Target&0xFF)
	p2 := uint32(s.Weight)<<24 | uint32(s.PreTrace)<<16 |
		uint32(s.PostTrace)<<8 | uint32(s.Delay)
	return p1, p2
}

// UnpackResSynapse decodes two uint32 pixels into a ResSynapse.
func UnpackResSynapse(p1, p2 uint32) ResSynapse {
	return ResSynapse{
		Source:    uint16(p1>>24)<<8 | uint16((p1>>16)&0xFF),
		Target:    uint16((p1>>8)&0xFF)<<8 | uint16(p1&0xFF),
		Weight:    uint8(p2 >> 24),
		PreTrace:  uint8(p2 >> 16),
		PostTrace: uint8(p2 >> 8),
		Delay:     uint8(p2),
	}
}

// ─────────────────────────────────────────────────────────────────────
// Network — The Reservoir
// ─────────────────────────────────────────────────────────────────────

// Network is a spiking neural network reservoir with STDP learning.
// It manages a NeuronPool of LIF neurons connected by plastic
// ResSynapses.
type Network struct {
	Neurons      *NeuronPool  // All neurons in the reservoir
	Synapses     []ResSynapse // All synaptic connections
	SpikeHistory [][]bool     // Last N timesteps of spike states
	Step         uint64       // Current simulation timestep
	Rng          *rand.Rand   // Deterministic random source
	Config       Config       // Runtime configuration (STDP params, neuron dynamics, etc.)

	// Transient fields (not serialized, used for reservoir stabilizers)
	ColSpikeTraces         []uint32
	LastTotalSpikes        int
	ChaosCounter           int
	MetacognitiveDampening uint16

	tickInputCurrents []uint16 // Pre-allocated transient slice for input currents (LIF)
	tickInputs        []uint8  // Pre-allocated transient slice for step inputs (LIF)
	tickSpikes        []bool   // Pre-allocated transient slice for current tick's spikes

	OutSynapses [][]int // Index of outgoing active synapses per source neuron
	InSynapses  [][]int // Index of incoming active synapses per target neuron
	PreTraces   []uint8 // Pre-synaptic eligibility traces per neuron
	PostTraces  []uint8 // Post-synaptic eligibility traces per neuron

	historyLen int // Maximum spike history length to retain
}

// NetworkStats holds statistics about the network's current state.
type NetworkStats struct {
	TotalNeurons    int    // Total neuron count
	ExcitatoryCount int    // Number of excitatory neurons
	InhibitoryCount int    // Number of inhibitory neurons
	TotalSynapses   int    // Total synapse count
	ActiveSynapses  int    // Synapses with weight > 0
	SpikesThisTick  int    // Neurons that spiked on current tick
	AvgWeight       uint16 // Average synapse weight (×256 for precision)
	MaxWeight       uint8  // Maximum synapse weight
	MinWeight       uint8  // Minimum non-zero synapse weight
	StepCount       uint64 // Current simulation step
	MemoryBytes     int    // Estimated memory usage
}

// DefaultHistoryLen is the default number of ticks of spike history
// retained for temporal pattern analysis and readout.
const DefaultHistoryLen = 100

// ─────────────────────────────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────────────────────────────

// NewNetwork creates a reservoir of 'size' spiking neurons connected
// with the given connectivity density (0.0–1.0).
//
// Neuron thresholds are randomized in the config-defined range.
// Excitatory ratio is configured via Config — matching the
// ratio observed in mammalian cortex.
//
// Synaptic weights are randomly initialized in configured range.
// Connectivity of 0.1 means ~10% of all possible pairs are connected.
func NewNetwork(size int, connectivity float64, cfg Config, rng *rand.Rand) *Network {
	if size <= 0 {
		size = 1
	}
	if size > 65535 {
		// ResSynapse Source/Target are uint16; exceeding 65535 silently truncates IDs.
		panic("network size exceeds uint16 neuron ID limit (65535)")
	}
	if connectivity < 0 {
		connectivity = 0
	}
	if connectivity > 1 {
		connectivity = 1
	}

	pool := NewNeuronPool(size)
	pool.SetDynamics(cfg)

	synCap := size * size / 10
	if synCap > 10_000_000 {
		synCap = 10_000_000 // Cap pre-allocation to prevent OOM
	}

	net := &Network{
		Neurons:      pool,
		Synapses:     make([]ResSynapse, 0, synCap),
		SpikeHistory: make([][]bool, 0, DefaultHistoryLen),
		Step:         0,
		Rng:          rng,
		Config:       cfg,
		historyLen:   DefaultHistoryLen,
	}

	// Initialize neurons with random thresholds.
	// Excitatory ratio is determined dynamically.
	excitatoryCutoff := (size * cfg.NetworkExcitatoryRatioNumerator) / 5
	threshRange := int(cfg.NeuronMaxThreshold) - int(cfg.NeuronMinThreshold) + 1
	leakRange := int(cfg.NeuronMaxLeak) - int(cfg.NeuronMinLeak) + 1

	for i := 0; i < size; i++ {
		// Threshold in [Min, Max] — uniform random via integer math.
		threshold := uint8(int(cfg.NeuronMinThreshold) + rng.Intn(threshRange))

		// Determine neuron type: excitatory or inhibitory.
		var neuronType uint8
		if i < excitatoryCutoff {
			neuronType = NTypeExcitatory
		} else {
			neuronType = NTypeInhibitory
		}

		// Assign to reservoir region (RegionWorkspace).
		neuron := NewSpikingNeuron(neuronType, RegionWorkspace, threshold)

		// Randomize leak factor in [Min, Max] for diversity.
		neuron.SetLeakFactor(uint8(int(cfg.NeuronMinLeak) + rng.Intn(leakRange)))

		pool.AddNeuron(neuron)
	}

	// Wire random connections.
	// We iterate all possible (i, j) pairs where i ≠ j and roll
	// against connectivity. This is O(n²) but only runs once at init.
	//
	// To avoid float comparison, we scale connectivity to [0, 65535].
	connectThreshold := uint16(connectivity * 65535)
	weightRange := int(cfg.SynapseMaxWeight) - int(cfg.SynapseMinWeight) + 1
	delayRange := int(cfg.SynapseMaxDelay) - int(cfg.SynapseMinDelay) + 1

	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			if i == j {
				continue // no self-connections
			}
			roll := uint16(rng.Intn(65536))
			if roll < connectThreshold {
				weight := uint8(int(cfg.SynapseMinWeight) + rng.Intn(weightRange))
				delay := uint8(int(cfg.SynapseMinDelay) + rng.Intn(delayRange))
				syn := ResSynapse{
					Source: uint16(i),
					Target: uint16(j),
					Weight: weight,
					Delay:  delay,
				}
				net.Synapses = append(net.Synapses, syn)
			}
		}
	}

	net.PreTraces = make([]uint8, size)
	net.PostTraces = make([]uint8, size)
	net.tickInputCurrents = make([]uint16, size)
	net.tickInputs = make([]uint8, size)
	net.tickSpikes = make([]bool, size)
	net.rebuildSynapseIndices()
	return net
}

// rebuildSynapseIndices clears and populates the transient OutSynapses and InSynapses indices.
// It maps each neuron's ID to slices of indices pointing into n.Synapses.
func (n *Network) rebuildSynapseIndices() {
	size := n.Neurons.Size()
	n.OutSynapses = make([][]int, size)
	n.InSynapses = make([][]int, size)
	for i := range n.Synapses {
		syn := &n.Synapses[i]
		if syn.Weight > 0 {
			srcIdx := int(syn.Source)
			tgtIdx := int(syn.Target)
			if srcIdx >= 0 && srcIdx < size {
				n.OutSynapses[srcIdx] = append(n.OutSynapses[srcIdx], i)
			}
			if tgtIdx >= 0 && tgtIdx < size {
				n.InSynapses[tgtIdx] = append(n.InSynapses[tgtIdx], i)
			}
		}
	}
}

// ─────────────────────────────────────────────────────────────────────
// Tick — One Simulation Timestep
// ─────────────────────────────────────────────────────────────────────

// Tick advances the reservoir by one timestep. The simulation loop:
//
//  1. Apply external inputs to specified neurons.
//  2. Propagate spikes from previous tick through synapses.
//  3. Step all neurons (leak, integrate, fire).
//  4. Apply STDP learning based on spike timing.
//  5. Decay all STDP eligibility traces.
//  6. Record spike pattern into history ring buffer.
//
// Returns a boolean slice indicating which neurons spiked this tick.
func (n *Network) Tick(externalInputs []uint8) []bool {
	size := n.Neurons.Size()

	// Lazy initialization check
	if len(n.tickInputCurrents) != size {
		n.tickInputCurrents = make([]uint16, size)
	}
	if len(n.tickInputs) != size {
		n.tickInputs = make([]uint8, size)
	}
	if len(n.tickSpikes) != size {
		n.tickSpikes = make([]bool, size)
	}

	// Fast clear inputCurrents (this uses Go compiler optimized memclr)
	for i := range n.tickInputCurrents {
		n.tickInputCurrents[i] = 0
	}

	// ── Columnar Activations & Dominance ─────────────────────────
	colSize := 100
	numCols := (size + colSize - 1) / colSize
	if len(n.ColSpikeTraces) != numCols {
		n.ColSpikeTraces = make([]uint32, numCols)
	}

	// Decay traces (low-pass filter)
	for i := range n.ColSpikeTraces {
		n.ColSpikeTraces[i] = (n.ColSpikeTraces[i] * 7) / 8
	}

	// Update traces with previous spikes
	if len(n.SpikeHistory) > 0 {
		prevSpikes := n.SpikeHistory[len(n.SpikeHistory)-1]
		for i, spiked := range prevSpikes {
			if spiked && i < size {
				col := i / colSize
				if col < len(n.ColSpikeTraces) {
					n.ColSpikeTraces[col] += 256
				}
			}
		}
	}

	// Find dominant column
	maxTrace := uint32(0)
	maxCol := -1
	for c, trace := range n.ColSpikeTraces {
		if trace > maxTrace {
			maxTrace = trace
			maxCol = c
		}
	}

	// ── Step 1: External inputs ──────────────────────────────────
	if len(externalInputs) > 0 {
		for idx, current := range externalInputs {
			if current > 0 && idx >= 0 && idx < size {
				n.tickInputCurrents[idx] += uint16(current)
			}
		}
	}

	// ── Step 2: Propagate spikes through synapses ────────────────
	// For each synapse whose source neuron spiked on the previous
	// tick, deliver current to the target neuron using the OutSynapses index.
	if len(n.SpikeHistory) > 0 {
		lastHistory := n.SpikeHistory[len(n.SpikeHistory)-1]
		for srcIdx, prevSpiked := range lastHistory {
			if !prevSpiked || srcIdx >= len(n.OutSynapses) {
				continue
			}
			src := n.Neurons.GetNeuron(srcIdx)
			if src == nil {
				continue
			}

			// Iterate only through outgoing active synapses of this pre-synaptic neuron
			for _, synIdx := range n.OutSynapses[srcIdx] {
				syn := &n.Synapses[synIdx]
				if syn.Weight == 0 {
					continue
				}
				tgtIdx := int(syn.Target)
				if tgtIdx >= size {
					continue
				}

				// Excitatory source → add current. Inhibitory → subtract.
				if src.IsExcitatory() {
					n.tickInputCurrents[tgtIdx] += uint16(syn.Weight)
				} else {
					// Inhibitory: reduce accumulated input. Floor at 0.
					if n.tickInputCurrents[tgtIdx] >= uint16(syn.Weight) {
						n.tickInputCurrents[tgtIdx] -= uint16(syn.Weight)
					} else {
						n.tickInputCurrents[tgtIdx] = 0
					}
				}
			}
		}
	}

	// ── WTA Lateral Columnar Inhibition ──────────────────────────
	if maxCol != -1 && maxTrace > 50 {
		inhibition := uint16(maxTrace / 32)
		if inhibition > 100 {
			inhibition = 100
		}
		for i := 0; i < size; i++ {
			c := i / colSize
			if c != maxCol {
				if n.tickInputCurrents[i] >= inhibition {
					n.tickInputCurrents[i] -= inhibition
				} else {
					n.tickInputCurrents[i] = 0
				}
			}
		}
	}

	// ── Metacognitive Dampening ──────────────────────────────────
	if n.MetacognitiveDampening > 0 {
		n.MetacognitiveDampening--
		bias := uint16(15)
		for i := 0; i < size; i++ {
			if n.tickInputCurrents[i] >= bias {
				n.tickInputCurrents[i] -= bias
			} else {
				n.tickInputCurrents[i] = 0
			}
		}
	}

	// ── Step 3: Step all neurons via NeuronPool ──────────────────
	// Clamp accumulated inputs to uint8 range for the pool.
	for i := range n.tickInputs {
		if n.tickInputCurrents[i] > 255 {
			n.tickInputs[i] = 255
		} else {
			n.tickInputs[i] = uint8(n.tickInputCurrents[i])
		}
	}
	n.Neurons.StepAllInto(n.tickInputs, n.tickSpikes)
	spikes := n.tickSpikes

	// ── Noradrenergic Metacognitive Reset Monitor ────────────────
	spikesCount := 0
	for _, spiked := range spikes {
		if spiked {
			spikesCount++
		}
	}

	chaosThreshPct := n.Config.NoradrenergicChaosThresholdPct
	if chaosThreshPct <= 0 {
		chaosThreshPct = 30
	}
	chaosCountLimit := n.Config.NoradrenergicChaosCounterLimit
	if chaosCountLimit <= 0 {
		chaosCountLimit = 8
	}
	cooldownPct := n.Config.NoradrenergicCooldownPct
	if cooldownPct <= 0 {
		cooldownPct = 5
	}

	highSpikesThreshold := size * chaosThreshPct / 100
	if spikesCount > highSpikesThreshold {
		n.ChaosCounter++
		if n.ChaosCounter > chaosCountLimit {
			// Trigger Noradrenergic Metacognitive Reset!
			threshBoost := n.Config.NoradrenergicThresholdBoost
			if threshBoost == 0 {
				threshBoost = 30
			}
			for i := 0; i < size; i++ {
				neuron := n.Neurons.GetNeuron(i)
				if neuron != nil {
					neuron.R = 0 // clear potentials
					newG := uint16(neuron.G) + uint16(threshBoost)
					if newG > 255 {
						neuron.G = 255
					} else {
						neuron.G = uint8(newG)
					}
				}
			}
			dampTicks := n.Config.NoradrenergicDampeningTicks
			if dampTicks == 0 {
				dampTicks = 10
			}
			n.MetacognitiveDampening = dampTicks
			n.ChaosCounter = 0
		}
	} else if spikesCount < size*cooldownPct/100 {
		if n.ChaosCounter > 0 {
			n.ChaosCounter--
		}
	}
	n.LastTotalSpikes = spikesCount

	// ── Step 4 & 5: Apply STDP learning and Decay ────────────────
	if len(n.PreTraces) != size || len(n.PostTraces) != size {
		n.PreTraces = make([]uint8, size)
		n.PostTraces = make([]uint8, size)
	}

	// 1. Decay all neuron-level traces (O(N))
	stdpDecay := n.Config.StdpTraceDecay
	if stdpDecay == 0 {
		stdpDecay = StdpTraceDecay // Fallback to const default
	}
	for i := 0; i < size; i++ {
		n.PreTraces[i] = uint8((uint16(n.PreTraces[i]) * uint16(255-stdpDecay)) >> 8)
		n.PostTraces[i] = uint8((uint16(n.PostTraces[i]) * uint16(255-stdpDecay)) >> 8)
	}

	// 2. Apply Potentiation (post-spikes)
	for tgtIdx, spiked := range spikes {
		if spiked {
			n.PostTraces[tgtIdx] = 255
			for _, synIdx := range n.InSynapses[tgtIdx] {
				syn := &n.Synapses[synIdx]
				if syn.Weight == 0 {
					continue
				}
				srcIdx := int(syn.Source)
				if n.PreTraces[srcIdx] > 0 {
					stdpPot := n.Config.StdpPotentiate
					if stdpPot == 0 {
						stdpPot = StdpPotentiate
					}
					stdpMax := n.Config.StdpMaxWeight
					if stdpMax == 0 {
						stdpMax = StdpMaxWeight
					}
					newW := uint16(syn.Weight) + uint16(stdpPot)
					if newW > uint16(stdpMax) {
						newW = uint16(stdpMax)
					}
					syn.Weight = uint8(newW)
				}
			}
		}
	}

	// 3. Apply Depression (pre-spikes)
	for srcIdx, spiked := range spikes {
		if spiked {
			n.PreTraces[srcIdx] = 255
			for _, synIdx := range n.OutSynapses[srcIdx] {
				syn := &n.Synapses[synIdx]
				if syn.Weight == 0 {
					continue
				}
				tgtIdx := int(syn.Target)
				if tgtIdx >= size {
					continue
				}
				if !spikes[tgtIdx] && n.PostTraces[tgtIdx] > 0 {
					stdpDep := n.Config.StdpDepress
					if stdpDep == 0 {
						stdpDep = StdpDepress
					}
					if syn.Weight > stdpDep {
						syn.Weight -= stdpDep
					} else {
						syn.Weight = 0
					}
				}
			}
		}
	}

	// ── Step 6: Record spike history ─────────────────────────────
	var record []bool
	if len(n.SpikeHistory) >= n.historyLen {
		// Ring buffer: shift left by one, recycle the oldest slot.
		record = n.SpikeHistory[0]
		copy(n.SpikeHistory, n.SpikeHistory[1:])
		n.SpikeHistory[len(n.SpikeHistory)-1] = record
	} else {
		record = make([]bool, size)
		n.SpikeHistory = append(n.SpikeHistory, record)
	}
	copy(record, spikes)

	n.Step++
	return record
}

// ─────────────────────────────────────────────────────────────────────
// RunTicks — Multiple Simulation Steps
// ─────────────────────────────────────────────────────────────────────

// RunTicks executes 'steps' simulation ticks with the same external
// input pattern applied on every tick. Returns the complete spike
// history for all ticks (steps × neurons).
func (n *Network) RunTicks(steps int, externalInputs []uint8) [][]bool {
	results := make([][]bool, 0, steps)
	for i := 0; i < steps; i++ {
		spikes := n.Tick(externalInputs)
		results = append(results, spikes)
	}
	return results
}

// ─────────────────────────────────────────────────────────────────────
// Input / Output — Interface to the outside world
// ─────────────────────────────────────────────────────────────────────

// InjectPattern injects the given current into specific neurons.
// This is used to feed SDR (Sparse Distributed Representation) input
// patterns directly into the reservoir's input layer.
func (n *Network) InjectPattern(neuronIDs []int, current uint8) {
	size := n.Neurons.Size()
	for _, id := range neuronIDs {
		if id < 0 || id >= size {
			continue
		}
		neuron := n.Neurons.GetNeuron(id)
		if neuron == nil {
			continue
		}
		// Directly add current to the neuron's membrane potential.
		sum := uint16(neuron.R) + uint16(current)
		if sum > 255 {
			sum = 255
		}
		neuron.R = uint8(sum)
	}
}

// ReadOutput reads the spike state of specific output neurons.
// Returns a boolean slice aligned with neuronIDs: true if that
// neuron spiked on the most recent tick. Checks the latest spike
// history entry.
func (n *Network) ReadOutput(neuronIDs []int) []bool {
	result := make([]bool, len(neuronIDs))
	if len(n.SpikeHistory) == 0 {
		return result
	}
	lastSpikes := n.SpikeHistory[len(n.SpikeHistory)-1]
	for i, id := range neuronIDs {
		if id >= 0 && id < len(lastSpikes) {
			result[i] = lastSpikes[id]
		}
	}
	return result
}

// GetActivationPattern returns the current spike state of all neurons.
// Returns the most recent tick's spike pattern from history.
func (n *Network) GetActivationPattern() []bool {
	if len(n.SpikeHistory) == 0 {
		return make([]bool, n.Neurons.Size())
	}
	src := n.SpikeHistory[len(n.SpikeHistory)-1]
	pattern := make([]bool, len(src))
	copy(pattern, src)
	return pattern
}

// GetActivationPatternReadOnly returns a direct reference to the latest spike pattern
// from history without allocating a new slice. Callers must treat it as read-only.
func (n *Network) GetActivationPatternReadOnly() []bool {
	if len(n.SpikeHistory) == 0 {
		return nil
	}
	return n.SpikeHistory[len(n.SpikeHistory)-1]
}

// ─────────────────────────────────────────────────────────────────────
// Stats — Network introspection
// ─────────────────────────────────────────────────────────────────────

// Stats returns statistics about the network's current state.
func (n *Network) Stats() NetworkStats {
	size := n.Neurons.Size()
	s := NetworkStats{
		TotalNeurons:  size,
		TotalSynapses: len(n.Synapses),
		StepCount:     n.Step,
		MinWeight:     255,
	}

	// Count excitatory / inhibitory neurons and current spikes.
	lastSpikes := n.GetActivationPattern()
	for i := 0; i < size; i++ {
		neuron := n.Neurons.GetNeuron(i)
		if neuron == nil {
			continue
		}
		if neuron.IsExcitatory() {
			s.ExcitatoryCount++
		} else {
			s.InhibitoryCount++
		}
		if i < len(lastSpikes) && lastSpikes[i] {
			s.SpikesThisTick++
		}
	}

	// Synapse statistics.
	var totalWeight uint64
	for i := range n.Synapses {
		w := n.Synapses[i].Weight
		if w > 0 {
			s.ActiveSynapses++
			totalWeight += uint64(w)
			if w > s.MaxWeight {
				s.MaxWeight = w
			}
			if w < s.MinWeight {
				s.MinWeight = w
			}
		}
	}
	if s.ActiveSynapses == 0 {
		s.MinWeight = 0
	}

	// Average weight scaled by 256 for integer precision.
	// To get the real average, divide AvgWeight by 256.
	if s.ActiveSynapses > 0 {
		s.AvgWeight = uint16((totalWeight * 256) / uint64(s.ActiveSynapses))
	}

	// Memory estimate: neurons (4 bytes each) + synapses (8 bytes each)
	// + spike history (1 byte per neuron per tick in Go booleans).
	s.MemoryBytes = size*4 + len(n.Synapses)*8
	for _, h := range n.SpikeHistory {
		s.MemoryBytes += len(h)
	}

	return s
}

// ─────────────────────────────────────────────────────────────────────
// Prune — Remove weak synapses
// ─────────────────────────────────────────────────────────────────────

// PruneWeak removes all synapses with weight at or below the given
// threshold. Returns the number of pruned synapses.
func (n *Network) PruneWeak(threshold uint8) int {
	alive := make([]ResSynapse, 0, len(n.Synapses))
	pruned := 0
	for i := range n.Synapses {
		if n.Synapses[i].Weight > threshold {
			alive = append(alive, n.Synapses[i])
		} else {
			pruned++
		}
	}
	n.Synapses = alive
	n.rebuildSynapseIndices()
	return pruned
}

// ─────────────────────────────────────────────────────────────────────
// Persistence — Save/Load as RGBA32 binary
// ─────────────────────────────────────────────────────────────────────
//
// File layout (.nxnet):
//
//   ┌──────────────────────────────────────────────┐
//   │  Magic: "NXRSNET1" (8 bytes)                 │
//   ├──────────────────────────────────────────────┤
//   │  NeuronCount:  uint32 LE                     │
//   ├──────────────────────────────────────────────┤
//   │  SynapseCount: uint32 LE                     │
//   ├──────────────────────────────────────────────┤
//   │  StepCount:    uint64 LE                     │
//   ├──────────────────────────────────────────────┤
//   │  HistoryLen:   uint32 LE                     │
//   ├──────────────────────────────────────────────┤
//   │  Reserved:     24 bytes                      │
//   ├──────────────────────────────────────────────┤
//   │  Neurons: N × 4 bytes (RGBA32 pixels)        │
//   ├──────────────────────────────────────────────┤
//   │  Synapses: M × 8 bytes (2 × RGBA32 pixels)  │
//   └──────────────────────────────────────────────┘

const (
	NetworkFileMagic  = "NXRSNET1"
	networkHeaderSize = 8 + 4 + 4 + 8 + 4 + 24 // = 52 bytes
)

// networkFileHeader is the binary header for .nxnet files.
type networkFileHeader struct {
	Magic        [8]byte
	NeuronCount  uint32
	SynapseCount uint32
	StepCount    uint64
	HistoryLen   uint32
	Reserved     [24]byte
}

// Save serializes the entire network to a binary file.
// Neurons are written as RGBA32 pixels via SpikingNeuron.PackRGBA32.
// Synapses are written as 2 × RGBA32 pixels.
func (n *Network) Save(path string) error {
	tmpPath := path + ".tmp"
	f, err := os.Create(tmpPath)
	if err != nil {
		return fmt.Errorf("create network file: %w", err)
	}

	size := n.Neurons.Size()

	// Write header.
	hdr := networkFileHeader{
		NeuronCount:  uint32(size),
		SynapseCount: uint32(len(n.Synapses)),
		StepCount:    n.Step,
		HistoryLen:   uint32(n.historyLen),
	}
	copy(hdr.Magic[:], NetworkFileMagic)

	if err := binary.Write(f, binary.LittleEndian, &hdr); err != nil {
		return fmt.Errorf("write header: %w", err)
	}

	// Write neurons as RGBA32 pixels (4 bytes each).
	for i := 0; i < size; i++ {
		neuron := n.Neurons.GetNeuron(i)
		if neuron == nil {
			continue
		}
		packed := neuron.PackRGBA32()
		if err := binary.Write(f, binary.LittleEndian, packed); err != nil {
			return fmt.Errorf("write neuron %d: %w", i, err)
		}
	}

	// Write synapses as 2 × RGBA32 pixels (8 bytes each).
	for i := range n.Synapses {
		syn := &n.Synapses[i]
		if int(syn.Source) < len(n.PreTraces) {
			syn.PreTrace = n.PreTraces[syn.Source]
		}
		if int(syn.Target) < len(n.PostTraces) {
			syn.PostTrace = n.PostTraces[syn.Target]
		}
		p1, p2 := syn.PackResSynapseRGBA32()
		if err := binary.Write(f, binary.LittleEndian, p1); err != nil {
			return fmt.Errorf("write synapse %d pixel 1: %w", i, err)
		}
		if err := binary.Write(f, binary.LittleEndian, p2); err != nil {
			return fmt.Errorf("write synapse %d pixel 2: %w", i, err)
		}
	}

	// Sync to disk before closing to ensure durability.
	if err := f.Sync(); err != nil {
		f.Close()
		os.Remove(tmpPath)
		return fmt.Errorf("sync network file: %w", err)
	}
	f.Close()

	// Atomic rename: old file is only replaced after new data is fully written.
	if err := os.Rename(tmpPath, path); err != nil {
		return fmt.Errorf("rename network file: %w", err)
	}
	return nil
}

// LoadNetwork deserializes a network from a binary file.
// Returns a fully reconstructed Network with all neurons, synapses,
// and simulation state restored.
func LoadNetwork(path string, cfg Config) (*Network, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open network file: %w", err)
	}
	defer f.Close()

	// Read header.
	var hdr networkFileHeader
	if err := binary.Read(f, binary.LittleEndian, &hdr); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}

	magic := string(hdr.Magic[:])
	if magic != NetworkFileMagic {
		return nil, fmt.Errorf("invalid magic: got %q, want %q", magic, NetworkFileMagic)
	}

	// Safety: reject absurdly large counts to prevent OOM from corrupted files.
	const MaxNeuronCount uint32 = 1_000_000      // 1M neurons × 4 bytes = 4MB
	const MaxResSynapseCount uint32 = 50_000_000 // 50M synapses × 8 bytes = 400MB
	if hdr.NeuronCount > MaxNeuronCount {
		return nil, fmt.Errorf("neuron count %d exceeds safety limit %d", hdr.NeuronCount, MaxNeuronCount)
	}
	if hdr.SynapseCount > MaxResSynapseCount {
		return nil, fmt.Errorf("synapse count %d exceeds safety limit %d", hdr.SynapseCount, MaxResSynapseCount)
	}
	const MaxHistoryLen uint32 = 1_000_000
	if hdr.HistoryLen > MaxHistoryLen {
		return nil, fmt.Errorf("history length %d exceeds safety limit %d", hdr.HistoryLen, MaxHistoryLen)
	}

	pool := NewNeuronPool(int(hdr.NeuronCount))
	pool.SetDynamics(cfg)

	net := &Network{
		Neurons:      pool,
		Synapses:     make([]ResSynapse, 0, hdr.SynapseCount),
		SpikeHistory: make([][]bool, 0, hdr.HistoryLen),
		Step:         hdr.StepCount,
		Rng:          rand.New(rand.NewSource(int64(hdr.StepCount))),
		Config:       cfg,
		historyLen:   int(hdr.HistoryLen),
	}

	// Read neurons from RGBA32 pixels.
	for i := uint32(0); i < hdr.NeuronCount; i++ {
		var packed uint32
		if err := binary.Read(f, binary.LittleEndian, &packed); err != nil {
			return nil, fmt.Errorf("read neuron %d (file truncated?): %w", i, err)
		}
		neuron := UnpackRGBA32Neuron(packed)
		pool.AddNeuron(neuron)
	}

	net.PreTraces = make([]uint8, hdr.NeuronCount)
	net.PostTraces = make([]uint8, hdr.NeuronCount)

	// Read synapses from paired RGBA32 pixels.
	for i := uint32(0); i < hdr.SynapseCount; i++ {
		var p1, p2 uint32
		if err := binary.Read(f, binary.LittleEndian, &p1); err != nil {
			return nil, fmt.Errorf("read synapse %d pixel 1 (file truncated?): %w", i, err)
		}
		if err := binary.Read(f, binary.LittleEndian, &p2); err != nil {
			return nil, fmt.Errorf("read synapse %d pixel 2: %w", i, err)
		}
		syn := UnpackResSynapse(p1, p2)
		net.Synapses = append(net.Synapses, syn)

		// Populate transient traces
		if int(syn.Source) < len(net.PreTraces) {
			if syn.PreTrace > net.PreTraces[syn.Source] {
				net.PreTraces[syn.Source] = syn.PreTrace
			}
		}
		if int(syn.Target) < len(net.PostTraces) {
			if syn.PostTrace > net.PostTraces[syn.Target] {
				net.PostTraces[syn.Target] = syn.PostTrace
			}
		}
	}

	net.tickInputCurrents = make([]uint16, hdr.NeuronCount)
	net.tickInputs = make([]uint8, hdr.NeuronCount)
	net.tickSpikes = make([]bool, hdr.NeuronCount)
	net.rebuildSynapseIndices()
	return net, nil
}
