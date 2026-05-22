package cortex

// ─────────────────────────────────────────────────────────────────────
// neuron.go — Leaky Integrate-and-Fire (LIF) Spiking Neuron
// ─────────────────────────────────────────────────────────────────────
//
// Each neuron is encoded as a single RGBA32 pixel (4 bytes), enabling
// GPU-friendly storage and bitwise-only processing:
//
//   R (uint8) = Membrane potential (voltage, 0–255)
//               The neuron's current charge. Resting state is 0.
//               Input current accumulates here until threshold.
//
//   G (uint8) = Firing threshold (0–255)
//               When membrane potential ≥ threshold, the neuron fires
//               an action potential (spike). Higher = harder to fire.
//
//   B (uint8) = High nibble [7:4]: leak factor (0–15)
//                 How much voltage decays per timestep.
//                 Models the passive ion channel leakage of a real neuron.
//               Low nibble [3:0]: refractory cooldown counter (0–15)
//                 After firing, the neuron enters a refractory period
//                 where it cannot fire again. Decrements each step.
//                 Models the absolute refractory period in biology.
//
//   A (uint8) = High nibble [7:4]: neuron type (0–15)
//                 Functional classification (excitatory, inhibitory, etc.)
//               Low nibble [3:0]: region ID (0–15)
//                 Which brain region this neuron belongs to.
//
// Neuroscience background:
//   The LIF model is the simplest biologically plausible spiking neuron.
//   It captures the key dynamics: charge accumulation from synaptic input,
//   passive leak through the membrane, threshold-triggered spike, and
//   post-spike refractory silence. Despite its simplicity, networks of
//   LIF neurons can produce rich temporal coding and oscillatory dynamics.

// ─────────────────────────────────────────────────────────────────────
// Neuron Type Constants
// ─────────────────────────────────────────────────────────────────────
//
// These classify the functional role of each neuron in the network.
// In biological brains, different neuron types have different
// morphologies and neurotransmitter profiles. Here we encode the
// functional analog in 4 bits.

const (
	// NTypeExcitatory releases "glutamate-like" signals that increase
	// the membrane potential of downstream neurons. ~80% of cortical neurons.
	NTypeExcitatory uint8 = 0

	// NTypeInhibitory releases "GABA-like" signals that decrease
	// the membrane potential of downstream neurons. ~20% of cortical neurons.
	// Critical for preventing runaway excitation (epilepsy prevention).
	NTypeInhibitory uint8 = 1
)

const (
	// RegionWorkspace is the global workspace for conscious integration.
	// Inspired by Global Workspace Theory (Baars, 1988) — a "blackboard"
	// where different modules broadcast and compete for attention.
	RegionWorkspace uint8 = 6
)

// ─────────────────────────────────────────────────────────────────────
// SpikingNeuron — The RGBA32 LIF Neuron
// ─────────────────────────────────────────────────────────────────────

// SpikingNeuron represents a single Leaky Integrate-and-Fire neuron
// packed into exactly 4 bytes (one RGBA32 pixel).
//
// The LIF model is governed by the discrete-time update rule:
//
//	V(t+1) = V(t) - leak + I(t)
//	if V(t+1) >= threshold: SPIKE, then V = 0, refractory = max
//
// All arithmetic is performed with uint8 integers. No floats.
type SpikingNeuron struct {
	R uint8 // Membrane potential (voltage)
	G uint8 // Firing threshold
	B uint8 // High nibble: leak factor, Low nibble: refractory counter
	A uint8 // High nibble: neuron type, Low nibble: region ID
}

// NewSpikingNeuron creates a neuron with the given type, region, and
// firing threshold. The neuron starts at resting potential (V=0) with
// a default leak factor of 1 and no refractory cooldown.
//
// Parameters:
//   - neuronType: functional classification (NeuronExcitatory, etc.)
//   - regionID:   brain region assignment (RegionThalamus, etc.)
//   - threshold:  firing threshold (higher = harder to activate)
func NewSpikingNeuron(neuronType, regionID, threshold uint8) SpikingNeuron {
	// Pack type into high nibble, region into low nibble of A.
	a := (neuronType & 0x0F) << 4 | (regionID & 0x0F)

	// Default leak factor = 1 (gentle passive decay), packed into
	// high nibble of B. Refractory counter starts at 0 (ready to fire).
	b := uint8(1) << 4 // leak=1, refractory=0

	return SpikingNeuron{
		R: 0,         // Resting potential
		G: threshold, // Firing threshold
		B: b,
		A: a,
	}
}

// ─────────────────────────────────────────────────────────────────────
// Accessors — Read packed bit fields
// ─────────────────────────────────────────────────────────────────────

// Voltage returns the current membrane potential.
func (n SpikingNeuron) Voltage() uint8 {
	return n.R
}

// Threshold returns the firing threshold.
func (n SpikingNeuron) Threshold() uint8 {
	return n.G
}

// LeakFactor returns the leak rate (0–15) from B's high nibble.
// A leak of 0 means no decay (perfect integrator).
// A leak of 15 means very fast decay (hard to charge).
func (n SpikingNeuron) LeakFactor() uint8 {
	return n.B >> 4
}

// SetLeakFactor sets the leak rate (0–15).
func (n *SpikingNeuron) SetLeakFactor(leak uint8) {
	n.B = (leak & 0x0F) << 4 | (n.B & 0x0F)
}

// RefractoryCounter returns the remaining refractory ticks (0–15).
// While > 0, the neuron cannot fire. Models the biological absolute
// refractory period when Na+ channels are inactivated.
func (n SpikingNeuron) RefractoryCounter() uint8 {
	return n.B & 0x0F
}

// SetRefractoryCounter sets the refractory cooldown (0–15).
func (n *SpikingNeuron) SetRefractoryCounter(count uint8) {
	n.B = (n.B & 0xF0) | (count & 0x0F)
}

// Type returns the neuron type (0–15) from A's high nibble.
func (n SpikingNeuron) Type() uint8 {
	return n.A >> 4
}

// RegionID returns the region assignment (0–15) from A's low nibble.
func (n SpikingNeuron) RegionID() uint8 {
	return n.A & 0x0F
}

// ─────────────────────────────────────────────────────────────────────
// Type helpers
// ─────────────────────────────────────────────────────────────────────

// IsExcitatory returns true if this neuron sends excitatory (+) signals.
// Excitatory neurons increase the probability of downstream firing,
// analogous to glutamatergic pyramidal cells in the cortex.
func (n SpikingNeuron) IsExcitatory() bool {
	return n.Type() == NTypeExcitatory
}

// IsInhibitory returns true if this neuron sends inhibitory (-) signals.
// Inhibitory neurons decrease the probability of downstream firing,
// analogous to GABAergic interneurons that maintain E/I balance.
func (n SpikingNeuron) IsInhibitory() bool {
	return n.Type() == NTypeInhibitory
}

// ─────────────────────────────────────────────────────────────────────
// Step — The core LIF dynamics (ALL INTEGER ARITHMETIC)
// ─────────────────────────────────────────────────────────────────────

// Step performs one discrete timestep of the LIF neuron model.
//
// The biological process modeled:
//  1. LEAK: ions passively diffuse back across the membrane, reducing
//     the membrane potential toward resting state (0). This models
//     the membrane's RC circuit time constant.
//  2. INTEGRATE: incoming synaptic current (inputCurrent) is added
//     to the membrane potential. This models post-synaptic potentials
//     (EPSPs and IPSPs) arriving at the soma.
//  3. FIRE: if the membrane potential reaches the threshold, the
//     neuron emits an action potential (spike). The voltage is then
//     reset to 0 and the refractory counter is set to maximum,
//     preventing immediate re-firing (absolute refractory period).
//
// Returns true if the neuron fired a spike this timestep.
//
// All arithmetic uses uint8. No floating-point operations.
func (n *SpikingNeuron) Step(inputCurrent, refractoryMax, homeoInc, homeoFloor, homeoDecay uint8) bool {
	// If in refractory period, decrement counter and skip processing.
	// In biology, voltage-gated Na+ channels need time to de-inactivate
	// before the neuron can fire again.
	refractory := n.RefractoryCounter()
	if refractory > 0 {
		n.SetRefractoryCounter(refractory - 1)
		// Voltage stays at 0 during refractory period.
		n.R = 0
		return false
	}

	// Phase 1: LEAK — passive decay toward resting potential.
	// Subtract the leak factor from the membrane potential.
	// Clamp at 0 to prevent underflow (uint8 wrapping).
	leak := n.LeakFactor()
	if n.R > leak {
		n.R -= leak
	} else {
		n.R = 0
	}

	// Phase 2: INTEGRATE — add synaptic input current.
	// Clamp at 255 to prevent overflow.
	sum := uint16(n.R) + uint16(inputCurrent)
	if sum > 255 {
		n.R = 255
	} else {
		n.R = uint8(sum)
	}

	// Phase 3: FIRE — threshold comparison and spike generation.
	// In biology, when the membrane potential reaches approximately
	// -55mV (threshold), voltage-gated Na+ channels open in a
	// positive feedback loop, producing the all-or-nothing action
	// potential (~+40mV spike).
	if n.R >= n.G {
		// Spike! Reset membrane potential to resting state.
		n.R = 0

		// Enter refractory period (configurable duration, max 15 due to nibble packing).
		n.SetRefractoryCounter(refractoryMax & 0x0F)

		// Dynamic Homeostatic Threshold Plasticity: increase threshold on spike to damp excitability.
		newG := uint16(n.G) + uint16(homeoInc)
		if newG > 255 {
			n.G = 255
		} else {
			n.G = uint8(newG)
		}

		return true // Action potential emitted.
	}

	// Dynamic Homeostatic Threshold Plasticity: decay threshold toward baseline floor when silent.
	if n.G > homeoFloor && homeoDecay > 0 {
		if n.G > homeoFloor+homeoDecay {
			n.G -= homeoDecay
		} else {
			n.G = homeoFloor
		}
	}

	return false // Sub-threshold — no spike.
}



// ─────────────────────────────────────────────────────────────────────
// Serialization — RGBA32 ↔ uint32
// ─────────────────────────────────────────────────────────────────────

// PackRGBA32 serializes the neuron into a single uint32.
// Layout: [R:31–24 | G:23–16 | B:15–8 | A:7–0]
//
// This format is directly loadable as a GPU texture pixel,
// enabling massively parallel neural simulation on the GPU.
func (n SpikingNeuron) PackRGBA32() uint32 {
	return uint32(n.R)<<24 | uint32(n.G)<<16 | uint32(n.B)<<8 | uint32(n.A)
}

// UnpackRGBA32 deserializes a uint32 back into a SpikingNeuron.
func UnpackRGBA32Neuron(v uint32) SpikingNeuron {
	return SpikingNeuron{
		R: uint8(v >> 24),
		G: uint8(v >> 16),
		B: uint8(v >> 8),
		A: uint8(v),
	}
}

// ─────────────────────────────────────────────────────────────────────
// NeuronPool — Manages a population of spiking neurons
// ─────────────────────────────────────────────────────────────────────
//
// In biological neural networks, neurons operate as populations.
// The pool provides batch operations over a slice of neurons,
// enabling efficient simulation of neural circuits.

// NeuronPool manages a collection of SpikingNeuron instances.
// It provides batch simulation, neuron access, and spike counting.
type NeuronPool struct {
	Neurons []SpikingNeuron

	// Configurable neuron dynamics parameters (set from Config).
	RefractoryPeriod    uint8
	HomeoSpikeIncrement uint8
	HomeoBaselineFloor  uint8
	HomeoDecayRate      uint8
}

// NewNeuronPool creates an empty pool with pre-allocated capacity.
// Uses default neuron dynamics parameters; call SetDynamics() to override.
func NewNeuronPool(capacity int) *NeuronPool {
	return &NeuronPool{
		Neurons:             make([]SpikingNeuron, 0, capacity),
		RefractoryPeriod:    15,
		HomeoSpikeIncrement: 8,
		HomeoBaselineFloor:  80,
		HomeoDecayRate:      1,
	}
}

// SetDynamics configures the neuron dynamics parameters from a Config.
func (p *NeuronPool) SetDynamics(cfg Config) {
	p.RefractoryPeriod = cfg.NeuronRefractoryPeriod
	p.HomeoSpikeIncrement = cfg.HomeostaticSpikeIncrement
	p.HomeoBaselineFloor = cfg.HomeostaticBaselineFloor
	p.HomeoDecayRate = cfg.HomeostaticDecayRate
}

// AddNeuron appends a neuron to the pool and returns its index.
// The index serves as the neuron's address in the network.
func (p *NeuronPool) AddNeuron(n SpikingNeuron) int {
	idx := len(p.Neurons)
	p.Neurons = append(p.Neurons, n)
	return idx
}

// GetNeuron returns a pointer to the neuron at the given index.
// Returns nil if the index is out of bounds.
func (p *NeuronPool) GetNeuron(index int) *SpikingNeuron {
	if index < 0 || index >= len(p.Neurons) {
		return nil
	}
	return &p.Neurons[index]
}

// Size returns the number of neurons in the pool.
func (p *NeuronPool) Size() int {
	return len(p.Neurons)
}

// StepAll advances every neuron in the pool by one timestep.
//
// Parameters:
//   - inputs: a slice of input currents, one per neuron. If the slice
//     is shorter than the pool, missing entries are treated as 0
//     (no synaptic input). This models a quiescent neuron receiving
//     no afferent stimulation.
//
// Returns a boolean slice indicating which neurons fired a spike.
// This spike vector is the primary output used by the synapse layer
// to propagate signals and apply STDP learning.
// StepAllInto steps all neurons in the pool, writing the spike outputs into the pre-allocated spikes slice.
func (p *NeuronPool) StepAllInto(inputs []uint8, spikes []bool) {
	n := len(p.Neurons)
	limit := len(spikes)
	if n < limit {
		limit = n
	}
	for i := 0; i < limit; i++ {
		var current uint8
		if i < len(inputs) {
			current = inputs[i]
		}
		spikes[i] = p.Neurons[i].Step(current, p.RefractoryPeriod, p.HomeoSpikeIncrement, p.HomeoBaselineFloor, p.HomeoDecayRate)
	}
}

// StepAll steps all neurons in the pool and returns a newly allocated boolean slice of spike states.
// Deprecated: Use StepAllInto to avoid transient heap allocations in the fast path.
func (p *NeuronPool) StepAll(inputs []uint8) []bool {
	n := len(p.Neurons)
	spikes := make([]bool, n)
	p.StepAllInto(inputs, spikes)
	return spikes
}


