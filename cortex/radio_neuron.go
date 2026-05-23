package cortex

// RadioNeuron is a single neuron packed into 4 bytes (one RGBA32 pixel).
//
// Layout:
//
//	R (byte 0) = FreqListen  — frequency this neuron listens on (0-255)
//	G (byte 1) = Phase       — oscillation phase (0-255 maps to 0°-360°)
//	B (byte 2) = Amplitude   — signal strength / confidence (0-255)
//	A (byte 3) = Routing     — bits 0-5: FreqEmit, bit 6: Inhibitory, bit 7: Refractory
//
// Two neurons with the same FreqListen are implicitly connected — no synapse list needed.
// Learning = adjusting FreqListen (re-tune) and Amplitude (strengthen/weaken).
// Phase determines WHEN in the cycle the neuron responds (temporal ordering).
type RadioNeuron uint32

// PackRadioNeuron creates a neuron from its RGBA32 components.
func PackRadioNeuron(freqListen, phase, amplitude, freqEmit uint8, inhibitory bool) RadioNeuron {
	a := freqEmit & 0x3F
	if inhibitory {
		a |= 0x40
	}
	return RadioNeuron(uint32(freqListen) | uint32(phase)<<8 | uint32(amplitude)<<16 | uint32(a)<<24)
}

// FreqListen returns the frequency this neuron listens on (byte R).
func (n RadioNeuron) FreqListen() uint8 { return uint8(n) }

// Phase returns the oscillation phase (byte G). 0-255 maps to 0°-360°.
func (n RadioNeuron) Phase() uint8 { return uint8(n >> 8) }

// Amplitude returns the signal strength (byte B). 0 = dead, 255 = maximum confidence.
func (n RadioNeuron) Amplitude() uint8 { return uint8(n >> 16) }

// FreqEmit returns the frequency this neuron emits on (bits 0-5 of byte A).
func (n RadioNeuron) FreqEmit() uint8 { return uint8(n>>24) & 0x3F }

// IsInhibitory returns true if this neuron suppresses signal (bit 6 of byte A).
func (n RadioNeuron) IsInhibitory() bool { return (n>>24)&0x40 != 0 }

// IsRefractory returns true if this neuron just fired and is cooling down (bit 7 of byte A).
func (n RadioNeuron) IsRefractory() bool { return (n>>24)&0x80 != 0 }

// SetFreqListen changes the listening frequency (byte R).
// This is how LEARNING works: re-tuning a neuron to a different "channel".
func (n *RadioNeuron) SetFreqListen(f uint8) {
	*n = RadioNeuron((uint32(*n) & 0xFFFFFF00) | uint32(f))
}

// SetPhase changes the oscillation phase (byte G).
func (n *RadioNeuron) SetPhase(p uint8) {
	*n = RadioNeuron((uint32(*n) & 0xFFFF00FF) | uint32(p)<<8)
}

// SetAmplitude changes the signal strength (byte B).
func (n *RadioNeuron) SetAmplitude(a uint8) {
	*n = RadioNeuron((uint32(*n) & 0xFF00FFFF) | uint32(a)<<16)
}

// SetRefractory sets or clears the refractory flag (bit 7 of byte A).
func (n *RadioNeuron) SetRefractory(r bool) {
	if r {
		*n = RadioNeuron(uint32(*n) | (0x80 << 24))
	} else {
		*n = RadioNeuron(uint32(*n) & ^(uint32(0x80) << 24))
	}
}

// AdvancePhase moves the phase forward by the neuron's listen frequency.
// Phase wraps naturally at 256 (uint8 overflow). Higher frequency = faster oscillation.
func (n *RadioNeuron) AdvancePhase() {
	newPhase := n.Phase() + n.FreqListen()
	n.SetPhase(newPhase) // wraps at 256 via uint8
}

// IsAlive returns true if the neuron has non-zero amplitude.
func (n RadioNeuron) IsAlive() bool { return n.Amplitude() > 0 }

// Resonance computes the phase match between a neuron and a bus signal.
// Uses cos256 lookup table from quantum_tile.go.
// Returns [-127, +127]: +127 = perfect resonance, -127 = anti-resonance, 0 = neutral.
func Resonance(neuronPhase, busPhase uint8) int8 {
	return cos256[neuronPhase-busPhase]
}
