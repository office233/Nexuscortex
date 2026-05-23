package cortex

// RadioNeuron is a single neuron packed into 4 bytes (one RGBA32 pixel).
//
// Layout (P0.2 FIX: full 0-255 emit frequency):
//
//	R (byte 0) = FreqListen  — frequency this neuron listens on (0-255)
//	G (byte 1) = Phase       — bits 0-6: oscillation phase (0-127 maps to 0°-360°)
//	                           bit 7: Inhibitory flag
//	B (byte 2) = Amplitude   — signal strength / confidence (0-255)
//	A (byte 3) = FreqEmit    — frequency this neuron emits on (0-255, FULL RANGE)
//
// Two neurons with the same FreqListen are implicitly connected — no synapse list needed.
// Learning = adjusting FreqListen (re-tune) and Amplitude (strengthen/weaken).
// Phase determines WHEN in the cycle the neuron responds (temporal ordering).
// Inhibitory flag indicates the neuron suppresses rather than excites.
type RadioNeuron uint32

// PackRadioNeuron creates a neuron from its RGBA32 components.
func PackRadioNeuron(freqListen, phase, amplitude, freqEmit uint8, inhibitory bool) RadioNeuron {
	g := phase & 0x7F // 7-bit phase
	if inhibitory {
		g |= 0x80 // bit 7 = inhibitory
	}
	return RadioNeuron(uint32(freqListen) | uint32(g)<<8 | uint32(amplitude)<<16 | uint32(freqEmit)<<24)
}

// FreqListen returns the frequency this neuron listens on (byte R).
func (n RadioNeuron) FreqListen() uint8 { return uint8(n) }

// Phase returns the oscillation phase (bits 0-6 of byte G). 0-127 maps to 0°-360°.
func (n RadioNeuron) Phase() uint8 { return uint8(n>>8) & 0x7F }

// Amplitude returns the signal strength (byte B). 0 = dead, 255 = maximum confidence.
func (n RadioNeuron) Amplitude() uint8 { return uint8(n >> 16) }

// FreqEmit returns the frequency this neuron emits on (byte A, FULL 0-255).
func (n RadioNeuron) FreqEmit() uint8 { return uint8(n >> 24) }

// IsInhibitory returns true if this neuron suppresses signal (bit 7 of byte G).
func (n RadioNeuron) IsInhibitory() bool { return (n>>8)&0x80 != 0 }

// IsRefractory is kept for API compat but always returns false.
// Refractory state is now managed externally (not enough bits in 4 bytes).
func (n RadioNeuron) IsRefractory() bool { return false }

// SetFreqListen changes the listening frequency (byte R).
func (n *RadioNeuron) SetFreqListen(f uint8) {
	*n = RadioNeuron((uint32(*n) & 0xFFFFFF00) | uint32(f))
}

// SetPhase changes the oscillation phase (bits 0-6 of byte G, preserving inhibitory).
func (n *RadioNeuron) SetPhase(p uint8) {
	inhib := uint32(*n) & (0x80 << 8) // preserve inhibitory bit
	*n = RadioNeuron((uint32(*n) & 0xFFFF00FF) | inhib | uint32(p&0x7F)<<8)
}

// SetAmplitude changes the signal strength (byte B).
func (n *RadioNeuron) SetAmplitude(a uint8) {
	*n = RadioNeuron((uint32(*n) & 0xFF00FFFF) | uint32(a)<<16)
}

// SetFreqEmit changes the emit frequency (byte A, FULL 0-255).
func (n *RadioNeuron) SetFreqEmit(f uint8) {
	*n = RadioNeuron((uint32(*n) & 0x00FFFFFF) | uint32(f)<<24)
}

// SetRefractory is a no-op (refractory managed externally now).
func (n *RadioNeuron) SetRefractory(r bool) {}

// AdvancePhase moves the phase forward by the neuron's listen frequency.
// Phase wraps naturally at 128 (7-bit). Higher frequency = faster oscillation.
func (n *RadioNeuron) AdvancePhase() {
	newPhase := (n.Phase() + n.FreqListen()) & 0x7F
	n.SetPhase(newPhase)
}

// IsAlive returns true if the neuron has non-zero amplitude.
func (n RadioNeuron) IsAlive() bool { return n.Amplitude() > 0 }

// Resonance computes the phase match between a neuron and a bus signal.
// Uses cos256 lookup table from quantum_tile.go.
// Returns [-127, +127]: +127 = perfect resonance, -127 = anti-resonance, 0 = neutral.
func Resonance(neuronPhase, busPhase uint8) int8 {
	// Scale 7-bit phase to 8-bit for cos256 table lookup
	delta := (neuronPhase & 0x7F) - (busPhase & 0x7F)
	return cos256[delta]
}
