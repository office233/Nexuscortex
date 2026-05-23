package cortex

// ═══════════════════════════════════════════════════════════════════
// NeuroRadioTile — The Unified Tile
// ═══════════════════════════════════════════════════════════════════
//
// Combines TernaryTile (content/weights) + RadioMeta (routing/frequency)
// into a single unit that knows WHAT to compute AND WHO to talk to.
//
// Layout (12 bytes per tile):
//   TernaryTile    [4 bytes] = 16 ternary weights {-1,0,+1}
//   ConfidenceTile [4 bytes] = 2-bit confidence per weight
//   RadioMeta      [4 bytes] = RadioNeuron format (freq/phase/amp/emit)
//
// Effective weight per micro-weight:
//   output = ternary_value × confidence_gate × freq_match × phase_sign × amplitude
//
// If frequency doesn't match → zero (tile never activates, zero cost)
// If confidence is zero → skip that weight
// If phase is anti-phase → negate the sign
// If amplitude is high → strong contribution

// NeuroRadioTile is the fundamental compute+routing unit.
type NeuroRadioTile struct {
	Weights    uint32      // TernaryTile: 16 weights packed as 2-bit pairs
	Confidence uint32      // ConfidenceTile: 2-bit confidence per weight
	Radio      RadioNeuron // RadioMeta: listen_freq, phase, amplitude, emit_freq
}

// NewNeuroRadioTile creates a tile with random weights and given radio params.
func NewNeuroRadioTile(weights, confidence uint32, listenFreq, phase, amplitude, emitFreq uint8, inhibitory bool) NeuroRadioTile {
	return NeuroRadioTile{
		Weights:    weights,
		Confidence: confidence,
		Radio:      PackRadioNeuron(listenFreq, phase, amplitude, emitFreq, inhibitory),
	}
}

// Forward computes the tile's output given a 16-element input vector.
// Returns the weighted sum, gated by confidence and radio parameters.
//
// This is the "radio-gated ternary forward":
//   1. Check frequency match (if not matched, return 0)
//   2. For each of 16 weights: ternary × confidence × input
//   3. Scale by amplitude and phase sign
func (t *NeuroRadioTile) Forward(input [16]int8, busSignal int32, busPhase uint8) int32 {
	// Gate 1: Frequency match — was there a signal on our listen frequency?
	if busSignal == 0 {
		return 0 // No signal on our freq → dormant, zero cost
	}

	// Gate 2: Phase resonance
	resonance := Resonance(t.Radio.Phase(), busPhase)
	if resonance < 32 {
		return 0 // Poor phase alignment → don't activate
	}

	// Gate 3: Ternary × Confidence forward pass
	var sum int32
	w := t.Weights
	c := t.Confidence

	for i := 0; i < 16; i++ {
		// Extract 2-bit ternary: 00=0, 01=+1, 10=-1, 11=reserved
		tw := (w >> (uint(i) * 2)) & 0x03
		// Extract 2-bit confidence: 00=skip, 01=low, 10=med, 11=high
		conf := (c >> (uint(i) * 2)) & 0x03

		if tw == 0 || conf == 0 {
			continue // Zero weight or zero confidence → skip
		}

		val := int32(input[i])
		if tw == 2 {
			val = -val // Inhibitory weight
		}
		// Scale by confidence (1, 2, or 3)
		sum += val * int32(conf)
	}

	// Gate 4: Amplitude scaling
	amp := int32(t.Radio.Amplitude())
	sum = (sum * amp) >> 7 // Scale by amplitude/128

	// Gate 5: Phase sign — anti-phase neurons negate output
	if resonance < 64 {
		sum = -sum // Weak resonance → partial inhibition
	}

	return sum
}

// ForwardSparse is the fast path: only processes if frequency matches.
// Returns (output, didActivate).
func (t *NeuroRadioTile) ForwardSparse(input [16]int8, bus *RadioBus) (int32, bool) {
	freq := t.Radio.FreqListen()
	signal, phase := bus.Read(freq)

	if signal == 0 {
		return 0, false
	}

	result := t.Forward(input, signal, phase)
	return result, result != 0
}

// Confirm strengthens this tile (Hebbian LTP).
func (t *NeuroRadioTile) Confirm() {
	amp := t.Radio.Amplitude()
	if amp < 255 {
		t.Radio.SetAmplitude(amp + 1)
	}
	// Also boost confidence of active weights
	t.Confidence |= 0x55555555 // Set at least bit 0 of each 2-bit pair
}

// Contradict weakens this tile (Hebbian LTD).
func (t *NeuroRadioTile) Contradict() {
	amp := t.Radio.Amplitude()
	if amp > 0 {
		t.Radio.SetAmplitude(amp - 1)
	}
	// Kill if dead — clear the alive bit (bit 0 of A byte)
	if amp <= 1 {
		raw := uint32(t.Radio)
		raw &^= 0x01 // Clear alive bit
		t.Radio = RadioNeuron(raw)
	}
}

// IsAlive returns whether this tile is active.
func (t *NeuroRadioTile) IsAlive() bool {
	return t.Radio.IsAlive()
}

// ListenFreq returns the frequency this tile listens on.
func (t *NeuroRadioTile) ListenFreq() uint8 {
	return t.Radio.FreqListen()
}

// EmitFreq returns the frequency this tile emits on.
func (t *NeuroRadioTile) EmitFreq() uint8 {
	return t.Radio.FreqEmit()
}

// Amplitude returns the tile's amplitude (confidence/strength).
func (t *NeuroRadioTile) Amplitude() uint8 {
	return t.Radio.Amplitude()
}
