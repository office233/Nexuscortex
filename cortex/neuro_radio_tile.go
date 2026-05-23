package cortex

// ═══════════════════════════════════════════════════════════════════
// NeuroRadioTile — The Unified Tile
// ═══════════════════════════════════════════════════════════════════
//
// Combines TernaryTile (content/weights) + RadioMeta (routing/frequency)
// into a single unit that knows WHAT to compute AND WHO to talk to.
//
// Layout (12 bytes per tile):
//   TernaryTile    [4 bytes] = 16 ternary weights {-1,0,+1} in sign/mask format
//   ConfidenceTile [4 bytes] = 2-bit confidence per weight
//   RadioMeta      [4 bytes] = RadioNeuron format (freq/phase/amp/emit)
//
// TernaryTile format (compatible with ternary.go):
//   R = sign bits 0-7     (1=negative)
//   G = mask bits 0-7     (1=active, 0=zero)
//   B = sign bits 8-15
//   A = mask bits 8-15
//
// Effective weight per micro-weight:
//   output = ternary_value × confidence_gate × freq_match × phase_sign × amplitude
// ─────────────────────────────────────────────────────────────────────
// Architecture constants for NeuroRadioTile
// ─────────────────────────────────────────────────────────────────────

const (
	// TileWeightCount is the number of ternary weights per tile (fixed by RGBA32 format).
	TileWeightCount = 16

	// PhaseResonanceExcite is the positive resonance threshold for excitatory firing.
	PhaseResonanceExcite int8 = 16

	// PhaseResonanceInhibit is the negative resonance threshold for inhibitory firing.
	PhaseResonanceInhibit int8 = -16

	// ConfidenceLowBitmask sets the lowest confidence bit for each 2-bit pair.
	ConfidenceLowBitmask uint32 = 0x55555555

	// AmplitudeShift is the right-shift used for amplitude scaling (divide by 128).
	AmplitudeShift = 7
)

// NeuroRadioTile is the fundamental compute+routing unit.
type NeuroRadioTile struct {
	Weights    uint32      // TernaryTile: sign/mask RGBA format (NOT 2-bit pairs)
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

// unpackTernary extracts 16 ternary weights from the sign/mask RGBA format.
// Returns values in {-1, 0, +1}.
func (t *NeuroRadioTile) unpackTernary() [TileWeightCount]int8 {
	var out [TileWeightCount]int8
	w := t.Weights
	// R = sign bits 0-7, G = mask bits 0-7 (low 8 weights)
	// B = sign bits 8-15, A = mask bits 8-15 (high 8 weights)
	signLo := uint8(w)         // R: sign bits 0-7
	maskLo := uint8(w >> 8)    // G: mask bits 0-7
	signHi := uint8(w >> 16)   // B: sign bits 8-15
	maskHi := uint8(w >> 24)   // A: mask bits 8-15

	for i := 0; i < 8; i++ {
		bit := uint8(1 << uint(i))
		if maskLo&bit != 0 {
			if signLo&bit != 0 {
				out[i] = -1
			} else {
				out[i] = +1
			}
		}
		if maskHi&bit != 0 {
			if signHi&bit != 0 {
				out[i+8] = -1
			} else {
				out[i+8] = +1
			}
		}
	}
	return out
}

// Forward computes the tile's output given a 16-element input vector.
// Returns the weighted sum, gated by confidence, phase, and amplitude.
//
// P0.3 FIX: uses sign/mask unpack compatible with TernaryTile format.
// P0.4 FIX: phase logic uses signed resonance for proper anti-phase.
func (t *NeuroRadioTile) Forward(input [TileWeightCount]int8, busSignal int32, busPhase uint8) int32 {
	// Gate 1: Frequency match — was there a signal on our listen frequency?
	if busSignal == 0 {
		return 0 // No signal on our freq → dormant, zero cost
	}

	// Gate 2: Phase resonance (returns [-127, +127])
	res := Resonance(t.Radio.Phase(), busPhase)

	// P0.4 FIX: proper three-way phase logic
	var phaseSign int32
	if res > PhaseResonanceExcite {
		phaseSign = +1 // In-phase: excitatory contribution
	} else if res < PhaseResonanceInhibit {
		phaseSign = -1 // Anti-phase: inhibitory contribution
	} else {
		return 0 // Near-zero resonance: skip (neutral zone)
	}

	// Gate 3: Ternary × Confidence forward pass
	// P0.3 FIX: use proper sign/mask unpack, not 2-bit pairs
	ternary := t.unpackTernary()
	c := t.Confidence

	var sum int32
	for i := 0; i < TileWeightCount; i++ {
		tw := ternary[i]
		if tw == 0 {
			continue // Zero weight → skip
		}
		// Extract 2-bit confidence: 00=skip, 01=low, 10=med, 11=high
		conf := (c >> (uint(i) * 2)) & 0x03
		if conf == 0 {
			continue // Zero confidence → skip
		}

		val := int32(input[i]) * int32(tw) // +1 or -1 applied
		sum += val * int32(conf)
	}

	// Gate 4: Phase sign (anti-phase neurons negate output)
	sum *= phaseSign

	// Gate 5: Amplitude scaling
	amp := int32(t.Radio.Amplitude())
	sum = (sum * amp) >> AmplitudeShift // Scale by amplitude/128

	return sum
}

// ForwardSparse is the fast path: only processes if frequency matches.
// Returns (output, didActivate).
func (t *NeuroRadioTile) ForwardSparse(input [TileWeightCount]int8, bus *RadioBus) (int32, bool) {
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
	t.Confidence |= ConfidenceLowBitmask // Set at least bit 0 of each 2-bit pair
}

// Contradict weakens this tile (Hebbian LTD).
// P0.5 FIX: use SetAmplitude(0) for death, not bit manipulation on emit_freq.
func (t *NeuroRadioTile) Contradict() {
	amp := t.Radio.Amplitude()
	if amp > 0 {
		t.Radio.SetAmplitude(amp - 1)
	}
}

// IsAlive returns whether this tile is active (amplitude > 0).
func (t *NeuroRadioTile) IsAlive() bool {
	return t.Radio.IsAlive()
}

// ListenFreq returns the frequency this tile listens on.
func (t *NeuroRadioTile) ListenFreq() uint8 {
	return t.Radio.FreqListen()
}

// EmitFreq returns the frequency this tile emits on (NOW FULL 0-255).
func (t *NeuroRadioTile) EmitFreq() uint8 {
	return t.Radio.FreqEmit()
}

// Amplitude returns the tile's amplitude (confidence/strength).
func (t *NeuroRadioTile) Amplitude() uint8 {
	return t.Radio.Amplitude()
}
