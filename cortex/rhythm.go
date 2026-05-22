package cortex

// ---------------------------------------------------------------------------
// Phase 7 — Biological Rhythms
//
// Biological brains have rhythms: circadian cycles, attention cycles,
// sleep-wake patterns. This module simulates internal timing using purely
// integer arithmetic.
// ---------------------------------------------------------------------------

// sineTable is a 256-entry lookup table approximating
//
//	128 + 127·sin(2π·i/256)
//
// Values range from 1 to 255 (uint8). Precomputed at package init time.
var sineTable [256]uint8

func init() {
	// Build the table with a fixed-point, integer-only Taylor-ish
	// approximation.  We only need the first quadrant (0..63 → 0..π/2)
	// and mirror the rest.
	//
	// For index i in [0,255] we want:
	//   sineTable[i] = round(128 + 127 * sin(2π * i / 256))
	//
	// Instead of float math we use a well-known parabolic approximation
	// that stays within ±1 of the true sine curve for 8-bit resolution.
	//
	// Parabolic sine for x in [0,255] mapped to one full period:
	//   half = 128
	//   For the first half (0..127): y =  4*x*(128-x) / 128
	//   For the second half (128..255): y = -4*(x-128)*(256-x) / 128
	//   Then scale to 0..127 and shift by 128.

	for i := 0; i < 256; i++ {
		var raw int
		if i < 128 {
			// rising half-period
			raw = 4 * i * (128 - i) // peak at i=64 → 4·64·64 = 16384
		} else {
			// falling half-period
			j := i - 128
			raw = -(4 * j * (128 - j))
		}
		// raw is in [-16384, +16384].  Scale to [-127, +127].
		scaled := (raw * 127) / 16384
		val := 128 + scaled
		if val < 0 {
			val = 0
		}
		if val > 255 {
			val = 255
		}
		sineTable[i] = uint8(val)
	}
}

// -------------------------------------------------------------------------
// RhythmState
// -------------------------------------------------------------------------

// RhythmState represents a single oscillatory rhythm inside the engine.
type RhythmState struct {
	Phase     uint8  // 0-255 representing one full cycle
	Frequency uint8  // How fast the rhythm oscillates (higher = faster)
	Amplitude uint8  // Current strength of the rhythm
	Name      string // "circadian", "attention", "theta", "gamma"
}

// -------------------------------------------------------------------------
// RhythmStats
// -------------------------------------------------------------------------

// RhythmStats is a snapshot of the engine's current timing state.
type RhythmStats struct {
	GlobalTick           uint64
	SleepPressure        uint8
	AlertnessLevel       uint8
	AttentionSpan        uint8
	CyclesSinceLastSleep uint64
	RhythmCount          int
}

// -------------------------------------------------------------------------
// RhythmEngine
// -------------------------------------------------------------------------

// RhythmEngine is the master biological clock. It owns a set of named
// rhythms and tracks sleep pressure, alertness, and attention span.
type RhythmEngine struct {
	Rhythms              map[string]*RhythmState
	GlobalTick           uint64 // Master clock
	SleepPressure        uint8  // Accumulates over time, resets on Sleep()
	AlertnessLevel       uint8  // Inverse of sleep pressure
	AttentionSpan        uint8  // How long focus can be maintained
	CyclesSinceLastSleep uint64 // Counts up until Sleep() is called
	SleepThreshold       uint8  // Dynamic sleep pressure threshold
}

// NewRhythmEngine creates a RhythmEngine pre-loaded with four default
// biological rhythms.
func NewRhythmEngine(cfg Config) *RhythmEngine {
	re := &RhythmEngine{
		Rhythms:        make(map[string]*RhythmState, 4),
		AlertnessLevel: 255,
		AttentionSpan:  255,
		SleepThreshold: cfg.RhythmSleepThreshold,
	}

	// Default rhythms ---------------------------------------------------
	re.Rhythms["circadian"] = &RhythmState{
		Name:      "circadian",
		Frequency: 1,
		Amplitude: 255, // slow, strong
	}
	re.Rhythms["attention"] = &RhythmState{
		Name:      "attention",
		Frequency: 10,
		Amplitude: 200, // medium speed
	}
	re.Rhythms["theta"] = &RhythmState{
		Name:      "theta",
		Frequency: 30,
		Amplitude: 150, // hippocampal rhythm
	}
	re.Rhythms["gamma"] = &RhythmState{
		Name:      "gamma",
		Frequency: 100,
		Amplitude: 100, // fast, binding rhythm
	}

	return re
}

// Tick advances the master clock by one step and updates every rhythm.
func (r *RhythmEngine) Tick() {
	r.GlobalTick++

	// Advance each rhythm's phase by its frequency (wraps naturally at 256
	// because Phase is uint8).
	for _, rs := range r.Rhythms {
		rs.Phase += rs.Frequency
	}

	// Increase sleep pressure (cap at 255).
	if r.SleepPressure < 255 {
		r.SleepPressure++
	}

	// Alertness is the complement of sleep pressure.
	r.AlertnessLevel = 255 - r.SleepPressure

	// Attention span degrades with alertness.  Simple proportional model:
	//   AttentionSpan = AlertnessLevel (they track together).
	r.AttentionSpan = r.AlertnessLevel

	r.CyclesSinceLastSleep++
}

// OnSleep resets the engine as if the organism just slept.
func (r *RhythmEngine) OnSleep() {
	r.SleepPressure = 0
	r.AlertnessLevel = 255
	r.CyclesSinceLastSleep = 0

	// Reset all rhythm phases.
	for _, rs := range r.Rhythms {
		rs.Phase = 0
	}
}

// GetAlertness returns the current alertness level (0-255).
func (r *RhythmEngine) GetAlertness() uint8 {
	return r.AlertnessLevel
}

// GetSleepPressure returns the current sleep pressure (0-255).
func (r *RhythmEngine) GetSleepPressure() uint8 {
	return r.SleepPressure
}

// NeedsSleep returns true when sleep pressure exceeds SleepThreshold.
func (r *RhythmEngine) NeedsSleep() bool {
	return r.SleepPressure > r.SleepThreshold
}

// GetRhythm returns the named rhythm, or nil if it does not exist.
func (r *RhythmEngine) GetRhythm(name string) *RhythmState {
	return r.Rhythms[name]
}

// GetAttentionModulator returns a 0-255 value that modulates attention.
//
// It is derived from the attention rhythm's current phase via the sine
// lookup table (peaks near phase 128) and then scaled down when alertness
// is low.
func (r *RhythmEngine) GetAttentionModulator() uint8 {
	att := r.Rhythms["attention"]
	if att == nil {
		return 128 // neutral fallback
	}

	// Base modulation from the sine table.
	base := sineTable[att.Phase] // 0-255, peaks near phase 64

	// Scale by alertness: result = base * alertness / 255.
	modulated := (uint16(base) * uint16(r.AlertnessLevel)) / 255

	return uint8(modulated)
}

// GetBindingStrength returns the gamma rhythm's current strength, which
// modulates how well different cortex modules synchronize with each other.
// Higher gamma → better binding.
func (r *RhythmEngine) GetBindingStrength() uint8 {
	g := r.Rhythms["gamma"]
	if g == nil {
		return 0
	}

	// Binding strength = amplitude scaled by the sine-table value at the
	// current phase, so it oscillates with the gamma wave.
	wave := sineTable[g.Phase]
	strength := (uint16(g.Amplitude) * uint16(wave)) / 255

	return uint8(strength)
}

// Stats returns a snapshot of the engine's timing state.
func (r *RhythmEngine) Stats() RhythmStats {
	return RhythmStats{
		GlobalTick:           r.GlobalTick,
		SleepPressure:        r.SleepPressure,
		AlertnessLevel:       r.AlertnessLevel,
		AttentionSpan:        r.AttentionSpan,
		CyclesSinceLastSleep: r.CyclesSinceLastSleep,
		RhythmCount:          len(r.Rhythms),
	}
}
