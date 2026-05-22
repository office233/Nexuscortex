package cortex

import (
	"encoding/json"
	"fmt"
	"os"
)

// ─────────────────────────────────────────────────────────────────────
// emotion.go — Emergent Functional States
// ─────────────────────────────────────────────────────────────────────
//
// Emotions are NOT hardcoded. They are functional states that EMERGE
// from the reward system and prediction error. The organism doesn't
// know it has emotions — we just measure its state.
//
// Each dimension is computed from raw signals:
//
//   Valence    = weighted blend of prior valence + incoming reward
//   Arousal    = mapped from prediction error magnitude
//   Curiosity  = peaks in the novelty sweet spot (prediction error 30–100)
//   Confidence = high when prediction error is consistently low
//   Social     = grows with interaction count
//
// All dimensions carry momentum: states don't snap instantly, they
// glide toward their targets. The decay rate controls how quickly
// the organism returns to neutral when no signals arrive.

// EmotionalState captures the organism's internal functional state
// along five orthogonal axes. These are measurements, not personality.
type EmotionalState struct {
	Valence    int8  // -128..+127: negative to positive
	Arousal    uint8 // 0..255: calm to excited
	Curiosity  uint8 // 0..255: how much novelty-seeking
	Confidence uint8 // 0..255: how sure of itself
	Social     uint8 // 0..255: how engaged with interaction
}

// EmotionEngine computes emergent emotional states from raw signals.
type EmotionEngine struct {
	State            EmotionalState   // Current state
	PreviousState    EmotionalState   // State before last update
	MomentumDecay    uint8            // How fast emotions decay (higher = faster)
	History          []EmotionalState // Recent state history
	MaxHistory       int              // Maximum history length
	ValenceWeight    uint8
	ArousalWeight    uint8
	CuriosityWeight  uint8
	ConfidenceWeight uint8
	SocialWeight     uint8
	Config           Config           // Runtime configuration
}

// NewEmotionEngine creates an emotion engine with the given config.
// MomentumDecay is loaded from config, producing smooth emotional transitions.
func NewEmotionEngine(cfg Config) *EmotionEngine {
	maxHistory := cfg.EmotionHistoryCapacity
	if maxHistory < 1 {
		maxHistory = 1
	}
	return &EmotionEngine{
		State:            EmotionalState{},
		PreviousState:    EmotionalState{},
		MomentumDecay:    cfg.EmotionMomentumDecay,
		History:          make([]EmotionalState, 0, maxHistory),
		MaxHistory:       maxHistory,
		ValenceWeight:    cfg.EmotionValenceWeight,
		ArousalWeight:    cfg.EmotionArousalWeight,
		CuriosityWeight:  cfg.EmotionCuriosityWeight,
		ConfidenceWeight: cfg.EmotionConfidenceWeight,
		SocialWeight:     cfg.EmotionSocialWeight,
		Config:           cfg,
	}
}

// Update computes a new emotional state from raw signals.
//
// Parameters:
//
//   - reward:           the most recent reward value (from RewardSystem)
//   - predictionError:  current prediction error magnitude
//   - interactionCount: total interactions the organism has had
//
// Nothing here is hardcoded personality — every value is derived from
// the incoming numbers through simple arithmetic.
func (e *EmotionEngine) Update(reward int8, predictionError uint8, interactionCount uint64) {
	e.PreviousState = e.State

	// ── Valence ──────────────────────────────────────────────────
	// Weighted average: e.ValenceWeight% prior state + (100 - e.ValenceWeight)% incoming reward.
	// This gives valence inertia so mood doesn't swing wildly.
	valWeight := int16(e.ValenceWeight)
	rewWeight := 100 - valWeight
	newValence := (int16(e.State.Valence)*valWeight + int16(reward)*rewWeight) / 100
	if newValence > 127 {
		newValence = 127
	}
	if newValence < -128 {
		newValence = -128
	}

	// ── Arousal ──────────────────────────────────────────────────
	// Direct mapping from prediction error. High error = high arousal.
	// Smooth toward target using momentum.
	targetArousal := uint16(predictionError) // 0..255 maps naturally
	newArousal := blendU8(e.State.Arousal, uint8(targetArousal), uint16(e.ArousalWeight))

	// ── Curiosity ────────────────────────────────────────────────
	// Peaks in the sweet spot (prediction error 30–100).
	// Below 30: curiosity drops (boring). Above 100: curiosity drops
	// (overwhelming). Within the band: scales to 255 at the center.
	var targetCuriosity uint8
	switch {
	case predictionError < 30:
		// Boring — low curiosity proportional to error.
		targetCuriosity = uint8(uint16(predictionError) * 80 / 30) // 0..80
	case predictionError <= 100:
		// Sweet spot — curiosity peaks.
		// Distance from center (65): 0 at center → 255.
		center := int16(65)
		dist := int16(predictionError) - center
		if dist < 0 {
			dist = -dist
		}
		// At center: 255, at edges (30 or 100): ~127.
		targetCuriosity = uint8(255 - uint16(dist)*128/35)
	default:
		// Overwhelming — curiosity crashes.
		// 101→60, 255→0.
		if predictionError < 161 {
			targetCuriosity = uint8(60 - uint16(predictionError-101)*60/60)
		}
		// else: 0 (already zero-initialized)
	}
	newCuriosity := blendU8(e.State.Curiosity, targetCuriosity, uint16(e.CuriosityWeight))

	// ── Confidence ───────────────────────────────────────────────
	// High when prediction error is consistently low.
	// Target: inversely proportional to prediction error.
	var targetConfidence uint8
	if predictionError < 128 {
		targetConfidence = uint8(255 - uint16(predictionError)*2)
	}
	// else: 0 — high error destroys confidence.
	// Confidence moves slowly (momentum %) — trust is earned.
	newConfidence := blendU8(e.State.Confidence, targetConfidence, uint16(e.ConfidenceWeight))

	// ── Social ───────────────────────────────────────────────────
	// Grows with interaction count, saturating at 255.
	// Uses a logarithmic-ish curve: fast initial growth, slow later.
	var targetSocial uint8
	switch {
	case interactionCount >= 1000:
		targetSocial = 255
	case interactionCount >= 100:
		// 100..999 → 180..255
		targetSocial = uint8(180 + (interactionCount-100)*75/900)
	case interactionCount >= 10:
		// 10..99 → 80..180
		targetSocial = uint8(80 + (interactionCount-10)*100/90)
	default:
		// 0..9 → 0..80
		targetSocial = uint8(interactionCount * 8)
	}
	newSocial := blendU8(e.State.Social, targetSocial, uint16(e.SocialWeight))

	// ── Commit ───────────────────────────────────────────────────
	e.State = EmotionalState{
		Valence:    int8(newValence),
		Arousal:    newArousal,
		Curiosity:  newCuriosity,
		Confidence: newConfidence,
		Social:     newSocial,
	}

	// Store in history.
	if len(e.History) >= e.MaxHistory {
		copy(e.History, e.History[1:])
		e.History[len(e.History)-1] = e.State
	} else {
		e.History = append(e.History, e.State)
	}
}

// GetState returns the current emotional state snapshot.
func (e *EmotionEngine) GetState() EmotionalState {
	return e.State
}

// GetMood returns a human-readable label computed from Valence and
// Arousal quadrants. These labels are for external observers, NOT
// personality the organism knows about.
//
//	High valence + high arousal = "energized"
//	High valence + low arousal  = "content"
//	Low valence  + high arousal = "stressed"
//	Low valence  + low arousal  = "depleted"
func (e *EmotionEngine) GetMood() string {
	highValence := e.State.Valence > 0
	highArousal := e.State.Arousal >= 128

	switch {
	case highValence && highArousal:
		return "energized"
	case highValence && !highArousal:
		return "content"
	case !highValence && highArousal:
		return "stressed"
	default:
		return "depleted"
	}
}

// IsStable returns true if emotional state is changing slowly.
// Measured by comparing the current state to the previous state
// across all dimensions. Small total delta = stable.
func (e *EmotionEngine) IsStable() bool {
	// Sum of absolute differences across all dimensions.
	delta := absDiffI8(e.State.Valence, e.PreviousState.Valence) +
		absDiffU8(e.State.Arousal, e.PreviousState.Arousal) +
		absDiffU8(e.State.Curiosity, e.PreviousState.Curiosity) +
		absDiffU8(e.State.Confidence, e.PreviousState.Confidence) +
		absDiffU8(e.State.Social, e.PreviousState.Social)

	// Threshold: total change across all 5 dimensions.
	stabThresh := e.Config.EmotionStabilityThreshold
	if stabThresh <= 0 {
		stabThresh = 30
	}
	return delta <= stabThresh
}

// Decay applies natural decay toward neutral (zero valence, low
// arousal/curiosity). The decay rate is controlled by MomentumDecay.
// Call this when no external signals arrive to let the organism
// gradually calm down.
func (e *EmotionEngine) Decay() {
	e.PreviousState = e.State

	rate := uint16(e.MomentumDecay) // 0..255

	// Valence decays toward 0.
	e.State.Valence = decayI8(e.State.Valence, rate)

	// Arousal decays toward 0.
	e.State.Arousal = decayU8(e.State.Arousal, rate)

	// Curiosity decays toward 0.
	e.State.Curiosity = decayU8(e.State.Curiosity, rate)

	// Confidence decays slowly toward a baseline (configurable, default 128).
	baseline := e.Config.EmotionConfidenceBaseline
	if baseline == 0 {
		baseline = 128
	}
	halfRate := rate / 2
	if halfRate == 0 {
		halfRate = 1
	}
	if e.State.Confidence > baseline {
		diff := uint16(e.State.Confidence-baseline) * halfRate / 256
		if diff == 0 && e.State.Confidence > baseline {
			diff = 1
		}
		e.State.Confidence -= uint8(diff)
	} else if e.State.Confidence < baseline {
		diff := uint16(baseline-e.State.Confidence) * halfRate / 256
		if diff == 0 && e.State.Confidence < baseline {
			diff = 1
		}
		e.State.Confidence += uint8(diff)
	}

	// Social does NOT decay — it only grows through interaction.
}

// ─────────────────────────────────────────────────────────────────────
// Internal helpers — pure arithmetic, no personality.
// ─────────────────────────────────────────────────────────────────────

// blendU8 smoothly moves current toward target by pct percent (0–100).
func blendU8(current, target uint8, pct uint16) uint8 {
	c := uint16(current)
	t := uint16(target)
	result := (c*(100-pct) + t*pct) / 100
	if result > 255 {
		result = 255
	}
	return uint8(result)
}

// decayU8 moves a uint8 toward zero by a fraction controlled by rate.
func decayU8(val uint8, rate uint16) uint8 {
	diff := uint16(val) * rate / 256
	if diff == 0 && val > 0 {
		diff = 1
	}
	if uint16(val) <= diff {
		return 0
	}
	return val - uint8(diff)
}

// decayI8 moves an int8 toward zero by a fraction controlled by rate.
func decayI8(val int8, rate uint16) int8 {
	if val > 0 {
		diff := int16(val) * int16(rate) / 256
		if diff == 0 {
			diff = 1
		}
		result := int16(val) - diff
		if result < 0 {
			result = 0
		}
		return int8(result)
	} else if val < 0 {
		diff := int16(-val) * int16(rate) / 256
		if diff == 0 {
			diff = 1
		}
		result := int16(val) + diff
		if result > 0 {
			result = 0
		}
		return int8(result)
	}
	return 0
}

// absDiffU8 returns the absolute difference between two uint8 values.
func absDiffU8(a, b uint8) int {
	if a > b {
		return int(a - b)
	}
	return int(b - a)
}

// absDiffI8 returns the absolute difference between two int8 values.
func absDiffI8(a, b int8) int {
	d := int16(a) - int16(b)
	if d < 0 {
		d = -d
	}
	return int(d)
}

// ─────────────────────────────────────────────────────────────────────
// Persistence — Save / Load
// ─────────────────────────────────────────────────────────────────────

// emotionSaveData is the JSON-serializable snapshot of EmotionEngine state.
type emotionSaveData struct {
	State            EmotionalState   `json:"state"`
	PreviousState    EmotionalState   `json:"previous_state"`
	MomentumDecay    uint8            `json:"momentum_decay"`
	History          []EmotionalState `json:"history"`
	MaxHistory       int              `json:"max_history"`
	ValenceWeight    uint8            `json:"valence_weight,omitempty"`
	ArousalWeight    uint8            `json:"arousal_weight,omitempty"`
	CuriosityWeight  uint8            `json:"curiosity_weight,omitempty"`
	ConfidenceWeight uint8            `json:"confidence_weight,omitempty"`
	SocialWeight     uint8            `json:"social_weight,omitempty"`
}

// Save persists the EmotionEngine state to a JSON file.
func (e *EmotionEngine) Save(path string) error {
	data := emotionSaveData{
		State:            e.State,
		PreviousState:    e.PreviousState,
		MomentumDecay:    e.MomentumDecay,
		History:          e.History,
		MaxHistory:       e.MaxHistory,
		ValenceWeight:    e.ValenceWeight,
		ArousalWeight:    e.ArousalWeight,
		CuriosityWeight:  e.CuriosityWeight,
		ConfidenceWeight: e.ConfidenceWeight,
		SocialWeight:     e.SocialWeight,
	}
	raw, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return fmt.Errorf("emotion save marshal: %w", err)
	}
	if err := os.WriteFile(path, raw, 0644); err != nil {
		return fmt.Errorf("emotion save write: %w", err)
	}
	return nil
}

// LoadEmotionEngine restores EmotionEngine state from a JSON file.
// Returns (nil, nil) if the file does not exist, allowing the caller
// to fall back to NewEmotionEngine.
func LoadEmotionEngine(path string, cfg Config) (*EmotionEngine, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("emotion load read: %w", err)
	}
	var data emotionSaveData
	if err := json.Unmarshal(raw, &data); err != nil {
		return nil, fmt.Errorf("emotion load unmarshal: %w", err)
	}
	if data.History == nil {
		data.History = make([]EmotionalState, 0)
	}
	return &EmotionEngine{
		State:            data.State,
		PreviousState:    data.PreviousState,
		MomentumDecay:    data.MomentumDecay,
		History:          data.History,
		MaxHistory:       data.MaxHistory,
		ValenceWeight:    cfg.EmotionValenceWeight,
		ArousalWeight:    cfg.EmotionArousalWeight,
		CuriosityWeight:  cfg.EmotionCuriosityWeight,
		ConfidenceWeight: cfg.EmotionConfidenceWeight,
		SocialWeight:     cfg.EmotionSocialWeight,
		Config:           cfg,
	}, nil
}
