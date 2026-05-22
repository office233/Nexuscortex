package cortex

// ─────────────────────────────────────────────────────────────────────
// attention.go — Self-Attention for SDR Processing
// ─────────────────────────────────────────────────────────────────────
//
// The AttentionModule re-weights individual bits in an SDR based on
// how relevant they are to the current context. This sharpens the
// representation by pruning low-relevance bits, producing a more
// focused signal for downstream modules.
//
// Design principles:
//   - INTEGER-ONLY arithmetic (uint8/uint16/uint32 — no float64)
//   - Ring buffer pattern (copy, NOT [1:] reslicing) for context history
//   - Clone SDRs when storing in history to prevent aliasing
//   - Biologically plausible: contextual priming + frequency-based salience

// AttentionWeights holds per-bit attention weights for an SDR.
// Only bits active in the input SDR have non-zero weights.
type AttentionWeights struct {
	Weights []uint8 // Per-bit attention weight (0-255)
	Size    int     // Must match SDR Size
}

// AttentionModule computes contextual relevance weights for SDR bits.
// It maintains a sliding window of recent context SDRs and uses them
// to determine which bits in the current input are most relevant.
type AttentionModule struct {
	ContextHistory []SDR  // Recent context SDRs (ring buffer)
	MaxHistory     int    // Ring buffer capacity (from Config)
	writePos       int    // Current write position in ring buffer
	count          int    // Number of valid entries (≤ MaxHistory)
	Cfg            Config // Central configuration
}

// NewAttentionModule creates a new AttentionModule with the given configuration.
func NewAttentionModule(cfg Config) *AttentionModule {
	maxHist := cfg.AttentionHistorySize
	if maxHist <= 0 {
		maxHist = 10
	}
	return &AttentionModule{
		ContextHistory: make([]SDR, maxHist),
		MaxHistory:     maxHist,
		writePos:       0,
		count:          0,
		Cfg:            cfg,
	}
}

// ObserveContext adds a context SDR to the history ring buffer.
// Uses copy-based ring buffer — never [1:] reslicing.
func (am *AttentionModule) ObserveContext(context SDR) {
	// Clone the SDR to prevent aliasing with caller's data.
	am.ContextHistory[am.writePos] = context.Clone()
	am.writePos = (am.writePos + 1) % am.MaxHistory
	if am.count < am.MaxHistory {
		am.count++
	}
}

// ComputeWeights calculates per-bit attention weights for the input SDR
// based on relevance to the current context and frequency in history.
//
// For each active bit in input:
//   - frequency: how many times this bit appears in recent context history
//   - overlap:   whether this bit is also active in the current context SDR
//
// Weight = min(255, frequency * AttentionFrequencyScale + overlapBoost)
// Bits NOT active in the input keep weight 0.
//
// All arithmetic is integer-only (uint16 intermediate, uint8 result).
func (am *AttentionModule) ComputeWeights(input SDR, context SDR) AttentionWeights {
	aw := AttentionWeights{
		Weights: make([]uint8, input.Size),
		Size:    input.Size,
	}

	freqScale := uint16(am.Cfg.AttentionFrequencyScale)
	if freqScale == 0 {
		freqScale = 25
	}
	contextBoost := uint16(am.Cfg.AttentionContextBoost)
	if contextBoost == 0 {
		contextBoost = 100
	}

	// Iterate over each active bit in the input SDR.
	for _, idx := range input.ActiveIndices() {
		var score uint16

		// Count frequency of this bit across the context history.
		var freq uint16
		for i := 0; i < am.count; i++ {
			hist := am.ContextHistory[i]
			if hist.Size > 0 && hist.IsActive(idx) {
				freq++
			}
		}
		score = freq * freqScale

		// Boost if the bit is also active in the current context.
		if context.Size > 0 && context.IsActive(idx) {
			score += contextBoost
		}

		// Clamp to uint8 range.
		if score > 255 {
			score = 255
		}

		aw.Weights[idx] = uint8(score)
	}

	return aw
}

// AttendSDR creates a sharpened SDR by keeping only bits whose attention
// weight exceeds the configured threshold (AttentionMinWeight).
// Returns a new SDR — the original is not mutated.
func (am *AttentionModule) AttendSDR(input SDR, weights AttentionWeights) SDR {
	result := NewSDR(input.Size)

	threshold := am.Cfg.AttentionMinWeight
	if threshold == 0 {
		threshold = 64
	}

	for _, idx := range input.ActiveIndices() {
		if idx < weights.Size && weights.Weights[idx] >= threshold {
			result.Set(idx)
		}
	}

	return result
}
