package cortex

// ─────────────────────────────────────────────────────────────────────
// Predictor — Free Energy Engine (Associative Temporal Prediction)
// ─────────────────────────────────────────────────────────────────────
//
// The Predictor implements a predictive processing engine inspired by
// the Free Energy Principle. It maintains a sliding window of recent
// input SDRs and forms predictions by computing the union of recent
// inputs (temporal prediction). Prediction error — the mismatch
// between what was predicted and what actually arrived — drives
// learning and signals novelty/surprise to higher-level modules.
//
// Architecture:
//   - Sliding window of the last N input SDRs (PredictorWindowSize)
//   - Prediction = union of the last PredictorUnionDepth inputs
//   - Both prediction and reality live in the same SDR space
//   - Prediction error history for running average
//
// Prediction error is encoded as uint8: 0 = perfect prediction,
// 255 = maximum surprise (zero overlap).

// DefaultPredictorWindowSize is the number of recent SDRs to retain.
const DefaultPredictorWindowSize = 5

// DefaultPredictorUnionDepth is how many recent SDRs to union for prediction.
// Must be <= DefaultPredictorWindowSize.
const DefaultPredictorUnionDepth = 3

// DefaultPredictorMaxHistory is the maximum error history entries retained.
const DefaultPredictorMaxHistory = 100

// Predictor generates predictions from recent input history and
// tracks prediction error over time as a surprise signal.
type Predictor struct {
	Window          []SDR   // Sliding window of recent input SDRs
	LastPrediction  SDR     // Most recent prediction output
	LastReality     SDR     // Most recent actual input
	PredictionError uint8   // Error from the last Update call
	ErrorHistory    []uint8 // Rolling history of prediction errors
	AvgError        uint8   // Running average of ErrorHistory
}

// NewPredictor creates a Predictor with an empty sliding window.
// The rng parameter is accepted for API compatibility but is unused.
func NewPredictor(_ interface{}) *Predictor {
	return &Predictor{
		Window:       make([]SDR, 0, DefaultPredictorWindowSize),
		ErrorHistory: make([]uint8, 0, DefaultPredictorMaxHistory),
	}
}

// ─────────────────────────────────────────────────────────────────────
// Predict — Generate a prediction from recent input history
// ─────────────────────────────────────────────────────────────────────

// Predict appends currentInput to the sliding window and produces a
// prediction by computing the union of the last PredictorUnionDepth
// inputs. The prediction and input are both in the same SDR space,
// so Compare works correctly.
func (p *Predictor) Predict(currentInput SDR) SDR {
	// Append to sliding window (ring buffer to avoid backing array leak).
	cloned := currentInput.Clone()
	if len(p.Window) >= DefaultPredictorWindowSize {
		copy(p.Window, p.Window[1:])
		p.Window[len(p.Window)-1] = cloned
	} else {
		p.Window = append(p.Window, cloned)
	}

	// Union the last DefaultPredictorUnionDepth entries for temporal prediction.
	depth := DefaultPredictorUnionDepth
	if depth > len(p.Window) {
		depth = len(p.Window)
	}

	start := len(p.Window) - depth
	prediction := p.Window[start]
	for i := start + 1; i < len(p.Window); i++ {
		prediction = prediction.Union(p.Window[i])
	}

	p.LastPrediction = prediction
	return prediction
}

// ─────────────────────────────────────────────────────────────────────
// Compare — Compute prediction error between two SDRs
// ─────────────────────────────────────────────────────────────────────

// Compare computes the prediction error between a prediction and
// reality. Error is scaled to uint8: 0 means perfect match (full
// overlap), 255 means no overlap at all.
//
// Formula: error = 255 - (overlap / maxActive) * 255
// All arithmetic is integer-only.
func (p *Predictor) Compare(prediction SDR, reality SDR) uint8 {
	maxActive := prediction.ActiveCount
	if reality.ActiveCount > maxActive {
		maxActive = reality.ActiveCount
	}
	if maxActive == 0 {
		// Both empty — nothing to compare, no error.
		return 0
	}

	overlap := prediction.Overlap(reality)

	// Scale overlap ratio to [0, 255] using integer math.
	// overlapScaled = (overlap * 255) / maxActive
	overlapScaled := (overlap * 255) / maxActive
	if overlapScaled > 255 {
		overlapScaled = 255
	}

	return 255 - uint8(overlapScaled)
}

// ─────────────────────────────────────────────────────────────────────
// Update — Compare last prediction with reality, store error
// ─────────────────────────────────────────────────────────────────────

// Update compares LastPrediction with the incoming reality SDR,
// computes the prediction error, stores it in the error history,
// updates the running average, and returns the error.
func (p *Predictor) Update(reality SDR) uint8 {
	p.LastReality = reality
	p.PredictionError = p.Compare(p.LastPrediction, reality)

	// Append to error history (ring buffer to avoid backing array leak).
	if len(p.ErrorHistory) >= DefaultPredictorMaxHistory {
		copy(p.ErrorHistory, p.ErrorHistory[1:])
		p.ErrorHistory[len(p.ErrorHistory)-1] = p.PredictionError
	} else {
		p.ErrorHistory = append(p.ErrorHistory, p.PredictionError)
	}

	// Recompute running average error (integer arithmetic).
	if len(p.ErrorHistory) > 0 {
		var sum uint32
		for _, e := range p.ErrorHistory {
			sum += uint32(e)
		}
		p.AvgError = uint8(sum / uint32(len(p.ErrorHistory)))
	}

	return p.PredictionError
}

// ─────────────────────────────────────────────────────────────────────
// Surprise & Novelty
// ─────────────────────────────────────────────────────────────────────

// GetSurprise returns the most recent prediction error as a surprise
// signal. High values indicate the input was unexpected.
func (p *Predictor) GetSurprise() uint8 {
	return p.PredictionError
}

// IsNovel returns true if the current prediction error exceeds the
// given threshold, indicating the input is novel or unexpected.
func (p *Predictor) IsNovel(threshold uint8) bool {
	return p.PredictionError > threshold
}
