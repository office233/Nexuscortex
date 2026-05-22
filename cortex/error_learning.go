package cortex

// ─────────────────────────────────────────────────────────────────────
// error_learning.go — Error-Driven Synaptic Learning (C.6)
// ─────────────────────────────────────────────────────────────────────
//
// Prediction error from the Predictor currently signals surprise to
// higher-order modules (Reward, Workspace, Curiosity) but does NOT
// directly influence synapse-level learning. This module closes that
// gap by using prediction error to SELECTIVELY strengthen or weaken
// synapses in the Prefrontal reservoir network.
//
// Algorithm:
//
//   1. Compute false positives: bits that were PREDICTED but NOT ACTUAL.
//      → These represent "over-confident" pathways — weaken their
//        incoming synapses so the network fires less on those neurons.
//
//   2. Compute false negatives: bits that were NOT PREDICTED but ACTUAL.
//      → These represent "missed surprises" — strengthen their incoming
//        synapses so the network is more receptive next time.
//
//   3. The adjustment amount is proportional to the prediction error:
//        - High error (>200):   full Config.ErrorLearningStrength
//        - Medium error (100–200): half Config.ErrorLearningStrength
//        - Low error (<100):    no adjustment (predictions are good)
//
// All arithmetic is integer-only (uint8/uint16/uint32). No float64.

// ErrorDrivenLearner applies prediction error signals to selectively
// adjust synaptic weights in a spiking neural network.
type ErrorDrivenLearner struct {
	Cfg Config
}

// NewErrorDrivenLearner creates an ErrorDrivenLearner with the given config.
func NewErrorDrivenLearner(cfg Config) *ErrorDrivenLearner {
	return &ErrorDrivenLearner{Cfg: cfg}
}

// AdjustSynapses modifies synaptic weights in the network based on the
// discrepancy between predicted and actual SDR patterns.
//
// The method:
//   - Weakens synapses feeding false-positive neurons (predicted ∧ ¬actual)
//   - Strengthens synapses feeding false-negative neurons (¬predicted ∧ actual)
//   - Scales adjustments proportionally to predictionError magnitude
//   - Respects StdpMaxWeight cap from Config
//
// No changes are made when predictionError < ErrorLearningThreshold.
func (e *ErrorDrivenLearner) AdjustSynapses(net *Network, predicted SDR, actual SDR, predictionError uint8) {
	if net == nil || net.Neurons == nil || net.Neurons.Size() == 0 {
		return
	}

	// ── Gate: skip if error is below threshold ───────────────────
	threshold := e.Cfg.ErrorLearningThreshold
	if threshold == 0 {
		threshold = 100 // default fallback
	}
	if predictionError < threshold {
		return // predictions are good enough — no adjustment needed
	}

	// ── Determine adjustment strength ───────────────────────────
	strength := e.Cfg.ErrorLearningStrength
	if strength == 0 {
		strength = 5 // default fallback
	}

	var adjustment uint8
	if predictionError > 200 {
		// High error: aggressive adjustment (full strength)
		adjustment = strength
	} else {
		// Medium error (threshold..200): moderate adjustment (half)
		adjustment = strength / 2
		if adjustment == 0 {
			adjustment = 1 // ensure at least 1 to have any effect
		}
	}

	// ── Build sets of false-positive and false-negative neuron IDs ─
	// False positives: predicted=1, actual=0  → weaken incoming synapses
	// False negatives: predicted=0, actual=1  → strengthen incoming synapses
	//
	// Build boolean lookup tables for false-positive and false-negative
	// neurons, mapped into the network's neuron index space.
	netSize := net.Neurons.Size()
	falsePos := make([]bool, netSize)
	falseNeg := make([]bool, netSize)

	// Iterate through SDR bits to find mismatches.
	maxSize := predicted.Size
	if actual.Size > maxSize {
		maxSize = actual.Size
	}

	for bitIdx := 0; bitIdx < maxSize; bitIdx++ {
		pActive := predicted.IsActive(bitIdx)
		aActive := actual.IsActive(bitIdx)

		if pActive == aActive {
			continue // match — no adjustment needed
		}

		// Map the SDR bit index into the network's neuron space.
		neuronIdx := bitIdx % netSize

		if pActive && !aActive {
			falsePos[neuronIdx] = true
		} else {
			falseNeg[neuronIdx] = true
		}
	}

	// ── Apply adjustments to synapses ───────────────────────────
	maxWeight := e.Cfg.StdpMaxWeight
	if maxWeight == 0 {
		maxWeight = StdpMaxWeight // fallback to const
	}

	for i := range net.Synapses {
		syn := &net.Synapses[i]
		if syn.Weight == 0 {
			continue // already pruned — skip
		}

		tgtIdx := int(syn.Target)
		if tgtIdx >= netSize {
			continue
		}

		// Weaken synapses feeding false-positive neurons.
		if falsePos[tgtIdx] {
			if syn.Weight > adjustment {
				syn.Weight -= adjustment
			} else {
				syn.Weight = 0
			}
		}

		// Strengthen synapses feeding false-negative neurons.
		if falseNeg[tgtIdx] {
			newW := uint16(syn.Weight) + uint16(adjustment)
			if newW > uint16(maxWeight) {
				newW = uint16(maxWeight)
			}
			syn.Weight = uint8(newW)
		}
	}
}
