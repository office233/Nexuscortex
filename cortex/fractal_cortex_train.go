package cortex

// fractal_cortex_train.go — Autoregressive STDP Training
//
// Trains the FractalCortex sequence generator using Probabilistic STDP.
// Since the architecture is highly recurrent and sparse, we use a Reservoir
// Computing (Liquid State Machine) approach: the internal ALBERT layers
// act as a non-linear dynamical reservoir, and we only train the 
// Output Projections using STDP to map the reservoir state to the next token.

// TrainSequence trains the cortex to predict the sequence of SDRs.
// It uses a local Hebbian learning rule on the Output layers.
func (fc *FractalCortex) TrainSequence(sequence []SDR, learningRate uint8) {
	if len(sequence) < 2 || len(fc.Blocks) == 0 {
		return
	}

	// Always train the most recent active block
	activeBlock := fc.Blocks[len(fc.Blocks)-1]
	activeBlock.Reset() // Reset temporal state for new sequence

	// Forward pass token by token
	for t := 0; t < len(sequence)-1; t++ {
		input := sequence[t]
		target := sequence[t+1]

		// To do STDP, we need the pre-synaptic activation of the output layer.
		// The output layer is activeBlock.SharedOutputLayer.
		// Its input is the output of the LinearScan layer.
		
		// 1. Process layer-by-layer (since it's a shared stack, we run all virtual layers)
		current := input
		for i := 0; i < activeBlock.NumLayers; i++ {
			// Reproduce the forward pass to capture the pre-synaptic state of the LAST virtual layer
			inputAct := sdrToActivations(current)
			featAct := activeBlock.SharedFeatureLayer.Forward(inputAct)
			features := activationsToSDR(featAct, current.ActiveCount)

			activeBlock.Attentions[i].Store(features, features)
			attended := activeBlock.Attentions[i].Query(features, activeBlock.TopK)

			scanned := activeBlock.Scans[i].StepFast(attended)
			scanAct := sdrToActivations(scanned) // THIS is the pre-synaptic input to SharedOutputLayer!
			
			outAct := activeBlock.SharedOutputLayer.Forward(scanAct)
			output := activationsToSDR(outAct, current.ActiveCount)
			
			current = output.Union(current)

			// If this is the LAST virtual layer, apply STDP to the SharedOutputLayer
			if i == activeBlock.NumLayers-1 {
				// Compute error signal based on target SDR
				errorSignal := make([]int16, activeBlock.SharedOutputLayer.OutputSize)
				
				// Hebbian reinforcement: bits that SHOULD be active
				for _, idx := range target.ActiveIndices() {
					if idx < activeBlock.SharedOutputLayer.OutputSize {
						errorSignal[idx] = 1
					}
				}
				
				// Anti-Hebbian penalty: bits that were incorrectly predicted active
				for _, idx := range current.ActiveIndices() {
					if idx < activeBlock.SharedOutputLayer.OutputSize && !target.IsActive(idx) {
						errorSignal[idx] = -1
					}
				}

				// Apply STDP (Reservoir Readout Training)
				activeBlock.SharedOutputLayer.UpdateProbabilisticSTDP(scanAct, errorSignal, learningRate)
			}
		}
	}
}
