package cortex

// radio_generate.go — Autoregressive text generation through frequency resonance.
//
// RadioGenerate produces text token-by-token using the RadioCortex:
//  1. Encode input tokens → frequencies via SignalCodec
//  2. Inject onto bus
//  3. Run N ticks (signal propagates input → hidden → output)
//  4. Decode output frequencies → next token
//  5. Feed next token back as input
//  6. Repeat until EOS or max tokens
//
// This is like GPT's autoregressive loop, but using frequency resonance
// instead of matrix multiplication.

// RadioTrainStep trains the RadioCortex on a single input→target association.
// It injects input frequencies, runs ticks, then compares output to target.
// Matching neurons get Confirm (amplitude++), wrong ones get Contradict (amplitude--).
//
// Returns the number of matching output frequencies (higher = better).
func (rc *RadioCortex) RadioTrainStep(codec *SignalCodec, inputTokenIDs []int, targetTokenID int, ticksPerStep int) int {
	// Clear any previous state
	rc.Bus.Clear()
	for i := range rc.Fired {
		rc.Fired[i] = false
	}
	for i := range rc.PrevFired {
		rc.PrevFired[i] = false
	}

	// Inject input tokens onto bus
	codec.EncodeTokens(&rc.Bus, inputTokenIDs, uint8(rc.TrainAmplitude))

	// Activate input neurons that match
	for i := rc.InputStart; i < rc.InputEnd; i++ {
		n := &rc.Neurons[i]
		signal, busPhase := rc.Bus.Read(n.FreqListen())
		if signal > 0 {
			resonance := Resonance(n.Phase(), busPhase)
			if resonance > int8(rc.ResonanceThreshold) {
				rc.Fired[i] = true
			}
		}
	}

	// Run ticks — GPU-accelerated if available, else CPU fallback
	if rc.GPU != nil && rc.GPU.IsAvailable() {
		// Upload current bus state to GPU, run all ticks, download result
		rc.SyncToGPU()
		var busArr [256]int32
		for ch := 0; ch < 256; ch++ {
			busArr[ch] = rc.Bus.Signal[ch]
		}
		rc.GPU.UploadBus(&busArr)
		rc.GPU.StepN(ticksPerStep, rc.FireThreshold, int32(rc.PhaseWindow))
		rc.GPU.DownloadBus(&busArr)
		for ch := 0; ch < 256; ch++ {
			rc.Bus.Signal[ch] = busArr[ch]
		}
		rc.SyncFromGPU()
		rc.TickCount += uint64(ticksPerStep)
	} else {
		for t := 0; t < ticksPerStep; t++ {
			rc.Step()
		}
	}

	// Read output bus state and compare to target
	targetFreqs := codec.TokenFreqs(targetTokenID)
	if targetFreqs == nil {
		return 0
	}

	// Check which output neurons fired
	outputFired := false
	for i := rc.OutputStart; i < rc.OutputEnd; i++ {
		if rc.Fired[i] {
			outputFired = true
			break
		}
	}

	// Build set of target frequencies for comparison
	targetSet := make(map[uint8]bool, len(targetFreqs))
	for _, f := range targetFreqs {
		targetSet[f] = true
	}

	// Score: how many of the fired output neurons are on target frequencies
	matches := 0
	for i := rc.OutputStart; i < rc.OutputEnd; i++ {
		if !rc.Fired[i] {
			continue
		}
		n := rc.Neurons[i]
		// Check if this neuron's emit frequency is in the target set
		if targetSet[n.FreqEmit()] || targetSet[n.FreqListen()] {
			matches++
		}
	}

	// Hebbian learning based on results
	if matches > 0 || outputFired {
		// Some neurons fired correctly — strengthen them
		for i := range rc.Neurons {
			if !rc.Fired[i] {
				continue
			}
			n := &rc.Neurons[i]
			freq := n.FreqListen()

			// Check if this neuron is on a target-relevant frequency
			isRelevant := targetSet[freq] || targetSet[n.FreqEmit()]

			if isRelevant {
				// Confirm: this neuron helped produce the right answer
				amp := n.Amplitude()
				if amp < 255 {
					n.SetAmplitude(amp + 1)
				}
			} else if i >= rc.OutputStart {
				// Output neuron fired but was NOT on target frequency — weaken
				amp := n.Amplitude()
				if amp > 0 {
					n.SetAmplitude(amp - 1)
				}
				// Weak neurons drift toward target frequencies
				if amp < uint8(rc.WeakNeuronThreshold) && len(targetFreqs) > 0 {
					// Re-tune to a random target frequency
					newFreq := targetFreqs[rc.rng.Intn(len(targetFreqs))]
					n.SetFreqListen(newFreq)
				}
			}
		}
	} else {
		// No output at all — very weak, contradict all fired neurons
		rc.Contradict()
	}

	return matches
}

// RadioGenerate produces text autoregressively through frequency resonance.
//
// Process:
//  1. Encode context tokens → frequencies
//  2. Run ticks → signal propagates
//  3. Decode bus output → next token
//  4. Feed next token back as context
//  5. Repeat until maxTokens
//
// Returns generated token IDs and their text representation.
func (rc *RadioCortex) RadioGenerate(codec *SignalCodec, vocab *Vocab, contextTokenIDs []int, maxTokens int, ticksPerStep int) string {
	if codec == nil || vocab == nil || len(contextTokenIDs) == 0 {
		return ""
	}

	generated := make([]int, 0, maxTokens)
	context := make([]int, len(contextTokenIDs))
	copy(context, contextTokenIDs)

	// Track recently generated tokens to avoid loops
	lastTokens := make(map[int]int) // tokenID → count

	for step := 0; step < maxTokens; step++ {
		// Clear state
		rc.Bus.Clear()
		for i := range rc.Fired {
			rc.Fired[i] = false
		}
		for i := range rc.PrevFired {
			rc.PrevFired[i] = false
		}

		// Encode current context (use last N tokens as window)
		windowSize := rc.GenerateWindowSize
		window := context
		if len(window) > windowSize {
			window = window[len(window)-windowSize:]
		}
		codec.EncodeTokens(&rc.Bus, window, uint8(rc.TrainAmplitude))

		// Activate matching input neurons
		for i := rc.InputStart; i < rc.InputEnd; i++ {
			n := &rc.Neurons[i]
			signal, busPhase := rc.Bus.Read(n.FreqListen())
			if signal > 0 {
				resonance := Resonance(n.Phase(), busPhase)
				if resonance > int8(rc.ResonanceThreshold) {
					rc.Fired[i] = true
				}
			}
		}

		// Run ticks — GPU-accelerated if available
		if rc.GPU != nil && rc.GPU.IsAvailable() {
			rc.SyncToGPU()
			var busArr [256]int32
			for ch := 0; ch < 256; ch++ {
				busArr[ch] = rc.Bus.Signal[ch]
			}
			rc.GPU.UploadBus(&busArr)
			rc.GPU.StepN(ticksPerStep, rc.FireThreshold, int32(rc.PhaseWindow))
			rc.GPU.DownloadBus(&busArr)
			for ch := 0; ch < 256; ch++ {
				rc.Bus.Signal[ch] = busArr[ch]
			}
			rc.SyncFromGPU()
			rc.TickCount += uint64(ticksPerStep)
		} else {
			for t := 0; t < ticksPerStep; t++ {
				rc.Step()
			}
		}

		// Decode: find token with highest frequency energy on the bus
		tokenID, score := codec.DecodeToken(&rc.Bus)
		if tokenID < 0 || score <= 0 {
			break // No signal = end of generation
		}

		// Anti-loop: skip tokens that appear too often
		if lastTokens[tokenID] >= rc.AntiLoopMaxRepeat {
			// Try second-best token
			topK := codec.DecodeTopK(&rc.Bus, rc.DecodeTopK)
			found := false
			for _, ts := range topK {
				if lastTokens[ts.TokenID] < rc.AntiLoopMaxRepeat {
					tokenID = ts.TokenID
					found = true
					break
				}
			}
			if !found {
				break // All top tokens are loops
			}
		}

		generated = append(generated, tokenID)
		context = append(context, tokenID)
		lastTokens[tokenID]++
	}

	// Convert token IDs to text
	var words []string
	for _, tid := range generated {
		word := vocab.Decode(uint32(tid))
		if word != "" {
			words = append(words, word)
		}
	}

	result := ""
	for i, w := range words {
		if i > 0 {
			result += " "
		}
		result += w
	}
	return result
}
