package cortex

// ternary_train.go — Probabilistic STDP for Ternary Weights
//
// Implements a biologically plausible learning rule for the RGBA32 ternary
// weights. Since discrete weights {-1, 0, 1} cannot be updated with
// continuous gradients, we use Spike-Timing-Dependent Plasticity (STDP)
// with a probabilistic update mechanism.

// UpdateProbabilisticSTDP applies a probabilistic weight update based on
// pre-synaptic activation and post-synaptic error signal.
//
// - preSynaptic: The input activations to this layer (size = InputSize).
// - errorSignal: The desired change for each output neuron (size = OutputSize).
//                Positive means we want more activation (Hebbian reinforcement).
//                Negative means we want less activation (Anti-Hebbian penalty).
// - learningRate: Probability (0-255) of a weight transition occurring.
func (l *TernaryLayer) UpdateProbabilisticSTDP(preSynaptic []int16, errorSignal []int16, learningRate uint8) {
	if learningRate == 0 {
		return
	}
	if len(preSynaptic) < l.InputSize || len(errorSignal) < l.OutputSize {
		return
	}

	// Fast xorshift PRNG state — stocat pe layer pentru reproductibilitate.
	// Dacă nu a fost seed-uit explicit (PRNGState==0), folosim o constantă
	// derivată din topologie (deterministă, dar diferită între layere).
	prngState := l.PRNGState
	if prngState == 0 {
		prngState = uint64(l.InputSize)*0x9E3779B97F4A7C15 + uint64(l.OutputSize) + 1
	}

	fastRand8 := func() uint8 {
		prngState ^= prngState << 13
		prngState ^= prngState >> 7
		prngState ^= prngState << 17
		return uint8(prngState)
	}
	// State-ul avansează între apeluri — esențial pentru ca epoci succesive
	// să nu reia aceeași secvență.
	defer func() { l.PRNGState = prngState }()

	for j := 0; j < l.OutputSize; j++ {
		err := errorSignal[j]
		if err == 0 {
			continue // No error for this neuron, skip updates
		}

		rowOffset := j * l.TilesPerRow

		for t := 0; t < l.TilesPerRow; t++ {
			tileIdx := rowOffset + t
			tile := l.Tiles[tileIdx]
			weights := tile.Unpack()
			modified := false

			baseIdx := t * 16
			for pos := 0; pos < 16; pos++ {
				i := baseIdx + pos
				if i >= l.InputSize {
					break
				}

				pre := preSynaptic[i]
				if pre <= 0 {
					continue // No pre-synaptic spike, STDP says no update
				}

				// Probabilistic transition
				if fastRand8() < learningRate {
					w := weights[pos]
					if err > 0 {
						// Hebbian reinforcement: increase weight
						if w == -1 {
							weights[pos] = 0
							modified = true
						} else if w == 0 {
							weights[pos] = 1
							modified = true
						}
					} else {
						// Anti-Hebbian penalty: decrease weight
						if w == 1 {
							weights[pos] = 0
							modified = true
						} else if w == 0 {
							weights[pos] = -1
							modified = true
						}
					}
				}
			}

			// Repack and save if modified
			if modified {
				l.Tiles[tileIdx] = PackTernaryTile(weights)
			}
		}

		// Update biases probabilistically
		if fastRand8() < learningRate {
			if err > 0 && l.Bias[j] < 32767 {
				l.Bias[j]++
			} else if err < 0 && l.Bias[j] > -32768 {
				l.Bias[j]--
			}
		}
	}
}
