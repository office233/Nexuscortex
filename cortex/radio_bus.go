package cortex

// RadioBus is a 256-channel frequency bus where neurons emit and receive signals.
//
// Like radio waves in the air: multiple transmitters emit on different frequencies,
// signals on the same frequency ADD together (constructive interference),
// and each receiver tunes to the frequency it wants to hear.
//
// The bus replaces explicit synapse lists. Connectivity is implicit:
// neurons on the same frequency are connected, different frequencies are not.
type RadioBus struct {
	Signal   [256]int32  // combined signal amplitude per frequency
	Phase    [256]uint8  // dominant phase per frequency (from strongest emitter)
	Emitters [256]uint16 // count of neurons that emitted on each frequency
	maxAmp   [256]int32  // track strongest emitter for phase dominance
}

// Emit adds a neuron's signal to its emit frequency on the bus.
// Inhibitory neurons subtract from the signal (suppression).
// Phase is tracked from the strongest emitter on each frequency.
func (b *RadioBus) Emit(freq uint8, amplitude uint8, phase uint8, inhibitory bool) {
	amp := int32(amplitude)
	if inhibitory {
		b.Signal[freq] -= amp
	} else {
		b.Signal[freq] += amp
	}
	b.Emitters[freq]++

	// Track dominant phase (from strongest emitter)
	if amp > b.maxAmp[freq] {
		b.maxAmp[freq] = amp
		b.Phase[freq] = phase
	}
}

// Read returns the combined signal and dominant phase on a frequency.
func (b *RadioBus) Read(freq uint8) (signal int32, phase uint8) {
	return b.Signal[freq], b.Phase[freq]
}

// Clear resets the bus for the next tick.
func (b *RadioBus) Clear() {
	b.Signal = [256]int32{}
	b.Phase = [256]uint8{}
	b.Emitters = [256]uint16{}
	b.maxAmp = [256]int32{}
}

// ActiveChannels returns frequencies where |signal| exceeds the threshold.
func (b *RadioBus) ActiveChannels(threshold int32) []uint8 {
	var result []uint8
	for i := 0; i < 256; i++ {
		sig := b.Signal[i]
		if sig > threshold || sig < -threshold {
			result = append(result, uint8(i))
		}
	}
	return result
}

// Spectrum returns a copy of the signal array for visualization/audio.
func (b *RadioBus) Spectrum() [256]int32 {
	return b.Signal
}

// TotalEnergy returns the sum of absolute signal values across all frequencies.
func (b *RadioBus) TotalEnergy() int64 {
	var total int64
	for i := 0; i < 256; i++ {
		sig := int64(b.Signal[i])
		if sig < 0 {
			sig = -sig
		}
		total += sig
	}
	return total
}

// PeakFrequency returns the frequency with the strongest absolute signal.
func (b *RadioBus) PeakFrequency() (freq uint8, amplitude int32) {
	var bestFreq uint8
	var bestAmp int32
	for i := 0; i < 256; i++ {
		sig := b.Signal[i]
		if sig < 0 {
			sig = -sig
		}
		if sig > bestAmp {
			bestAmp = sig
			bestFreq = uint8(i)
		}
	}
	return bestFreq, bestAmp
}
