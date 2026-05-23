package cortex

import (
	"math/rand"
)

// RadioCortex is a frequency-based neural processor where neurons communicate
// through a shared radio bus instead of explicit synapses.
//
// Key insight: FREQUENCY IS CONNECTIVITY.
// Neurons on the same frequency are connected. No synapse list needed.
// Learning = re-tuning frequencies + adjusting amplitudes.
//
// One tick:
//  1. EMIT: fired neurons broadcast on their emit frequency
//  2. RECEIVE: each neuron reads its listen frequency from the bus
//  3. RESONATE: phase match determines if the neuron activates
//  4. FIRE: neurons above threshold fire and enter refractory period
//  5. CLEAR: bus is reset for next tick
//
// Cost: O(N) per tick. No O(N × synapses) multiplication.
type RadioCortex struct {
	Neurons   []RadioNeuron // all neurons as RGBA32 pixels
	Bus       RadioBus      // 256-channel frequency bus
	Fired     []bool        // who fired this tick
	PrevFired []bool        // who fired last tick (for emit)
	Size      int           // number of neurons
	TickCount uint64        // current tick number

	// Regions: neurons are organized into functional regions
	// Each region occupies a frequency range on the bus
	InputStart  int // first neuron index for input region
	InputEnd    int // last neuron index for input region
	OutputStart int // first neuron index for output region
	OutputEnd   int // last neuron index for output region

	// Config
	FireThreshold int32 // minimum activation to fire (default 64)
	PhaseWindow   uint8 // phase match tolerance (default 90 = ~127°)

	// GPU acceleration (nil = CPU fallback)
	GPU *RadioCUDA

	rng *rand.Rand
}

// RadioCortexStats holds statistics about the radio cortex state.
type RadioCortexStats struct {
	TotalNeurons  int
	AliveNeurons  int
	FiredNeurons  int
	ActiveFreqs   int
	TotalEnergy   int64
	PeakFreq      uint8
	PeakAmplitude int32
	TickCount     uint64
	AvgAmplitude  int
}

// NewRadioCortex creates a frequency-based neural processor.
//
// Neurons are initialized with random frequencies and phases.
// 80% excitatory, 20% inhibitory (matching biological ratio).
// Frequency ranges:
//   - Input region:  neurons 0..inputSize-1 (freq 0-63)
//   - Hidden region: neurons inputSize..size-outputSize-1 (freq 64-191)
//   - Output region: neurons size-outputSize..size-1 (freq 192-255)
func NewRadioCortex(size int, rng *rand.Rand) *RadioCortex {
	inputSize := size / 10  // 10% input neurons
	outputSize := size / 10 // 10% output neurons

	rc := &RadioCortex{
		Neurons:       make([]RadioNeuron, size),
		Fired:         make([]bool, size),
		PrevFired:     make([]bool, size),
		Size:          size,
		InputStart:    0,
		InputEnd:      inputSize,
		OutputStart:   size - outputSize,
		OutputEnd:     size,
		FireThreshold: 64,
		PhaseWindow:   90,
		rng:           rng,
	}

	for i := 0; i < size; i++ {
		var freqListen, freqEmit uint8
		inhibitory := rng.Intn(5) == 0 // 20% inhibitory

		if i < inputSize {
			// Input neurons: listen on freq 0-63, emit on 64-127
			freqListen = uint8(rng.Intn(64))
			freqEmit = uint8(64 + rng.Intn(64))
		} else if i >= size-outputSize {
			// Output neurons: listen on freq 128-191, emit on 192-255
			freqListen = uint8(128 + rng.Intn(64))
			freqEmit = uint8(192 + rng.Intn(64)) // full range output band
		} else {
			// Hidden neurons: full range listen and emit
			freqListen = uint8(rng.Intn(256))
			freqEmit = uint8((int(freqListen) + 32 + rng.Intn(64)) % 256) // full 256
		}

		phase := uint8(rng.Intn(256))
		amplitude := uint8(64 + rng.Intn(128)) // start at medium strength

		rc.Neurons[i] = PackRadioNeuron(freqListen, phase, amplitude, freqEmit, inhibitory)
	}

	return rc
}

// Step runs one tick of the radio cortex.
// Returns the number of neurons that fired.
func (rc *RadioCortex) Step() int {
	// Save previous firing state and clear current
	rc.PrevFired, rc.Fired = rc.Fired, rc.PrevFired
	for i := range rc.Fired {
		rc.Fired[i] = false
	}

	// Phase 1: EMIT — previously fired neurons broadcast on the bus
	for i, fired := range rc.PrevFired {
		if !fired {
			continue
		}
		n := rc.Neurons[i]
		rc.Bus.Emit(n.FreqEmit(), n.Amplitude(), n.Phase(), n.IsInhibitory())
	}

	// Phase 2: RECEIVE + RESONATE + FIRE
	firedCount := 0
	for i := range rc.Neurons {
		n := &rc.Neurons[i]

		// Skip refractory neurons (just fired, cooling down)
		if n.IsRefractory() {
			n.SetRefractory(false) // clear for next tick
			n.AdvancePhase()       // keep oscillating
			continue
		}

		// Skip dead neurons
		if !n.IsAlive() {
			continue
		}

		// Read signal from bus on this neuron's listen frequency
		signal, busPhase := rc.Bus.Read(n.FreqListen())
		if signal == 0 {
			n.AdvancePhase()
			continue
		}

		// Compute resonance (phase match)
		resonance := Resonance(n.Phase(), busPhase)

		// Activation = signal × resonance / 128
		// This is the ONLY multiplication — and it's just scaling
		activation := (signal * int32(resonance)) >> 7

		// Fire if activation exceeds threshold
		if activation > rc.FireThreshold || activation < -rc.FireThreshold {
			rc.Fired[i] = true
			n.SetRefractory(true)
			firedCount++
		}

		// Advance phase (oscillate)
		n.AdvancePhase()
	}

	// Phase 3: CLEAR bus for next tick
	rc.Bus.Clear()
	rc.TickCount++

	return firedCount
}

// InjectSignal manually activates input neurons matching the given frequencies.
// This is how external input enters the radio cortex.
func (rc *RadioCortex) InjectSignal(frequencies []uint8, amplitude uint8) {
	freqSet := make(map[uint8]bool, len(frequencies))
	for _, f := range frequencies {
		freqSet[f] = true
	}

	for i := rc.InputStart; i < rc.InputEnd; i++ {
		n := &rc.Neurons[i]
		if freqSet[n.FreqListen()%64] { // input neurons use freq 0-63
			rc.Bus.Emit(n.FreqEmit(), amplitude, n.Phase(), n.IsInhibitory())
			rc.Fired[i] = true
		}
	}
}

// InjectSDR converts an SDR into frequency activations on the bus.
// Active SDR bits are mapped to frequencies: freq = bit_index % 256
func (rc *RadioCortex) InjectSDR(sdr SDR) {
	indices := sdr.ActiveIndices()
	for _, idx := range indices {
		freq := uint8(idx % 256)
		rc.Bus.Emit(freq, 200, uint8(idx%256), false)
	}
	// Also activate matching input neurons
	for i := rc.InputStart; i < rc.InputEnd; i++ {
		n := &rc.Neurons[i]
		signal, busPhase := rc.Bus.Read(n.FreqListen())
		if signal > 0 {
			resonance := Resonance(n.Phase(), busPhase)
			if resonance > 32 { // reasonable match
				rc.Fired[i] = true
			}
		}
	}
}

// ReadFiringPattern returns which neurons fired as a boolean slice.
func (rc *RadioCortex) ReadFiringPattern() []bool {
	result := make([]bool, rc.Size)
	copy(result, rc.Fired)
	return result
}

// ReadOutputSDR reads the output region's firing pattern as an SDR.
func (rc *RadioCortex) ReadOutputSDR(sdrSize int) SDR {
	sdr := NewSDR(sdrSize)
	for i := rc.OutputStart; i < rc.OutputEnd; i++ {
		if rc.Fired[i] {
			// Map output neuron index to SDR bit position
			bitIdx := (i - rc.OutputStart) * sdrSize / (rc.OutputEnd - rc.OutputStart)
			if bitIdx >= 0 && bitIdx < sdrSize {
				sdr.Set(bitIdx)
			}
		}
	}
	return sdr
}

// Confirm strengthens neurons that contributed to a correct output (Hebbian LTP).
// Amplitude increases by 1 (capped at 255).
func (rc *RadioCortex) Confirm() {
	for i := range rc.Neurons {
		if !rc.Fired[i] {
			continue
		}
		n := &rc.Neurons[i]
		amp := n.Amplitude()
		if amp < 255 {
			n.SetAmplitude(amp + 1)
		}
	}
}

// Contradict weakens neurons that contributed to a wrong output (Hebbian LTD).
// Amplitude decreases by 1. Weak neurons drift their frequency (re-tune).
func (rc *RadioCortex) Contradict() {
	for i := range rc.Neurons {
		if !rc.Fired[i] {
			continue
		}
		n := &rc.Neurons[i]
		amp := n.Amplitude()
		if amp > 0 {
			n.SetAmplitude(amp - 1)
		}
		// Weak neurons re-tune: drift frequency by ±1
		if amp < 32 {
			freq := n.FreqListen()
			if rc.rng.Intn(2) == 0 {
				n.SetFreqListen(freq + 1) // wraps via uint8
			} else {
				n.SetFreqListen(freq - 1) // wraps via uint8
			}
		}
	}
}

// Neurogenesis replaces dead neurons (amplitude=0) with fresh random ones.
// Returns the number of neurons replaced.
func (rc *RadioCortex) Neurogenesis() int {
	replaced := 0
	for i := range rc.Neurons {
		if rc.Neurons[i].IsAlive() {
			continue
		}

		// Determine region for frequency assignment
		var freqListen, freqEmit uint8
		inhibitory := rc.rng.Intn(5) == 0

		if i < rc.InputEnd {
			freqListen = uint8(rc.rng.Intn(64))
			freqEmit = uint8(64 + rc.rng.Intn(64))
		} else if i >= rc.OutputStart {
			freqListen = uint8(128 + rc.rng.Intn(64))
			freqEmit = uint8(192 + rc.rng.Intn(64))
		} else {
			freqListen = uint8(rc.rng.Intn(256))
			freqEmit = uint8((int(freqListen) + 32 + rc.rng.Intn(64)) % 256)
		}

		rc.Neurons[i] = PackRadioNeuron(
			freqListen,
			uint8(rc.rng.Intn(256)),
			uint8(64+rc.rng.Intn(64)), // moderate starting amplitude
			freqEmit,
			inhibitory,
		)
		replaced++
	}
	return replaced
}

// Stats returns current statistics about the radio cortex.
func (rc *RadioCortex) Stats() RadioCortexStats {
	alive := 0
	fired := 0
	var ampSum int64

	for i := range rc.Neurons {
		if rc.Neurons[i].IsAlive() {
			alive++
			ampSum += int64(rc.Neurons[i].Amplitude())
		}
		if rc.Fired[i] {
			fired++
		}
	}

	avgAmp := 0
	if alive > 0 {
		avgAmp = int(ampSum / int64(alive))
	}

	peakFreq, peakAmp := rc.Bus.PeakFrequency()

	return RadioCortexStats{
		TotalNeurons:  rc.Size,
		AliveNeurons:  alive,
		FiredNeurons:  fired,
		ActiveFreqs:   len(rc.Bus.ActiveChannels(rc.FireThreshold)),
		TotalEnergy:   rc.Bus.TotalEnergy(),
		PeakFreq:      peakFreq,
		PeakAmplitude: peakAmp,
		TickCount:     rc.TickCount,
		AvgAmplitude:  avgAmp,
	}
}

// InitGPU attempts to initialize CUDA acceleration.
// Call this once after creating the RadioCortex.
// Returns true if GPU is available and initialized.
func (rc *RadioCortex) InitGPU() bool {
	gpu := NewRadioCUDA(rc.Size)
	if gpu == nil {
		return false
	}

	// Upload all neurons to GPU
	neurons := make([]uint32, rc.Size)
	for i, n := range rc.Neurons {
		neurons[i] = uint32(n)
	}
	gpu.UploadNeurons(neurons)
	gpu.ClearBus()

	rc.GPU = gpu
	return true
}

// StepGPU runs N ticks entirely on GPU memory.
// Neurons stay on GPU between ticks — no PCIe transfer per tick.
// This is the key optimization: 20 ticks in ~0.1ms instead of ~50ms on CPU.
func (rc *RadioCortex) StepGPU(nTicks int) {
	if rc.GPU == nil || !rc.GPU.IsAvailable() {
		// Fallback to CPU
		for i := 0; i < nTicks; i++ {
			rc.Step()
		}
		return
	}

	rc.GPU.StepN(nTicks, rc.FireThreshold, int32(rc.PhaseWindow))
	rc.TickCount += uint64(nTicks)
}

// InjectGPU injects frequencies into the GPU bus.
func (rc *RadioCortex) InjectGPU(freqs []uint8, amplitude int16) {
	if rc.GPU == nil || !rc.GPU.IsAvailable() {
		// CPU fallback: use InjectSignal
		rc.InjectSignal(freqs, uint8(amplitude))
		return
	}

	amps := make([]int16, len(freqs))
	for i := range amps {
		amps[i] = amplitude
	}
	rc.GPU.Inject(freqs, amps)
}

// HebbianGPU runs confirm/contradict learning on GPU.
func (rc *RadioCortex) HebbianGPU(targetFreqs []uint8) {
	if rc.GPU == nil || !rc.GPU.IsAvailable() {
		return
	}

	var mask [256]int32
	for _, f := range targetFreqs {
		mask[f] = 1
	}
	rc.GPU.Hebbian(&mask)
}

// SyncFromGPU downloads neuron state from GPU back to CPU.
// Call this after GPU training to get updated neurons.
func (rc *RadioCortex) SyncFromGPU() {
	if rc.GPU == nil || !rc.GPU.IsAvailable() {
		return
	}

	neurons := make([]uint32, rc.Size)
	rc.GPU.DownloadNeurons(neurons)
	for i, n := range neurons {
		rc.Neurons[i] = RadioNeuron(n)
	}
}

// SyncToGPU uploads current CPU neuron state to GPU.
// Call this after neurogenesis or other CPU-side changes.
func (rc *RadioCortex) SyncToGPU() {
	if rc.GPU == nil || !rc.GPU.IsAvailable() {
		return
	}

	neurons := make([]uint32, rc.Size)
	for i, n := range rc.Neurons {
		neurons[i] = uint32(n)
	}
	rc.GPU.UploadNeurons(neurons)
}

