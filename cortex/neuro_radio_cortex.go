package cortex

import (
	"math/rand"
)

// ═══════════════════════════════════════════════════════════════════
// NeuroRadioCortex — The Full Unified Cortex
// ═══════════════════════════════════════════════════════════════════
//
// Combines:
//   - NeuroRadioTile array (weights + routing)
//   - RadioBucketIndex (sparse lookup by frequency)
//   - SemanticFreqCodec (meaningful frequency assignment)
//   - OutputNeuronDecoder (indexed token decode)
//   - RadioBus (256-channel signal bus)
//
// Pipeline per tick:
//   1. Input tokens → semantic frequencies → inject onto bus
//   2. BucketIndex: only tiles on active frequencies process
//   3. Each active tile: Forward(ternary × confidence × phase × amplitude)
//   4. Emit result onto emit_freq
//   5. OutputDecoder: active bus freqs → candidate tokens → top token
//
// Cost: O(active_tiles) per tick, NOT O(total_tiles)
// With 1M tiles and 3 active frequencies → ~12K tiles process, not 1M

// RadioBucketIndex maps frequencies to tile indices for sparse activation.
// This is what makes the cortex O(active) instead of O(N).
type RadioBucketIndex struct {
	Buckets [256][]int // freq → list of tile indices
}

// NewRadioBucketIndex builds the bucket index from a tile array.
func NewRadioBucketIndex(tiles []NeuroRadioTile) *RadioBucketIndex {
	idx := &RadioBucketIndex{}
	for i, t := range tiles {
		if t.IsAlive() {
			freq := t.ListenFreq()
			idx.Buckets[freq] = append(idx.Buckets[freq], i)
		}
	}
	return idx
}

// Rebuild reconstructs the index (call after neurogenesis or frequency drift).
func (idx *RadioBucketIndex) Rebuild(tiles []NeuroRadioTile) {
	for i := range idx.Buckets {
		idx.Buckets[i] = idx.Buckets[i][:0]
	}
	for i, t := range tiles {
		if t.IsAlive() {
			freq := t.ListenFreq()
			idx.Buckets[freq] = append(idx.Buckets[freq], i)
		}
	}
}

// TilesOnFreq returns indices of tiles listening on a given frequency.
func (idx *RadioBucketIndex) TilesOnFreq(freq uint8) []int {
	return idx.Buckets[freq]
}

// OutputNeuron represents a token decoder tuned to specific frequencies.
type OutputNeuron struct {
	TokenID int     // Which token this neuron decodes
	Freqs   []uint8 // Frequencies this token responds to
	Bias    int32   // Accumulated bias from learning
}

// OutputNeuronDecoder maps bus frequencies → candidate tokens.
// No brute-force vocabulary scan needed.
type OutputNeuronDecoder struct {
	Neurons []OutputNeuron
	// Inverse index: freq → list of neuron indices
	FreqIndex [256][]int
	// Minimum signal threshold for ActiveChannels during decode (default 5)
	threshold int
}

// NewOutputNeuronDecoder creates a decoder from the semantic codec.
func NewOutputNeuronDecoder(codec *SemanticFreqCodec, vocabSize int) *OutputNeuronDecoder {
	d := &OutputNeuronDecoder{
		Neurons:   make([]OutputNeuron, 0, vocabSize),
		threshold: 5,
	}

	for tokenID := 0; tokenID < vocabSize; tokenID++ {
		freqs := codec.Encode(tokenID)
		neuron := OutputNeuron{
			TokenID: tokenID,
			Freqs:   freqs,
			Bias:    0,
		}
		nIdx := len(d.Neurons)
		d.Neurons = append(d.Neurons, neuron)

		for _, f := range freqs {
			d.FreqIndex[f] = append(d.FreqIndex[f], nIdx)
		}
	}
	return d
}

// Decode returns the top token and its score from the current bus state.
// Only checks tokens whose frequencies are active — no full vocab scan.
func (d *OutputNeuronDecoder) Decode(bus *RadioBus) (tokenID int, score int64) {
	// Find active frequencies
	activeFreqs := bus.ActiveChannels(int32(d.threshold))
	if len(activeFreqs) == 0 {
		return -1, 0
	}

	// Determine max token ID for slice sizing
	maxTID := 0
	for _, freq := range activeFreqs {
		for _, nIdx := range d.FreqIndex[freq] {
			if d.Neurons[nIdx].TokenID > maxTID {
				maxTID = d.Neurons[nIdx].TokenID
			}
		}
	}

	// Use slice instead of map for zero-alloc scoring
	scores := make([]int64, maxTID+1)
	seen := make([]bool, maxTID+1)

	for _, freq := range activeFreqs {
		signal, _ := bus.Read(freq)
		for _, nIdx := range d.FreqIndex[freq] {
			n := &d.Neurons[nIdx]
			scores[n.TokenID] += int64(signal) + int64(n.Bias)
			seen[n.TokenID] = true
		}
	}

	// Find best
	bestID := -1
	var bestScore int64
	for tid := range scores {
		if seen[tid] && scores[tid] > bestScore {
			bestScore = scores[tid]
			bestID = tid
		}
	}
	return bestID, bestScore
}

// DecodeTopK returns top-K tokens and scores.
func (d *OutputNeuronDecoder) DecodeTopK(bus *RadioBus, k int) []TokenScore {
	activeFreqs := bus.ActiveChannels(int32(d.threshold))
	scores := make(map[int]int64)
	for _, freq := range activeFreqs {
		signal, _ := bus.Read(freq)
		for _, nIdx := range d.FreqIndex[freq] {
			n := &d.Neurons[nIdx]
			scores[n.TokenID] += int64(signal) + int64(n.Bias)
		}
	}

	all := make([]TokenScore, 0, len(scores))
	for tid, s := range scores {
		all = append(all, TokenScore{tid, s})
	}
	// Sort descending
	for i := 0; i < len(all); i++ {
		for j := i + 1; j < len(all); j++ {
			if all[j].Score > all[i].Score {
				all[i], all[j] = all[j], all[i]
			}
		}
	}
	if len(all) > k {
		all = all[:k]
	}
	return all
}

// NeuroRadioCortex is the full unified cortex.
type NeuroRadioCortex struct {
	Tiles     []NeuroRadioTile
	Bus       RadioBus
	Index     *RadioBucketIndex
	Codec     *SemanticFreqCodec
	Decoder   *OutputNeuronDecoder
	Size      int
	TickCount uint64
	rng       *rand.Rand

	// Configurable parameters (set from central Config)
	DecodeActiveThreshold int // ActiveChannels threshold in Decode (default 5)
	InitAmpMin            int // Initial amplitude minimum (default 100)
	InitAmpRange          int // Initial amplitude range (default 156)
	InhibitoryRatioDiv    int // Inhibitory ratio divisor: 1/N (default 5)
	InjectAmplitude       int // Inject amplitude for ProcessInput/TrainStep (default 200)

	// Stats tracking
	LastActiveTiles int // How many tiles activated last tick
	LastTotalTiles  int // Total alive tiles
}

// NewNeuroRadioCortex creates a new cortex with the given number of tiles.
func NewNeuroRadioCortex(size int, rng *rand.Rand) *NeuroRadioCortex {
	// Use default config values; callers should override these from Config.
	initAmpMin := 100
	initAmpRange := 156
	inhibitoryDiv := 5
	if inhibitoryDiv <= 0 {
		inhibitoryDiv = 5
	}

	tiles := make([]NeuroRadioTile, size)

	for i := range tiles {
		// Random ternary weights
		weights := rng.Uint32()
		// Random initial confidence (medium)
		confidence := uint32(0x55555555) // All confidence = 1 (low)
		// Random radio params
		listenFreq := uint8(rng.Intn(256))
		phase := uint8(rng.Intn(256))
		amplitude := uint8(initAmpMin + rng.Intn(initAmpRange)) // Start strong
		emitFreq := uint8(rng.Intn(256))                         // full 0-255 emit freq
		inhibitory := rng.Intn(inhibitoryDiv) == 0               // configurable inhibitory ratio

		tiles[i] = NewNeuroRadioTile(weights, confidence, listenFreq, phase, amplitude, emitFreq, inhibitory)
	}

	codec := NewSemanticFreqCodec()
	index := NewRadioBucketIndex(tiles)

	nrc := &NeuroRadioCortex{
		Tiles:                 tiles,
		Index:                 index,
		Codec:                 codec,
		Size:                  size,
		rng:                   rng,
		DecodeActiveThreshold: 5,
		InitAmpMin:            initAmpMin,
		InitAmpRange:          initAmpRange,
		InhibitoryRatioDiv:    inhibitoryDiv,
		InjectAmplitude:       200,
	}

	// Build decoder and wire the decode threshold
	nrc.Decoder = NewOutputNeuronDecoder(codec, 0)
	nrc.Decoder.threshold = nrc.DecodeActiveThreshold

	return nrc
}

// Step runs one tick of the cortex.
// Only tiles on active bus frequencies are processed (sparse activation).
func (nrc *NeuroRadioCortex) Step() {
	emitBus := RadioBus{} // Temporary bus for this tick's emissions
	activeTiles := 0

	// For each active frequency on the bus, process only those tiles
	for freq := uint8(0); ; freq++ {
		signal, phase := nrc.Bus.Read(freq)
		if signal == 0 {
			if freq == 255 {
				break
			}
			continue
		}

		// Get tiles listening on this frequency
		tileIndices := nrc.Index.TilesOnFreq(freq)
		for _, idx := range tileIndices {
			tile := &nrc.Tiles[idx]
			if !tile.IsAlive() {
				continue
			}

			// Simple forward: use bus signal as input magnitude
			// Create a simple input vector from the bus signal
			var input [16]int8
			sigClamped := signal
			if sigClamped > 127 {
				sigClamped = 127
			}
			if sigClamped < -127 {
				sigClamped = -127
			}
			for k := range input {
				input[k] = int8(sigClamped)
			}

			result := tile.Forward(input, signal, phase)
			if result != 0 {
				activeTiles++
				// Emit onto emit frequency
				ef := tile.EmitFreq()
				amp := tile.Amplitude()
				emitBus.Emit(ef, amp, tile.Radio.Phase(), tile.Radio.IsInhibitory())
			}
		}

		if freq == 255 {
			break
		}
	}

	// Swap bus
	nrc.Bus = emitBus
	nrc.TickCount++
	nrc.LastActiveTiles = activeTiles
}

// InjectTokens puts token frequencies onto the bus.
func (nrc *NeuroRadioCortex) InjectTokens(tokenIDs []int, amplitude int16) {
	for _, tid := range tokenIDs {
		freqs := nrc.Codec.Encode(tid)
		for _, f := range freqs {
			nrc.Bus.Emit(f, uint8(amplitude), 128, false)
		}
	}
}

// ProcessInput does a full input→output cycle.
func (nrc *NeuroRadioCortex) ProcessInput(tokenIDs []int, ticks int) (int, int64) {
	nrc.Bus.Clear()
	nrc.InjectTokens(tokenIDs, int16(nrc.InjectAmplitude))

	for t := 0; t < ticks; t++ {
		nrc.Step()
	}

	if nrc.Decoder == nil {
		return -1, 0
	}
	return nrc.Decoder.Decode(&nrc.Bus)
}

// TrainStep does one train iteration: input→forward→compare→Hebbian.
func (nrc *NeuroRadioCortex) TrainStep(inputTokenIDs []int, targetTokenID int, ticks int) int {
	nrc.Bus.Clear()
	nrc.InjectTokens(inputTokenIDs, int16(nrc.InjectAmplitude))

	for t := 0; t < ticks; t++ {
		nrc.Step()
	}

	// Check which output frequencies are active
	targetFreqs := nrc.Codec.Encode(targetTokenID)
	targetSet := make(map[uint8]bool, len(targetFreqs))
	for _, f := range targetFreqs {
		targetSet[f] = true
	}

	// Hebbian: confirm tiles whose emit freq matches target, contradict others
	matches := 0
	for i := range nrc.Tiles {
		tile := &nrc.Tiles[i]
		if !tile.IsAlive() {
			continue
		}

		ef := tile.EmitFreq()
		lf := tile.ListenFreq()
		amp := tile.Amplitude()

		if amp < 10 {
			continue // Dead tile, skip
		}

		isRelevant := targetSet[ef] || targetSet[lf]

		if isRelevant && amp > 0 {
			tile.Confirm()
			matches++
		} else if !isRelevant && amp < 50 {
			tile.Contradict()

			// Drift weak tiles toward target frequencies
			if amp < 20 && len(targetFreqs) > 0 {
				newFreq := targetFreqs[nrc.rng.Intn(len(targetFreqs))]
				tile.Radio.SetFreqListen(newFreq)
			}
		}
	}

	return matches
}

// Neurogenesis replaces dead tiles with fresh ones.
func (nrc *NeuroRadioCortex) Neurogenesis() int {
	if nrc.InhibitoryRatioDiv <= 0 {
		nrc.InhibitoryRatioDiv = 5
	}
	replaced := 0
	for i := range nrc.Tiles {
		if !nrc.Tiles[i].IsAlive() {
			nrc.Tiles[i] = NewNeuroRadioTile(
				nrc.rng.Uint32(),
				0x55555555,
				uint8(nrc.rng.Intn(256)),
				uint8(nrc.rng.Intn(256)),
				uint8(nrc.InitAmpMin+nrc.rng.Intn(nrc.InitAmpRange)),
				uint8(nrc.rng.Intn(256)),
				nrc.rng.Intn(nrc.InhibitoryRatioDiv) == 0,
			)
			replaced++
		}
	}
	if replaced > 0 {
		nrc.Index.Rebuild(nrc.Tiles)
	}
	return replaced
}

// Stats returns cortex statistics.
type NeuroRadioStats struct {
	TotalTiles   int
	AliveTiles   int
	ActiveLast   int
	AvgAmplitude int
	TickCount    uint64
}

func (nrc *NeuroRadioCortex) Stats() NeuroRadioStats {
	alive := 0
	var totalAmp int64
	for i := range nrc.Tiles {
		if nrc.Tiles[i].IsAlive() {
			alive++
			totalAmp += int64(nrc.Tiles[i].Amplitude())
		}
	}
	avgAmp := 0
	if alive > 0 {
		avgAmp = int(totalAmp / int64(alive))
	}
	return NeuroRadioStats{
		TotalTiles:   nrc.Size,
		AliveTiles:   alive,
		ActiveLast:   nrc.LastActiveTiles,
		AvgAmplitude: avgAmp,
		TickCount:    nrc.TickCount,
	}
}
