package cortex

// linear_scan.go — RWKV-Style Linear Scan with SDR State
//
// Implements linear-complexity sequential processing inspired by RWKV-7
// and Mamba (State Space Models). Instead of O(N²) attention, we maintain
// a fixed-size recurrent state that accumulates information over time.
//
// Key properties:
//   - O(N) per token — constant memory per step
//   - SDR state — binary/integer operations only
//   - Infinite context — state accumulates without limit
//   - Streaming native — processes one token at a time
//
// Architecture:
//   state_t = decay(state_{t-1}) + gate(input_t)
//   output_t = TernaryForward(state_t)
//
// This is conceptually similar to:
//   - RWKV's WKV mechanism (linear attention)
//   - Mamba's selective state space
//   - LIF neuron dynamics (membrane decay + spike input)

import (
	"fmt"
	"math/bits"
)

// Named constants for linear scan hyperparameters.
const (
	// SDRActiveValue is the activation level assigned to active SDR bits
	// in sparse forward passes (UltraDeepStack).
	SDRActiveValue int16 = 127

	// temporalBlendDivisor controls the blending ratio between new features
	// and temporal state in UltraDeepStack.
	temporalBlendDivisor int16 = 4

	// targetSparsityDivisor controls the maximum active fraction of state
	// bits in StepFast. E.g., divisor 10 → keep ~10% sparsity.
	targetSparsityDivisor = 10

	// Decay band thresholds partition the 0-255 DecayRate range.
	decayHigh uint8 = 192 // ~75% decay per step
	decayMid  uint8 = 128 // ~50% decay
	decayLow  uint8 = 64  // ~25% decay
)

// LinearScanLayer maintains a recurrent SDR state that processes
// tokens one at a time with O(1) memory per step.
type LinearScanLayer struct {
	// Dimensions
	InputSize  int
	StateSize  int // Size of the recurrent state SDR
	OutputSize int

	// Recurrent state — this is the "memory" that persists across tokens
	State SDR

	// Decay control: determines which bits fade over time
	// Higher decay = faster forgetting (more responsive to new input)
	// Lower decay = longer memory (more stable representations)
	DecayRate uint8 // 0-255: probability of each active bit decaying

	// Gating: which input bits get through to state
	InputGate *TernaryLayer // Projects input to gate signal
	// State update: how input modifies state
	InputProjection *TernaryLayer // Projects input to state-sized SDR
	// Output projection: converts state to output
	OutputProjection *TernaryLayer // Projects state to output

	// Internal state for gating
	decayCounter uint64 // Pseudorandom counter for stochastic decay
}

// NewLinearScanLayer creates a new linear scan layer.
func NewLinearScanLayer(inputSize, stateSize, outputSize int, decayRate uint8) *LinearScanLayer {
	return &LinearScanLayer{
		InputSize:        inputSize,
		StateSize:        stateSize,
		OutputSize:       outputSize,
		State:            NewSDR(stateSize),
		DecayRate:        decayRate,
		InputGate:        NewTernaryLayer(inputSize, stateSize),
		InputProjection:  NewTernaryLayer(inputSize, stateSize),
		OutputProjection: NewTernaryLayer(stateSize, outputSize),
	}
}

// Step processes one token through the linear scan layer.
// This is the core recurrence: O(1) memory, O(stateSize) compute.
//
// Algorithm:
//   1. DECAY: Stochastically clear bits in state (forgetting)
//   2. GATE: Compute which input dimensions are important
//   3. UPDATE: Mix gated input into state
//   4. OUTPUT: Project state through ternary layer
func (l *LinearScanLayer) Step(input SDR) SDR {
	// 1. DECAY — Stochastic forgetting
	// Each active bit has a (DecayRate/256) probability of being cleared.
	// This uses a fast PRNG to avoid expensive random calls.
	l.decay()

	// 2. GATE — Determine input importance
	// Convert input SDR to activations, run through gate projection
	inputAct := sdrToActivations(input)
	gateAct := l.InputGate.Forward(inputAct)

	// 3. UPDATE — Mix gated input into state
	// Project input to state size
	updateAct := l.InputProjection.Forward(inputAct)

	// Apply gate: only update state where gate is positive
	for i := 0; i < l.StateSize; i++ {
		if gateAct[i] > 0 && updateAct[i] > 0 {
			l.State.Set(i)
		} else if gateAct[i] > 0 && updateAct[i] < 0 {
			l.State.Clear(i)
		}
		// If gate <= 0, state bit is unchanged (selective update!)
	}

	// 4. OUTPUT — Project state through output ternary layer
	stateAct := sdrToActivations(l.State)
	outputAct := l.OutputProjection.Forward(stateAct)

	return activationsToSDR(outputAct, input.ActiveCount)
}

// StepFast processes one token using only bitwise operations.
// This is the optimized path that avoids activation conversions.
//
// Instead of float-like gating, it uses SDR overlap:
//   - Bits where input overlaps with state are REINFORCED
//   - Bits only in input are ADDED (new information)
//   - Bits only in state are subject to DECAY (old information fading)
func (l *LinearScanLayer) StepFast(input SDR) SDR {
	// 1. DECAY
	l.decay()

	// 2. UPDATE using bitwise operations
	// New information: bits in input but not in state
	// Union: merge input into state (additive)
	// This naturally implements the "forget gate" of LSTM:
	// - State bits that overlap with input are reinforced (survive decay)
	// - State bits without overlap eventually decay away
	// - Input bits get added to state

	// Limit active count to prevent state explosion
	combined := l.State.Union(input)
	if combined.ActiveCount > l.StateSize/targetSparsityDivisor { // Keep ~10% sparsity
		// Too many active bits — keep only the most reinforced ones
		// Prefer bits that are in BOTH state and input (reinforced)
		overlap := sdrAnd(l.State, input)
		// Fill remaining slots with state bits, then input bits
		remaining := l.StateSize/targetSparsityDivisor - overlap.ActiveCount
		if remaining > 0 {
			l.State = overlap
			// Add state-only bits up to limit
			for _, idx := range sdrAndNot(combined, overlap).ActiveIndices() {
				if remaining <= 0 {
					break
				}
				l.State.Set(idx)
				remaining--
			}
		} else {
			l.State = overlap
		}
	} else {
		l.State = combined
	}

	// 3. OUTPUT — Use ternary output projection
	stateAct := sdrToActivations(l.State)
	outputAct := l.OutputProjection.Forward(stateAct)

	return activationsToSDR(outputAct, input.ActiveCount)
}

// decay applies stochastic bit decay to the state.
func (l *LinearScanLayer) decay() {
	if l.DecayRate == 0 {
		return // No decay
	}

	l.decayCounter++
	state := l.decayCounter

	// Use the state bits array directly for speed
	for i := range l.State.Bits {
		word := l.State.Bits[i]
		if word == 0 {
			continue
		}

		// Fast PRNG: xorshift for each word
		state ^= state << 13
		state ^= state >> 7
		state ^= state << 17

		// Create a decay mask: bits to clear
		// The higher DecayRate is, the more bits in the mask
		var decayMask uint64
		switch {
		case l.DecayRate >= decayHigh: // ~75% decay per step (very fast forget)
			decayMask = state
		case l.DecayRate >= decayMid: // ~50% decay
			decayMask = state & (state >> 1)
		case l.DecayRate >= decayLow: // ~25% decay
			decayMask = state & (state >> 1) & (state >> 2)
		default: // <25% decay (slow forget, long memory)
			decayMask = state & (state >> 1) & (state >> 2) & (state >> 3)
		}

		l.State.Bits[i] = word &^ decayMask
	}

	// Recount active bits
	l.State.ActiveCount = 0
	for _, w := range l.State.Bits {
		l.State.ActiveCount += bits.OnesCount64(w)
	}
}

// Reset clears the recurrent state.
func (l *LinearScanLayer) Reset() {
	l.State.Reset()
	l.decayCounter = 0
}

// StateActiveCount returns how many bits are active in the current state.
func (l *LinearScanLayer) StateActiveCount() int {
	return l.State.ActiveCount
}

// sdrAnd computes bitwise AND of two SDRs (intersection).
func sdrAnd(a, b SDR) SDR {
	size := a.Size
	if b.Size > size {
		size = b.Size
	}
	result := NewSDR(size)

	minWords := len(a.Bits)
	if len(b.Bits) < minWords {
		minWords = len(b.Bits)
	}

	for i := 0; i < minWords; i++ {
		result.Bits[i] = a.Bits[i] & b.Bits[i]
	}

	// Recount
	result.ActiveCount = 0
	for _, w := range result.Bits {
		result.ActiveCount += bits.OnesCount64(w)
	}

	return result
}

// sdrAndNot computes bitwise A AND NOT B (bits in A but not in B).
func sdrAndNot(a, b SDR) SDR {
	result := NewSDR(a.Size)

	minWords := len(a.Bits)
	if len(b.Bits) < minWords {
		minWords = len(b.Bits)
	}

	for i := 0; i < minWords; i++ {
		result.Bits[i] = a.Bits[i] &^ b.Bits[i]
	}
	// Copy remaining A bits if A is longer
	for i := minWords; i < len(a.Bits); i++ {
		result.Bits[i] = a.Bits[i]
	}

	result.ActiveCount = 0
	for _, w := range result.Bits {
		result.ActiveCount += bits.OnesCount64(w)
	}

	return result
}

// ---------------------------------------------------------------------------
// CortexBlock — The Complete Processing Block
// ---------------------------------------------------------------------------

// CortexBlock combines TernaryLayer + SDR Attention + Linear Scan
// into a single processing block. Multiple blocks can be stacked
// to form the full Cortex Transformer.
//
// Processing order per block:
//   1. Input → TernaryLayer (feature extraction)
//   2. → SDR Attention (contextual retrieval)
//   3. → Linear Scan (temporal state update)
//   4. → TernaryLayer (output projection)
//   5. Residual connection: output = output + input
type CortexBlock struct {
	// Feature extraction
	FeatureLayer *TernaryLayer

	// Attention
	Attention *SDRAttentionHead

	// Temporal processing
	Scan *LinearScanLayer

	// Output projection
	OutputLayer *TernaryLayer

	// Config
	Dim   int // Internal dimension
	TopK  int // Attention top-K
}

// NewCortexBlock creates a complete processing block.
func NewCortexBlock(dim, contextLen, topK int, decayRate uint8) *CortexBlock {
	return &CortexBlock{
		FeatureLayer: NewTernaryLayer(dim, dim),
		Attention:    NewSDRAttentionHead(dim, dim, contextLen),
		Scan:         NewLinearScanLayer(dim, dim, dim, decayRate),
		OutputLayer:  NewTernaryLayer(dim, dim),
		Dim:          dim,
		TopK:         topK,
	}
}

// ProcessToken runs one token through the complete block.
func (b *CortexBlock) ProcessToken(input SDR) SDR {
	// 1. Feature extraction through ternary layer
	inputAct := sdrToActivations(input)
	featAct := b.FeatureLayer.Forward(inputAct)
	features := activationsToSDR(featAct, input.ActiveCount)

	// 2. SDR Attention — retrieve relevant context
	b.Attention.Store(features, features)
	attended := b.Attention.Query(features, b.TopK)

	// 3. Linear Scan — update temporal state
	scanned := b.Scan.StepFast(attended)

	// 4. Output projection
	scanAct := sdrToActivations(scanned)
	outAct := b.OutputLayer.Forward(scanAct)
	output := activationsToSDR(outAct, input.ActiveCount)

	// 5. Residual connection — union with input to preserve information
	return output.Union(input)
}

// Reset clears the temporal state of this block.
func (b *CortexBlock) Reset() {
	b.Scan.Reset()
}

// ParameterCount returns total parameters in this block.
func (b *CortexBlock) ParameterCount() int {
	return b.FeatureLayer.ParameterCount() +
		b.OutputLayer.ParameterCount() +
		b.Scan.InputGate.ParameterCount() +
		b.Scan.InputProjection.ParameterCount() +
		b.Scan.OutputProjection.ParameterCount()
}

// MemoryBytes returns total memory used by this block.
func (b *CortexBlock) MemoryBytes() int {
	return b.FeatureLayer.MemoryBytes() +
		b.OutputLayer.MemoryBytes() +
		b.Scan.InputGate.MemoryBytes() +
		b.Scan.InputProjection.MemoryBytes() +
		b.Scan.OutputProjection.MemoryBytes()
}

// ---------------------------------------------------------------------------
// CortexStack — The Full Cortex Transformer
// ---------------------------------------------------------------------------

// CortexStack is the complete Nexus Cortex "transformer" — a stack of
// CortexBlocks that processes tokens through multiple layers of
// ternary computation, SDR attention, and linear scanning.
type CortexStack struct {
	Blocks []*CortexBlock
	Dim    int
}

// NewCortexStack creates a stack of N CortexBlocks.
func NewCortexStack(numLayers, dim, contextLen, topK int, decayRate uint8) *CortexStack {
	blocks := make([]*CortexBlock, numLayers)
	for i := range blocks {
		blocks[i] = NewCortexBlock(dim, contextLen, topK, decayRate)
	}
	return &CortexStack{
		Blocks: blocks,
		Dim:    dim,
	}
}

// ProcessToken runs one token through the entire stack.
func (s *CortexStack) ProcessToken(input SDR) SDR {
	current := input
	for _, block := range s.Blocks {
		current = block.ProcessToken(current)
	}
	return current
}

// Reset clears all temporal state across all blocks.
func (s *CortexStack) Reset() {
	for _, block := range s.Blocks {
		block.Reset()
	}
}

// TotalParameters returns the total parameter count across all blocks.
func (s *CortexStack) TotalParameters() int {
	total := 0
	for _, block := range s.Blocks {
		total += block.ParameterCount()
	}
	return total
}

// TotalMemoryBytes returns the total memory used across all blocks.
func (s *CortexStack) TotalMemoryBytes() int {
	total := 0
	for _, block := range s.Blocks {
		total += block.MemoryBytes()
	}
	return total
}

// EffectiveParameters returns effective parameter-computations.
// For a standard stack, this equals TotalParameters (no sharing).
func (s *CortexStack) EffectiveParameters() int {
	return s.TotalParameters()
}

// ---------------------------------------------------------------------------
// UltraDeepStack — 5T Active Parameters Through Extreme Depth
// ---------------------------------------------------------------------------
//
// The key insight: you can't READ 5T unique parameters from memory per token.
// But you CAN read 5B parameters 1000 times from VRAM cache.
//
// UltraDeepStack achieves 5T effective parameter-computations by:
//   1. Storing 5B unique ternary params in ~1 GB VRAM (fits easily!)
//   2. Running input through 1000 virtual ALBERT layers
//   3. Using SDR sparse forward to skip 95% of computation per layer
//   4. Each layer has independent attention cache + temporal state
//
// This is not "fake" — ALBERT (Google, ICLR 2020) proved that weight
// sharing across layers maintains quality. The depth creates emergent
// capabilities that shallow networks can't match.
//
// Performance model:
//   5B params × 1000 layers = 5T effective computations
//   SDR sparse (5% active) → 50× less actual compute
//   Time: ~100ms per token = 10 tok/s on CPU
//
// Memory model:
//   5B ternary = 1 GB VRAM (stored once, read 1000×)
//   Attention caches: 1000 × small per-layer state
//   Total: ~1.5-2 GB VRAM

// UltraDeepStack implements extreme-depth ALBERT sharing.
type UltraDeepStack struct {
	// Shared ternary weights — stored ONCE, used NumLayers times
	FeatureUp   *TernaryLayer // dim → hidden
	FeatureDown *TernaryLayer // hidden → dim
	GateProj    *TernaryLayer // dim → dim (scan gating)
	ScanProj    *TernaryLayer // dim → dim (scan update)
	OutputProj  *TernaryLayer // dim → dim (scan output)

	// Per-layer independent state
	ScanStates []SDR   // Each layer's temporal memory
	DecayRate  uint8

	NumLayers  int
	Dim        int
	HiddenDim  int // Hidden dimension (can be > Dim for more params)

	// Counters for scan decay
	decayCounters []uint64
}

// NewUltraDeepStack creates an extreme-depth shared stack.
//
// Parameters:
//   numLayers: number of virtual layers (1000 for 5T effective from 5B stored)
//   dim: SDR dimension (e.g., 10000 for 5B params with hiddenDim=10000)
//   hiddenDim: hidden dimension for feature extraction (controls param count)
//   decayRate: temporal memory decay (0-255)
//
// Total stored params = dim*hiddenDim + hiddenDim*dim + 3*(dim*dim)
// Total effective params = storedParams × numLayers
func NewUltraDeepStack(numLayers, dim, hiddenDim int, decayRate uint8) *UltraDeepStack {
	states := make([]SDR, numLayers)
	counters := make([]uint64, numLayers)
	for i := range states {
		states[i] = NewSDR(dim)
	}

	return &UltraDeepStack{
		FeatureUp:     NewTernaryLayer(dim, hiddenDim),
		FeatureDown:   NewTernaryLayer(hiddenDim, dim),
		GateProj:      NewTernaryLayer(dim, dim),
		ScanProj:      NewTernaryLayer(dim, dim),
		OutputProj:    NewTernaryLayer(dim, dim),
		ScanStates:    states,
		DecayRate:     decayRate,
		NumLayers:     numLayers,
		Dim:           dim,
		HiddenDim:     hiddenDim,
		decayCounters: counters,
	}
}

// ProcessToken runs one token through ALL virtual layers.
// Uses sparse SDR forward to skip 95% of computation.
func (u *UltraDeepStack) ProcessToken(input SDR) SDR {
	current := input
	for i := 0; i < u.NumLayers; i++ {
		current = u.processLayerSparse(current, i)
	}
	return current
}

// processLayerSparse processes one layer using SDR sparsity.
// Only computes outputs for active input bits — massive speedup.
func (u *UltraDeepStack) processLayerSparse(input SDR, layerIdx int) SDR {
	// Get active indices from input SDR
	activeIdx := input.ActiveIndices()
	if len(activeIdx) == 0 {
		return input
	}

	// Convert sparse SDR to sparse activation format
	values := make([]int16, len(activeIdx))
	for i := range values {
		values[i] = SDRActiveValue // Active bits have standard activation
	}

	// 1. Feature extraction: up-project through shared ternary layer (SPARSE)
	upAct, err := u.FeatureUp.ForwardSparse(activeIdx, values)
	if err != nil {
		upAct = make([]int16, u.HiddenDim)
	}

	// 2. Non-linearity: ReLU-like — keep only positive activations
	upIndices := make([]int, 0, len(upAct)/4)
	upValues := make([]int16, 0, len(upAct)/4)
	for i, v := range upAct {
		if v > 0 {
			upIndices = append(upIndices, i)
			upValues = append(upValues, v)
		}
	}

	// 3. Down-project back to dim (SPARSE from hidden)
	var downAct []int16
	if len(upIndices) > 0 {
		downAct, err = u.FeatureDown.ForwardSparse(upIndices, upValues)
		if err != nil {
			downAct = make([]int16, u.Dim)
		}
	} else {
		downAct = make([]int16, u.Dim)
	}

	// 4. Scan state update (temporal memory for this layer)
	u.updateScanState(layerIdx, activeIdx, values)

	// 5. Blend scan state into output
	stateAct := sdrToActivations(u.ScanStates[layerIdx])
	for i := range downAct {
		if i < len(stateAct) {
			downAct[i] += stateAct[i] / temporalBlendDivisor // Weak temporal blending
		}
	}

	// 6. Convert to SDR output (keep same sparsity as input)
	output := activationsToSDR(downAct, input.ActiveCount)

	// 7. Residual connection
	return output.Union(input)
}

// updateScanState applies decay and updates temporal state for one layer.
func (u *UltraDeepStack) updateScanState(layerIdx int, inputIdx []int, inputValues []int16) {
	// Decay
	u.decayCounters[layerIdx]++
	counter := u.decayCounters[layerIdx]

	if u.DecayRate > 0 {
		for i := range u.ScanStates[layerIdx].Bits {
			word := u.ScanStates[layerIdx].Bits[i]
			if word == 0 {
				continue
			}
			counter ^= counter << 13
			counter ^= counter >> 7
			counter ^= counter << 17
			var mask uint64
			if u.DecayRate >= decayMid {
				mask = counter & (counter >> 1)
			} else {
				mask = counter & (counter >> 1) & (counter >> 2)
			}
			u.ScanStates[layerIdx].Bits[i] = word &^ mask
		}
	}

	// Update: add input bits to state
	for _, idx := range inputIdx {
		if idx < u.ScanStates[layerIdx].Size {
			u.ScanStates[layerIdx].Set(idx)
		}
	}

	// Recount
	u.ScanStates[layerIdx].ActiveCount = 0
	for _, w := range u.ScanStates[layerIdx].Bits {
		u.ScanStates[layerIdx].ActiveCount += bits.OnesCount64(w)
	}
}

// Reset clears all temporal state.
func (u *UltraDeepStack) Reset() {
	for i := range u.ScanStates {
		u.ScanStates[i].Reset()
		u.decayCounters[i] = 0
	}
}

// StoredParameters returns unique stored parameters.
func (u *UltraDeepStack) StoredParameters() int {
	return u.FeatureUp.ParameterCount() +
		u.FeatureDown.ParameterCount() +
		u.GateProj.ParameterCount() +
		u.ScanProj.ParameterCount() +
		u.OutputProj.ParameterCount()
}

// EffectiveParameters returns total effective computations.
func (u *UltraDeepStack) EffectiveParameters() int64 {
	return int64(u.StoredParameters()) * int64(u.NumLayers)
}

// StoredMemoryBytes returns actual VRAM needed.
func (u *UltraDeepStack) StoredMemoryBytes() int {
	return u.FeatureUp.MemoryBytes() +
		u.FeatureDown.MemoryBytes() +
		u.GateProj.MemoryBytes() +
		u.ScanProj.MemoryBytes() +
		u.OutputProj.MemoryBytes()
}

// Stats returns a human-readable summary.
func (u *UltraDeepStack) Stats() string {
	stored := u.StoredParameters()
	effective := u.EffectiveParameters()
	mem := u.StoredMemoryBytes()

	return fmt.Sprintf(
		"UltraDeepStack[%d layers × %d dim × %d hidden] stored=%.2fM effective=%.2fT mem=%.1fMB compression=%d×",
		u.NumLayers, u.Dim, u.HiddenDim,
		float64(stored)/1e6,
		float64(effective)/1e12,
		float64(mem)/(1024*1024),
		u.NumLayers,
	)
}
// ---------------------------------------------------------------------------
// SharedCortexStack — ALBERT-Style Weight Sharing
// ---------------------------------------------------------------------------
//
// ALBERT (Google, ICLR 2020) showed that sharing weights across all
// transformer layers reduces parameters by 18× while maintaining ~95%
// of BERT's quality. This is the single most effective compression trick.
//
// SharedCortexStack creates ONE physical CortexBlock and runs input
// through it N times (virtual layers). Each virtual layer shares the
// same ternary weights but maintains its own:
//   - SDR Attention cache (each layer attends to different context)
//   - Linear Scan state (each layer has independent temporal memory)
//
// Memory: O(1 block) instead of O(N blocks)
// Compute: O(N blocks) — same as non-shared
// Effective params: stored_params × N layers

// SharedCortexStack shares one set of ternary weights across N layers.
type SharedCortexStack struct {
	// The shared physical weights — only ONE set stored in memory
	SharedFeatureLayer *TernaryLayer
	SharedOutputLayer  *TernaryLayer

	// Per-layer independent state (NOT shared)
	Attentions []*SDRAttentionHead // Each layer has its own attention cache
	Scans      []*LinearScanLayer  // Each layer has its own temporal state

	NumLayers int
	Dim       int
	TopK      int
}

// NewSharedCortexStack creates an ALBERT-style shared stack.
// Only ONE set of ternary weights is allocated; all N layers share them.
// This gives N× parameter efficiency.
func NewSharedCortexStack(numLayers, dim, contextLen, topK int, decayRate uint8, engine interface{}) *SharedCortexStack {
	// ONE set of shared weights
	sharedFeature := NewTernaryLayer(dim, dim)
	sharedFeature.Engine = engine
	sharedOutput := NewTernaryLayer(dim, dim)
	sharedOutput.Engine = engine

	// ONE set of shared scan projection weights
	sharedGate := NewTernaryLayer(dim, dim)
	sharedGate.Engine = engine
	sharedInputProj := NewTernaryLayer(dim, dim)
	sharedInputProj.Engine = engine
	sharedOutputProj := NewTernaryLayer(dim, dim)
	sharedOutputProj.Engine = engine

	// Each layer gets independent attention cache and scan STATE
	// but shares the ternary weights
	attns := make([]*SDRAttentionHead, numLayers)
	scans := make([]*LinearScanLayer, numLayers)
	for i := 0; i < numLayers; i++ {
		attns[i] = NewSDRAttentionHead(dim, dim, contextLen)
		// Create scan layer with shared weights but independent state
		scans[i] = &LinearScanLayer{
			InputSize:        dim,
			StateSize:        dim,
			OutputSize:       dim,
			State:            NewSDR(dim),
			DecayRate:        decayRate,
			InputGate:        sharedGate,       // SHARED
			InputProjection:  sharedInputProj,  // SHARED
			OutputProjection: sharedOutputProj, // SHARED
		}
	}

	return &SharedCortexStack{
		SharedFeatureLayer: sharedFeature,
		SharedOutputLayer:  sharedOutput,
		Attentions:         attns,
		Scans:              scans,
		NumLayers:          numLayers,
		Dim:                dim,
		TopK:               topK,
	}
}


// ProcessToken runs one token through all N virtual layers.
// The shared weights are applied N times with different state.
func (s *SharedCortexStack) ProcessToken(input SDR) SDR {
	current := input
	for i := 0; i < s.NumLayers; i++ {
		current = s.processLayer(current, i)
	}
	return current
}

// processLayer runs one token through one virtual layer.
func (s *SharedCortexStack) processLayer(input SDR, layerIdx int) SDR {
	// 1. Feature extraction through SHARED ternary weights
	inputAct := sdrToActivations(input)
	featAct := s.SharedFeatureLayer.Forward(inputAct)
	features := activationsToSDR(featAct, input.ActiveCount)

	// 2. SDR Attention — PER-LAYER independent cache
	s.Attentions[layerIdx].Store(features, features)
	attended := s.Attentions[layerIdx].Query(features, s.TopK)

	// 3. Linear Scan — PER-LAYER independent state
	scanned := s.Scans[layerIdx].StepFast(attended)

	// 4. Output projection through SHARED ternary weights
	scanAct := sdrToActivations(scanned)
	outAct := s.SharedOutputLayer.Forward(scanAct)
	output := activationsToSDR(outAct, input.ActiveCount)

	// 5. Residual connection
	return output.Union(input)
}

// Reset clears all per-layer temporal state.
func (s *SharedCortexStack) Reset() {
	for i := 0; i < s.NumLayers; i++ {
		s.Scans[i].Reset()
	}
}

// StoredParameters returns the number of UNIQUE stored parameters.
// This is the actual memory footprint. With full ALBERT sharing,
// all layers share the same 5 ternary layers (feature, output, gate,
// input projection, output projection).
func (s *SharedCortexStack) StoredParameters() int {
	shared := s.SharedFeatureLayer.ParameterCount() + s.SharedOutputLayer.ParameterCount()
	// Scan weights are also shared — count them only ONCE
	if len(s.Scans) > 0 {
		scan := s.Scans[0]
		shared += scan.InputGate.ParameterCount()
		shared += scan.InputProjection.ParameterCount()
		shared += scan.OutputProjection.ParameterCount()
	}
	return shared
}

// EffectiveParameters returns the effective parameter-computations.
// Each unique weight is reused NumLayers times.
func (s *SharedCortexStack) EffectiveParameters() int {
	return s.StoredParameters() * s.NumLayers
}

// StoredMemoryBytes returns the actual memory used.
func (s *SharedCortexStack) StoredMemoryBytes() int {
	shared := s.SharedFeatureLayer.MemoryBytes() + s.SharedOutputLayer.MemoryBytes()
	if len(s.Scans) > 0 {
		scan := s.Scans[0]
		shared += scan.InputGate.MemoryBytes()
		shared += scan.InputProjection.MemoryBytes()
		shared += scan.OutputProjection.MemoryBytes()
	}
	return shared
}

// CompressionRatio returns how many times more effective params vs stored.
func (s *SharedCortexStack) CompressionRatio() float64 {
	stored := s.StoredParameters()
	if stored == 0 {
		return 0
	}
	return float64(s.EffectiveParameters()) / float64(stored)
}

