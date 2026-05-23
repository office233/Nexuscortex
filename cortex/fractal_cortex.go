package cortex

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// MaxFractalBlocks is the default maximum number of cortex blocks.
// Override via Config.FractalMaxBlocks.
const MaxFractalBlocks = 8

// FractalCortex is a dynamically growing "Mixture of Cortexes" (MoC).
// It starts with one CortexBlock (a SharedCortexStack) and spawns new ones
// when it encounters high novelty/error, allowing incremental parameter scaling
// incrementally while keeping latency extremely low (sparse routing).
type FractalCortex struct {
	Blocks     []*SharedCortexStack
	Engine     interface{}
	Config     Config
	DataDir    string
	GrowthLock bool // Prevents spawning too fast

	// Router enables top-k expert routing instead of all-block voting.
	// When nil, all blocks are evaluated (backward compatible).
	Router  *ExpertRouter

	// QRouter enables quantum-inspired phase+SDR routing.
	// When non-nil, used instead of Router for expert selection.
	QRouter *QuantumRouter

	// Journal accumulates weight changes for batch merge during Sleep().
	Journal *PlasticityJournal
}

func NewFractalCortex(cfg Config, engine interface{}) *FractalCortex {
	fc := &FractalCortex{
		Blocks:  make([]*SharedCortexStack, 0),
		Engine:  engine,
		Config:  cfg,
		DataDir: filepath.Join(cfg.DataDir, "fractal_cortex"),
	}

	_ = os.MkdirAll(fc.DataDir, 0700)

	// Start with exactly 1 block
	fc.SpawnNeurogenesis()

	// Conditionally wire quantum-inspired routing.
	// QRouter is sized to actual block count (not MaxFractalBlocks) to
	// avoid routing to non-existent experts.
	if cfg.EnableQuantumInspired && len(fc.Blocks) > 0 {
		fc.QRouter = NewQuantumRouter(len(fc.Blocks), cfg.SDRSize, 2)
		fc.Journal = NewPlasticityJournal()
	}

	return fc
}

// SpawnNeurogenesis dynamically allocates a new physical ALBERT stack.
// If existing blocks exist, the new block CLONES the best block's trained
// weights and applies a small perturbation for diversity. This ensures the
// new block inherits learned knowledge while exploring new representations.
func (fc *FractalCortex) SpawnNeurogenesis() {
	blockID := len(fc.Blocks)
	fmt.Printf("[Neurogenesis] Spawning Cortex Block #%d...\n", blockID)

	if len(fc.Blocks) == 0 {
		// First block: create with fresh random weights
		newBlock := NewSharedCortexStack(fc.Config.FractalNumLayers, fc.Config.SDRSize, fc.Config.FractalContextLen, fc.Config.FractalTopK, uint8(fc.Config.FractalDecayRate), fc.Engine)
		fc.Blocks = append(fc.Blocks, newBlock)
	} else {
		// Clone the most recent block's TRAINED weights
		source := fc.Blocks[len(fc.Blocks)-1]
		newBlock := NewSharedCortexStack(source.NumLayers, source.Dim, fc.Config.FractalContextLen, source.TopK, uint8(fc.Config.FractalDecayRate), fc.Engine)

		// Copy trained weights from source → new block
		copyTernaryWeights(source.SharedFeatureLayer, newBlock.SharedFeatureLayer)
		copyTernaryWeights(source.SharedOutputLayer, newBlock.SharedOutputLayer)
		if len(source.Scans) > 0 && len(newBlock.Scans) > 0 {
			copyTernaryWeights(source.Scans[0].InputGate, newBlock.Scans[0].InputGate)
			copyTernaryWeights(source.Scans[0].InputProjection, newBlock.Scans[0].InputProjection)
			copyTernaryWeights(source.Scans[0].OutputProjection, newBlock.Scans[0].OutputProjection)
			// Propagate shared weights to all scan layers
			for i := 1; i < len(newBlock.Scans); i++ {
				newBlock.Scans[i].InputGate = newBlock.Scans[0].InputGate
				newBlock.Scans[i].InputProjection = newBlock.Scans[0].InputProjection
				newBlock.Scans[i].OutputProjection = newBlock.Scans[0].OutputProjection
			}
		}

		// Apply perturbation: flip ~10% of ternary tiles for diversity
		perturbTernaryLayer(newBlock.SharedFeatureLayer, uint8(fc.Config.FractalPerturbRate))
		perturbTernaryLayer(newBlock.SharedOutputLayer, uint8(fc.Config.FractalPerturbRate))

		fc.Blocks = append(fc.Blocks, newBlock)
	}

	// Extend QRouter if it exists — add a new expert slot for the new block.
	if fc.QRouter != nil {
		fc.QRouter.ExpertPhases = append(fc.QRouter.ExpertPhases, 0)
		fc.QRouter.ExpertAmps = append(fc.QRouter.ExpertAmps, 128)
		fc.QRouter.ExpertEmbeddings = append(fc.QRouter.ExpertEmbeddings, NewSDR(fc.Config.SDRSize))
		fc.QRouter.UsageCounts = append(fc.QRouter.UsageCounts, 0)
	}

	totalParams := len(fc.Blocks) * fc.Blocks[0].StoredParameters()
	fmt.Printf("[Neurogenesis] Block #%d Active. Total Unique Params: ~%d\n", blockID, totalParams)
}

// copyTernaryWeights copies all tiles and biases from src to dst.
func copyTernaryWeights(src, dst *TernaryLayer) {
	if src == nil || dst == nil {
		return
	}
	n := len(src.Tiles)
	if len(dst.Tiles) < n {
		n = len(dst.Tiles)
	}
	copy(dst.Tiles[:n], src.Tiles[:n])

	bn := len(src.Bias)
	if len(dst.Bias) < bn {
		bn = len(dst.Bias)
	}
	copy(dst.Bias[:bn], src.Bias[:bn])
}

// perturbTernaryLayer randomly flips a fraction of ternary tiles.
// rate is 0-255 where 0=no change, 255=flip all.
func perturbTernaryLayer(layer *TernaryLayer, rate uint8) {
	if layer == nil || rate == 0 {
		return
	}
	// Simple XOR-based perturbation of the RGBA32 packed tiles
	state := uint64(0xDEADBEEF42)
	for i := range layer.Tiles {
		state ^= state << 13
		state ^= state >> 7
		state ^= state << 17
		if uint8(state) < rate {
			// Flip a random bit in this tile's mask (changes a weight's sign or activity)
			flipBit := uint32(1) << (uint(state>>8) % 32)
			layer.Tiles[i] = TernaryTile(uint32(layer.Tiles[i]) ^ flipBit)
		}
	}
}

// ProcessToken routes the input SDR through CortexBlocks.
// If a Router is set and there are enough blocks, uses top-k routing.
// Otherwise falls back to all-block voting for backward compatibility.
func (fc *FractalCortex) ProcessToken(input SDR) SDR {
	if len(fc.Blocks) == 0 {
		return NewSDR(fc.Config.SDRSize)
	}
	if len(fc.Blocks) == 1 {
		return fc.Blocks[0].ProcessToken(input)
	}

	// Use quantum-inspired routing if QRouter is available
	if fc.QRouter != nil && len(fc.Blocks) > fc.QRouter.TopK {
		return fc.ProcessTokenQuantum(input)
	}

	// Use top-k routing if Router is available
	if fc.Router != nil && len(fc.Blocks) > fc.Router.TopK {
		return fc.ProcessTokenTopK(input)
	}

	// Fallback: collect outputs from ALL blocks via weighted voting
	dim := fc.Config.SDRSize
	voteCounts := make([]int, dim)
	var bestBlock SDR
	var bestActive int
	for _, block := range fc.Blocks {
		output := block.ProcessToken(input)
		if output.ActiveCount > bestActive {
			bestActive = output.ActiveCount
			bestBlock = output
		}
		for _, idx := range output.ActiveIndices() {
			if idx < dim {
				voteCounts[idx]++
			}
		}
	}

	// Majority voting: a bit is active if >50% of blocks voted for it
	threshold := len(fc.Blocks) / 2
	result := NewSDR(dim)
	for idx, count := range voteCounts {
		if count > threshold {
			result.Set(idx)
		}
	}

	// Fallback: if voting produced too sparse output, use the best single block
	minActive := input.ActiveCount / 4
	if minActive < 2 {
		minActive = 2
	}
	if result.ActiveCount < minActive && bestActive > 0 {
		return bestBlock
	}

	return result
}

// ProcessTokenTopK routes input through only the top-k most relevant
// expert blocks instead of all blocks. This is the key to scaling:
// with 64 experts, running only top-2 does 32x less work.
func (fc *FractalCortex) ProcessTokenTopK(input SDR) SDR {
	if fc.Router == nil {
		return fc.ProcessToken(input)
	}

	selected := fc.Router.Route(input)
	if len(selected) == 0 {
		return NewSDR(fc.Config.SDRSize)
	}

	// Single expert: no voting needed
	if len(selected) == 1 {
		result := fc.Blocks[selected[0]].ProcessToken(input)
		fc.Router.UpdateEmbedding(selected[0], input)
		return result
	}

	// Multiple experts: vote among selected only
	dim := fc.Config.SDRSize
	voteCounts := make([]int, dim)
	var bestBlock SDR
	var bestActive int

	for _, idx := range selected {
		if idx >= len(fc.Blocks) {
			continue
		}
		output := fc.Blocks[idx].ProcessToken(input)
		if output.ActiveCount > bestActive {
			bestActive = output.ActiveCount
			bestBlock = output
		}
		for _, bitIdx := range output.ActiveIndices() {
			if bitIdx < dim {
				voteCounts[bitIdx]++
			}
		}
		fc.Router.UpdateEmbedding(idx, input)
	}

	// Majority among selected
	threshold := len(selected) / 2
	result := NewSDR(dim)
	for idx, count := range voteCounts {
		if count > threshold {
			result.Set(idx)
		}
	}

	minActive := input.ActiveCount / 4
	if minActive < 2 {
		minActive = 2
	}
	if result.ActiveCount < minActive && bestActive > 0 {
		return bestBlock
	}

	return result
}

// CheckPredictionError monitors the discrepancy between the prediction and reality.
// If the error is consistently high, it triggers Neurogenesis to expand the physical parameter space.
func (fc *FractalCortex) CheckPredictionError(errorMagnitude float64) {
	maxBlocks := fc.Config.FractalMaxBlocks
	if maxBlocks <= 0 {
		maxBlocks = MaxFractalBlocks
	}
	if len(fc.Blocks) >= maxBlocks {
		return
	}
	// Threshold for spawning a new cortex block
	if errorMagnitude > fc.Config.FractalNeurogenesisThreshold && !fc.GrowthLock {
		fc.SpawnNeurogenesis()
		fc.GrowthLock = true // Reset via ResetGrowthLock() during Sleep()
	}
}

// Save persists all blocks in the FractalCortex to the given directory.
func (fc *FractalCortex) Save(dataDir string) error {
	fcDir := filepath.Join(dataDir, "fractal_cortex")
	if err := os.MkdirAll(fcDir, 0700); err != nil {
		return fmt.Errorf("fractal_cortex save mkdir: %w", err)
	}

	// Save metadata JSON — includes quantum router state.
	type qRouterMeta struct {
		Phases []uint8 `json:"phases"`
		Amps   []uint8 `json:"amps"`
	}
	type fcMeta struct {
		BlocksCount int          `json:"blocks_count"`
		NumLayers   int          `json:"num_layers"`
		Dim         int          `json:"dim"`
		TopK        int          `json:"top_k"`
		ContextLen  int          `json:"context_len"`
		DecayRate   int          `json:"decay_rate"`
		QRouter     *qRouterMeta `json:"qrouter,omitempty"`
	}
	
	numLayers := fc.Config.FractalNumLayers
	dim := fc.Config.SDRSize
	topK := fc.Config.FractalTopK
	contextLen := fc.Config.FractalContextLen
	decayRate := fc.Config.FractalDecayRate
	if len(fc.Blocks) > 0 {
		numLayers = fc.Blocks[0].NumLayers
		dim = fc.Blocks[0].Dim
		topK = fc.Blocks[0].TopK
	}

	meta := fcMeta{
		BlocksCount: len(fc.Blocks),
		NumLayers:   numLayers,
		Dim:         dim,
		TopK:        topK,
		ContextLen:  contextLen,
		DecayRate:   decayRate,
	}

	// Persist QRouter state so learned phases/amps survive restart.
	if fc.QRouter != nil {
		meta.QRouter = &qRouterMeta{
			Phases: fc.QRouter.ExpertPhases,
			Amps:   fc.QRouter.ExpertAmps,
		}
	}

	metaData, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return fmt.Errorf("fractal_cortex marshal meta: %w", err)
	}

	if err := os.WriteFile(filepath.Join(fcDir, "metadata.json"), metaData, 0600); err != nil {
		return fmt.Errorf("fractal_cortex write meta: %w", err)
	}

	// Save each block
	for id, block := range fc.Blocks {
		blockDir := filepath.Join(fcDir, fmt.Sprintf("block_%d", id))
		if err := os.MkdirAll(blockDir, 0700); err != nil {
			return fmt.Errorf("fractal_cortex block mkdir: %w", err)
		}

		// Save 5 ternary layers
		if err := os.WriteFile(filepath.Join(blockDir, "feature.nxt1"), block.SharedFeatureLayer.MarshalRGBA32(), 0600); err != nil {
			return fmt.Errorf("save feature layer: %w", err)
		}
		if err := os.WriteFile(filepath.Join(blockDir, "output.nxt1"), block.SharedOutputLayer.MarshalRGBA32(), 0600); err != nil {
			return fmt.Errorf("save output layer: %w", err)
		}
		
		if len(block.Scans) > 0 {
			scan := block.Scans[0]
			if err := os.WriteFile(filepath.Join(blockDir, "gate.nxt1"), scan.InputGate.MarshalRGBA32(), 0600); err != nil {
				return fmt.Errorf("save gate layer: %w", err)
			}
			if err := os.WriteFile(filepath.Join(blockDir, "inproj.nxt1"), scan.InputProjection.MarshalRGBA32(), 0600); err != nil {
				return fmt.Errorf("save inproj layer: %w", err)
			}
			if err := os.WriteFile(filepath.Join(blockDir, "outproj.nxt1"), scan.OutputProjection.MarshalRGBA32(), 0600); err != nil {
				return fmt.Errorf("save outproj layer: %w", err)
			}
		}
	}
	return nil
}

// Load restores the FractalCortex state from the saved directory.
func (fc *FractalCortex) Load(dataDir string) error {
	fcDir := filepath.Join(dataDir, "fractal_cortex")
	metaPath := filepath.Join(fcDir, "metadata.json")
	metaData, err := os.ReadFile(metaPath)
	if err != nil {
		return fmt.Errorf("fractal_cortex load meta: %w", err)
	}

	type qRouterMeta struct {
		Phases []uint8 `json:"phases"`
		Amps   []uint8 `json:"amps"`
	}
	type fcMeta struct {
		BlocksCount int          `json:"blocks_count"`
		NumLayers   int          `json:"num_layers"`
		Dim         int          `json:"dim"`
		TopK        int          `json:"top_k"`
		ContextLen  int          `json:"context_len"`
		DecayRate   int          `json:"decay_rate"`
		QRouter     *qRouterMeta `json:"qrouter,omitempty"`
	}

	var meta fcMeta
	if err := json.Unmarshal(metaData, &meta); err != nil {
		return fmt.Errorf("fractal_cortex unmarshal meta: %w", err)
	}

	maxBlocks := fc.Config.FractalMaxBlocks
	if maxBlocks <= 0 {
		maxBlocks = MaxFractalBlocks
	}
	if meta.BlocksCount < 0 || meta.BlocksCount > maxBlocks {
		return fmt.Errorf("invalid blocks_count: %d (max %d)", meta.BlocksCount, maxBlocks)
	}
	if meta.NumLayers <= 0 || meta.NumLayers > 64 {
		return fmt.Errorf("invalid num_layers: %d", meta.NumLayers)
	}
	if meta.Dim <= 0 || meta.Dim > 100_000 {
		return fmt.Errorf("invalid dim: %d", meta.Dim)
	}
	if meta.TopK <= 0 || meta.TopK > 32 {
		return fmt.Errorf("invalid top_k: %d", meta.TopK)
	}

	// Reconstruct blocks
	loadedBlocks := make([]*SharedCortexStack, 0, meta.BlocksCount)
	for id := 0; id < meta.BlocksCount; id++ {
		blockDir := filepath.Join(fcDir, fmt.Sprintf("block_%d", id))
		
		featureData, err := os.ReadFile(filepath.Join(blockDir, "feature.nxt1"))
		if err != nil {
			return fmt.Errorf("load feature layer file: %w", err)
		}
		featureLayer, err := UnmarshalTernaryLayer(featureData)
		if err != nil {
			return fmt.Errorf("unmarshal feature layer: %w", err)
		}
		featureLayer.Engine = fc.Engine

		outputData, err := os.ReadFile(filepath.Join(blockDir, "output.nxt1"))
		if err != nil {
			return fmt.Errorf("load output layer file: %w", err)
		}
		outputLayer, err := UnmarshalTernaryLayer(outputData)
		if err != nil {
			return fmt.Errorf("unmarshal output layer: %w", err)
		}
		outputLayer.Engine = fc.Engine

		gateData, err := os.ReadFile(filepath.Join(blockDir, "gate.nxt1"))
		if err != nil {
			return fmt.Errorf("load gate layer file: %w", err)
		}
		gateLayer, err := UnmarshalTernaryLayer(gateData)
		if err != nil {
			return fmt.Errorf("unmarshal gate layer: %w", err)
		}
		gateLayer.Engine = fc.Engine

		inprojData, err := os.ReadFile(filepath.Join(blockDir, "inproj.nxt1"))
		if err != nil {
			return fmt.Errorf("load inproj layer file: %w", err)
		}
		inprojLayer, err := UnmarshalTernaryLayer(inprojData)
		if err != nil {
			return fmt.Errorf("unmarshal inproj layer: %w", err)
		}
		inprojLayer.Engine = fc.Engine

		outprojData, err := os.ReadFile(filepath.Join(blockDir, "outproj.nxt1"))
		if err != nil {
			return fmt.Errorf("load outproj layer file: %w", err)
		}
		outprojLayer, err := UnmarshalTernaryLayer(outprojData)
		if err != nil {
			return fmt.Errorf("unmarshal outproj layer: %w", err)
		}
		outprojLayer.Engine = fc.Engine

		block := &SharedCortexStack{
			SharedFeatureLayer: featureLayer,
			SharedOutputLayer:  outputLayer,
			NumLayers:          meta.NumLayers,
			Dim:                meta.Dim,
			TopK:               meta.TopK,
			Attentions:         make([]*SDRAttentionHead, meta.NumLayers),
			Scans:              make([]*LinearScanLayer, meta.NumLayers),
		}

		contextLen := meta.ContextLen
		if contextLen <= 0 {
			contextLen = fc.Config.FractalContextLen
		}
		decayRate := uint8(meta.DecayRate)
		if meta.DecayRate <= 0 {
			decayRate = uint8(fc.Config.FractalDecayRate)
		}
		for i := 0; i < meta.NumLayers; i++ {
			block.Attentions[i] = NewSDRAttentionHead(meta.Dim, meta.Dim, contextLen)
			block.Scans[i] = &LinearScanLayer{
				InputSize:        meta.Dim,
				StateSize:        meta.Dim,
				OutputSize:       meta.Dim,
				State:            NewSDR(meta.Dim),
				DecayRate:        decayRate,
				InputGate:        gateLayer,
				InputProjection:  inprojLayer,
				OutputProjection: outprojLayer,
			}
		}

		loadedBlocks = append(loadedBlocks, block)
	}

	fc.Blocks = loadedBlocks

	// Restore QRouter state from saved metadata.
	if meta.QRouter != nil && fc.Config.EnableQuantumInspired {
		fc.QRouter = NewQuantumRouter(len(fc.Blocks), fc.Config.SDRSize, 2)
		// Restore learned phases and amplitudes (capped to actual block count)
		for i := 0; i < len(fc.Blocks) && i < len(meta.QRouter.Phases); i++ {
			fc.QRouter.ExpertPhases[i] = meta.QRouter.Phases[i]
		}
		for i := 0; i < len(fc.Blocks) && i < len(meta.QRouter.Amps); i++ {
			fc.QRouter.ExpertAmps[i] = meta.QRouter.Amps[i]
		}
		fc.Journal = NewPlasticityJournal()
	}

	return nil
}

// ResetGrowthLock allows new block spawning after sleep consolidation.
// This must be called from Organism.Sleep() to re-enable neurogenesis.
func (fc *FractalCortex) ResetGrowthLock() {
	fc.GrowthLock = false
}

// ─────────────────────────────────────────────────────────────────────
// Quantum-Inspired Routing Integration
// ─────────────────────────────────────────────────────────────────────

// ProcessTokenQuantum routes input through the top-K experts selected by
// the QuantumRouter (phase interference + SDR similarity scoring).
func (fc *FractalCortex) ProcessTokenQuantum(input SDR) SDR {
	if fc.QRouter == nil {
		return fc.ProcessToken(input)
	}

	selected := fc.QRouter.RouteSDR(input)
	if len(selected) == 0 {
		return NewSDR(fc.Config.SDRSize)
	}

	// Single expert: no voting needed
	if len(selected) == 1 {
		idx := selected[0]
		if idx >= len(fc.Blocks) {
			return NewSDR(fc.Config.SDRSize)
		}
		result := fc.Blocks[idx].ProcessToken(input)
		// Update expert embedding for learning
		fc.QRouter.ExpertEmbeddings[idx] = fc.QRouter.ExpertEmbeddings[idx].Union(input)
		return result
	}

	// Multiple experts: vote among selected only
	dim := fc.Config.SDRSize
	voteCounts := make([]int, dim)
	var bestBlock SDR
	var bestActive int

	for _, idx := range selected {
		if idx >= len(fc.Blocks) {
			continue
		}
		output := fc.Blocks[idx].ProcessToken(input)
		if output.ActiveCount > bestActive {
			bestActive = output.ActiveCount
			bestBlock = output
		}
		for _, bitIdx := range output.ActiveIndices() {
			if bitIdx < dim {
				voteCounts[bitIdx]++
			}
		}
		// Update expert embedding for learning
		fc.QRouter.ExpertEmbeddings[idx] = fc.QRouter.ExpertEmbeddings[idx].Union(input)
	}

	// Majority among selected
	threshold := len(selected) / 2
	result := NewSDR(dim)
	for idx, count := range voteCounts {
		if count > threshold {
			result.Set(idx)
		}
	}

	minActive := input.ActiveCount / 4
	if minActive < 2 {
		minActive = 2
	}
	if result.ActiveCount < minActive && bestActive > 0 {
		return bestBlock
	}

	return result
}

// MergeJournal applies all pending PlasticityJournal entries into the
// ternary weights of the cortex blocks. Called during Sleep().
// Returns the total number of weight changes applied.
// Only clears entries that were successfully applied. Unapplied entries
// (for non-existent blocks/layers) are preserved for the next cycle.
func (fc *FractalCortex) MergeJournal() int {
	if fc.Journal == nil || fc.Journal.Size() == 0 {
		return 0
	}

	totalApplied := 0
	for blockID, block := range fc.Blocks {
		totalApplied += fc.Journal.Merge(block.SharedFeatureLayer, blockID, "feature")
		totalApplied += fc.Journal.Merge(block.SharedOutputLayer, blockID, "output")
	}

	// Only clear if all entries were applied. If some remain unmatched,
	// keep them for the next Sleep() cycle when more blocks may exist.
	if totalApplied > 0 {
		fc.Journal.ClearApplied()
	}

	return totalApplied
}
