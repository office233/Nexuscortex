package cortex

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// MaxFractalBlocks limits the maximum number of cortex blocks to prevent unbounded growth.
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
}

func NewFractalCortex(cfg Config, engine interface{}) *FractalCortex {
	fc := &FractalCortex{
		Blocks:  make([]*SharedCortexStack, 0),
		Engine:  engine,
		Config:  cfg,
		DataDir: filepath.Join(cfg.DataDir, "fractal_cortex"),
	}

	_ = os.MkdirAll(fc.DataDir, 0755)

	// Start with exactly 1 block
	fc.SpawnNeurogenesis()

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
		newBlock := NewSharedCortexStack(24, fc.Config.SDRSize, 50, 3, 64, fc.Engine)
		fc.Blocks = append(fc.Blocks, newBlock)
	} else {
		// Clone the most recent block's TRAINED weights
		source := fc.Blocks[len(fc.Blocks)-1]
		newBlock := NewSharedCortexStack(source.NumLayers, source.Dim, 50, source.TopK, 64, fc.Engine)

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
		perturbTernaryLayer(newBlock.SharedFeatureLayer, 25) // 25/256 ≈ 10%
		perturbTernaryLayer(newBlock.SharedOutputLayer, 25)

		fc.Blocks = append(fc.Blocks, newBlock)
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

// ProcessToken routes the input SDR through ALL CortexBlocks and combines
// their outputs via bitwise majority voting. Each block acts as an
// independent expert with slightly different weights, and the consensus
// SDR contains bits that >50% of blocks agree on.
func (fc *FractalCortex) ProcessToken(input SDR) SDR {
	if len(fc.Blocks) == 0 {
		return NewSDR(fc.Config.SDRSize)
	}
	if len(fc.Blocks) == 1 {
		return fc.Blocks[0].ProcessToken(input)
	}

	// Collect outputs from ALL blocks
	dim := fc.Config.SDRSize
	voteCounts := make([]int, dim)
	for _, block := range fc.Blocks {
		output := block.ProcessToken(input)
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

	return result
}

// CheckPredictionError monitors the discrepancy between the prediction and reality.
// If the error is consistently high, it triggers Neurogenesis to expand the physical parameter space.
func (fc *FractalCortex) CheckPredictionError(errorMagnitude float64) {
	if len(fc.Blocks) >= MaxFractalBlocks {
		return
	}
	// Threshold for spawning a new cortex block
	if errorMagnitude > 0.8 && !fc.GrowthLock {
		fc.SpawnNeurogenesis()
		fc.GrowthLock = true // Reset lock after sleep consolidation
	}
}

// Save persists all blocks in the FractalCortex to the given directory.
func (fc *FractalCortex) Save(dataDir string) error {
	fcDir := filepath.Join(dataDir, "fractal_cortex")
	if err := os.MkdirAll(fcDir, 0755); err != nil {
		return fmt.Errorf("fractal_cortex save mkdir: %w", err)
	}

	// Save metadata JSON
	type fcMeta struct {
		BlocksCount int `json:"blocks_count"`
		NumLayers   int `json:"num_layers"`
		Dim         int `json:"dim"`
		TopK        int `json:"top_k"`
	}
	
	numLayers := 24
	dim := fc.Config.SDRSize
	topK := 3
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
	}

	metaData, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return fmt.Errorf("fractal_cortex marshal meta: %w", err)
	}

	if err := os.WriteFile(filepath.Join(fcDir, "metadata.json"), metaData, 0644); err != nil {
		return fmt.Errorf("fractal_cortex write meta: %w", err)
	}

	// Save each block
	for id, block := range fc.Blocks {
		blockDir := filepath.Join(fcDir, fmt.Sprintf("block_%d", id))
		if err := os.MkdirAll(blockDir, 0755); err != nil {
			return fmt.Errorf("fractal_cortex block mkdir: %w", err)
		}

		// Save 5 ternary layers
		if err := os.WriteFile(filepath.Join(blockDir, "feature.nxt1"), block.SharedFeatureLayer.MarshalRGBA32(), 0644); err != nil {
			return fmt.Errorf("save feature layer: %w", err)
		}
		if err := os.WriteFile(filepath.Join(blockDir, "output.nxt1"), block.SharedOutputLayer.MarshalRGBA32(), 0644); err != nil {
			return fmt.Errorf("save output layer: %w", err)
		}
		
		if len(block.Scans) > 0 {
			scan := block.Scans[0]
			if err := os.WriteFile(filepath.Join(blockDir, "gate.nxt1"), scan.InputGate.MarshalRGBA32(), 0644); err != nil {
				return fmt.Errorf("save gate layer: %w", err)
			}
			if err := os.WriteFile(filepath.Join(blockDir, "inproj.nxt1"), scan.InputProjection.MarshalRGBA32(), 0644); err != nil {
				return fmt.Errorf("save inproj layer: %w", err)
			}
			if err := os.WriteFile(filepath.Join(blockDir, "outproj.nxt1"), scan.OutputProjection.MarshalRGBA32(), 0644); err != nil {
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

	type fcMeta struct {
		BlocksCount int `json:"blocks_count"`
		NumLayers   int `json:"num_layers"`
		Dim         int `json:"dim"`
		TopK        int `json:"top_k"`
	}

	var meta fcMeta
	if err := json.Unmarshal(metaData, &meta); err != nil {
		return fmt.Errorf("fractal_cortex unmarshal meta: %w", err)
	}

	if meta.BlocksCount < 0 || meta.BlocksCount > 16 {
		return fmt.Errorf("invalid blocks_count: %d", meta.BlocksCount)
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
		
		// Load the 5 ternary layers
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

		// Reconstruct SharedCortexStack
		block := &SharedCortexStack{
			SharedFeatureLayer: featureLayer,
			SharedOutputLayer:  outputLayer,
			NumLayers:          meta.NumLayers,
			Dim:                meta.Dim,
			TopK:               meta.TopK,
			Attentions:         make([]*SDRAttentionHead, meta.NumLayers),
			Scans:              make([]*LinearScanLayer, meta.NumLayers),
		}

		// Fill in attention caches and temporal states
		contextLen := 50
		decayRate := uint8(64)
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
	return nil
}
