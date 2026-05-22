package cortex

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// FractalCortex is a dynamically growing "Mixture of Cortexes" (MoC).
// It starts with one CortexBlock (a SharedCortexStack) and spawns new ones
// when it encounters high novelty/error, allowing it to scale towards 5T parameters
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

// SpawnNeurogenesis dynamically allocates a new physical ALBERT stack
// and optionally maps it to disk if Mmap is enabled.
func (fc *FractalCortex) SpawnNeurogenesis() {
	blockID := len(fc.Blocks)
	fmt.Printf("[Neurogenesis] Spawning Cortex Block #%d...\n", blockID)

	// Add a new shared stack
	// By default, 24 layers, SDRSize dim.
	newBlock := NewSharedCortexStack(24, fc.Config.SDRSize, 50, 3, 64, fc.Engine)

	fc.Blocks = append(fc.Blocks, newBlock)

	// Calculate scale
	totalParams := len(fc.Blocks) * 24 * (fc.Config.SDRSize * fc.Config.SDRSize) // Rough estimate
	fmt.Printf("[Neurogenesis] Block #%d Active. Total Effective Params: ~%d\n", blockID, totalParams)
}

// ProcessToken routes the input SDR to the top-K most active CortexBlocks.
// In biological terms, different brain regions specialize.
// For now, if we have few blocks, we route to all and average, or pick the best.
func (fc *FractalCortex) ProcessToken(input SDR) SDR {
	if len(fc.Blocks) == 0 {
		return NewSDR(fc.Config.SDRSize)
	}
	if len(fc.Blocks) == 1 {
		return fc.Blocks[0].ProcessToken(input)
	}

	// Dynamic MoC Routing:
	// Route to the most recently created block, which is focusing on the newest concepts.
	activeBlock := fc.Blocks[len(fc.Blocks)-1]
	return activeBlock.ProcessToken(input)
}

// CheckPredictionError monitors the discrepancy between the prediction and reality.
// If the error is consistently high, it triggers Neurogenesis to expand the physical parameter space.
func (fc *FractalCortex) CheckPredictionError(errorMagnitude float64) {
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
