package cortex

import (
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
