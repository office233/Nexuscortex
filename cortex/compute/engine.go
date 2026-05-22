package compute

// Engine defines the interface for hardware-accelerated computation
// of the Nexus Cortex neural primitives.
type Engine interface {
	// Init initializes the compute engine (e.g., allocates buffers, compiles shaders).
	Init() error

	// Close releases any hardware resources held by the engine.
	Close()

	// ForwardSparse computes the activation of a ternary layer given sparse inputs.
	// activeIndices and activeValues represent the non-zero inputs.
	// tiles is the array of TernaryTile (uint32).
	// bias is the array of bias per output neuron.
	// tilesPerRow is ceil(InputSize / 16).
	ForwardSparse(activeIndices []uint32, activeValues []int16, tiles []uint32, bias []int16, tilesPerRow int, outputSize int) []int16

	// BatchSDRSimilarity computes the intersection size between a query SDR
	// and a batch of memory SDRs in parallel.
	BatchSDRSimilarity(querySDR []uint32, memorySDRs [][]uint32) []uint8
}
