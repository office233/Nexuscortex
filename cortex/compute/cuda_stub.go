//go:build !cuda
// +build !cuda

package compute

import (
	"fmt"
)

// CUDAEngine is a fallback stub when compiled without CUDA support.
// It redirects all execution paths to the robust CPUEngine.
type CUDAEngine struct {
	cpu *CPUEngine
}

// NewCUDAEngine constructs a CPU-backed CUDAEngine stub.
func NewCUDAEngine() *CUDAEngine {
	return &CUDAEngine{
		cpu: NewCPUEngine(),
	}
}

// Init returns a non-nil error indicating CUDA is disabled.
func (e *CUDAEngine) Init() error {
	return fmt.Errorf("CUDA support is disabled in this build (compile with -tags cuda)")
}

// Close is a no-op fallback.
func (e *CUDAEngine) Close() {}

// ForwardSparse delegates to the CPUEngine.
func (e *CUDAEngine) ForwardSparse(activeIndices []uint32, activeValues []int16, tiles []uint32, bias []int16, tilesPerRow int, outputSize int) ([]int16, error) {
	return e.cpu.ForwardSparse(activeIndices, activeValues, tiles, bias, tilesPerRow, outputSize)
}

// BatchSDRSimilarity delegates to the CPUEngine.
func (e *CUDAEngine) BatchSDRSimilarity(querySDR []uint32, memorySDRs [][]uint32) ([]uint8, error) {
	return e.cpu.BatchSDRSimilarity(querySDR, memorySDRs)
}
