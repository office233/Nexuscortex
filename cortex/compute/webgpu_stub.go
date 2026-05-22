//go:build !webgpu
// +build !webgpu

package compute

import "fmt"

// WebGPUEngine is a fallback stub when compiled without WebGPU support.
// It redirects all execution paths to the robust CPUEngine.
type WebGPUEngine struct {
	cpu *CPUEngine
}

// NewWebGPUEngine constructs a CPU-backed WebGPUEngine stub.
func NewWebGPUEngine() *WebGPUEngine {
	return &WebGPUEngine{
		cpu: NewCPUEngine(),
	}
}

// Init returns a non-nil error indicating WebGPU is disabled.
func (e *WebGPUEngine) Init() error {
	return fmt.Errorf("webgpu support is disabled in this build configuration")
}

// Close is a no-op fallback.
func (e *WebGPUEngine) Close() {}

// ForwardSparse delegates sparse neural layer computations directly to the CPUEngine.
func (e *WebGPUEngine) ForwardSparse(activeIndices []uint32, activeValues []int16, tiles []uint32, bias []int16, tilesPerRow int, outputSize int) []int16 {
	return e.cpu.ForwardSparse(activeIndices, activeValues, tiles, bias, tilesPerRow, outputSize)
}

// BatchSDRSimilarity delegates SDR overlap comparisons to the CPUEngine.
func (e *WebGPUEngine) BatchSDRSimilarity(querySDR []uint32, memorySDRs [][]uint32) []uint8 {
	return e.cpu.BatchSDRSimilarity(querySDR, memorySDRs)
}
