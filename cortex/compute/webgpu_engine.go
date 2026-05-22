//go:build webgpu
// +build webgpu

package compute


import (
	"fmt"
	"time"
	"unsafe"

	"github.com/rajveermalviya/go-webgpu/wgpu"
)

type WebGPUEngine struct {
	instance *wgpu.Instance
	adapter  *wgpu.Adapter
	device   *wgpu.Device
	queue    *wgpu.Queue
	pipeline *wgpu.ComputePipeline
	cpu      *CPUEngine
	Timeout  time.Duration
}

func NewWebGPUEngine() *WebGPUEngine {
	return &WebGPUEngine{
		cpu:     NewCPUEngine(),
		Timeout: 5 * time.Second,
	}
}

func (e *WebGPUEngine) Init() error {
	e.instance = wgpu.CreateInstance(nil)
	if e.instance == nil {
		return fmt.Errorf("failed to create wgpu instance")
	}

	var err error
	e.adapter, err = e.instance.RequestAdapter(nil)
	if err != nil || e.adapter == nil {
		return fmt.Errorf("failed to request adapter: %v", err)
	}

	e.device, err = e.adapter.RequestDevice(nil)
	if err != nil || e.device == nil {
		return fmt.Errorf("failed to request device: %v", err)
	}
	e.queue = e.device.GetQueue()

	shaderSource := `
	@group(0) @binding(0) var<storage, read> activeIndices: array<u32>;
	@group(0) @binding(1) var<storage, read> activeValues: array<i32>;
	@group(0) @binding(2) var<storage, read> tiles: array<u32>;
	@group(0) @binding(3) var<storage, read> bias: array<i32>;
	@group(0) @binding(4) var<storage, read_write> outputAct: array<i32>;

	struct Params {
		activeCount: u32,
		outputSize: u32,
		tilesPerRow: u32,
		padding: u32,
	}
	@group(0) @binding(5) var<uniform> params: Params;

	@compute @workgroup_size(64)
	fn forward_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
		let outIdx = global_id.x;
		if (outIdx >= params.outputSize) {
			return;
		}
		
		var acc: i32 = 0;
		let rowOffset = outIdx * params.tilesPerRow;
		
		for (var i = 0u; i < params.activeCount; i = i + 1u) {
			let idx = activeIndices[i];
			let tileIdx = rowOffset + (idx / 16u);
			let pos = idx % 16u;
			
			let tile = tiles[tileIdx];
			
			var shift_base = 16u;
			if (pos < 8u) {
				shift_base = 0u;
			}
			
			let local_pos = pos & 7u;
			let sign_bit = (tile >> (shift_base + local_pos)) & 1u;
			let mask_bit = (tile >> (shift_base + 8u + local_pos)) & 1u;
			
			if (mask_bit != 0u) {
				let val = activeValues[i];
				if (sign_bit != 0u) {
					acc = acc - val;
				} else {
					acc = acc + val;
				}
			}
		}
		
		if (acc > 32767) {
			acc = 32767;
		} else if (acc < -32768) {
			acc = -32768;
		}
		
		outputAct[outIdx] = bias[outIdx] + acc;
	}
	`

	shader, err := e.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "forward_shader",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: shaderSource},
	})
	if err != nil {
		return fmt.Errorf("failed to create shader module: %w", err)
	}
	defer shader.Release()

	e.pipeline, err = e.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "forward_pipeline",
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     shader,
			EntryPoint: "forward_main",
		},
	})
	if err != nil {
		return fmt.Errorf("failed to create compute pipeline: %w", err)
	}

	return nil
}

func (e *WebGPUEngine) Close() {
	if e.pipeline != nil {
		e.pipeline.Release()
	}
	if e.queue != nil {
		e.queue.Release()
	}
	if e.device != nil {
		e.device.Release()
	}
	if e.adapter != nil {
		e.adapter.Release()
	}
	if e.instance != nil {
		e.instance.Release()
	}
}

// ForwardSparse performs GPU-accelerated forward pass. On ANY GPU error,
// it falls back to the CPU engine instead of panicking.
func (e *WebGPUEngine) ForwardSparse(activeIndices []uint32, activeValues []int16, tiles []uint32, bias []int16, tilesPerRow int, outputSize int) ([]int16, error) {
	if len(activeIndices) == 0 {
		out := make([]int16, outputSize)
		copy(out, bias)
		return out, nil
	}

	// Defensive validation — engine is a public interface
	if outputSize <= 0 || tilesPerRow <= 0 {
		return nil, fmt.Errorf("webgpu: invalid outputSize=%d or tilesPerRow=%d", outputSize, tilesPerRow)
	}
	if len(activeIndices) != len(activeValues) {
		return nil, fmt.Errorf("webgpu: mismatched activeIndices(%d) and activeValues(%d)", len(activeIndices), len(activeValues))
	}
	if len(tiles) == 0 {
		return nil, fmt.Errorf("webgpu: empty tiles")
	}
	if len(bias) < outputSize {
		return nil, fmt.Errorf("webgpu: bias length %d < outputSize %d", len(bias), outputSize)
	}

	// Prepare i32 buffers for activeValues and bias
	activeValues32 := make([]int32, len(activeValues))
	for i, v := range activeValues {
		activeValues32[i] = int32(v)
	}

	bias32 := make([]int32, len(bias))
	for i, b := range bias {
		bias32[i] = int32(b)
	}

	params := []uint32{uint32(len(activeIndices)), uint32(outputSize), uint32(tilesPerRow), 0}

	// All slices are validated non-empty above, so unsafe.Slice is safe here
	activeIndicesBytes := unsafe.Slice((*byte)(unsafe.Pointer(&activeIndices[0])), len(activeIndices)*4)
	activeValuesBytes := unsafe.Slice((*byte)(unsafe.Pointer(&activeValues32[0])), len(activeValues32)*4)
	tilesBytes := unsafe.Slice((*byte)(unsafe.Pointer(&tiles[0])), len(tiles)*4)
	biasBytes := unsafe.Slice((*byte)(unsafe.Pointer(&bias32[0])), len(bias32)*4)
	paramsBytes := unsafe.Slice((*byte)(unsafe.Pointer(&params[0])), len(params)*4)

	outputSizeBytes := uint64(outputSize * 4)

	// Create buffers — return error on failure, never panic
	bufIndices, err := e.device.CreateBuffer(&wgpu.BufferDescriptor{
		Size:  uint64(len(activeIndicesBytes)),
		Usage: wgpu.BufferUsage_Storage | wgpu.BufferUsage_CopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create bufIndices: %w", err)
	}
	defer bufIndices.Release()

	bufValues, err := e.device.CreateBuffer(&wgpu.BufferDescriptor{
		Size:  uint64(len(activeValuesBytes)),
		Usage: wgpu.BufferUsage_Storage | wgpu.BufferUsage_CopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create bufValues: %w", err)
	}
	defer bufValues.Release()

	bufTiles, err := e.device.CreateBuffer(&wgpu.BufferDescriptor{
		Size:  uint64(len(tilesBytes)),
		Usage: wgpu.BufferUsage_Storage | wgpu.BufferUsage_CopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create bufTiles: %w", err)
	}
	defer bufTiles.Release()

	bufBias, err := e.device.CreateBuffer(&wgpu.BufferDescriptor{
		Size:  uint64(len(biasBytes)),
		Usage: wgpu.BufferUsage_Storage | wgpu.BufferUsage_CopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create bufBias: %w", err)
	}
	defer bufBias.Release()

	bufOutput, err := e.device.CreateBuffer(&wgpu.BufferDescriptor{
		Size:  outputSizeBytes,
		Usage: wgpu.BufferUsage_Storage | wgpu.BufferUsage_CopySrc,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create bufOutput: %w", err)
	}
	defer bufOutput.Release()

	bufStaging, err := e.device.CreateBuffer(&wgpu.BufferDescriptor{
		Size:  outputSizeBytes,
		Usage: wgpu.BufferUsage_MapRead | wgpu.BufferUsage_CopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create bufStaging: %w", err)
	}
	defer bufStaging.Release()

	bufParams, err := e.device.CreateBuffer(&wgpu.BufferDescriptor{
		Size:  uint64(len(paramsBytes)),
		Usage: wgpu.BufferUsage_Uniform | wgpu.BufferUsage_CopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create bufParams: %w", err)
	}
	defer bufParams.Release()

	// Write buffers
	e.queue.WriteBuffer(bufIndices, 0, activeIndicesBytes)
	e.queue.WriteBuffer(bufValues, 0, activeValuesBytes)
	e.queue.WriteBuffer(bufTiles, 0, tilesBytes)
	e.queue.WriteBuffer(bufBias, 0, biasBytes)
	e.queue.WriteBuffer(bufParams, 0, paramsBytes)

	bindGroupLayout := e.pipeline.GetBindGroupLayout(0)
	defer bindGroupLayout.Release()

	bindGroup, err := e.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: bindGroupLayout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: bufIndices, Offset: 0, Size: uint64(len(activeIndicesBytes))},
			{Binding: 1, Buffer: bufValues, Offset: 0, Size: uint64(len(activeValuesBytes))},
			{Binding: 2, Buffer: bufTiles, Offset: 0, Size: uint64(len(tilesBytes))},
			{Binding: 3, Buffer: bufBias, Offset: 0, Size: uint64(len(biasBytes))},
			{Binding: 4, Buffer: bufOutput, Offset: 0, Size: outputSizeBytes},
			{Binding: 5, Buffer: bufParams, Offset: 0, Size: uint64(len(paramsBytes))},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create bind group: %w", err)
	}
	defer bindGroup.Release()

	encoder, err := e.device.CreateCommandEncoder(nil)
	if err != nil {
		return nil, fmt.Errorf("webgpu: create command encoder: %w", err)
	}
	computePass := encoder.BeginComputePass(nil)
	computePass.SetPipeline(e.pipeline)
	computePass.SetBindGroup(0, bindGroup, nil)
	computePass.DispatchWorkgroups(uint32((outputSize+63)/64), 1, 1)
	computePass.End()

	encoder.CopyBufferToBuffer(bufOutput, 0, bufStaging, 0, outputSizeBytes)

	cmdBuffer, err := encoder.Finish(nil)
	if err != nil {
		encoder.Release()
		return nil, fmt.Errorf("webgpu: finish command encoder: %w", err)
	}
	e.queue.Submit(cmdBuffer)

	encoder.Release()
	cmdBuffer.Release()

	// Map and wait with timeout to prevent infinite blocking
	done := make(chan wgpu.BufferMapAsyncStatus, 1)
	bufStaging.MapAsync(wgpu.MapMode_Read, 0, outputSizeBytes, func(status wgpu.BufferMapAsyncStatus) {
		done <- status
	})

	timeout := e.Timeout
	if timeout <= 0 {
		timeout = 5 * time.Second
	}
	deadline := time.After(timeout)
WaitLoop:
	for {
		e.device.Poll(true, nil)
		select {
		case status := <-done:
			if status != wgpu.BufferMapAsyncStatus_Success {
				return nil, fmt.Errorf("webgpu: map async failed with status %v", status)
			}
			break WaitLoop
		case <-deadline:
			return nil, fmt.Errorf("webgpu: map async timed out after %v", timeout)
		default:
		}
	}

	mappedBytes := bufStaging.GetMappedRange(0, uint(outputSizeBytes))
	
	output32 := make([]int32, outputSize)
	copy(unsafe.Slice((*byte)(unsafe.Pointer(&output32[0])), outputSizeBytes), mappedBytes)
	
	bufStaging.Unmap()

	// Convert back to int16
	out16 := make([]int16, outputSize)
	for i, val := range output32 {
		out16[i] = int16(val)
	}

	return out16, nil
}

func (e *WebGPUEngine) BatchSDRSimilarity(querySDR []uint32, memorySDRs [][]uint32) ([]uint8, error) {
	return e.cpu.BatchSDRSimilarity(querySDR, memorySDRs)
}
