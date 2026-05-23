package cortex

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"syscall"
	"unsafe"
)

// ═══════════════════════════════════════════════════════════════════
// RadioCUDA — GPU-accelerated RadioCortex via CUDA DLL
// ═══════════════════════════════════════════════════════════════════
//
// Loads radio_cuda.dll at runtime and provides Go wrappers for all
// CUDA kernel functions. Falls back gracefully to CPU if DLL not found.

// RadioCUDA wraps the CUDA DLL for GPU-accelerated radio processing.
type RadioCUDA struct {
	dll         *syscall.DLL
	numNeurons  int
	initialized bool

	// DLL function handles
	fnInit            *syscall.Proc
	fnCleanup         *syscall.Proc
	fnUploadNeurons   *syscall.Proc
	fnDownloadNeurons *syscall.Proc
	fnUploadBus       *syscall.Proc
	fnDownloadBus     *syscall.Proc
	fnStep            *syscall.Proc
	fnStepN           *syscall.Proc
	fnInject          *syscall.Proc
	fnClearBus        *syscall.Proc
	fnStats           *syscall.Proc
	fnHebbian         *syscall.Proc
	fnDecodeToken     *syscall.Proc
	fnDecodeTopK      *syscall.Proc
}

// NewRadioCUDA loads the CUDA DLL and initializes GPU memory.
// Returns nil if CUDA is not available (CPU fallback).
func NewRadioCUDA(numNeurons int) *RadioCUDA {
	if runtime.GOOS != "windows" {
		fmt.Println("[RadioCUDA] Not on Windows, using CPU fallback.")
		return nil
	}

	// Search for DLL in multiple locations
	searchPaths := []string{
		"radio_cuda.dll",
		"cuda/radio_cuda.dll",
		filepath.Join("cuda", "radio_cuda.dll"),
	}

	// Also search next to the executable
	if exe, err := os.Executable(); err == nil {
		searchPaths = append(searchPaths, filepath.Join(filepath.Dir(exe), "radio_cuda.dll"))
	}

	var dll *syscall.DLL
	var loadErr error
	for _, p := range searchPaths {
		dll, loadErr = syscall.LoadDLL(p)
		if loadErr == nil {
			break
		}
	}
	if dll == nil {
		fmt.Printf("[RadioCUDA] DLL not found (tried %d paths), using CPU fallback.\n", len(searchPaths))
		return nil
	}

	rc := &RadioCUDA{
		dll:        dll,
		numNeurons: numNeurons,
	}

	// Load all function pointers
	var err error
	rc.fnInit, err = dll.FindProc("radio_cuda_init")
	if err != nil {
		fmt.Printf("[RadioCUDA] Missing function radio_cuda_init: %v\n", err)
		dll.Release()
		return nil
	}
	rc.fnCleanup, _ = dll.FindProc("radio_cuda_cleanup")
	rc.fnUploadNeurons, _ = dll.FindProc("radio_cuda_upload_neurons")
	rc.fnDownloadNeurons, _ = dll.FindProc("radio_cuda_download_neurons")
	rc.fnUploadBus, _ = dll.FindProc("radio_cuda_upload_bus")
	rc.fnDownloadBus, _ = dll.FindProc("radio_cuda_download_bus")
	rc.fnStep, _ = dll.FindProc("radio_cuda_step")
	rc.fnStepN, _ = dll.FindProc("radio_cuda_step_n")
	rc.fnInject, _ = dll.FindProc("radio_cuda_inject")
	rc.fnClearBus, _ = dll.FindProc("radio_cuda_clear_bus")
	rc.fnStats, _ = dll.FindProc("radio_cuda_stats")
	rc.fnHebbian, _ = dll.FindProc("radio_cuda_hebbian")
	rc.fnDecodeToken, _ = dll.FindProc("radio_cuda_decode_token")
	rc.fnDecodeTopK, _ = dll.FindProc("radio_cuda_decode_top_k")

	// Initialize GPU memory
	ret, _, _ := rc.fnInit.Call(uintptr(numNeurons))
	if int32(ret) != 0 {
		fmt.Println("[RadioCUDA] GPU init failed, using CPU fallback.")
		dll.Release()
		return nil
	}

	rc.initialized = true
	fmt.Printf("[RadioCUDA] ✅ GPU initialized: %d neurons on CUDA\n", numNeurons)
	return rc
}

// Close releases GPU memory and unloads the DLL.
func (rc *RadioCUDA) Close() {
	if rc == nil || !rc.initialized {
		return
	}
	if rc.fnCleanup != nil {
		rc.fnCleanup.Call()
	}
	rc.dll.Release()
	rc.initialized = false
}

// UploadNeurons copies neuron data from CPU to GPU.
func (rc *RadioCUDA) UploadNeurons(neurons []uint32) {
	if !rc.initialized || len(neurons) == 0 {
		return
	}
	rc.fnUploadNeurons.Call(
		uintptr(unsafe.Pointer(&neurons[0])),
		uintptr(len(neurons)),
	)
}

// DownloadNeurons copies neuron data from GPU to CPU.
func (rc *RadioCUDA) DownloadNeurons(neurons []uint32) {
	if !rc.initialized || len(neurons) == 0 {
		return
	}
	rc.fnDownloadNeurons.Call(
		uintptr(unsafe.Pointer(&neurons[0])),
		uintptr(len(neurons)),
	)
}

// UploadBus copies bus state from CPU to GPU.
func (rc *RadioCUDA) UploadBus(bus *[256]int32) {
	if !rc.initialized {
		return
	}
	rc.fnUploadBus.Call(uintptr(unsafe.Pointer(&bus[0])))
}

// DownloadBus copies bus state from GPU to CPU.
func (rc *RadioCUDA) DownloadBus(bus *[256]int32) {
	if !rc.initialized {
		return
	}
	rc.fnDownloadBus.Call(uintptr(unsafe.Pointer(&bus[0])))
}

// Step runs one complete tick on GPU.
func (rc *RadioCUDA) Step(fireThreshold int32, phaseWindow int32) {
	if !rc.initialized {
		return
	}
	rc.fnStep.Call(uintptr(fireThreshold), uintptr(phaseWindow))
}

// StepN runs N ticks entirely on GPU without CPU round-trips.
// This is the key performance win — 20 ticks in one GPU call.
func (rc *RadioCUDA) StepN(nTicks int, fireThreshold int32, phaseWindow int32) {
	if !rc.initialized {
		return
	}
	rc.fnStepN.Call(
		uintptr(nTicks),
		uintptr(fireThreshold),
		uintptr(phaseWindow),
	)
}

// Inject adds signals to the bus for specific frequencies.
func (rc *RadioCUDA) Inject(freqs []uint8, amplitudes []int16) {
	if !rc.initialized || len(freqs) == 0 {
		return
	}
	count := len(freqs)
	if count > len(amplitudes) {
		count = len(amplitudes)
	}
	rc.fnInject.Call(
		uintptr(unsafe.Pointer(&freqs[0])),
		uintptr(unsafe.Pointer(&amplitudes[0])),
		uintptr(count),
	)
}

// ClearBus resets all bus channels to zero.
func (rc *RadioCUDA) ClearBus() {
	if !rc.initialized {
		return
	}
	rc.fnClearBus.Call()
}

// Stats returns [alive_count, fired_count, avg_amplitude].
func (rc *RadioCUDA) Stats() (alive, fired, avgAmp int32) {
	if !rc.initialized {
		return 0, 0, 0
	}
	var out [3]int32
	rc.fnStats.Call(uintptr(unsafe.Pointer(&out[0])))
	return out[0], out[1], out[2]
}

// Hebbian runs confirm/contradict learning on GPU.
// targetMask[freq] > 0 means neurons emitting on that freq get confirmed.
func (rc *RadioCUDA) Hebbian(targetMask *[256]int32) {
	if !rc.initialized {
		return
	}
	rc.fnHebbian.Call(uintptr(unsafe.Pointer(&targetMask[0])))
}

// DecodeToken finds the token with highest bus energy match.
func (rc *RadioCUDA) DecodeToken(vocabSpectrum []uint8, vocabSize, freqsPerToken int) int {
	if !rc.initialized || len(vocabSpectrum) == 0 {
		return 0
	}
	ret, _, _ := rc.fnDecodeToken.Call(
		uintptr(unsafe.Pointer(&vocabSpectrum[0])),
		uintptr(vocabSize),
		uintptr(freqsPerToken),
	)
	return int(ret)
}

// IsAvailable returns true if CUDA is initialized and ready.
func (rc *RadioCUDA) IsAvailable() bool {
	return rc != nil && rc.initialized
}
