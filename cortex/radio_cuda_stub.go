//go:build !windows
// +build !windows

package cortex

// RadioCUDA stub for non-Windows platforms.
// CUDA DLL loading is Windows-only (syscall.DLL).

type RadioCUDA struct{}

func NewRadioCUDA(numNeurons int) *RadioCUDA { return nil }
func (rc *RadioCUDA) IsAvailable() bool      { return false }
func (rc *RadioCUDA) Close()                 {}
func (rc *RadioCUDA) UploadNeurons(neurons []uint32)           {}
func (rc *RadioCUDA) DownloadNeurons(neurons []uint32)         {}
func (rc *RadioCUDA) UploadBus(bus *[256]int32)                {}
func (rc *RadioCUDA) DownloadBus(bus *[256]int32)              {}
func (rc *RadioCUDA) Step(fireThreshold int32, phaseWindow int32) {}
func (rc *RadioCUDA) StepN(nTicks int, fireThreshold int32, phaseWindow int32) {}
func (rc *RadioCUDA) Inject(freqs []uint8, amplitudes []int16) {}
func (rc *RadioCUDA) ClearBus()                                {}
func (rc *RadioCUDA) Stats() (alive, fired, avgAmp int32)      { return 0, 0, 0 }
func (rc *RadioCUDA) Hebbian(targetMask *[256]int32)           {}
func (rc *RadioCUDA) DecodeToken(vocabSpectrum []uint8, vocabSize, freqsPerToken int) int { return 0 }
