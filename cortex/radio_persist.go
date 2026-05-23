package cortex

import (
	"encoding/binary"
	"fmt"
	"math/rand"
	"os"
)

// Binary format for RadioCortex persistence:
//
//   Offset  Size   Field
//   ──────  ─────  ─────────────────
//   0       4      Magic bytes "NXRC"
//   4       1      Version (1)
//   5       4      Size (uint32, neuron count)
//   9       8      TickCount (uint64)
//   17      4      InputStart (uint32)
//   21      4      InputEnd (uint32)
//   25      4      OutputStart (uint32)
//   29      4      OutputEnd (uint32)
//   33      4      FireThreshold (int32)
//   37      1      PhaseWindow (uint8)
//   38      4      VocabSize (uint32, for SignalCodec)
//   42      N×4    Neurons (each RadioNeuron is uint32)

var radioMagic = [4]byte{'N', 'X', 'R', 'C'}

const radioVersion uint8 = 1

// radioHeaderSize is the fixed header size before the neuron array.
const radioHeaderSize = 4 + 1 + 4 + 8 + 4 + 4 + 4 + 4 + 4 + 1 + 4 // = 42

// SaveRadioCortex writes all neurons + metadata to a binary file.
func SaveRadioCortex(rc *RadioCortex, path string) error {
	totalSize := radioHeaderSize + rc.Size*4
	buf := make([]byte, totalSize)

	off := 0

	// Magic.
	copy(buf[off:], radioMagic[:])
	off += 4

	// Version.
	buf[off] = radioVersion
	off++

	// Size.
	binary.LittleEndian.PutUint32(buf[off:], uint32(rc.Size))
	off += 4

	// TickCount.
	binary.LittleEndian.PutUint64(buf[off:], rc.TickCount)
	off += 8

	// Region indices.
	binary.LittleEndian.PutUint32(buf[off:], uint32(rc.InputStart))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:], uint32(rc.InputEnd))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:], uint32(rc.OutputStart))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:], uint32(rc.OutputEnd))
	off += 4

	// Config.
	binary.LittleEndian.PutUint32(buf[off:], uint32(rc.FireThreshold))
	off += 4
	buf[off] = rc.PhaseWindow
	off++

	// VocabSize placeholder (caller fills via SaveRadioCortexWithCodec,
	// or 0 if no codec).
	binary.LittleEndian.PutUint32(buf[off:], 0)
	off += 4

	// Neurons.
	for i := 0; i < rc.Size; i++ {
		binary.LittleEndian.PutUint32(buf[off:], uint32(rc.Neurons[i]))
		off += 4
	}

	return os.WriteFile(path, buf, 0600)
}

// SaveRadioCortexWithCodec writes all neurons + metadata + SignalCodec
// vocab size to a binary file. The vocab size is stored so the codec
// can be reconstructed on load.
func SaveRadioCortexWithCodec(rc *RadioCortex, codec *SignalCodec, path string) error {
	totalSize := radioHeaderSize + rc.Size*4
	buf := make([]byte, totalSize)

	off := 0

	// Magic.
	copy(buf[off:], radioMagic[:])
	off += 4

	// Version.
	buf[off] = radioVersion
	off++

	// Size.
	binary.LittleEndian.PutUint32(buf[off:], uint32(rc.Size))
	off += 4

	// TickCount.
	binary.LittleEndian.PutUint64(buf[off:], rc.TickCount)
	off += 8

	// Region indices.
	binary.LittleEndian.PutUint32(buf[off:], uint32(rc.InputStart))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:], uint32(rc.InputEnd))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:], uint32(rc.OutputStart))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:], uint32(rc.OutputEnd))
	off += 4

	// Config.
	binary.LittleEndian.PutUint32(buf[off:], uint32(rc.FireThreshold))
	off += 4
	buf[off] = rc.PhaseWindow
	off++

	// VocabSize from codec.
	vocabSize := uint32(0)
	if codec != nil {
		vocabSize = uint32(codec.vocabSize)
	}
	binary.LittleEndian.PutUint32(buf[off:], vocabSize)
	off += 4

	// Neurons.
	for i := 0; i < rc.Size; i++ {
		binary.LittleEndian.PutUint32(buf[off:], uint32(rc.Neurons[i]))
		off += 4
	}

	return os.WriteFile(path, buf, 0600)
}

// LoadRadioCortex reads neurons from a binary file and returns a
// reconstructed RadioCortex. The Fired and PrevFired slices are
// rebuilt at the correct size. A new RNG is seeded from the tick count.
//
// If the file contained a non-zero vocabSize, a new SignalCodec is
// also returned; otherwise codec is nil.
func LoadRadioCortex(path string) (*RadioCortex, *SignalCodec, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, fmt.Errorf("read radio cortex file: %w", err)
	}

	if len(data) < radioHeaderSize {
		return nil, nil, fmt.Errorf("radio cortex file too short: %d bytes", len(data))
	}

	off := 0

	// Magic.
	if data[0] != radioMagic[0] || data[1] != radioMagic[1] ||
		data[2] != radioMagic[2] || data[3] != radioMagic[3] {
		return nil, nil, fmt.Errorf("radio cortex bad magic: %q", data[:4])
	}
	off += 4

	// Version.
	version := data[off]
	if version != radioVersion {
		return nil, nil, fmt.Errorf("radio cortex unsupported version: %d", version)
	}
	off++

	// Size.
	size := int(binary.LittleEndian.Uint32(data[off:]))
	off += 4

	// Validate total file size.
	expectedSize := radioHeaderSize + size*4
	if len(data) < expectedSize {
		return nil, nil, fmt.Errorf("radio cortex file truncated: want %d bytes, got %d", expectedSize, len(data))
	}

	// TickCount.
	tickCount := binary.LittleEndian.Uint64(data[off:])
	off += 8

	// Region indices.
	inputStart := int(binary.LittleEndian.Uint32(data[off:]))
	off += 4
	inputEnd := int(binary.LittleEndian.Uint32(data[off:]))
	off += 4
	outputStart := int(binary.LittleEndian.Uint32(data[off:]))
	off += 4
	outputEnd := int(binary.LittleEndian.Uint32(data[off:]))
	off += 4

	// Config.
	fireThreshold := int32(binary.LittleEndian.Uint32(data[off:]))
	off += 4
	phaseWindow := data[off]
	off++

	// VocabSize.
	vocabSize := int(binary.LittleEndian.Uint32(data[off:]))
	off += 4

	// Neurons.
	neurons := make([]RadioNeuron, size)
	for i := 0; i < size; i++ {
		neurons[i] = RadioNeuron(binary.LittleEndian.Uint32(data[off:]))
		off += 4
	}

	rc := &RadioCortex{
		Neurons:       neurons,
		Fired:         make([]bool, size),
		PrevFired:     make([]bool, size),
		Size:          size,
		TickCount:     tickCount,
		InputStart:    inputStart,
		InputEnd:      inputEnd,
		OutputStart:   outputStart,
		OutputEnd:     outputEnd,
		FireThreshold: fireThreshold,
		PhaseWindow:   phaseWindow,
		// Defaults matching NewRadioCortex so standalone callers work
		TrainAmplitude:      200,
		ResonanceThreshold:  20,
		WeakNeuronThreshold: 32,
		GenerateWindowSize:  8,
		AntiLoopMaxRepeat:   2,
		DecodeTopK:          5,
		rng:           rand.New(rand.NewSource(int64(tickCount))),
	}

	// Reconstruct SignalCodec if a vocab size was stored.
	var codec *SignalCodec
	if vocabSize > 0 {
		codec = NewSignalCodec(vocabSize)
	}

	return rc, codec, nil
}
