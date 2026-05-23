package cortex

import (
	"encoding/binary"
	"fmt"
	"math/rand"
	"os"
)

// Binary format for NeuroRadioCortex persistence:
//
//   Offset  Size   Field
//   ──────  ─────  ─────────────────
//   0       4      Magic bytes "NXNR"
//   4       1      Version (1)
//   5       4      Size (uint32, tile count)
//   9       8      TickCount (uint64)
//   17      N×12   Tiles (each: 4 bytes weights + 4 bytes confidence + 4 bytes radio)

var neuroRadioMagic = [4]byte{'N', 'X', 'N', 'R'}

const neuroRadioVersion uint8 = 1

// neuroRadioHeaderSize is the fixed header size before the tile array.
const neuroRadioHeaderSize = 4 + 1 + 4 + 8 // = 17

// SaveNeuroRadioCortex writes all tiles + metadata to a binary file.
func SaveNeuroRadioCortex(nrc *NeuroRadioCortex, path string) error {
	if nrc == nil {
		return fmt.Errorf("SaveNeuroRadioCortex: nil cortex")
	}

	totalSize := neuroRadioHeaderSize + nrc.Size*12
	buf := make([]byte, totalSize)

	off := 0

	// Magic.
	copy(buf[off:], neuroRadioMagic[:])
	off += 4

	// Version.
	buf[off] = neuroRadioVersion
	off++

	// Size.
	binary.LittleEndian.PutUint32(buf[off:], uint32(nrc.Size))
	off += 4

	// TickCount.
	binary.LittleEndian.PutUint64(buf[off:], nrc.TickCount)
	off += 8

	// Tiles: 12 bytes each (weights + confidence + radio).
	for i := 0; i < nrc.Size; i++ {
		t := &nrc.Tiles[i]
		binary.LittleEndian.PutUint32(buf[off:], t.Weights)
		off += 4
		binary.LittleEndian.PutUint32(buf[off:], t.Confidence)
		off += 4
		binary.LittleEndian.PutUint32(buf[off:], uint32(t.Radio))
		off += 4
	}

	return os.WriteFile(path, buf, 0600)
}

// LoadNeuroRadioCortex reads tiles from a binary file and returns a
// reconstructed NeuroRadioCortex. The BucketIndex, Codec, and Decoder
// are rebuilt from the tile data.
func LoadNeuroRadioCortex(path string) (*NeuroRadioCortex, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read neuro-radio cortex file: %w", err)
	}

	if len(data) < neuroRadioHeaderSize {
		return nil, fmt.Errorf("neuro-radio cortex file too short: %d bytes", len(data))
	}

	off := 0

	// Magic.
	if data[0] != neuroRadioMagic[0] || data[1] != neuroRadioMagic[1] ||
		data[2] != neuroRadioMagic[2] || data[3] != neuroRadioMagic[3] {
		return nil, fmt.Errorf("neuro-radio cortex bad magic: %q", data[:4])
	}
	off += 4

	// Version.
	version := data[off]
	if version != neuroRadioVersion {
		return nil, fmt.Errorf("neuro-radio cortex unsupported version: %d", version)
	}
	off++

	// Size.
	size := int(binary.LittleEndian.Uint32(data[off:]))
	off += 4

	// Validate total file size.
	expectedSize := neuroRadioHeaderSize + size*12
	if len(data) < expectedSize {
		return nil, fmt.Errorf("neuro-radio cortex file truncated: want %d bytes, got %d", expectedSize, len(data))
	}

	// TickCount.
	tickCount := binary.LittleEndian.Uint64(data[off:])
	off += 8

	// Tiles.
	tiles := make([]NeuroRadioTile, size)
	for i := 0; i < size; i++ {
		tiles[i].Weights = binary.LittleEndian.Uint32(data[off:])
		off += 4
		tiles[i].Confidence = binary.LittleEndian.Uint32(data[off:])
		off += 4
		tiles[i].Radio = RadioNeuron(binary.LittleEndian.Uint32(data[off:]))
		off += 4
	}

	// Reconstruct the cortex.
	rng := rand.New(rand.NewSource(int64(tickCount)))

	codec := NewSemanticFreqCodec()

	nrc := &NeuroRadioCortex{
		Tiles:     tiles,
		Size:      size,
		TickCount: tickCount,
		Codec:     codec,
		rng:       rng,
		// Set safe defaults for config fields that aren't persisted
		DecodeActiveThreshold: 5,
		InitAmpMin:            100,
		InitAmpRange:          156,
		InhibitoryRatioDiv:    5,
		InjectAmplitude:       200,
	}

	// Rebuild bucket index.
	nrc.Index = NewRadioBucketIndex(tiles)

	// Reconstruct the Decoder from codec
	nTok, _, _ := codec.Stats()
	nrc.Decoder = NewOutputNeuronDecoder(codec, nTok)
	nrc.Decoder.threshold = nrc.DecodeActiveThreshold

	return nrc, nil
}
