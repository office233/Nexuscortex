package cortex

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
)

// Binary format for NeuroRadioCortex persistence:
//
// Version 2:
//
//   Offset  Size   Field
//   ------  -----  ----------------
//   0       4      Magic bytes "NXNR"
//   4       1      Version (2)
//   5       4      Size (uint32, tile count)
//   9       8      TickCount (uint64)
//   17      4      DecodeActiveThreshold (int32)
//   21      4      InitAmpMin (int32)
//   25      4      InitAmpRange (int32)
//   29      4      InhibitoryRatioDiv (int32)
//   33      4      InjectAmplitude (int32)
//   37      4      CodecJSON length (uint32)
//   41      N*12   Tiles (each: 4 bytes weights + 4 bytes confidence + 4 bytes radio)
//   ...     M      SemanticFreqCodec JSON snapshot
//
// Version 1 files are still accepted for backward compatibility, but they only
// contain the legacy header and tiles. The semantic codec is initialized empty.
var neuroRadioMagic = [4]byte{'N', 'X', 'N', 'R'}

const (
	neuroRadioVersion           uint8 = 2
	neuroRadioLegacyVersion     uint8 = 1
	neuroRadioLegacyHeaderSize        = 4 + 1 + 4 + 8           // = 17
	neuroRadioHeaderSize              = 4 + 1 + 4 + 8 + 5*4 + 4 // = 41
	neuroRadioTileSize                = 12
	neuroRadioMaxTiles                = 100_000_000
)

type neuroRadioCodecSnapshot struct {
	TokenFreqs   map[int]uint8         `json:"token_freqs"`
	TokenFreqSet map[int][]uint8       `json:"token_freq_set"`
	Cooccurrence map[int]map[int]int   `json:"cooccurrence"`
	VocabSize    int                   `json:"vocab_size"`
	Dirty        bool                  `json:"dirty"`
}

func snapshotSemanticFreqCodec(codec *SemanticFreqCodec) neuroRadioCodecSnapshot {
	if codec == nil {
		return neuroRadioCodecSnapshot{}
	}
	return neuroRadioCodecSnapshot{
		TokenFreqs:   cloneUint8Map(codec.tokenFreqs),
		TokenFreqSet: cloneFreqSetMap(codec.tokenFreqSet),
		Cooccurrence: cloneCooccurrenceMap(codec.cooccurrence),
		VocabSize:    codec.vocabSize,
		Dirty:        codec.dirty,
	}
}

func restoreSemanticFreqCodec(snapshot neuroRadioCodecSnapshot) *SemanticFreqCodec {
	codec := NewSemanticFreqCodec()
	codec.tokenFreqs = cloneUint8Map(snapshot.TokenFreqs)
	codec.tokenFreqSet = cloneFreqSetMap(snapshot.TokenFreqSet)
	codec.cooccurrence = cloneCooccurrenceMap(snapshot.Cooccurrence)
	codec.vocabSize = snapshot.VocabSize
	codec.dirty = snapshot.Dirty

	// Rebuild inverse index from the persisted token frequency sets. It is
	// intentionally derived instead of trusted from disk.
	codec.freqToTokens = make(map[uint8][]int)
	for tokenID, freqs := range codec.tokenFreqSet {
		for _, freq := range freqs {
			codec.freqToTokens[freq] = append(codec.freqToTokens[freq], tokenID)
		}
	}
	return codec
}

func cloneUint8Map(src map[int]uint8) map[int]uint8 {
	if len(src) == 0 {
		return make(map[int]uint8)
	}
	dst := make(map[int]uint8, len(src))
	for k, v := range src {
		dst[k] = v
	}
	return dst
}

func cloneFreqSetMap(src map[int][]uint8) map[int][]uint8 {
	if len(src) == 0 {
		return make(map[int][]uint8)
	}
	dst := make(map[int][]uint8, len(src))
	for k, v := range src {
		dst[k] = append([]uint8(nil), v...)
	}
	return dst
}

func cloneCooccurrenceMap(src map[int]map[int]int) map[int]map[int]int {
	if len(src) == 0 {
		return make(map[int]map[int]int)
	}
	dst := make(map[int]map[int]int, len(src))
	for k, inner := range src {
		dstInner := make(map[int]int, len(inner))
		for kk, vv := range inner {
			dstInner[kk] = vv
		}
		dst[k] = dstInner
	}
	return dst
}

// SaveNeuroRadioCortex writes all tiles, runtime config, and semantic frequency
// codec state to a binary file. Persisting the codec is critical: without it,
// trained token-to-frequency semantics are lost after reload.
func SaveNeuroRadioCortex(nrc *NeuroRadioCortex, path string) error {
	if nrc == nil {
		return fmt.Errorf("SaveNeuroRadioCortex: nil cortex")
	}

	nrc.mu.Lock()
	defer nrc.mu.Unlock()

	codecJSON, err := json.Marshal(snapshotSemanticFreqCodec(nrc.Codec))
	if err != nil {
		return fmt.Errorf("marshal neuro-radio codec: %w", err)
	}
	if len(codecJSON) > int(^uint32(0)) {
		return fmt.Errorf("neuro-radio codec snapshot too large: %d bytes", len(codecJSON))
	}

	totalSize := neuroRadioHeaderSize + nrc.Size*neuroRadioTileSize + len(codecJSON)
	buf := make([]byte, totalSize)
	off := 0

	copy(buf[off:], neuroRadioMagic[:])
	off += 4
	buf[off] = neuroRadioVersion
	off++

	binary.LittleEndian.PutUint32(buf[off:], uint32(nrc.Size))
	off += 4
	binary.LittleEndian.PutUint64(buf[off:], nrc.TickCount)
	off += 8

	binary.LittleEndian.PutUint32(buf[off:], uint32(int32(nrc.DecodeActiveThreshold)))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:], uint32(int32(nrc.InitAmpMin)))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:], uint32(int32(nrc.InitAmpRange)))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:], uint32(int32(nrc.InhibitoryRatioDiv)))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:], uint32(int32(nrc.InjectAmplitude)))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:], uint32(len(codecJSON)))
	off += 4

	for i := 0; i < nrc.Size; i++ {
		t := &nrc.Tiles[i]
		binary.LittleEndian.PutUint32(buf[off:], t.Weights)
		off += 4
		binary.LittleEndian.PutUint32(buf[off:], t.Confidence)
		off += 4
		binary.LittleEndian.PutUint32(buf[off:], uint32(t.Radio))
		off += 4
	}

	copy(buf[off:], codecJSON)
	return os.WriteFile(path, buf, 0600)
}

// LoadNeuroRadioCortex reads a persisted NeuroRadioCortex. It supports both the
// current v2 format and legacy v1 files. Optional Config overrides runtime
// defaults after load while preserving persisted codec semantics.
func LoadNeuroRadioCortex(path string, cfg ...Config) (*NeuroRadioCortex, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read neuro-radio cortex file: %w", err)
	}
	if len(data) < neuroRadioLegacyHeaderSize {
		return nil, fmt.Errorf("neuro-radio cortex file too short: %d bytes", len(data))
	}

	if data[0] != neuroRadioMagic[0] || data[1] != neuroRadioMagic[1] ||
		data[2] != neuroRadioMagic[2] || data[3] != neuroRadioMagic[3] {
		return nil, fmt.Errorf("neuro-radio cortex bad magic: %q", data[:4])
	}

	off := 4
	version := data[off]
	off++
	if version != neuroRadioVersion && version != neuroRadioLegacyVersion {
		return nil, fmt.Errorf("neuro-radio cortex unsupported version: %d", version)
	}

	size := int(binary.LittleEndian.Uint32(data[off:]))
	off += 4
	if size < 0 || size > neuroRadioMaxTiles {
		return nil, fmt.Errorf("neuro-radio cortex file has invalid tile count: %d", size)
	}

	tickCount := binary.LittleEndian.Uint64(data[off:])
	off += 8

	cfgDefaults := DefaultConfig()
	decodeActiveThreshold := cfgDefaults.NRCDecodeActiveThreshold
	initAmpMin := cfgDefaults.NRCInitAmpMin
	initAmpRange := cfgDefaults.NRCInitAmpRange
	inhibitoryRatioDiv := cfgDefaults.NRCInhibitoryRatioDiv
	injectAmplitude := cfgDefaults.NRCInjectAmplitude
	codecLen := 0

	if version == neuroRadioVersion {
		if len(data) < neuroRadioHeaderSize {
			return nil, fmt.Errorf("neuro-radio cortex v2 file too short: %d bytes", len(data))
		}
		decodeActiveThreshold = int(int32(binary.LittleEndian.Uint32(data[off:])))
		off += 4
		initAmpMin = int(int32(binary.LittleEndian.Uint32(data[off:])))
		off += 4
		initAmpRange = int(int32(binary.LittleEndian.Uint32(data[off:])))
		off += 4
		inhibitoryRatioDiv = int(int32(binary.LittleEndian.Uint32(data[off:])))
		off += 4
		injectAmplitude = int(int32(binary.LittleEndian.Uint32(data[off:])))
		off += 4
		codecLen = int(binary.LittleEndian.Uint32(data[off:]))
		off += 4
	}

	expectedSize := off + size*neuroRadioTileSize + codecLen
	if expectedSize < off || expectedSize > len(data) {
		return nil, fmt.Errorf("neuro-radio cortex file truncated: want %d bytes, got %d", expectedSize, len(data))
	}

	tiles := make([]NeuroRadioTile, size)
	for i := 0; i < size; i++ {
		tiles[i].Weights = binary.LittleEndian.Uint32(data[off:])
		off += 4
		tiles[i].Confidence = binary.LittleEndian.Uint32(data[off:])
		off += 4
		tiles[i].Radio = RadioNeuron(binary.LittleEndian.Uint32(data[off:]))
		off += 4
	}

	codec := NewSemanticFreqCodec()
	if codecLen > 0 {
		var snapshot neuroRadioCodecSnapshot
		if err := json.Unmarshal(data[off:off+codecLen], &snapshot); err != nil {
			return nil, fmt.Errorf("unmarshal neuro-radio codec: %w", err)
		}
		codec = restoreSemanticFreqCodec(snapshot)
	}

	if len(cfg) > 0 {
		c := cfg[0]
		decodeActiveThreshold = c.NRCDecodeActiveThreshold
		initAmpMin = c.NRCInitAmpMin
		initAmpRange = c.NRCInitAmpRange
		inhibitoryRatioDiv = c.NRCInhibitoryRatioDiv
		injectAmplitude = c.NRCInjectAmplitude
	}

	nrc := &NeuroRadioCortex{
		Tiles:                 tiles,
		Size:                  size,
		TickCount:             tickCount,
		Codec:                 codec,
		rng:                   rand.New(rand.NewSource(int64(tickCount))),
		DecodeActiveThreshold: decodeActiveThreshold,
		InitAmpMin:            initAmpMin,
		InitAmpRange:          initAmpRange,
		InhibitoryRatioDiv:    inhibitoryRatioDiv,
		InjectAmplitude:       injectAmplitude,
	}

	nrc.Index = NewRadioBucketIndex(tiles)
	nTok, _, _ := codec.Stats()
	nrc.Decoder = NewOutputNeuronDecoder(codec, nTok)
	nrc.Decoder.threshold = nrc.DecodeActiveThreshold

	return nrc, nil
}
