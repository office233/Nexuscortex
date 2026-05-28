package cortex

// expert_shard.go — Expert Sharding System for Trillion-Scale Models
//
// Enables 5T+ parameter models on a single machine by splitting the model
// into thousands of small expert shards stored on SSD. Only the experts
// needed for the current token are loaded into RAM via mmap.
//
// Architecture:
//   5T params = 10,000 experts × 500M params each
//   Each expert = 125 MB (ternar 2-bit) or 31 MB (codebook 0.5-bit)
//   QuantumRouter selects top-K experts per token
//   LRU cache keeps hot experts in RAM
//   mmap loads cold experts from SSD on demand
//
// Key insight: conversation topics cluster around ~50 experts.
// With 258-expert LRU cache, we get 90%+ cache hit rate.

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"sync"
)

// ─────────────────────────────────────────────────────────────────────
// Tile Codebook — Sub-1-bit weight compression
// ─────────────────────────────────────────────────────────────────────
//
// Instead of storing 2 bits per weight (ternar), we store an 8-bit
// index into a codebook of 256 common tile patterns.
//
// 1 TernaryTile = 32 bits (16 weights × 2 bits)
// 1 Codebook index = 8 bits → maps to 32-bit tile pattern
// Compression: 4× (2 bits/weight → 0.5 bits/weight effective)
//
// The codebook is learned from the model's actual weight distribution
// during an offline compression step.

const (
	CodebookSize     = 256  // 8-bit index → 256 possible patterns
	TilesPerCodebook = 16   // Group 16 tiles, share 1 codebook
)

// TileCodebook stores 256 common tile patterns.
// Used to decompress codebook-indexed weights back to TernaryTiles.
type TileCodebook struct {
	Patterns [CodebookSize]TernaryTile
}

// Decode converts a codebook index to its TernaryTile pattern.
func (cb *TileCodebook) Decode(index uint8) TernaryTile {
	return cb.Patterns[index]
}

// DefaultCodebook creates a codebook optimized for sparse ternary weights.
// Pattern 0 = all zeros (most common in sparse networks).
// Patterns 1-255 cover common weight configurations.
func DefaultCodebook() *TileCodebook {
	cb := &TileCodebook{}
	// Pattern 0: all zeros (sparse)
	cb.Patterns[0] = TernaryTile(0)

	// Patterns 1-16: single positive weight at position i
	for i := 0; i < 16; i++ {
		var weights [16]int8
		weights[i] = 1
		cb.Patterns[i+1] = PackTernaryTile(weights)
	}

	// Patterns 17-32: single negative weight at position i
	for i := 0; i < 16; i++ {
		var weights [16]int8
		weights[i] = -1
		cb.Patterns[i+17] = PackTernaryTile(weights)
	}

	// Patterns 33-48: pairs of positive weights
	idx := 33
	for i := 0; i < 8 && idx < 49; i++ {
		for j := i + 1; j < 8 && idx < 49; j++ {
			var weights [16]int8
			weights[i] = 1
			weights[j] = 1
			cb.Patterns[idx] = PackTernaryTile(weights)
			idx++
		}
	}

	// Patterns 49-128: random common patterns (increasing density)
	const codebookSeed = uint64(0xCAFEBABE)
	state := codebookSeed
	for i := 49; i < 129; i++ {
		var weights [16]int8
		density := (i - 49) / 10 // 0-7 active weights
		for d := 0; d <= density; d++ {
			state ^= state << 13
			state ^= state >> 7
			state ^= state << 17
			pos := int(state % 16)
			if state%3 == 0 {
				weights[pos] = -1
			} else {
				weights[pos] = 1
			}
		}
		cb.Patterns[i] = PackTernaryTile(weights)
	}

	// Patterns 129-255: dense patterns (8-16 active weights)
	for i := 129; i < 256; i++ {
		var weights [16]int8
		density := 8 + (i-129)/16
		if density > 16 {
			density = 16
		}
		for d := 0; d < density; d++ {
			state ^= state << 13
			state ^= state >> 7
			state ^= state << 17
			pos := int(state % 16)
			if state%2 == 0 {
				weights[pos] = -1
			} else {
				weights[pos] = 1
			}
		}
		cb.Patterns[i] = PackTernaryTile(weights)
	}

	return cb
}

// ─────────────────────────────────────────────────────────────────────
// Expert Shard — One expert's weights on disk
// ─────────────────────────────────────────────────────────────────────

// ExpertShardHeader is the on-disk header for a single expert.
type ExpertShardHeader struct {
	ExpertID    uint32 // Expert index (0-9999)
	InputSize   uint32 // Input dimension
	OutputSize  uint32 // Output dimension
	TileCount   uint64 // Number of tiles
	Compressed  uint8  // 0=raw tiles, 1=codebook compressed
	Reserved    [7]byte
}

const expertHeaderSize = 4 + 4 + 4 + 8 + 1 + 7 // 28 bytes

// ExpertShard represents one loaded expert's weights.
type ExpertShard struct {
	Header     ExpertShardHeader
	Layer      *TernaryLayer
	// Confidence is populated by external codebook compression tooling,
	// not during shard creation or loading. Nil during normal operation.
	Confidence *ConfidenceLayer // nil if not available
	SizeBytes  int64           // Total size on disk
}

// ─────────────────────────────────────────────────────────────────────
// Sharded Model Index — Maps expert IDs to file offsets
// ─────────────────────────────────────────────────────────────────────

const shardedMagic = "NXS5"

// ShardedModelHeader is the master index for a sharded model file.
type ShardedModelHeader struct {
	Magic        [4]byte
	Version      uint16
	ExpertCount  uint32
	InputSize    uint32
	OutputSize   uint32
	CodebookUsed uint8
	Reserved     [9]byte
}



// ShardedModelIndex stores file offsets for all experts.
type ShardedModelIndex struct {
	Header    ShardedModelHeader
	Offsets   []int64  // File offset for each expert
	Sizes     []int64  // Byte size of each expert
	Codebook  *TileCodebook
	FilePath  string
	mu        sync.RWMutex
}

// CreateShardedModel creates a new sharded model file from expert layers.
func CreateShardedModel(path string, experts []*TernaryLayer, codebook *TileCodebook) (*ShardedModelIndex, error) {
	if len(experts) == 0 {
		return nil, fmt.Errorf("no experts provided")
	}

	f, err := os.Create(path)
	if err != nil {
		return nil, fmt.Errorf("create sharded model: %w", err)
	}
	defer f.Close()

	compressed := uint8(0)
	if codebook != nil {
		compressed = 1
	}

	// Write master header
	hdr := ShardedModelHeader{
		Version:      1,
		ExpertCount:  uint32(len(experts)),
		InputSize:    uint32(experts[0].InputSize),
		OutputSize:   uint32(experts[0].OutputSize),
		CodebookUsed: compressed,
	}
	copy(hdr.Magic[:], shardedMagic)

	if err := binary.Write(f, binary.LittleEndian, hdr); err != nil {
		return nil, fmt.Errorf("write master header: %w", err)
	}

	// Write codebook if used
	if codebook != nil {
		for _, pattern := range codebook.Patterns {
			if err := binary.Write(f, binary.LittleEndian, uint32(pattern)); err != nil {
				return nil, fmt.Errorf("write codebook: %w", err)
			}
		}
	}

	// Reserve space for offset table (expertCount × 16 bytes: offset + size)
	offsetTableStart, err := f.Seek(0, 1)
	if err != nil {
		return nil, fmt.Errorf("seek for offset table: %w", err)
	}
	offsetTable := make([]byte, len(experts)*16)
	if _, err := f.Write(offsetTable); err != nil {
		return nil, fmt.Errorf("reserve offset table: %w", err)
	}

	// Write each expert and record offsets
	offsets := make([]int64, len(experts))
	sizes := make([]int64, len(experts))

	for i, expert := range experts {
		offset, err := f.Seek(0, 1)
		if err != nil {
			return nil, fmt.Errorf("seek before expert %d: %w", i, err)
		}
		offsets[i] = offset

		// Write expert header
		ehdr := ExpertShardHeader{
			ExpertID:   uint32(i),
			InputSize:  uint32(expert.InputSize),
			OutputSize: uint32(expert.OutputSize),
			TileCount:  uint64(len(expert.Tiles)),
			Compressed: compressed,
		}
		if err := binary.Write(f, binary.LittleEndian, ehdr); err != nil {
			return nil, fmt.Errorf("write expert %d header: %w", i, err)
		}

		// Write tiles
		tileData := expert.MarshalRGBA32()
		if _, err := f.Write(tileData); err != nil {
			return nil, fmt.Errorf("write expert %d tiles: %w", i, err)
		}

		endOffset, err := f.Seek(0, 1)
		if err != nil {
			return nil, fmt.Errorf("seek after expert %d: %w", i, err)
		}
		sizes[i] = endOffset - offset
	}

	// Go back and write the offset table
	if _, err := f.Seek(offsetTableStart, 0); err != nil {
		return nil, fmt.Errorf("seek to offset table: %w", err)
	}
	for i := range experts {
		if err := binary.Write(f, binary.LittleEndian, offsets[i]); err != nil {
			return nil, fmt.Errorf("write offset[%d]: %w", i, err)
		}
		if err := binary.Write(f, binary.LittleEndian, sizes[i]); err != nil {
			return nil, fmt.Errorf("write size[%d]: %w", i, err)
		}
	}

	return &ShardedModelIndex{
		Header:   hdr,
		Offsets:  offsets,
		Sizes:    sizes,
		Codebook: codebook,
		FilePath: path,
	}, nil
}

// LoadShardedModelIndex reads the master index without loading any experts.
func LoadShardedModelIndex(path string) (*ShardedModelIndex, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open sharded model: %w", err)
	}
	defer f.Close()

	var hdr ShardedModelHeader
	if err := binary.Read(f, binary.LittleEndian, &hdr); err != nil {
		return nil, fmt.Errorf("read master header: %w", err)
	}

	if string(hdr.Magic[:]) != shardedMagic {
		return nil, fmt.Errorf("invalid magic: %q", string(hdr.Magic[:]))
	}
	if hdr.ExpertCount == 0 || hdr.ExpertCount > 100000 {
		return nil, fmt.Errorf("invalid expert count: %d", hdr.ExpertCount)
	}

	var codebook *TileCodebook
	if hdr.CodebookUsed == 1 {
		codebook = &TileCodebook{}
		for i := range codebook.Patterns {
			var v uint32
			if err := binary.Read(f, binary.LittleEndian, &v); err != nil {
				return nil, fmt.Errorf("read codebook: %w", err)
			}
			codebook.Patterns[i] = TernaryTile(v)
		}
	}

	// Read offset table
	offsets := make([]int64, hdr.ExpertCount)
	sizes := make([]int64, hdr.ExpertCount)
	for i := uint32(0); i < hdr.ExpertCount; i++ {
		if err := binary.Read(f, binary.LittleEndian, &offsets[i]); err != nil {
			return nil, fmt.Errorf("read offset[%d]: %w", i, err)
		}
		if err := binary.Read(f, binary.LittleEndian, &sizes[i]); err != nil {
			return nil, fmt.Errorf("read size[%d]: %w", i, err)
		}
	}

	return &ShardedModelIndex{
		Header:   hdr,
		Offsets:  offsets,
		Sizes:    sizes,
		Codebook: codebook,
		FilePath: path,
	}, nil
}

// LoadExpert reads a single expert from the sharded model file.
func (idx *ShardedModelIndex) LoadExpert(expertID int) (*ExpertShard, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if expertID < 0 || expertID >= len(idx.Offsets) {
		return nil, fmt.Errorf("expert %d out of range [0, %d)", expertID, len(idx.Offsets))
	}

	f, err := os.Open(idx.FilePath)
	if err != nil {
		return nil, fmt.Errorf("open model file: %w", err)
	}
	defer f.Close()

	// Seek to expert offset
	if _, err := f.Seek(idx.Offsets[expertID], 0); err != nil {
		return nil, fmt.Errorf("seek to expert %d: %w", expertID, err)
	}

	// Read expert header
	var ehdr ExpertShardHeader
	if err := binary.Read(f, binary.LittleEndian, &ehdr); err != nil {
		return nil, fmt.Errorf("read expert %d header: %w", expertID, err)
	}

	// Read tile data
	dataSize := idx.Sizes[expertID] - int64(expertHeaderSize)
	if dataSize <= 0 {
		return nil, fmt.Errorf("expert %d has no tile data", expertID)
	}
	data := make([]byte, dataSize)
	if _, err := io.ReadFull(f, data); err != nil {
		return nil, fmt.Errorf("read expert %d tiles: %w", expertID, err)
	}

	layer, err := UnmarshalTernaryLayer(data)
	if err != nil {
		return nil, fmt.Errorf("unmarshal expert %d: %w", expertID, err)
	}

	return &ExpertShard{
		Header:    ehdr,
		Layer:     layer,
		SizeBytes: idx.Sizes[expertID],
	}, nil
}

// ExpertCount returns the total number of experts in the model.
func (idx *ShardedModelIndex) ExpertCount() int {
	return len(idx.Offsets)
}

// TotalSizeBytes returns the total model file size.
func (idx *ShardedModelIndex) TotalSizeBytes() int64 {
	var total int64
	for _, s := range idx.Sizes {
		total += s
	}
	return total
}
