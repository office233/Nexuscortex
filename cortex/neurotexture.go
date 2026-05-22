package cortex

import (
	"encoding/binary"
	"fmt"
	"math/bits"
	"os"
	"sync"
	"unsafe"
)

// ─────────────────────────────────────────────────────────────────────
// NeuroTexture Engine — RGBA32 Ternary Computation Accelerator
// ─────────────────────────────────────────────────────────────────────
//
// Formalizes the RGBA32 ternary weight packing into a high-performance
// compute engine with:
//
//   1. NeuroTexture file format (.ntx) — raw RGBA32 tiles, mmap-ready
//   2. PopcountLUT — precomputed popcount table for 16-bit masks
//   3. TileCache — hot tile→contribution cache for repeated inputs
//   4. SDRCache — full input→output cache for exact/similar queries
//   5. ForwardBatch — SIMD-style batch processing with popcount
//
// The key insight: ternary weights + sparse SDR input means most
// computation reduces to AND + popcount + add/sub. No multiplication.

// ─────────────────────────────────────────────────────────────────────
// NeuroTexture File Format
// ─────────────────────────────────────────────────────────────────────
//
// Binary format (.ntx):
//   [4B  magic "NTX2"]
//   [2B  version]
//   [4B  width  (tilesPerRow)]
//   [4B  height (outputSize)]
//   [2B  layout (0=row-major, 1=col-major)]
//   [8B  tileCount]
//   [8B  checksum (FNV-1a of tile data)]
//   [2B  reserved]
//   [... raw uint32 tiles]
//   [... int16 biases (2 * height bytes)]

const (
	ntxMagic   = "NTX2"
	ntxVersion = 1
	ntxHeaderSize = 34 // 4+2+4+4+2+8+8+2
)

// NeuroTextureHeader describes a neurotexture file.
type NeuroTextureHeader struct {
	Magic     [4]byte
	Version   uint16
	Width     uint32 // TilesPerRow
	Height    uint32 // OutputSize
	Layout    uint16
	TileCount uint64
	Checksum  uint64
	Reserved  uint16
}

// SaveNeuroTexture writes a TernaryLayer as a .ntx file.
func SaveNeuroTexture(path string, layer *TernaryLayer) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("neurotexture create: %w", err)
	}
	defer f.Close()

	// Compute checksum
	checksum := fnv1aHash(layer.Tiles)

	// Write header
	hdr := NeuroTextureHeader{
		Version:   ntxVersion,
		Width:     uint32(layer.TilesPerRow),
		Height:    uint32(layer.OutputSize),
		Layout:    0, // row-major
		TileCount: uint64(len(layer.Tiles)),
		Checksum:  checksum,
	}
	copy(hdr.Magic[:], ntxMagic)

	if err := binary.Write(f, binary.LittleEndian, hdr); err != nil {
		return fmt.Errorf("neurotexture write header: %w", err)
	}

	// Write tiles as raw uint32
	tileBytes := make([]byte, len(layer.Tiles)*4)
	for i, t := range layer.Tiles {
		binary.LittleEndian.PutUint32(tileBytes[i*4:], uint32(t))
	}
	if _, err := f.Write(tileBytes); err != nil {
		return fmt.Errorf("neurotexture write tiles: %w", err)
	}

	// Write biases
	biasBytes := make([]byte, len(layer.Bias)*2)
	for i, b := range layer.Bias {
		binary.LittleEndian.PutUint16(biasBytes[i*2:], uint16(b))
	}
	if _, err := f.Write(biasBytes); err != nil {
		return fmt.Errorf("neurotexture write biases: %w", err)
	}

	return nil
}

// LoadNeuroTexture reads a .ntx file into a TernaryLayer.
func LoadNeuroTexture(path string) (*TernaryLayer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("neurotexture read: %w", err)
	}

	if len(data) < ntxHeaderSize {
		return nil, fmt.Errorf("neurotexture too short: %d bytes", len(data))
	}

	var hdr NeuroTextureHeader
	if err := binary.Read(
		newByteReader(data[:ntxHeaderSize]),
		binary.LittleEndian, &hdr,
	); err != nil {
		return nil, fmt.Errorf("neurotexture read header: %w", err)
	}

	if string(hdr.Magic[:]) != ntxMagic {
		return nil, fmt.Errorf("neurotexture invalid magic: %q", string(hdr.Magic[:]))
	}

	if hdr.Width == 0 || hdr.Height == 0 || hdr.TileCount == 0 {
		return nil, fmt.Errorf("neurotexture invalid dimensions: %dx%d tiles=%d",
			hdr.Width, hdr.Height, hdr.TileCount)
	}

	// Safety limits
	if hdr.Width > 100000 || hdr.Height > 100000 || hdr.TileCount > 100_000_000 {
		return nil, fmt.Errorf("neurotexture dimensions exceed safety limits")
	}

	expectedDataSize := ntxHeaderSize + int(hdr.TileCount)*4 + int(hdr.Height)*2
	if len(data) < expectedDataSize {
		return nil, fmt.Errorf("neurotexture data too short: got %d, need %d",
			len(data), expectedDataSize)
	}

	// Read tiles
	layer := &TernaryLayer{
		InputSize:   int(hdr.Width) * 16,
		OutputSize:  int(hdr.Height),
		TilesPerRow: int(hdr.Width),
		Tiles:       make([]TernaryTile, hdr.TileCount),
		Bias:        make([]int16, hdr.Height),
	}

	offset := ntxHeaderSize
	for i := range layer.Tiles {
		layer.Tiles[i] = TernaryTile(binary.LittleEndian.Uint32(data[offset:]))
		offset += 4
	}
	for i := range layer.Bias {
		layer.Bias[i] = int16(binary.LittleEndian.Uint16(data[offset:]))
		offset += 2
	}

	// Verify checksum
	checksum := fnv1aHash(layer.Tiles)
	if checksum != hdr.Checksum {
		return nil, fmt.Errorf("neurotexture checksum mismatch: file=%x computed=%x",
			hdr.Checksum, checksum)
	}

	return layer, nil
}

// ─────────────────────────────────────────────────────────────────────
// Popcount LUT — Precomputed popcount for 16-bit values
// ─────────────────────────────────────────────────────────────────────
//
// Used for the core operation: AND + popcount + add/sub.
// Instead of iterating bits, we do:
//   contribution = pop16[positive_mask & input_mask] - pop16[negative_mask & input_mask]

var pop16 [65536]uint8

func init() {
	for i := 0; i < 65536; i++ {
		pop16[i] = uint8(bits.OnesCount16(uint16(i)))
	}
}

// ForwardPopcount performs forward pass using popcount-based computation.
// This is the fastest path for SDR (binary) input where all active values = 1.
//
// Instead of iterating individual bits:
//   For each tile: contribution = popcount(pos_mask & active) - popcount(neg_mask & active)
//
// This reduces the inner loop to 2 ANDs + 2 table lookups + 1 subtraction per tile.
func (l *TernaryLayer) ForwardPopcount(activeMask []uint16) []int16 {
	output := make([]int16, l.OutputSize)
	copy(output, l.Bias)

	for j := 0; j < l.OutputSize; j++ {
		var acc int32
		rowOffset := j * l.TilesPerRow

		for t := 0; t < l.TilesPerRow; t++ {
			tile := uint32(l.Tiles[rowOffset+t])
			if tile == 0 {
				continue // Fully sparse tile — skip entirely
			}

			// Extract sign and mask for lower and upper 8 weights
			signLo := uint8(tile)
			maskLo := uint8(tile >> 8)
			signHi := uint8(tile >> 16)
			maskHi := uint8(tile >> 24)

			// Combine into 16-bit masks
			posMask := uint16(maskLo&^signLo) | uint16(maskHi&^signHi)<<8
			negMask := uint16(maskLo&signLo) | uint16(maskHi&signHi)<<8

			// AND with input active mask for this tile position
			if t < len(activeMask) {
				active := activeMask[t]
				acc += int32(pop16[posMask&active]) - int32(pop16[negMask&active])
			}
		}

		if acc > 32767 {
			acc = 32767
		} else if acc < -32768 {
			acc = -32768
		}
		output[j] += int16(acc)
	}

	return output
}

// SDRToActiveMask converts an SDR to a tile-aligned active mask for ForwardPopcount.
// Returns a []uint16 where each uint16 corresponds to one tile (16 bits).
func SDRToActiveMask(sdr SDR, tilesPerRow int) []uint16 {
	masks := make([]uint16, tilesPerRow)
	for _, idx := range sdr.ActiveIndices() {
		tileIdx := idx / 16
		bitPos := idx % 16
		if tileIdx < tilesPerRow {
			masks[tileIdx] |= 1 << uint(bitPos)
		}
	}
	return masks
}

// ─────────────────────────────────────────────────────────────────────
// SDR Cache — Skip computation entirely for repeated/similar inputs
// ─────────────────────────────────────────────────────────────────────

// SDRCache stores input→output mappings for exact SDR matches.
// Cache hit means ZERO computation — instant response.
type SDRCache struct {
	mu       sync.RWMutex
	cache    map[uint64]SDR // hash → output SDR
	maxSize  int
	hits     uint64
	misses   uint64
}

// NewSDRCache creates a cache with the given capacity.
func NewSDRCache(maxSize int) *SDRCache {
	return &SDRCache{
		cache:   make(map[uint64]SDR, maxSize),
		maxSize: maxSize,
	}
}

// Lookup checks if an exact match exists for the input SDR.
func (c *SDRCache) Lookup(input SDR) (SDR, bool) {
	h := input.Hash()
	c.mu.RLock()
	result, ok := c.cache[h]
	c.mu.RUnlock()

	if ok {
		c.mu.Lock()
		c.hits++
		c.mu.Unlock()
	} else {
		c.mu.Lock()
		c.misses++
		c.mu.Unlock()
	}
	return result, ok
}

// Store saves an input→output mapping.
func (c *SDRCache) Store(input, output SDR) {
	h := input.Hash()
	c.mu.Lock()
	defer c.mu.Unlock()

	// Simple eviction: clear half the cache when full
	if len(c.cache) >= c.maxSize {
		count := 0
		for k := range c.cache {
			delete(c.cache, k)
			count++
			if count >= c.maxSize/2 {
				break
			}
		}
	}

	c.cache[h] = output.Clone()
}

// HitRate returns the cache hit ratio (0.0 to 1.0).
func (c *SDRCache) HitRate() float64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	total := c.hits + c.misses
	if total == 0 {
		return 0
	}
	return float64(c.hits) / float64(total)
}

// Stats returns cache statistics.
func (c *SDRCache) Stats() (size int, hits, misses uint64) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.cache), c.hits, c.misses
}

// ─────────────────────────────────────────────────────────────────────
// Expert Router — Top-K sparse routing for FractalCortex blocks
// ─────────────────────────────────────────────────────────────────────

// ExpertRouter selects the top-K most relevant CortexBlocks for a
// given input, avoiding running ALL blocks (which is wasteful for
// large numbers of experts).
type ExpertRouter struct {
	// Each expert has a domain embedding (compact SDR fingerprint)
	ExpertEmbeddings []SDR
	// Running usage statistics per expert
	UsageCounts []uint64
	TopK        int
}

// NewExpertRouter creates a router for the given number of experts.
func NewExpertRouter(numExperts, sdrSize, topK int) *ExpertRouter {
	embeddings := make([]SDR, numExperts)
	for i := range embeddings {
		embeddings[i] = NewSDR(sdrSize)
	}
	return &ExpertRouter{
		ExpertEmbeddings: embeddings,
		UsageCounts:      make([]uint64, numExperts),
		TopK:             topK,
	}
}

// Route selects the top-K experts most similar to the input SDR.
// Returns expert indices sorted by relevance (most relevant first).
func (r *ExpertRouter) Route(input SDR) []int {
	if len(r.ExpertEmbeddings) <= r.TopK {
		// Fewer experts than K — use all
		result := make([]int, len(r.ExpertEmbeddings))
		for i := range result {
			result[i] = i
		}
		return result
	}

	type scored struct {
		idx   int
		score uint8
	}

	// Score all experts
	scores := make([]scored, len(r.ExpertEmbeddings))
	for i, emb := range r.ExpertEmbeddings {
		scores[i] = scored{i, emb.Similarity(input)}
	}

	// Partial sort: find top-K
	topK := make([]scored, 0, r.TopK)
	for _, s := range scores {
		if len(topK) < r.TopK {
			topK = append(topK, s)
			// Bubble up
			for j := len(topK) - 1; j > 0; j-- {
				if topK[j].score > topK[j-1].score {
					topK[j], topK[j-1] = topK[j-1], topK[j]
				}
			}
		} else if s.score > topK[r.TopK-1].score {
			topK[r.TopK-1] = s
			for j := r.TopK - 1; j > 0; j-- {
				if topK[j].score > topK[j-1].score {
					topK[j], topK[j-1] = topK[j-1], topK[j]
				}
			}
		}
	}

	result := make([]int, len(topK))
	for i, s := range topK {
		result[i] = s.idx
		r.UsageCounts[s.idx]++
	}
	return result
}

// UpdateEmbedding updates an expert's domain embedding based on its
// recent inputs (running average via OR + decay).
func (r *ExpertRouter) UpdateEmbedding(expertIdx int, input SDR) {
	if expertIdx < 0 || expertIdx >= len(r.ExpertEmbeddings) {
		return
	}
	// Simple update: set bits that are active in the input
	emb := &r.ExpertEmbeddings[expertIdx]
	for _, idx := range input.ActiveIndices() {
		if idx < emb.Size {
			emb.Set(idx)
		}
	}
}

// ─────────────────────────────────────────────────────────────────────
// Plasticity Journal — Delta-based weight updates
// ─────────────────────────────────────────────────────────────────────

// PlasticityEntry records a single weight change for replay/merge.
type PlasticityEntry struct {
	ExpertID  int          `json:"expert_id"`
	LayerName string       `json:"layer_name"`
	TileIndex int          `json:"tile_index"`
	OldTile   TernaryTile  `json:"old_tile"`
	NewTile   TernaryTile  `json:"new_tile"`
	Reason    string       `json:"reason"` // "hebbian", "error", "reward", "sleep"
}

// PlasticityJournal accumulates weight changes that can be merged
// into neurotexture files during Sleep().
type PlasticityJournal struct {
	mu      sync.Mutex
	entries []PlasticityEntry
}

// NewPlasticityJournal creates an empty journal.
func NewPlasticityJournal() *PlasticityJournal {
	return &PlasticityJournal{
		entries: make([]PlasticityEntry, 0, 1024),
	}
}

// Record adds a plasticity event to the journal.
func (j *PlasticityJournal) Record(entry PlasticityEntry) {
	j.mu.Lock()
	j.entries = append(j.entries, entry)
	j.mu.Unlock()
}

// Merge applies all recorded changes to the given layer and clears the journal.
// Returns the number of changes applied.
func (j *PlasticityJournal) Merge(layer *TernaryLayer, expertID int, layerName string) int {
	j.mu.Lock()
	defer j.mu.Unlock()

	applied := 0
	remaining := make([]PlasticityEntry, 0, len(j.entries))

	for _, e := range j.entries {
		if e.ExpertID == expertID && e.LayerName == layerName {
			if e.TileIndex >= 0 && e.TileIndex < len(layer.Tiles) {
				// Verify the old tile matches (conflict detection)
				if layer.Tiles[e.TileIndex] == e.OldTile {
					layer.Tiles[e.TileIndex] = e.NewTile
					applied++
				}
			}
		} else {
			remaining = append(remaining, e)
		}
	}

	j.entries = remaining
	return applied
}

// Size returns the number of pending entries.
func (j *PlasticityJournal) Size() int {
	j.mu.Lock()
	defer j.mu.Unlock()
	return len(j.entries)
}

// Clear removes all entries.
func (j *PlasticityJournal) Clear() {
	j.mu.Lock()
	j.entries = j.entries[:0]
	j.mu.Unlock()
}

// ─────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────

// fnv1aHash computes FNV-1a hash of tile data for checksum.
func fnv1aHash(tiles []TernaryTile) uint64 {
	h := uint64(14695981039346656037) // FNV offset basis
	for _, t := range tiles {
		v := uint32(t)
		h ^= uint64(v & 0xFF)
		h *= 1099511628211
		h ^= uint64((v >> 8) & 0xFF)
		h *= 1099511628211
		h ^= uint64((v >> 16) & 0xFF)
		h *= 1099511628211
		h ^= uint64((v >> 24) & 0xFF)
		h *= 1099511628211
	}
	return h
}

// byteReader adapts a byte slice for binary.Read.
type byteReader struct {
	data []byte
	pos  int
}

func newByteReader(data []byte) *byteReader {
	return &byteReader{data: data}
}

func (r *byteReader) Read(p []byte) (int, error) {
	n := copy(p, r.data[r.pos:])
	r.pos += n
	return n, nil
}

// ─────────────────────────────────────────────────────────────────────
// Benchmark helpers
// ─────────────────────────────────────────────────────────────────────

// LayerMemoryReport returns a detailed memory breakdown.
func LayerMemoryReport(l *TernaryLayer) string {
	tilesBytes := len(l.Tiles) * int(unsafe.Sizeof(TernaryTile(0)))
	biasBytes := len(l.Bias) * 2
	totalBytes := tilesBytes + biasBytes

	equivFloat32 := l.InputSize * l.OutputSize * 4
	equivInt8 := l.InputSize * l.OutputSize

	return fmt.Sprintf(
		"NeuroTexture: %dx%d = %d params\n"+
			"  RGBA32 tiles: %d bytes (%.1f KB)\n"+
			"  Biases:       %d bytes\n"+
			"  Total:        %d bytes (%.1f KB)\n"+
			"  vs float32:   %d bytes (%.1fx compression)\n"+
			"  vs int8:      %d bytes (%.1fx compression)\n"+
			"  Sparsity:     %.1f%%",
		l.InputSize, l.OutputSize, l.ParameterCount(),
		tilesBytes, float64(tilesBytes)/1024,
		biasBytes,
		totalBytes, float64(totalBytes)/1024,
		equivFloat32, float64(equivFloat32)/float64(totalBytes),
		equivInt8, float64(equivInt8)/float64(totalBytes),
		l.SparsityRatio()*100,
	)
}
