package cortex

import (
	"math/bits"
	"math/rand"
)

// ─────────────────────────────────────────────────────────────────────
// SDR — Sparse Distributed Representations
// ─────────────────────────────────────────────────────────────────────
//
// An SDR represents a concept as a sparse binary pattern: a small
// number of active bits in a large bitfield. For example, with
// Size=10000 and ActiveCount=50, only 0.5% of bits are on.
//
// Properties that make SDRs powerful for associative memory:
//   - Two SDRs with significant overlap represent related concepts.
//   - Union of SDRs creates a combined representation.
//   - Overlap counting is a single AND + popcount — extremely fast.
//
// The Bits field is packed into []uint64 words for efficient
// bitwise operations (Size=10000 requires ceil(10000/64)=157 words).

// SDR is a sparse distributed representation stored as a packed bitfield.
type SDR struct {
	Bits        []uint64 // Packed bitfield (ceil(Size/64) words)
	Size        int      // Total number of bit positions
	ActiveCount int      // Number of bits currently set (target sparsity)
}

// wordsNeeded returns the number of uint64 words required to hold n bits.
func wordsNeeded(n int) int {
	return (n + 63) / 64
}

// NewSDR creates an empty SDR with no active bits.
func NewSDR(size int) SDR {
	return SDR{
		Bits:        make([]uint64, wordsNeeded(size)),
		Size:        size,
		ActiveCount: 0,
	}
}

// RandomSDR creates an SDR with exactly activeCount random bits set.
// Uses Fisher-Yates partial shuffle to select bit positions uniformly.
func RandomSDR(size, activeCount int, rng *rand.Rand) SDR {
	if activeCount > size {
		activeCount = size
	}

	sdr := NewSDR(size)

	// Generate activeCount unique random indices via partial Fisher-Yates.
	indices := make([]int, size)
	for i := range indices {
		indices[i] = i
	}
	for i := 0; i < activeCount; i++ {
		j := i + rng.Intn(size-i)
		indices[i], indices[j] = indices[j], indices[i]
	}

	for i := 0; i < activeCount; i++ {
		idx := indices[i]
		sdr.Bits[idx/64] |= 1 << uint(idx%64)
	}
	sdr.ActiveCount = activeCount
	return sdr
}

// Set activates the bit at the given index.
func (s *SDR) Set(index int) {
	if index < 0 || index >= s.Size {
		return
	}
	word := index / 64
	bit := uint(index % 64)
	if s.Bits[word]&(1<<bit) == 0 {
		s.Bits[word] |= 1 << bit
		s.ActiveCount++
	}
}

// Clear deactivates the bit at the given index.
func (s *SDR) Clear(index int) {
	if index < 0 || index >= s.Size {
		return
	}
	word := index / 64
	bit := uint(index % 64)
	if s.Bits[word]&(1<<bit) != 0 {
		s.Bits[word] &^= 1 << bit
		s.ActiveCount--
	}
}

// Reset deactivates all bits in the SDR without re-allocating the underlying Bits slice.
func (s *SDR) Reset() {
	for i := range s.Bits {
		s.Bits[i] = 0
	}
	s.ActiveCount = 0
}

// IsActive returns true if the bit at the given index is set.
func (s SDR) IsActive(index int) bool {
	if index < 0 || index >= s.Size {
		return false
	}
	return s.Bits[index/64]&(1<<uint(index%64)) != 0
}

// ActiveIndices returns a sorted list of all active bit positions.
func (s SDR) ActiveIndices() []int {
	result := make([]int, 0, s.ActiveCount)
	for w, word := range s.Bits {
		if word == 0 {
			continue
		}
		base := w * 64
		for word != 0 {
			// Find lowest set bit position.
			tz := bits.TrailingZeros64(word)
			idx := base + tz
			if idx < s.Size {
				result = append(result, idx)
			}
			word &= word - 1 // Clear lowest set bit.
		}
	}
	return result
}

// Overlap returns the number of active bits shared between two SDRs
// (population count of the AND of their bitfields).
// Delegates to FastOverlap for unrolled loop performance.
func (a SDR) Overlap(b SDR) int {
	return FastOverlap(a, b)
}

// Similarity returns the overlap normalized by the larger active count,
// scaled to a uint8 range: 0 = no overlap, 255 = perfect match.
// Uses integer-only arithmetic.
// Delegates to FastSimilarity for unrolled loop performance.
func (a SDR) Similarity(b SDR) uint8 {
	return FastSimilarity(a, b)
}

// ─── Pool vs Allocation strategy ────────────────────────────────────
// Hot-path operations (Union, Intersect) use AcquireSDR to get pooled
// backing storage. Callers may optionally call ReleaseSDR(result) when
// the SDR is no longer needed to return the backing array to the pool.
// If ReleaseSDR is never called, the SDR is garbage-collected normally.
//
// Long-lived SDR constructors (NewSDR, Clone, RandomSDR) allocate
// fresh slices and do NOT use the pool, because those SDRs may be
// stored indefinitely and returning their backing arrays would risk
// aliasing bugs.
// ────────────────────────────────────────────────────────────────────

// Union returns a new SDR that is the bitwise OR of a and b.
// The result uses pooled backing storage; call ReleaseSDR when done
// to return it to the pool (optional — GC handles it if not released).
func (a SDR) Union(b SDR) SDR {
	size := a.Size
	if b.Size > size {
		size = b.Size
	}
	result := AcquireSDR(size)

	na, nb := len(a.Bits), len(b.Bits)
	for i := 0; i < len(result.Bits); i++ {
		var va, vb uint64
		if i < na {
			va = a.Bits[i]
		}
		if i < nb {
			vb = b.Bits[i]
		}
		result.Bits[i] = va | vb
	}
	result.ActiveCount = result.recount()
	return result
}

// Intersect returns a new SDR that is the bitwise AND of a and b.
// The result uses pooled backing storage; call ReleaseSDR when done
// to return it to the pool (optional — GC handles it if not released).
func (a SDR) Intersect(b SDR) SDR {
	size := a.Size
	if b.Size < size {
		size = b.Size
	}
	result := AcquireSDR(size)

	n := len(result.Bits)
	if len(a.Bits) < n {
		n = len(a.Bits)
	}
	if len(b.Bits) < n {
		n = len(b.Bits)
	}
	for i := 0; i < n; i++ {
		result.Bits[i] = a.Bits[i] & b.Bits[i]
	}
	result.ActiveCount = result.recount()
	return result
}

// ToNeuronInputs converts active bit positions into a neuron input map.
// Each active bit index maps to the provided current value, suitable
// for feeding into the spiking neural network.
func (s SDR) ToNeuronInputs(current uint8) map[int]uint8 {
	inputs := make(map[int]uint8, s.ActiveCount)
	for _, idx := range s.ActiveIndices() {
		inputs[idx] = current
	}
	return inputs
}

// Difference returns a new SDR containing bits in a but NOT in b:
// result = a AND (NOT b). This is the set-difference operation used
// for computing analogical transformations between SDR patterns.
func (a SDR) Difference(b SDR) SDR {
	size := a.Size
	result := NewSDR(size)

	na := len(a.Bits)
	nb := len(b.Bits)
	nr := len(result.Bits)
	n := na
	if nr < n {
		n = nr
	}
	for i := 0; i < n; i++ {
		var vb uint64
		if i < nb {
			vb = b.Bits[i]
		}
		result.Bits[i] = a.Bits[i] & ^vb
	}
	result.ActiveCount = result.recount()
	return result
}

// ─────────────────────────────────────────────────────────────────────
// Serialization — Pack/Unpack to raw bytes
// ─────────────────────────────────────────────────────────────────────

// PackBytes serializes the SDR bitfield into a compact byte slice.
// Each uint64 word is written as 8 bytes in little-endian order.
func (s SDR) PackBytes() []byte {
	data := make([]byte, len(s.Bits)*8)
	for i, word := range s.Bits {
		off := i * 8
		data[off+0] = byte(word)
		data[off+1] = byte(word >> 8)
		data[off+2] = byte(word >> 16)
		data[off+3] = byte(word >> 24)
		data[off+4] = byte(word >> 32)
		data[off+5] = byte(word >> 40)
		data[off+6] = byte(word >> 48)
		data[off+7] = byte(word >> 56)
	}
	return data
}

// UnpackBytes deserializes a byte slice into an SDR of the given size.
// The data must have been produced by PackBytes.
func UnpackBytes(data []byte, size int) SDR {
	nw := wordsNeeded(size)
	sdr := SDR{
		Bits: make([]uint64, nw),
		Size: size,
	}
	for i := 0; i < nw && i*8+7 < len(data); i++ {
		off := i * 8
		sdr.Bits[i] = uint64(data[off]) |
			uint64(data[off+1])<<8 |
			uint64(data[off+2])<<16 |
			uint64(data[off+3])<<24 |
			uint64(data[off+4])<<32 |
			uint64(data[off+5])<<40 |
			uint64(data[off+6])<<48 |
			uint64(data[off+7])<<56
	}
	sdr.ActiveCount = sdr.recount()
	return sdr
}

// recount recalculates ActiveCount from the bitfield. Used after
// set operations that may change sparsity unpredictably.
func (s SDR) recount() int {
	count := 0
	for _, word := range s.Bits {
		count += bits.OnesCount64(word)
	}
	return count
}

// Clone returns a deep copy of the SDR, duplicating the backing
// Bits array. Use this whenever an SDR needs to be stored long-term
// (e.g., in queues, workspaces, hippocampus) to prevent aliasing
// where the caller can mutate the SDR after submission.
func (s SDR) Clone() SDR {
	newBits := make([]uint64, len(s.Bits))
	copy(newBits, s.Bits)
	return SDR{
		Bits:        newBits,
		Size:        s.Size,
		ActiveCount: s.ActiveCount,
	}
}
