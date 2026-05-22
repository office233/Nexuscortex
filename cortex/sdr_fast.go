package cortex

import (
	"math/bits"
)

// ─────────────────────────────────────────────────────────────────────
// Batch / Unrolled SDR Operations
// ─────────────────────────────────────────────────────────────────────
//
// These functions perform the same operations as the SDR methods but
// use loop-unrolling (4 words per iteration) to reduce loop overhead
// on hot paths. All arithmetic is integer-only.

// FastOverlap returns the number of active bits shared between two SDRs.
// Processes 4 uint64 words per iteration with remainder handling.
func FastOverlap(a, b SDR) int {
	aBits := a.Bits
	bBits := b.Bits
	n := len(aBits)
	if len(bBits) < n {
		n = len(bBits)
	}
	if n == 0 {
		return 0
	}

	count := 0
	// Unrolled loop: process 4 words at a time.
	limit := n - (n % 4)
	i := 0
	for ; i < limit; i += 4 {
		count += bits.OnesCount64(aBits[i] & bBits[i])
		count += bits.OnesCount64(aBits[i+1] & bBits[i+1])
		count += bits.OnesCount64(aBits[i+2] & bBits[i+2])
		count += bits.OnesCount64(aBits[i+3] & bBits[i+3])
	}
	// Handle remainder words (0–3 words).
	for ; i < n; i++ {
		count += bits.OnesCount64(aBits[i] & bBits[i])
	}
	return count
}

// FastSimilarity returns the overlap of a and b normalized by the larger
// active count, scaled to uint8 (0 = no overlap, 255 = perfect match).
// Integer-only arithmetic — no float64.
func FastSimilarity(a, b SDR) uint8 {
	maxActive := a.ActiveCount
	if b.ActiveCount > maxActive {
		maxActive = b.ActiveCount
	}
	if maxActive == 0 {
		return 0
	}
	result := FastOverlap(a, b) * 255 / maxActive
	if result > 255 {
		result = 255
	}
	return uint8(result)
}

// FastUnion returns a new SDR that is the bitwise OR of a and b.
// Processes 4 words at a time with inline popcount for ActiveCount.
func FastUnion(a, b SDR) SDR {
	size := a.Size
	if b.Size > size {
		size = b.Size
	}
	result := NewSDR(size)

	rBits := result.Bits
	aBits := a.Bits
	bBits := b.Bits
	na := len(aBits)
	nb := len(bBits)
	nr := len(rBits)

	active := 0

	// Unrolled loop: process 4 words at a time.
	limit := nr - (nr % 4)
	i := 0
	for ; i < limit; i += 4 {
		var va0, va1, va2, va3 uint64
		var vb0, vb1, vb2, vb3 uint64

		if i < na {
			va0 = aBits[i]
		}
		if i+1 < na {
			va1 = aBits[i+1]
		}
		if i+2 < na {
			va2 = aBits[i+2]
		}
		if i+3 < na {
			va3 = aBits[i+3]
		}
		if i < nb {
			vb0 = bBits[i]
		}
		if i+1 < nb {
			vb1 = bBits[i+1]
		}
		if i+2 < nb {
			vb2 = bBits[i+2]
		}
		if i+3 < nb {
			vb3 = bBits[i+3]
		}

		r0 := va0 | vb0
		r1 := va1 | vb1
		r2 := va2 | vb2
		r3 := va3 | vb3
		rBits[i] = r0
		rBits[i+1] = r1
		rBits[i+2] = r2
		rBits[i+3] = r3
		active += bits.OnesCount64(r0)
		active += bits.OnesCount64(r1)
		active += bits.OnesCount64(r2)
		active += bits.OnesCount64(r3)
	}

	// Handle remainder words.
	for ; i < nr; i++ {
		var va, vb uint64
		if i < na {
			va = aBits[i]
		}
		if i < nb {
			vb = bBits[i]
		}
		w := va | vb
		rBits[i] = w
		active += bits.OnesCount64(w)
	}

	result.ActiveCount = active
	return result
}

// BatchSimilarity computes the Similarity of target against every candidate,
// returning a []uint8 of scores. This avoids repeated target.Bits extraction
// and is ideal for vocabulary scans (e.g. AnalogyEngine).
func BatchSimilarity(target SDR, candidates []SDR) []uint8 {
	if len(candidates) == 0 {
		return nil
	}

	scores := make([]uint8, len(candidates))
	tBits := target.Bits
	tActive := target.ActiveCount
	tn := len(tBits)

	for ci, cand := range candidates {
		cBits := cand.Bits
		cn := len(cBits)
		n := tn
		if cn < n {
			n = cn
		}

		// Compute overlap with unrolled loop.
		overlap := 0
		limit := n - (n % 4)
		j := 0
		for ; j < limit; j += 4 {
			overlap += bits.OnesCount64(tBits[j] & cBits[j])
			overlap += bits.OnesCount64(tBits[j+1] & cBits[j+1])
			overlap += bits.OnesCount64(tBits[j+2] & cBits[j+2])
			overlap += bits.OnesCount64(tBits[j+3] & cBits[j+3])
		}
		for ; j < n; j++ {
			overlap += bits.OnesCount64(tBits[j] & cBits[j])
		}

		// Similarity: integer-only.
		maxActive := tActive
		if cand.ActiveCount > maxActive {
			maxActive = cand.ActiveCount
		}
		if maxActive == 0 {
			scores[ci] = 0
			continue
		}
		result := overlap * 255 / maxActive
		if result > 255 {
			result = 255
		}
		scores[ci] = uint8(result)
	}

	return scores
}
