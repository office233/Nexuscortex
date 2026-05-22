package cortex

import "sync"

// ─────────────────────────────────────────────────────────────────────
// SDR Pool Allocator — Memory Optimization
// ─────────────────────────────────────────────────────────────────────
//
// Every SDR operation (Union, Intersect, etc.) allocates a new []uint64
// backing slice. During Process(), SDRs are created and discarded at
// high frequency, creating GC pressure. This pool reuses backing arrays
// to reduce allocation frequency for short-lived intermediates.
//
// Usage pattern:
//   - Hot-path operations (Union, Intersect) use AcquireSDR internally
//     to get pooled backing storage.
//   - Callers who know they have a temporary SDR can optionally call
//     ReleaseSDR when done to return the backing array to the pool.
//   - If ReleaseSDR is never called, the SDR is GC'd normally — no leaks.
//
// Operations that create LONG-LIVED SDRs (NewSDR, Clone, RandomSDR)
// do NOT use the pool, as those SDRs may persist indefinitely and
// returning their backing arrays would cause aliasing bugs.

// sdrPools maps word-count → sync.Pool for that size class.
var sdrPools = map[int]*sync.Pool{}

// sdrPoolMu protects the sdrPools map during lazy initialization.
var sdrPoolMu sync.Mutex

type pooledSDRBuffer struct {
	bits []uint64
}

// getSDRPool returns (or creates) a sync.Pool for slices of the given
// word count. Each distinct word count gets its own pool to avoid
// mixing slice capacities.
func getSDRPool(words int) *sync.Pool {
	sdrPoolMu.Lock()
	defer sdrPoolMu.Unlock()
	if p, ok := sdrPools[words]; ok {
		return p
	}
	w := words // capture for closure
	p := &sync.Pool{
		New: func() interface{} {
			return &pooledSDRBuffer{bits: make([]uint64, w)}
		},
	}
	sdrPools[words] = p
	return p
}

// AcquireSDR obtains an SDR with the given bit-size from the pool.
// The backing slice is zeroed before returning to ensure a clean state.
// The returned SDR has ActiveCount=0 and all bits cleared.
func AcquireSDR(size int) SDR {
	words := wordsNeeded(size)
	pool := getSDRPool(words)
	buf := pool.Get().(*pooledSDRBuffer).bits

	// Zero the slice — it may contain data from a previous use.
	for i := range buf {
		buf[i] = 0
	}

	return SDR{
		Bits:        buf,
		Size:        size,
		ActiveCount: 0,
	}
}

// ReleaseSDR returns the SDR's backing array to the pool for reuse.
// After calling ReleaseSDR, the SDR must not be read or written.
// It is safe (no-op) to call ReleaseSDR on an SDR with nil Bits.
func ReleaseSDR(sdr SDR) {
	if sdr.Bits == nil {
		return
	}
	words := wordsNeeded(sdr.Size)
	// Only return slices that match the expected length for their size
	// class to prevent pool corruption from manually constructed SDRs.
	if len(sdr.Bits) != words {
		return
	}
	pool := getSDRPool(words)
	pool.Put(&pooledSDRBuffer{bits: sdr.Bits})
}
