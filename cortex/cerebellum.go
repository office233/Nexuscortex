package cortex

import (
	"hash/fnv"
)

// ─────────────────────────────────────────────────────────────────────
// cerebellum.go — Automation & Forward Model Cache
// ─────────────────────────────────────────────────────────────────────
//
// The biological cerebellum contains more than half of all neurons in
// the brain yet occupies only ~10% of brain volume. Its primary role
// is to build internal forward models: predictions of sensory outcomes
// given motor commands. Once a movement (or cognitive operation) is
// well-practiced, the cerebellum can predict the result without
// engaging the slower cortical deliberation pathway.
//
// This implementation provides a hash-addressed cache of SDR→response
// mappings. When the cortex has successfully computed a response for
// an input pattern, the cerebellum caches it. On future encounters
// of the same (or hash-colliding) input, the cached response can be
// returned instantly — bypassing the full neural simulation.
//
// Key properties:
//   - O(1) lookup via FNV-1a hash of the SDR bitfield.
//   - Use-count tracking: frequently accessed entries are valuable.
//   - Confidence score: the cortex can express certainty in the cache.
//   - Pruning: low-use-count entries can be evicted to bound memory.
//   - Hit/miss counters for monitoring cache effectiveness.

// CachedResponse holds a single cached cortical computation result.
type CachedResponse struct {
	Output     SDR     // The output SDR that was produced
	Text       string  // The textual response associated with this output
	UseCount   uint32  // Number of times this cache entry has been hit
	Confidence uint8   // Cortex's confidence in this cached result (0 = none, 255 = max)
}

// Cerebellum is a hash-addressed forward-model cache for SDR patterns.
type Cerebellum struct {
	Cache     map[uint64]CachedResponse
	HitCount  uint64
	MissCount uint64
	MaxSize   int // Maximum cache entries (0 = unlimited)
}

// NewCerebellum creates an empty cerebellum cache with capacity from Config.
func NewCerebellum(cfg Config) *Cerebellum {
	maxSize := cfg.CerebellumMaxCacheSize
	if maxSize <= 0 {
		maxSize = 10000 // Default: 10K entries
	}
	return &Cerebellum{
		Cache:   make(map[uint64]CachedResponse),
		MaxSize: maxSize,
	}
}

// ─────────────────────────────────────────────────────────────────────
// HashSDR — Deterministic SDR fingerprint via FNV-1a
// ─────────────────────────────────────────────────────────────────────

// HashSDR computes a 64-bit FNV-1a hash of the SDR's packed bitfield.
// The hash is used as the cache key. FNV-1a provides excellent
// distribution with minimal computational cost — important because
// hashing runs on every lookup.
func (c *Cerebellum) HashSDR(sdr SDR) uint64 {
	h := fnv.New64a()
	packed := sdr.PackBytes()
	h.Write(packed)
	return h.Sum64()
}

// ─────────────────────────────────────────────────────────────────────
// Lookup — Fast-path cache retrieval
// ─────────────────────────────────────────────────────────────────────

// Lookup attempts to retrieve a cached response for the given input
// SDR. If found, the entry's use count is incremented and HitCount
// is bumped; otherwise MissCount is incremented.
//
// Returns the cached response and true on hit, or a zero
// CachedResponse and false on miss.
func (c *Cerebellum) Lookup(input SDR) (CachedResponse, bool) {
	key := c.HashSDR(input)
	entry, ok := c.Cache[key]
	if !ok {
		c.MissCount++
		return CachedResponse{}, false
	}

	// Promote: increment use count (cap at max uint32).
	if entry.UseCount < ^uint32(0) {
		entry.UseCount++
	}
	c.Cache[key] = entry
	c.HitCount++
	return entry, true
}

// ─────────────────────────────────────────────────────────────────────
// Learn — Cache a new cortical computation result
// ─────────────────────────────────────────────────────────────────────

// Learn stores an input→output mapping in the cache. If an entry with
// the same hash already exists, it is overwritten only if the new
// confidence is higher — modeling the cerebellum's preference for
// more reliable forward models.
func (c *Cerebellum) Learn(input SDR, output SDR, text string, confidence uint8) {
	key := c.HashSDR(input)

	// Keep the higher-confidence entry.
	if existing, ok := c.Cache[key]; ok {
		if confidence <= existing.Confidence {
			return
		}
	}

	c.Cache[key] = CachedResponse{
		Output:     output.Clone(), // Deep copy to prevent aliasing
		Text:       text,
		UseCount:   0,
		Confidence: confidence,
	}

	// Auto-evict if over capacity: remove lowest UseCount entry.
	if c.MaxSize > 0 && len(c.Cache) > c.MaxSize {
		var evictKey uint64
		var evictUse uint32 = ^uint32(0) // max uint32
		for k, entry := range c.Cache {
			if entry.UseCount < evictUse {
				evictUse = entry.UseCount
				evictKey = k
			}
		}
		delete(c.Cache, evictKey)
	}
}

// ─────────────────────────────────────────────────────────────────────
// Stats — Cache performance metrics
// ─────────────────────────────────────────────────────────────────────

// Stats returns the cumulative hit count, miss count, and current
// number of entries in the cache.
func (c *Cerebellum) Stats() (hits, misses uint64, cacheSize int) {
	return c.HitCount, c.MissCount, len(c.Cache)
}

// ─────────────────────────────────────────────────────────────────────
// Prune — Evict low-value cache entries
// ─────────────────────────────────────────────────────────────────────

// Prune removes all cache entries whose use count is strictly below
// minUseCount. Returns the number of entries removed.
//
// This models cerebellar synaptic pruning: forward-model pathways
// that are rarely activated are eliminated to free resources for
// more frequently needed predictions.
func (c *Cerebellum) Prune(minUseCount uint32) int {
	removed := 0
	for key, entry := range c.Cache {
		if entry.UseCount < minUseCount {
			delete(c.Cache, key)
			removed++
		}
	}
	return removed
}

// EvictByResponse removes cache entries matching the specified response text.
func (c *Cerebellum) EvictByResponse(text string) int {
	removed := 0
	for key, entry := range c.Cache {
		if entry.Text == text {
			delete(c.Cache, key)
			removed++
		}
	}
	return removed
}

