package cortex

// expert_cache.go — LRU Expert Cache for Trillion-Scale Inference
//
// Keeps the most recently used expert shards in RAM to avoid
// repeated SSD reads. With 8 GB budget and 31 MB per expert,
// we cache ~258 experts. Conversations typically use ~50 experts,
// giving 90%+ cache hit rate.
//
// Cache hits = 0ms (instant, from RAM)
// Cache misses = ~10ms (SSD read via mmap)

import (
	"container/list"
	"fmt"
	"sync"
)

// ExpertCache is an LRU cache for loaded expert shards.
// Thread-safe for concurrent access during inference.
type ExpertCache struct {
	mu       sync.Mutex
	capacity int                         // Max number of experts to cache
	items    map[int]*list.Element       // expertID → list element
	order    *list.List                  // LRU order (front = most recent)
	index    *ShardedModelIndex          // Model index for loading misses
	
	// inflight deduplicates concurrent disk reads for the same expert.
	inflight map[int]chan struct{}
	
	// Statistics
	hits     uint64
	misses   uint64
	evictions uint64
}

type cacheEntry struct {
	expertID int
	shard    *ExpertShard
}

// NewExpertCache creates a cache with the given capacity (number of experts).
// index is used to load experts on cache miss.
func NewExpertCache(capacity int, index *ShardedModelIndex) *ExpertCache {
	if capacity < 1 {
		capacity = 1
	}
	return &ExpertCache{
		capacity: capacity,
		items:    make(map[int]*list.Element, capacity),
		order:    list.New(),
		index:    index,
		inflight: make(map[int]chan struct{}),
	}
}

// Get retrieves an expert from cache, loading from disk on miss.
// Returns the expert shard and whether it was a cache hit.
func (c *ExpertCache) Get(expertID int) (*ExpertShard, bool, error) {
	// Fast path: check cache under lock
	c.mu.Lock()
	if elem, ok := c.items[expertID]; ok {
		c.order.MoveToFront(elem)
		c.hits++
		c.mu.Unlock()
		return elem.Value.(*cacheEntry).shard, true, nil
	}

	// Check if another goroutine is already loading this expert
	if wait, ok := c.inflight[expertID]; ok {
		c.mu.Unlock()
		// Wait for the in-flight load to complete
		<-wait
		// Re-check cache — the loading goroutine will have inserted it
		c.mu.Lock()
		if elem, ok := c.items[expertID]; ok {
			c.order.MoveToFront(elem)
			c.mu.Unlock()
			return elem.Value.(*cacheEntry).shard, false, nil
		}
		c.mu.Unlock()
		// The in-flight load must have failed; fall through to try again
		return nil, false, fmt.Errorf("cache miss load expert %d: in-flight load failed", expertID)
	}

	// Mark this expert as in-flight so concurrent callers wait
	done := make(chan struct{})
	c.inflight[expertID] = done
	c.mu.Unlock()

	// Slow path: disk I/O happens WITHOUT holding the lock
	shard, err := c.index.LoadExpert(expertID)

	// Clean up in-flight state and notify waiters
	c.mu.Lock()
	delete(c.inflight, expertID)
	close(done)

	if err != nil {
		c.mu.Unlock()
		return nil, false, fmt.Errorf("cache miss load expert %d: %w", expertID, err)
	}

	// Evict if at capacity
	for c.order.Len() >= c.capacity {
		c.evictOldest()
	}

	// Insert
	entry := &cacheEntry{expertID: expertID, shard: shard}
	elem := c.order.PushFront(entry)
	c.items[expertID] = elem
	c.misses++
	c.mu.Unlock()

	return shard, false, nil
}

// GetMulti retrieves multiple experts, batching cache lookups.
// Returns shards in the same order as expertIDs.
func (c *ExpertCache) GetMulti(expertIDs []int) ([]*ExpertShard, int, error) {
	shards := make([]*ExpertShard, len(expertIDs))
	cacheHits := 0

	for i, id := range expertIDs {
		shard, hit, err := c.Get(id)
		if err != nil {
			return nil, cacheHits, err
		}
		shards[i] = shard
		if hit {
			cacheHits++
		}
	}

	return shards, cacheHits, nil
}

// Prefetch loads experts into cache without returning them.
// Useful for pre-loading predicted experts based on QuantumRouter.
func (c *ExpertCache) Prefetch(expertIDs []int) int {
	loaded := 0
	for _, id := range expertIDs {
		_, hit, err := c.Get(id)
		if err == nil && !hit {
			loaded++
		}
	}
	return loaded
}

// evictOldest removes the least recently used expert.
// Must be called with c.mu held.
func (c *ExpertCache) evictOldest() {
	oldest := c.order.Back()
	if oldest == nil {
		return
	}
	entry := oldest.Value.(*cacheEntry)
	delete(c.items, entry.expertID)
	c.order.Remove(oldest)
	c.evictions++
}

// Stats returns cache statistics.
func (c *ExpertCache) Stats() (hits, misses, evictions uint64, size, capacity int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.hits, c.misses, c.evictions, c.order.Len(), c.capacity
}

// HitRate returns the cache hit ratio (0.0 to 1.0).
func (c *ExpertCache) HitRate() float64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	total := c.hits + c.misses
	if total == 0 {
		return 0
	}
	return float64(c.hits) / float64(total)
}

// Clear removes all cached experts.
func (c *ExpertCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.items = make(map[int]*list.Element, c.capacity)
	c.order.Init()
	c.hits = 0
	c.misses = 0
	c.evictions = 0
}

// Contains checks if an expert is in the cache without affecting LRU order.
func (c *ExpertCache) Contains(expertID int) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	_, ok := c.items[expertID]
	return ok
}

// CachedExperts returns the list of currently cached expert IDs,
// ordered from most recently used to least.
func (c *ExpertCache) CachedExperts() []int {
	c.mu.Lock()
	defer c.mu.Unlock()
	result := make([]int, 0, c.order.Len())
	for elem := c.order.Front(); elem != nil; elem = elem.Next() {
		result = append(result, elem.Value.(*cacheEntry).expertID)
	}
	return result
}

// MemoryUsageBytes returns approximate memory usage of cached experts.
func (c *ExpertCache) MemoryUsageBytes() int64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	var total int64
	for elem := c.order.Front(); elem != nil; elem = elem.Next() {
		entry := elem.Value.(*cacheEntry)
		if entry.shard != nil {
			total += entry.shard.SizeBytes
		}
	}
	return total
}
