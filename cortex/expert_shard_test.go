package cortex

import (
	"os"
	"path/filepath"
	"testing"
)

func TestTileCodebook(t *testing.T) {
	cb := DefaultCodebook()

	// Pattern 0 should be all zeros
	if cb.Patterns[0] != 0 {
		t.Errorf("pattern 0 should be all zeros, got %08X", uint32(cb.Patterns[0]))
	}

	// Pattern 1 should be single positive weight at position 0
	weights := cb.Decode(1).Unpack()
	if weights[0] != 1 {
		t.Errorf("pattern 1, weight 0: expected 1, got %d", weights[0])
	}
	activeCount := 0
	for _, w := range weights {
		if w != 0 {
			activeCount++
		}
	}
	if activeCount != 1 {
		t.Errorf("pattern 1: expected 1 active weight, got %d", activeCount)
	}

	// Pattern 17 should be single negative weight at position 0
	weights = cb.Decode(17).Unpack()
	if weights[0] != -1 {
		t.Errorf("pattern 17, weight 0: expected -1, got %d", weights[0])
	}

	// All 256 patterns should be unique (or at least decodable)
	for i := 0; i < 256; i++ {
		_ = cb.Decode(uint8(i)) // should not panic
	}

	t.Logf("Codebook: 256 patterns verified")
}

func TestShardedModelCreateAndLoad(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test_model.nxs5")

	// Create 4 small experts
	experts := make([]*TernaryLayer, 4)
	for i := range experts {
		experts[i] = NewTernaryLayer(64, 32)
		// Fill with different patterns per expert
		for j := range experts[i].Tiles {
			experts[i].Tiles[j] = TernaryTile(uint32(i*1000 + j))
		}
	}

	// Create sharded model
	idx, err := CreateShardedModel(path, experts, nil)
	if err != nil {
		t.Fatalf("CreateShardedModel: %v", err)
	}

	if idx.ExpertCount() != 4 {
		t.Errorf("expert count: expected 4, got %d", idx.ExpertCount())
	}

	// Verify file exists
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat model file: %v", err)
	}
	t.Logf("Model file size: %d bytes", info.Size())

	// Load index
	loaded, err := LoadShardedModelIndex(path)
	if err != nil {
		t.Fatalf("LoadShardedModelIndex: %v", err)
	}

	if loaded.ExpertCount() != 4 {
		t.Errorf("loaded expert count: expected 4, got %d", loaded.ExpertCount())
	}

	// Load each expert and verify
	for i := 0; i < 4; i++ {
		shard, err := loaded.LoadExpert(i)
		if err != nil {
			t.Fatalf("LoadExpert(%d): %v", i, err)
		}
		if shard.Header.ExpertID != uint32(i) {
			t.Errorf("expert %d: expected ID %d, got %d", i, i, shard.Header.ExpertID)
		}
		if shard.Layer == nil {
			t.Fatalf("expert %d: layer is nil", i)
		}
		// Verify first tile matches what we wrote
		expected := TernaryTile(uint32(i * 1000))
		if shard.Layer.Tiles[0] != expected {
			t.Errorf("expert %d, tile 0: expected %08X, got %08X",
				i, uint32(expected), uint32(shard.Layer.Tiles[0]))
		}
	}
}

func TestShardedModelWithCodebook(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test_codebook.nxs5")

	experts := make([]*TernaryLayer, 2)
	for i := range experts {
		experts[i] = NewTernaryLayer(32, 16)
	}

	cb := DefaultCodebook()
	idx, err := CreateShardedModel(path, experts, cb)
	if err != nil {
		t.Fatalf("CreateShardedModel with codebook: %v", err)
	}

	if idx.Codebook == nil {
		t.Fatal("codebook should be saved")
	}

	// Reload and verify codebook
	loaded, err := LoadShardedModelIndex(path)
	if err != nil {
		t.Fatalf("LoadShardedModelIndex: %v", err)
	}
	if loaded.Codebook == nil {
		t.Fatal("loaded codebook should not be nil")
	}

	// Verify codebook patterns match
	for i := 0; i < 256; i++ {
		if loaded.Codebook.Patterns[i] != cb.Patterns[i] {
			t.Errorf("codebook pattern %d: expected %08X, got %08X",
				i, uint32(cb.Patterns[i]), uint32(loaded.Codebook.Patterns[i]))
		}
	}
}

func TestExpertCache(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "cache_test.nxs5")

	// Create 10 experts
	experts := make([]*TernaryLayer, 10)
	for i := range experts {
		experts[i] = NewTernaryLayer(64, 32)
		for j := range experts[i].Tiles {
			experts[i].Tiles[j] = TernaryTile(uint32(i*100 + j))
		}
	}

	_, err := CreateShardedModel(path, experts, nil)
	if err != nil {
		t.Fatalf("CreateShardedModel: %v", err)
	}

	idx, err := LoadShardedModelIndex(path)
	if err != nil {
		t.Fatalf("LoadShardedModelIndex: %v", err)
	}

	// Create cache with capacity 3
	cache := NewExpertCache(3, idx)

	// First access = miss
	shard, hit, err := cache.Get(0)
	if err != nil {
		t.Fatalf("Get(0): %v", err)
	}
	if hit {
		t.Error("first access should be a miss")
	}
	if shard == nil {
		t.Fatal("shard should not be nil")
	}

	// Second access to same expert = hit
	_, hit, err = cache.Get(0)
	if err != nil {
		t.Fatalf("Get(0) again: %v", err)
	}
	if !hit {
		t.Error("second access should be a hit")
	}

	// Load 3 more experts to trigger eviction
	cache.Get(1) // miss
	cache.Get(2) // miss
	cache.Get(3) // miss, should evict expert 0

	if cache.Contains(0) {
		t.Error("expert 0 should have been evicted (capacity=3)")
	}
	if !cache.Contains(3) {
		t.Error("expert 3 should be in cache")
	}

	// Verify stats
	hits, misses, evictions, size, capacity := cache.Stats()
	t.Logf("Cache stats: hits=%d misses=%d evictions=%d size=%d capacity=%d",
		hits, misses, evictions, size, capacity)

	if hits != 1 {
		t.Errorf("expected 1 hit, got %d", hits)
	}
	if misses != 4 {
		t.Errorf("expected 4 misses, got %d", misses)
	}
	if evictions != 1 {
		t.Errorf("expected 1 eviction, got %d", evictions)
	}
	if size != 3 {
		t.Errorf("expected size 3, got %d", size)
	}
}

func TestExpertCacheGetMulti(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "multi_test.nxs5")

	experts := make([]*TernaryLayer, 5)
	for i := range experts {
		experts[i] = NewTernaryLayer(32, 16)
	}

	_, err := CreateShardedModel(path, experts, nil)
	if err != nil {
		t.Fatalf("CreateShardedModel: %v", err)
	}

	idx, err := LoadShardedModelIndex(path)
	if err != nil {
		t.Fatalf("LoadShardedModelIndex: %v", err)
	}

	cache := NewExpertCache(10, idx)

	// Pre-load expert 1
	cache.Get(1)

	// Get multi: 0 (miss), 1 (hit), 2 (miss)
	shards, hits, err := cache.GetMulti([]int{0, 1, 2})
	if err != nil {
		t.Fatalf("GetMulti: %v", err)
	}
	if len(shards) != 3 {
		t.Fatalf("expected 3 shards, got %d", len(shards))
	}
	if hits != 1 {
		t.Errorf("expected 1 cache hit, got %d", hits)
	}

	t.Logf("GetMulti: %d shards, %d cache hits, hit rate: %.1f%%",
		len(shards), hits, cache.HitRate()*100)
}

func TestExpertCachePrefetch(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "prefetch_test.nxs5")

	experts := make([]*TernaryLayer, 5)
	for i := range experts {
		experts[i] = NewTernaryLayer(32, 16)
	}

	_, err := CreateShardedModel(path, experts, nil)
	if err != nil {
		t.Fatalf("CreateShardedModel: %v", err)
	}

	idx, err := LoadShardedModelIndex(path)
	if err != nil {
		t.Fatalf("LoadShardedModelIndex: %v", err)
	}

	cache := NewExpertCache(10, idx)

	// Prefetch 3 experts
	loaded := cache.Prefetch([]int{0, 1, 2})
	if loaded != 3 {
		t.Errorf("expected 3 loaded, got %d", loaded)
	}

	// All should now be hits
	for _, id := range []int{0, 1, 2} {
		_, hit, _ := cache.Get(id)
		if !hit {
			t.Errorf("expert %d should be a hit after prefetch", id)
		}
	}
}

func TestExpertCacheLRUOrder(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "lru_test.nxs5")

	experts := make([]*TernaryLayer, 5)
	for i := range experts {
		experts[i] = NewTernaryLayer(32, 16)
	}

	_, err := CreateShardedModel(path, experts, nil)
	if err != nil {
		t.Fatalf("CreateShardedModel: %v", err)
	}

	idx, err := LoadShardedModelIndex(path)
	if err != nil {
		t.Fatalf("LoadShardedModelIndex: %v", err)
	}

	cache := NewExpertCache(3, idx)

	// Load 0, 1, 2
	cache.Get(0)
	cache.Get(1)
	cache.Get(2)

	// Access 0 again (moves to front)
	cache.Get(0)

	// Load 3 — should evict 1 (oldest unused), not 0
	cache.Get(3)

	if !cache.Contains(0) {
		t.Error("expert 0 should still be in cache (was recently accessed)")
	}
	if cache.Contains(1) {
		t.Error("expert 1 should have been evicted (LRU)")
	}
	if !cache.Contains(2) {
		t.Error("expert 2 should still be in cache")
	}
	if !cache.Contains(3) {
		t.Error("expert 3 should be in cache")
	}

	order := cache.CachedExperts()
	t.Logf("LRU order (MRU→LRU): %v", order)
	if len(order) != 3 {
		t.Errorf("expected 3 cached, got %d", len(order))
	}
	if order[0] != 3 {
		t.Errorf("MRU should be expert 3, got %d", order[0])
	}
}
