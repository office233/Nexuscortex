package cortex

import (
	"math/rand"
	"testing"
)

func TestWorkingMemoryStoreAndRecall(t *testing.T) {
	cfg := DefaultConfig()
	cfg.WorkingMemoryCapacity = 4
	wm := NewWorkingMemory(cfg)

	rng := rand.New(rand.NewSource(42))
	sdr1 := RandomSDR(1000, 50, rng)
	sdr2 := RandomSDR(1000, 50, rng)
	sdr3 := RandomSDR(1000, 50, rng)

	// Store 3 items.
	wm.Store(sdr1, "hello world", 200)
	wm.Store(sdr2, "how are you", 150)
	wm.Store(sdr3, "goodbye", 100)

	if wm.ActiveCount() != 3 {
		t.Fatalf("expected 3 active slots, got %d", wm.ActiveCount())
	}

	// Recall should find exact match.
	text, sim := wm.Recall(sdr1)
	if text != "hello world" {
		t.Errorf("expected 'hello world', got %q", text)
	}
	if sim < 200 {
		t.Errorf("expected high similarity, got %d", sim)
	}
}

func TestWorkingMemoryEviction(t *testing.T) {
	cfg := DefaultConfig()
	cfg.WorkingMemoryCapacity = 2
	wm := NewWorkingMemory(cfg)

	rng := rand.New(rand.NewSource(99))
	sdr1 := RandomSDR(1000, 50, rng)
	sdr2 := RandomSDR(1000, 50, rng)
	sdr3 := RandomSDR(1000, 50, rng)

	wm.Store(sdr1, "first", 100)
	wm.Store(sdr2, "second", 200)
	wm.Store(sdr3, "third", 150) // Should evict "first" (lowest relevance)

	if wm.ActiveCount() != 2 {
		t.Fatalf("expected 2 active slots, got %d", wm.ActiveCount())
	}

	// "first" should be gone.
	text, _ := wm.Recall(sdr1)
	if text == "first" {
		t.Error("expected 'first' to be evicted")
	}

	// "second" should still be there.
	text, _ = wm.Recall(sdr2)
	if text != "second" {
		t.Errorf("expected 'second', got %q", text)
	}

	if wm.TotalEvicts != 1 {
		t.Errorf("expected 1 eviction, got %d", wm.TotalEvicts)
	}
}

func TestWorkingMemoryDecay(t *testing.T) {
	cfg := DefaultConfig()
	cfg.WorkingMemoryCapacity = 4
	cfg.WorkingMemoryDecayRate = 50
	cfg.WorkingMemoryMinRelevance = 10
	wm := NewWorkingMemory(cfg)

	rng := rand.New(rand.NewSource(77))
	sdr := RandomSDR(1000, 50, rng)
	wm.Store(sdr, "ephemeral", 60)

	// After 2 ticks: relevance 60 - 50 - 50 = 0 → auto-clear.
	wm.Tick()
	if wm.ActiveCount() != 1 {
		t.Fatalf("slot should survive first tick, got active=%d", wm.ActiveCount())
	}
	wm.Tick()
	if wm.ActiveCount() != 0 {
		t.Fatalf("slot should be cleared after decay, got active=%d", wm.ActiveCount())
	}
}

func TestWorkingMemoryBlendContext(t *testing.T) {
	cfg := DefaultConfig()
	wm := NewWorkingMemory(cfg)

	rng := rand.New(rand.NewSource(55))
	sdr1 := RandomSDR(1000, 50, rng)
	sdr2 := RandomSDR(1000, 50, rng)

	wm.Store(sdr1, "context a", 200)
	wm.Store(sdr2, "context b", 200)

	blended := wm.BlendContext(1000)
	// Blended should have bits from both SDRs.
	if blended.ActiveCount < 50 {
		t.Errorf("blended context too sparse: %d active bits", blended.ActiveCount)
	}
}

func TestWorkingMemoryRefresh(t *testing.T) {
	cfg := DefaultConfig()
	wm := NewWorkingMemory(cfg)

	rng := rand.New(rand.NewSource(33))
	sdr := RandomSDR(1000, 50, rng)

	wm.Store(sdr, "original", 100)
	// Storing the SAME pattern again should refresh, not duplicate.
	wm.Store(sdr, "updated", 100)

	if wm.ActiveCount() != 1 {
		t.Errorf("expected 1 slot (refreshed), got %d", wm.ActiveCount())
	}

	text, _ := wm.Recall(sdr)
	if text != "updated" {
		t.Errorf("expected 'updated' after refresh, got %q", text)
	}
}

func TestWorkingMemoryActiveTexts(t *testing.T) {
	cfg := DefaultConfig()
	wm := NewWorkingMemory(cfg)

	rng := rand.New(rand.NewSource(11))
	wm.Store(RandomSDR(1000, 50, rng), "low", 50)
	wm.Store(RandomSDR(1000, 50, rng), "high", 250)
	wm.Store(RandomSDR(1000, 50, rng), "mid", 150)

	texts := wm.ActiveTexts()
	if len(texts) != 3 {
		t.Fatalf("expected 3 texts, got %d", len(texts))
	}
	// Should be sorted by relevance: high, mid, low.
	if texts[0] != "high" {
		t.Errorf("expected 'high' first, got %q", texts[0])
	}
	if texts[1] != "mid" {
		t.Errorf("expected 'mid' second, got %q", texts[1])
	}
}

func TestWorkingMemoryEmptyPattern(t *testing.T) {
	cfg := DefaultConfig()
	wm := NewWorkingMemory(cfg)

	empty := NewSDR(1000) // Zero active bits
	wm.Store(empty, "should not store", 200)

	if wm.ActiveCount() != 0 {
		t.Error("should not store empty SDR")
	}
}
