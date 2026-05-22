package cortex

import (
	"testing"
)

func TestSequenceMemoryLearnAndPredict(t *testing.T) {
	sm := NewSequenceMemory(DefaultConfig())

	// Learn: "the cat sat on the mat"
	ids := []uint32{1, 2, 3, 4, 1, 5}
	sm.Learn(ids)

	// Predict next after [1] ("the") — should predict 2 ("cat") or 5 ("mat")
	nextID, ok := sm.Predict([]uint32{1}, nil)
	if !ok {
		t.Fatal("expected prediction after learning, got no prediction")
	}
	if nextID != 2 && nextID != 5 {
		t.Errorf("expected prediction to be 2 or 5, got %d", nextID)
	}
	t.Logf("after context [1]: predicted %d", nextID)
}

func TestSequenceMemoryLongerContextWins(t *testing.T) {
	sm := NewSequenceMemory(DefaultConfig())

	// Learn two sequences that share word 10 but diverge after:
	// Sequence A: 10, 20, 30  (after [10] → 20)
	// Sequence B: 99, 10, 40  (after [99, 10] → 40)
	sm.Learn([]uint32{10, 20, 30})
	sm.Learn([]uint32{99, 10, 40})

	// Short context [10] should predict 20 (from sequence A, bigram match)
	// or 40 (from sequence B, bigram match). Both have window-1 evidence.
	pred1, ok1 := sm.Predict([]uint32{10}, nil)
	if !ok1 {
		t.Fatal("expected prediction for short context [10]")
	}
	t.Logf("short context [10]: predicted %d", pred1)

	// Long context [99, 10] should predict 40 (window-2 match gives ×2 weight)
	pred2, ok2 := sm.Predict([]uint32{99, 10}, nil)
	if !ok2 {
		t.Fatal("expected prediction for long context [99, 10]")
	}
	if pred2 != 40 {
		t.Errorf("expected long context [99, 10] to predict 40 (longer match), got %d", pred2)
	}
	t.Logf("long context [99, 10]: predicted %d", pred2)
}

func TestSequenceMemoryAntiRepetition(t *testing.T) {
	sm := NewSequenceMemory(DefaultConfig())

	// Learn: 1 → 2, 1 → 3
	sm.Learn([]uint32{1, 2})
	sm.Learn([]uint32{1, 3})

	// Without penalty, both 2 and 3 are valid.
	_, ok := sm.Predict([]uint32{1}, nil)
	if !ok {
		t.Fatal("expected prediction without penalty")
	}

	// With heavy penalty on 2, should predict 3.
	recentlyUsed := map[uint32]int{2: 10}
	pred, ok := sm.Predict([]uint32{1}, recentlyUsed)
	if !ok {
		t.Fatal("expected prediction with penalty")
	}
	if pred != 3 {
		t.Errorf("expected 3 (2 is penalized), got %d", pred)
	}
}

func TestSequenceMemoryPersistence(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/seqmem.json"

	sm := NewSequenceMemory(DefaultConfig())
	sm.Learn([]uint32{10, 20, 30, 40})

	// Save.
	if err := sm.Save(path); err != nil {
		t.Fatalf("save failed: %v", err)
	}

	// Load.
	loaded, err := LoadSequenceMemory(path, DefaultConfig())
	if err != nil {
		t.Fatalf("load failed: %v", err)
	}
	if loaded == nil {
		t.Fatal("loaded is nil")
	}

	// Verify prediction still works after load.
	pred, ok := loaded.Predict([]uint32{10}, nil)
	if !ok {
		t.Fatal("expected prediction from loaded memory")
	}
	if pred != 20 {
		t.Errorf("expected 20 from loaded memory, got %d", pred)
	}
}

func TestSequenceMemoryStats(t *testing.T) {
	sm := NewSequenceMemory(DefaultConfig())
	sm.Learn([]uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17})

	stats := sm.Stats()
	if len(stats.WindowSizes) != 5 {
		t.Errorf("expected 5 window sizes, got %d", len(stats.WindowSizes))
	}
	if stats.TotalTransitions == 0 {
		t.Error("expected non-zero total transitions after learning")
	}
	if stats.TotalContexts == 0 {
		t.Error("expected non-zero total contexts after learning")
	}
	t.Logf("stats: %+v", stats)
}

func TestSequenceMemoryEmptyContext(t *testing.T) {
	sm := NewSequenceMemory(DefaultConfig())
	sm.Learn([]uint32{1, 2, 3})

	_, ok := sm.Predict([]uint32{}, nil)
	if ok {
		t.Error("expected no prediction for empty context")
	}

	_, ok = sm.Predict(nil, nil)
	if ok {
		t.Error("expected no prediction for nil context")
	}
}
