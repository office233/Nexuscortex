package cortex

import (
	"math/rand"
	"testing"
)

// ─────────────────────────────────────────────────────────────────────
// attention_test.go — Tests for the Self-Attention Module
// ─────────────────────────────────────────────────────────────────────

func testAttentionConfig() Config {
	cfg := DefaultConfig()
	cfg.AttentionHistorySize = 10
	cfg.AttentionMinWeight = 64
	cfg.AttentionContextBoost = 100
	cfg.AttentionFrequencyScale = 25
	cfg.DataDir = "testdata/attention"
	return cfg
}

// TestAttentionComputeWeights verifies that bits appearing in context
// history and in the current context SDR receive higher weights.
func TestAttentionComputeWeights(t *testing.T) {
	cfg := testAttentionConfig()
	am := NewAttentionModule(cfg)

	rng := rand.New(rand.NewSource(42))

	// Create an input SDR with specific bits active.
	input := NewSDR(1000)
	input.Set(10)
	input.Set(50)
	input.Set(100)
	input.Set(200)

	// Create context SDRs that share bits 10 and 50.
	for i := 0; i < 5; i++ {
		ctx := RandomSDR(1000, 20, rng)
		ctx.Set(10) // bit 10 always in context
		ctx.Set(50) // bit 50 always in context
		am.ObserveContext(ctx)
	}

	// Current context also has bit 10 and 100 active.
	currentCtx := NewSDR(1000)
	currentCtx.Set(10)
	currentCtx.Set(100)

	weights := am.ComputeWeights(input, currentCtx)

	// Bit 10: 5 × 25 (freq) + 100 (context boost) = 225
	if weights.Weights[10] != 225 {
		t.Errorf("bit 10 weight: got %d, want 225", weights.Weights[10])
	}

	// Bit 50: 5 × 25 (freq) + 0 (not in current context) = 125
	if weights.Weights[50] != 125 {
		t.Errorf("bit 50 weight: got %d, want 125", weights.Weights[50])
	}

	// Bit 100: 0 × 25 (no freq) + 100 (context boost) = 100
	if weights.Weights[100] != 100 {
		t.Errorf("bit 100 weight: got %d, want 100", weights.Weights[100])
	}

	// Bit 200: 0 × 25 (no freq) + 0 (not in context) = 0
	if weights.Weights[200] != 0 {
		t.Errorf("bit 200 weight: got %d, want 0", weights.Weights[200])
	}

	// Verify inactive bits remain zero.
	if weights.Weights[500] != 0 {
		t.Errorf("inactive bit 500 weight: got %d, want 0", weights.Weights[500])
	}
}

// TestAttendSDRPrunes verifies that bits below the threshold are removed.
func TestAttendSDRPrunes(t *testing.T) {
	cfg := testAttentionConfig()
	cfg.AttentionMinWeight = 64
	am := NewAttentionModule(cfg)

	input := NewSDR(1000)
	input.Set(10)
	input.Set(50)
	input.Set(100)
	input.Set(200)

	// Manually create weights: bit 10 = 200 (keep), bit 50 = 100 (keep),
	// bit 100 = 63 (prune, below threshold), bit 200 = 0 (prune).
	weights := AttentionWeights{
		Weights: make([]uint8, 1000),
		Size:    1000,
	}
	weights.Weights[10] = 200
	weights.Weights[50] = 100
	weights.Weights[100] = 63
	weights.Weights[200] = 0

	attended := am.AttendSDR(input, weights)

	// Bits 10, 50 should survive; bits 100, 200 should be pruned.
	if !attended.IsActive(10) {
		t.Error("bit 10 should survive attention (weight=200)")
	}
	if !attended.IsActive(50) {
		t.Error("bit 50 should survive attention (weight=100)")
	}
	if attended.IsActive(100) {
		t.Error("bit 100 should be pruned (weight=63, threshold=64)")
	}
	if attended.IsActive(200) {
		t.Error("bit 200 should be pruned (weight=0)")
	}
	if attended.ActiveCount != 2 {
		t.Errorf("attended active count: got %d, want 2", attended.ActiveCount)
	}
}

// TestAttendSDRThresholdBoundary tests exact threshold boundary behavior.
func TestAttendSDRThresholdBoundary(t *testing.T) {
	cfg := testAttentionConfig()
	cfg.AttentionMinWeight = 64
	am := NewAttentionModule(cfg)

	input := NewSDR(100)
	input.Set(0)
	input.Set(1)

	weights := AttentionWeights{
		Weights: make([]uint8, 100),
		Size:    100,
	}
	weights.Weights[0] = 64 // Exactly at threshold — should survive
	weights.Weights[1] = 63 // Just below — should be pruned

	attended := am.AttendSDR(input, weights)
	if !attended.IsActive(0) {
		t.Error("bit 0 at exact threshold should survive")
	}
	if attended.IsActive(1) {
		t.Error("bit 1 below threshold should be pruned")
	}
}

// TestAttentionHistoryRingBuffer verifies that the ring buffer does not
// grow beyond MaxHistory and uses copy-based wrapping.
func TestAttentionHistoryRingBuffer(t *testing.T) {
	cfg := testAttentionConfig()
	cfg.AttentionHistorySize = 4
	am := NewAttentionModule(cfg)

	rng := rand.New(rand.NewSource(99))

	// Add more entries than capacity.
	for i := 0; i < 10; i++ {
		sdr := RandomSDR(100, 5, rng)
		am.ObserveContext(sdr)
	}

	// Ring buffer backing array should never exceed capacity.
	if len(am.ContextHistory) != 4 {
		t.Errorf("history length: got %d, want 4", len(am.ContextHistory))
	}
	if am.count != 4 {
		t.Errorf("count: got %d, want 4 (capped at MaxHistory)", am.count)
	}
	if am.MaxHistory != 4 {
		t.Errorf("MaxHistory: got %d, want 4", am.MaxHistory)
	}

	// Verify all entries are valid SDRs (not nil/empty).
	for i := 0; i < am.count; i++ {
		if am.ContextHistory[i].Size == 0 {
			t.Errorf("history slot %d has zero Size", i)
		}
	}
}

// TestAttentionEmptyInputs tests edge cases with empty SDRs.
func TestAttentionEmptyInputs(t *testing.T) {
	cfg := testAttentionConfig()
	am := NewAttentionModule(cfg)

	// Empty input SDR.
	emptyInput := NewSDR(1000)
	emptyCtx := NewSDR(1000)

	weights := am.ComputeWeights(emptyInput, emptyCtx)
	if len(weights.Weights) != 1000 {
		t.Errorf("weights size: got %d, want 1000", len(weights.Weights))
	}
	// All weights should be 0.
	for i, w := range weights.Weights {
		if w != 0 {
			t.Errorf("weight[%d] = %d, want 0 for empty input", i, w)
			break
		}
	}

	// Attend on empty should produce empty.
	attended := am.AttendSDR(emptyInput, weights)
	if attended.ActiveCount != 0 {
		t.Errorf("attended empty input: got %d active, want 0", attended.ActiveCount)
	}

	// Non-empty input with empty context and no history.
	input := NewSDR(1000)
	input.Set(10)
	input.Set(20)

	weights2 := am.ComputeWeights(input, emptyCtx)
	// With no history and no context, all weights should be 0.
	if weights2.Weights[10] != 0 {
		t.Errorf("bit 10 with no context/history: got %d, want 0", weights2.Weights[10])
	}

	// AttendSDR should prune everything (all weights below threshold).
	attended2 := am.AttendSDR(input, weights2)
	if attended2.ActiveCount != 0 {
		t.Errorf("attended with zero weights: got %d active, want 0", attended2.ActiveCount)
	}
}

// TestAttentionComputeWeightsClamp verifies clamping to 255.
func TestAttentionComputeWeightsClamp(t *testing.T) {
	cfg := testAttentionConfig()
	cfg.AttentionHistorySize = 20
	cfg.AttentionFrequencyScale = 25
	cfg.AttentionContextBoost = 100
	am := NewAttentionModule(cfg)

	// Fill 15 history entries with bit 5 active.
	for i := 0; i < 15; i++ {
		ctx := NewSDR(100)
		ctx.Set(5)
		am.ObserveContext(ctx)
	}

	input := NewSDR(100)
	input.Set(5)

	// Current context also has bit 5.
	currentCtx := NewSDR(100)
	currentCtx.Set(5)

	weights := am.ComputeWeights(input, currentCtx)
	// 15 × 25 + 100 = 475, clamped to 255.
	if weights.Weights[5] != 255 {
		t.Errorf("bit 5 should be clamped to 255, got %d", weights.Weights[5])
	}
}

// TestAttentionClonesSDRs verifies that modifying the original SDR
// after ObserveContext does not corrupt the stored history.
func TestAttentionClonesSDRs(t *testing.T) {
	cfg := testAttentionConfig()
	am := NewAttentionModule(cfg)

	ctx := NewSDR(100)
	ctx.Set(10)
	ctx.Set(20)
	am.ObserveContext(ctx)

	// Mutate the original — history should be unaffected.
	ctx.Clear(10)
	ctx.Set(30)

	// Check the stored entry still has bit 10.
	stored := am.ContextHistory[0]
	if !stored.IsActive(10) {
		t.Error("stored history should still have bit 10 (clone should prevent aliasing)")
	}
	if stored.IsActive(30) {
		t.Error("stored history should NOT have bit 30 (mutation leaked)")
	}
}
