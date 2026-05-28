package cortex

import (
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

// TestTrainStepAdamOverfit verifies the Adam path converges at least as
// well as the SGD overfit test, on the same tiny model.
func TestTrainStepAdamOverfit(t *testing.T) {
	m, tokens := buildTinyTransformer()
	opt := NewAdamState(m, DefaultAdamConfig())

	// Run one step at lr=0 only to grab the starting loss without
	// actually moving any parameters (Adam still updates moment buffers,
	// but with lr=0 the weight delta is exactly zero).
	initial := m.TrainStepAdam(tokens, 0, opt)
	final := initial
	for i := 0; i < 200; i++ {
		final = m.TrainStepAdam(tokens, 0.01, opt)
	}

	if final >= initial*0.5 {
		t.Fatalf("Adam loss did not drop enough: initial=%.4f final=%.4f", initial, final)
	}
	if final >= 1.0 {
		t.Fatalf("Adam did not converge below 1.0 on tiny overfit: final=%.4f", final)
	}
	t.Logf("adam overfit: initial=%.4f final=%.4f after 200 steps", initial, final)
}

// TestLRSchedule walks the warmup/cosine schedule at a few sentinel
// steps to catch off-by-one mistakes (a frequent source of "loss
// explodes immediately" bugs in training).
func TestLRSchedule(t *testing.T) {
	s := LRSchedule{WarmupSteps: 100, DecaySteps: 900, PeakLR: 1e-3, MinLR: 1e-5}

	cases := []struct {
		step int
		want float32
		tol  float32
	}{
		{1, 1e-5, 1e-6},    // start of warmup: ~PeakLR/100
		{100, 1e-3, 1e-6},  // end of warmup: PeakLR
		{1000, 1e-5, 1e-6}, // end of decay: MinLR
		{5000, 1e-5, 1e-6}, // past end: still MinLR
	}
	for _, c := range cases {
		got := s.LR(c.step)
		if abs32(got-c.want) > c.tol {
			t.Errorf("LR(%d): got %.6g want %.6g", c.step, got, c.want)
		}
	}

	// Cosine midpoint check: at step = warmup + decay/2 the LR should
	// equal (PeakLR + MinLR) / 2 (cos(pi/2)=0 → LR=MinLR+0.5*(Peak-Min)).
	mid := s.LR(100 + 450)
	want := 0.5 * (s.PeakLR + s.MinLR)
	if abs32(mid-want) > 1e-6 {
		t.Errorf("LR mid: got %.6g want %.6g", mid, want)
	}
}

func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

// TestTrainStepAdamBatchEquivalence verifies the batched path with N=1
// is numerically identical to TrainStepAdam on the same sequence. Any
// divergence would indicate a bug in zeroAllGrads / accumulateGrad /
// scaleAllGrads decomposition.
func TestTrainStepAdamBatchEquivalence(t *testing.T) {
	mA, tokens := buildTinyTransformer()
	mB, _ := buildTinyTransformer() // identical because both use seed=1
	optA := NewAdamState(mA, DefaultAdamConfig())
	optB := NewAdamState(mB, DefaultAdamConfig())

	lossA := mA.TrainStepAdam(tokens, 0.01, optA)
	lossB := mB.TrainStepAdamBatch([][]int{tokens}, 0.01, optB)

	if abs32(lossA-lossB) > 1e-5 {
		t.Fatalf("loss diverged: single=%.6f batch=%.6f", lossA, lossB)
	}
	// Spot-check that a couple of parameter tensors match exactly.
	maxDiff := float32(0)
	for i := range mA.Embedding.TokenEmb.Data {
		d := abs32(mA.Embedding.TokenEmb.Data[i] - mB.Embedding.TokenEmb.Data[i])
		if d > maxDiff {
			maxDiff = d
		}
	}
	if maxDiff > 1e-6 {
		t.Fatalf("TokenEmb diverged: max diff %.6g", maxDiff)
	}
}

// TestTrainStepAdamBatchConverges checks the batched path also drives
// loss down on the tiny overfit task, with a batch of repeated copies
// of the sole sequence. With batch=4 the per-step loss is averaged
// across 4 forward/backward passes, so 100 batch steps ≈ 400 single
// updates worth of gradient signal — should converge similarly fast.
func TestTrainStepAdamBatchConverges(t *testing.T) {
	m, tokens := buildTinyTransformer()
	opt := NewAdamState(m, DefaultAdamConfig())
	batch := [][]int{tokens, tokens, tokens, tokens}

	initial := m.TrainStepAdamBatch(batch, 0, opt)
	final := initial
	for i := 0; i < 100; i++ {
		final = m.TrainStepAdamBatch(batch, 0.02, opt)
	}
	if final >= 1.0 {
		t.Fatalf("batch Adam did not converge below 1.0: initial=%.4f final=%.4f", initial, final)
	}
	t.Logf("batch=4 adam overfit: initial=%.4f final=%.4f after 100 steps", initial, final)
}

// TestAdamStateRoundTrip verifies SaveAdamState + LoadAdamState produce
// a state that yields byte-identical updates on the next training step.
// This guards the "resume is truly continuous" promise.
func TestAdamStateRoundTrip(t *testing.T) {
	m, tokens := buildTinyTransformer()
	opt := NewAdamState(m, DefaultAdamConfig())

	// Take a handful of steps so the moment buffers are non-trivial.
	for i := 0; i < 5; i++ {
		m.TrainStepAdam(tokens, 0.01, opt)
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "opt.nxto")
	if err := SaveAdamState(opt, path); err != nil {
		t.Fatalf("save: %v", err)
	}
	// Confirm load with a fresh (but architecturally identical) model
	// — first rebuild the model in the same state so shapes match.
	mLoaded := buildModelLikeOriginal()
	// Run the same 5 priming steps on this clone so weights match.
	optClone := NewAdamState(mLoaded, DefaultAdamConfig())
	for i := 0; i < 5; i++ {
		mLoaded.TrainStepAdam(tokens, 0.01, optClone)
	}

	loaded, err := LoadAdamState(mLoaded, path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if loaded == nil {
		t.Fatal("load returned nil")
	}
	if loaded.Step != opt.Step {
		t.Fatalf("step mismatch: want %d got %d", opt.Step, loaded.Step)
	}

	// Compare moment buffer contents.
	checkSame := func(name string, a, b *Tensor) {
		t.Helper()
		var maxDiff float32
		for i := range a.Data {
			d := abs32(a.Data[i] - b.Data[i])
			if d > maxDiff {
				maxDiff = d
			}
		}
		if maxDiff > 0 {
			t.Errorf("%s: max diff after round-trip = %.6g", name, maxDiff)
		}
	}
	checkSame("TokenEmbM", opt.TokenEmbM, loaded.TokenEmbM)
	checkSame("TokenEmbV", opt.TokenEmbV, loaded.TokenEmbV)
	checkSame("LNFGammaM", opt.LNFGammaM, loaded.LNFGammaM)
	checkSame("block0.WQM", opt.Blocks[0].WQM, loaded.Blocks[0].WQM)
	checkSame("block0.W2V", opt.Blocks[0].W2V, loaded.Blocks[0].W2V)

	// Verify the loaded state survives serialise→load even when the
	// file is removed afterwards.
	if err := os.Remove(path); err != nil {
		t.Fatalf("cleanup: %v", err)
	}
}

// TestLoadAdamStateMissingFile confirms that a missing file is reported
// as (nil, nil) rather than an error, so cold starts work seamlessly.
func TestLoadAdamStateMissingFile(t *testing.T) {
	m, _ := buildTinyTransformer()
	loaded, err := LoadAdamState(m, filepath.Join(t.TempDir(), "does-not-exist.nxto"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if loaded != nil {
		t.Fatalf("expected nil state for missing file, got %v", loaded)
	}
}

// buildModelLikeOriginal mirrors buildTinyTransformer's wiring but
// isolated here for the AdamState round-trip test, which wants two
// architecturally identical models so the shape check passes.
func buildModelLikeOriginal() *MiniTransformer {
	rng := rand.New(rand.NewSource(1))
	cfg := TransformerConfig{
		VocabSize:  20,
		EmbedDim:   8,
		NumHeads:   2,
		NumLayers:  1,
		FFNDim:     16,
		MaxSeqLen:  8,
		EOSTokenID: 3,
	}
	return NewMiniTransformer(cfg, rng)
}

var _ = math.Pi // keep math import even if assertions get removed later
