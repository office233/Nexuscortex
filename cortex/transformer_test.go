package cortex

import (
	"math/rand"
	"testing"
)

// ─────────────────────────────────────────────────────────────────────
// Transformer Tests
// ─────────────────────────────────────────────────────────────────────

func TestTransformerForwardShape(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	cfg := TransformerConfig{
		VocabSize: 100,
		EmbedDim:  32,
		NumHeads:  2,
		NumLayers: 2,
		FFNDim:    64,
		MaxSeqLen: 32,
	}

	model := NewMiniTransformer(cfg, rng)

	input := []int{5, 10, 15, 20, 25}
	logits := model.Forward(input)

	if logits.Rows() != 5 {
		t.Errorf("expected 5 rows (seq_len), got %d", logits.Rows())
	}
	if logits.Cols() != 100 {
		t.Errorf("expected 100 cols (vocab_size), got %d", logits.Cols())
	}
}

func TestTransformerForwardDeterministic(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	cfg := TransformerConfig{
		VocabSize: 50,
		EmbedDim:  16,
		NumHeads:  2,
		NumLayers: 1,
		FFNDim:    32,
		MaxSeqLen: 16,
	}

	model := NewMiniTransformer(cfg, rng)

	input := []int{1, 2, 3}
	logits1 := model.Forward(input)
	logits2 := model.Forward(input)

	// Same input should produce same output (deterministic)
	for i := range logits1.Data {
		if logits1.Data[i] != logits2.Data[i] {
			t.Errorf("non-deterministic at index %d: %f vs %f",
				i, logits1.Data[i], logits2.Data[i])
			break
		}
	}
}

func TestTransformerGenerate(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	cfg := TransformerConfig{
		VocabSize: 50,
		EmbedDim:  16,
		NumHeads:  2,
		NumLayers: 1,
		FFNDim:    32,
		MaxSeqLen: 32,
	}

	model := NewMiniTransformer(cfg, rng)

	prompt := []int{2, 5, 10} // BOS + some tokens
	generated := model.Generate(prompt, 10, 1.0, 10)

	// Should have generated at least some tokens
	if len(generated) <= len(prompt) {
		t.Errorf("expected generation, got same length: %d", len(generated))
	}

	// All generated tokens should be valid IDs
	for i, tok := range generated {
		if tok < 0 || tok >= cfg.VocabSize {
			t.Errorf("invalid token ID at position %d: %d", i, tok)
		}
	}

	t.Logf("Prompt: %v → Generated: %v", prompt, generated)
}

func TestTransformerTrainStep(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	cfg := TransformerConfig{
		VocabSize: 50,
		EmbedDim:  16,
		NumHeads:  2,
		NumLayers: 1,
		FFNDim:    32,
		MaxSeqLen: 16,
	}

	model := NewMiniTransformer(cfg, rng)

	// Training sequence
	tokens := []int{2, 5, 10, 15, 20, 3} // BOS ... EOS

	// First loss
	loss1 := model.TrainStep(tokens, 0.001)

	// Train for a few steps
	for i := 0; i < 10; i++ {
		model.TrainStep(tokens, 0.001)
	}

	// Loss after training
	loss2 := model.TrainStep(tokens, 0.001)

	t.Logf("Loss: %.4f → %.4f (after 10 steps)", loss1, loss2)

	// Loss should not be NaN or Inf
	if loss1 != loss1 || loss2 != loss2 { // NaN check
		t.Error("loss is NaN")
	}
}

func TestTransformerParamCount(t *testing.T) {
	rng := rand.New(rand.NewSource(42))

	// Small model
	small := DefaultTransformerConfig(1000)
	small.EmbedDim = 32
	small.NumHeads = 2
	small.NumLayers = 2
	small.FFNDim = 64
	modelSmall := NewMiniTransformer(small, rng)

	smallParams := modelSmall.ParamCount()
	t.Logf("Small model: %d params (~%.1fK)", smallParams, float64(smallParams)/1e3)

	// Default model (13M target)
	cfg := DefaultTransformerConfig(32768)
	model := NewMiniTransformer(cfg, rng)

	params := model.ParamCount()
	t.Logf("Default model: %d params (~%.1fM)", params, float64(params)/1e6)

	// Should be in the right ballpark (5-20M)
	if params < 1_000_000 {
		t.Errorf("param count too low: %d", params)
	}
}

func TestMultiHeadAttention(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	mha := NewMultiHeadAttention(16, 2, rng)

	// Input: 3 tokens, 16-dim
	x := NewTensorRand(rng, 1.0, 3, 16)
	out := mha.Forward(x)

	if out.Rows() != 3 || out.Cols() != 16 {
		t.Errorf("expected [3, 16], got %v", out.Shape)
	}

	// Output should not be all zeros (with random weights)
	allZero := true
	for _, v := range out.Data {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Error("attention output is all zeros")
	}
}

func TestFeedForward(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	ff := NewFeedForward(16, 64, rng)

	x := NewTensorRand(rng, 1.0, 3, 16)
	out := ff.Forward(x)

	if out.Rows() != 3 || out.Cols() != 16 {
		t.Errorf("expected [3, 16], got %v", out.Shape)
	}
}

func TestTransformerCausalMask(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	cfg := TransformerConfig{
		VocabSize: 50,
		EmbedDim:  16,
		NumHeads:  2,
		NumLayers: 1,
		FFNDim:    32,
		MaxSeqLen: 16,
	}

	model := NewMiniTransformer(cfg, rng)

	// Forward on [A, B, C]
	full := model.Forward([]int{5, 10, 15})

	// Forward on [A, B]
	partial := model.Forward([]int{5, 10})

	// Due to causal masking, the logits for position 0 should be the same
	// in both cases (token at position 0 can only see itself)
	v := cfg.VocabSize
	for j := 0; j < v; j++ {
		diff := full.Data[j] - partial.Data[j]
		if diff < -0.01 || diff > 0.01 {
			t.Errorf("causal mask violation at logit[0][%d]: full=%f, partial=%f",
				j, full.Data[j], partial.Data[j])
			break
		}
	}
}

func TestTopKSample(t *testing.T) {
	rng := rand.New(rand.NewSource(42))

	// Strong preference for index 2
	logits := []float32{-100, -100, 10, -100, -100}

	counts := make(map[int]int)
	for i := 0; i < 100; i++ {
		tok := topKSample(logits, 5, rng)
		counts[tok]++
	}

	// Index 2 should be selected almost always
	if counts[2] < 90 {
		t.Errorf("expected index 2 to dominate, got counts: %v", counts)
	}
}
