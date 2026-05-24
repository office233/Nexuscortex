package cortex

import (
	"math/rand"
	"testing"
)

// BenchmarkTransformerForward measures one full forward pass on a
// realistic transformer shape (matches the ~5M-param model used by
// cortex-broca-train). Compare CPU walltime before/after tensor
// parallelism changes:
//
//	go test -bench=BenchmarkTransformerForward -benchtime=10x -run=^$ ./cortex
//
// The benchmark seeds RNG deterministically so the comparison is
// apples-to-apples across runs.
func BenchmarkTransformerForward(b *testing.B) {
	cfg := TransformerConfig{
		VocabSize:  8192,
		EmbedDim:   128,
		NumHeads:   4,
		NumLayers:  6,
		FFNDim:     512,
		MaxSeqLen:  64,
		EOSTokenID: -1,
	}
	rng := rand.New(rand.NewSource(1))
	m := NewMiniTransformer(cfg, rng)

	// 32-token sequence – typical of GenerateFast prefill on prompts of
	// a few words after BPE encoding.
	tokens := make([]int, 32)
	for i := range tokens {
		tokens[i] = (i*7 + 13) % cfg.VocabSize
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = m.Forward(tokens)
	}
}

// BenchmarkTransformerGenerateFast covers the path Broca 2.0 actually
// uses for inference: prefill + 16 cached steps. This is what the
// eval loop spends its time in.
func BenchmarkTransformerGenerateFast(b *testing.B) {
	cfg := TransformerConfig{
		VocabSize:  8192,
		EmbedDim:   128,
		NumHeads:   4,
		NumLayers:  6,
		FFNDim:     512,
		MaxSeqLen:  64,
		EOSTokenID: -1,
	}
	rng := rand.New(rand.NewSource(1))
	m := NewMiniTransformer(cfg, rng)

	prompt := make([]int, 20)
	for i := range prompt {
		prompt[i] = (i*5 + 3) % cfg.VocabSize
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Rng = rand.New(rand.NewSource(int64(i)))
		_ = m.GenerateFast(prompt, 16, 1.0, 8)
	}
}

// BenchmarkTransformerTrainStep measures one training step (forward +
// backward + update). This is what cortex-broca-train spends its time
// in; ~14 steps/min before parallelism is what we are trying to lift.
func BenchmarkTransformerTrainStep(b *testing.B) {
	cfg := TransformerConfig{
		VocabSize:  8192,
		EmbedDim:   128,
		NumHeads:   4,
		NumLayers:  6,
		FFNDim:     512,
		MaxSeqLen:  64,
		EOSTokenID: -1,
	}
	rng := rand.New(rand.NewSource(1))
	m := NewMiniTransformer(cfg, rng)

	tokens := make([]int, 32)
	for i := range tokens {
		tokens[i] = (i*7 + 13) % cfg.VocabSize
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = m.TrainStep(tokens, 0.0001)
	}
}
