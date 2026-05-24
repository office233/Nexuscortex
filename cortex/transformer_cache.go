package cortex

// transformer_cache.go — KV-cache based fast autoregressive generation.
//
// Background. The training-time MultiHeadAttention.Forward recomputes Q,K,V
// from scratch every call. During greedy/sampled generation the prompt
// prefix never changes, so re-projecting old tokens every step is pure
// waste: N tokens × N steps = O(N^2) full forwards. With KV-cache we
// project new tokens once, append to per-layer K and V buffers, and
// attention becomes O(N) per step.
//
// Design constraints:
//  - Do NOT change the training path. Forward(), TrainStep(), Generate()
//    keep their semantics. New code is additive only.
//  - Mirror the math exactly. The cached path uses the same projections,
//    same scale, same softmax, same residual; output should be
//    numerically equivalent (within float32 noise) for the same prompt.
//  - Skip backward bookkeeping. Cached path is inference-only; setting
//    lastInput / lastQ etc. is unnecessary and would corrupt a real
//    training step that ran later. We DO NOT touch those fields.

import (
	"math"
	"math/rand"
)

// KVCache stores accumulated keys and values for one MultiHeadAttention
// layer across generation steps. K and V grow by one row per token
// emitted; we keep them flat as [seqLenSoFar, embedDim] tensors so the
// existing MatMulTransposed and MatMul helpers can be reused without
// reshaping per head.
type KVCache struct {
	K *Tensor // [seqLen, embedDim], grows over time
	V *Tensor // [seqLen, embedDim], grows over time
}

// appendRow returns a new tensor with the given [embedDim] row appended
// as the last row. We allocate fresh on each append rather than growing
// in place because Tensor.Data is a flat slice and we want predictable
// shape semantics. Concatenations are O(N*embedDim) which is dominated
// by attention itself, so this stays cheap.
func appendRow(existing *Tensor, row *Tensor, embedDim int) *Tensor {
	if existing == nil {
		out := NewTensor(1, embedDim)
		copy(out.Data, row.Data)
		return out
	}
	oldSeq := existing.Shape[0]
	out := NewTensor(oldSeq+1, embedDim)
	copy(out.Data, existing.Data)
	copy(out.Data[oldSeq*embedDim:], row.Data)
	return out
}

// ─────────────────────────────────────────────────────────────────────
// MultiHeadAttention — cached single-token step
// ─────────────────────────────────────────────────────────────────────

// ForwardCachedStep runs attention on a single new token using a growing
// KV cache. Input x is [1, embedDim] (the new token's normalised
// embedding). Returns [1, embedDim] (attention output before residual)
// and updates cache.K / cache.V in place.
//
// Causal masking is implicit: the query attends to all rows currently in
// cache, which are exactly the past tokens plus the new one. We always
// append before computing attention so the new token can attend to
// itself, mirroring the diagonal of the causal mask in training.
func (mha *MultiHeadAttention) ForwardCachedStep(x *Tensor, cache *KVCache) *Tensor {
	if x.Shape[0] != 1 {
		panic("ForwardCachedStep expects a single-token input")
	}
	embedDim := x.Shape[1]
	numHeads := mha.NumHeads
	headDim := mha.HeadDim

	// Project the new token: Q, K_new, V_new each [1, embedDim].
	Q := x.MatMul(mha.WQ).Add(mha.BQ)
	Knew := x.MatMul(mha.WK).Add(mha.BK)
	Vnew := x.MatMul(mha.WV).Add(mha.BV)

	// Append to cache so future tokens can attend back. The new query
	// will read from cache.K / cache.V right after.
	cache.K = appendRow(cache.K, Knew, embedDim)
	cache.V = appendRow(cache.V, Vnew, embedDim)

	seqLen := cache.K.Shape[0]
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	attnOut := NewTensor(1, embedDim)

	// Per-head attention. With a single-row query the attention reduces
	// to a vector × matrix^T softmax × matrix, no masking needed because
	// the new token is the last row of cache (everything is "past").
	for h := 0; h < numHeads; h++ {
		hStart := h * headDim

		// Extract this head's Q row [1, headDim].
		Qh := make([]float32, headDim)
		for j := 0; j < headDim; j++ {
			Qh[j] = Q.Data[hStart+j]
		}

		// Compute scores Qh · Kh^T → [seqLen]
		scores := make([]float32, seqLen)
		for t := 0; t < seqLen; t++ {
			kOff := t*embedDim + hStart
			s := float32(0)
			for j := 0; j < headDim; j++ {
				s += Qh[j] * cache.K.Data[kOff+j]
			}
			scores[t] = s * scale
		}

		// Softmax over scores.
		maxScore := scores[0]
		for _, s := range scores {
			if s > maxScore {
				maxScore = s
			}
		}
		expSum := float32(0)
		for i, s := range scores {
			scores[i] = float32(math.Exp(float64(s - maxScore)))
			expSum += scores[i]
		}
		if expSum == 0 {
			expSum = 1
		}
		for i := range scores {
			scores[i] /= expSum
		}

		// Weighted sum: attn_h = scores · Vh → [headDim]
		for j := 0; j < headDim; j++ {
			acc := float32(0)
			for t := 0; t < seqLen; t++ {
				acc += scores[t] * cache.V.Data[t*embedDim+hStart+j]
			}
			attnOut.Data[hStart+j] = acc
		}
	}

	// Output projection.
	return attnOut.MatMul(mha.WO).Add(mha.BO)
}

// ─────────────────────────────────────────────────────────────────────
// TransformerBlock — cached single-token step
// ─────────────────────────────────────────────────────────────────────

// ForwardCachedStep mirrors TransformerBlock.Forward semantics for a
// single new token, given the per-layer KV cache.
func (tb *TransformerBlock) ForwardCachedStep(x *Tensor, cache *KVCache) *Tensor {
	normed1 := x.LayerNorm(tb.LN1Gamma, tb.LN1Beta)
	attnOut := tb.Attn.ForwardCachedStep(normed1, cache)
	x = x.Add(attnOut)

	normed2 := x.LayerNorm(tb.LN2Gamma, tb.LN2Beta)
	ffnOut := tb.FFN.Forward(normed2)
	x = x.Add(ffnOut)

	return x
}

// ─────────────────────────────────────────────────────────────────────
// MiniTransformer — fast generation with KV-cache
// ─────────────────────────────────────────────────────────────────────

// transformerCache holds the per-layer KVCache list plus the running
// sequence length (needed for positional embeddings during step).
type transformerCache struct {
	Layers []*KVCache
	SeqLen int // number of tokens in cache (== prompt length after prefill)
}

func newTransformerCache(numLayers int) *transformerCache {
	tc := &transformerCache{Layers: make([]*KVCache, numLayers)}
	for i := range tc.Layers {
		tc.Layers[i] = &KVCache{}
	}
	return tc
}

// prefill runs a normal forward over the prompt and populates the cache
// by hooking into the cached step for every token in order. This is
// simpler than re-engineering MultiHeadAttention.Forward to also emit
// K/V, and the cost is dominated by the prompt itself (O(N^2) once)
// which is what the user already paid for in the un-cached path.
func (m *MiniTransformer) prefill(promptIDs []int) (*transformerCache, *Tensor) {
	cache := newTransformerCache(len(m.Blocks))
	if len(promptIDs) == 0 {
		return cache, nil
	}

	embedDim := m.Config.EmbedDim
	var lastHidden *Tensor // [1, embedDim] from the final token after the stack

	for pos, id := range promptIDs {
		// Look up token + positional embedding for this single position.
		x := m.singleTokenEmbedding(id, pos)

		// Run the stack with cached step, populating each layer's cache.
		for i, block := range m.Blocks {
			x = block.ForwardCachedStep(x, cache.Layers[i])
		}
		x = x.LayerNorm(m.LNFGamma, m.LNFBeta)
		lastHidden = x

		_ = embedDim
	}
	cache.SeqLen = len(promptIDs)
	return cache, lastHidden
}

// singleTokenEmbedding returns [1, embedDim] = TokenEmb[id] + PosEmb[pos]
// with the same out-of-range fallback as EmbeddingTable.Forward.
func (m *MiniTransformer) singleTokenEmbedding(id, pos int) *Tensor {
	emb := m.Embedding
	if id < 0 || id >= emb.VocabSize {
		id = emb.UnkTokenID
		if id < 0 || id >= emb.VocabSize {
			id = 0
		}
	}
	if pos >= emb.MaxSeqLen {
		pos = emb.MaxSeqLen - 1 // clamp; mirrors the training-time truncation
	}
	out := NewTensor(1, emb.EmbedDim)
	tokOff := id * emb.EmbedDim
	posOff := pos * emb.EmbedDim
	for j := 0; j < emb.EmbedDim; j++ {
		out.Data[j] = emb.TokenEmb.Data[tokOff+j] + emb.PosEmb.Data[posOff+j]
	}
	return out
}

// logitsFromHidden projects a single [1, embedDim] hidden state to the
// vocabulary using tied embedding weights, matching Forward().
func (m *MiniTransformer) logitsFromHidden(hidden *Tensor) []float32 {
	if hidden == nil {
		return nil
	}
	logits := hidden.MatMulTransposed(m.Embedding.TokenEmb)
	out := make([]float32, logits.Shape[1])
	copy(out, logits.Data)
	return out
}

// GenerateFast is the KV-cached counterpart to Generate. Same signature,
// same sampling behaviour, just O(N) per emitted token instead of
// O(N^2). Safe to use anywhere Generate was used; falls back to the
// classical path if the prompt is empty.
func (m *MiniTransformer) GenerateFast(prompt []int, maxNewTokens int, temperature float32, topK int) []int {
	if len(prompt) == 0 {
		return prompt
	}
	if temperature <= 0 {
		temperature = 1.0
	}
	if topK <= 0 {
		topK = m.Config.VocabSize
	}

	// Clamp prompt to MaxSeqLen - room for at least one new token.
	maxPrompt := m.Config.MaxSeqLen - 1
	if maxPrompt < 1 {
		maxPrompt = 1
	}
	if len(prompt) > maxPrompt {
		prompt = prompt[len(prompt)-maxPrompt:]
	}

	cache, lastHidden := m.prefill(prompt)
	if lastHidden == nil {
		return prompt
	}

	generated := make([]int, len(prompt), len(prompt)+maxNewTokens)
	copy(generated, prompt)

	for i := 0; i < maxNewTokens; i++ {
		// Sample next token from current logits.
		logits := m.logitsFromHidden(lastHidden)
		if len(logits) == 0 {
			break
		}
		for j := range logits {
			logits[j] /= temperature
		}
		next := topKSample(logits, topK, m.Rng)
		generated = append(generated, next)

		// EOS short-circuit.
		if next == m.Config.EOSTokenID {
			break
		}

		// If we already hit the model's positional budget, stop. There is
		// no positional embedding beyond MaxSeqLen and silently reusing
		// the clamped value would degrade the next token quality.
		if cache.SeqLen >= m.Config.MaxSeqLen {
			break
		}

		// Run a single cached step for the just-emitted token.
		x := m.singleTokenEmbedding(next, cache.SeqLen)
		for j, block := range m.Blocks {
			x = block.ForwardCachedStep(x, cache.Layers[j])
		}
		x = x.LayerNorm(m.LNFGamma, m.LNFBeta)
		lastHidden = x
		cache.SeqLen++
	}

	return generated
}

// Compile-time guard: ensure rand.Rand is still imported when sampling
// helpers move around (referenced via topKSample in transformer.go).
var _ = rand.New
