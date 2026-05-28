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

// appendRow appends one [embedDim] row to existing and returns the
// updated tensor. If existing was created with extra cap (see
// newKVCacheLayer), the append is O(embedDim) with no reallocation:
// we grow the Data slice in place and bump Shape[0]. Otherwise we fall
// back to a fresh allocation (used by callers that don't preallocate,
// e.g. tests that build a KVCache by hand).
func appendRow(existing *Tensor, row *Tensor, embedDim int) *Tensor {
	if existing == nil {
		// No preallocated capacity: caller is fine with a one-row tensor.
		out := NewTensor(1, embedDim)
		copy(out.Data, row.Data)
		return out
	}
	oldSeq := existing.Shape[0]
	oldLen := oldSeq * embedDim
	needed := oldLen + embedDim
	if cap(existing.Data) >= needed {
		existing.Data = existing.Data[:needed]
		copy(existing.Data[oldLen:], row.Data)
		existing.Shape[0] = oldSeq + 1
		return existing
	}
	// Cold path: no headroom. Allocate fresh.
	out := NewTensor(oldSeq+1, embedDim)
	copy(out.Data, existing.Data)
	copy(out.Data[oldLen:], row.Data)
	return out
}

// newKVCacheLayer pre-allocates K and V with zero length but capacity
// for maxSeqLen rows, so appendRow can grow in place without realloc.
func newKVCacheLayer(maxSeqLen, embedDim int) *KVCache {
	capSlots := maxSeqLen * embedDim
	return &KVCache{
		K: &Tensor{Data: make([]float32, 0, capSlots), Shape: []int{0, embedDim}},
		V: &Tensor{Data: make([]float32, 0, capSlots), Shape: []int{0, embedDim}},
	}
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
	// MatMul returns a fresh tensor so the bias add is safe in place.
	Q := x.MatMul(mha.WQ)
	Q.AddInPlace(mha.BQ)
	Knew := x.MatMul(mha.WK)
	Knew.AddInPlace(mha.BK)
	Vnew := x.MatMul(mha.WV)
	Vnew.AddInPlace(mha.BV)

	// Append to cache so future tokens can attend back. The new query
	// will read from cache.K / cache.V right after.
	cache.K = appendRow(cache.K, Knew, embedDim)
	cache.V = appendRow(cache.V, Vnew, embedDim)

	seqLen := cache.K.Shape[0]
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	attnOut := NewTensor(1, embedDim)

	// Grow per-head scratch slices if needed. Reused across heads here
	// and across consecutive cached steps (they hold no state).
	if cap(mha.stepQhBuf) < headDim {
		mha.stepQhBuf = make([]float32, headDim)
	}
	if cap(mha.stepScoresBuf) < seqLen {
		mha.stepScoresBuf = make([]float32, seqLen)
	}
	Qh := mha.stepQhBuf[:headDim]
	scores := mha.stepScoresBuf[:seqLen]

	// Per-head attention. With a single-row query the attention reduces
	// to a vector × matrix^T softmax × matrix, no masking needed because
	// the new token is the last row of cache (everything is "past").
	for h := 0; h < numHeads; h++ {
		hStart := h * headDim

		// Extract this head's Q row [1, headDim] into the reused buffer.
		for j := 0; j < headDim; j++ {
			Qh[j] = Q.Data[hStart+j]
		}

		// Compute scores Qh · Kh^T → [seqLen] in the reused buffer.
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
	out := attnOut.MatMul(mha.WO)
	out.AddInPlace(mha.BO)
	return out
}

// ─────────────────────────────────────────────────────────────────────
// TransformerBlock — cached single-token step
// ─────────────────────────────────────────────────────────────────────

// ForwardCachedStep mirrors TransformerBlock.Forward semantics for a
// single new token, given the per-layer KV cache.
func (tb *TransformerBlock) ForwardCachedStep(x *Tensor, cache *KVCache) *Tensor {
	// No backward in the cache path — nothing aliases x or attnOut/ffnOut,
	// so fold the residual directly into the fresh sub-layer outputs.
	normed1 := x.LayerNorm(tb.LN1Gamma, tb.LN1Beta)
	attnOut := tb.Attn.ForwardCachedStep(normed1, cache)
	attnOut.AddInPlace(x)
	x = attnOut

	normed2 := x.LayerNorm(tb.LN2Gamma, tb.LN2Beta)
	ffnOut := tb.FFN.Forward(normed2)
	ffnOut.AddInPlace(x)
	x = ffnOut

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

// newTransformerCachePrealloc creates a cache whose K/V slices have
// capacity for maxSeqLen rows so the per-step append is realloc-free.
func newTransformerCachePrealloc(numLayers, maxSeqLen, embedDim int) *transformerCache {
	tc := &transformerCache{Layers: make([]*KVCache, numLayers)}
	for i := range tc.Layers {
		tc.Layers[i] = newKVCacheLayer(maxSeqLen, embedDim)
	}
	return tc
}

// prefill runs a normal forward over the prompt and populates the cache
// by hooking into the cached step for every token in order. This is
// simpler than re-engineering MultiHeadAttention.Forward to also emit
// K/V, and the cost is dominated by the prompt itself (O(N^2) once)
// which is what the user already paid for in the un-cached path.
func (m *MiniTransformer) prefill(promptIDs []int) (*transformerCache, *Tensor) {
	embedDim := m.Config.EmbedDim
	// Preallocate K/V capacity for the full max sequence length so the
	// per-step appendRow inside ForwardCachedStep grows in place.
	cache := newTransformerCachePrealloc(len(m.Blocks), m.Config.MaxSeqLen, embedDim)
	if len(promptIDs) == 0 {
		return cache, nil
	}

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
//
// Thin wrapper around GenerateFastMin with minNewTokens=0 (no EOS
// suppression). See GenerateFastMin for the variant that forces a
// minimum generation length — useful when the model has learned to
// emit EOS too early after short prompts (vezi
// docs/plans/2026-05-26-eos-degeneration.md).
func (m *MiniTransformer) GenerateFast(prompt []int, maxNewTokens int, temperature float32, topK int) []int {
	return m.GenerateFastMin(prompt, maxNewTokens, 0, temperature, topK)
}

// GenerateFastMin is like GenerateFast but suppresses the EOS token for
// the first minNewTokens emitted tokens. After that the EOS token can
// terminate generation normally. If minNewTokens <= 0 the behaviour is
// identical to GenerateFast.
//
// Suppression is done by setting logits[EOSTokenID] = -Inf BEFORE
// top-K filtering, so EOS cannot be in the top-K candidate set during
// the suppression window.
func (m *MiniTransformer) GenerateFastMin(prompt []int, maxNewTokens, minNewTokens int, temperature float32, topK int) []int {
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

	eosID := m.Config.EOSTokenID

	for i := 0; i < maxNewTokens; i++ {
		// Sample next token from current logits.
		logits := m.logitsFromHidden(lastHidden)
		if len(logits) == 0 {
			break
		}
		for j := range logits {
			logits[j] /= temperature
		}

		// EOS suppression for first minNewTokens emitted tokens.
		// i is the 0-based index of the token we're about to emit.
		if i < minNewTokens && eosID >= 0 && eosID < len(logits) {
			logits[eosID] = float32(math.Inf(-1))
		}

		next := topKSample(logits, topK, m.Rng)
		generated = append(generated, next)

		// EOS short-circuit (only triggers after suppression window).
		if next == eosID {
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
