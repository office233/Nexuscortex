package cortex

import (
	"fmt"
	"math"
	"math/rand"

	"nexus-cortex/cortex/compute"
)

// gpuMatmulMinFlops is the lower flop threshold (M*N*K) at which a
// matmul becomes profitable on GPU. Below this the host<->device copy
// dominates and CPU is faster.
//
// Tuned for GTX 1660 Ti (PCIe 3.0 x16). Roughly: a 32^3 matmul = 32k
// flops takes ~20us H2D + 20us D2H + 5us compute = 45us total, while
// CPU does it in ~10us. A 128^3 = 2M flops matmul takes ~80us GPU vs
// ~600us CPU — GPU wins. Threshold is conservative.
const gpuMatmulMinFlops = 1 << 16 // 65536

// ─────────────────────────────────────────────────────────────────────
// Tensor — Minimal Tensor Library for Nexus Cortex
// ─────────────────────────────────────────────────────────────────────
//
// A lightweight tensor implementation providing the operations needed
// for transformer inference and training. CPU-only, float32, designed
// for clarity and correctness over maximum performance.
//
// Supported operations:
//   - MatMul:    matrix multiplication
//   - Add:       element-wise addition (with broadcasting)
//   - Softmax:   along last dimension
//   - LayerNorm: layer normalization
//   - GELU:      Gaussian Error Linear Unit activation
//   - ReLU:      Rectified Linear Unit activation
//   - Scale:     scalar multiplication
//   - Transpose: 2D matrix transpose
//   - Slice:     extract a sub-tensor along the first dimension

// Tensor holds a multi-dimensional array of float32 values.
// Data is stored in row-major (C-contiguous) order.
type Tensor struct {
	Data  []float32
	Shape []int
}

// NewTensor creates a zero-initialized tensor with the given shape.
func NewTensor(shape ...int) *Tensor {
	size := 1
	for _, d := range shape {
		size *= d
	}
	return &Tensor{
		Data:  make([]float32, size),
		Shape: append([]int{}, shape...),
	}
}

// NewTensorRand creates a tensor with random values from N(0, std).
func NewTensorRand(rng *rand.Rand, std float32, shape ...int) *Tensor {
	t := NewTensor(shape...)
	for i := range t.Data {
		t.Data[i] = float32(rng.NormFloat64()) * std
	}
	return t
}

// NewTensorFrom creates a tensor from existing data. Data is NOT copied.
func NewTensorFrom(data []float32, shape ...int) *Tensor {
	return &Tensor{
		Data:  data,
		Shape: append([]int{}, shape...),
	}
}

// Zeros fills the tensor with zeros.
func (t *Tensor) Zeros() {
	for i := range t.Data {
		t.Data[i] = 0
	}
}

// Clone creates a deep copy of the tensor.
func (t *Tensor) Clone() *Tensor {
	data := make([]float32, len(t.Data))
	copy(data, t.Data)
	shape := make([]int, len(t.Shape))
	copy(shape, t.Shape)
	return &Tensor{Data: data, Shape: shape}
}

// Size returns the total number of elements.
func (t *Tensor) Size() int {
	s := 1
	for _, d := range t.Shape {
		s *= d
	}
	return s
}

// Rows returns the first dimension (for 2D tensors).
func (t *Tensor) Rows() int {
	if len(t.Shape) < 1 {
		return 0
	}
	return t.Shape[0]
}

// Cols returns the second dimension (for 2D tensors).
func (t *Tensor) Cols() int {
	if len(t.Shape) < 2 {
		return 0
	}
	return t.Shape[1]
}

// At returns the element at the given indices (row-major).
func (t *Tensor) At(indices ...int) float32 {
	return t.Data[t.offset(indices...)]
}

// Set writes a value at the given indices.
func (t *Tensor) Set(val float32, indices ...int) {
	t.Data[t.offset(indices...)] = val
}

func (t *Tensor) offset(indices ...int) int {
	off := 0
	stride := 1
	for i := len(t.Shape) - 1; i >= 0; i-- {
		if i < len(indices) {
			off += indices[i] * stride
		}
		stride *= t.Shape[i]
	}
	return off
}

// ─────────────────────────────────────────────────────────────────────
// Core Operations
// ─────────────────────────────────────────────────────────────────────

// MatMul performs matrix multiplication: C = A × B.
// A is [M, K], B is [K, N], result is [M, N].
//
// Two optimisations vs. the naive triple loop:
//  1. i-k-j loop order. The innermost loop streams contiguous bytes of
//     B (b.Data[bOff+j]) instead of jumping by N (b.Data[k*N+j]); this
//     keeps the L1 cache hot and is ~3–4× faster on modern x86 for the
//     transformer's matmul shapes.
//  2. Per-row goroutines when M is big enough. The output rows are
//     independent (different i) so we can split the M range across all
//     CPU cores with no synchronisation. The threshold avoids spawning
//     work for tiny matmuls (single-token attention queries) where
//     goroutine overhead would dominate.
func (a *Tensor) MatMul(b *Tensor) *Tensor {
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		panic(fmt.Sprintf("MatMul requires 2D tensors, got %v × %v", a.Shape, b.Shape))
	}
	M, _ := a.Shape[0], a.Shape[1]
	N := b.Shape[1]
	c := NewTensor(M, N)
	a.MatMulInto(c, b)
	return c
}

// MatMulInto writes a × b into out. out must already be shaped [M, N]
// where M = a.Shape[0] and N = b.Shape[1]. Lets callers recycle the
// output buffer across iterations to avoid per-call allocations.
func (a *Tensor) MatMulInto(out, b *Tensor) {
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		panic(fmt.Sprintf("MatMulInto requires 2D tensors, got %v × %v", a.Shape, b.Shape))
	}
	M, K := a.Shape[0], a.Shape[1]
	K2, N := b.Shape[0], b.Shape[1]
	if K != K2 {
		panic(fmt.Sprintf("MatMulInto dimension mismatch: %v × %v", a.Shape, b.Shape))
	}
	if len(out.Shape) != 2 || out.Shape[0] != M || out.Shape[1] != N {
		panic(fmt.Sprintf("MatMulInto out shape mismatch: want [%d %d], got %v", M, N, out.Shape))
	}

	if compute.IsCuBLASAvailable() && M*N*K >= gpuMatmulMinFlops {
		if data, err := compute.MatMulGPU(a.Data, b.Data, M, N, K); err == nil {
			copy(out.Data, data)
			return
		}
	}

	matmulRows(a.Data, b.Data, out.Data, 0, M, K, N)
}

// matmulRowsSequential computes C[rowStart:rowEnd, :] = A × B with the
// cache-friendly i-k-j loop order. Pulled out so the parallel dispatcher
// can call it per worker on a row slab.
func matmulRowsSequential(a, b, c []float32, rowStart, rowEnd, K, N int) {
	for i := rowStart; i < rowEnd; i++ {
		aOff := i * K
		cOff := i * N
		// Zero the row first so we can accumulate with +=.
		for j := 0; j < N; j++ {
			c[cOff+j] = 0
		}
		for k := 0; k < K; k++ {
			aik := a[aOff+k]
			bOff := k * N
			for j := 0; j < N; j++ {
				c[cOff+j] += aik * b[bOff+j]
			}
		}
	}
}

// matmulRows dispatches the row range to a worker pool when worthwhile.
// Threshold tuned empirically: below ~4 rows the goroutine spin-up cost
// outweighs the gain (Generate's per-token single-row matmuls).
func matmulRows(a, b, c []float32, rowStart, rowEnd, K, N int) {
	rows := rowEnd - rowStart
	workers := tensorParallelism()
	if workers <= 1 || rows < tensorParallelMinRows {
		matmulRowsSequential(a, b, c, rowStart, rowEnd, K, N)
		return
	}
	runRowParallel(rows, workers, func(lo, hi int) {
		matmulRowsSequential(a, b, c, rowStart+lo, rowStart+hi, K, N)
	})
}

// MatMulTransposed performs A × B^T.
// A is [M, K], B is [N, K], result is [M, N].
// Avoids explicitly transposing B (cache-friendlier).
//
// The original i-j-k order is already cache-friendly here (both a and b
// rows are streamed contiguously inside the innermost loop), so we keep
// it and only add per-row goroutine parallelism.
func (a *Tensor) MatMulTransposed(b *Tensor) *Tensor {
	M := a.Shape[0]
	N := b.Shape[0]
	c := NewTensor(M, N)
	a.MatMulTransposedInto(c, b)
	return c
}

// MatMulTransposedInto writes a × b^T into out. Same buffer-recycling
// contract as MatMulInto. out must be [M, N] where M = a.Shape[0] and
// N = b.Shape[0].
func (a *Tensor) MatMulTransposedInto(out, b *Tensor) {
	M, K := a.Shape[0], a.Shape[1]
	N := b.Shape[0]
	if b.Shape[1] != K {
		panic(fmt.Sprintf("MatMulTransposedInto dimension mismatch: %v × %v^T", a.Shape, b.Shape))
	}
	if len(out.Shape) != 2 || out.Shape[0] != M || out.Shape[1] != N {
		panic(fmt.Sprintf("MatMulTransposedInto out shape mismatch: want [%d %d], got %v", M, N, out.Shape))
	}

	if compute.IsCuBLASAvailable() && M*N*K >= gpuMatmulMinFlops {
		if data, err := compute.MatMulNTGPU(a.Data, b.Data, M, N, K); err == nil {
			copy(out.Data, data)
			return
		}
	}

	matmulTRows(a.Data, b.Data, out.Data, 0, M, K, N)
}

func matmulTRowsSequential(a, b, c []float32, rowStart, rowEnd, K, N int) {
	for i := rowStart; i < rowEnd; i++ {
		aOff := i * K
		for j := 0; j < N; j++ {
			sum := float32(0)
			bOff := j * K
			for k := 0; k < K; k++ {
				sum += a[aOff+k] * b[bOff+k]
			}
			c[i*N+j] = sum
		}
	}
}

func matmulTRows(a, b, c []float32, rowStart, rowEnd, K, N int) {
	rows := rowEnd - rowStart
	workers := tensorParallelism()
	if workers <= 1 || rows < tensorParallelMinRows {
		matmulTRowsSequential(a, b, c, rowStart, rowEnd, K, N)
		return
	}
	runRowParallel(rows, workers, func(lo, hi int) {
		matmulTRowsSequential(a, b, c, rowStart+lo, rowStart+hi, K, N)
	})
}

// Add performs element-wise addition. Supports broadcasting: if b has
// fewer dimensions, it is broadcast along the leading dimensions of a.
func (a *Tensor) Add(b *Tensor) *Tensor {
	if len(a.Data) == len(b.Data) {
		// Same shape: element-wise
		c := a.Clone()
		for i := range c.Data {
			c.Data[i] += b.Data[i]
		}
		return c
	}

	// Broadcasting: b is a bias vector [N] added to each row of a [M, N]
	if len(a.Shape) == 2 && len(b.Shape) == 1 && b.Shape[0] == a.Shape[1] {
		c := a.Clone()
		M, N := a.Shape[0], a.Shape[1]
		for i := 0; i < M; i++ {
			off := i * N
			for j := 0; j < N; j++ {
				c.Data[off+j] += b.Data[j]
			}
		}
		return c
	}

	panic(fmt.Sprintf("Add shape mismatch: %v + %v", a.Shape, b.Shape))
}

// AddInPlace adds b to a in-place (modifies a). Supports the same
// element-wise and 2D+bias broadcast shapes as Add.
func (a *Tensor) AddInPlace(b *Tensor) {
	if len(a.Data) == len(b.Data) {
		for i := range a.Data {
			a.Data[i] += b.Data[i]
		}
		return
	}
	if len(a.Shape) == 2 && len(b.Shape) == 1 && b.Shape[0] == a.Shape[1] {
		M, N := a.Shape[0], a.Shape[1]
		for i := 0; i < M; i++ {
			off := i * N
			for j := 0; j < N; j++ {
				a.Data[off+j] += b.Data[j]
			}
		}
		return
	}
	panic(fmt.Sprintf("AddInPlace shape mismatch: %v + %v", a.Shape, b.Shape))
}

// Scale multiplies every element by a scalar.
func (t *Tensor) Scale(s float32) *Tensor {
	c := t.Clone()
	for i := range c.Data {
		c.Data[i] *= s
	}
	return c
}

// ScaleInPlace multiplies every element by s in place.
func (t *Tensor) ScaleInPlace(s float32) {
	for i := range t.Data {
		t.Data[i] *= s
	}
}

// Transpose returns the transpose of a 2D tensor.
func (t *Tensor) Transpose() *Tensor {
	if len(t.Shape) != 2 {
		panic("Transpose requires 2D tensor")
	}
	M, N := t.Shape[0], t.Shape[1]
	c := NewTensor(N, M)
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			c.Data[j*M+i] = t.Data[i*N+j]
		}
	}
	return c
}

// Slice extracts rows [start, end) from a 2D tensor.
func (t *Tensor) Slice(start, end int) *Tensor {
	if len(t.Shape) != 2 {
		panic("Slice requires 2D tensor")
	}
	N := t.Shape[1]
	rows := end - start
	data := make([]float32, rows*N)
	copy(data, t.Data[start*N:end*N])
	return NewTensorFrom(data, rows, N)
}

// Row returns a single row from a 2D tensor as a [1, N] tensor.
func (t *Tensor) Row(i int) *Tensor {
	return t.Slice(i, i+1)
}

// ─────────────────────────────────────────────────────────────────────
// Activation Functions
// ─────────────────────────────────────────────────────────────────────

// GELU applies the Gaussian Error Linear Unit activation function.
// Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
func (t *Tensor) GELU() *Tensor {
	c := t.Clone()
	c.GELUInPlace()
	return c
}

// GELUInPlace applies GELU to every element in place.
func (t *Tensor) GELUInPlace() {
	sqrt2pi := float32(math.Sqrt(2.0 / math.Pi))
	for i, x := range t.Data {
		inner := sqrt2pi * (x + 0.044715*x*x*x)
		t.Data[i] = 0.5 * x * (1.0 + float32(math.Tanh(float64(inner))))
	}
}

// ReLU applies the Rectified Linear Unit: max(0, x).
func (t *Tensor) ReLU() *Tensor {
	c := t.Clone()
	for i, x := range c.Data {
		if x < 0 {
			c.Data[i] = 0
		}
	}
	return c
}

// ─────────────────────────────────────────────────────────────────────
// Softmax
// ─────────────────────────────────────────────────────────────────────

// Softmax applies softmax along the last dimension.
// For a [M, N] tensor, softmax is computed independently for each row.
func (t *Tensor) Softmax() *Tensor {
	c := t.Clone()
	c.SoftmaxInPlace()
	return c
}

// SoftmaxInPlace applies softmax along the last dimension in place.
func (t *Tensor) SoftmaxInPlace() {
	if len(t.Shape) == 1 {
		softmaxRow(t.Data)
		return
	}
	if len(t.Shape) == 2 {
		M, N := t.Shape[0], t.Shape[1]
		for i := 0; i < M; i++ {
			softmaxRow(t.Data[i*N : (i+1)*N])
		}
		return
	}
	panic(fmt.Sprintf("SoftmaxInPlace supports 1D/2D tensors, got shape %v", t.Shape))
}

func softmaxRow(data []float32) {
	if len(data) == 0 {
		return
	}
	// Numerical stability: subtract max
	maxVal := data[0]
	for _, v := range data[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	sum := float32(0)
	for i, v := range data {
		e := float32(math.Exp(float64(v - maxVal)))
		data[i] = e
		sum += e
	}

	if sum > 0 {
		invSum := 1.0 / sum
		for i := range data {
			data[i] *= invSum
		}
	}
}

// ─────────────────────────────────────────────────────────────────────
// Layer Normalization
// ─────────────────────────────────────────────────────────────────────

// LayerNorm applies layer normalization along the last dimension.
// gamma (scale) and beta (shift) are learned parameters with shape [dim].
// Formula: y = gamma * (x - mean) / sqrt(var + eps) + beta
func (t *Tensor) LayerNorm(gamma, beta *Tensor) *Tensor {
	if len(t.Shape) != 2 {
		panic("LayerNorm requires 2D tensor [seq_len, dim]")
	}

	M, N := t.Shape[0], t.Shape[1]
	if gamma.Size() != N || beta.Size() != N {
		panic(fmt.Sprintf("LayerNorm gamma/beta size mismatch: dim=%d, gamma=%d, beta=%d",
			N, gamma.Size(), beta.Size()))
	}

	c := t.Clone()
	eps := float32(1e-5)

	for i := 0; i < M; i++ {
		off := i * N

		// Compute mean
		mean := float32(0)
		for j := 0; j < N; j++ {
			mean += c.Data[off+j]
		}
		mean /= float32(N)

		// Compute variance
		variance := float32(0)
		for j := 0; j < N; j++ {
			d := c.Data[off+j] - mean
			variance += d * d
		}
		variance /= float32(N)

		// Normalize
		invStd := float32(1.0 / math.Sqrt(float64(variance+eps)))
		for j := 0; j < N; j++ {
			normalized := (c.Data[off+j] - mean) * invStd
			c.Data[off+j] = gamma.Data[j]*normalized + beta.Data[j]
		}
	}

	return c
}

// ─────────────────────────────────────────────────────────────────────
// Cross-Entropy Loss
// ─────────────────────────────────────────────────────────────────────

// CrossEntropyLoss computes the average cross-entropy loss.
// logits is [N, VocabSize], targets is []int of length N.
// Returns the scalar loss value.
func CrossEntropyLoss(logits *Tensor, targets []int) float32 {
	N, V := logits.Shape[0], logits.Shape[1]
	if len(targets) != N {
		panic(fmt.Sprintf("CrossEntropyLoss: %d logits rows but %d targets", N, len(targets)))
	}

	totalLoss := float32(0)
	for i := 0; i < N; i++ {
		row := logits.Data[i*V : (i+1)*V]

		// Compute log-softmax for numerical stability
		maxVal := row[0]
		for _, v := range row[1:] {
			if v > maxVal {
				maxVal = v
			}
		}

		logSum := float32(0)
		for _, v := range row {
			logSum += float32(math.Exp(float64(v - maxVal)))
		}
		logSumExp := maxVal + float32(math.Log(float64(logSum)))

		targetIdx := targets[i]
		if targetIdx >= 0 && targetIdx < V {
			totalLoss += -(row[targetIdx] - logSumExp)
		}
	}

	return totalLoss / float32(N)
}

// CrossEntropySoftmaxGrad computes the gradient of cross-entropy loss
// w.r.t. logits. Returns dLogits [N, VocabSize].
// This is simply softmax(logits) - one_hot(targets).
func CrossEntropySoftmaxGrad(logits *Tensor, targets []int) *Tensor {
	N, V := logits.Shape[0], logits.Shape[1]
	grad := logits.Softmax()

	// Subtract 1 from the target positions
	for i := 0; i < N; i++ {
		idx := targets[i]
		if idx >= 0 && idx < V {
			grad.Data[i*V+idx] -= 1.0
		}
	}

	// Average over batch
	scale := 1.0 / float32(N)
	for i := range grad.Data {
		grad.Data[i] *= scale
	}

	return grad
}
