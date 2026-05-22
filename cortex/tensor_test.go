package cortex

import (
	"math"
	"math/rand"
	"testing"
)

// ─────────────────────────────────────────────────────────────────────
// Tensor Tests
// ─────────────────────────────────────────────────────────────────────

func TestTensorNewAndShape(t *testing.T) {
	x := NewTensor(3, 4)
	if x.Rows() != 3 || x.Cols() != 4 {
		t.Errorf("expected 3x4, got %dx%d", x.Rows(), x.Cols())
	}
	if x.Size() != 12 {
		t.Errorf("expected size 12, got %d", x.Size())
	}
	// All zeros
	for _, v := range x.Data {
		if v != 0 {
			t.Fatal("expected all zeros")
		}
	}
}

func TestTensorSetAt(t *testing.T) {
	x := NewTensor(2, 3)
	x.Set(5.0, 1, 2)
	if x.At(1, 2) != 5.0 {
		t.Errorf("expected 5.0, got %f", x.At(1, 2))
	}
	if x.At(0, 0) != 0 {
		t.Error("expected 0 at (0,0)")
	}
}

func TestTensorMatMul(t *testing.T) {
	// [2,3] × [3,2] = [2,2]
	a := NewTensorFrom([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	b := NewTensorFrom([]float32{7, 8, 9, 10, 11, 12}, 3, 2)

	c := a.MatMul(b)

	if c.Rows() != 2 || c.Cols() != 2 {
		t.Fatalf("expected 2x2, got %v", c.Shape)
	}

	// Manual: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
	//         [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
	expected := []float32{58, 64, 139, 154}
	for i, v := range expected {
		if math.Abs(float64(c.Data[i]-v)) > 1e-4 {
			t.Errorf("MatMul[%d]: expected %f, got %f", i, v, c.Data[i])
		}
	}
}

func TestTensorMatMulTransposed(t *testing.T) {
	a := NewTensorFrom([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	b := NewTensorFrom([]float32{7, 8, 9, 10, 11, 12}, 2, 3) // B^T is [3,2]

	c := a.MatMulTransposed(b)

	// A × B^T = [2,3] × [3,2] where B^T is [[7,10],[8,11],[9,12]]
	// row 0: [1*7+2*8+3*9, 1*10+2*11+3*12] = [50, 68]
	// row 1: [4*7+5*8+6*9, 4*10+5*11+6*12] = [122, 167]
	expected := []float32{50, 68, 122, 167}
	for i, v := range expected {
		if math.Abs(float64(c.Data[i]-v)) > 1e-4 {
			t.Errorf("MatMulTransposed[%d]: expected %f, got %f", i, v, c.Data[i])
		}
	}
}

func TestTensorAdd(t *testing.T) {
	a := NewTensorFrom([]float32{1, 2, 3, 4}, 2, 2)
	b := NewTensorFrom([]float32{10, 20, 30, 40}, 2, 2)

	c := a.Add(b)
	expected := []float32{11, 22, 33, 44}
	for i, v := range expected {
		if c.Data[i] != v {
			t.Errorf("Add[%d]: expected %f, got %f", i, v, c.Data[i])
		}
	}
}

func TestTensorAddBroadcast(t *testing.T) {
	// [2,3] + [3] → broadcast bias
	a := NewTensorFrom([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	bias := NewTensorFrom([]float32{10, 20, 30}, 3)

	c := a.Add(bias)
	expected := []float32{11, 22, 33, 14, 25, 36}
	for i, v := range expected {
		if c.Data[i] != v {
			t.Errorf("AddBroadcast[%d]: expected %f, got %f", i, v, c.Data[i])
		}
	}
}

func TestTensorSoftmax(t *testing.T) {
	x := NewTensorFrom([]float32{1, 2, 3}, 3)
	s := x.Softmax()

	// Sum should be 1.0
	sum := float32(0)
	for _, v := range s.Data {
		sum += v
	}
	if math.Abs(float64(sum-1.0)) > 1e-5 {
		t.Errorf("softmax sum: expected 1.0, got %f", sum)
	}

	// Values should be monotonically increasing
	if s.Data[0] >= s.Data[1] || s.Data[1] >= s.Data[2] {
		t.Errorf("softmax not monotonic: %v", s.Data)
	}
}

func TestTensorSoftmax2D(t *testing.T) {
	x := NewTensorFrom([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	s := x.Softmax()

	// Each row should sum to 1
	for i := 0; i < 2; i++ {
		sum := float32(0)
		for j := 0; j < 3; j++ {
			sum += s.Data[i*3+j]
		}
		if math.Abs(float64(sum-1.0)) > 1e-5 {
			t.Errorf("row %d softmax sum: expected 1.0, got %f", i, sum)
		}
	}
}

func TestTensorLayerNorm(t *testing.T) {
	x := NewTensorFrom([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	gamma := NewTensorFrom([]float32{1, 1, 1}, 3)
	beta := NewTensorFrom([]float32{0, 0, 0}, 3)

	y := x.LayerNorm(gamma, beta)

	// Each row should have mean≈0, var≈1
	for i := 0; i < 2; i++ {
		mean := float32(0)
		for j := 0; j < 3; j++ {
			mean += y.Data[i*3+j]
		}
		mean /= 3

		if math.Abs(float64(mean)) > 1e-4 {
			t.Errorf("row %d mean: expected ~0, got %f", i, mean)
		}

		variance := float32(0)
		for j := 0; j < 3; j++ {
			d := y.Data[i*3+j] - mean
			variance += d * d
		}
		variance /= 3

		if math.Abs(float64(variance-1.0)) > 1e-3 {
			t.Errorf("row %d variance: expected ~1, got %f", i, variance)
		}
	}
}

func TestTensorGELU(t *testing.T) {
	x := NewTensorFrom([]float32{-2, -1, 0, 1, 2}, 5)
	y := x.GELU()

	// GELU(0) ≈ 0
	if math.Abs(float64(y.Data[2])) > 1e-4 {
		t.Errorf("GELU(0) should be ~0, got %f", y.Data[2])
	}

	// GELU(x) ≈ x for large positive x
	if math.Abs(float64(y.Data[4]-2.0)) > 0.1 {
		t.Errorf("GELU(2) should be ~2, got %f", y.Data[4])
	}

	// GELU(x) ≈ 0 for large negative x
	if math.Abs(float64(y.Data[0])) > 0.1 {
		t.Errorf("GELU(-2) should be ~0, got %f", y.Data[0])
	}
}

func TestTensorTranspose(t *testing.T) {
	x := NewTensorFrom([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	y := x.Transpose()

	if y.Rows() != 3 || y.Cols() != 2 {
		t.Fatalf("expected 3x2, got %v", y.Shape)
	}

	expected := []float32{1, 4, 2, 5, 3, 6}
	for i, v := range expected {
		if y.Data[i] != v {
			t.Errorf("Transpose[%d]: expected %f, got %f", i, v, y.Data[i])
		}
	}
}

func TestTensorSlice(t *testing.T) {
	x := NewTensorFrom([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3, 3)
	y := x.Slice(1, 3) // rows 1 and 2

	if y.Rows() != 2 || y.Cols() != 3 {
		t.Fatalf("expected 2x3, got %v", y.Shape)
	}

	expected := []float32{4, 5, 6, 7, 8, 9}
	for i, v := range expected {
		if y.Data[i] != v {
			t.Errorf("Slice[%d]: expected %f, got %f", i, v, y.Data[i])
		}
	}
}

func TestCrossEntropyLoss(t *testing.T) {
	// Logits [2, 4]: two examples, 4 classes
	logits := NewTensorFrom([]float32{
		10, 0, 0, 0, // strong prediction for class 0
		0, 0, 10, 0, // strong prediction for class 2
	}, 2, 4)

	targets := []int{0, 2} // correct classes

	loss := CrossEntropyLoss(logits, targets)

	// Loss should be very small since predictions are correct
	if loss > 0.1 {
		t.Errorf("expected low loss for correct predictions, got %f", loss)
	}

	// Wrong targets should give high loss
	wrongTargets := []int{3, 0}
	wrongLoss := CrossEntropyLoss(logits, wrongTargets)
	if wrongLoss < 5.0 {
		t.Errorf("expected high loss for wrong targets, got %f", wrongLoss)
	}
}

func TestCrossEntropySoftmaxGrad(t *testing.T) {
	logits := NewTensorFrom([]float32{1, 2, 3, 4}, 1, 4)
	targets := []int{2}

	grad := CrossEntropySoftmaxGrad(logits, targets)

	// Gradient at target position should be negative (softmax - 1)
	// Gradients at non-target positions should be positive (softmax)
	if grad.Data[2] >= 0 {
		t.Error("gradient at target should be negative")
	}

	// Sum of gradients in a row should be ~0
	sum := float32(0)
	for j := 0; j < 4; j++ {
		sum += grad.Data[j]
	}
	if math.Abs(float64(sum)) > 1e-5 {
		t.Errorf("gradient row sum should be ~0, got %f", sum)
	}
}

// ─────────────────────────────────────────────────────────────────────
// Embedding Tests
// ─────────────────────────────────────────────────────────────────────

func TestEmbeddingForward(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	emb := NewEmbeddingTable(100, 16, 64, rng)

	ids := []int{5, 10, 15}
	output := emb.Forward(ids)

	if output.Rows() != 3 || output.Cols() != 16 {
		t.Fatalf("expected [3, 16], got %v", output.Shape)
	}

	// Verify the output is token_emb[id] + pos_emb[pos]
	for pos, id := range ids {
		for j := 0; j < 16; j++ {
			expected := emb.TokenEmb.Data[id*16+j] + emb.PosEmb.Data[pos*16+j]
			got := output.Data[pos*16+j]
			if math.Abs(float64(expected-got)) > 1e-6 {
				t.Errorf("pos=%d, j=%d: expected %f, got %f", pos, j, expected, got)
			}
		}
	}
}

func TestEmbeddingBackward(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	emb := NewEmbeddingTable(100, 4, 64, rng)

	ids := []int{5, 5, 10} // id 5 appears twice
	emb.ZeroGrad()

	// Create gradient of ones
	dOutput := NewTensor(3, 4)
	for i := range dOutput.Data {
		dOutput.Data[i] = 1.0
	}

	emb.Backward(dOutput, ids)

	// Token 5 should have gradient accumulated from 2 positions
	for j := 0; j < 4; j++ {
		grad := emb.TokenEmbGrad.Data[5*4+j]
		if math.Abs(float64(grad-2.0)) > 1e-6 {
			t.Errorf("token 5 grad[%d]: expected 2.0 (accumulated), got %f", j, grad)
		}
	}

	// Token 10 should have gradient from 1 position
	for j := 0; j < 4; j++ {
		grad := emb.TokenEmbGrad.Data[10*4+j]
		if math.Abs(float64(grad-1.0)) > 1e-6 {
			t.Errorf("token 10 grad[%d]: expected 1.0, got %f", j, grad)
		}
	}
}

func TestEmbeddingParamCount(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	emb := NewEmbeddingTable(32768, 256, 512, rng)

	params := emb.ParamCount()
	expected := 32768*256 + 512*256
	if params != expected {
		t.Errorf("expected %d params, got %d", expected, params)
	}
	t.Logf("Embedding params: %d (~%.1fM)", params, float64(params)/1e6)
}

func TestEmbeddingUpdate(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	emb := NewEmbeddingTable(10, 4, 8, rng)

	// Save original value
	orig := emb.TokenEmb.Data[0]

	// Set gradient
	emb.ZeroGrad()
	emb.TokenEmbGrad.Data[0] = 1.0

	// Update with lr=0.1
	emb.Update(0.1)

	// Value should decrease by lr * grad
	expected := orig - 0.1*1.0
	if math.Abs(float64(emb.TokenEmb.Data[0]-expected)) > 1e-6 {
		t.Errorf("after update: expected %f, got %f", expected, emb.TokenEmb.Data[0])
	}
}
