package cortex

// tensor_parallel.go — goroutine helpers for embarrassingly-parallel
// per-row tensor work (matmul, layernorm, softmax). Pulled out of
// tensor.go so the math stays readable.

import (
	"runtime"
)

// tensorParallelMinRows is the smallest row count that justifies
// spawning a worker pool. Below this we keep the work sequential to
// avoid goroutine setup cost dominating (typical case: single-token
// queries during autoregressive generation).
const tensorParallelMinRows = 4

// tensorParallelism returns the worker count for row-parallel ops.
// Reads NumCPU each call so a runtime GOMAXPROCS bump takes effect
// without a restart. Caps at 16 to avoid scheduler churn on machines
// with very high core counts where the matmul shapes don't benefit.
func tensorParallelism() int {
	n := runtime.GOMAXPROCS(0)
	if n <= 0 {
		n = runtime.NumCPU()
	}
	if n > 16 {
		n = 16
	}
	return n
}

// runRowParallel splits [0, rows) into roughly equal chunks and runs fn
// on each chunk concurrently. Blocks until every chunk has finished.
// The callee receives [lo, hi) and should treat the slice as its
// exclusive write region (no synchronisation provided).
//
// The last chunk runs on the calling goroutine instead of a fresh one;
// that saves one goroutine spawn (and its closure allocation) per call.
// Worker completion is signalled through a buffered channel rather than
// sync.WaitGroup so the synchronisation primitive stays on the stack
// (WaitGroup escapes to the heap when captured by the spawned
// goroutines' defer).
func runRowParallel(rows, workers int, fn func(lo, hi int)) {
	if workers <= 1 || rows <= 1 {
		fn(0, rows)
		return
	}
	if workers > rows {
		workers = rows
	}
	chunk := (rows + workers - 1) / workers // ceil(rows / workers)

	// Pre-compute chunk boundaries.
	type span struct{ lo, hi int }
	var spans [16]span // capped at tensorParallelism's max
	nSpans := 0
	for w := 0; w < workers; w++ {
		lo := w * chunk
		if lo >= rows {
			break
		}
		hi := lo + chunk
		if hi > rows {
			hi = rows
		}
		spans[nSpans] = span{lo, hi}
		nSpans++
	}
	if nSpans == 1 {
		fn(spans[0].lo, spans[0].hi)
		return
	}

	// done is buffered to nSpans-1 so background workers never block.
	// One spawn (and one closure alloc) saved by running the last span
	// here on the caller's goroutine.
	done := make(chan struct{}, nSpans-1)
	for i := 0; i < nSpans-1; i++ {
		s := spans[i]
		go func() {
			fn(s.lo, s.hi)
			done <- struct{}{}
		}()
	}
	last := spans[nSpans-1]
	fn(last.lo, last.hi)
	for i := 0; i < nSpans-1; i++ {
		<-done
	}
}
