package cortex

// tensor_parallel.go — goroutine helpers for embarrassingly-parallel
// per-row tensor work (matmul, layernorm, softmax). Pulled out of
// tensor.go so the math stays readable.

import (
	"runtime"
	"sync"
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
func runRowParallel(rows, workers int, fn func(lo, hi int)) {
	if workers <= 1 || rows <= 1 {
		fn(0, rows)
		return
	}
	if workers > rows {
		workers = rows
	}
	chunk := (rows + workers - 1) / workers // ceil(rows / workers)

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		lo := w * chunk
		if lo >= rows {
			break
		}
		hi := lo + chunk
		if hi > rows {
			hi = rows
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			fn(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}
