//go:build cuda
// +build cuda

// cublas_matmul.go — Go-side cuBLAS dense float32 matmul bridge.
//
// Exposes three things:
//
//   InitCuBLAS()  — call once at process startup. Returns nil if a GPU
//                   was successfully grabbed; non-nil error otherwise.
//                   Safe to call multiple times (idempotent).
//   CloseCuBLAS() — release the GPU handle. Call at shutdown.
//   IsCuBLASAvailable() bool — returns true once Init has succeeded.
//   MatMulGPU(A, B)   — row-major C[M,N] = A[M,K] * B[K,N].
//   MatMulNTGPU(A, B) — row-major C[M,N] = A[M,K] * B[N,K]^T.
//
// The cuBLAS handle is process-global and NOT thread-safe. We serialise
// every GPU call with a single sync.Mutex. That is fine because the
// Broca 2.0 trainer is single-stream: even when goroutines parallelise
// row-slabs of a matmul, the row-slab parallelism is bypassed when GPU
// is active (one big sgemm replaces N small ones).

package compute

/*
#cgo CFLAGS: -I${SRCDIR}/cuda
#cgo LDFLAGS: -L${SRCDIR}/cuda -lcuda_nexus
#include "cuda_bridge.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"unsafe"
)

var (
	cublasMu        sync.Mutex  // serialises every cuBLAS call (handle is non-reentrant)
	cublasReady     atomic.Bool // true once init succeeded
	cublasInitOnce  sync.Once
	cublasInitError error
)

// InitCuBLAS grabs device 0 and creates a cuBLAS handle. Idempotent:
// repeated calls return the result of the first attempt.
func InitCuBLAS() error {
	cublasInitOnce.Do(func() {
		ret := C.nexus_cublas_init(C.int(0))
		if ret != 0 {
			cublasInitError = fmt.Errorf("nexus_cublas_init returned %d", int(ret))
			return
		}
		cublasReady.Store(true)
	})
	return cublasInitError
}

// CloseCuBLAS releases the handle. After this, IsCuBLASAvailable returns
// false until a fresh process re-inits.
func CloseCuBLAS() {
	cublasMu.Lock()
	defer cublasMu.Unlock()
	if !cublasReady.Load() {
		return
	}
	C.nexus_cublas_close()
	cublasReady.Store(false)
}

// IsCuBLASAvailable reports whether GPU matmul is ready to use.
func IsCuBLASAvailable() bool {
	return cublasReady.Load()
}

// MatMulGPU computes C[M,N] = A[M,K] * B[K,N] in row-major layout.
// A must have len M*K, B must have len K*N, returned slice has len M*N.
func MatMulGPU(A, B []float32, M, N, K int) ([]float32, error) {
	if !cublasReady.Load() {
		return nil, errors.New("cublas not initialised")
	}
	if M <= 0 || N <= 0 || K <= 0 {
		return nil, fmt.Errorf("invalid dims M=%d N=%d K=%d", M, N, K)
	}
	if len(A) != M*K {
		return nil, fmt.Errorf("A length %d != M*K=%d", len(A), M*K)
	}
	if len(B) != K*N {
		return nil, fmt.Errorf("B length %d != K*N=%d", len(B), K*N)
	}
	C_ := make([]float32, M*N)

	cublasMu.Lock()
	ret := C.nexus_cublas_sgemm(
		(*C.float)(unsafe.Pointer(&A[0])),
		(*C.float)(unsafe.Pointer(&B[0])),
		(*C.float)(unsafe.Pointer(&C_[0])),
		C.int(M), C.int(N), C.int(K),
	)
	cublasMu.Unlock()

	if ret != 0 {
		return nil, fmt.Errorf("nexus_cublas_sgemm returned %d", int(ret))
	}
	return C_, nil
}

// MatMulNTGPU computes C[M,N] = A[M,K] * B[N,K]^T in row-major layout.
// Equivalent to Tensor.MatMulTransposed on CPU.
func MatMulNTGPU(A, B []float32, M, N, K int) ([]float32, error) {
	if !cublasReady.Load() {
		return nil, errors.New("cublas not initialised")
	}
	if M <= 0 || N <= 0 || K <= 0 {
		return nil, fmt.Errorf("invalid dims M=%d N=%d K=%d", M, N, K)
	}
	if len(A) != M*K {
		return nil, fmt.Errorf("A length %d != M*K=%d", len(A), M*K)
	}
	if len(B) != N*K {
		return nil, fmt.Errorf("B length %d != N*K=%d", len(B), N*K)
	}
	C_ := make([]float32, M*N)

	cublasMu.Lock()
	ret := C.nexus_cublas_sgemm_nt(
		(*C.float)(unsafe.Pointer(&A[0])),
		(*C.float)(unsafe.Pointer(&B[0])),
		(*C.float)(unsafe.Pointer(&C_[0])),
		C.int(M), C.int(N), C.int(K),
	)
	cublasMu.Unlock()

	if ret != 0 {
		return nil, fmt.Errorf("nexus_cublas_sgemm_nt returned %d", int(ret))
	}
	return C_, nil
}
