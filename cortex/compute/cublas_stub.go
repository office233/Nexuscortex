//go:build !cuda
// +build !cuda

// cublas_stub.go — pure-Go fallback when the binary is built without
// `-tags cuda`. All entry points are no-ops or sentinel errors so that
// callers can probe IsCuBLASAvailable() and gracefully use the CPU path.

package compute

import "errors"

// InitCuBLAS in stub mode always errors; the rest of the code is
// expected to react by staying on the CPU MatMul path.
func InitCuBLAS() error {
	return errors.New("cublas not compiled in (rebuild with -tags cuda)")
}

// CloseCuBLAS is a no-op stub.
func CloseCuBLAS() {}

// IsCuBLASAvailable always reports false in stub builds.
func IsCuBLASAvailable() bool { return false }

// MatMulGPU stub always errors; callers must check IsCuBLASAvailable
// first and route to the CPU matmul instead.
func MatMulGPU(A, B []float32, M, N, K int) ([]float32, error) {
	return nil, errors.New("cublas not compiled in (rebuild with -tags cuda)")
}

// MatMulNTGPU stub always errors.
func MatMulNTGPU(A, B []float32, M, N, K int) ([]float32, error) {
	return nil, errors.New("cublas not compiled in (rebuild with -tags cuda)")
}
