// cublas_matmul.cu — float32 dense matmul bridge backed by cuBLAS sgemm.
//
// We expose two entry points to Go:
//   nexus_cublas_sgemm   : C = A * B
//   nexus_cublas_sgemm_nt: C = A * B^T  (B is given as row-major [N,K])
//
// Row-major vs column-major trick
// --------------------------------
// cuBLAS is column-major. Go tensors are row-major. The identity
//     (A_row * B_row)^T  ==  B_col * A_col
// lets us compute a row-major C[M,N] = A[M,K] * B[K,N] by calling cuBLAS
// as if it were   C_col[N,M] = B_col[N,K] * A_col[K,M]
// and reading the result back as row-major C[M,N]. No actual transpose
// happens — the underlying memory layout is identical.
//
// We allocate device buffers per call (small overhead — for the matmuls
// in a 5M-param transformer the H2D/D2H copy is the bottleneck anyway,
// not allocation). A future optimisation is a pinned-host arena +
// device buffer pool, but we keep it simple until profiled.

#include "cuda_bridge.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace {

cublasHandle_t g_handle = nullptr;
bool           g_inited = false;

// ─── Persistent device buffer arena ──────────────────────────────────
// cudaMalloc / cudaFree cost ~100 microseconds each on Windows. For a
// 100us matmul, that's pure waste. We keep three reusable device buffers
// (A, B, C) that grow on demand and never shrink, so the typical matmul
// pays zero allocation cost.
struct DeviceBuf {
    float* ptr;
    size_t bytes;
};
DeviceBuf g_bufA{nullptr, 0};
DeviceBuf g_bufB{nullptr, 0};
DeviceBuf g_bufC{nullptr, 0};

inline int ensureBuf(DeviceBuf& b, size_t needed) {
    if (b.bytes >= needed) return 0;
    if (b.ptr) cudaFree(b.ptr);
    b.ptr = nullptr;
    b.bytes = 0;
    cudaError_t e = cudaMalloc(&b.ptr, needed);
    if (e != cudaSuccess) return 1;
    b.bytes = needed;
    return 0;
}

inline void freeBuf(DeviceBuf& b) {
    if (b.ptr) cudaFree(b.ptr);
    b.ptr = nullptr;
    b.bytes = 0;
}

inline int check(cudaError_t e) {
    return (e == cudaSuccess) ? 0 : 1;
}

inline int check(cublasStatus_t s) {
    return (s == CUBLAS_STATUS_SUCCESS) ? 0 : 1;
}

} // namespace

extern "C" {

NEXUS_API int nexus_cublas_init(int device_id) {
    if (g_inited) return 0;
    if (check(cudaSetDevice(device_id)) != 0) return 1;
    if (check(cublasCreate(&g_handle)) != 0) return 2;
    g_inited = true;
    return 0;
}

NEXUS_API void nexus_cublas_close(void) {
    freeBuf(g_bufA);
    freeBuf(g_bufB);
    freeBuf(g_bufC);
    if (g_handle != nullptr) {
        cublasDestroy(g_handle);
        g_handle = nullptr;
    }
    g_inited = false;
}

// C[M,N] = A[M,K] * B[K,N]   (all row-major)
//
// Trick: ask cuBLAS to compute C_col[N,M] = B_col[N,K] * A_col[K,M] using
// no transpositions. Since column-major(X[r,c]) = row-major(X^T[c,r]),
// the bytes we write back are exactly the row-major C[M,N] we want.
NEXUS_API int nexus_cublas_sgemm(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    if (!g_inited) return -1;
    if (M <= 0 || N <= 0 || K <= 0) return -2;

    const size_t bytesA = static_cast<size_t>(M) * K * sizeof(float);
    const size_t bytesB = static_cast<size_t>(K) * N * sizeof(float);
    const size_t bytesC = static_cast<size_t>(M) * N * sizeof(float);

    if (ensureBuf(g_bufA, bytesA) != 0) return 10;
    if (ensureBuf(g_bufB, bytesB) != 0) return 11;
    if (ensureBuf(g_bufC, bytesC) != 0) return 12;

    if (check(cudaMemcpy(g_bufA.ptr, A, bytesA, cudaMemcpyHostToDevice)) != 0) return 20;
    if (check(cudaMemcpy(g_bufB.ptr, B, bytesB, cudaMemcpyHostToDevice)) != 0) return 21;

    const float alpha = 1.0f, beta = 0.0f;
    // Compute C_col[N,M] = B_col[N,K] * A_col[K,M]
    //   op(B) is [N,K] column-major, leading dim ldb = N
    //   op(A) is [K,M] column-major, leading dim lda = K
    //   C is [N,M] column-major, leading dim ldc = N
    if (check(cublasSgemm(
            g_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            g_bufB.ptr, N,
            g_bufA.ptr, K,
            &beta,
            g_bufC.ptr, N)) != 0) return 30;

    if (check(cudaMemcpy(C, g_bufC.ptr, bytesC, cudaMemcpyDeviceToHost)) != 0) return 40;
    return 0;
}

// C[M,N] = A[M,K] * B[N,K]^T   (B is row-major [N,K])
//
// Same column-major trick. We want row-major C[M,N] = A * B^T.
// Equivalent column-major: C_col[N,M] = B_col[N,K] * A_col[K,M]^? — no:
// reformulate. We want row-major C = A * B^T. Take transpose:
//   C^T = B * A^T   (row-major identity)
// Column-major of C is row-major of C^T. So column-major C[N,M] = B[N,K] * A^T[K,M].
// In cuBLAS column-major:
//   op(B) is B treated as column-major [K,N] but we want it as [N,K] —
//   that means we need to TRANSPOSE B for cuBLAS (CUBLAS_OP_T).
//   But B's bytes are row-major [N,K]; that's identical to column-major [K,N].
//   Apply CUBLAS_OP_T → cuBLAS sees [N,K] (which is what we want).
//   op(A^T) means we want A^T column-major [K,M]; A's bytes are row-major [M,K]
//   = column-major [K,M] already — exactly what we want, so CUBLAS_OP_N on A.
//
//   sgemm(opA=T on B, opB=N on A, m=N, n=M, k=K,
//         A=B (ld=K), B=A (ld=K), C (ld=N))
NEXUS_API int nexus_cublas_sgemm_nt(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    if (!g_inited) return -1;
    if (M <= 0 || N <= 0 || K <= 0) return -2;

    const size_t bytesA = static_cast<size_t>(M) * K * sizeof(float);
    const size_t bytesB = static_cast<size_t>(N) * K * sizeof(float);
    const size_t bytesC = static_cast<size_t>(M) * N * sizeof(float);

    if (ensureBuf(g_bufA, bytesA) != 0) return 10;
    if (ensureBuf(g_bufB, bytesB) != 0) return 11;
    if (ensureBuf(g_bufC, bytesC) != 0) return 12;

    if (check(cudaMemcpy(g_bufA.ptr, A, bytesA, cudaMemcpyHostToDevice)) != 0) return 20;
    if (check(cudaMemcpy(g_bufB.ptr, B, bytesB, cudaMemcpyHostToDevice)) != 0) return 21;

    const float alpha = 1.0f, beta = 0.0f;
    if (check(cublasSgemm(
            g_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            g_bufB.ptr, K,
            g_bufA.ptr, K,
            &beta,
            g_bufC.ptr, N)) != 0) return 30;

    if (check(cudaMemcpy(C, g_bufC.ptr, bytesC, cudaMemcpyDeviceToHost)) != 0) return 40;
    return 0;
}

} // extern "C"
