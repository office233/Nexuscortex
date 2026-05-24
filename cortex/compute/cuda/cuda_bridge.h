// cuda_bridge.h — C API for Nexus Cortex CUDA compute kernels.
// Called from Go via CGO. All functions return 0 on success, non-zero on error.

#ifndef CUDA_BRIDGE_H
#define CUDA_BRIDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
  #ifdef CUDA_NEXUS_EXPORTS
    #define NEXUS_API __declspec(dllexport)
  #else
    #define NEXUS_API __declspec(dllimport)
  #endif
#else
  #define NEXUS_API
#endif

// Initialize CUDA device. Returns 0 on success.
NEXUS_API int nexus_cuda_init(int device_id);

// Release CUDA resources.
NEXUS_API void nexus_cuda_close(void);

// ForwardSparse: ternary neural layer forward pass.
NEXUS_API int nexus_cuda_forward_sparse(
    const uint32_t* activeIndices,
    const int32_t*  activeValues,
    uint32_t        activeCount,
    const uint32_t* tiles,
    const int32_t*  bias,
    int32_t*        output,
    uint32_t        tilesPerRow,
    uint32_t        outputSize
);

// BatchSDRSimilarity: compute popcount(query & memory[i]) for each memory SDR.
NEXUS_API int nexus_cuda_batch_sdr_similarity(
    const uint32_t* querySDR,
    const uint32_t* memorySDRs,
    uint8_t*        results,
    uint32_t        queryWords,
    uint32_t        numMemories
);

#ifdef __cplusplus
}
#endif

#endif // CUDA_BRIDGE_H
