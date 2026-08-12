
#include "embedding_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace {

template <typename T>
__global__ void embeddingKernel(
    T *output,
    const std::int64_t *indices,
    const T *weight,
    size_t seqlen,
    size_t embedding_dim
) {
    const size_t idx =
        static_cast<size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;

    const size_t numel = seqlen * embedding_dim;

    if (idx >= numel) {
        return;
    }

    // output[row, col]
    const size_t row = idx / embedding_dim;
    const size_t col = idx % embedding_dim;

    const size_t weight_row =
        static_cast<size_t>(indices[row]);

    output[idx] =
        weight[weight_row * embedding_dim + col];
}

template <typename T>
void launchEmbedding(
    T *output,
    const std::int64_t *indices,
    const T *weight,
    size_t seqlen,
    size_t embedding_dim
) {
    const size_t numel = seqlen * embedding_dim;

    if (numel == 0) {
        return;
    }

    constexpr int BLOCK_SIZE = 256;

    const int grid_size =
        static_cast<int>(
            (numel + BLOCK_SIZE - 1) / BLOCK_SIZE
        );

    embeddingKernel<T><<<grid_size, BLOCK_SIZE>>>(
        output,
        indices,
        weight,
        seqlen,
        embedding_dim
    );

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        throw std::runtime_error(
            cudaGetErrorString(error)
        );
    }
}

} // namespace

namespace llaisys::ops::nvidia {

void embedding(
    std::byte *output,
    const std::byte *indices,
    const std::byte *weight,
    llaisysDataType_t type,
    size_t seqlen,
    size_t embedding_dim
) {
    switch (type) {

    case LLAISYS_DTYPE_F32:
        return launchEmbedding<float>(
            reinterpret_cast<float *>(output),
            reinterpret_cast<const std::int64_t *>(indices),
            reinterpret_cast<const float *>(weight),
            seqlen,
            embedding_dim
        );

    case LLAISYS_DTYPE_F16:
        return launchEmbedding<__half>(
            reinterpret_cast<__half *>(output),
            reinterpret_cast<const std::int64_t *>(indices),
            reinterpret_cast<const __half *>(weight),
            seqlen,
            embedding_dim
        );

    case LLAISYS_DTYPE_BF16:
        return launchEmbedding<__nv_bfloat16>(
            reinterpret_cast<__nv_bfloat16 *>(output),
            reinterpret_cast<const std::int64_t *>(indices),
            reinterpret_cast<const __nv_bfloat16 *>(weight),
            seqlen,
            embedding_dim
        );

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia