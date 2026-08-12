#include "argmax_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <climits>
#include <cstdint>
#include <stdexcept>

namespace {

template <typename T>
struct CudaCast;

template <>
struct CudaCast<float> {
    __device__ static float toFloat(float value) {
        return value;
    }
};

template <>
struct CudaCast<__half> {
    __device__ static float toFloat(__half value) {
        return __half2float(value);
    }
};

template <>
struct CudaCast<__nv_bfloat16> {
    __device__ static float toFloat(__nv_bfloat16 value) {
        return __bfloat162float(value);
    }
};

template <typename T>
__global__ void argmaxKernel(
    std::int64_t *max_idx,
    T *max_val,
    const T *vals,
    size_t numel
) {
    constexpr int BLOCK_SIZE = 256;

    __shared__ float shared_val[BLOCK_SIZE];
    __shared__ std::int64_t shared_idx[BLOCK_SIZE];

    float local_max = -CUDART_INF_F;
    std::int64_t local_idx = INT64_MAX;

    // 每个线程扫描：
    // tid, tid + blockDim.x, tid + 2 * blockDim.x, ...
    for (
        size_t i = threadIdx.x;
        i < numel;
        i += blockDim.x
    ) {
        const float current =
            CudaCast<T>::toFloat(vals[i]);

        // 如果值更大，更新。
        // 如果值相同，保留更小的 index，
        // 从而和 CPU 版本一样返回第一次出现的最大值。
        if (
            current > local_max ||
            (
                current == local_max &&
                static_cast<std::int64_t>(i) < local_idx
            )
        ) {
            local_max = current;
            local_idx =
                static_cast<std::int64_t>(i);
        }
    }

    shared_val[threadIdx.x] = local_max;
    shared_idx[threadIdx.x] = local_idx;

    __syncthreads();

    // block 内 reduction
    for (
        unsigned int stride = blockDim.x / 2;
        stride > 0;
        stride >>= 1
    ) {
        if (threadIdx.x < stride) {
            const unsigned int other =
                threadIdx.x + stride;

            const float my_val =
                shared_val[threadIdx.x];

            const std::int64_t my_idx =
                shared_idx[threadIdx.x];

            const float other_val =
                shared_val[other];

            const std::int64_t other_idx =
                shared_idx[other];

            if (
                other_val > my_val ||
                (
                    other_val == my_val &&
                    other_idx < my_idx
                )
            ) {
                shared_val[threadIdx.x] =
                    other_val;

                shared_idx[threadIdx.x] =
                    other_idx;
            }
        }

        __syncthreads();
    }

    // 最终结果由 thread 0 写出
    if (threadIdx.x == 0) {
        const std::int64_t best_idx =
            shared_idx[0];

        max_idx[0] = best_idx;

        // 直接复制原始 T。
        // 对 F16/BF16 来说这样可以保留原始 bit pattern，
        // 不需要 float -> half/bfloat16 再转换一次。
        max_val[0] = vals[best_idx];
    }
}

template <typename T>
void launchArgmax(
    std::int64_t *max_idx,
    T *max_val,
    const T *vals,
    size_t numel
) {
    if (numel == 0) {
        return;
    }

    constexpr int BLOCK_SIZE = 256;

    argmaxKernel<T><<<1, BLOCK_SIZE>>>(
        max_idx,
        max_val,
        vals,
        numel
    );

    const cudaError_t error =
        cudaGetLastError();

    if (error != cudaSuccess) {
        throw std::runtime_error(
            cudaGetErrorString(error)
        );
    }
}

} // namespace

namespace llaisys::ops::nvidia {

void argmax(
    std::byte *max_idx,
    std::byte *max_val,
    const std::byte *vals,
    llaisysDataType_t type,
    size_t numel
) {
    switch (type) {

    case LLAISYS_DTYPE_F32:
        return launchArgmax<float>(
            reinterpret_cast<std::int64_t *>(max_idx),
            reinterpret_cast<float *>(max_val),
            reinterpret_cast<const float *>(vals),
            numel
        );

    case LLAISYS_DTYPE_F16:
        return launchArgmax<__half>(
            reinterpret_cast<std::int64_t *>(max_idx),
            reinterpret_cast<__half *>(max_val),
            reinterpret_cast<const __half *>(vals),
            numel
        );

    case LLAISYS_DTYPE_BF16:
        return launchArgmax<__nv_bfloat16>(
            reinterpret_cast<std::int64_t *>(max_idx),
            reinterpret_cast<__nv_bfloat16 *>(max_val),
            reinterpret_cast<const __nv_bfloat16 *>(vals),
            numel
        );

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia