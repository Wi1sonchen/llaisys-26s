#include "rms_norm_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>

namespace {

template <typename T>
struct CudaCast;

template <>
struct CudaCast<float> {
    __device__ static float toFloat(float value) {
        return value;
    }

    __device__ static float fromFloat(float value) {
        return value;
    }
};

template <>
struct CudaCast<__half> {
    __device__ static float toFloat(__half value) {
        return __half2float(value);
    }

    __device__ static __half fromFloat(float value) {
        return __float2half_rn(value);
    }
};

template <>
struct CudaCast<__nv_bfloat16> {
    __device__ static float toFloat(__nv_bfloat16 value) {
        return __bfloat162float(value);
    }

    __device__ static __nv_bfloat16 fromFloat(float value) {
        return __float2bfloat16_rn(value);
    }
};


template <typename T>
__global__ void rmsNormKernel(
    T *out,
    const T *in,
    const T *weight,
    size_t rows,
    size_t dim,
    float eps
) {
    const size_t row = blockIdx.x;

    if (row >= rows) {
        return;
    }

    constexpr int BLOCK_SIZE = 256;

    __shared__ float shared[BLOCK_SIZE];

    float local_sum = 0.0f;

    // 每个线程处理这一行中的若干元素
    for (
        size_t j = threadIdx.x;
        j < dim;
        j += blockDim.x
    ) {
        const float x =
            CudaCast<T>::toFloat(
                in[row * dim + j]
            );

        local_sum += x * x;
    }

    shared[threadIdx.x] = local_sum;

    __syncthreads();

    // block 内 reduction
    for (
        unsigned int stride = blockDim.x / 2;
        stride > 0;
        stride >>= 1
    ) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] +=
                shared[threadIdx.x + stride];
        }

        __syncthreads();
    }

    // thread 0 计算这一行的 inverse RMS
    if (threadIdx.x == 0) {
        const float mean_square =
            shared[0] /
            static_cast<float>(dim);

        shared[0] =
            rsqrtf(mean_square + eps);
    }

    __syncthreads();

    const float inv_rms = shared[0];

    // normalization + weight
    for (
        size_t j = threadIdx.x;
        j < dim;
        j += blockDim.x
    ) {
        const float x =
            CudaCast<T>::toFloat(
                in[row * dim + j]
            );

        const float w =
            CudaCast<T>::toFloat(
                weight[j]
            );

        const float value =
            x * inv_rms * w;

        out[row * dim + j] =
            CudaCast<T>::fromFloat(value);
    }
}


template <typename T>
void launchRmsNorm(
    T *out,
    const T *in,
    const T *weight,
    size_t rows,
    size_t dim,
    float eps
) {
    if (rows == 0 || dim == 0) {
        return;
    }

    constexpr int BLOCK_SIZE = 256;

    rmsNormKernel<T><<<rows, BLOCK_SIZE>>>(
        out,
        in,
        weight,
        rows,
        dim,
        eps
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

void rms_norm(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    llaisysDataType_t type,
    size_t rows,
    size_t dim,
    float eps
) {
    switch (type) {

    case LLAISYS_DTYPE_F32:
        return launchRmsNorm<float>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            rows,
            dim,
            eps
        );

    case LLAISYS_DTYPE_F16:
        return launchRmsNorm<__half>(
            reinterpret_cast<__half *>(out),
            reinterpret_cast<const __half *>(in),
            reinterpret_cast<const __half *>(weight),
            rows,
            dim,
            eps
        );

    case LLAISYS_DTYPE_BF16:
        return launchRmsNorm<__nv_bfloat16>(
            reinterpret_cast<__nv_bfloat16 *>(out),
            reinterpret_cast<const __nv_bfloat16 *>(in),
            reinterpret_cast<const __nv_bfloat16 *>(weight),
            rows,
            dim,
            eps
        );

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia