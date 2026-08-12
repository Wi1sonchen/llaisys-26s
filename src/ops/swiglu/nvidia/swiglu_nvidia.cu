#include "swiglu_nvidia.cuh"

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
__global__ void swigluKernel(
    T *out,
    const T *gate,
    const T *up,
    size_t numel
) {
    const size_t i =
        static_cast<size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;

    if (i >= numel) {
        return;
    }

    const float gate_val =
        CudaCast<T>::toFloat(gate[i]);

    const float up_val =
        CudaCast<T>::toFloat(up[i]);

    const float silu =
        gate_val / (1.0f + expf(-gate_val));

    const float result =
        up_val * silu;

    out[i] =
        CudaCast<T>::fromFloat(result);
}

template <typename T>
void launchSwiglu(
    T *out,
    const T *gate,
    const T *up,
    size_t numel
) {
    if (numel == 0) {
        return;
    }

    constexpr int BLOCK_SIZE = 256;

    const int grid_size =
        static_cast<int>(
            (numel + BLOCK_SIZE - 1) / BLOCK_SIZE
        );

    swigluKernel<T><<<grid_size, BLOCK_SIZE>>>(
        out,
        gate,
        up,
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

void swiglu(
    std::byte *out,
    const std::byte *gate,
    const std::byte *up,
    llaisysDataType_t type,
    size_t numel
) {
    switch (type) {

    case LLAISYS_DTYPE_F32:
        return launchSwiglu<float>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(gate),
            reinterpret_cast<const float *>(up),
            numel
        );

    case LLAISYS_DTYPE_F16:
        return launchSwiglu<__half>(
            reinterpret_cast<__half *>(out),
            reinterpret_cast<const __half *>(gate),
            reinterpret_cast<const __half *>(up),
            numel
        );

    case LLAISYS_DTYPE_BF16:
        return launchSwiglu<__nv_bfloat16>(
            reinterpret_cast<__nv_bfloat16 *>(out),
            reinterpret_cast<const __nv_bfloat16 *>(gate),
            reinterpret_cast<const __nv_bfloat16 *>(up),
            numel
        );

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia