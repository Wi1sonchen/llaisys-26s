#include "add_nvidia.cuh"

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
__global__ void addKernel(
    T *c,
    const T *a,
    const T *b,
    size_t numel
) {
    const size_t i =
        static_cast<size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;

    if (i >= numel) {
        return;
    }

    const float av = CudaCast<T>::toFloat(a[i]);
    const float bv = CudaCast<T>::toFloat(b[i]);

    c[i] = CudaCast<T>::fromFloat(av + bv);
}

template <typename T>
void launchAdd(
    T *c,
    const T *a,
    const T *b,
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

    addKernel<T><<<grid_size, BLOCK_SIZE>>>(
        c,
        a,
        b,
        numel
    );

    const cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        throw std::runtime_error(
            cudaGetErrorString(error)
        );
    }
}

} // namespace

namespace llaisys::ops::nvidia {

void add(
    std::byte *c,
    const std::byte *a,
    const std::byte *b,
    llaisysDataType_t type,
    size_t numel
) {
    switch (type) {

    case LLAISYS_DTYPE_F32:
        return launchAdd<float>(
            reinterpret_cast<float *>(c),
            reinterpret_cast<const float *>(a),
            reinterpret_cast<const float *>(b),
            numel
        );

    case LLAISYS_DTYPE_F16:
        return launchAdd<__half>(
            reinterpret_cast<__half *>(c),
            reinterpret_cast<const __half *>(a),
            reinterpret_cast<const __half *>(b),
            numel
        );

    case LLAISYS_DTYPE_BF16:
        return launchAdd<__nv_bfloat16>(
            reinterpret_cast<__nv_bfloat16 *>(c),
            reinterpret_cast<const __nv_bfloat16 *>(a),
            reinterpret_cast<const __nv_bfloat16 *>(b),
            numel
        );

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia