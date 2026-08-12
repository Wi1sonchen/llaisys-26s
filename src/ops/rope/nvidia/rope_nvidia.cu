#include "rope_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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
__global__ void ropeKernel(
    T *out,
    const T *in,
    const std::int64_t *pos_ids,
    size_t seqlen,
    size_t nhead,
    size_t dim,
    float theta
) {
    const size_t half_dim = dim / 2;

    // 一个 thread 负责一组 (a_j, b_j)
    const size_t pair_idx =
        static_cast<size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;

    const size_t total_pairs =
        seqlen * nhead * half_dim;

    if (pair_idx >= total_pairs) {
        return;
    }

    // pair_idx -> [seq, head, j]
    const size_t j =
        pair_idx % half_dim;

    const size_t temp =
        pair_idx / half_dim;

    const size_t h =
        temp % nhead;

    const size_t seq =
        temp / nhead;

    const size_t base =
        (seq * nhead + h) * dim;

    const float pos =
        static_cast<float>(pos_ids[seq]);

    const float exponent =
        2.0f * static_cast<float>(j)
        / static_cast<float>(dim);

    const float phi =
        pos / powf(theta, exponent);

    float sin_phi;
    float cos_phi;

    sincosf(
        phi,
        &sin_phi,
        &cos_phi
    );

    const float a =
        CudaCast<T>::toFloat(
            in[base + j]
        );

    const float b =
        CudaCast<T>::toFloat(
            in[base + half_dim + j]
        );

    const float a_out =
        a * cos_phi - b * sin_phi;

    const float b_out =
        b * cos_phi + a * sin_phi;

    out[base + j] =
        CudaCast<T>::fromFloat(a_out);

    out[base + half_dim + j] =
        CudaCast<T>::fromFloat(b_out);
}


template <typename T>
void launchRope(
    T *out,
    const T *in,
    const std::int64_t *pos_ids,
    size_t seqlen,
    size_t nhead,
    size_t dim,
    float theta
) {
    if (seqlen == 0 || nhead == 0 || dim == 0) {
        return;
    }

    const size_t half_dim = dim / 2;

    const size_t total_pairs =
        seqlen * nhead * half_dim;

    constexpr int BLOCK_SIZE = 256;

    const int grid_size =
        static_cast<int>(
            (total_pairs + BLOCK_SIZE - 1)
            / BLOCK_SIZE
        );

    ropeKernel<T><<<grid_size, BLOCK_SIZE>>>(
        out,
        in,
        pos_ids,
        seqlen,
        nhead,
        dim,
        theta
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

void rope(
    std::byte *out,
    const std::byte *in,
    const std::byte *pos_ids,
    llaisysDataType_t type,
    size_t seqlen,
    size_t nhead,
    size_t dim,
    float theta
) {
    switch (type) {

    case LLAISYS_DTYPE_F32:
        return launchRope<float>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const std::int64_t *>(pos_ids),
            seqlen,
            nhead,
            dim,
            theta
        );

    case LLAISYS_DTYPE_F16:
        return launchRope<__half>(
            reinterpret_cast<__half *>(out),
            reinterpret_cast<const __half *>(in),
            reinterpret_cast<const std::int64_t *>(pos_ids),
            seqlen,
            nhead,
            dim,
            theta
        );

    case LLAISYS_DTYPE_BF16:
        return launchRope<__nv_bfloat16>(
            reinterpret_cast<__nv_bfloat16 *>(out),
            reinterpret_cast<const __nv_bfloat16 *>(in),
            reinterpret_cast<const std::int64_t *>(pos_ids),
            seqlen,
            nhead,
            dim,
            theta
        );

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia