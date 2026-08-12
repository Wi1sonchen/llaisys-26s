#include "self_attention_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

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
    __device__ static float toFloat(
        __nv_bfloat16 value
    ) {
        return __bfloat162float(value);
    }

    __device__ static __nv_bfloat16 fromFloat(
        float value
    ) {
        return __float2bfloat16_rn(value);
    }
};


template <typename T>
__global__ void selfAttentionKernel(
    T *attn_val,
    const T *q,
    const T *k,
    const T *v,
    size_t seqlen,
    size_t total_len,
    size_t nhead,
    size_t nkvhead,
    size_t d,
    size_t dv,
    float scale
) {
    constexpr int BLOCK_SIZE = 256;

    // 每个 block 对应一个：
    //
    // (query_idx, query_head)
    //
    const size_t block_idx =
        static_cast<size_t>(blockIdx.x);

    const size_t query_idx =
        block_idx / nhead;

    const size_t head =
        block_idx % nhead;

    if (query_idx >= seqlen) {
        return;
    }

    // ---------------------------------------------
    // GQA / MQA:
    //
    // 例如：
    // nhead   = 12
    // nkvhead = 2
    //
    // Q heads 0~5  -> KV head 0
    // Q heads 6~11 -> KV head 1
    // ---------------------------------------------
    const size_t heads_per_kv =
        nhead / nkvhead;

    const size_t kv_head =
        head / heads_per_kv;


    // ---------------------------------------------
    // KV cache / causal position
    //
    // 例如：
    // total_len = 10
    // seqlen    = 1
    //
    // 当前 query 对应 absolute position 9
    //
    // prefill:
    // total_len == seqlen
    // query_pos == query_idx
    // ---------------------------------------------
    const size_t query_pos =
        total_len - seqlen + query_idx;


    // 动态 shared memory：
    //
    // scores[0 ... total_len-1]
    //
    extern __shared__ float scores[];

    // 用于 block reduction
    __shared__ float reduction[BLOCK_SIZE];


    // =========================================================
    // 1. Q K^T * scale
    // =========================================================

    float local_max = -CUDART_INF_F;

    // 每个线程负责若干 key position
    for (
        size_t t = threadIdx.x;
        t <= query_pos;
        t += blockDim.x
    ) {
        float score = 0.0f;

        // q[query_idx, head, x]
        const size_t q_base =
            (query_idx * nhead + head) * d;

        // k[t, kv_head, x]
        const size_t k_base =
            (t * nkvhead + kv_head) * d;

        for (size_t x = 0; x < d; x++) {
            const float q_val =
                CudaCast<T>::toFloat(
                    q[q_base + x]
                );

            const float k_val =
                CudaCast<T>::toFloat(
                    k[k_base + x]
                );

            score += q_val * k_val;
        }

        score *= scale;

        scores[t] = score;

        local_max =
            fmaxf(local_max, score);
    }

    __syncthreads();


    // =========================================================
    // 2. softmax max reduction
    // =========================================================

    reduction[threadIdx.x] =
        local_max;

    __syncthreads();

    for (
        unsigned int stride =
            blockDim.x / 2;
        stride > 0;
        stride >>= 1
    ) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] =
                fmaxf(
                    reduction[threadIdx.x],
                    reduction[
                        threadIdx.x + stride
                    ]
                );
        }

        __syncthreads();
    }

    const float max_score =
        reduction[0];


    // =========================================================
    // 3. exp(score - max) + sum
    // =========================================================

    float local_sum = 0.0f;

    for (
        size_t t = threadIdx.x;
        t <= query_pos;
        t += blockDim.x
    ) {
        const float exp_score =
            expf(
                scores[t] - max_score
            );

        scores[t] = exp_score;

        local_sum += exp_score;
    }

    reduction[threadIdx.x] =
        local_sum;

    __syncthreads();


    // =========================================================
    // 4. softmax sum reduction
    // =========================================================

    for (
        unsigned int stride =
            blockDim.x / 2;
        stride > 0;
        stride >>= 1
    ) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] +=
                reduction[
                    threadIdx.x + stride
                ];
        }

        __syncthreads();
    }

    const float sum =
        reduction[0];


    // =========================================================
    // 5. normalize softmax
    // =========================================================

    for (
        size_t t = threadIdx.x;
        t <= query_pos;
        t += blockDim.x
    ) {
        scores[t] /= sum;
    }

    // 很重要：
    //
    // 接下来每个输出线程会读取所有 scores[t]，
    // 必须确保整个 block 都已经 normalize 完成。
    __syncthreads();


    // =========================================================
    // 6. Attention × V
    // =========================================================

    for (
        size_t x = threadIdx.x;
        x < dv;
        x += blockDim.x
    ) {
        float result = 0.0f;

        for (
            size_t t = 0;
            t <= query_pos;
            t++
        ) {
            const size_t v_offset =
                (t * nkvhead + kv_head)
                    * dv
                + x;

            const float v_val =
                CudaCast<T>::toFloat(
                    v[v_offset]
                );

            result +=
                scores[t] * v_val;
        }

        const size_t out_offset =
            (query_idx * nhead + head)
                * dv
            + x;

        attn_val[out_offset] =
            CudaCast<T>::fromFloat(
                result
            );
    }
}


template <typename T>
void launchSelfAttention(
    T *attn_val,
    const T *q,
    const T *k,
    const T *v,
    size_t seqlen,
    size_t total_len,
    size_t nhead,
    size_t nkvhead,
    size_t d,
    size_t dv,
    float scale
) {
    if (
        seqlen == 0 ||
        total_len == 0 ||
        nhead == 0 ||
        nkvhead == 0 ||
        d == 0 ||
        dv == 0
    ) {
        return;
    }

    constexpr int BLOCK_SIZE = 256;

    const size_t blocks =
        seqlen * nhead;

    // 每个 block 需要：
    //
    // total_len 个 float score
    //
    const size_t shared_bytes =
        total_len * sizeof(float);

    selfAttentionKernel<T>
        <<<blocks, BLOCK_SIZE, shared_bytes>>>(
            attn_val,
            q,
            k,
            v,
            seqlen,
            total_len,
            nhead,
            nkvhead,
            d,
            dv,
            scale
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

void self_attention(
    std::byte *attn_val,
    const std::byte *q,
    const std::byte *k,
    const std::byte *v,
    llaisysDataType_t type,
    size_t seqlen,
    size_t total_len,
    size_t nhead,
    size_t nkvhead,
    size_t d,
    size_t dv,
    float scale
) {
    switch (type) {

    case LLAISYS_DTYPE_F32:
        return launchSelfAttention<float>(
            reinterpret_cast<float *>(
                attn_val
            ),

            reinterpret_cast<
                const float *
            >(q),

            reinterpret_cast<
                const float *
            >(k),

            reinterpret_cast<
                const float *
            >(v),

            seqlen,
            total_len,
            nhead,
            nkvhead,
            d,
            dv,
            scale
        );


    case LLAISYS_DTYPE_F16:
        return launchSelfAttention<__half>(
            reinterpret_cast<__half *>(
                attn_val
            ),

            reinterpret_cast<
                const __half *
            >(q),

            reinterpret_cast<
                const __half *
            >(k),

            reinterpret_cast<
                const __half *
            >(v),

            seqlen,
            total_len,
            nhead,
            nkvhead,
            d,
            dv,
            scale
        );


    case LLAISYS_DTYPE_BF16:
        return launchSelfAttention<
            __nv_bfloat16
        >(
            reinterpret_cast<
                __nv_bfloat16 *
            >(attn_val),

            reinterpret_cast<
                const __nv_bfloat16 *
            >(q),

            reinterpret_cast<
                const __nv_bfloat16 *
            >(k),

            reinterpret_cast<
                const __nv_bfloat16 *
            >(v),

            seqlen,
            total_len,
            nhead,
            nkvhead,
            d,
            dv,
            scale
        );


    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia