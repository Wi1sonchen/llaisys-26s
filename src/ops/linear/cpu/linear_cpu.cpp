#include "linear_cpu.hpp"

#include "../../../utils.hpp"

template <typename T>
void linear_(
    T *out,
    const T *in,
    const T *weight,
    const T *bias,
    size_t M,
    size_t N,
    size_t K
) {
    // TODO:
    // i 遍历 M 行
    //
    // j 遍历 N 个输出特征
    //
    // 用 float acc 做累加
    //
    // k 遍历 K：
    //
    //   in[i * K + k]
    //   weight[j * K + k]
    //
    // 都 cast<float> 后相乘并累加
    //
    // 如果 bias != nullptr：
    //   加上 bias[j]
    //
    // 最后把 acc cast<T> 写到 out[i * N + j]
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float acc = 0.0f;
            for (size_t k = 0; k < K; k++) {
                acc += llaisys::utils::cast<float>(in[i * K + k]) * llaisys::utils::cast<float>(weight[j * K + k]);
            }
            if (bias != nullptr) {
                acc += llaisys::utils::cast<float>(bias[j]);
            }
            out[i * N + j] = llaisys::utils::cast<T>(acc);
        }
    }
}

namespace llaisys::ops::cpu {
void linear(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    const std::byte *bias,
    llaisysDataType_t type,
    size_t M,
    size_t N,
    size_t K
) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                       reinterpret_cast<const float *>(weight), bias == nullptr ? nullptr : reinterpret_cast<const float *>(bias), M, N, K);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                       reinterpret_cast<const llaisys::bf16_t *>(weight), bias == nullptr ? nullptr : reinterpret_cast<const llaisys::bf16_t *>(bias), M, N, K);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                       reinterpret_cast<const llaisys::fp16_t *>(weight), bias == nullptr ? nullptr : reinterpret_cast<const llaisys::fp16_t *>(bias), M, N, K);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu