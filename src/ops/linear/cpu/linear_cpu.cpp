#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace {

template <typename T>
inline float to_float(T value) {
    if constexpr (std::is_same_v<T, float>) {
        return value;
    } else if constexpr (std::is_same_v<T, llaisys::bf16_t>) {
        // BF16 -> F32 is just placing the 16 payload bits in the high half.
        // Keep this conversion inline because it is on Linear's hottest path.
        const uint32_t bits = static_cast<uint32_t>(value._v) << 16;
        float out;
        std::memcpy(&out, &bits, sizeof(out));
        return out;
    } else {
        return llaisys::utils::cast<float>(value);
    }
}

template <typename T>
inline T from_float(float value) {
    if constexpr (std::is_same_v<T, float>) {
        return value;
    } else if constexpr (std::is_same_v<T, llaisys::bf16_t>) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        const uint32_t rounding_bias = 0x00007FFFu + ((bits >> 16) & 1u);
        return llaisys::bf16_t{
            static_cast<uint16_t>((bits + rounding_bias) >> 16)};
    } else {
        return llaisys::utils::cast<T>(value);
    }
}

template <typename T>
void linear_(
    T *out,
    const T *in,
    const T *weight,
    const T *bias,
    size_t M,
    size_t N,
    size_t K) {
    // Each output element is independent. Parallelizing this outer work keeps
    // the accumulation order inside every dot product deterministic while
    // making the 1-token decode path usable on multi-core CPUs.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (M * N >= 256)
#endif
    for (ptrdiff_t flat = 0; flat < static_cast<ptrdiff_t>(M * N); ++flat) {
        const size_t i = static_cast<size_t>(flat) / N;
        const size_t j = static_cast<size_t>(flat) % N;

        const T *x_row = in + i * K;
        const T *w_row = weight + j * K;

        float acc = 0.0f;
        for (size_t k = 0; k < K; ++k) {
            acc += to_float(x_row[k]) * to_float(w_row[k]);
        }

        if (bias != nullptr) {
            acc += to_float(bias[j]);
        }

        out[i * N + j] = from_float<T>(acc);
    }
}

} // namespace

namespace llaisys::ops::cpu {
void linear(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    const std::byte *bias,
    llaisysDataType_t type,
    size_t M,
    size_t N,
    size_t K) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out),
                       reinterpret_cast<const float *>(in),
                       reinterpret_cast<const float *>(weight),
                       bias == nullptr ? nullptr : reinterpret_cast<const float *>(bias),
                       M, N, K);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out),
                       reinterpret_cast<const llaisys::bf16_t *>(in),
                       reinterpret_cast<const llaisys::bf16_t *>(weight),
                       bias == nullptr ? nullptr : reinterpret_cast<const llaisys::bf16_t *>(bias),
                       M, N, K);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out),
                       reinterpret_cast<const llaisys::fp16_t *>(in),
                       reinterpret_cast<const llaisys::fp16_t *>(weight),
                       bias == nullptr ? nullptr : reinterpret_cast<const llaisys::fp16_t *>(bias),
                       M, N, K);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
