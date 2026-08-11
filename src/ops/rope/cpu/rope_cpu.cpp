#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstdint>

template <typename T>
void rope_(
    T *out,
    const T *in,
    const std::int64_t *pos_ids,
    size_t seqlen,
    size_t nhead,
    size_t dim,
    float theta
) {
    const size_t half_dim = dim / 2;

    for (size_t i = 0; i < seqlen; i++) {
        const float pos = static_cast<float>(pos_ids[i]);

        for (size_t h = 0; h < nhead; h++) {
            const size_t base = (i * nhead + h) * dim;

            for (size_t j = 0; j < half_dim; j++) {
                // phi = pos / theta^(2j/d)
                const float exponent =
                    2.0f * static_cast<float>(j) /
                    static_cast<float>(dim);

                const float phi =
                    pos / std::pow(theta, exponent);

                const float cos_phi = std::cos(phi);
                const float sin_phi = std::sin(phi);

                // 输入向量 [a, b]
                const float a =
                    llaisys::utils::cast<float>(
                        in[base + j]
                    );

                const float b =
                    llaisys::utils::cast<float>(
                        in[base + half_dim + j]
                    );

                // a' = a cos(phi) - b sin(phi)
                // b' = b cos(phi) + a sin(phi)
                const float a_out =
                    a * cos_phi - b * sin_phi;

                const float b_out =
                    b * cos_phi + a * sin_phi;

                out[base + j] =
                    llaisys::utils::cast<T>(a_out);

                out[base + half_dim + j] =
                    llaisys::utils::cast<T>(b_out);
            }
        }
    }
}

namespace llaisys::ops::cpu {

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
        return rope_<float>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const std::int64_t *>(pos_ids),
            seqlen,
            nhead,
            dim,
            theta
        );

    case LLAISYS_DTYPE_F16:
        return rope_<llaisys::fp16_t>(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const std::int64_t *>(pos_ids),
            seqlen,
            nhead,
            dim,
            theta
        );

    case LLAISYS_DTYPE_BF16:
        return rope_<llaisys::bf16_t>(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
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

} // namespace llaisys::ops::cpu