#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <limits>
#include <vector>

template <typename T>
void self_attention_(
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
    const size_t heads_per_kv = nhead / nkvhead;

    // 用来暂存一个 query/head 对所有 key 的 attention score
    std::vector<float> scores(total_len);

    for (size_t i = 0; i < seqlen; i++) {
        // 当前 query 在整个 context 中的绝对位置。
        // total_len == seqlen 时就是 i；
        // 有 KV cache 时，query 对应最后 seqlen 个 token。
        const size_t query_pos = total_len - seqlen + i;

        for (size_t h = 0; h < nhead; h++) {
            const size_t kv_head = h / heads_per_kv;

            float max_score =
                -std::numeric_limits<float>::infinity();

            // ------------------------------------------------
            // 1. Q K^T * scale
            // ------------------------------------------------
            // causal mask 后，只计算 [0, query_pos]
            for (size_t t = 0; t <= query_pos; t++) {
                float score = 0.0f;

                for (size_t x = 0; x < d; x++) {
                    // q[i, h, x]
                    const size_t q_offset =
                        (i * nhead + h) * d + x;

                    // k[t, kv_head, x]
                    const size_t k_offset =
                        (t * nkvhead + kv_head) * d + x;

                    const float q_val =
                        llaisys::utils::cast<float>(
                            q[q_offset]
                        );

                    const float k_val =
                        llaisys::utils::cast<float>(
                            k[k_offset]
                        );

                    score += q_val * k_val;
                }

                score *= scale;

                scores[t] = score;

                // softmax 数值稳定需要最大值
                if (score > max_score) {
                    max_score = score;
                }
            }

            // ------------------------------------------------
            // 2. stable softmax
            // ------------------------------------------------
            float exp_sum = 0.0f;

            for (size_t t = 0; t <= query_pos; t++) {
                scores[t] =
                    std::exp(scores[t] - max_score);

                exp_sum += scores[t];
            }

            for (size_t t = 0; t <= query_pos; t++) {
                scores[t] /= exp_sum;
            }

            // ------------------------------------------------
            // 3. softmax(A) V
            // ------------------------------------------------
            for (size_t x = 0; x < dv; x++) {
                float result = 0.0f;

                for (size_t t = 0; t <= query_pos; t++) {
                    // v[t, kv_head, x]
                    const size_t v_offset =
                        (t * nkvhead + kv_head) * dv + x;

                    const float v_val =
                        llaisys::utils::cast<float>(
                            v[v_offset]
                        );

                    result += scores[t] * v_val;
                }

                // attn_val[i, h, x]
                const size_t out_offset =
                    (i * nhead + h) * dv + x;

                attn_val[out_offset] =
                    llaisys::utils::cast<T>(result);
            }
        }
    }
}

namespace llaisys::ops::cpu {

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
        return self_attention_<float>(
            reinterpret_cast<float *>(attn_val),
            reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k),
            reinterpret_cast<const float *>(v),
            seqlen,
            total_len,
            nhead,
            nkvhead,
            d,
            dv,
            scale
        );

    case LLAISYS_DTYPE_F16:
        return self_attention_<llaisys::fp16_t>(
            reinterpret_cast<llaisys::fp16_t *>(attn_val),
            reinterpret_cast<const llaisys::fp16_t *>(q),
            reinterpret_cast<const llaisys::fp16_t *>(k),
            reinterpret_cast<const llaisys::fp16_t *>(v),
            seqlen,
            total_len,
            nhead,
            nkvhead,
            d,
            dv,
            scale
        );

    case LLAISYS_DTYPE_BF16:
        return self_attention_<llaisys::bf16_t>(
            reinterpret_cast<llaisys::bf16_t *>(attn_val),
            reinterpret_cast<const llaisys::bf16_t *>(q),
            reinterpret_cast<const llaisys::bf16_t *>(k),
            reinterpret_cast<const llaisys::bf16_t *>(v),
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

} // namespace llaisys::ops::cpu