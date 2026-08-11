#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);

    CHECK_SAME_DTYPE(
        attn_val->dtype(),
        q->dtype(),
        k->dtype(),
        v->dtype()
    );

    ASSERT(
        attn_val->isContiguous() &&
        q->isContiguous() &&
        k->isContiguous() &&
        v->isContiguous(),
        "SelfAttention: all tensors must be contiguous."
    );

    ASSERT(
        attn_val->shape().size() == 3 &&
        q->shape().size() == 3 &&
        k->shape().size() == 3 &&
        v->shape().size() == 3,
        "SelfAttention: all tensors must be 3D."
    );

    const size_t seqlen = q->shape()[0];
    const size_t nhead = q->shape()[1];
    const size_t d = q->shape()[2];

    const size_t total_len = k->shape()[0];
    const size_t nkvhead = k->shape()[1];

    const size_t dv = v->shape()[2];

    ASSERT(
        k->shape()[2] == d,
        "SelfAttention: q and k head dimensions must match."
    );

    ASSERT(
        v->shape()[0] == total_len,
        "SelfAttention: k and v sequence lengths must match."
    );

    ASSERT(
        v->shape()[1] == nkvhead,
        "SelfAttention: k and v must have the same number of KV heads."
    );

    ASSERT(
        nhead % nkvhead == 0,
        "SelfAttention: nhead must be divisible by nkvhead."
    );

    ASSERT(
        total_len >= seqlen,
        "SelfAttention: total_len must be >= seqlen."
    );

    ASSERT(
        attn_val->shape()[0] == seqlen &&
        attn_val->shape()[1] == nhead &&
        attn_val->shape()[2] == dv,
        "SelfAttention: invalid output shape."
    );

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(
            attn_val->data(),
            q->data(),
            k->data(),
            v->data(),
            attn_val->dtype(),
            seqlen,
            total_len,
            nhead,
            nkvhead,
            d,
            dv,
            scale
        );
    }

    llaisys::core::context().setDevice(
        attn_val->deviceType(),
        attn_val->deviceId()
    );

    switch (attn_val->deviceType()) {

    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(
            attn_val->data(),
            q->data(),
            k->data(),
            v->data(),
            attn_val->dtype(),
            seqlen,
            total_len,
            nhead,
            nkvhead,
            d,
            dv,
            scale
        );

#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif

    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
