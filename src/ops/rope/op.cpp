#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);

    CHECK_SAME_DTYPE(
        out->dtype(),
        in->dtype()
    );

    ASSERT(
        pos_ids->dtype() == LLAISYS_DTYPE_I64,
        "RoPE: pos_ids must be int64."
    );

    ASSERT(
        out->shape().size() == 3 &&
        in->shape().size() == 3,
        "RoPE: out and in must be 3D."
    );

    ASSERT(
        pos_ids->shape().size() == 1,
        "RoPE: pos_ids must be 1D."
    );

    ASSERT(
        out->shape() == in->shape(),
        "RoPE: out and in must have the same shape."
    );

    ASSERT(
        pos_ids->shape()[0] == in->shape()[0],
        "RoPE: pos_ids length must match seqlen."
    );

    ASSERT(
        in->shape()[2] % 2 == 0,
        "RoPE: head dimension must be even."
    );

    ASSERT(
        out->isContiguous() &&
        in->isContiguous() &&
        pos_ids->isContiguous(),
        "RoPE: all tensors must be contiguous."
    );

    const size_t seqlen = in->shape()[0];
    const size_t nhead = in->shape()[1];
    const size_t dim = in->shape()[2];

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(
            out->data(),
            in->data(),
            pos_ids->data(),
            out->dtype(),
            seqlen,
            nhead,
            dim,
            theta
        );
    }

    llaisys::core::context().setDevice(
        out->deviceType(),
        out->deviceId()
    );

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(
            out->data(),
            in->data(),
            pos_ids->data(),
            out->dtype(),
            seqlen,
            nhead,
            dim,
            theta
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
