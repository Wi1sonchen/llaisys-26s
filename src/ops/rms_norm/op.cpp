#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(
        out->dtype(),
        in->dtype(),
        weight->dtype()
    );

    ASSERT(
        out->isContiguous() &&
        in->isContiguous() &&
        weight->isContiguous(),
        "RMSNorm: all tensors must be contiguous."
    );

    ASSERT(
        out->shape().size() == 2 &&
        in->shape().size() == 2,
        "RMSNorm: out and in must be 2D."
    );

    ASSERT(
        weight->shape().size() == 1,
        "RMSNorm: weight must be 1D."
    );

    ASSERT(
        out->shape()[0] == in->shape()[0] &&
        out->shape()[1] == in->shape()[1],
        "RMSNorm: out and in must have the same shape."
    );

    ASSERT(
        weight->shape()[0] == in->shape()[1],
        "RMSNorm: weight size must match the last dimension of input."
    );

    size_t rows = in->shape()[0];
    size_t dim = in->shape()[1];

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(
            out->data(),
            in->data(),
            weight->data(),
            out->dtype(),
            rows,
            dim,
            eps
        );
    }

    llaisys::core::context().setDevice(
        out->deviceType(),
        out->deviceId()
    );

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(
            out->data(),
            in->data(),
            weight->data(),
            out->dtype(),
            rows,
            dim,
            eps
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
