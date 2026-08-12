#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_nvidia.cuh"
#endif

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "Linear: out, in and weight must be contiguous.");   
    ASSERT(out->shape().size() == 2 && in->shape().size() == 2 && weight->shape().size() == 2,
           "Linear: all tensors must be 2D.");
    ASSERT(out->shape()[0] == in->shape()[0], "Linear: out and in must have the same batch size.");
    ASSERT(out->shape()[1] == weight->shape()[0], "Linear: out and weight must have the same output feature size.");
    ASSERT(in->shape()[1] == weight->shape()[1], "Linear: in and weight must have the same input feature size.");
    if (bias != nullptr) {
        CHECK_SAME_DEVICE(out, bias);
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
        ASSERT(bias->shape().size() == 1, "Linear: bias must be 1D.");
        ASSERT(bias->shape()[0] == out->shape()[1], "Linear: bias must have the same size as out's output feature size.");
    }

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr, out->dtype(),
                           out->shape()[0], out->shape()[1], in->shape()[1]);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr, out->dtype(),
                           out->shape()[0], out->shape()[1], in->shape()[1]);

#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::linear(
        out->data(),
        in->data(),
        weight->data(),
        bias ? bias->data() : nullptr,
        out->dtype(),
        out->shape()[0],
        out->shape()[1],
        in->shape()[1]
    );
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
