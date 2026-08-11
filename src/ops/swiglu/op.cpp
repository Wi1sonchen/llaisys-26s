#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/swiglu_cpu.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);

    CHECK_SAME_DTYPE(
        out->dtype(),
        gate->dtype(),
        up->dtype()
    );

    ASSERT(
        out->shape().size() == 2 &&
        gate->shape().size() == 2 &&
        up->shape().size() == 2,
        "SwiGLU: all tensors must be 2D."
    );

    ASSERT(
        out->shape()[0] == gate->shape()[0] &&
        out->shape()[1] == gate->shape()[1] &&
        out->shape()[0] == up->shape()[0] &&
        out->shape()[1] == up->shape()[1],
        "SwiGLU: out, gate and up must have the same shape."
    );

    ASSERT(
        out->isContiguous() &&
        gate->isContiguous() &&
        up->isContiguous(),
        "SwiGLU: all tensors must be contiguous."
    );

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(
            out->data(),
            gate->data(),
            up->data(),
            out->dtype(),
            out->numel()
        );
    }

    llaisys::core::context().setDevice(
        out->deviceType(),
        out->deviceId()
    );

    switch (out->deviceType()) {

    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(
            out->data(),
            gate->data(),
            up->data(),
            out->dtype(),
            out->numel()
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
