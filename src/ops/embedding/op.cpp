#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), "Embedding: all tensors must be contiguous.");
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index tensor must be of type int64.");
    ASSERT(out->shape().size() == 2 && index->shape().size() == 1 && weight->shape().size() == 2,
           "Embedding: output must be 2D, index must be 1D, and weight must be 2D.");
    ASSERT(out->shape()[0] == index->shape()[0], "Embedding: output's first dimension must match index's size.");
    ASSERT(out->shape()[1] == weight->shape()[1], "Embedding: output's second dimension must match weight's second dimension.");

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), out->dtype(), index->shape()[0], weight->shape()[1]);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), out->dtype(), index->shape()[0], weight->shape()[1]);
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
