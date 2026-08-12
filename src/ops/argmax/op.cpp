#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/argmax_nvidia.cuh"
#endif

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    // TODO 1:
    // 三个 tensor 必须在同一个 device 上
    //
    // 可以参考 add:
    // CHECK_SAME_DEVICE(...)
    CHECK_SAME_DEVICE(max_idx, max_val, vals);


    // TODO 2:
    // 检查 vals 和 max_val 的 dtype 一致
    //
    // 注意：
    // 不要把 max_idx 也放进 CHECK_SAME_DTYPE，
    // 因为 max_idx 固定是 i64，
    // 而 vals/max_val 是 f32/f16/bf16。
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());


    // TODO 3:
    // 单独检查 max_idx 的 dtype 是不是 I64。
    //
    // 去 utils.hpp / 其他算子搜索 ASSERT(dtype == ...)
    // 看仓库已有的写法。
    ASSERT(max_idx->dtype() == LLAISYS_DTYPE_I64, "argmax: max_idx must be I64.");


    // TODO 4:
    // 检查 shape。
    //
    // 题目告诉你目前可以假设：
    //
    // vals     是 1-D
    // max_idx  是 [1]
    // max_val  是 [1]
    //
    // 注意这里不能像 add 那样：
    //
    // CHECK_SAME_SHAPE(...)
    //
    // 因为 vals 和两个输出本来就不是同 shape。
    ASSERT(vals->ndim() == 1, "argmax: vals must be 1-D.");
    ASSERT(vals->numel() > 0, "argmax: vals must have at least one element.");
    ASSERT(max_idx->ndim() == 1 && max_idx->shape()[0] == 1, "argmax: max_idx must be [1].");
    ASSERT(max_val->ndim() == 1 && max_val->shape()[0] == 1, "argmax: max_val must be [1].");




    // TODO 5:
    // 检查 contiguous。
    //
    // 三个 tensor 当前都应该是 contiguous。
    ASSERT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(), "argmax: all tensors must be contiguous.");


    // TODO 6:
    // CPU 情况下调用：
    //
    // cpu::argmax(
    //     max_idx->data(),
    //     max_val->data(),
    //     vals->data(),
    //     vals->dtype(),
    //     vals->numel()
    // );
    //
    // 参数顺序由你在 argmax_cpu.hpp 中声明的接口决定。
    if (max_idx->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(
            max_idx->data(),
            max_val->data(),
            vals->data(),
            vals->dtype(),
            vals->numel()
        );
    }

    llaisys::core::context().setDevice(max_idx->deviceType(), max_idx->deviceId());


    // TODO 7:
    // 后面的 device dispatch 基本可以参考 add。
    //
    // NVIDIA 当前仍然可以 TO_BE_IMPLEMENTED()
    switch (max_idx->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(
            max_idx->data(),
            max_val->data(),
            vals->data(),
            vals->dtype(),
            vals->numel()
        );
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::argmax(
        max_idx->data(),
        max_val->data(),
        vals->data(),
        vals->dtype(),
        vals->numel()
    );
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;   
    }
}
} // namespace llaisys::ops
