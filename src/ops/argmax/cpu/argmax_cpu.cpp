#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstdint>
#include <type_traits>

template <typename T>
void argmax_(std::int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    // TODO 1:
    // 保存当前最大值的下标。
    // 从第 0 个元素开始，而不是虚构一个最大值 0。
    std::int64_t best_idx = 0;
    float best_val = llaisys::utils::cast<float>(vals[0]);


    // TODO 2:
    // 保存 vals[0] 对应的、用于“比较”的值。
    //
    float current = llaisys::utils::cast<float>(vals[0]);
    // 提示：
    // utils::cast<float>(...) 是你刚才在 add 中看到的东西。


    // TODO 3:
    // 从第 1 个元素开始遍历。
    //
    // 每次：
    //   current = 将 vals[i] 转成用于比较的值
    //
    //   如果 current 严格大于目前最大值：
    //       更新最大值
    //       更新最大值下标
    for (size_t i = 1; i < numel; i++) {
        current = llaisys::utils::cast<float>(vals[i]);
        if (current > best_val) {
            best_val = current;
            best_idx = static_cast<std::int64_t>(i);
        }
    }


    // TODO 4:
    // 把最终下标写进 max_idx[0]
    *max_idx = static_cast<std::int64_t>(best_idx);

    // TODO 5:
    // 把对应的原始元素写进 max_val[0]
    //
    // 可以思考为什么直接 vals[best_idx]
    // 比 float -> T 转换更自然。
    *max_val = vals[best_idx];
}

namespace llaisys::ops::cpu {
    void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
        // TODO:
        // max_idx 永远是 i64，
        // 所以这里需要把 byte pointer 转成正确的指针类型。

        // std::int64_t *max_idx_ptr = reinterpret_cast<std::int64_t *>(max_idx);

        switch (type) {

        case LLAISYS_DTYPE_F32:
            // TODO:
            // max_val -> float *
            // vals    -> const float *
            // 调用 argmax_<float>
            return argmax_<float>(
                reinterpret_cast<std::int64_t *>(max_idx),
                reinterpret_cast<float *>(max_val),
                reinterpret_cast<const float *>(vals),
                numel
            );
        case LLAISYS_DTYPE_BF16:
            // TODO:
            // 对应 bf16_t
            return argmax_<llaisys::bf16_t>(
                reinterpret_cast<std::int64_t *>(max_idx),
                reinterpret_cast<llaisys::bf16_t *>(max_val),
                reinterpret_cast<const llaisys::bf16_t *>(vals),
                numel
            );

        case LLAISYS_DTYPE_F16:
            // TODO:
            // 对应 fp16_t
            return argmax_<llaisys::fp16_t>(
                reinterpret_cast<std::int64_t *>(max_idx),
                reinterpret_cast<llaisys::fp16_t *>(max_val),
                reinterpret_cast<const llaisys::fp16_t *>(vals),
                numel
            );

        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }
    }

}