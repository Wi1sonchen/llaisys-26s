#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstdint>

template <typename T>
void embedding_(T *output, const std::int64_t *indices, const T *weight, size_t seqlen, size_t embedding_dim) {
    for (size_t i = 0; i < seqlen; i++) {
        size_t index = static_cast<size_t>(indices[i]);
        for (size_t j = 0; j < embedding_dim; j++) {
            output[i * embedding_dim + j] = weight[index * embedding_dim + j];
        }
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *output, const std::byte *indices, const std::byte *weight, llaisysDataType_t type, size_t seqlen, size_t embedding_dim) {
    // Implementation for CPU embedding operation
    switch(type) {
        case LLAISYS_DTYPE_F32:
            return embedding_(reinterpret_cast<float*>(output), reinterpret_cast<const std::int64_t*>(indices), reinterpret_cast<const float*>(weight), seqlen, embedding_dim);
        case LLAISYS_DTYPE_F16:
            return embedding_<llaisys::fp16_t>(reinterpret_cast<llaisys::fp16_t*>(output), reinterpret_cast<const std::int64_t*>(indices), reinterpret_cast<const llaisys::fp16_t*>(weight), seqlen, embedding_dim);
        case LLAISYS_DTYPE_BF16:
            return embedding_<llaisys::bf16_t>(reinterpret_cast<llaisys::bf16_t*>(output), reinterpret_cast<const std::int64_t*>(indices), reinterpret_cast<const llaisys::bf16_t*>(weight), seqlen, embedding_dim);

        // Add more cases for other data types as needed
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu