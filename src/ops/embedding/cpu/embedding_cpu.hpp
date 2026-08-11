#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void embedding(std::byte *output, const std::byte *indices, const std::byte *weight, llaisysDataType_t type, size_t seqlen, size_t embedding_dim);
}