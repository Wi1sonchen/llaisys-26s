#pragma once

#include "llaisys/models/qwen2.h"

#include "../../tensor/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace llaisys::models {

struct Qwen2Weights {
    tensor_t in_embed;
    tensor_t out_embed;
    tensor_t out_norm_w;

    std::vector<tensor_t> attn_norm_w;
    std::vector<tensor_t> attn_q_w;
    std::vector<tensor_t> attn_q_b;
    std::vector<tensor_t> attn_k_w;
    std::vector<tensor_t> attn_k_b;
    std::vector<tensor_t> attn_v_w;
    std::vector<tensor_t> attn_v_b;
    std::vector<tensor_t> attn_o_w;

    std::vector<tensor_t> mlp_norm_w;
    std::vector<tensor_t> mlp_gate_w;
    std::vector<tensor_t> mlp_up_w;
    std::vector<tensor_t> mlp_down_w;
};

class Qwen2Model {
public:
    Qwen2Model(const LlaisysQwen2Meta &meta,
               llaisysDeviceType_t device,
               int device_id);

    Qwen2Model(const Qwen2Model &) = delete;
    Qwen2Model &operator=(const Qwen2Model &) = delete;

    const LlaisysQwen2Meta &meta() const { return _meta; }
    const Qwen2Weights &weights() const { return _weights; }

    tensor_t weight(const std::string &name) const;
    bool loadWeight(const std::string &name, const void *data, size_t nbytes);
    bool allWeightsLoaded() const;

    void resetCache();
    size_t cacheLength() const { return _cache_len; }

    int64_t infer(const int64_t *token_ids, size_t ntoken);

private:
    tensor_t makeTensor(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype) const;
    tensor_t makeTensor(const std::vector<size_t> &shape) const;

    void registerWeights();
    void ensureCacheCapacity(size_t required);
    void copyTensorData(const tensor_t &dst, const tensor_t &src) const;
    void copyRawSameDevice(void *dst, const void *src, size_t bytes) const;
    int64_t readI64(const tensor_t &tensor) const;

    LlaisysQwen2Meta _meta;
    llaisysDeviceType_t _device;
    int _device_id;

    Qwen2Weights _weights;
    std::unordered_map<std::string, tensor_t> _weight_map;
    std::unordered_set<std::string> _loaded_weights;

    std::vector<tensor_t> _k_cache;
    std::vector<tensor_t> _v_cache;
    size_t _cache_len{0};
    size_t _cache_capacity{0};
};

} // namespace llaisys::models
