#include "qwen2.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace llaisys::models {

namespace {

std::string layerName(size_t layer, const char *suffix) {
    std::ostringstream oss;
    oss << "model.layers." << layer << suffix;
    return oss.str();
}

} // namespace

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta &meta,
                       llaisysDeviceType_t device,
                       int device_id)
    : _meta(meta), _device(device), _device_id(device_id) {
    if (_meta.nlayer == 0 || _meta.hs == 0 || _meta.nh == 0 ||
        _meta.nkvh == 0 || _meta.dh == 0 || _meta.di == 0 ||
        _meta.maxseq == 0 || _meta.voc == 0) {
        throw std::runtime_error("Qwen2: invalid model metadata.");
    }
    if (_meta.nh % _meta.nkvh != 0) {
        throw std::runtime_error("Qwen2: num_attention_heads must be divisible by num_key_value_heads.");
    }
    if (_meta.nh * _meta.dh != _meta.hs) {
        throw std::runtime_error("Qwen2: hidden_size must equal num_attention_heads * head_dim.");
    }
    if (_meta.dtype != LLAISYS_DTYPE_F32 &&
        _meta.dtype != LLAISYS_DTYPE_F16 &&
        _meta.dtype != LLAISYS_DTYPE_BF16) {
        throw std::runtime_error("Qwen2: unsupported model dtype.");
    }

    // Model weights.
    _weights.in_embed = makeTensor({_meta.voc, _meta.hs});
    _weights.out_embed = makeTensor({_meta.voc, _meta.hs});
    _weights.out_norm_w = makeTensor({_meta.hs});

    _weights.attn_norm_w.resize(_meta.nlayer);
    _weights.attn_q_w.resize(_meta.nlayer);
    _weights.attn_q_b.resize(_meta.nlayer);
    _weights.attn_k_w.resize(_meta.nlayer);
    _weights.attn_k_b.resize(_meta.nlayer);
    _weights.attn_v_w.resize(_meta.nlayer);
    _weights.attn_v_b.resize(_meta.nlayer);
    _weights.attn_o_w.resize(_meta.nlayer);
    _weights.mlp_norm_w.resize(_meta.nlayer);
    _weights.mlp_gate_w.resize(_meta.nlayer);
    _weights.mlp_up_w.resize(_meta.nlayer);
    _weights.mlp_down_w.resize(_meta.nlayer);

    const size_t qdim = _meta.nh * _meta.dh;
    const size_t kvdim = _meta.nkvh * _meta.dh;

    for (size_t i = 0; i < _meta.nlayer; ++i) {
        _weights.attn_norm_w[i] = makeTensor({_meta.hs});
        _weights.attn_q_w[i] = makeTensor({qdim, _meta.hs});
        _weights.attn_q_b[i] = makeTensor({qdim});
        _weights.attn_k_w[i] = makeTensor({kvdim, _meta.hs});
        _weights.attn_k_b[i] = makeTensor({kvdim});
        _weights.attn_v_w[i] = makeTensor({kvdim, _meta.hs});
        _weights.attn_v_b[i] = makeTensor({kvdim});
        _weights.attn_o_w[i] = makeTensor({_meta.hs, qdim});

        _weights.mlp_norm_w[i] = makeTensor({_meta.hs});
        _weights.mlp_gate_w[i] = makeTensor({_meta.di, _meta.hs});
        _weights.mlp_up_w[i] = makeTensor({_meta.di, _meta.hs});
        _weights.mlp_down_w[i] = makeTensor({_meta.hs, _meta.di});
    }

    registerWeights();

    _k_cache.resize(_meta.nlayer);
    _v_cache.resize(_meta.nlayer);
}

tensor_t Qwen2Model::makeTensor(const std::vector<size_t> &shape,
                                llaisysDataType_t dtype) const {
    return Tensor::create(shape, dtype, _device, _device_id);
}

tensor_t Qwen2Model::makeTensor(const std::vector<size_t> &shape) const {
    return makeTensor(shape, _meta.dtype);
}

void Qwen2Model::registerWeights() {
    _weight_map.reserve(3 + 12 * _meta.nlayer);

    _weight_map.emplace("model.embed_tokens.weight", _weights.in_embed);
    _weight_map.emplace("lm_head.weight", _weights.out_embed);
    _weight_map.emplace("model.norm.weight", _weights.out_norm_w);

    for (size_t i = 0; i < _meta.nlayer; ++i) {
        _weight_map.emplace(layerName(i, ".input_layernorm.weight"), _weights.attn_norm_w[i]);

        _weight_map.emplace(layerName(i, ".self_attn.q_proj.weight"), _weights.attn_q_w[i]);
        _weight_map.emplace(layerName(i, ".self_attn.q_proj.bias"), _weights.attn_q_b[i]);
        _weight_map.emplace(layerName(i, ".self_attn.k_proj.weight"), _weights.attn_k_w[i]);
        _weight_map.emplace(layerName(i, ".self_attn.k_proj.bias"), _weights.attn_k_b[i]);
        _weight_map.emplace(layerName(i, ".self_attn.v_proj.weight"), _weights.attn_v_w[i]);
        _weight_map.emplace(layerName(i, ".self_attn.v_proj.bias"), _weights.attn_v_b[i]);
        _weight_map.emplace(layerName(i, ".self_attn.o_proj.weight"), _weights.attn_o_w[i]);

        _weight_map.emplace(layerName(i, ".post_attention_layernorm.weight"), _weights.mlp_norm_w[i]);
        _weight_map.emplace(layerName(i, ".mlp.gate_proj.weight"), _weights.mlp_gate_w[i]);
        _weight_map.emplace(layerName(i, ".mlp.up_proj.weight"), _weights.mlp_up_w[i]);
        _weight_map.emplace(layerName(i, ".mlp.down_proj.weight"), _weights.mlp_down_w[i]);
    }
}

tensor_t Qwen2Model::weight(const std::string &name) const {
    const auto it = _weight_map.find(name);
    if (it == _weight_map.end()) {
        return nullptr;
    }
    return it->second;
}

bool Qwen2Model::loadWeight(const std::string &name, const void *data, size_t nbytes) {
    const auto w = weight(name);
    if (!w) {
        return false;
    }
    if (data == nullptr) {
        throw std::runtime_error("Qwen2: null source passed while loading weight " + name);
    }
    const size_t expected = w->numel() * w->elementSize();
    if (nbytes != expected) {
        throw std::runtime_error("Qwen2: checkpoint byte size mismatch for weight " + name);
    }
    w->load(data);
    _loaded_weights.insert(name);
    return true;
}

bool Qwen2Model::allWeightsLoaded() const {
    return _loaded_weights.size() == _weight_map.size();
}

void Qwen2Model::resetCache() {
    _cache_len = 0;
}

void Qwen2Model::copyRawSameDevice(void *dst, const void *src, size_t bytes) const {
    if (bytes == 0) {
        return;
    }
    core::context().setDevice(_device, _device_id);
    const auto kind = _device == LLAISYS_DEVICE_CPU
                          ? LLAISYS_MEMCPY_H2H
                          : LLAISYS_MEMCPY_D2D;
    core::context().runtime().api()->memcpy_sync(dst, src, bytes, kind);
}

void Qwen2Model::copyTensorData(const tensor_t &dst, const tensor_t &src) const {
    if (!dst || !src) {
        throw std::runtime_error("Qwen2: null tensor in copyTensorData.");
    }
    if (dst->dtype() != src->dtype() || dst->numel() != src->numel()) {
        throw std::runtime_error("Qwen2: incompatible tensors in copyTensorData.");
    }
    if (!dst->isContiguous() || !src->isContiguous()) {
        throw std::runtime_error("Qwen2: copyTensorData requires contiguous tensors.");
    }
    copyRawSameDevice(dst->data(), src->data(), dst->numel() * dst->elementSize());
}

void Qwen2Model::ensureCacheCapacity(size_t required) {
    if (required <= _cache_capacity) {
        return;
    }
    if (required > _meta.maxseq) {
        throw std::runtime_error("Qwen2: sequence length exceeds max_position_embeddings.");
    }

    size_t new_capacity = _cache_capacity == 0 ? std::min<size_t>(_meta.maxseq, 256) : _cache_capacity;
    if (new_capacity == 0) {
        new_capacity = required;
    }
    while (new_capacity < required) {
        const size_t doubled = new_capacity > _meta.maxseq / 2 ? _meta.maxseq : new_capacity * 2;
        if (doubled == new_capacity) {
            break;
        }
        new_capacity = doubled;
    }
    if (new_capacity < required) {
        new_capacity = required;
    }

    const size_t cached_elems = _cache_len * _meta.nkvh * _meta.dh;
    const size_t cached_bytes = cached_elems * utils::dsize(_meta.dtype);

    for (size_t i = 0; i < _meta.nlayer; ++i) {
        auto new_k = makeTensor({new_capacity, _meta.nkvh, _meta.dh});
        auto new_v = makeTensor({new_capacity, _meta.nkvh, _meta.dh});

        if (_k_cache[i] && cached_bytes != 0) {
            copyRawSameDevice(new_k->data(), _k_cache[i]->data(), cached_bytes);
            copyRawSameDevice(new_v->data(), _v_cache[i]->data(), cached_bytes);
        }

        _k_cache[i] = std::move(new_k);
        _v_cache[i] = std::move(new_v);
    }

    _cache_capacity = new_capacity;
}

int64_t Qwen2Model::readI64(const tensor_t &tensor) const {
    if (!tensor || tensor->dtype() != LLAISYS_DTYPE_I64 || tensor->numel() < 1) {
        throw std::runtime_error("Qwen2: readI64 received an invalid tensor.");
    }

    int64_t value = 0;
    core::context().setDevice(_device, _device_id);
    const auto kind = _device == LLAISYS_DEVICE_CPU
                          ? LLAISYS_MEMCPY_H2H
                          : LLAISYS_MEMCPY_D2H;
    core::context().runtime().api()->memcpy_sync(
        &value, tensor->data(), sizeof(value), kind);
    return value;
}

int64_t Qwen2Model::infer(const int64_t *token_ids, size_t ntoken) {
    if (token_ids == nullptr || ntoken == 0) {
        throw std::runtime_error("Qwen2: infer requires at least one token.");
    }
    if (!allWeightsLoaded()) {
        throw std::runtime_error("Qwen2: model inference requested before all checkpoint weights were loaded.");
    }

    const size_t total_len = _cache_len + ntoken;
    ensureCacheCapacity(total_len);

    auto ids = makeTensor({ntoken}, LLAISYS_DTYPE_I64);
    ids->load(token_ids);

    std::vector<int64_t> positions(ntoken);
    for (size_t i = 0; i < ntoken; ++i) {
        positions[i] = static_cast<int64_t>(_cache_len + i);
    }
    auto pos_ids = makeTensor({ntoken}, LLAISYS_DTYPE_I64);
    pos_ids->load(positions.data());

    auto hidden = makeTensor({ntoken, _meta.hs});
    ops::embedding(hidden, ids, _weights.in_embed);

    const size_t qdim = _meta.nh * _meta.dh;
    const size_t kvdim = _meta.nkvh * _meta.dh;
    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(_meta.dh));

    for (size_t layer = 0; layer < _meta.nlayer; ++layer) {
        // Pre-attention RMSNorm.
        auto attn_norm = makeTensor({ntoken, _meta.hs});
        ops::rms_norm(attn_norm, hidden, _weights.attn_norm_w[layer], _meta.epsilon);

        // Q/K/V projections.
        auto q_flat = makeTensor({ntoken, qdim});
        auto k_flat = makeTensor({ntoken, kvdim});
        auto v_flat = makeTensor({ntoken, kvdim});

        ops::linear(q_flat, attn_norm, _weights.attn_q_w[layer], _weights.attn_q_b[layer]);
        ops::linear(k_flat, attn_norm, _weights.attn_k_w[layer], _weights.attn_k_b[layer]);
        ops::linear(v_flat, attn_norm, _weights.attn_v_w[layer], _weights.attn_v_b[layer]);

        auto q = q_flat->view({ntoken, _meta.nh, _meta.dh});
        auto k = k_flat->view({ntoken, _meta.nkvh, _meta.dh});
        auto v = v_flat->view({ntoken, _meta.nkvh, _meta.dh});

        // Rotary positional embedding is applied to Q and K only.
        auto q_rot = makeTensor({ntoken, _meta.nh, _meta.dh});
        auto k_rot = makeTensor({ntoken, _meta.nkvh, _meta.dh});
        ops::rope(q_rot, q, pos_ids, _meta.theta);
        ops::rope(k_rot, k, pos_ids, _meta.theta);

        // Append new K/V states to this layer's cache.
        auto k_dst = _k_cache[layer]->slice(0, _cache_len, total_len);
        auto v_dst = _v_cache[layer]->slice(0, _cache_len, total_len);
        copyTensorData(k_dst, k_rot);
        copyTensorData(v_dst, v);

        // Attend over the complete prefix, including the newly appended states.
        auto k_all = _k_cache[layer]->slice(0, 0, total_len);
        auto v_all = _v_cache[layer]->slice(0, 0, total_len);
        auto attn_value = makeTensor({ntoken, _meta.nh, _meta.dh});
        ops::self_attention(attn_value, q_rot, k_all, v_all, attn_scale);

        // Output projection + first residual connection.
        auto attn_2d = attn_value->view({ntoken, qdim});
        auto attn_out = makeTensor({ntoken, _meta.hs});
        ops::linear(attn_out, attn_2d, _weights.attn_o_w[layer], nullptr);

        auto after_attn = makeTensor({ntoken, _meta.hs});
        ops::add(after_attn, hidden, attn_out);

        // MLP pre-norm.
        auto mlp_norm = makeTensor({ntoken, _meta.hs});
        ops::rms_norm(mlp_norm, after_attn, _weights.mlp_norm_w[layer], _meta.epsilon);

        auto gate = makeTensor({ntoken, _meta.di});
        auto up = makeTensor({ntoken, _meta.di});
        ops::linear(gate, mlp_norm, _weights.mlp_gate_w[layer], nullptr);
        ops::linear(up, mlp_norm, _weights.mlp_up_w[layer], nullptr);

        auto activated = makeTensor({ntoken, _meta.di});
        ops::swiglu(activated, gate, up);

        auto mlp_out = makeTensor({ntoken, _meta.hs});
        ops::linear(mlp_out, activated, _weights.mlp_down_w[layer], nullptr);

        auto next_hidden = makeTensor({ntoken, _meta.hs});
        ops::add(next_hidden, after_attn, mlp_out);
        hidden = std::move(next_hidden);
    }

    _cache_len = total_len;

    // Only the last position is needed to choose the next token.
    auto last_hidden = hidden->slice(0, ntoken - 1, ntoken);
    auto normed = makeTensor({1, _meta.hs});
    ops::rms_norm(normed, last_hidden, _weights.out_norm_w, _meta.epsilon);

    auto logits = makeTensor({1, _meta.voc});
    ops::linear(logits, normed, _weights.out_embed, nullptr);

    auto logits_1d = logits->view({_meta.voc});
    auto max_idx = makeTensor({1}, LLAISYS_DTYPE_I64);
    auto max_val = makeTensor({1});
    ops::argmax(max_idx, max_val, logits_1d);

    return readI64(max_idx);
}

} // namespace llaisys::models
