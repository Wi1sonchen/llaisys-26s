#include "llaisys/models/qwen2.h"

#include "llaisys_tensor.hpp"
#include "../models/qwen2/qwen2.hpp"

#include <memory>
#include <string>
#include <vector>

struct LlaisysQwen2Model {
    std::unique_ptr<llaisys::models::Qwen2Model> impl;
    LlaisysQwen2Weights cweights{};

    std::vector<llaisysTensor_t> owned_handles;

    std::vector<llaisysTensor_t> attn_norm_w;
    std::vector<llaisysTensor_t> attn_q_w;
    std::vector<llaisysTensor_t> attn_q_b;
    std::vector<llaisysTensor_t> attn_k_w;
    std::vector<llaisysTensor_t> attn_k_b;
    std::vector<llaisysTensor_t> attn_v_w;
    std::vector<llaisysTensor_t> attn_v_b;
    std::vector<llaisysTensor_t> attn_o_w;
    std::vector<llaisysTensor_t> mlp_norm_w;
    std::vector<llaisysTensor_t> mlp_gate_w;
    std::vector<llaisysTensor_t> mlp_up_w;
    std::vector<llaisysTensor_t> mlp_down_w;

    explicit LlaisysQwen2Model(std::unique_ptr<llaisys::models::Qwen2Model> model)
        : impl(std::move(model)) {
        const auto &w = impl->weights();
        const size_t nlayer = impl->meta().nlayer;

        // 3 top-level weights + 12 weights per decoder layer.
        owned_handles.reserve(3 + 12 * nlayer);

        auto make_handle = [this](const llaisys::tensor_t &tensor) {
            auto *handle = new LlaisysTensor{tensor};
            owned_handles.push_back(handle);
            return handle;
        };

        cweights.in_embed = make_handle(w.in_embed);
        cweights.out_embed = make_handle(w.out_embed);
        cweights.out_norm_w = make_handle(w.out_norm_w);

        attn_norm_w.reserve(nlayer);
        attn_q_w.reserve(nlayer);
        attn_q_b.reserve(nlayer);
        attn_k_w.reserve(nlayer);
        attn_k_b.reserve(nlayer);
        attn_v_w.reserve(nlayer);
        attn_v_b.reserve(nlayer);
        attn_o_w.reserve(nlayer);
        mlp_norm_w.reserve(nlayer);
        mlp_gate_w.reserve(nlayer);
        mlp_up_w.reserve(nlayer);
        mlp_down_w.reserve(nlayer);

        for (size_t i = 0; i < nlayer; ++i) {
            attn_norm_w.push_back(make_handle(w.attn_norm_w[i]));
            attn_q_w.push_back(make_handle(w.attn_q_w[i]));
            attn_q_b.push_back(make_handle(w.attn_q_b[i]));
            attn_k_w.push_back(make_handle(w.attn_k_w[i]));
            attn_k_b.push_back(make_handle(w.attn_k_b[i]));
            attn_v_w.push_back(make_handle(w.attn_v_w[i]));
            attn_v_b.push_back(make_handle(w.attn_v_b[i]));
            attn_o_w.push_back(make_handle(w.attn_o_w[i]));
            mlp_norm_w.push_back(make_handle(w.mlp_norm_w[i]));
            mlp_gate_w.push_back(make_handle(w.mlp_gate_w[i]));
            mlp_up_w.push_back(make_handle(w.mlp_up_w[i]));
            mlp_down_w.push_back(make_handle(w.mlp_down_w[i]));
        }

        cweights.attn_norm_w = attn_norm_w.data();
        cweights.attn_q_w = attn_q_w.data();
        cweights.attn_q_b = attn_q_b.data();
        cweights.attn_k_w = attn_k_w.data();
        cweights.attn_k_b = attn_k_b.data();
        cweights.attn_v_w = attn_v_w.data();
        cweights.attn_v_b = attn_v_b.data();
        cweights.attn_o_w = attn_o_w.data();
        cweights.mlp_norm_w = mlp_norm_w.data();
        cweights.mlp_gate_w = mlp_gate_w.data();
        cweights.mlp_up_w = mlp_up_w.data();
        cweights.mlp_down_w = mlp_down_w.data();
    }

    ~LlaisysQwen2Model() {
        for (auto *handle : owned_handles) {
            delete handle;
        }
    }
};

__C {

struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta *meta,
    llaisysDeviceType_t device,
    int *device_ids,
    int ndevice) {
    if (meta == nullptr || device_ids == nullptr || ndevice <= 0) {
        return nullptr;
    }

    // Assignment #3 uses a single device. Keep the public ndevice argument so
    // the ABI remains extensible, but use the first selected device for now.
    auto impl = std::make_unique<llaisys::models::Qwen2Model>(
        *meta, device, device_ids[0]);
    return new LlaisysQwen2Model(std::move(impl));
}

void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    delete model;
}

struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(
    struct LlaisysQwen2Model *model) {
    return model == nullptr ? nullptr : &model->cweights;
}

int llaisysQwen2ModelLoadWeight(
    struct LlaisysQwen2Model *model,
    const char *name,
    const void *data,
    size_t nbytes) {
    if (model == nullptr || name == nullptr || data == nullptr) {
        return 0;
    }
    return model->impl->loadWeight(std::string(name), data, nbytes) ? 1 : 0;
}

int llaisysQwen2ModelAllWeightsLoaded(struct LlaisysQwen2Model *model) {
    if (model == nullptr) {
        return 0;
    }
    return model->impl->allWeightsLoaded() ? 1 : 0;
}

void llaisysQwen2ModelResetCache(struct LlaisysQwen2Model *model) {
    if (model != nullptr) {
        model->impl->resetCache();
    }
}

size_t llaisysQwen2ModelCacheLength(struct LlaisysQwen2Model *model) {
    return model == nullptr ? 0 : model->impl->cacheLength();
}

int64_t llaisysQwen2ModelInfer(
    struct LlaisysQwen2Model *model,
    int64_t *token_ids,
    size_t ntoken) {
    if (model == nullptr) {
        return -1;
    }
    return model->impl->infer(token_ids, ntoken);
}

} // extern "C"
