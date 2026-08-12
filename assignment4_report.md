# 作业 #4：LLAISYS CUDA 平台适配报告

## 1. 实现内容

本次作业完成 LLAISYS CUDA 后端实现，并适配以下两款平台：

- NVIDIA CUDA
- 天数智芯 CoreX / IXUCA

主要实现：

- CUDA Runtime API
- Add、Embedding、Argmax、Linear、RMSNorm、RoPE、SwiGLU、SelfAttention CUDA 算子
- Qwen2 / DeepSeek-R1-Distill-Qwen-1.5B GPU 推理支持

两个平台复用同一套 CUDA backend 和算子实现，根据平台选择不同编译工具链，并对少量 API 差异进行兼容处理。

## 2. NVIDIA 平台

### 环境

- GPU：NVIDIA GeForce RTX 3090
- CUDA Toolkit：12.4

### 编译

```bash
xmake f --nv-gpu=y -cv
xmake
xmake install
```

### 测试

```bash
python test/test_runtime.py --device nvidia
python test/ops/add.py --device nvidia
python test/ops/linear.py --device nvidia
python test/ops/self_attention.py --device nvidia

python test/test_infer.py \
  --model /path/to/DeepSeek-R1-Distill-Qwen-1.5B \
  --device nvidia
```

### 结果

- CUDA Runtime API：PASS
- CUDA 算子：PASS
- SelfAttention：PASS
- Qwen2 整模型推理：PASS

## 3. 天数智芯平台

### 环境

- GPU：天垓150 / BI-V150
- CoreX：4.4.0
- 编译器：CoreX clang++ 18.1.8
- GPU Arch：`ivcore11`

### 编译

增加 `xmake/iluvatar.lua`，使用 CoreX 工具链编译 CUDA-compatible 代码：

```bash
xmake f --iluvatar-gpu=y -cv
xmake
xmake install
```

核心编译参数：

```text
-x ivcore
--cuda-gpu-arch=ivcore11
--cuda-path=/usr/local/corex-4.4.0
```

天数平台复用 LLAISYS 原有 CUDA 逻辑设备，因此运行测试时仍使用：

```text
--device nvidia
```

针对 CoreX 4.4.0 的 cuBLAS 接口差异，对 `cublasGemmEx` 做了少量条件编译兼容。

### 测试

```bash
python3 test/test_runtime.py --device nvidia
python3 test/ops/self_attention.py --device nvidia

python3 test/test_infer.py \
  --model /data/models/DeepSeek-R1-Distill-Qwen-1.5B \
  --device nvidia \
  --max_steps 128 \
  --test
```

### 结果

- CoreX 编译：PASS
- Runtime API：PASS
- FP32 / FP16 / BF16：PASS
- CUDA 算子：PASS
- SelfAttention / GQA：PASS
- Qwen2 KV Cache 推理：PASS
- 128-step Hugging Face token 对比：PASS

最终输出：

```text
Test passed!
```

Hugging Face reference 与 LLAISYS 生成的 token 序列一致。

## 4. 支持状态

| 平台 | Runtime | CUDA 算子 | 整模型推理 | 状态 |
|---|---|---|---|---|
| NVIDIA RTX 3090 | PASS | PASS | PASS | Supported |
| 天数智芯 天垓150 | PASS | PASS | PASS | Supported |

本次作业完成了 NVIDIA 和天数智芯两款 CUDA / 类 CUDA 平台的适配。
