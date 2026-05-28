---
name: sglang-skill
description: "Develop, debug, and optimize SGLang LLM serving engine on CUDA and Ascend NPU. Use when the user mentions SGLang, sglang, srt, sgl-kernel, sgl-kernel-npu, LLM serving, model inference, KV cache, attention backend, FlashInfer backend, MLA, MoE routing, MoE dispatch, expert parallelism SGLang, speculative decoding, disaggregated serving, TP/PP/EP, radix cache, continuous batching, chunked prefill, CUDA graph SGLang, model loading, quantization FP8/GPTQ/AWQ, JIT kernel, triton kernel SGLang, DeepSeek serving, EPLB (expert load balancing), HiCache, launch_server, sglang Engine API, LoRA inference, torch.compile SGLang, or asks about serving LLMs with SGLang. Also use when the user works on Ascend NPU kernels, CANN toolkit, torch-npu, AscendC kernel development, DeepEP-Ascend, Ascend attention backend, NPU MoE dispatch/combine, sgl-kernel-npu ops, lightning_indexer, mla_preprocess, alloc_extend, transfer_kv_dim_exchange, or any custom kernel for Huawei Atlas/Ascend hardware."
---

# SGLang Development — CUDA & Ascend NPU

This skill covers both the upstream SGLang serving engine (CUDA/ROCm) and the **sgl-kernel-npu** Ascend NPU kernel library that plugs into it.

---

## Part 1 — Upstream SGLang 

### Source Code Locations

SGLang 源码位于 skill 安装目录下的 `repos/sglang/`:
- Claude Code: `~/.claude/skills/sglang-skill/repos/sglang/`
- Cursor: `~/.cursor/skills/sglang-skill/repos/sglang/`

如果该路径不存在，运行 `bash update-repos.sh sglang`。

### Core Runtime (SRT)

```
SGLANG_REPO/python/sglang/srt/
├── layers/
│   ├── attention/          # Attention backends
│   │   ├── flashinfer_backend.py      # FlashInfer (默认)
│   │   ├── flashinfer_mla_backend.py  # FlashInfer MLA (DeepSeek)
│   │   ├── cutlass_mla_backend.py     # CUTLASS MLA
│   │   ├── flashattention_backend.py  # FlashAttention
│   │   ├── triton_backend.py          # Triton attention
│   │   ├── flashmla_backend.py        # FlashMLA
│   │   ├── nsa_backend.py             # Native Sparse Attention
│   │   ├── tbo_backend.py             # TBO
│   │   ├── fla/                       # Flash Linear Attention
│   │   ├── triton_ops/                # Triton attention ops
│   │   └── wave_ops/                  # Wave attention ops
│   ├── moe/                # MoE routing and dispatch
│   ├── quantization/       # FP8, GPTQ, AWQ, Marlin, etc.
│   ├── deep_gemm_wrapper/  # DeepGEMM 集成
│   └── utils/
├── models/                 # 模型实现 (LLaMA, DeepSeek, Qwen, etc.)
│   └── deepseek_common/    # DeepSeek V2/V3 共享组件
├── managers/               # Scheduler, TokenizerManager, Detokenizer
├── mem_cache/              # KV cache, Radix cache
├── model_executor/         # 模型执行器, forward batch
├── model_loader/           # 模型加载, 权重映射
├── entrypoints/            # 启动入口: Engine, OpenAI API server
├── speculative/            # Speculative decoding
├── disaggregation/         # Disaggregated prefill/decode
├── distributed/            # TP/PP/EP 分布式
├── compilation/            # CUDA Graph, Torch.compile
├── configs/                # 模型配置
├── lora/                   # LoRA 推理
├── eplb/                   # Expert-level load balancing
├── hardware_backend/       # 硬件适配 (CUDA, ROCm, XPU)
└── utils/                  # 工具函数
```

### JIT Kernels

```
SGLANG_REPO/python/sglang/jit_kernel/
├── flash_attention/        # Flash Attention 自定义实现
├── cutedsl_gdn.py          # CuTeDSL GDN kernel
├── concat_mla.py           # MLA concat kernel
├── norm.py                 # Normalization kernels
├── rope.py                 # RoPE position encoding
├── per_tensor_quant_fp8.py # FP8 量化
├── kvcache.py              # KV cache kernels
└── hicache.py              # HiCache kernels
```

### sgl-kernel (C++/CUDA)

```
SGLANG_REPO/sgl-kernel/csrc/
├── attention/          # Custom attention CUDA kernels
├── cutlass_extensions/ # CUTLASS GEMM extensions
├── gemm/               # GEMM kernels
├── moe/                # MoE dispatch/combine kernels
├── quantization/       # Quantization CUDA kernels
├── allreduce/          # AllReduce CUDA kernels
├── speculative/        # Speculative decoding kernels
├── kvcacheio/          # KV cache I/O
└── mamba/              # Mamba SSM kernels
```

---

## Part 2 — sgl-kernel-npu (Ascend NPU)

当前工作目录: `d:/Github/sgl-kernel-npu`

### Hardware Context

| 代号 | 硬件 | 互联 |
|------|------|------|
| A3   | Ascend 910C (Atlas 800T A2) | Full-mesh HCCS |
| A2   | Ascend 910B (Atlas 800T A2) | HCCS intra-node + RDMA inter-node |

软件栈: CANN 8.3.RC1+ / 8.2.RC1+ (deepep), torch-npu 2.5.1-7.0.0+, PyTorch 2.5.1+, AscendC (Ascend C 设备端编程语言)

### Project Layout

```
sgl-kernel-npu/
├── csrc/                    # Ascend C kernel 源码
│   ├── helloworld/          # 参考示例 (用来学 kernel 结构)
│   ├── alloc_extend/        # KV cache 页面分配
│   ├── assign_cache_op/     # KV cache 赋值
│   ├── build_tree/          # 投机解码树构建
│   ├── batch_matmul_transpose/  # 批量矩阵乘 + 转置
│   ├── catlass/             # Catlass GEMM (Ascend CUTLASS)
│   ├── deepep/              # DeepEP A3 (全mesh HCCS)
│   │   ├── ops/             # A3 MoE dispatch/combine ops
│   │   └── ops2/            # A2 MoE dispatch/combine ops
│   └── pytorch_extensions.cpp  # TORCH_LIBRARY 注册入口
├── include/
│   └── sgl_kenel_npu_ops.h  # 所有 op 的 C++ 声明
├── python/
│   ├── sgl_kernel_npu/      # SGLang kernel NPU Python 包
│   │   └── sgl_kernel_npu/
│   │       ├── __init__.py          # 加载 libsgl_kernel_npu.so
│   │       ├── activation/          # SwiGLU, quantized activation
│   │       ├── attention/           # Decode attention, attention sinks
│   │       ├── fla/                 # Flash Linear Attention (GDN, chunk, etc.)
│   │       ├── mamba/               # Mamba causal conv
│   │       ├── mem_cache/           # Memory allocator
│   │       ├── moe/                 # MoE utilities
│   │       ├── norm/                # RMSNorm, add_rmsnorm_bias, split_qkv_rmsnorm_rope
│   │       ├── sample/              # Tree verification (speculative decoding)
│   │       ├── speculative.py       # Speculative decoding helpers
│   │       ├── kvcacheio.py         # KV cache I/O
│   │       └── utils/               # Triton utilities
│   └── deep_ep/             # DeepEP Python 包
│       └── deep_ep/
│           ├── buffer.py    # Dispatch/combine 内存 buffer 管理
│           └── utils.py
├── tests/
│   ├── python/sgl_kernel_npu/   # 各 kernel 单测
│   └── python/deepep/           # DeepEP 通信测试
└── build.sh                 # 构建入口
```

### Registered Ops (`torch.ops.npu.*`)

| Op | 功能 |
|----|------|
| `helloworld` | 参考 kernel |
| `alloc_extend` | KV cache 页面分配 (prefill/extend) |
| `cache_loc_assign` | KV token pool 位置分配 |
| `cache_loc_update` | KV token pool 位置更新 |
| `assign_cache_op` | KV cache 数据赋值 |
| `build_tree_kernel_efficient` | 投机解码树构建 |
| `mla_preprocess` | MLA 预处理 (fused RMSNorm + QKV proj + RoPE + 写 KV cache) |
| `batch_matmul_transpose` | 批量矩阵乘转置 |
| `transfer_kv_dim_exchange` | KV cache device↔host 搬运 (disaggregated serving) |
| `bgmv_expand/shrink` | LoRA BGMV expand/shrink |
| `sgmv_expand/shrink` | LoRA SGMV expand/shrink |
| `sgemmv_expand/shrink` | LoRA SGEMMV (multi-rank) expand/shrink |
| `lightning_indexer` | Sparse attention indexer (NPU 稀疏注意力) |
| `catlass_matmul_basic` | Catlass GEMM (需 BUILD_CATLASS_MODULE) |

### Build

```bash
# 全量构建
./build.sh

# 按模块构建
./build.sh -a deepep       # DeepEP A3
./build.sh -a deepep2      # DeepEP A2
./build.sh -a kernels      # SGLang kernels
./build.sh -a memory-saver # Torch memory optimizer

# Debug 模式
./build.sh -d

# 安装 wheel
pip install output/*.whl
```

CMake 选项: `BUILD_DEEPEP_MODULE`, `BUILD_KERNELS_MODULE`, `BUILD_CATLASS_MODULE` (默认 off)

### Testing

测试需要真实 Ascend NPU 设备。远程 Ascend 910C 节点见 `sglang-debug-ascend` skill。

```bash
# 单个测试文件
python3 tests/python/sgl_kernel_npu/test_mla_preprocess.py
python3 tests/python/deepep/test_fused_deep_moe.py

# 目录批量
python3 -m unittest discover tests/python/sgl_kernel_npu

# 指定 class/method
python3 -m unittest tests.python.sgl_kernel_npu.test_alloc_extend_slot.TestAllocExtend.test_case1_prefill
```

测试文件:
- `tests/python/sgl_kernel_npu/` — 各个 kernel 的单测
- `tests/python/deepep/` — DeepEP 通信测试 (intra-node, inter-node, low-latency)

### Adding a New Ascend C Kernel

```
1. 在 csrc/<kernel_name>/ 实现 kernel:
   op_host/   — 主机侧: tiling 计算, tensor descriptor, Python interface
   op_kernel/ — 设备侧: AscendC kernel 实现

2. 参考 csrc/helloworld/ 作为最简模板

3. 在 include/sgl_kenel_npu_ops.h 声明 C++ 函数

4. 在 csrc/pytorch_extensions.cpp 注册:
   - TORCH_LIBRARY_FRAGMENT(npu, m) { m.def("op_name(...)") }
   - TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) { m.impl("op_name", ...) }

5. 在 csrc/CMakeLists.txt 添加源文件

6. 在 tests/python/sgl_kernel_npu/ 写测试 (对比 PyTorch/CPU 参考实现)
```

### DeepEP-Ascend (Expert Parallelism)

```
csrc/deepep/ops/    — A3 (910C): full-mesh HCCS
csrc/deepep/ops2/   — A2 (910B): HCCS intra + RDMA inter

关键 ops:
- cam_moe_dispatch_normal     — MoE token dispatch (normal mode)
- cam_moe_combine_normal      — MoE token combine (normal mode)  
- moe_distribute_dispatch_v2  — MoE dispatch v2
- notify_dispatch             — Notify-based dispatch (low-latency)
- fused_deep_moe              — Fused dispatch (A3 only)
- dispatch_normal_a2          — A2 专用 dispatch

python/deep_ep/deep_ep/buffer.py — Buffer 管理 (dispatch/combine 内存)
```

### Linting

```bash
pip install pre-commit && pre-commit install
pre-commit run --all-files        # isort + black + clang-format + ruff + codespell
pre-commit run clang-format --all-files  # C++ 格式化
```

---

## Search Patterns

```bash
# sgl-kernel-npu: 查找 op 注册
grep -n "m.def\|m.impl" csrc/pytorch_extensions.cpp

# sgl-kernel-npu: 查找 kernel 实现
grep -rn "def forward\|class.*Attention" python/sgl_kernel_npu/

# sgl-kernel-npu: 查找 AscendC kernel tiling
grep -rn "SetSysWorkspaceSize\|CalcTilingData" csrc/

# upstream SGLang: 查找 attention backend
rg "class.*Backend\|def forward" ~/.claude/skills/sglang-skill/repos/sglang/python/sglang/srt/layers/attention/

# upstream SGLang: 查找 MoE
rg "TopK\|router\|expert" ~/.claude/skills/sglang-skill/repos/sglang/python/sglang/srt/layers/moe/
```

---

## Source Lookup Table

| Need | Project | Path |
|------|---------|------|
| Ascend C kernel 实现 | sgl-kernel-npu | `csrc/<kernel>/op_kernel/` |
| Ascend C tiling (host) | sgl-kernel-npu | `csrc/<kernel>/op_host/` |
| Op 注册 | sgl-kernel-npu | `csrc/pytorch_extensions.cpp` |
| Op C++ 声明 | sgl-kernel-npu | `include/sgl_kenel_npu_ops.h` |
| Python wrapper | sgl-kernel-npu | `python/sgl_kernel_npu/sgl_kernel_npu/` |
| DeepEP A3 | sgl-kernel-npu | `csrc/deepep/ops/` |
| DeepEP A2 | sgl-kernel-npu | `csrc/deepep/ops2/` |
| DeepEP Python | sgl-kernel-npu | `python/deep_ep/deep_ep/buffer.py` |
| KV cache 分配 | sgl-kernel-npu | `csrc/alloc_extend/`, `python/.../mem_cache/allocator.py` |
| MLA preprocess | sgl-kernel-npu | `csrc/mla_preprocess/` (Ascend C), `python/.../attention/decode_attention.py` |
| LoRA kernels | sgl-kernel-npu | `csrc/bgmv/`, `csrc/sgmv/`, `python/.../` |
| Speculative decoding | sgl-kernel-npu | `csrc/build_tree/`, `python/.../sample/verify_tree_greedy.py` |
| Disagg KV transfer | sgl-kernel-npu | `csrc/kvcacheio/`, `python/.../kvcacheio.py` |
| Normalization | sgl-kernel-npu | `python/.../norm/` (add_rmsnorm_bias, split_qkv_rmsnorm_rope, l1_norm) |
| Attention backend 接口 | upstream SGLang | `srt/layers/attention/base_attn_backend.py` |
| FlashInfer MLA | upstream SGLang | `srt/layers/attention/flashinfer_mla_backend.py` |
| Scheduler | upstream SGLang | `srt/managers/` |
| KV cache / Radix cache | upstream SGLang | `srt/mem_cache/` |
| 模型实现 | upstream SGLang | `srt/models/` |
| Quantization | upstream SGLang | `srt/layers/quantization/` |
| Speculative decoding | upstream SGLang | `srt/speculative/` |
| Disaggregated serving | upstream SGLang | `srt/disaggregation/` |
| TP/PP/EP 分布式 | upstream SGLang | `srt/distributed/` |

---

## Common Scenarios

### 添加 Ascend NPU Attention Backend

1. 参考 `python/sgl_kernel_npu/sgl_kernel_npu/attention/decode_attention.py`
2. 对应的 Ascend C kernel 在 `csrc/` 下实现
3. 在 `pytorch_extensions.cpp` 注册 op
4. Upstream SGLang Ascend backend 适配在 `srt/hardware_backend/` 下扩展

### 调试 Ascend NPU Kernel

使用 `sglang-debug-ascend` skill 登录远程 Ascend 910C 节点，在 `/home/<your-user>/sglang` 下运行。

### 启动 SGLang (CUDA)

```bash
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --tp 1

from sglang import Engine
engine = Engine(model_path="meta-llama/Meta-Llama-3-8B-Instruct")
```

### Profiling Ascend NPU

使用 `sglang-perfermance-ascend` skill 分析 Ascend PyTorch Profiler 输出。

---

## Additional References

- SGLang 文档: https://docs.sglang.ai/
- SGLang GitHub: https://github.com/sgl-project/sglang
- sgl-kernel-npu: `d:/Github/sgl-kernel-npu`
- CANN 文档: Ascend C Programming Guide
