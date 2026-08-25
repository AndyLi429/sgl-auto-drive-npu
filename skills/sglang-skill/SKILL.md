---
name: sglang-skill
description: Use when developing, debugging, reviewing, testing, or optimizing SGLang serving; Ascend NPU backends; sgl-kernel-npu Ascend C or Triton operators; DeepEP-Ascend dispatch/combine; attention, KV cache, MoE, graph capture, speculative decoding, or disaggregated serving integrations.
---

# SGLang and Ascend NPU Development

## Overview

Trace behavior from the current checkout instead of trusting a remembered directory or operator list. SGLang owns serving orchestration; `sgl-kernel-npu` owns Ascend C/Triton compute kernels and the Ascend `deep_ep` package; CANN/torch-npu owns many `torch.ops.npu.*` operators that are not implemented in `sgl-kernel-npu`.

## Establish the Source Snapshot

Before answering a code-specific question, locate the repositories and record their state:

```bash
git status --short --branch
git rev-parse HEAD
git log -1 --date=iso-strict --format='%H%n%ad%n%s'
git remote -v
```

Prefer the user's current checkout, including intentional branch changes. If the request says “latest,” compare it with the official `sgl-project/sglang` or `sgl-project/sgl-kernel-npu` default branch and state which snapshot supports the answer. Do not silently replace local code with upstream code.

## Route the Task

- Read [references/sglang-runtime.md](references/sglang-runtime.md) for SRT architecture, Ascend attention, NPU graph runners, configuration, launch, and SGLang tests.
- Read [references/sgl-kernel-npu.md](references/sgl-kernel-npu.md) for Ascend C/Triton kernels, Torch registration, A2/A3/A5 builds, operator tests, and the SGLang-package boundary.
- Read [references/deepep-ascend.md](references/deepep-ascend.md) for DeepEP normal/low-latency/fused MoE, strategies, custom OPP operators, and SGLang consumption.

Read only the references relevant to the request. For ordinary SGLang-to-kernel integration, read the runtime and kernel references. Add the DeepEP reference only when expert-parallel communication or FuseEP is involved.

## Core Trace Pattern

For a runtime or operator question, follow this chain until the requested behavior is proven:

```text
server argument / environment
  -> SGLang registry or factory
  -> SGLang consumer
  -> sgl_kernel_npu Python function OR torch.ops.npu schema OR deep_ep.Buffer
  -> C++ host/pybind and tiling
  -> Ascend C/Triton device implementation
  -> focused unit test and NPU integration test
```

At every boundary, record tensor shape/layout, dtype/quant mode, device, mutation/aliasing, workspace/tiling, stream/event semantics, and chip guards. Search call sites before changing a public schema.

## Quick Reference

| Question | Start here |
|---|---|
| Attention backend selection | `python/sglang/srt/layers/attention/attention_registry.py` |
| Ascend runtime backend | `python/sglang/srt/hardware_backend/npu/` |
| MoE/DeepEP consumer | `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` |
| Ascend C Torch schemas | `sgl-kernel-npu/csrc/pytorch_extensions.cpp` |
| Ascend C declarations | `sgl-kernel-npu/include/sgl_kenel_npu_ops.h` (spelling is intentional) |
| Triton-Ascend functions | `sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/` |
| DeepEP public API | `sgl-kernel-npu/python/deep_ep/deep_ep/buffer.py` |
| DeepEP A3/A5 vs A2 kernels | `csrc/deepep/ops/` vs `csrc/deepep/ops2/` |
| Version compatibility | SGLang Ascend installation docs plus installed package versions |

## Verification Contract

Use the narrowest relevant check first, then broaden:

1. Static trace: registry, consumer, schema/wrapper, implementation, test.
2. Host-safe unit test or import/compile check when available.
3. Rebuild and reinstall the affected wheel for any compiled or packaged source change.
4. Run the focused test on a matching A2/A3/A5 NPU and CANN stack.
5. Run the relevant SGLang NPU integration/accuracy/performance suite.

Local CPU inspection cannot prove Ascend kernel correctness, collective synchronization, OPP loading, or performance. Report the exact unrun NPU command instead of claiming success.

## Common Mistakes

- Treating every `torch.ops.npu.*` call as an `sgl-kernel-npu` op. Verify its schema in `csrc/pytorch_extensions.cpp`; otherwise it may come from torch-npu/CANN or another loaded library.
- Editing an installed wheel while testing a source checkout. Rebuild and reinstall, or prove editable/in-place loading.
- Copying CUDA launch constants into Triton-Ascend. Derive tuning from NPU vector-core and Unified Buffer constraints.
- Assuming `ops/` means A3 only. It serves A3 and A5 with build-time guards; `ops2/` is the A2 tree.
- Assuming `--moe-a2a-backend ascend_fuseep` is ordinary DeepEP dispatch/combine. It bypasses the standard dispatcher path.
- Using `.item()`, `_local_scalar_dense`, tensor truth checks, or unnecessary synchronization in a hot NPU path.

## One Complete Example

To trace `mla_preprocess`, find its SGLang call in `hardware_backend/npu/attention/mla_preprocess.py`, confirm the `torch.ops.npu.mla_preprocess` schema and chip guard in `csrc/pytorch_extensions.cpp`, follow the declaration and `csrc/mla_preprocess/{op_host,op_kernel}`, inspect `csrc/CMakeLists.txt` for workspace/tiling classification, then compare `tests/python/sgl_kernel_npu/test_mla_preprocess.py` with the SGLang NPU attention tests. Rebuild `sgl_kernel_npu` before real-device validation.
