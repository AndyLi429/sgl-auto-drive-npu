# sgl-kernel-npu Operator Development

This map was checked against `sgl-project/sgl-kernel-npu` origin/main `9cddbc840bab8d6d6686df8efcad777b38a2ba6b` plus a local A5-compressor branch at `963e601a2b27ddff9619d18c96bb6ee047ada604` on 2026-08-22. Distinguish official main from local branch changes in conclusions.

## Deliverables and Boundaries

The repository produces three relevant packages/artifacts:

| Deliverable | Source | Runtime surface |
|---|---|---|
| `sgl_kernel_npu` | `csrc/`, `python/sgl_kernel_npu/` | `torch.ops.npu.*` plus imported Triton functions |
| `deep_ep` | `csrc/deepep/`, `python/deep_ep/` | `deep_ep.Buffer`, `deep_ep_cpp`, packaged custom OPP |
| `attentions` | `csrc/attentions/`, `python/attentions/` | separately built attention package |

`python/sgl_kernel_npu/sgl_kernel_npu/__init__.py` loads `libsgl_kernel_npu.so`. A direct Python import such as `from sgl_kernel_npu.norm...` resolves to package code that may be a Triton-Ascend function or a wrapper around a registered op. `torch.ops.npu.<name>` may select an Ascend C op from this library, torch-npu/CANN, or another library.

To classify an op:

```bash
rg -n 'm\.def\("?<op_name>|m\.impl\("?<op_name>' csrc/pytorch_extensions.cpp
rg -n '<op_name>' include csrc python/sgl_kernel_npu tests/python/sgl_kernel_npu
```

If no schema exists in `pytorch_extensions.cpp`, do not label it an `sgl-kernel-npu` Ascend C op without locating another registration path.

## Current Layout

- `csrc/<op>/op_host/`: validation, tiling, workspace and launch.
- `csrc/<op>/op_kernel/`: Ascend C device implementation.
- `include/sgl_kenel_npu_ops.h`: public C++ declarations; the `kenel` typo is load-bearing.
- `csrc/pytorch_extensions.cpp`: Torch schemas and `PrivateUse1` implementations.
- `csrc/CMakeLists.txt`: source lists, SoC guards, workspace classification and optional modules.
- `python/sgl_kernel_npu/sgl_kernel_npu/`: Triton kernels and Python wrappers.
- `tests/python/sgl_kernel_npu/`: focused operator tests and PyTorch references.

Current operator areas include allocation/cache assignment, speculative sampling/tree building, LoRA, MLA/GQA/sparse attention and indexers, FLA/GDN/KDA/Mamba, normalization/activation, KV transfer, compressor, and optional Catlass GEMM.

## Ascend C Integration Contract

For a new or changed compiled op, inspect and normally update all applicable layers:

1. `csrc/<op>/op_host` and `op_kernel`.
2. Declaration in `include/sgl_kenel_npu_ops.h`.
3. Schema and `PrivateUse1` implementation in `csrc/pytorch_extensions.cpp`.
4. Host/device source registration in `csrc/CMakeLists.txt`.
5. A focused test in `tests/python/sgl_kernel_npu/`.
6. A Python wrapper and SGLang consumer if the public surface needs them.

The device source must be placed in the correct CMake list. Workspace/tiling kernels require the compile definitions associated with the workspace list; a wrong classification causes missing tiling/workspace behavior.

Torch schemas are API contracts. Preserve alias/mutation annotations, optional defaults, dtype/layout semantics, and backward compatibility. Search all SGLang consumers before changing them.

## Triton-Ascend Contract

Triton kernels live mainly under `attention/`, `indexer/`, `fla/`, `mamba/`, `norm/`, `activation/`, `moe/`, and `sample/`. They are imported directly rather than registered through `pytorch_extensions.cpp`.

Tune for Ascend vector cores and Unified Buffer. Use `utils/triton_utils.py::get_device_properties` and the local kernel's tuning pattern; CUDA SM/shared-memory launch values are not portable. Preserve upstream attribution when porting FLA code.

## Build Matrix

Read `bash build.sh -h` in the active checkout. Current targets are:

```bash
git submodule update --init --recursive

bash build.sh                         # full A3 build
bash build.sh -a kernels              # sgl_kernel_npu, auto/default A3
bash build.sh -a kernels Ascend910B1  # A2 kernels
bash build.sh -a kernels Ascend950PR_9599  # A5 kernel SoC name
bash build.sh -a deepep               # auto-detect A2/A3/A5 when possible
bash build.sh -a deepep Ascend950     # A5 DeepEP alias
bash build.sh -a deepep2              # A2 DeepEP compatibility target
bash build.sh -a memory-saver
bash build.sh -a attentions            # separate attentions package
bash build.sh -d                      # debug full build
```

`Ascend950` is the DeepEP build alias; `sgl_kernel_npu` kernels require an AscendC-supported SoC name such as `Ascend950PR_9599`. Do not reuse the old `deepep-adapter` or `deepep-kernels` targets unless they appear in the current `build.sh -h`.

Install fresh artifacts:

```bash
pip install --force-reinstall output/sgl_kernel_npu*.whl
pip install --force-reinstall output/deep_ep*.whl
python -c "import torch, sgl_kernel_npu; print(torch.ops.npu.sgl_kernel_npu_version())"
```

Compiled changes are invisible until rebuild and reinstall. Triton files are also packaged in the wheel unless the environment is explicitly editable/in-place.

## Chip Guards

`csrc/CMakeLists.txt` derives feature guards from `SOC_VERSION`. Some schemas and implementations, including several MLA/GDN/sparse-attention paths, are compiled only when `SGL_KERNEL_ENABLE_A3_ONLY_OPS` is enabled. Despite the historical name, inspect its current SoC mapping before inferring A5 support.

Optional Catlass schemas require `BUILD_CATLASS_MODULE`. An op appearing in source does not prove it exists in the installed wheel.

## Tests and Diagnostics

Run the exact test matching the operator:

```bash
python3 tests/python/sgl_kernel_npu/test_mla_preprocess.py
python3 tests/python/sgl_kernel_npu/test_transfer_kv_dim_exchange.py
python3 tests/python/sgl_kernel_npu/test_compressor_a5.py
python3 -m unittest discover tests/python/sgl_kernel_npu
pre-commit run --all-files
```

Before real-device testing:

```bash
python -c "import torch, torch_npu, sgl_kernel_npu; print(torch.npu.is_available()); print(torch.ops.npu.sgl_kernel_npu_version())"
python -c "import torch; print(getattr(torch.ops.npu, '<op_name>')._schemas)"
```

For performance-sensitive code, inspect `.item()`, `_local_scalar_dense`, tensor truth checks, host copies, stream synchronization, tiling-key coverage, workspace sizing, and device guards. Correctness needs a PyTorch/CPU reference; performance needs warmup, synchronization at measurement boundaries, and matching shapes/dtypes/layouts.
