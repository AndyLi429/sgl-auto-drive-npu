# DeepEP-Ascend Operators and SGLang Integration

This map was checked against `sgl-project/sgl-kernel-npu` code on 2026-08-22. DeepEP behavior depends on chip topology, CANN, HCCL/RDMA setup, rank count, strategy, quant mode, and installed custom OPP; static inspection alone cannot prove communication correctness.

## Build and Kernel Trees

| Platform | DeepEP command | Kernel tree | Notes |
|---|---|---|---|
| A2 / Ascend 910B1 | `bash build.sh -a deepep Ascend910B1` or `-a deepep2` | `csrc/deepep/ops2/` | HCCS intranode plus internode/RDMA-aware paths |
| A3 / Ascend910_9382 | `bash build.sh -a deepep Ascend910_9382` | `csrc/deepep/ops/` | full-mesh HCCS-oriented implementation |
| A5 / Ascend950 | `bash build.sh -a deepep Ascend950` | `csrc/deepep/ops/` with A5 guards | A5-specific fused and communication implementations inside the shared tree |

`bash build.sh -a deepep` attempts hardware detection. If `npu-smi` is absent, current code falls back to A3; use an explicit SoC when cross-building or documenting a result.

Build flow:

```text
root build.sh
  -> csrc/deepep build and deep_ep_cpp pybind
  -> ops or ops2 custom operator project
  -> custom_opp*.run
  -> python/deep_ep/deep_ep/vendors/
  -> deep_ep wheel
```

`deep_ep/__init__.py` prepends the packaged custom OPP and library directories before importing `deep_ep_cpp`. Import order is functional, not cosmetic.

## Python Strategy Architecture

```text
deep_ep.Buffer public API
  -> normal_strategy or low_latency_strategy
  -> deep_ep_cpp.Buffer OR torch.distributed/torch_npu strategy
  -> ACLNN/custom OPP kernel
```

Key files:

- `python/deep_ep/deep_ep/buffer.py`: public DeepEP-compatible API.
- `ep_strategy.py`: strategy names, abstract contracts, registry, and `DEEP_USE_MODE` map.
- `strategies/normal_strategy.py`: normal `default` and `alltoall`.
- `strategies/low_latency_strategy.py`: low-latency `default`, `ops`, and `alltoall`.
- `csrc/deepep/pybind_extension.cpp`: `deep_ep_cpp.Buffer` Python surface.
- `csrc/deepep/deep_ep.cpp`: tensor validation, buffers, topology choice, and ACLNN calls.

Current environment mapping:

| `DEEP_USE_MODE` | Normal | Low latency |
|---|---|---|
| unset / `default` | custom `deep_ep_cpp` | custom `deep_ep_cpp` |
| `alltoall` | torch.distributed all-to-all | torch.distributed all-to-all |
| `ops` | custom `deep_ep_cpp` | torch-npu distribute ops |

Do not claim that every strategy reaches the repository's custom communication kernels. The `alltoall` and low-latency `ops` strategies deliberately take different paths.

## Public Modes

- Normal: `Buffer.get_dispatch_layout`, `dispatch`, `combine`; intended for throughput/prefill and larger token counts.
- Low latency: `low_latency_dispatch`, `low_latency_combine`; intended for decode and graph-compatible small batches.
- Fused: `Buffer.fused_deep_moe` with `FuseMode.FUSED_DEEP_MOE` or `DISPATCH_FFN_COMBINE`.

The exact quant modes are defined by `VALID_QUANT_MODES` and strategy implementations. Inspect them in the active checkout; mode support differs between default, ops, alltoall, normal, low-latency, and fused paths.

## Operator Map

Start at the `EXEC_NPU_CMD` calls in `csrc/deepep/deep_ep.cpp`, then follow `op_host/op_api/aclnn_*.cpp`, operator definitions/tiling, and device kernels.

| Stage | Representative operators |
|---|---|
| Layout/notify | `DispatchLayout`, `NotifyDispatch`, `NotifyDispatchA2` |
| Normal A3/A5 | `CamMoeDispatchNormal`, `CamMoeCombineNormal` |
| Normal A2/internode | `DispatchNormalA2`, `MoeDistributeCombineA2` |
| Low latency | `MoeLowLatencyDispatchV2`, `MoeLowLatencyCombineV2` |
| Fused | `FusedDeepMoe`, `DispatchFFNCombine` |

Registration uses CANN custom-op definitions (`OP_ADD`) and tiling registration (`IMPL_OP_OPTILING`), not `csrc/pytorch_extensions.cpp`. A3/A5 definitions live under `ops/op_host`; A2 definitions live under `ops2/op_host`; ACLNN wrappers live in each tree's `op_host/op_api`; kernels live in `op_kernel`.

The A2 tree also contains shared `CamMoe*` and low-latency V2 operators. Directory or operator suffix alone does not determine the runtime path; trace topology checks such as RDMA-rank count in `deep_ep.cpp`.

A5 has specialized `fused_deep_moe_a5` sources and build-time substitutions. Verify the selected fuse mode on A5 before claiming support: the Python enum may expose a mode whose custom definition is omitted or delegated differently in an A5 build.

## SGLang Consumption

Standard DeepEP chain:

```text
--moe-a2a-backend deepep / --deepep-mode <mode>
  -> layers/moe/fused_moe_triton/layer.py dispatcher factory
  -> layers/moe/token_dispatcher/deepep.py::DeepEPBuffer
  -> deep_ep.Buffer
  -> normal dispatch+combine or low_latency dispatch+combine
```

`DeepEPDispatcher` resolves `auto` based on whether the batch contains extend/prefill work. The main expert orchestration is also visible in `layers/moe/ep_moe/layer.py`.

FuseEP chain:

```text
--moe-a2a-backend ascend_fuseep
  -> FusedMoE bypass of the ordinary dispatcher
  -> hardware_backend/npu/moe/fuseep.py
  -> shared DeepEPBuffer
  -> Buffer.fused_deep_moe
```

This distinction matters for hooks, EPLB compatibility, graph capture, quantization, and performance analysis.

## Static Trace Commands

```bash
rg -n "Ascend910B1|Ascend910_9382|Ascend950|DEEPEP_VARIANT|kernel_dir=" build.sh
rg -n "StrategyMap|register_(normal|low_latency)_strategy|DEEP_USE_MODE" python/deep_ep/deep_ep
rg -n "EXEC_NPU_CMD|fused_deep_moe|dispatch_ffn_combine" csrc/deepep/deep_ep.cpp csrc/deepep/pybind_extension.cpp
rg -n "OP_ADD\(|IMPL_OP_OPTILING\(" csrc/deepep/ops/op_host csrc/deepep/ops2/op_host
rg -n "from deep_ep|buffer\.(dispatch|combine|low_latency_dispatch|low_latency_combine)|fused_deep_moe" ../sglang/python/sglang/srt
```

## NPU Validation

Build and reinstall the wheel for the target chip, then run the narrowest topology-matching test:

```bash
python3 tests/python/deepep/test_intranode.py
python3 tests/python/deepep/test_internode_a2.py
python3 tests/python/deepep/test_low_latency.py
python3 tests/python/deepep/test_fused_deep_moe.py
python3 tests/python/deepep/test_fused_deep_moe_a5.py
python3 -m unittest discover tests/python/deepep
```

Multi-rank launch parameters differ by topology. Read the selected test's entrypoint plus `tests/python/deepep/run_ascend_testcase.sh`, `run_test_internode.sh`, or enumeration scripts rather than inventing a generic `torchrun` command. All ranks require the correct HCCL/RDMA environment and consistent buffer-related variables. If a run hangs, capture rank-local logs and identify the last completed collective before changing kernel synchronization.
