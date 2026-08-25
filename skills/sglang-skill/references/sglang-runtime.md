# SGLang Runtime and Ascend NPU Map

This map was checked against `sgl-project/sglang` main at commit `5c03069d4bce87c97b257ad05f9d497729a47c4f` on 2026-08-22. Treat paths and flags as a snapshot: verify them in the active checkout before use.

## Runtime Ownership

`python/sglang/srt/` is the serving runtime:

| Area | Current path |
|---|---|
| HTTP/gRPC/Engine entrypoints | `entrypoints/` |
| Scheduler and process managers | `managers/` |
| Model/device/forward execution | `model_executor/` |
| KV, Radix, HiCache and pools | `mem_cache/` |
| Attention, MoE, quantization, sampling | `layers/` |
| Model implementations | `models/` |
| Speculative decoding | `speculative/` |
| Prefill/decode disaggregation | `disaggregation/` |
| TP/PP/DP/EP communication | `distributed/` |
| EPLB and elastic EP | `eplb/`, `elastic_ep/` |
| Central server configuration | `server_args.py` |

Search first:

```bash
rg -n "<flag-or-symbol>" python/sglang/srt test/registered/npu docs/docs/hardware-platforms/ascend-npus
```

## Ascend Backend

NPU-specific code is under `python/sglang/srt/hardware_backend/npu/`:

- `utils.py`: NPU defaults and backend initialization.
- `attention/ascend_backend.py`: main MLA/MHA attention backend.
- `attention/ascend_dsv4_backend.py`: DSV4/compressed-attention path.
- `attention/ascend_gdn_backend.py`, `ascend_kda_backend.py`, `ascend_hybrid_linear_attn_backend.py`: linear/hybrid attention.
- `attention/ascend_torch_native_backend.py`: native/reference-like path.
- `attention/mla_preprocess.py`: fused MLA preprocessing integration.
- `graph_runner/`: decode graph, EAGLE draft/extend, multilayer EAGLE, and ViT graph runners.
- `moe/`: Ascend routing, quantization, grouped matmul, finalize routing, and FuseEP.
- `quantization/`: NPU linear/MoE AWQ, GPTQ, W8A8, MX formats.
- `allocator_npu.py`, `memory_pool_npu.py`, `dsv4/`: NPU memory behavior.
- `extra_ops_loader.py`: separate shared-library loading; do not attribute these ops to `sgl_kernel_npu` without evidence.

### Attention selection chain

```text
server_args.py::_handle_npu_backends
  -> hardware_backend/npu/utils.py::set_default_server_args
  -> model_executor/model_runner_components/attention_backend_setup.py
  -> layers/attention/attention_registry.py
  -> hardware_backend/npu/attention/<backend>.py
  -> layers/radix_attention.py consumer
```

The main registration is `@register_attention_backend("ascend")`. The registry also resolves NPU-specific DSV4/GDN/KDA variants; inspect the conditional factory instead of assuming all models instantiate `AscendAttnBackend` directly.

### NPU decode graph selection chain

```text
model_executor/model_runner_components/cuda_graph_setup.py
  -> hardware_backend/npu/graph_runner/npu_graph_runner.py
  -> graph_runner/npu_cudagraph_backend.py
```

This is the decode path. Prefill graph selection still goes through the shared `resolve_prefill_backend()` logic and may choose full, breakable, or piecewise compilation; do not generalize the decode backend chain to prefill. Speculative and VLM paths select dedicated NPU graph runners from the corresponding worker/model code. Although shared abstractions retain “cuda graph” names, the NPU decode implementation uses `torch.npu.NPUGraph` and `torch.npu.graph`.

## Finding `sgl_kernel_npu` Consumption

Use both import and Torch-op searches:

```bash
rg -n "from sgl_kernel_npu|import sgl_kernel_npu" python/sglang
rg -n "torch\.ops\.npu|torch_npu\." python/sglang/srt/hardware_backend/npu python/sglang/srt/models python/sglang/srt/layers
```

Common consumers include attention sinks/indexers, MLA preprocess, GDN/KDA/Mamba, NPU KV allocation/store, LoRA, speculative tree building/verification, QKV-RMSNorm-RoPE fusions, and host KV transfer. The imports are the authoritative inventory; a static table will become stale.

## Ascend Installation and Launch

The compatibility matrix changes frequently. Check:

- `docs/docs/hardware-platforms/ascend-npus/getting-started/installation.mdx`
- `python/pyproject_npu.toml`
- installed `torch`, `torch_npu`, `triton`, `sgl_kernel_npu`, and `deep_ep` versions

The NPU project metadata intentionally does not pin the external kernel wheels; install them using the matching Ascend image or official kernel repository instructions.

Minimal server smoke test:

```bash
python3 -m sglang.launch_server \
  --model-path <model> \
  --device npu \
  --attention-backend ascend \
  --tp 1 \
  --host 127.0.0.1 \
  --port 30000

curl http://127.0.0.1:30000/health
```

For current examples, also inspect `docs/docs/hardware-platforms/ascend-npus/model-deployment/`; model-specific flags override generic recipes.

## DeepEP/FuseEP Runtime Flags

- `--moe-a2a-backend deepep`: standard dispatcher path.
- `--deepep-mode auto`: selects normal for extend/prefill and low-latency for decode at runtime.
- PD-separated serving: use `normal` on prefill and `low_latency` on decode when following the official deployment guide.
- `--moe-a2a-backend ascend_fuseep`: separate fused NPU path, normally decode-oriented and not the same call chain as standard DeepEP.
- `--deepep-dispatcher-output-dtype`: verify supported values and quantization behavior in the active `server_args.py`; do not reuse an old environment-variable recipe blindly.

## Tests

Current NPU integration tests are under `test/registered/npu/`, and host-safe focused tests are under `test/registered/unit/npu/`; do not use the obsolete `test/registered/ascend/` path.

Host-safe focused tests include:

```bash
pytest -q test/registered/unit/npu/attention/test_npu_ascend_backend.py
pytest -q test/registered/unit/npu/attention/test_npu_ascend_dsv4_backend.py
pytest -q test/registered/unit/npu/attention/test_npu_ascend_torch_native_backend.py
pytest -q test/registered/unit/npu/attention/test_npu_mla_preprocess.py
```

NPU suites are registered in `python/sglang/test/ci/ci_register.py` and run through:

```bash
cd test
python3 run_suite.py --hw npu --suite base-a-test-1-npu-a2
python3 run_suite.py --hw npu --suite base-b-test-1-npu-a3
```

Check each file's `register_npu_ci(...)` annotation and the active `test/run_suite.py` list before naming a suite; a focused test may still carry a legacy suite annotation. Direct `pytest` remains the narrowest way to run that file. Unit tests that mock NPU modules validate Python routing and metadata, not kernel correctness.
