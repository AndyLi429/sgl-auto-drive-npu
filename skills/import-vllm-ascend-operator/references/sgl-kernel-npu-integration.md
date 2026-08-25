# SGLang-Kernel-NPU operator integration map

Use this reference after identifying a vLLM-ascend operator. Inspect the closest existing operator instead of assuming every item applies; workspace kernels and optional SoC features may need extra pieces.

| Concern | Target location | Verify |
| --- | --- | --- |
| Ascend C kernel | `csrc/<op>/op_kernel/` | Kernel entry point and host launch agree on names, layouts, workspace, and tiling. |
| Host launcher and tiling | `csrc/<op>/op_host/` | Uses target helpers and target error/stream conventions. |
| Public declaration | `include/sgl_kenel_npu_ops.h` | Signature matches the host implementation; retain the existing `kenel` spelling. |
| Torch schema | `csrc/pytorch_extensions.cpp` (`TORCH_LIBRARY_FRAGMENT(npu, m)`) | Schema accurately marks in-place/aliased tensors such as `Tensor(a!)`. |
| PrivateUse1 binding | `csrc/pytorch_extensions.cpp` | `TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)` binds the same symbol. |
| Build | `csrc/CMakeLists.txt` | Add host source plus the correct workspace/no-workspace Ascend C list and relevant guards. |
| Python wrapper | `python/sgl_kernel_npu/sgl_kernel_npu/` | Add only where a caller needs a stable Python API; invoke `torch.ops.npu.<op>`. |
| Test | `tests/python/sgl_kernel_npu/test_<op>.py` | Compare with a PyTorch reference and use supported NPU dtypes/devices. |

## Target references

- Minimal custom-op shape: `csrc/helloworld/`, `tests/python/sgl_kernel_npu/test_hello_world.py`.
- Registration and schemas: `csrc/pytorch_extensions.cpp`.
- Host function declarations: `include/sgl_kenel_npu_ops.h`.
- Build source lists and SoC guards: `csrc/CMakeLists.txt`.

## Verification commands

Run the narrowest applicable test first. Source changes to compiled ops need a rebuilt and reinstalled wheel before Python exercises the updated implementation.

```bash
bash build.sh -a kernels
pip install output/sgl_kernel_npu*.whl
python3 tests/python/sgl_kernel_npu/test_<op>.py
```

Use `bash build.sh` if the change spans other modules. The build requires the Ascend/CANN environment; do not mask failures caused by a missing toolkit or `torch-npu`. `pre-commit run --all-files` is the repository-wide formatting check when the environment permits it.
