---
name: import-vllm-ascend-operator
description: Port a specific Ascend C custom operator from a vLLM-ascend checkout into the sgl-kernel-npu repository and expose it as torch.ops.npu. Use when asked to copy, adapt, integrate, register, or test an Ascend C operator from vLLM-ascend for SGLang-Kernel-NPU.
---

# Import a vLLM-ascend operator

Port one named operator at a time. Keep the change surgical; do not copy an unrelated build system, Python framework, or group of operators.

## Inputs and boundaries

Require the operator name and an accessible vLLM-ascend source checkout or a source revision/URL. If either is absent, ask for it before copying code. Record the upstream commit and source paths in the handoff.

Before changing semantics, state the source and target assumptions for:

- input/output shapes, layouts, dtypes, and mutation/aliasing;
- workspace, stream, synchronization, and tiling behavior;
- supported SoCs and feature guards;
- source license and required notices.

Do not SSH to NPU hardware or start remote work unless explicitly requested. Do not claim kernel correctness from CPU-only checks.

## Workflow

1. **Inspect the source completely.** Find its Ascend C kernel, host launcher/tiling, Python wrapper, registration, build entries, tests, and direct dependencies. Identify all code that must move and every dependency that must instead use an existing SGLang equivalent. Preserve license notices.
2. **Map to the target integration path.** Read [references/sgl-kernel-npu-integration.md](references/sgl-kernel-npu-integration.md). Choose the closest target op as the style reference. Do not silently retain vLLM-specific namespaces, macros, wrappers, environment variables, or build targets.
3. **Adapt minimally.** Copy only required files beneath `csrc/<op_name>/`; update includes, namespaces, checks, tiling data, launch conventions, and feature guards to match the target. Reuse target utilities where they are equivalent. Keep the source API unless a target incompatibility requires an intentional, documented change.
4. **Integrate end-to-end.** Update the public C++ declaration, `TORCH_LIBRARY_FRAGMENT(npu, m)` schema, `PrivateUse1` implementation binding, and CMake source lists. Add a narrow Python wrapper only if callers need one. Confirm `torch.ops.npu.<op_name>` maps to the intended implementation and schemas accurately declare mutated/output aliases.
5. **Test against a reference.** Add a focused `unittest` covering normal inputs and each realistic edge case (dtype/layout/empty or boundary shape). Compare output with a PyTorch reference where practical. Include tolerances and explicit NPU-device handling.
6. **Verify in layers.** Run formatting/static checks that work locally, then use the narrow build and test commands in the reference. If CANN, torch-npu, or NPU hardware is unavailable, report the exact unrun command and why; require rebuild/reinstall before testing compiled changes.

## Review gates

Before handoff, inspect the diff and answer these questions:

- Does every copied file have a justified dependency and license provenance?
- Are all C++ declarations, schema arguments, alias annotations, implementation bindings, and build sources consistent?
- Does the operator avoid accidental device-to-host synchronization in performance paths?
- Are Soc guards consistent across source, CMake, registration, and tests?
- Is the installed wheel rebuilt before an NPU test is treated as evidence?

Report changed files, upstream commit/source paths, local checks, NPU checks still required, and any behavior intentionally different from upstream.
