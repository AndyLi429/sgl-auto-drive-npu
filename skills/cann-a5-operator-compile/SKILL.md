---
name: cann-a5-operator-compile
description: Preflight, port, and debug CANN 9.1 AscendC direct-launch operator builds for Ascend A5/Ascend950. Use when moving native CANN/OPP kernels into sgl-kernel-npu or diagnosing bisheng, host-bisheng, device compilation, tiling-address-space, Torch registration, or SoC-gating failures. Do not use for post-build precision or performance analysis.
---

# CANN A5 Operator Compile

Make the smallest integration change that preserves the original device algorithm and produces an A5 kernel, host wrapper, and loadable Torch extension.

## Establish the build topology first

Before editing, identify:

- exact `SOC_VERSION`; treat `Ascend950*` as A5 only after confirming the repository's mapping;
- CANN version and active toolkit path;
- whether the source came from an OPP build or a direct `ascendc_library` build;
- which stages failed: precompile/preprocess, AIC/AIV device, host-bisheng, host C++, link, or Python load;
- the first compiler `error:`, not the final cascade of `gmake` failures.

For sgl-kernel-npu on an A5 machine, run `scripts/preflight_a5.sh <repo-root> <SOC_VERSION> [CANN-home]` before the first build and after changing build integration. This script is read-only.

## Apply these A5 invariants

1. Keep A5-only sources and definitions gated by the repository's Ascend950 condition. Do not expose A5 headers to A3 targets.
2. For MicroAPI kernels built through `ascendc_library`, leave only generally parseable includes outside and wrap the complete device implementation—including the global kernel entry—with `#if defined(__NPU_ARCH__)`. The generated host stub is compiled separately; do not create an empty host kernel body merely to satisfy host-bisheng.
3. Do not pass OPP-only flags directly to bisheng. In particular, investigate `--op_relocatable_kernel_binary=true` based on the CMake API that consumes it; it is not a valid direct bisheng option in the CANN 9.1 failure observed here.
4. A direct-launch `GM_ADDR tiling` is global memory. If a kernel class constructor expects a local pointer, materialize a local tiling struct first. Prefer the repository's established `kernel_utils::CopyTiling`; otherwise use an explicit, alignment-safe scalar copy. Do not use `GET_TILING_DATA*` in a direct-launch project unless the active CANN toolchain explicitly supports it.
5. Keep public host declarations, Torch schemas, and Torch implementations visible on every SoC where their host source is compiled. An A5 op declaration must not sit inside `SGL_KERNEL_ENABLE_A3_ONLY_OPS`.
6. Treat host/device tiling definitions as an ABI: verify identical field order, widths, packing, total size, and tiling-key encoding before changing either side.

## Fixed workflow

1. Capture environment and run the preflight script.
2. Compare the original OPP build API with the target direct-launch API; transfer kernel semantics, not build flags or helper macros blindly.
3. Complete the integration path: device source, host wrapper/tiling, public declaration, Torch schema and implementation, CMake source/target/include definitions, focused test.
4. Build the narrowest kernel target from a known clean build directory. Ask before deleting a build directory unless the user already authorized a clean rebuild.
5. Fix one compiler stage and one root cause at a time. Rebuild and confirm that stage advances before addressing later diagnostics.
6. After linking, reinstall the generated wheel or shared library before testing; an existing installation does not contain source changes.
7. Run import/schema smoke tests, then precision comparison, then performance comparison. A successful compile does not establish precision or performance.

When a build fails, read [references/error-playbook.md](references/error-playbook.md). Read only the matching section. For migration sign-off, read [references/verification.md](references/verification.md).

## Boundaries

- Preserve imported CANN kernel bodies unless evidence shows an A5 incompatibility.
- Do not suppress errors globally or add broad compiler flags to make one source compile.
- Do not modify unrelated warnings during a migration fix. Format warnings in validation logs are worth recording but do not explain a successful link failure.
- Do not claim A5 behavior from local static checks. Require a clean CANN 9.1 A5 build and hardware tests.
- Do not SSH to an A5 node or start remote builds without explicit authorization.
