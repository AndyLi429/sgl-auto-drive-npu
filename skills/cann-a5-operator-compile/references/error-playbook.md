# A5 compilation error playbook

Use the first matching compiler diagnostic. Later `gmake` errors usually report propagation, not another root cause.

## `unsupported option '--op_relocatable_kernel_binary=true'`

**Stage:** direct bisheng precompile.

**Meaning:** an option intended for the OPP build API reached bisheng as a raw compiler option.

**Action:** inspect the exact CMake call. Remove the option only from direct `ascendc_compile_options`; retain it where an OPP-specific API consumes it correctly. Reconfigure from a known clean target build because generated sub-build command lines may be cached.

## Constructor rejects a `gm` tiling pointer

Typical diagnostic:

```text
2nd argument ('gm ...TilingData *') is in address space gm,
but parameter must be in Local Memory
```

**Stage:** AIC/AIV preprocess or device compile.

**Meaning:** the entry passed `GM_ADDR tiling` directly to a class designed for a local tiling struct.

**Action:** allocate the tiling struct as a local variable and copy the global-memory bytes or fields into it before constructing the kernel class. Prefer `kernel_utils::CopyTiling` when present. Verify that the struct size is a multiple of the copy word size or handle the tail explicitly.

## `GET_TILING_DATA*` parse failures or undeclared output variable

Typical diagnostics:

```text
expected '(' for function-style cast or type construction
use of undeclared identifier 'tilingDataIn'
```

**Stage:** direct-launch preprocess.

**Meaning:** OPP tiling macros were copied into a kernel-direct project. Namespaced types can expose an additional macro parsing limitation, but removing the namespace does not make the macro supported by direct launch.

**Action:** use a local GM-to-local copy. Do not keep changing macro spellings after the build mode has been identified as direct launch.

## `MicroAPI` namespace or type is missing in `host_bisheng_obj`

Typical diagnostics:

```text
expected namespace name
use of undeclared identifier 'MicroAPI'
```

**Stage:** host-bisheng compilation after device targets may already succeed.

**Meaning:** `ascendc_library` recompiles the source for host handling, but MicroAPI is device-only.

**Action:** follow the CANN 9.1 A5 pattern:

```cpp
#include "kernel_operator.h"

#if defined(__NPU_ARCH__)
#include "device_microapi_implementation.h"

extern "C" __global__ __aicore__ void kernel_entry(...) {
    // unchanged device implementation
}
#endif
```

Wrap the entry itself. Avoid a host-visible empty kernel entry unless the target build system explicitly requires it.

## Torch registration says the function is not a namespace member

Typical diagnostic:

```text
'operator_name' is not a member of 'sglang::npu_kernel'
```

**Stage:** host C++ compilation of `pytorch_extensions.cpp`.

**Meaning:** the function definition may exist, but its public declaration was excluded by a SoC/build macro or omitted entirely.

**Action:** trace all four points independently:

1. public declaration;
2. host definition;
3. `m.def` schema;
4. `m.impl` registration.

Compare their preprocessor guards with the CMake source guard. For A5, specifically check accidental nesting under `SGL_KERNEL_ENABLE_A3_ONLY_OPS`.

## Warnings after a successful link

Examples include ignored `const` on a value return, `%u` versus `size_t`, and narrowing conversion warnings.

If the log reaches `Linking CXX shared library` and returns success, these are not compilation failures. Record them separately. A format mismatch can corrupt an error-only log value and should eventually be fixed, but it normally does not affect the kernel's hot path, precision, or generated device binary.

## `ASCConfig.cmake` not found

Classify this using actual CMake progress. If configure completes and the AscendC compiler targets are generated, it may be a non-fatal package-discovery warning in that repository. If AscendC commands or targets are missing, verify the active CANN package layout and sourced `set_env.sh`; do not conceal the missing environment with guessed paths.
