# A5 migration verification

Use this only after the compiler and linker succeed.

## Build and packaging

- Confirm configuration reports the intended `SOC_VERSION`, CANN root, A5 feature gate, and A3-only gate.
- Confirm the AIC and AIV device objects, generated host stub, host wrapper, and final shared library all build.
- Rebuild and reinstall the wheel after compiled-source changes.
- Check that the extension imports and the Torch schema is present.

## Tiling and dispatch

- Compare `sizeof(TilingData)` and relevant offsets on host and device where the toolchain permits.
- Exercise every supported tiling-key family and reject unsupported keys before launch.
- Include empty-input, normal, and full-load paths when implemented.
- Verify the launch block dimension and workspace contract against the original CANN implementation.

## Precision

- Compare against the original CANN operator or a PyTorch reference with production shapes and dtypes.
- Validate both cache modes and layouts that the host exposes.
- Verify in-place state-cache updates as well as returned output.
- Use dtype-appropriate absolute/relative tolerances and report maximum error, not only pass/fail.

## Performance

- Warm up before measurement and synchronize only at measurement boundaries.
- Compare the same input, stream, graph/eager mode, workspace state, and tiling cache state.
- Inspect for host synchronization, per-call tiling allocation/copy, or fallback dispatch added by integration.
- Treat compile-time `#if defined(__NPU_ARCH__)` as zero runtime overhead; it must not change the device code path.

Do not sign off “no precision or performance regression” without A5 hardware results.
