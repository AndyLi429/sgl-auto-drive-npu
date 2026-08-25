# Ascend 950 (A5) hardware facts

Snapshot verified: **2026-08-22**. Values below come from Huawei primary sources. Recheck `sources.md` when a newer product or CANN release matters.

## Contents

- Product and SKU peaks
- Compute architecture
- Local and global memory
- Cube–Vector and data movement
- SIMD/SIMT
- Interconnect and host IO
- DVPP / computer-vision hardware
- Facts that still require target-system discovery

## Product and SKU peaks

Slash-separated entries are different product bins. Keep columns aligned and confirm the deployed bin; do not combine a compute value from one bin with capacity/cache from another.

| Specification | Ascend 950PR | Ascend 950DT |
|---|---:|---:|
| Cube Core count | 32 / 28 | 36 / 32 / 28 |
| Vector Core count | 64 / 56 | 72 / 64 / 56 |
| MXFP4 total TFLOPS | 1784 / 1561 | 2007 / 1784 / 1561 |
| HiF8/MXFP8/FP8 total TFLOPS | 919 / 804 | 1034 / 919 / 804 |
| BF16/FP16 total TFLOPS | 486 / 425 | 547 / 486 / 425 |
| TF32 total TFLOPS | 243 / 212 | 273 / 243 / 212 |
| BF16/FP16 Cube TFLOPS | 432 / 378 | 486 / 432 / 378 |
| BF16/FP16 Vector TFLOPS | 54 / 47 | 60 / 54 / 47 |
| FP32 Vector TFLOPS | 27 / 23 | 30 / 27 / 23 |
| Memory capacity | 128 / 112 GB | 144 / 96 GB |
| Memory bandwidth | 1.6 / 1.4 TB/s | 4 TB/s |
| L2 capacity | 128 / 112 MB | 128 MB |

The official Atlas 350 card is a concrete 950PR product: 28 Cube + 56 Vector cores are implied by its 425 total BF16/FP16 TFLOPS bin; the product page directly specifies 425 BF16/FP16 TFLOPS, 112 GB memory, 1.4 TB/s memory bandwidth, PCIe 5.0 x16 (128 GB/s bidirectional), and ≤600 W.

## Compute architecture

- The full Ascend 950 design contains 36 AI subsystems. Each subsystem has **1 Cube Core + 2 Vector Cores**; product bins expose 36/32/28 Cube and 72/64/56 Vector cores.
- The chiplet contains 2 AI Dies and 2 IO Dies. 950PR uses 8 memory modules; 950DT uses 4 higher-bandwidth memory modules. The two products share the Ascend 950 compute die architecture but target different workload balance points.
- Cube handles dense tensor/matrix work. The third-generation Cube supports TF32, FP16, BF16, FP8, MXFP8, HiF8, INT8, and MXFP4. Relative to FP16/BF16 Cube throughput, FP8-class formats are 2× and MXFP4 is 4× at the same frequency.
- Vector uses a dual-issue, register-based SIMD microarchitecture with out-of-order execution. It adds native BF16, conversion support, and targeted Softmax/GELU improvements. FP16 and FP32 per-core throughput is documented as 2× the previous generation.
- Scalar cores perform address/control/instruction issue. Excess scalar work, branches, or instruction-generation pressure can therefore starve Cube, Vector, and MTE pipelines.

## Local and global memory

Official per-AI-subsystem local capacities:

| Memory | Capacity |
|---|---:|
| L1 Buffer | 512 KB per AI Core |
| L0A Buffer | 64 KB per AI Core |
| L0B Buffer | 64 KB per AI Core |
| L0C Buffer | 256 KB per AI Core |
| Unified Buffer (UB) | 512 KB per AI Core |

Memory roles:

- Cube operands stage through L1 → L0A/L0B; accumulation/results use L0C.
- Vector SIMD uses GM → UB → programmable RegFile → UB → GM. CANN documents each programmable Vector register as 256 B.
- The chip has up to 128 MB shared L2. It is a distributed multi-bank cache with 512 B cache lines and four 128 B sectors. It supports simultaneous read/write per bank.
- The 2-AI-Die design has a unified address space and hardware-maintained L2 coherence, but L2 is locally affine. Schedule/tile for locality instead of assuming all L2 hits have identical cost.
- L2 hints can select allocate/non-allocate behavior. SDMA CMO supports prefetch, writeback, invalidate, and flush. Use these only with a reuse/lifetime argument and measurement.
- Published 128/112 GB (PR) and 144/96 GB (DT) “Memory” is global DRAM/HBM-class device memory, not UB/L1 SRAM.

Common CANN alignment defaults (a specific API may override them):

| Storage | Start-address alignment |
|---|---:|
| UB, L1 | 32 B |
| L0A, L0B | 512 B |
| L0C | 64 B |
| BiasTable, Fixpipe | 64 B |

Treat 128 B sector/granularity as a transfer-efficiency fact, not a universal local-memory API alignment.

## Cube–Vector and data movement

- A5 exposes a direct CV path between Cube L1 Buffer and Vector UB, reducing L2/GM round trips. It can perform in-flight dtype and layout conversion during Cube↔Vector exchange.
- CANN's A5 DataCopy documentation recommends `ENABLE_CV_COMM_VIA_SSBUF=true` for UB→L1 communication through the hardware SSBuffer path. With it disabled, the documented fallback passes through GM and requires Matmul API registration.
- Cube results can be quantized and layout-converted on writeback: FP32/INT32 → BF16/FP16/FP8/INT8 and NZ → ND/DN, including L0C→UB in-flight quantization.
- NDDMA fuses data movement with up-to-5D rearrangement/transpose from GM to Vector UB. Its cache coalesces element-granular reads into 128 B reads when locality permits.
- BufferID synchronization (`get_buf`/`rel_buf`) expresses buffer lifetime and can reduce coupling relative to raw set/wait flags. Still validate generated dependencies and steady-state overlap.
- A5 mixed kernels use `__cube__`, `__vector__`, or `__mix__(cube, vec)`. Documented mix ratios are `(1,0)`, `(0,1)`, `(1,1)`, and `(1,2)`.

## SIMD/SIMT

- A5 Vector cores support SIMD and SIMT in the same generation. More than 90% of hardware compute is described as SIMD-oriented; use SIMT as a flexibility path, not a default performance path.
- Prefer SIMD for regular, contiguous/coalesced elementwise, reduction, tensor, and matrix-adjacent work.
- Prefer SIMT for irregular per-thread addresses, gather/scatter, hash/branch-heavy fragments, or code whose complexity blocks an efficient SIMD implementation.
- SIMT uses per-thread registers/stack and reserves part of UB as shared memory/Data Cache. The configurable Data Cache range is 32–128 KB, reducing UB available to other live data.

## Interconnect and host IO

- 72 HiLink lanes form 18 x4 Unified Bus ports. The published peak is 2016 GB/s bidirectional (marketed as 2 TB/s); out-of-box rate can be limited by optical modules/topology.
- UBoE: 2 × 400 Gb/s, statically sharing two UB ports.
- PCIe 5.0 x16: 128 GB/s bidirectional, sharing four UB ports.
- These are not HBM bandwidths. Add a separate communication roof when the workload actually crosses a card/die/host boundary.

## DVPP / computer-vision hardware

Use this only when the user means Computer Vision rather than Cube–Vector:

| Unit | Published capability by bin |
|---|---|
| VPC | 4 / 2 cores; 5760 / 2880 FPS at 1080p equivalent |
| JPEG decode | 8 JPEGD cores; 4096 FPS at 1080p equivalent; up to 32K×32K |
| JPEG encode | 4 / 2 JPEGE cores; 1024 / 512 FPS at 1080p equivalent; up to 32K×32K |

## Discover on the target system

Even when a whitepaper number exists, query the deployed system for binning and usable resources:

- `GetSocVersion()` / `GetCurNpuArch()`
- `GetCoreNumAic()` and `GetCoreNumAiv()`
- `GetCoreMemSize()` for UB/L1/L0 types
- `GetCoreMemBw()` for platform-exposed local-memory roofs
- `GetVecRegLen()`
- product/card identity, power/frequency state, ECC/reserved memory, topology, and CANN/driver/firmware versions

Measure sustained Cube, Vector, HBM, local path, CV, PCIe, and interconnect performance under workload-representative layouts; published peaks remain ceilings.

