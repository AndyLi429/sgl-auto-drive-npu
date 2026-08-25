---
name: ascend-a5-optimization
description: Use when analyzing, porting, developing, or tuning kernels, operators, inference workloads, memory access, Cube–Vector pipelines, SIMD/SIMT code, low-precision paths, or roofline performance for Huawei Ascend 950PR/950DT (A5).
---

# Ascend A5 Optimization

## Overview

Treat A5 as the Ascend 950PR/950DT generation, not as an A2/A3 variant. Separate product/SKU facts from measured limits, model every important compute and data-movement roof, and optimize the roof that the target workload actually reaches.

In operator discussions, interpret **CV** as **Cube–Vector** unless the user explicitly means computer vision. If they mean computer vision, use the DVPP subsection in `references/hardware-facts.md`.

## Start here

1. Identify the exact target: Ascend 950PR, Ascend 950DT, Atlas 350, or another product/SKU; record CANN, compiler, driver, firmware, frequency/power mode, tensor shapes, layouts, dtypes, and parallel topology.
2. Read the references needed for the request:
   - Numeric hardware/architecture claims → `references/hardware-facts.md`
   - FLOP/byte/roofline calculations → `references/performance-models.md`
   - Kernel implementation or profiling decisions → `references/tuning-playbook.md`
   - Freshness or provenance checks → `references/sources.md`
3. Label material claims as:
   - **Official**: Huawei product page, architecture whitepaper, or matching CANN documentation.
   - **Derived**: arithmetic from official or user-supplied values; show the formula.
   - **Measured**: target-system result; include method and environment.
   - **Unknown**: not established for the exact target; propose a query or microbenchmark.
4. Build the workload model before recommending changes: useful FLOPs/ops, bytes moved at GM/HBM and each relevant local path, synchronization/launch work, reuse, and overlap assumptions.
5. Compare model predictions with profiling. Revise the byte/overlap model when measured time is materially above the predicted roof; do not force the observation into the first hypothesis.

## Non-negotiable distinctions

- Do not silently choose the largest slash-separated SKU value in the whitepaper. Match core count, compute, memory capacity, bandwidth, and cache values to the deployed product.
- Keep **total**, **Cube**, and **Vector** peaks separate. GEMM uses the Cube roof; elementwise/softmax work uses Vector behavior; a fused kernel can encounter both.
- Keep HBM/Memory bandwidth, L2/local-memory bandwidth, Unified Bus interconnect, UBoE, and PCIe bandwidth separate. They constrain different paths.
- Treat a published peak as a ceiling, not sustained application bandwidth or compute. Calibrate effective roofs with microbenchmarks using the same dtype, layout, concurrency, and power state.
- Use decimal SI for published `TFLOPS` and `GB/s`/`TB/s`. State explicitly when tensor sizes use binary `MiB`/`GiB`.
- Do not reuse A2/A3 core counts, local-memory sizes, event rules, data paths, alignments, or profiler-field availability when A5 documentation differs.
- A5 provides a direct Cube L1 Buffer ↔ Vector UB CV path. Do not inherit the older claim that all AIC/AIV exchange must round-trip through GM.
- Missing profiler fields are not zero. The A5 `op_summary` has documented gaps, including Vector FLOP and some L2-bandwidth fields.

## Quantitative answer contract

For performance analysis, return enough information to reproduce the conclusion:

| Field | Required content |
|---|---|
| Target | Exact product/SKU and software/power environment, or an explicit unknown |
| Workload | Shapes, layouts, dtypes, batching, parallelism, cache assumptions |
| Compute | Useful and executed Cube/Vector FLOPs or ops, with formulas |
| Traffic | Bytes at HBM/GM and relevant local/CV/interconnect paths |
| Roofs | Published and calibrated compute/bandwidth values, clearly separated |
| Prediction | Arithmetic intensity, machine balance, time lower bounds, dominant roof |
| Evidence | Official/derived/measured/unknown tag for every decisive input |
| Validation | Profiler counters or microbenchmarks that would confirm/refute the diagnosis |
| Actions | Ordered optimizations tied to the observed limiting roof |

Use `python scripts/roofline.py --help` for deterministic arithmetic-intensity and single-memory-roof calculations. For multi-stage kernels, use the equations in `references/performance-models.md`.

## Freshness rule

Before quoting availability, current CANN feature support, profiler fields, or a product SKU not listed in the bundled snapshot, use the `web-access` skill to recheck Huawei/CANN primary sources. Update facts only when the new source has equal or higher authority, and preserve the exact product/version scope.

## Common mistakes

| Mistake | Correction |
|---|---|
| Calling all Ascend 950 products “A5” and using one number | Resolve PR vs DT and deployed SKU first. |
| Using combined BF16 peak for a pure GEMM | Use Cube BF16 peak; model Vector post-processing separately. |
| Declaring HBM-bound from low Cube utilization alone | Compute bytes/FLOPs, compare roofs, then confirm HBM/MTE counters and timeline. |
| Enabling CV parallelism without modeling handoff | Account for L1↔UB transfer, conversion, BufferID/event cost, and producer/consumer imbalance. |
| Applying SIMT to regular dense math | Default to SIMD for regular/coalesced work; use SIMT for irregular addressing or branch-heavy fragments. |
| Treating 128 B as every API's alignment | Distinguish HBM/L2 sector efficiency from each local-memory/API alignment requirement. |
| Reporting a theoretical speedup as expected gain | Cap by all remaining roofs and validate end-to-end, including launch and tail effects. |

