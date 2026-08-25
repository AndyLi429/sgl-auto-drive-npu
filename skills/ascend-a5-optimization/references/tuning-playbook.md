# A5 operator tuning playbook

## Contents

- Establish the target
- Choose Cube, Vector SIMD, SIMT, or mixed execution
- Design memory and CV flow
- Profile and classify
- Optimization map
- Minimum microbenchmark matrix

## Establish the target

Record before comparing results:

- Product/card/SKU and `soc_version`
- AIC/AIV counts and mix ratio
- CANN, compiler, driver, firmware, framework/kernel package commit
- Frequency/power mode, temperature/throttling, ECC
- Shape distribution, dtype/accumulation/quantization, layout, tail ratio
- TP/DP/EP/CP topology and active communication links

Use `PlatformAscendC` in Host tiling code to query `GetCoreNumAic`, `GetCoreNumAiv`, `GetCoreMemSize`, `GetCoreMemBw`, `GetVecRegLen`, and architecture/version information. Tiling should consume platform facts instead of embedding another SKU's constants.

## Choose the execution model

| Work | Default | Why |
|---|---|---|
| Dense GEMM/convolution/tensor contraction | Cube SIMD | Highest dense tensor throughput |
| Regular elementwise, reduction, softmax fragments | Vector SIMD/Reg | Dual-issue vector path and RegFile reuse |
| Gather/scatter, hash, irregular addresses, branch-heavy fragments | Vector SIMT | Independent thread addressing and simpler irregular control |
| Matrix plus substantial epilogue/softmax/layout work | `__mix__(cube, vec)` | Enables explicit Cube–Vector pipeline |

Start with SIMD for regular work. A5's SIMT is a flexibility feature; thread registers, stack, UB shared memory, and 32–128 KB Data Cache reduce other available resources.

## Design memory and CV flow

1. Draw the actual path for each tensor: HBM/GM → L2 → L1/UB → L0/Reg → compute → output. Mark bytes, layout, dtype, ownership, and lifetime at each edge.
2. Size tiles against simultaneous live storage, not individual tensor size. Include ping/pong buffers, accumulators, queues, conversion scratch, SIMT Data Cache, and compiler reservations.
3. For mixed kernels, prefer A5's direct Cube L1 ↔ Vector UB path. Test `ENABLE_CV_COMM_VIA_SSBUF=true`; compare with the fallback to expose whether CV transfer or synchronization dominates.
4. Use L0C→UB in-flight quantization/layout conversion when it removes a separate pass without violating accuracy.
5. Use NDDMA for regular up-to-5D rearrangement/transpose fused with GM→UB movement. Confirm that 128 B coalescing and padding make traffic smaller than a SIMD/SIMT gather implementation.
6. Pipeline independent tiles. Model `Cube(i+1) || CV(i) || Vector(i-1)` and use BufferID/event dependencies to protect producer/consumer lifetimes.
7. Add double buffering only when multiple steady-state tiles exist and the overlapped stage is visible. Compare single vs double buffering; local capacity loss can outweigh overlap.
8. Exploit L2 locality intentionally. Use group/Die affinity for reused data, L2 allocate hints for near-term consumers, non-allocate for streaming outputs, and CMO only when lifetime is known.

## Profile and classify

Collect task time/timeline plus AIC/AIV metrics. Useful `op_summary` fields include:

- `aic_total_cycles`, `aiv_total_cycles`
- `*_mac_time/ratio`, `*_vec_time/ratio`, `*_scalar_time/ratio`
- `*_mte1_time/ratio`, `*_mte2_time/ratio`, `*_mte3_time/ratio`
- `aic_fixpipe_time/ratio`
- main-memory, L1, L0A/B/C, UB, and `ub_fixp2ub` bandwidth fields
- A5 local-L2 hit/miss/victim counters

A5 caveats: `vector_fops` and Vector arithmetic subtype ratios are not supported in the documented summary; L2 read/write bandwidth fields and some conflict fields are also unavailable. Use elapsed cycles, instruction/timeline data, byte models, and microbenchmarks rather than substituting zero.

| Evidence pattern | Classification | First actions |
|---|---|---|
| Cube time near calibrated Cube roof; large/regular shapes benefit least from more reuse | Cube compute | Improve tile occupancy, format/precision, tail handling; remove scalar issue gaps |
| Vector stage/ratio dominates or mixed kernel stalls after Cube; softmax/GELU/layout cost scales | Vector compute/dependency | RegFile reuse, SIMD scheduling, fewer passes/conversions; SIMT only for irregular fragments |
| MTE2 and measured HBM approach sustained copy roof; AI below machine balance | HBM/GM | Reduce bytes, fuse, improve KV/weight reuse, coalesce 128 B sectors, avoid materialization |
| HBM below roof but MTE1/L0/UB/fixpipe/CV path is saturated | Local movement | Retile, adjust buffer depth, use direct CV/SSBuffer, in-flight conversion, remove redundant copies |
| Both Cube and Vector show bubbles at handoff; barriers dominate | CV synchronization | Recheck BufferID/event lifetime, increase independent tiles, balance stages, remove round trips |
| Local L2 miss/victim high with reuse; cross-Die scheduling unstable | Cache locality | Group/Die affinity, L2 hints/CMO, smaller working set, separate streaming data |
| Small/tail shapes slow while no roof is approached | Latency/launch/scalar | Fuse launches, specialize tails, reduce branches/address arithmetic, batch work |
| Scale-out time dominates | Communication | Topology-aware collectives, overlap, message fusion, EP balance; use effective link roof |

## Minimum microbenchmark matrix

Run each with representative dtype/layout and aligned plus misaligned/tail cases:

1. Empty kernel / launch and synchronization latency.
2. Cube GEMM tiles across M/N/K and precision; report useful and executed FLOPs.
3. Vector SIMD and SIMT versions of regular and irregular patterns.
4. HBM sequential read, write, read+write; cold/hot L2; concurrency sweep.
5. GM↔UB, GM↔L1, L1↔L0A/B, L0C/fixpipe, UB↔Reg.
6. Direct CV SSBuffer path vs GM fallback, with/without conversion.
7. NDDMA vs explicit rearrangement for 1D–5D patterns.
8. Single/double buffering and `__mix__` ratios `(1,1)` vs `(1,2)`.
9. Real fused kernel vs decomposed operators, including accuracy and end-to-end framework time.

For each result, store target identity, versions, command/config, shape/layout/dtype, warmup/repetitions, clock/power state, counters, and derived roofline inputs. Never merge calibration data across different A5 bins without revalidation.

