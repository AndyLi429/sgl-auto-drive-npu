# ASCEND_PROFILER_OUTPUT — File-by-File Reference

Every file inside `ASCEND_PROFILER_OUTPUT/`, what's in it, when to open it.

## step_trace_time.csv — **start here**

Per-step wall-clock breakdown. One row per sampled step.

| Column | Meaning |
|---|---|
| Step | step id (profile a middle step, not the first — first includes warmup) |
| Computing (B) | total time of non-comm ops on NPU |
| Communication(Not Overlapped) (C) | exposed comm time — this is what actually slows you down |
| Communication (D) | comm time overlapped with compute |
| Communication Total (E) | C + D |
| Free (F) | blank time on the NPU stream |
| Stage Time (G) | end-to-end time of the step |
| Bubble (H) | `B + C + F - G` — waiting / sync / scheduling gap |

Identity: `B + C + F = G + H`.

Heuristic:
- `C / E > 30%` → overlap is broken, look at comm scheduling.
- `F / G > 15%` → host-side bottleneck or sync — check `operator_details.csv`.
- Large `H` → stream bubble, op launch gaps.

## kernel_details.csv — per-kernel NPU execution

Every kernel that ran on the NPU. Main columns:

| Column | Use |
|---|---|
| Name | unique instance name (e.g., `Cast60`); also appears in `trace_view.json` |
| Type | op type (Cast, MatMul, Softmax, AllReduce, …) |
| Accelerator Core | `AI Core` / `AI Vector Core` / `HCCL` — filter by `HCCL` to get all comm kernels |
| Start Time(us) | absolute timestamp; use for per-layer slicing and for joining with `memory_record.csv` |
| Duration(us) | actual execution time |
| Wait Time(us) | gap before this kernel started on its stream — indicates upstream blocking |
| Input Shapes / Input Data Types | shape- or dtype-driven slow kernels show here |
| Output Shapes / Output Data Types | same |

Tips:
- To find per-layer boundaries: pick a kernel that appears once per layer in forward, find its repeated occurrences and use `Start Time` deltas.
- Large `Wait Time` with small `Duration` = this op is fine, something upstream is slow.

## op_statistic.csv — op-type aggregates (compute only, no HCCL)

One row per op **type**, e.g., Cast, MatMul. Columns: `Count`, `Total Time(us)`, `Min`, `Max`, `Avg`, `Ratio(%)`.

Use to answer "which op type is eating the most time." Sort by `Total Time(us)` desc. Top 3 often account for >60% of compute.

## operator_details.csv — host-side (aten / Python) ops

Every PyTorch op call on the host. Includes shape, dtype, pipeline occupancy, and — critically — the `Call stack` column showing where the op was dispatched from in user code.

Key columns:
- `Host Self Duration(us)` — time spent in this op on host, excluding children
- `Host Total Duration(us)` — including children
- `Call stack` — Python call stack at the time of dispatch

When to use: the NPU is idle (high `Free` in step_trace_time) but the run is slow — the CPU side is the bottleneck. Sort by `Host Self Duration` desc to find what to fix.

## operator_memory.csv — per-op memory lifecycle

CPU-side aten op allocations. One row per allocation.

Columns include `Size(KB)`, `Allocation Time(us)`, `Release Time(us)`, `Duration(us)`, `Allocation Total Allocated(MB)`, `Allocation Total Reserved(MB)`, `Release Total Allocated`, `Release Total Reserved`, `Device Type`.

**Default sort is by release time**, so rows appear overlapped in time — don't be confused. For "what lived longest," compute `Release Time - Allocation Time` and sort desc.

## memory_record.csv — memory time series

Time series of `Total Allocated` and `Total Reserved` per component (PTA, GE). Good for spotting the peak and when it happens. Cross-reference timestamp with `kernel_details.csv` to find the culprit op.

## communication.json — HCCL op timing

Every HCCL op, its start time, duration, wait time, and metadata. Maps one-to-one with HCCL rows in `kernel_details.csv`. Useful when you need more than kernel_details shows (e.g., the internal phases of an AllReduce).

## communication_matrix.json — per-rank-pair comm

For each HCCL op, shows every rank pair involved: transport type, time, transmitted size, bandwidth. `total op info` at the end is the aggregated stat for the step.

Use to find straggler pairs: if one pair consistently has lower bandwidth, suspect hardware / NIC / topology.

## trace_view.json — full chrome trace

Everything from `operator_details.csv` plus timestamps (`ts`) and flow events linking host → device.

**Do not load with `json.load`** — it may OOM. Instead:
- Open in Perfetto UI (https://ui.perfetto.dev) by drag-and-drop. This is the best visual tool for spotting gaps, unexpected syncs, serialization of comm and compute.
- Or stream-parse with `ijson` if extracting specific events.

## Less-common files

- `profiler_info.json` — metadata (rank id, config). Useful only to confirm which rank and what level this data is.
- `l2_cache.csv` — generated only with `l2_cache=True`.
- `data_preprocess.csv` — generated only at `Level2`. Data loader timing.
- `npu_module_mem.csv` — module-level memory (generated in some configs).
- `FRAMEWORK/torch.op_range|op_mark|memory_usage` — raw framework binary data. Ignore unless doing framework-internals debugging. See framework-internals notes below.

## Framework data (for deep debugging only)

If you need to understand PTA framework-level behavior (rare):
- `OpRangeData` — op call ranges (start/end, shapes, stack). Collected via `at::addThreadLocalCallback` on enter/exit.
- `OpMarkData` — enqueue/dequeue events for NPU task submission; `correlation_id` links enqueue ↔ dequeue ↔ acl ↔ npu kernel. This is what connects host-side op trees to NPU kernel execution.
- `MemoryData` — tensor alloc/free events from `c10::MemoryReportingInfoBase`.

You almost never read these directly — `torch_npu` processes them into the CSVs above.
