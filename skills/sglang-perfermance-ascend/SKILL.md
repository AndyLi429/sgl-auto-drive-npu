---
name: sglang-ascend-perf-analysis
description: Analyze SGLang inference performance on Ascend NPU using Ascend PyTorch Profiler output. Use this skill whenever the user is profiling, tracing, or debugging performance of SGLang (or any PyTorch model) on Ascend NPU — including phrases like "性能分析", "性能调优", "性能瓶颈", "profiling", "算子耗时", "通信耗时", "HCCL", "慢", "掉点", "卡顿", "profiler_level", "ascend_pt", "ASCEND_PROFILER_OUTPUT", "kernel_details", "trace_view", "step_trace_time", "msprof", or when the user references a profiler output directory ending in `_ascend_pt` or files like `kernel_details.csv` / `trace_view.json` / `communication.json`. Also trigger when the user asks how to collect Ascend profiling data, how to interpret it, or how to find the slow part of a model run.
---

# SGLang Performance Analysis on Ascend NPU

This skill guides the analysis of SGLang (or any PyTorch model) performance on Ascend NPU using data produced by Ascend PyTorch Profiler.

## When this skill applies

The user has (or needs) profiler output from an Ascend NPU run and wants to:
- Find the performance bottleneck (compute? communication? memory? bubble?)
- Understand per-layer or per-op timing
- Compare two runs (e.g., before/after a change, NPU vs GPU)
- Interpret a specific file in the profiler output

If the user doesn't have profiling data yet, go to [references/collection.md](references/collection.md).

If they have data, jump straight to the decision flow below.

## Two profiling strategies — pick the right one for the question

SGLang on Ascend has **two distinct profiling strategies**. They are not interchangeable: each is collected differently and each answers a different question. Standard practice is to run them in order — **fix the CPU/framework side first, then go after NPU device-side**, because Python/scheduler stalls show up as "NPU is idle" and will mislead any device-side analysis you do before clearing the CPU side.

| Strategy | Collected via | `with_stack` | What it's for | Look at |
|---|---|---|---|---|
| **A. Framework / CPU profiling (with stack)** | SGLang's built-in profiler — HTTP endpoint `/start_profile` + `/stop_profile`, or `python -m sglang.profiler` | `True` (Python call stacks recorded) | Frame the **framework / scheduler / Python** overhead. Find host-side hotspots, slow tokenizer prep, scheduler bubbles, dispatch/launch latency, sync points. Primary focus is **CPU**. | `operator_details.csv` (Host Self Duration, Call stack), `trace_view.json` for gaps on the host timeline, `step_trace_time.csv` `Free` column |
| **B. NPU operator profiling (no stack)** | `torch_npu.profiler` wrapping the scheduler — preferred via the monkey-patch helper [`scripts/enable_scheduler_profiling.py`](scripts/enable_scheduler_profiling.py) (env-var driven, no source edits); the legacy `d:tmpscheduler_profiling.patch` is the fallback | `False` (no stack — keeps trace small and accurate on device) | Frame the **device-side NPU operator** performance. Find slow kernels, bad shapes/dtypes, HCCL waits, unoverlapped comm. Primary focus is **NPU**. | `kernel_details.csv`, `op_statistic.csv`, `communication.json`, `trace_view.json` device tracks |

**Order of investigation (do not skip):**

1. **First pass — Strategy A (CPU/framework).** If the Python scheduler is the bottleneck, NPU traces will just show idle gaps and you will chase ghosts. Confirm host-side is clean before touching device-side.
2. **Second pass — Strategy B (NPU operators).** Only meaningful once CPU side is no longer the dominant cost. Then you can trust `Computing` / `Communication` numbers and dig into kernels.

If the user hands you a trace and isn't sure which one it is: check `profiler_info.json` / the trace metadata for `with_stack`, or look at whether host frames carry Python call stacks. With stack → Strategy A. Without → Strategy B.

Collection details (commands, the patch, what to apply where) are in [references/collection.md](references/collection.md).

## Mental model: the output directory

A profiling run produces a directory ending in `_ascend_pt`. The pieces you will actually use:

```
<worker>_<ts>_ascend_pt/
├── profiler_info.json                     # metadata (rank id, config)
└── ASCEND_PROFILER_OUTPUT/                # ← 99% of analysis happens here
    ├── step_trace_time.csv                # START HERE: per-step breakdown
    ├── kernel_details.csv                 # every kernel on NPU (timing + shapes)
    ├── op_statistic.csv                   # op-type level aggregates
    ├── operator_details.csv               # aten/python op level (host-side)
    ├── operator_memory.csv                # per-op memory alloc/free
    ├── memory_record.csv                  # time-series memory footprint
    ├── communication.json                 # HCCL op timing (intra + inter card)
    ├── communication_matrix.json          # per-rank-pair comm (size, bandwidth)
    └── trace_view.json                    # full chrome trace (opens in ui.perfetto.dev)
```

`FRAMEWORK/` and `PROF_*/` are raw framework / CANN data — ignore unless doing deep framework-level debugging.

## Decision flow: which file answers which question

| Question the user is asking | Open this first |
|---|---|
| "Where does a step actually spend time?" (compute vs comm vs bubble) | `step_trace_time.csv` |
| "Which op type is eating time?" | `op_statistic.csv` |
| "Which specific kernel instance is slow? What shape?" | `kernel_details.csv` |
| "Is each layer's forward/backward regular? Where's the stall?" | `kernel_details.csv` + `trace_view.json` |
| "Is communication blocking compute?" | `step_trace_time.csv` (unoverlapped column) → `communication.json` |
| "Which rank pair has slow HCCL? Bandwidth OK?" | `communication_matrix.json` |
| "Memory peak? OOM cause?" | `memory_record.csv` + `operator_memory.csv` |
| "What is the host-side (aten) hotspot? Call stack?" | `operator_details.csv` |
| "I want to see the full timeline visually" | `trace_view.json` → load in Perfetto |

Full field-by-field reference in [references/output-files.md](references/output-files.md).

## Standard analysis workflow

Follow this order for a typical "why is SGLang slow on NPU" investigation. Do NOT jump to op-level detail before checking the step-level numbers — you'll waste time optimizing the wrong thing.

### 1. Quantify the problem: `step_trace_time.csv`

One row per sampled step. Columns (in order):

- `Step` — step id
- `Computing` — total non-comm op time (B)
- `Communication(Not Overlapped)` — exposed comm time (C)
- `Communication` (overlapped portion, D)
- `Communication Total` = C + D (E)
- `Free` — blank / idle time on the NPU stream (F)
- `Stage Time` — end-to-end step time (G)
- `Bubble` — `B + C + F - G` (H)

**Decision:**
- If `Computing` dominates → go to step 2 (compute-bound).
- If `Communication(Not Overlapped)` is large → go to step 3 (comm-bound or poor overlap).
- If `Free` / `Bubble` is large → schedule issue, host-side bottleneck, or sync; check `operator_details.csv` host time and trace_view for gaps.

Prefer a middle step (not step 0) — the first step includes warmup.

### 2. Compute-bound: find the hot op

1. Open `op_statistic.csv`, sort by `Total Time(us)` desc. Top 3-5 op types account for most of compute time.
2. For a specific op type, filter `kernel_details.csv` by `Type` = that op to see each instance's `Duration`, `Input Shapes`, `Input Data Types`, `Output Shapes`. Look for:
   - Outlier instances much slower than others with same shape
   - Unexpected dtypes (e.g., fp32 where you expected fp16/bf16)
   - Shapes that hit slow kernel paths
3. For host-side (Python / aten) overhead, open `operator_details.csv` and sort by `Host Self Duration(us)`. The `Call stack` column tells you where in the code it was launched from.

### 3. Communication-bound: is overlap broken?

1. From `step_trace_time.csv`, the ratio `Communication(Not Overlapped) / Communication Total` tells you how much of comm is exposed. High ratio → overlap is broken.
2. `kernel_details.csv` — filter rows with `Accelerator Core` = `HCCL`. Large `Wait Time` on an HCCL kernel means the stream was blocked before the comm even started (usually waiting on a compute op upstream).
3. `communication.json` has per-op `start_time`, `duration`, `wait_time` — use to line up specific comm ops with their compute neighbors.
4. `communication_matrix.json` has per-rank-pair bandwidth. If one rank pair has much lower bandwidth than others, suspect topology / NIC / straggler rank.

### 4. Memory-bound / OOM

1. `memory_record.csv` is a time series: `Timestamp`, `Total Allocated`, `Total Reserved`, per component (PTA / GE). Peak `Total Allocated` is the interesting number.
2. Cross-reference timestamp of the peak with `kernel_details.csv` to find which op was running.
3. `operator_memory.csv` lists each alloc: `Size(KB)`, `Allocation Time`, `Release Time`. Long-lived tensors = large `Release Time - Allocation Time`. These are candidates for activation checkpointing or earlier free.

### 5. Per-layer breakdown

Transformer layers have repeating op patterns. To get per-layer timing:

1. In `kernel_details.csv`, find the first op that repeats once per layer (e.g., the first `Cast` of each layer, or a RMSNorm). The example in the source wiki uses `Cast60` as the first op of each layer.
2. The `start time` of that op at layer N, minus its `start time` at layer N+1, gives the wall-clock time for layer N.
3. Using those boundaries, you can slice `kernel_details.csv` per layer and also look up the memory band in `memory_record.csv` for the same time window.

### 6. Visual inspection: `trace_view.json`

Large file — **do not** load with naive `json.load`; it may OOM. Either:
- Open it directly in Perfetto UI (https://ui.perfetto.dev) — drag & drop.
- Or stream-parse with `ijson` if you need to extract specific events programmatically.

Perfetto view is the fastest way to spot gaps, serialization of comm/compute, unexpected syncs.

## Quick Python helpers

For the csv files, use pandas. Typical starting snippet:

```python
import pandas as pd
ROOT = "path/to/<worker>_<ts>_ascend_pt/ASCEND_PROFILER_OUTPUT"

step = pd.read_csv(f"{ROOT}/step_trace_time.csv")
print(step)                                # quick sanity check on step-level time

op = pd.read_csv(f"{ROOT}/op_statistic.csv")
print(op.sort_values("Total Time(us)", ascending=False).head(10))

kd = pd.read_csv(f"{ROOT}/kernel_details.csv")
# Top 10 kernel instances
print(kd.sort_values("Duration(us)", ascending=False).head(10)
        [["Name", "Type", "Duration(us)", "Wait Time(us)", "Input Shapes"]])

# HCCL-only
hccl = kd[kd["Accelerator Core"] == "HCCL"]
```

For multi-card runs, each rank has its own `_ascend_pt` directory. Compare `step_trace_time.csv` across ranks — a straggler rank usually has higher `Computing` or larger `Free`.

## References

- [references/collection.md](references/collection.md) — how to actually collect profiling data (3 methods)
- [references/output-files.md](references/output-files.md) — field-by-field description of every output file
- [references/analysis-patterns.md](references/analysis-patterns.md) — worked examples of common investigations (poor overlap, host-bound, memory peak, straggler rank)

## Reporting back to the user

When giving the user conclusions, structure the answer as:

1. **What the data shows** (specific numbers from specific files, e.g., "step time 42ms, of which 18ms is exposed HCCL")
2. **Likely root cause** (a hypothesis grounded in which file told you this)
3. **What to try next** (one concrete action, e.g., "enable compute-comm overlap for allreduce", "rerun with larger batch", "check rank 3 — it's a straggler")

Do not dump every metric — a fresh engineer wants a clear "here's what's wrong" with the receipts, not a full table readout.
