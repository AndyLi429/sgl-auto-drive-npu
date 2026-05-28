# Common Analysis Patterns

Worked templates for the investigations you'll do most often. Each starts from `step_trace_time.csv` and drills down.

## Pattern 1: "My SGLang decode is slow — why?"

```python
import pandas as pd
ROOT = ".../ASCEND_PROFILER_OUTPUT"

step = pd.read_csv(f"{ROOT}/step_trace_time.csv")
print(step)
```

Read the one-step row. Compute:
- `compute_ratio = Computing / Stage Time`
- `exposed_comm_ratio = Communication(Not Overlapped) / Stage Time`
- `bubble_ratio = Bubble / Stage Time`
- `free_ratio = Free / Stage Time`

Now branch:
- `compute_ratio` is dominant → Pattern 2 (compute-bound)
- `exposed_comm_ratio` > ~20% → Pattern 3 (comm-bound)
- `free_ratio` or `bubble_ratio` large → Pattern 4 (host-bound / bubble)

## Pattern 2: Compute-bound → find the hot op

```python
op = pd.read_csv(f"{ROOT}/op_statistic.csv")
top = op.sort_values("Total Time(us)", ascending=False).head(5)
print(top[["Type", "Count", "Total Time(us)", "Avg Time(us)", "Ratio(%)"]])
```

Pick the #1 type. Drill into per-instance:

```python
kd = pd.read_csv(f"{ROOT}/kernel_details.csv")
hot = kd[kd["Type"] == top.iloc[0]["Type"]]
print(hot.sort_values("Duration(us)", ascending=False).head(10)
        [["Name", "Duration(us)", "Input Shapes", "Input Data Types"]])
```

What to look for:
- Same shape, different duration → scheduling issue, not the op itself
- Unexpected dtype (fp32 when it should be bf16) → cast missing or promoted
- Very small shapes × high count → kernel launch overhead; consider fusion

Common SGLang cases:
- Attention kernel slow on specific seqlen → check if fused attention is actually hitting the fast path
- MatMul with non-aligned M/N/K → shape-driven slow tile selection

## Pattern 3: Comm-bound or broken overlap

```python
hccl = kd[kd["Accelerator Core"] == "HCCL"]
print(hccl[["Name", "Type", "Duration(us)", "Wait Time(us)"]]
        .sort_values("Duration(us)", ascending=False).head(10))
```

- Large `Duration` on AllReduce/AllGather → either genuinely slow link, or large payload. Cross-check with `communication_matrix.json` for bandwidth.
- Large `Wait Time` on an HCCL kernel → the comm stream is blocked before comm even starts. The op upstream on the same stream is the real problem.
- `Communication(Not Overlapped)` high but individual HCCL durations are normal → overlap not happening. SGLang-specific checks: TP all-reduce scheduling, whether comm is launched on a separate stream.

For rank comparison (multi-card):

```python
# Across all ranks, read each step_trace_time.csv and diff
import glob
for p in sorted(glob.glob(".../rank*/ASCEND_PROFILER_OUTPUT/step_trace_time.csv")):
    s = pd.read_csv(p)
    print(p, s.iloc[0][["Computing", "Communication(Not Overlapped)", "Stage Time"]].to_dict())
```

A rank with notably higher `Computing` than others is a straggler — everyone else waits at the next collective.

## Pattern 4: Host-bound / bubble

`Free` large or `Bubble` large and `Computing` is fine → NPU is sitting idle waiting for the host.

```python
det = pd.read_csv(f"{ROOT}/operator_details.csv")
print(det.sort_values("Host Self Duration(us)", ascending=False).head(10)
        [["Name", "Host Self Duration(us)", "Host Total Duration(us)", "Call stack"]])
```

The `Call stack` column tells you where in user code the time is going. Common SGLang offenders:
- Python-side token sampling / logits processing between decode steps
- Tensor copies / reshapes without fused alternatives
- Scheduler overhead in the request manager

Also open `trace_view.json` in Perfetto and look for gaps between consecutive NPU kernels — you'll see the host timeline stretched across the gap.

## Pattern 5: Memory peak / OOM

```python
mem = pd.read_csv(f"{ROOT}/memory_record.csv")
peak = mem.sort_values("Total Allocated(MB)", ascending=False).head(1)
print(peak)
peak_ts = peak.iloc[0]["Timestamp"]  # or whatever the column is called
```

Find what was running at `peak_ts`:

```python
kd = pd.read_csv(f"{ROOT}/kernel_details.csv")
around = kd[(kd["Start Time(us)"] < peak_ts) & 
            (kd["Start Time(us)"] + kd["Duration(us)"] > peak_ts)]
print(around)
```

Then look at `operator_memory.csv` for long-lived allocations (large `Release Time - Allocation Time`). These are the candidates for:
- Activation checkpointing
- Earlier free / smaller cache
- Smaller batch / kv-cache page size

## Pattern 6: Per-layer timing

Transformer layers repeat the same op sequence. To split the kernel list into per-layer segments:

```python
kd = pd.read_csv(f"{ROOT}/kernel_details.csv")

# Find a kernel that happens once per forward pass per layer.
# Cast ops named CastN with sequential N at fixed spacing work well,
# as do RMSNorm or QKV-proj MatMuls. Inspect first.
anchor_type = "Cast"   # adjust per model
cands = kd[kd["Type"] == anchor_type].sort_values("Start Time(us)")

# Inspect the series; the per-layer repeats will show a regular period.
```

Once you have layer boundaries, slice kd by `Start Time(us)` range to get each layer's op list and total duration.

## Pattern 7: Before / after comparison

User made a change (fused op, new kernel, new config) and wants to verify improvement:

1. Collect profiles both runs with identical settings (same step count, same batch).
2. Diff `step_trace_time.csv` first — did `Stage Time` drop? By how much?
3. Diff `op_statistic.csv` by op type to see which type's `Total Time` changed.
4. If the claimed fix is for a specific op, check that op in `kernel_details.csv` — did `Duration` drop? Did `Count` change?

Always report the deltas, not just the new numbers.

## Tips that save time

- Always skip the first step. Warmup artifacts are not representative.
- For multi-rank runs, the slowest rank determines the step time. Don't draw conclusions from one rank.
- `trace_view.json` is the fastest visual sanity check — a 30-second look in Perfetto often reveals what an hour of CSV analysis misses.
- If numbers in CSVs look suspicious, cross-check with Perfetto before chasing a ghost.
