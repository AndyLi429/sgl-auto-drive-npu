# Collecting Ascend Profiling Data

SGLang on Ascend has **two distinct profiling strategies**. Pick by the question you're answering, and as a rule of thumb run them in this order — **CPU/framework first, NPU operator second** — because Python/scheduler stalls show up as NPU idle gaps and will mislead any device-side analysis you do before clearing them.

| | Strategy A — Framework / CPU (with stack) | Strategy B — NPU operator (no stack) |
|---|---|---|
| Goal | Find framework / scheduler / Python overhead | Find slow NPU kernels, bad shapes, HCCL waits |
| Primary focus | **CPU / host side** | **NPU / device side** |
| Collected via | SGLang's built-in profiler (`/start_profile` HTTP endpoint, or `python -m sglang.profiler`) | `torch_npu.profiler` patched into the scheduler |
| `with_stack` | `True` — Python call stacks recorded | `False` — keeps trace small and accurate |
| What to look at after | `operator_details.csv` Host Self Duration + Call stack; host-side gaps in `trace_view.json`; `Free` in `step_trace_time.csv` | `kernel_details.csv`, `op_statistic.csv`, `communication.json`, device tracks in `trace_view.json` |

## Strategy A — SGLang built-in profiler (framework / CPU, with stack)

Use this **first**. Launch the SGLang server normally, then trigger profiling via the HTTP endpoint while running real traffic:

```bash
# Start profiling (on the rank-0 host of the running server)
curl -X POST http://127.0.0.1:30000/start_profile \
     -H 'Content-Type: application/json' \
     -d '{"output_dir": "/tmp/sglang_cpu_prof", "num_steps": 5, "with_stack": true}'

# ... send a small representative workload (e.g., bench_serving with a handful of prompts) ...

curl -X POST http://127.0.0.1:30000/stop_profile
```

Or equivalently, drive it from a script:

```bash
python -m sglang.profiler --url http://127.0.0.1:30000 \
                          --output-dir /tmp/sglang_cpu_prof \
                          --num-steps 5 \
                          --profile-by-stage  # captures prefill and decode separately
```

Notes:
- `with_stack=True` is the whole point — it gives you the Python call stack for each host-side op, so `operator_details.csv → Call stack` actually tells you which scheduler / tokenizer / dispatch code is hot.
- Keep the workload small (a handful of requests). Stack-enabled traces grow fast.
- This is the only strategy that exposes the SGLang Python scheduler loop, request-manager handoffs, and dispatcher costs in a readable way.

## Strategy B — torch_npu profiler wrapping the scheduler (NPU operators, no stack)

Use this **after** Strategy A is clean. It captures NPU device-side operator detail with stacks disabled (smaller, more accurate device timing).

### Recommended: import the helper script (no source edits)

The skill ships a monkey-patch helper at [`scripts/enable_scheduler_profiling.py`](scripts/enable_scheduler_profiling.py). Importing it before the server boots wraps `Scheduler.event_loop_normal` / `event_loop_overlap` with `torch.profiler.profile` — same behavior as the old patch, but decoupled from `scheduler.py` line numbers so it survives upstream refactors.

```bash
# 1. Put the helper on PYTHONPATH (or copy it next to your launch script)
export PYTHONPATH="$HOME/.claude/skills/sglang-perfermance-ascend/scripts:$PYTHONPATH"

# 2. Configure via env vars
export SGLANG_PROF_ENABLE=1
export SGLANG_PROF_STAGE=prefill        # or "decode"
export SGLANG_PROF_BS=1                 # min batch size to start capture
export SGLANG_PROF_STEP=10              # number of qualifying batches
export SGLANG_PROF_OUTDIR=./profiling
export SGLANG_PROF_ACTIVITIES=cpu,npu   # add npu for device-side; default "cpu"
export SGLANG_PROF_WITH_STACK=0         # Strategy B: keep stacks off

# 3. Launch SGLang so the helper imports first
python -c "import enable_scheduler_profiling; \
           import runpy; runpy.run_module('sglang.launch_server', run_name='__main__')" \
    -- --model-path <...> --tp 8
```

Alternative wiring: drop a one-liner `import enable_scheduler_profiling` into any module SGLang loads early (e.g., a `sitecustomize.py` on `PYTHONPATH`), then just `export SGLANG_PROF_ENABLE=1` before `python -m sglang.launch_server ...`.

Profiling activates automatically on rank 0 (override with `SGLANG_PROF_RANK`) once the chosen stage runs with `len(batch.reqs) >= SGLANG_PROF_BS`, and stops after `SGLANG_PROF_STEP` qualifying batches. Output: `<host>_<pid>_<ts>_ascend_pt/` under `SGLANG_PROF_OUTDIR`.

Unset `SGLANG_PROF_ENABLE` (or simply don't import the helper) before normal benchmarking — the active profiler adds overhead.

### Fallback: hand-patch the scheduler

If the helper can't wrap the targets (renamed methods on your SGLang fork, or you want to tweak the schedule), edit `scheduler.py` directly. The historical reference patch lives at `d:/Github/sglang/d:tmpscheduler_profiling.patch` and the helper script's `_wrap_event_loop` body documents the exact start/step/stop sequence. Apply via `git apply` and revert with `git apply -R` after the run.

### Ad-hoc cases

For profiling outside the scheduler (a single unit test, a kernel script), the standalone `torch_npu.profiler` block in Method 2 below works without any monkey patching.

## Other collection methods (when neither of the two above fits)

Three ways to collect raw profiling data. Method 2 (torch_npu profiler) is the underlying mechanism Strategy B uses; methods 1 and 3 are fallbacks for cases where you cannot patch SGLang.

## Method 1: Environment variables (CANN native)

Only use when you cannot modify the training / inference script.

```bash
export PROFILING_MODE=true
export PROFILING_OPTIONS='{"output":"/home/prof","training_trace":"on","task_trace":"on","fp_point":"","bp_point":"","aic_metrics":"PipeUtilization"}'
```

Then export the collected data:

```bash
python3 msprof.py export timeline -dir /home/user/profiler_data/PROF_XXX
python3 msprof.py export summary  -dir /home/user/profiler_data/PROF_XXX
```

Output lives under `PROF_XXX/` in raw CANN format. Less convenient to analyze than method 2.

## Method 2: torch_npu profiler (recommended)

Wrap the part of the code you want to measure. Collect at least Level1 data — Level0 is too coarse to find real bottlenecks.

```python
import torch_npu

experimental_config = torch_npu.profiler._ExperimentalConfig(
    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    # l2_cache=True,   # enable if you suspect L2 cache issues; adds l2_cache.csv
)

with torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU,
    ],
    schedule=torch_npu.profiler.schedule(
        wait=0, warmup=1, active=1, repeat=1, skip_first=0
    ),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("/home/wgw/prof"),
    experimental_config=experimental_config,
) as prof:
    for step in range(total_steps):
        run_one_step()
        prof.step()
```

Notes specific to SGLang:
- For inference benchmarking, wrap a single `generate` / decode loop, not the server boot.
- `warmup=1, active=1` means: skip 1 step, profile 1 step. Always profile a warm step, never step 0.
- Avoid profiling over many steps — the output files grow quickly and `trace_view.json` becomes unwieldy.

Level guide:
- `Level0` — basic op timing, no shape/dtype — too shallow.
- `Level1` — adds op details, HCCL, memory. **Default.**
- `Level2` — adds communication matrix detail, data preprocess, more. Use when debugging comm or preprocess.

## Method 3: `msprof` command (no code changes)

Launch any command under `msprof` to profile it end-to-end:

```bash
# Python example
msprof --application="python -m unittest test_unpad_paged_attention.TestUnpadPagedAttention.test_pa_fp16_case_decoder" \
       --out=./output

# C++ example
/opt/Ascend/ascend-toolkit/latest/tools/profiler/bin/msprof \
    --application="./out.exe $batch_size $seqlen" --out=./output
```

Good for quick one-off runs. Produces CANN-format output; convert with `msprof.py export` as in method 1.

## Output directory to hand off to analysis

After method 2, you will see:

```
/home/wgw/prof/<worker>_<pid>_<ts>_ascend_pt/
├── profiler_info.json
├── ASCEND_PROFILER_OUTPUT/     # ← point the analysis at this
├── FRAMEWORK/                   # raw framework data (optional, deleted if data_simplification=True)
└── PROF_<id>_<ts>_<hash>/       # raw CANN data
```

For multi-rank runs, one `_ascend_pt` directory per rank. Collect them all and compare across ranks.
