---
name: ascend-910c
description: SSH into the Ascend 910C node at 192.168.25.216, enter Docker container `lph-050600`, work in `/home/l00567497/sglang`, and use the ready 910C remote environment for SGLang development and validation. Use when a task needs remote NPU work, HCCL-aware tests, multi-NPU smoke tests, graph-mode checks, or a safe remote copy instead of local-only execution. Trigger this whenever the user asks to run, test, validate, benchmark, or smoke-test SGLang code on Ascend 910C / NPU hardware — even if they don't say "910C" explicitly. Also trigger for phrases like "在我的设备上跑", "远端跑一下", "在NPU上验证", "在910C上跑", "去lph-050600里跑", "用我的开发机", "on my NPU box", "on the Ascend node", or any reference to `lph-050600` or `192.168.25.216`.
---

# Ascend 910C

## Overview

Use this skill to do SGLang development on the Ascend 910C node through `192.168.25.216`.
The default container is `lph-050600` and the repo lives at `/home/l00567497/sglang`.
Prefer it whenever local validation is insufficient for NPU, HCCL, graph-mode, torch_npu, or other Ascend-backed SGLang behavior.

This environment is already prepared:

- container `lph-050600` is running on an SGLang-Ascend image
- the repo is cloned at `/home/l00567497/sglang`
- 8 × Ascend 910C NPUs are exposed inside the container as `/dev/davinci0` through `/dev/davinci7`
- HCCL device nodes (`/dev/davinci_manager`, `/dev/devmm_svm`, `/dev/hisi_hdc`) are mounted for collective communication
- CANN driver and `npu-smi` are available inside the container
- Ascend toolkit env can be sourced from `/usr/local/Ascend/ascend-toolkit/set_env.sh`

> **One-time setup (recommended):** add this to your local `~/.ssh/config` so every command below stays short:
> ```
> Host ascend_910c
>     HostName 192.168.25.212
>     User <your_user>
> ```
> Then replace every `<ssh_alias>` below with `ascend_910c`. Until you do, fall back to `ssh <user>@192.168.25.212`.

Hugging Face cache may not be mounted in this environment. If a workflow reads gated models from Hugging Face Hub, verify `HF_TOKEN` is set inside the container before starting. Interactive shells and non-interactive `docker exec ... bash -lc "<cmd>"` can behave differently — always verify with `echo ${HF_TOKEN:+set}`.

## Quick Start

1. Check the host, container, and NPU state.

```bash
ssh <ssh_alias> 'hostname && whoami'
ssh <ssh_alias> 'docker ps --format "table {{.Names}}\t{{.Status}}" | sed -n "1,20p"'
ssh <ssh_alias> 'docker exec lph-050600 npu-smi info'
```

2. Enter the container and repo.

```bash
ssh <ssh_alias> 'docker exec -it lph-050600 /bin/bash'
cd /home/l00567497/sglang
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
echo ${HF_TOKEN:+set}
```

If `HF_TOKEN` is unexpectedly missing in the current shell:

```bash
export HF_TOKEN=<your-hf-token>
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
```

For non-interactive `docker exec ... bash -lc "<cmd>"` runs, prefer exporting the
CANN env and the HF tokens inside the command itself instead of assuming the
shell startup path will populate them.

3. Pick a free NPU.

`npu-smi info` shows AICore Usage(%) and HBM-Usage(MB) per chip. A NPU with `0%` AICore and only a few MiB HBM is free.
Set `ASCEND_RT_VISIBLE_DEVICES=<npu_id>` for every NPU-backed validation command (this is the Ascend equivalent of `CUDA_VISIBLE_DEVICES`).

A quick "is this NPU idle?" probe:

```bash
ssh <ssh_alias> 'docker exec lph-050600 bash -lc "npu-smi info | head -40"'
```

For a single-line summary across all 8 NPUs:

```bash
ssh <ssh_alias> 'docker exec lph-050600 bash -lc "for i in 0 1 2 3 4 5 6 7; do npu-smi info -t usages -i \$i -c 0 2>/dev/null | head -5 ; echo --- ; done"'
```

4. This host currently does not provide an automatic `kill-idle` helper.

Do not assume you can reclaim other users' allocations. If the free NPU list is tight, re-check `npu-smi info`, choose another NPU, or coordinate before proceeding.

5. If the container is not running, start it first.

```bash
ssh <ssh_alias> 'docker start lph-050600'
```

## Safe Remote Workflow

1. Inspect the default repo before editing it.

```bash
ssh <ssh_alias> 'docker exec lph-050600 bash -lc "cd /home/l00567497/sglang && git branch --show-current && git status --short"'
```

2. Fast-forward `/home/l00567497/sglang` to the latest clean `main` before creating any validation worktree.

```bash
ssh <ssh_alias> 'docker exec lph-050600 bash -lc "cd /home/l00567497/sglang && git fetch origin && git checkout main && git pull --ff-only origin main"'
```

3. Avoid writing directly into `/home/l00567497/sglang` when it is dirty or when the local snapshot differs from the remote `HEAD`.

4. Prefer one of these isolation strategies.

**Detached worktree** for remote-only experiments:

```bash
ssh <ssh_alias> 'docker exec lph-050600 bash -lc "cd /home/l00567497/sglang && git worktree add --detach /tmp/sglang_validate_npu HEAD"'
```

**Stream the local working tree** into the container when validating the current local snapshot exactly:

```bash
COPYFILE_DISABLE=1 tar --exclude=.git -cf - . | \
  ssh <ssh_alias> 'docker exec -i lph-050600 sh -lc "rm -rf /tmp/sglang_local_validate && mkdir -p /tmp/sglang_local_validate && tar -xf - -C /tmp/sglang_local_validate"'
ssh <ssh_alias> 'docker exec lph-050600 bash -lc "find /tmp/sglang_local_validate -name '\''._*'\'' -delete"'
```

Use the streamed copy when the goal is "validate exactly what is in the local repo right now". For patch-oriented remote validation, another good option is:

- update remote `main`
- create a detached worktree from that clean commit
- stream or apply a focused local patch diff into the worktree only

That keeps `/home/l00567497/sglang` clean while still validating the exact local delta.

## Validation Workflow

1. Start with import or syntax-level checks (no NPU needed).

```bash
ssh <ssh_alias> 'docker exec lph-050600 bash -lc "cd /tmp/sglang_local_validate && python -m compileall python/sglang"'
```

2. Run targeted tests for the changed area.

```bash
ssh <ssh_alias> 'docker exec lph-050600 env PYTHONPATH=python bash -lc "cd /tmp/sglang_local_validate && pytest -q path/to/test.py -q"'
```

3. For NPU-backed changes, pin a free NPU explicitly with `ASCEND_RT_VISIBLE_DEVICES`.

```bash
ssh <ssh_alias> 'docker exec lph-050600 env ASCEND_RT_VISIBLE_DEVICES=0 PYTHONPATH=python bash -lc "cd /tmp/sglang_local_validate && pytest -q path/to/npu_test.py -q"'
```

4. For multi-NPU / HCCL work, set a visible device list and use the standard launch flow. Example for TP=4:

```bash
ssh <ssh_alias> 'docker exec lph-050600 env ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=python bash -lc "cd /tmp/sglang_local_validate && python -m sglang.launch_server --model-path <model_path> --tp 4 --device npu --port 30000"'
```

Common HCCL knobs (export inside the env block when needed):
- `HCCL_BUFFSIZE` — HCCL collective buffer size in MB
- `HCCL_CONNECT_TIMEOUT` — connect timeout in seconds
- `HCCL_EXEC_TIMEOUT` — kernel execution timeout in seconds
- `HCCL_DETERMINISTIC=true` — when chasing nondeterministic accuracy bugs

5. Use a real `.py` file with `if __name__ == "__main__":` for any flow that uses `multiprocessing.spawn`. It will fail if executed from stdin or unguarded top-level code. This bites especially hard with `DP` / `PD-disaggregation` launch paths on Ascend.

6. Attempt model-level or server-level smoke only after unit / kernel / targeted regression checks pass. Treat the following as separate failure classes from code regressions:
   - missing CANN / wrong driver version (`source set_env.sh` not run)
   - missing or mismatched `torch_npu` wheel
   - HCCL init failure (usually a device-visibility or `/dev/davinci_manager` mount issue)
   - checkpoint not present locally and HF_TOKEN missing

If a workflow reads from Hugging Face Hub, verify `HF_TOKEN` first and re-export it explicitly in the current shell or command when needed.

## Profiling and Accuracy Cross-Reference

This skill is the **runner**. Hand off to the right specialist skill for interpretation:

- **Performance work** → after capturing an Ascend PyTorch Profiler dump (the `*_ascend_pt` directory containing `kernel_details.csv`, `trace_view.json`, `step_trace_time.csv`), use the `sglang-ascend-perf-analysis` skill to find the bottleneck.
- **Accuracy regression** (gibberish output, dataset score drop, first-token mismatch, multi-batch divergence, DP/TP/PD/graph-mode/DeepEP/MTP-related divergence) → use the `sglang-accuracy-debugging` skill, which has the structured triage and bisection flow.

A typical profiler capture run on this host:

```bash
ssh <ssh_alias> 'docker exec lph-050600 env ASCEND_RT_VISIBLE_DEVICES=0 PYTHONPATH=python bash -lc "cd /tmp/sglang_local_validate && python <your_profiler_script>.py"'
# then pull the *_ascend_pt directory back to the local box for analysis
scp -r <ssh_alias>:/tmp/sglang_local_validate/<run_dir>_ascend_pt ./
```

## Cleanup

Remove temporary validation directories when finished so `/tmp` does not fill up:

```bash
ssh <ssh_alias> 'docker exec lph-050600 rm -rf /tmp/sglang_local_validate /tmp/sglang_validate_npu'
```