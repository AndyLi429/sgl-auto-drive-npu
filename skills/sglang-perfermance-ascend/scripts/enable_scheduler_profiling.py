"""
Enable torch_npu profiling around the SGLang scheduler's event loop, without
editing scheduler.py.

This is the script equivalent of the old scheduler_profiling.patch — Strategy B
("NPU operator profiling, no stack") in the sglang-perfermance-ascend skill.
Prefer this over the patch: it is decoupled from scheduler.py line numbers and
survives upstream refactors.

How it works
------------
Monkey-patches `Scheduler.event_loop_normal` and `Scheduler.event_loop_overlap`
to wrap their per-batch execution with `torch.profiler.profile`. Only rank 0
profiles. Profiling activates when a batch of the configured stage
(prefill/decode) with at least `prof_bs` requests is observed.

Activation (two options)
------------------------
Option 1 — preload via PYTHONSTARTUP-style import:

    SGLANG_PROF_ENABLE=1 \
    SGLANG_PROF_STAGE=prefill \
    SGLANG_PROF_BS=1 \
    SGLANG_PROF_STEP=10 \
    SGLANG_PROF_OUTDIR=./profiling \
    python -c "import enable_scheduler_profiling; import runpy; \
               runpy.run_module('sglang.launch_server', run_name='__main__')" \
        -- --model-path ... --tp 8

Option 2 — import once from inside any module loaded before the server starts
(e.g., a small wrapper script or a sitecustomize.py on PYTHONPATH):

    import enable_scheduler_profiling  # noqa: F401

Environment variables
---------------------
SGLANG_PROF_ENABLE   "1" to activate. Default: "0" (no-op import).
SGLANG_PROF_STAGE    "prefill" or "decode". Default: "prefill".
SGLANG_PROF_BS       Minimum batch size that triggers capture. Default: 1.
SGLANG_PROF_STEP     Number of qualifying batches to profile. Default: 10.
SGLANG_PROF_OUTDIR   Output directory for the trace. Default: "./profiling".
SGLANG_PROF_RANK     Which TP rank profiles. Default: 0.
SGLANG_PROF_ACTIVITIES   Comma list: "cpu", "npu". Default: "cpu" (matches
                         the original patch; add "npu" for device-side too).
SGLANG_PROF_WITH_STACK   "1" to record Python stacks. Default: "0" (Strategy B).
                         For Strategy A (CPU/framework with stack) prefer
                         SGLang's `/start_profile` HTTP endpoint instead.

What you get
------------
A `<host>_<pid>_<ts>_ascend_pt/` directory under SGLANG_PROF_OUTDIR per the
torch_npu tensorboard handler. Feed it into the sglang-perfermance-ascend
skill's analysis flow (start with `step_trace_time.csv`).

Caveats
-------
- Adds non-trivial overhead while active — revert / unset SGLANG_PROF_ENABLE
  before normal benchmarking.
- `Scheduler.event_loop_normal` and `event_loop_overlap` must exist on the
  Scheduler class at import time. If SGLang renames them, update the
  `_TARGETS` list below.
- One-shot: the profiler stops after `prof_step` qualifying batches and does
  not restart in the same process.
"""

from __future__ import annotations

import logging
import os
from typing import Callable

logger = logging.getLogger("sglang.scheduler_profiling")


def _env_bool(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _build_profiler():
    """Construct the torch.profiler.profile object, matching the original patch."""
    import torch
    import torch.profiler as tprof

    activities_raw = os.environ.get("SGLANG_PROF_ACTIVITIES", "cpu").lower()
    activity_map = {
        "cpu": tprof.ProfilerActivity.CPU,
        "npu": getattr(tprof.ProfilerActivity, "NPU", None)
        or getattr(tprof.ProfilerActivity, "PrivateUse1", None),
    }
    activities = [
        activity_map[a.strip()]
        for a in activities_raw.split(",")
        if a.strip() in activity_map and activity_map[a.strip()] is not None
    ]
    if not activities:
        activities = [tprof.ProfilerActivity.CPU]

    outdir = os.environ.get("SGLANG_PROF_OUTDIR", "./profiling")
    os.makedirs(outdir, exist_ok=True)

    prof_step = _env_int("SGLANG_PROF_STEP", 10)

    return torch.profiler.profile(
        activities=activities,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(outdir),
        schedule=torch.profiler.schedule(
            wait=1, warmup=1, active=max(1, prof_step - 2), repeat=1, skip_first=1
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=_env_bool("SGLANG_PROF_WITH_STACK"),
        with_flops=False,
        with_modules=False,
    )


def _wrap_event_loop(orig: Callable) -> Callable:
    """Return a replacement event-loop method that drives the profiler."""
    target_stage = os.environ.get("SGLANG_PROF_STAGE", "prefill").lower()
    target_rank = _env_int("SGLANG_PROF_RANK", 0)
    prof_bs = _env_int("SGLANG_PROF_BS", 1)
    prof_step = _env_int("SGLANG_PROF_STEP", 10)

    def wrapped(self, *args, **kwargs):
        # Only the chosen rank profiles.
        if getattr(self, "tp_rank", 0) != target_rank:
            return orig(self, *args, **kwargs)

        import torch

        try:
            import torch_npu  # noqa: F401
            has_npu = True
        except ImportError:
            has_npu = False

        prof = _build_profiler()
        state = {"cnt": 0, "started": False, "stopped": False}

        # Hook run_batch to drive prof.start / prof.step / prof.stop in lockstep
        # with the original patch's logic.
        orig_run_batch = self.run_batch

        def run_batch_with_prof(batch):
            is_target_stage = (
                target_stage == "decode" and batch.forward_mode.is_decode()
            ) or (
                target_stage == "prefill" and batch.forward_mode.is_extend()
            )

            if (
                not state["stopped"]
                and is_target_stage
                and len(batch.reqs) >= prof_bs
                and state["cnt"] == 0
            ):
                logger.warning(
                    "[sglang-prof] starting profiler (stage=%s bs>=%d step=%d outdir=%s)",
                    target_stage, prof_bs, prof_step,
                    os.environ.get("SGLANG_PROF_OUTDIR", "./profiling"),
                )
                prof.start()
                state["started"] = True
                state["cnt"] = 1
            elif state["started"] and not state["stopped"] and is_target_stage:
                state["cnt"] += 1

            result = orig_run_batch(batch)

            if state["started"] and not state["stopped"]:
                if state["cnt"] >= prof_step and is_target_stage:
                    if has_npu:
                        torch.npu.synchronize()
                    prof.stop()
                    state["stopped"] = True
                    logger.warning("[sglang-prof] profiler stopped after %d batches", state["cnt"])
                elif is_target_stage:
                    prof.step()

            return result

        self.run_batch = run_batch_with_prof  # type: ignore[assignment]
        try:
            return orig(self, *args, **kwargs)
        finally:
            self.run_batch = orig_run_batch  # type: ignore[assignment]
            if state["started"] and not state["stopped"]:
                try:
                    prof.stop()
                except Exception:
                    pass

    wrapped.__name__ = getattr(orig, "__name__", "wrapped")
    wrapped.__qualname__ = getattr(orig, "__qualname__", "wrapped")
    return wrapped


_TARGETS = ("event_loop_normal", "event_loop_overlap")


def install() -> bool:
    """Install the monkey patch. Returns True if at least one target was wrapped."""
    if not _env_bool("SGLANG_PROF_ENABLE"):
        return False

    try:
        from sglang.srt.managers.scheduler import Scheduler
    except Exception as e:
        logger.warning("[sglang-prof] cannot import Scheduler yet: %s", e)
        return False

    wrapped_any = False
    for name in _TARGETS:
        orig = getattr(Scheduler, name, None)
        if orig is None or getattr(orig, "_sglang_prof_wrapped", False):
            continue
        new = _wrap_event_loop(orig)
        new._sglang_prof_wrapped = True  # type: ignore[attr-defined]
        setattr(Scheduler, name, new)
        wrapped_any = True
        logger.warning("[sglang-prof] wrapped Scheduler.%s", name)

    if not wrapped_any:
        logger.warning("[sglang-prof] no event loop targets wrapped — check SGLang version")
    return wrapped_any


# Auto-install on import so callers only need `import enable_scheduler_profiling`.
install()
