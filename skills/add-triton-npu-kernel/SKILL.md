---
name: add-triton-npu-kernel
description: Step-by-step guide for writing a lightweight JIT Triton kernel inside `sgl-kernel-npu` that compiles via Triton-Ascend and runs efficiently on Huawei Ascend NPU (910B/910C) vector cores. Use this whenever the user asks to add, port, or rewrite a Triton kernel for NPU — including phrases like "写一个 Triton 算子", "add a triton kernel for NPU", "fuse this on NPU with Triton", "port this kernel to Ascend", "triton.jit on 910", "tl.load on NPU", or when modifying any file under `python/sgl_kernel_npu/sgl_kernel_npu/{fla,norm,moe,activation,attention,mamba,sample}/*.py`. Triton-Ascend kernels look like CUDA Triton kernels but tune to a different hardware model (vector cores + Unified Buffer instead of SMs + shared memory) — do NOT blindly copy GPU launch configs.
---

# Adding a JIT Triton Kernel for Ascend NPU

This skill captures how this repo writes Triton kernels that compile through the **Triton-Ascend** backend and run on Ascend NPU vector cores. The goal is correctness first, then good vector-core utilization without exceeding Unified Buffer (UB) memory.

We'll use a tiny **element-wise scale** kernel (`y = x * factor`) as the running example.

## When to use Triton (vs. AscendC / torch_npu op)

- **Triton** (this skill): vector / elementwise / fused-reduction / lightweight epilogue kernels. Iterates fast, lives in Python, compiles on first use. The right default for fused norms, gating, activations, pre/post-processing around larger ops.
- **AscendC / torch_npu custom op**: heavy matmul (cube core), kernels that need explicit double-buffered pipelines between cube and vector, or fused-attention shapes that Triton-Ascend can't yet express. Lives in C++ and ships in a wheel — slow iteration.

If the kernel is mostly `load → math → reduce → store` on vector data, write it in Triton. If it needs cube/MMA, use AscendC.

---

## Mental model: how Ascend NPU differs from CUDA

Triton's Python API is the same, but the hardware it targets is not. Internalize this before you size grids or blocks:

| Concept | CUDA (GPU) | Ascend NPU |
|---|---|---|
| Parallel unit | SM with many warps | **Vector core** (or AI / cube core for matmul) — there are ~24–48 of them |
| On-chip memory | Shared memory + L1 (~100KB / SM) | **Unified Buffer (UB)** — about **1.5 MB per vector core** (`UNIFIED_BUFFER_SIZE = 1_572_864` bytes in this repo) |
| `num_warps` | Maps to GPU warps (32 lanes × N) | Largely ignored on Ascend; **set `num_warps=1`** unless you've measured something better |
| Grid sizing | Oversubscribe SMs heavily | **Persistent style**: launch one (or a few) programs per core, loop inside |
| Tensor cores | `tl.dot` → MMA | `tl.dot` goes through cube core — only use it for matmul-shaped problems |
| `tl.atomic_*` | Cheap | Avoid; prefer reduction trees and clean writes |
| `.item()` / D2H | Slow but tolerated | **D2H sync is brutal** — never inside a hot path; see project `CLAUDE.md` |

The two practical consequences:

1. **Pick the grid to match cores, not data.** Launch `num_vectorcore` (e.g. 48) programs and have each program iterate over its share of the work with `tl.range(pid, total, num_programs)`. This is how every kernel in `sgl_kernel_npu/` is written.
2. **Sum of all live tensors in the kernel × element-size must fit in UB.** With ~1.5 MB per core, a `BLOCK_SIZE × hidden_size × bf16` tile fills up fast. Tile your inner loop accordingly.

---

## Repo conventions

All Triton-NPU kernels live under `python/sgl_kernel_npu/sgl_kernel_npu/<domain>/`. Pick the domain that fits:

- `norm/` — RMSNorm, LayerNorm, fused-norm variants
- `activation/` — SwiGLU, GELU, etc.
- `fla/` — Gated Delta Net / linear attention pieces
- `moe/` — MoE gating, dispatch, combine epilogues
- `mamba/` — SSM / conv1d
- `attention/` — attention-adjacent fused kernels
- `sample/` — sampling / verification

The folder name dictates where tests go too: `tests/python/sgl_kernel_npu/test_<your_kernel>.py`.

### Required imports

```python
import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties
```

If you need Ascend-specific intrinsics (`extract_slice`, etc.):

```python
import triton.language.extra.cann.extension as al
```

`get_device_properties()` returns `(num_aicore, num_vectorcore)`. For pure vector kernels use `num_vectorcore`; for kernels routed through cube use `num_aicore`. **Don't hardcode** `48` or `24` — there's a reason a helper exists (910B vs 910C differ).

---

## Step 1: Write the kernel

Create `python/sgl_kernel_npu/sgl_kernel_npu/activation/scale.py`:

```python
import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def scale_kernel(
    x_ptr,
    y_ptr,
    factor,                       # runtime scalar — keep runtime unless specialization is needed
    num_elements,                 # runtime int
    BLOCK_SIZE: tl.constexpr,     # tile size each program processes per inner step
    NUM_PROGRAMS: tl.constexpr,   # total programs launched; constexpr so the loop bound is known
):
    pid = tl.program_id(0)

    # Persistent loop: each program covers a strided slice of the data.
    # This is the canonical pattern for this repo — one program per vector core,
    # work stays resident, no oversubscription.
    total_tiles = tl.cdiv(num_elements, BLOCK_SIZE)
    for tile in range(pid, total_tiles, NUM_PROGRAMS):
        offsets = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        # Promote to fp32 for the math. NPU vector units are happiest in fp32 for
        # transcendentals / divides; cast back at the store site.
        y = (x.to(tl.float32) * factor).to(y_ptr.dtype.element_ty)
        tl.store(y_ptr + offsets, y, mask=mask)


def scale(x: torch.Tensor, factor: float, out: torch.Tensor | None = None) -> torch.Tensor:
    assert x.is_contiguous(), "scale: input must be contiguous"
    if out is None:
        out = torch.empty_like(x)

    n = x.numel()
    _, num_vectorcore = get_device_properties()

    # Block size: keep the tile inside UB. For 1.5MB UB and bf16/fp16 inputs
    # with one input + one output live at once (~4 bytes/elt incl. fp32 promo),
    # tiles up to ~256K elements are safe; pick a power of two and bound by data.
    BLOCK_SIZE = min(triton.next_power_of_2(triton.cdiv(n, num_vectorcore)), 8192)
    BLOCK_SIZE = max(BLOCK_SIZE, 64)

    grid = (num_vectorcore,)
    scale_kernel[grid](
        x,
        out,
        factor,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_PROGRAMS=num_vectorcore,
        num_warps=1,           # NPU: leave at 1 unless profiling shows otherwise
    )
    return out
```

### What this template establishes

- **Persistent grid**: `grid = (num_vectorcore,)`, then loop `for tile in range(pid, total, NUM_PROGRAMS)`. Match this in every new kernel — see `mul_add.py`, `rmsnorm_bias.py`, `rmsnorm_without_weight.py`.
- **`NUM_PROGRAMS` as `tl.constexpr`**: lets the compiler unroll / bound the loop. The repo uses this idiom (e.g. `kernel_num` in `rmsnorm_without_weight.py`, `row_step = tl.num_programs(0)` in `mul_add.py`).
- **`num_warps=1`**: Triton-Ascend ignores most warp tuning; setting 4/8 wastes scheduling decisions. If a kernel really benefits from more, that should be measured and commented.
- **fp32 accumulation**: always promote before reduce / exp / log / divide. Cast back via `y_ptr.dtype.element_ty` to honor the caller's dtype.

---

## Step 2: Size the block — UB budget is the rule

The single biggest correctness/performance trap on Ascend is exceeding UB. Walk through the math for every kernel:

```
live_bytes ≈ Σ_t (tile_elements_t × bytes_per_elem_t × promotion_factor_t)
live_bytes  must be  ≲  0.95 × UNIFIED_BUFFER_SIZE   (≈ 1.5 MB)
```

`fused_gdn_gating.py` shows the explicit form of this calculation (`UNIFIED_BUFFER_SIZE = 1572864`, then derive `BLK_BATCHES` from `BLK_HEADS × element_size`). Use the same pattern when the kernel keeps multiple tensors live at once.

Rules of thumb:

- 2 live tiles, bf16, fp32 promo → ~6 bytes per element → keep tile ≤ ~200K elements.
- A row-major tile of shape `[BLK_BATCHES, HIDDEN]` must satisfy `BLK_BATCHES * HIDDEN * elem_size * num_live_tensors ≤ 0.95 * UB`.
- Always round `BLOCK_SIZE` to a power of two (`triton.next_power_of_2`) — Triton-Ascend tiling assumes this.
- For very large `hidden_size`, **don't** allocate a `[BLOCK_SIZE, hidden_size]` tile; split the column loop with `al.extract_slice` (see `rmsnorm_bias.py:43–69`).

---

## Step 3: Optional autotune

If a kernel has a clear shape-driven knob (e.g. block_l in `rmsnorm_without_weight.py`), use `@triton.autotune`. Keep configs small (5–6 entries) — every cache miss recompiles, and the Ascend compile is not free.

```python
@triton.autotune(
    configs=[triton.Config({"BLOCK_L": s}) for s in (64, 32, 16, 8, 4, 2)],
    key=["num_tokens"],
)
@triton.jit
def kernel(...):
    ...
```

Skip autotune if there's only one sensible block size — adds latency for no gain.

---

## Step 4: Wire it up

Re-export the public function in the parent `__init__.py` of the domain folder (look at the existing file to see the convention — single-line imports, no `__all__` churn). If the kernel is consumed from SGLang, add the dispatch wrapper there.

After editing source, **remind the user to rebuild / reinstall the wheel** if their environment uses an installed `sgl-kernel-npu` rather than an editable install — per project `CLAUDE.md`, wheel updates don't pick up source changes automatically.

---

## Step 5: Tests

Tests live in `tests/python/sgl_kernel_npu/test_<your_kernel>.py`. Mirror the structure in `test_fused_gdn_gating_without_sigmoid.py`:

- Build a CPU reference (`reference_<op>` function) in plain torch.
- Run the kernel on the NPU device.
- `torch.npu.synchronize()` before pulling results back to CPU.
- Compare with `torch.testing.assert_close(atol, rtol)` — typical tolerances: `atol=5e-2, rtol=1e-2` for bf16/fp16, much tighter for fp32.
- Cover at least: minimal (1 row), normal (batch 16–32), large batch, non-power-of-two shapes (to exercise the mask path).

Tests use `argparse` + a `main()` entry point — not pytest fixtures — because they're driven from a runner on the NPU host. Match that style unless you have a reason not to.

**D2H discipline (from `CLAUDE.md`):** the test driver code may call `.item()` / `.cpu()`, but the kernel and its launcher must not. If you find yourself doing `.item()` on intermediate tensors to drive control flow, redesign — that sync will dominate runtime on the model side.

---

## Performance checklist before declaring done

Run through this list — it catches most of what we've broken historically:

- [ ] Grid is `(num_vectorcore,)` (or `(num_aicore,)` for cube-bound work), **not** `triton.cdiv(n, BLOCK_SIZE)`.
- [ ] Inside the kernel: `for ... in range(pid, total, NUM_PROGRAMS)` persistent loop.
- [ ] `num_warps=1` (unless explicitly tuned).
- [ ] `BLOCK_SIZE` rounded to power of two via `triton.next_power_of_2`.
- [ ] UB budget worked out on paper (or in a comment) and fits in ~1.5 MB.
- [ ] fp32 promotion for any reduce / div / `exp` / `log` / `rsqrt`.
- [ ] Store cast back via `out_ptr.dtype.element_ty`.
- [ ] No `.item()`, `_local_scalar_dense`, `torch.is_nonzero` in the launcher.
- [ ] No `torch.cuda.*` calls — use `torch.npu.synchronize()`, guard with try/except if shared code must run on both.
- [ ] Launcher pre-allocates outputs with `torch.empty(...)`, doesn't realloc in a loop.
- [ ] Public function added to the relevant `__init__.py` (if applicable).
- [ ] Test file exists under `tests/python/sgl_kernel_npu/` with CPU reference + at least 4 shape cases.

---

## Diagnosing problems

Per project `CLAUDE.md`: before changing code, list top-3 root-cause hypotheses with evidence and pick which to verify first. Common Ascend-Triton failure modes:

- **Silent wrong numerics** → almost always missing fp32 promo on a reduce, or a `mask` mismatch on the boundary tile.
- **OOM / UB overflow at compile or launch** → block size × live tensors exceeds UB. Shrink the tile or split with `al.extract_slice`.
- **Compile slow / cache thrash** → too many autotune configs; or the launcher is passing non-constexpr values that should be constexpr.
- **Kernel runs but is 10× slower than expected** → check `num_warps != 1`, check that the grid actually matches `num_vectorcore` (a typo can give you 1 program), check whether you're causing a hidden D2H by passing a Python scalar derived from `.item()`.
- **Works on 910B, breaks on 910C (or vice versa)** → hardcoded `48` somewhere. Use `get_device_properties()`.

---

## Reference reading in this repo

Start here when you need a working example to copy from:

- `python/sgl_kernel_npu/sgl_kernel_npu/moe/mul_add.py` — minimal persistent kernel template.
- `python/sgl_kernel_npu/sgl_kernel_npu/norm/rmsnorm_without_weight.py` — autotune + reduction + 2D tile.
- `python/sgl_kernel_npu/sgl_kernel_npu/norm/rmsnorm_bias.py` — `al.extract_slice` for column tiling + optional int8 quant epilogue.
- `python/sgl_kernel_npu/sgl_kernel_npu/fla/fused_gdn_gating.py` — explicit UB-budget arithmetic to derive `BLK_BATCHES`.
- `python/sgl_kernel_npu/sgl_kernel_npu/activation/swiglu_oai.py` — 1D persistent kernel with miniblock tiling.
- `python/sgl_kernel_npu/sgl_kernel_npu/utils/triton_utils.py` — `get_device_properties` and where to add shared helpers.
- `tests/python/sgl_kernel_npu/test_fused_gdn_gating_without_sigmoid.py` — test scaffold to copy.

---

## Summary of files you'll touch

```
python/sgl_kernel_npu/sgl_kernel_npu/<domain>/<your_kernel>.py   # NEW: kernel + launcher
python/sgl_kernel_npu/sgl_kernel_npu/<domain>/__init__.py        # EDIT: re-export
tests/python/sgl_kernel_npu/test_<your_kernel>.py                # NEW: CPU-ref test
```
