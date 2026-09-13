---
name: sglang-ascend-pr-review
description: Review pull requests and local diffs for sgl-project/sglang and sgl-project/sgl-kernel-npu through an Ascend NPU lens. Produce a concise Chinese review report with severity-sorted findings about NPU correctness, numerical accuracy, performance, distributed communication, CANN/torch-npu compatibility, tests, and maintainability. Use whenever the user asks to review an SGLang or sgl-kernel-npu PR/diff involving Ascend, NPU, torch_npu, CANN, AscendC, Triton-Ascend, HCCL, DeepEP-Ascend, attention, MoE, kernels, graph capture, or an NPU backend path.
---

# SGLang and Ascend NPU PR Review

Review the actual changed execution path, not just the diff hunk. A review finding is useful only when it connects a concrete code location to a concrete NPU triggering condition, runtime impact, and an actionable correction.

## Scope

Support:

- GitHub PR number or URL for `sgl-project/sglang` or `sgl-project/sgl-kernel-npu`;
- a local branch, commit range, or diff in either repository;
- a focused directory, symbol, kernel, or backend path supplied by the user.

Default to Simplified Chinese. This skill reviews code; do not modify it, post GitHub comments, change PR state, or claim hardware validation that was not run.

## Evidence collection

For a GitHub PR, prefer `gh` and collect metadata, diff, files, discussion, inline review comments, top-level reviews, and CI:

```bash
gh pr view PR_NUMBER --repo OWNER/REPO \
  --json number,title,body,author,state,createdAt,updatedAt,url,labels,additions,deletions,\
baseRefName,headRefName,isDraft,comments,reviews
gh pr diff PR_NUMBER --repo OWNER/REPO
gh api --paginate repos/OWNER/REPO/pulls/PR_NUMBER/files
gh api --paginate repos/OWNER/REPO/pulls/PR_NUMBER/comments
gh api --paginate repos/OWNER/REPO/issues/PR_NUMBER/comments
gh pr checks PR_NUMBER --repo OWNER/REPO
```

For local changes, record repository, base revision, HEAD, changed files, and diff:

```bash
git status --short --branch
git rev-parse HEAD
git diff --stat BASE...HEAD
git diff --find-renames BASE...HEAD
git log --oneline BASE..HEAD
```

If `.codegraph/` exists, use CodeGraph before text search to trace changed symbols and callers. Otherwise use `rg` and read the full relevant function, schema, wrapper, implementation, and focused test—not only changed lines.

Record missing evidence explicitly: truncated diff, unavailable comments, unrun CI, unknown CANN version, unavailable NPU device, or absent accuracy/performance results.

## Ascend relevance gate

Classify the change before deep review.

- **完全相关**: changes NPU runtime/backend, `torch_npu`, CANN/ACLNN, AscendC, Triton-Ascend, HCCL, DeepEP-Ascend, NPU tests, NPU CI, or NPU packaging/build paths.
- **部分相关**: shared scheduler, attention, model, cache, distributed, or public API code can affect NPU dispatch or behavior.
- **不相关**: no NPU-exposed path changed. Perform a lightweight generic review and say which Ascend checks were skipped.

Useful NPU markers include `npu`, `ascend`, `torch_npu`, `aclnn`, `aclrt`, `CANN`, `AscendC`, `HCCL`, `deep_ep`, `fuseep`, `op_host`, `op_kernel`, `tiling`, `pytorch_extensions.cpp`, `torch.ops.npu`, and `sgl_kernel_npu`.

## Understand the change before judging it

Answer these questions from code and PR evidence:

1. What changes computationally or operationally: dispatch, tensor transformation, kernel algorithm, communication schedule, memory ownership, or configuration?
2. Which hardware, chip family, CANN/torch-npu version, dtype/quantization mode, model, and parallel mode are in scope?
3. Which API or schema changes: Python wrapper, `torch.ops` schema, pybind/C++ host, AscendC/Triton kernel, environment variable, or configuration?
4. If there is a performance claim, what mechanism could cause it—less memory traffic, fewer launches, different tiling, more overlap, or a faster operator path—and where is the comparable measurement?
5. Which input makes the changed path execute: shape, dtype, chip, world size, attention/MoE configuration, graph mode, or fallback condition?

If the fifth question cannot be answered, do not emit a blocking correctness or performance finding. Downgrade it to a question or omit it.

## Trace contracts across boundaries

For an NPU path, follow as much of this chain as the PR touches:

```text
SGLang server argument / environment
  -> backend registry or factory
  -> scheduler, attention, MoE, or model consumer
  -> sgl_kernel_npu Python wrapper or torch.ops.npu schema
  -> C++ host / pybind / tiling
  -> AscendC or Triton-Ascend device implementation
  -> focused unit, accuracy, and NPU integration test
```

At each changed boundary check tensor device, shape, layout, dtype/quantization, aliasing or mutation, workspace lifetime, stream/event ordering, error propagation, and chip/version guards. Do not assume every `torch.ops.npu.*` symbol belongs to `sgl-kernel-npu`; confirm its registration source.

## Review categories

Apply categories selected by the changed path. All reviews include correctness, tests, and maintenance checks.

| PR touches | Required focus |
|---|---|
| NPU runtime/backend, backend selection, shared dispatch | device guard, fallback, backend registration, shared-path regression, configuration defaults |
| Attention, MLA, KV cache, graph capture | shape/layout, causal/mask semantics, prefill/decode split, cache ownership, graph-safe allocation, synchronization |
| MoE, DeepEP, HCCL, FuseEP, distributed path | rank/world-size assumptions, token order, buffer ownership, collective matching, stream ordering, timeout/error path |
| AscendC operator, ACLNN, C++ host, tiling | schema/wrapper agreement, workspace/tiling size, dtype/shape constraints, UB/vector-core assumptions, error checking, chip guard |
| Triton-Ascend kernel | tensor layout/stride, masks and boundary handling, program-id assumptions, dtype conversion, autotune keys, NPU-specific launch constraints |
| Quantization or mixed precision | scale/zero-point layout, accumulation dtype, rounding/saturation, fp8/int8/bf16 fallback, tolerance and reference test |
| Build, packaging, CI, dependencies | CANN/torch-npu compatibility, conditional import/build, wheel contents, test matrix, no accidental CUDA-only dependency |

## High-risk review checks

### Correctness and accuracy

- Check shape, stride, dtype, contiguous assumptions, empty inputs, odd sizes, and dynamic-shape boundaries against the kernel/wrapper contract.
- Check that output order, indices, expert routing, cache slots, and rank-local buffers preserve the caller's semantic order.
- Check every changed dtype conversion, scale, mask, padding, and reduction for an appropriate reference or tolerance-based test.
- Check fallbacks return equivalent semantics and do not silently select a CUDA/CPU path for NPU input.

### Async execution and resource lifetime

- Flag unsafe synchronization only when a concrete hot-path trigger exists; `.item()`, tensor truth tests, `_local_scalar_dense`, host reads, and explicit synchronize calls warrant inspection.
- Check stream/event dependencies and that tensors, workspace, host metadata, and communication buffers outlive asynchronous use.
- Check graph-capture paths for allocation, dynamic Python control flow, address stability, and fallback behavior.

### Performance

- Check avoidable device-host copies, repeated allocations, layout conversions, redundant casts, unfused small ops, and exposed communication.
- Require benchmark provenance: command, model, hardware/chip, CANN/torch-npu version, workload, batch/concurrency, baseline, and repeated-run variance when a performance claim is made.
- Treat a benchmark number without source/configuration as unverified; do not repeat it as fact.

### Distributed communication

- Verify collective participants, operation order, tensor shape/dtype, group selection, stream dependencies, and error/fallback handling are consistent on every rank.
- Check uneven token/expert counts, empty ranks, world-size one, and A2 versus A3/A5 implementation differences where relevant.
- A local compile or single-rank test cannot prove HCCL correctness, ordering, or overlap.

### Compatibility and maintainability

- Check chip guards and version checks match actual capability requirements rather than directory names or assumptions.
- Check new environment variables/configuration are registered, documented when user-facing, have safe defaults, and are covered by tests.
- Check paired implementations and wrappers for drift: Python signature, `torch.ops` schema, C++ declaration, host code, kernel parameters, and tests.

## Finding discipline

Before reporting a finding, verify it against the complete symbol path and record:

- location: file and line/symbol;
- trigger: concrete shape, dtype, chip, configuration, or distributed condition;
- impact: what fails, becomes inaccurate, regresses, or becomes unmaintainable;
- evidence level: `已验证` or `推测`;
- action: a specific correction, test, or reviewer question.

Classify CI failures before attributing them to the PR. A red check can be infrastructure, flaky, expired, or pre-existing; compare against the base branch when evidence is available.

Do not report speculative NPU API replacements, predicted speedup percentages, or imaginary tests. For uncertain vendor/API behavior, state what must be verified in official CANN/torch-npu documentation or on target hardware.

## Output

Return a concise report directly unless the user asks to save it. If saving without a path, use `./outputs/sglang-ascend-pr-PR_NUMBER-review.md`. Aim for 3–10 meaningful findings; omit the findings section when there are none.

```markdown
# Ascend NPU Code Review: <PR #number or local range> — <title>

> **仓库**: `OWNER/REPO` | **范围**: <PR / commit range>
> **变更规模**: +X -Y，N 个文件 | **Ascend 相关性**: 完全相关 / 部分相关 / 不相关
> **审查快照**: <HEAD or PR updatedAt> | **数据截至**: <timestamp>

## 1. 变更与 NPU 执行路径

<动机、关键模块、NPU 触发条件，以及本次实际追踪的调用链。>

## 2. Review 意见

| 类型 | 🔴 | ⚠️ | 📝 |
|---|---:|---:|---:|
| 正确性 / 精度 / 性能 / 通信 / 兼容性 / 测试 / 可维护性 | N | N | N |

### 🔴 <类型>: <标题> `[已验证]`

- **位置**: `path:line` / `<symbol>`
- **触发条件**: <shape, dtype, chip, config, ranks, or request path>
- **问题与影响**: <evidence-based explanation>
- **建议**: 作者应当 <specific correction or test>。

### ⚠️ <类型>: <标题> `[推测]`

- **位置**: ...
- **触发条件**: ...
- **问题与影响**: ...
- **建议**: 建议作者 ...

## 3. 已有讨论与 CI

<实质 review 讨论、设计决定、已解决/未解决线程，以及 CI 证据和局限。>

## 4. 未覆盖验证

- <精度、性能、A2/A3/A5、CANN 版本、多卡 HCCL 或集成测试缺口>

## 5. 结论

`LGTM` / `NEEDS WORK` / `BLOCK` — <1–2 sentence rationale>。
```

Use `BLOCK` only when at least one high-severity finding has a concrete trigger and credible runtime impact. Use `NEEDS WORK` for actionable non-blocking findings or material missing validation. Use `LGTM` only for the reviewed scope and state the unverified NPU hardware coverage.

## Related skills

- Use `sglang-pr-summary` to explain a PR comprehensively without generating review findings.
- Use `sglang-pr-describer` to draft a PR description.
- Use `sglang-ascend-perf-analysis` for profiler-led root-cause work after a performance concern is identified.
- Use `ascendc-operator-code-review` or `triton-operator-code-review` for a deeper review of a specific device operator.
