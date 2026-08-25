# sgl-auto-drive-npu

A collection of Claude Code / Codex skills for **AI-agent-driven SGLang development on Huawei Ascend NPU**.

These skills encode the institutional knowledge needed to develop, debug, and optimize the [SGLang](https://github.com/sgl-project/sglang) LLM serving engine and its Ascend NPU kernel library (`sgl-kernel-npu`) — so an AI coding agent can act as a competent NPU engineer without you having to re-explain the environment every session.

---

## Skills

The repository currently contains 54 top-level skills and 75 `SKILL.md` playbooks, including nested ATB, OpenMMLab, and SSH workflow skills. The SGLang and Ascend skill set is synchronized from the local Codex skill library.

| Area | Skills | Purpose |
|------|--------|---------|
| SGLang | [`sglang-skill`](skills/sglang-skill/SKILL.md), [`sglang-debug-ascend`](skills/sglang-debug-ascend/SKILL.md), [`sglang-accuracy-debugging`](skills/sglang-accuracy-debugging/SKILL.md), [`sglang-npu-adapter`](skills/sglang-npu-adapter/SKILL.md), [`sglang-ascend-to-main-port`](skills/sglang-ascend-to-main-port/SKILL.md), [`sglang-perfermance-ascend`](skills/sglang-perfermance-ascend/SKILL.md), [`sglang-pr-describer`](skills/sglang-pr-describer/SKILL.md) | Development, remote validation, accuracy and performance debugging, model adaptation, upstream porting, and PR preparation. |
| Ascend platform | [`ascend-docker`](skills/ascend-docker/SKILL.md), [`ascend-a5-optimization`](skills/ascend-a5-optimization/SKILL.md), [`ascend-profiling-anomaly`](skills/ascend-profiling-anomaly/SKILL.md), [`npu-smi`](skills/npu-smi/SKILL.md), [`hccl-test`](skills/hccl-test/SKILL.md) | Containers, A5 optimization, profiling analysis, NPU health checks, and collective-communication testing. |
| AscendC / CANN | [`ascendc-operator-dev`](skills/ascendc-operator-dev/SKILL.md), [`ascendc-operator-code-review`](skills/ascendc-operator-code-review/SKILL.md), [`cann-a5-operator-compile`](skills/cann-a5-operator-compile/SKILL.md), [`cann-operator-env-config`](skills/cann-operator-env-config/SKILL.md) | End-to-end operator development, review, compilation, precision/performance work, and CANN environment setup. |
| ATB / ACLNN | [`ascend-transformer-boost`](skills/ascend-transformer-boost/SKILL.md), [`cann-nnal-installer`](skills/cann-nnal-installer/SKILL.md) | ATB operator workflows, ACLNN migration, CSV testing, debugging, and NNAL installation. |
| Triton / CATLASS | [`add-triton-npu-kernel`](skills/add-triton-npu-kernel/SKILL.md), [`triton-operator-dev`](skills/triton-operator-dev/SKILL.md), [`catlass-operator-dev`](skills/catlass-operator-dev/SKILL.md), [`vector-triton-ascend-ops-optimizer`](skills/vector-triton-ascend-ops-optimizer/SKILL.md) | NPU kernels and operators: design, generation, testing, correctness evaluation, and optimization. |
| Model ecosystem | [`drivingsdk-ascend-model-migration`](skills/drivingsdk-ascend-model-migration/SKILL.md), [`npu-adapter-reviewer`](skills/npu-adapter-reviewer/SKILL.md), [`vllm-ascend-deploy`](skills/vllm-ascend-deploy/SKILL.md), [`import-vllm-ascend-operator`](skills/import-vllm-ascend-operator/SKILL.md) | Model migration/training, OpenMMLab installation, NPU adaptation review, vLLM-Ascend deployment, and operator import. |

---

## Installation

### Windows (PowerShell)

```powershell
.\install_skill.ps1 https://github.com/AndyLi429/sgl-auto-drive-npu
```

This fetches every skill under `skills/` from the GitHub API and writes each `SKILL.md` to both `~/.claude/skills/<name>/` and `~/.codex/skills/<name>/`.

### Linux / macOS — install manually

Clone the repo and copy the skill directories you need:

```bash
git clone https://github.com/AndyLi429/sgl-auto-drive-npu.git
cp -r sgl-auto-drive-npu/skills/sglang-skill ~/.claude/skills/
# repeat for other skills
```

### Sync Claude → Codex skills (Linux/macOS)

If you use both Claude Code and Codex, keep their skill sets in sync:

```bash
bash sync_skills_claude_to_codex.sh
```

---

## Requirements

- [Claude Code](https://claude.ai/code) CLI (skills are invoked automatically when relevant)
- SSH access to an Ascend 910C node with Docker (for `sglang-debug-ascend`)
- CANN toolkit and `torch_npu` installed inside the container (for NPU validation)

---

## How skills work

Each skill is a `SKILL.md` file placed in `~/.claude/skills/<name>/`. Claude Code loads skills at session start and invokes the right one based on what you ask. No manual activation needed — just describe what you want to do and the agent picks up the correct playbook.

> [!NOTE]
> Skills are local to your machine. Nothing in this repo runs automatically — the agent reads and follows the skill instructions only when you trigger it in a conversation.

---

## Related projects

- [sgl-project/sglang](https://github.com/sgl-project/sglang) — upstream SGLang serving engine
- [Ascend/sgl-kernel-npu](https://github.com/Ascend/sgl-kernel-npu) — Ascend NPU kernel library for SGLang
