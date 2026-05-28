# sgl-auto-drive-npu

A collection of Claude Code / Codex skills for **AI-agent-driven SGLang development on Huawei Ascend NPU**.

These skills encode the institutional knowledge needed to develop, debug, and optimize the [SGLang](https://github.com/sgl-project/sglang) LLM serving engine and its Ascend NPU kernel library (`sgl-kernel-npu`) — so an AI coding agent can act as a competent NPU engineer without you having to re-explain the environment every session.

---

## Skills

| Skill | Purpose |
|-------|---------|
| [`sglang-skill`](skills/sglang-skill/SKILL.md) | Core SGLang development on both CUDA and Ascend NPU — architecture map, kernel layout, debugging patterns, and launch commands |
| [`sglang-debug-ascend`](skills/sglang-debug-ascend/SKILL.md) | Remote workflow for running and validating code on an Ascend 910C node via SSH + Docker |
| [`sglang-perfermance-ascend`](skills/sglang-perfermance-ascend/SKILL.md) | Performance analysis using Ascend PyTorch Profiler — collection, bottleneck identification, and interpretation of `kernel_details.csv` / `trace_view.json` |
| [`add-triton-npu-kernel`](skills/add-triton-npu-kernel/SKILL.md) | Step-by-step guide for writing JIT Triton kernels that compile via Triton-Ascend and run on Ascend NPU vector cores |
| [`sglang-pr-describer`](skills/sglang-pr-describer/SKILL.md) | Draft PR descriptions conforming to `sgl-project/sglang` community conventions |

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
