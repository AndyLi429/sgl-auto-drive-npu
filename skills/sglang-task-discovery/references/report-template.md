# SGLang 需求发现报告模板

按目标任务调整分类和篇幅，不为空分类保留章节。事实尽量在相邻位置附一手来源。

```markdown
# SGLang <目标领域>开发需求报告

## 调研范围

| 项目 | 内容 |
|---|---|
| 数据截止时间 | YYYY-MM-DD HH:mm TZ |
| 仓库快照 | `OWNER/REPO@SHA`（branch）；... |
| 活动窗口 | YYYY-MM-DD 至 YYYY-MM-DD |
| 目标范围 | 模块、特性、模型或路径 |
| 查询与来源 | merged/open PR、issue、discussion、roadmap、TODO、CI、docs |
| 候选统计 | 原始 N；去重 N；排除 N；入选 N |
| 未覆盖项 | 无；或权限、API、分页、设备等限制 |

## 结论摘要

用 2–4 段说明主要方向、最值得认领的任务和关键不确定性。区分维护者明确需求与分析推断。

## 优先任务总览

| # | Task | 类型 | 状态 | 难度 | 紧急度 | 置信度 | 核心依据 |
|---|---|---|---|---|---|---|---|
| 1 | <可执行标题> | Bug / Feature / Perf / Model / Test / Docs / Refactor | 可认领 | D2 | U3 | High | [Issue #N](URL) |

## Task 详情

### T01. <可执行标题>

- **任务主张**：<目标用户、当前缺口、交付结果>
- **目标位置**：`OWNER/REPO`；模块、路径或关键符号
- **结论类型**：明确需求 / 分析推断
- **当前状态**：可认领 / 已有 PR / 已有负责人 / 需维护者确认
- **难度**：D1–D4 — <改动面、调用链和验证环境依据>
- **前置知识**：<具体 SGLang 子系统>；<Ascend C/Triton/CANN/torch-npu/HCCL 等>；<测试或 profiling 工具>
- **紧急程度**：U0–U3 — <deadline、priority、回归影响，或为何未知>
- **证据置信度**：High / Medium / Low — <证据强弱>
- **排名理由**：<相对其他候选更靠前或靠后的直接理由>
- **依赖与风险**：<上游版本、硬件、精度、性能、多仓协作>
- **建议入口**：`path/to/file.py:line`、符号或现有测试

**需求与证据**

说明需求来源、当前实现缺口、近期 PR/TODO/Roadmap 的关系。分析推断要写出推理链。

**完成边界**

- <可观察的交付结果>
- <关键测试、模型、芯片及精度/性能验证>
- <明确不属于本 Task 的内容>

**来源与新鲜度**

- [Issue/PR/Discussion #N](URL) — 状态；读取日期
- [`path/to/file.py` @ `SHA`](permalink) — TODO 或实现证据
- TODO 来源（适用时）：引入 commit、`todo_introduced_at`、注释年龄、最近相关修改
- `last_verified_at`: YYYY-MM-DD HH:mm TZ；默认分支 `SHA`

## 协作机会

列出已有活跃 PR 的相关事项、负责人和可协作范围。默认不计入 Top N。

## 候选项审计

| 候选项 | 处理 | 原因与证据 |
|---|---|---|
| <标题> | 排除 / 合并到 Txx / 降级 | 已解决 / 已有活跃实现 / 证据不足 / 重复根因 |

## 数据来源统计

| 来源 | 检查数量 | 入选数量 | 查询范围或限制 |
|---|---:|---:|---|
| 近期 merged PR | N | N | 日期与关键词 |
| open PR / Issues / Discussions | N | N | 状态与标签 |
| TODO / FIXME | N | N | 路径与 HEAD |
| Roadmap / Milestones / Releases | N | N | 可访问范围 |

## 建议下一步

给出 1–3 个动作，如在 Issue 确认范围、复现问题、阅读指定调用链或准备最小 benchmark。不要替用户认领或联系维护者。
```
