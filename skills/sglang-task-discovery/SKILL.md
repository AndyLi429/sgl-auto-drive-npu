---
name: sglang-task-discovery
description: Use when a user asks what to contribute to SGLang or its Ascend NPU ecosystem, requests module-specific demand analysis, or wants current contribution tasks ranked by difficulty, prerequisites, and urgency.
---

# SGLang 开发需求发现

生成可追溯、可执行的贡献机会报告。核心原则：搜索结果只是线索；Task 必须有一手证据，并在当前默认分支确认仍未解决。

## 默认范围

从用户输入提取目标模块、仓库、时间窗口、Top N 和输出语言。未指定时采用：

| 项目 | 默认值 |
|---|---|
| 仓库 | `sgl-project/sglang`、`sgl-project/sgl-kernel-npu` |
| 时间 | 近期活动 90 天；最多回看 180 天补充上下文 |
| 数量 | Top 10；弱证据不足时少于 10，不凑数 |
| 领域 | runtime、Ascend backend、Ascend C/Triton、DeepEP、模型、测试、文档 |

用户指定模块时，先从仓库结构、标签和近期 PR 确定同义词与代码路径。只有歧义会显著改变结论时才询问；否则说明采用的范围并继续。

## 建立快照

优先使用用户已有 checkout，记录 remote、branch、HEAD、默认分支和采集时间。若请求包含“最近/当前”，必须核对官方远端，不把本地旧分支当成最新状态。没有 checkout 时，可通过 GitHub CLI/API 获取元数据，并在临时目录浅克隆默认分支用于代码搜索。

先运行 `gh auth status`。认证不可用时退回公开 API 或网页，并记录分页、速率限制以及不可访问的 Projects/Discussions。单一渠道缺失不等于“没有需求”。

## 收集证据

围绕目标关键词和代码路径收集以下来源；对进入候选池的条目读取正文、评论和关联项，不能只看搜索摘要。

1. **近期 merged PR**：从正文、review、评论和改动文件寻找明确 follow-up、已知限制、未覆盖 NPU 平台或拆出的后续工作。用 `merged_at` 判断时间窗口。
2. **open PR**：搜索同义任务和目标路径，识别正在实现或已认领的范围。
3. **Issues 与 Discussions**：检查未解决 bug、feature/model request、help wanted、assignee、最近评论和维护者表态。
4. **Roadmap 与发布信号**：检查仓库内 Roadmap、milestones、Projects、release notes 和实际存在的 priority/roadmap labels。只有明确来源才能声称 deadline、版本或季度归属。
5. **TODO/FIXME/HACK/XXX**：用 `rg` 搜索目标路径，阅读实现、调用者与测试，并用 `git blame` / commit 历史理解注释来源和年龄。注释可能只是说明，不能自动转成 Task。
6. **测试、CI、文档和兼容矩阵**：核对 skip/xfail、NPU 缺失覆盖、实验性能力、已知限制及芯片/CANN/torch-npu 兼容缺口。

命令按仓库、日期和关键词调整：

```bash
gh pr list --repo OWNER/REPO --state merged --search "merged:>=YYYY-MM-DD KEYWORD" --limit 100 \
  --json number,title,url,mergedAt,updatedAt,labels,author,body,files
gh pr list --repo OWNER/REPO --state open --search "KEYWORD" --limit 100 \
  --json number,title,url,updatedAt,labels,assignees,author,body,files
gh issue list --repo OWNER/REPO --state open --search "updated:>=YYYY-MM-DD KEYWORD" --limit 100 \
  --json number,title,url,updatedAt,labels,assignees,author,body
gh repo view OWNER/REPO --json url,defaultBranchRef --jq '{url: .url, branch: .defaultBranchRef.name}'
gh api repos/OWNER/REPO/commits/DEFAULT_BRANCH --jq .sha
gh api --paginate "repos/OWNER/REPO/milestones?state=all"
rg -n -i "TODO|FIXME|HACK|XXX" <relevant-paths>
git blame -L <start>,<end> -- <path>
git log --since=YYYY-MM-DD -- <path>
```

这些是 Bash 形式的起点，不是穷举脚本；替换占位符并适配当前 shell。已有 checkout 时先确认 remote 指向官方仓库，再用 `git fetch --depth=1 <official-remote> <default-branch>` 获取 `FETCH_HEAD`，无需切换或修改用户分支；没有 checkout 时浅克隆到临时目录。报告中的默认分支 SHA 必须与 GitHub API 返回值一致。GitHub 搜索存在结果上限，宽泛查询应按日期、标签或模块拆分并记录覆盖范围。Discussions、Projects v2 和 Releases 按仓库实际启用情况通过 `gh api`、GraphQL 或官方网页读取。优先引用 GitHub 原始页面、仓库文件和 commit permalink。博客或二手文章只能辅助发现，不能单独证明任务仍有效。

## 形成 Task

把线索改写为“在哪个模块，为哪类用户补齐什么能力，当前缺口是什么，完成后如何验证”。每个 Task 必须：

- 有明确价值、交付边界和可观察验收标准，而非“优化性能”之类泛化描述。
- 至少一条一手证据；代码推断还需文件、行号或符号以及 HEAD。
- 标明依据：`明确需求` 或 `分析推断`；推断不得冒充维护者承诺。
- 标明状态：`可认领`、`已有 PR`、`已有负责人`、`需维护者确认`。
- 检查后续 PR、当前代码、关闭讨论和关联仓库是否已解决该缺口。

按 `仓库 + 子系统 + 缺失能力 + 平台/模型 + 预期产物` 聚类。合并同一根因的 Issue、PR follow-up 与 TODO；大型 Roadmap 项拆成可独立验收的子任务。Top N 只包含可独立认领的任务。已有活跃 PR 的事项移入单独的“协作机会”，不要称为无人认领；只有用户明确要求寻找协作项时才把它们并入主榜。

## 评估

### 难度

- **D1 Beginner**：文档、配置或局部测试；接口边界清晰。
- **D2 Intermediate**：跨多个文件，或需理解一条 runtime/backend 调用链。
- **D3 Advanced**：涉及调度、分布式、attention/MoE、图捕获或 NPU kernel，并需设备验证。
- **D4 Expert**：跨 SGLang 与 kernel/CANN/通信栈，涉及并发、数值精度或系统性能权衡。

必须解释改动面、调用链和验证环境。前置知识写具体的 SGLang 组件、Ascend 技术、工具链及设备条件，不列“会 Python/Git”之类通用项。

### 紧急程度

- **U3 High**：明确 deadline/release blocker、阻塞性回归、安全或数据正确性问题，或维护者明确最高优先级。
- **U2 Medium**：活跃且有明确用户影响，或已确认的近期 follow-up，但无硬 deadline。
- **U1 Low**：有效的长期改进，无近期承诺或阻塞影响。
- **U0 Unknown**：证据不足，需维护者确认。

热度、评论数、新旧程度不能单独证明紧急。另标 `High / Medium / Low` 证据置信度。难度用于匹配开发者，不作为优先级加分项；不要用加权总分制造虚假精确度。

## 排序与新鲜度审计

先排除已解决、重复、被否决、无法验收或实际上不可贡献的条目。排序依次比较：紧急度（U3→U0）、明确需求优先于同等级推断、证据置信度（High→Low）、影响范围、证据新鲜度。仍并列时优先验收边界更清楚的任务。报告为每项写一句“排名理由”，不用加权总分。

对最终 Top N 逐项重新检查默认分支：

1. 目标符号、TODO、测试缺口仍存在。
2. 没有同语义的 merged/open PR 或已完成 Issue。
3. 原始证据之后的相关提交没有间接解决问题。
4. assignee、Roadmap、milestone 和 release 状态未变化。
5. 记录 `last_verified_at` 与验证所基于的 commit SHA。

状态无法确认时标为“需维护者确认”，并降低置信度；不要强行纳入 Top N。

## 输出

读取 [references/report-template.md](references/report-template.md)，按其结构生成报告。默认直接返回，不写文件。用户明确要求落盘时写入其指定位置；若其未指定路径，才使用当前工作目录下的 `sglang_task_discovery_<topic>_<YYYYMMDD>.md`。

交付前确认：每个 Task 的链接可达、状态经过新鲜度审计、难度/前置知识/紧急度都有理由，显著的排除项也记录了原因。无法确认的字段写“未知/待确认”。

## 常见误判

| 误判 | 正确处理 |
|---|---|
| TODO 仍在，所以功能未实现 | 检查当前行为；重构后注释可能已失效 |
| Roadmap 提到方向，所以是紧急任务 | 只有 deadline/priority 的一手证据才能提高紧急度 |
| merged PR 留有 follow-up，所以仍待做 | 搜索该 PR 合入后的提交与关联 PR |
| GPU 已支持，所以 NPU 一定需要照搬 | 先证明真实用户价值、平台可行性与维护者接受度 |
