---
name: sglang-model-tutorial
description: Use when a developer wants a beginner-friendly Chinese tutorial for a specific model in SGLang, including its architecture, model implementation, registration, loading, and execution path.
---

# SGLang 模型源码导读

让读者同时理解“模型本身如何从 token 产生 logits”和“这份模型定义如何被 SGLang 选择、加载、批处理与调用”。模型论文、Hugging Face 配置和 SGLang 源码分别说明不同层面的事实，不能互相替代。

## 确定模型与快照

提取模型名、具体变体/checkpoint、目标仓库、读者基础、NPU 范围和输出位置。先区分：基础 LLM、VLM、MoE、量化 checkpoint 与同名模型家族；只有这种歧义会改变模型定义或执行路径时才追问。

优先使用用户的 checkout，记录 remote、branch、HEAD 和采集时间。用户要求“最新”时，核对 SGLang 官方默认分支。存在 `.codegraph/` 时先用 CodeGraph 定位模型注册、模型类、`forward`、模型 runner 和调用者；否则用 `rg` 并阅读实现、调用者、配置和测试。

为每个结论标明证据层：

| 结论 | 首选证据 |
|---|---|
| 架构、维度、attention/MoE/VLM 设计 | 官方技术报告、model card、模型 `config` |
| SGLang 模型选择、类层级、权重加载 | 当前 SGLang 源码、配置和测试 |
| 请求到 `forward` 的参数与时序 | runner/worker 调用点、`forward` 签名、返回值消费者 |

模型卡或论文不能证明 SGLang 如何实现模型；源码也不能替代模型版本的架构规格。无法核验时写“待确认”，不要以同名变体、Hugging Face 类或其他推理框架的习惯补全。

## 建立两条阅读线

教程必须并行追踪下列两条线，并在交点解释数据契约：

```text
模型线：config / 权重 -> embedding -> decoder blocks -> logits
运行时线：架构标识 -> registry / factory -> loader -> model runner -> forward -> sampler
```

模型线说明张量、层和权重如何组织；运行时线说明谁构造模型、谁提供 positions/batch metadata/KV 引用、谁消费输出。不要默认 `forward` 接收 `input_ids` 或直接返回 logits；以真实签名和调用点为准。

对关键站点记录：位置与符号、职责、输入/输出、状态或不变量、下一跳和证据类型。需要解释 prefill/decode 差异时，先确认它们是否实际分支或使用不同 metadata。只有调用链确实跨越 NPU backend、`sgl-kernel-npu`、DeepEP 或 CANN/torch-npu 时，才纳入 NPU 边界与优化说明。

## 讲解模型结构

先用最小 Transformer 术语表解释 token、embedding、hidden state、attention、MLP/MoE、layer、logits、prefill、decode 与 KV cache。再按当前模型实际具备的组件展开：

- Dense decoder：embedding、decoder block、attention、MLP、norm、LM head。
- MoE：仅在源码/配置证实时解释 router、expert、dispatch/combine 和并行边界。
- VLM：仅在模型与 SGLang 都支持该变体时解释视觉编码、投影、多模态输入处理与融合。
- 特殊 attention、量化或长上下文：以当前配置和实现为准，说明适用变体与边界。

不要为“完整”而添加不适用的 ViT、MoE、MLA、GQA 或性能对比章节。

## 图表与输出

读取 [references/model-tutorial-template.md](references/model-tutorial-template.md)。默认使用两张 Mermaid 图：模型结构数据流图、SGLang 接入/请求时序图。仅在能解释真实分支时增加 prefill/decode 对照、MoE 路由或 VLM 输入图。每张图只回答一个问题，控制在约 8–12 个节点。

使用 **`mermaid-syntax`**：使用 `flowchart`，含特殊字符的标签使用双引号，采用安全英文 ID，避免 `end`、`default` 等保留字；时序图中的字面分号写作 `#59;`。若本地具备渲染器则渲染验证；否则完成静态检查并说明未渲染。

默认直接在回复中输出 Markdown。用户明确要求落盘时写入指定位置；未给路径时使用当前工作目录下的 `sglang_<model>_tutorial.md`。

## 交付检查

最终文档不含 `<...>` 或“已核验”等占位文字。确认：

| 检查项 | 要求 |
|---|---|
| 初学者路径 | 先讲模型问题与术语，再讲实现 |
| 双重映射 | 模型组件与 SGLang 运行时职责均可定位 |
| 真实执行链 | 至少一条从模型选择/加载到 `forward` 结果消费的调用链 |
| 变体边界 | 明确本教程适用的 model/config/checkpoint，未混用同名变体 |
| 源码阅读 | 每站给出路径、符号、输入/输出/不变量和下一跳 |
| 可观察性 | 测试、配置、日志或最小请求中的至少一种证据 |
| 图表与不确定性 | 图表合规；推断、未运行命令和版本差异明确标记 |

## 常见误读

| 误读 | 正确处理 |
|---|---|
| 模型论文等于 SGLang 实现 | 分别引用模型规格与 SGLang 代码 |
| 同名 checkpoint 使用同一执行路径 | 核对 config、量化、多模态和后端条件 |
| `forward(input_ids)` 就是运行时接口 | 从真实调用点和签名确认参数与返回值 |
| KV cache 属于模型类 | 核对缓存所有权、复用时机和 runner/调度边界 |
| 模型文件里有 NPU 算子 | 只有跨越真实 NPU 调用边界时才可作此结论 |
