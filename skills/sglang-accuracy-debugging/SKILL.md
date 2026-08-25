---
name: sglang-accuracy-debugging
description: 定位和排查 SGLang 推理框架的精度问题（accuracy issues / 精度问题）。当用户遇到 SGLang 模型输出胡言乱语、语句不通顺、数据集压测分数下降、首 token 错误、多 batch 精度问题、DP/TP/PD 分离/图模式/DeepEP/MTP 等特性导致的精度异常、NPU vs GPU 精度对比、单算子精度异常、或需要在 SGLang 框架中加打印对比中间结果时，应使用本 skill。触发关键词包括：精度问题、accuracy issue、模型输出错误、胡言乱语、首 token 错误、数据集分数下降、多 batch 精度、SGLang 推理结果不对、logits 对不齐、attention 输出异常、对比 baseline 等。即使用户没有明确说 "SGLang"，只要上下文涉及 SGLang 的模型推理精度排查，也应使用此 skill。
---
 
# SGLang 精度问题定位指南
 
按照四步法定位：**确认问题 → 明确场景 → 找基线对比 → 加打印缩小范围 → 修复**。每一步都有明确产出，不要跳步。
 
## 第 0 步：确认是否真的是精度问题
 
先判断问题存在性，避免误判：
 
1. **手动 curl**：多发几次相同/不同请求，肉眼看输出是否通顺。相同请求多次推理有细微差异是正常的。
2. **数据集压测**：跑分数对比论文/官网数据。
> 如果只是一两次输出有细微差别，很可能不是 bug，是采样随机性。
 
## 第 1 步：明确错误场景
 
定位前必须回答三个问题，缩小怀疑范围。
 
### 1.1 模型配置：是哪个特性引入的？
 
默认做法：**先把所有性能特性关掉**，看基础模型是否有精度问题。然后逐个叠加，找出肇事特性。
 
| 特性 | 默认状态 | 关闭方式 |
|---|---|---|
| CUDA Graph (图模式) | 开 | `--disable-cuda-graph` |
| Prefix Cache | 开 | `--disable-radix-cache` |
| Chunked Prefill | 开 | `--chunked-prefill-size 32768000`（设极大值） |
| Scheduler Overlap | 开 | `--disable-overlap-schedule` |
| DeepEP | 关 | — |
| DP / DP Attention | 关 | — |
| MTP | 关 | — |
| PD 分离 | — | 对比混步（PD 混合部署）结果 |
 
**判断逻辑**：
- 基础模型（全关）无问题 → 逐步叠加，找引入问题的特性
- 基础模型也有问题 → 大概率是模型某个模块本身有问题，进入第 2 步
另外测试不同 TP size 下是否有差异。
 
### 1.2 测试数据：什么输入会出错？
 
保存好出错的 prompt。区分 **单 batch 问题** vs **多 batch 问题**：
 
- curl 一条就胡言乱语 → 严重问题，任选 prompt 调试即可
- curl 正确但数据集压测错 → 可能是多 batch 精度问题
  - 把数据集并行度改成 1 重测
  - 并行度 1 正确 → **确认是多 batch 问题**，用 for 循环 curl 多条触发
  - 并行度 1 还是错 → 从数据集里挑出错的 prompt 来调试
### 1.3 错误表现：缩小到 prefill 还是 decode？
 
- **首 token 错** → prefill 阶段问题
- **首 token 对，后面错** → 大概率 decode 阶段问题（也可能是 prefill 的 KV cache 算错/存错导致）
- **多次 curl 结果完全随机** → 怀疑地址越界、未定义行为
- **多次 curl 结果种类 ≤ DP size** → 可能某一路 DP 出问题
> 开 DP 时 curl 次数必须 > DP size，否则测不全所有 DP 分支。
 
## 第 2 步：找精度基线对比
 
没有基线就没法定位。按场景选基线：
 
| 场景 | 推荐基线 |
|---|---|
| 某特性引入的问题 | 关闭该特性的配置 |
| 多 batch 精度问题 | 单 batch 结果。重点查 attention padding、prepare attn/mlp、postprocess layer 的通信（DP Attention、DeepEP 等） |
| 同一 prompt 多次结果差异大 | 两次中间结果对比，找第一次出分歧的位置 |
| NPU 上没有正常基线 | GPU 结果；如果 GPU 也错且一致，问题在框架层；可换 transformers 或裸模型验证 |
 
## 第 3 步：加打印，缩小范围
 
### 3.1 SGLang 框架侧流程（知道在哪儿下手）
 
入口 → Scheduler 主循环 → tp_worker → model_runner → 模型 forward → 返回 logits → sample：
 
- `scheduler.event_loop_normal`（无 overlap）或 `event_loop_overlap`（有 overlap）是入口，`while True` 循环组 batch、跑 `run_batch`
- `forward_batch_generation` → `tp_worker` → `model_runner.forward` → 模型 forward
- 模型返回 logits，`tp_worker` 和 `model_runner` 调 `sample` 出 token
### 3.2 找到当前模型的入口类
 
1. 打开模型权重目录下的 `config.json`
2. 找 `architectures` 字段，值就是入口类名（例：`DeepseekV3ForCausalLM`）
3. 在 `python/sglang/srt/models/` 下搜这个类名
4. 看入口类有没有 `forward`；没有就找父类的（例：`DeepseekV3ForCausalLM` 继承自 `DeepseekV2ForCausalLM`，forward 在父类里）
### 3.3 torch 模型运行流程（看懂结构）
 
关键点：`self.model(...)` 这种调用能跑，是因为 `nn.Module` 重写了 `__call__`，最终调用 `forward`。追踪方法：
 
1. 从入口 forward 的 `self.xxx(...)` 看起
2. 在 `__init__` 里找 `self.xxx` 是什么类
3. 进入那个类找它的 `forward`
4. 重复直到看到底层算子
大多数 LLM 的结构是：`embedding → num_hidden_layers × DecoderLayer → norm → lm_head`。DecoderLayer 通常是 `Attention + MoE/MLP`，前后有 `LayerCommunicator` 做通信和 normalization。
 
**打印位置建议**：在 DecoderLayer 内 Attention/MLP 前后打印 `hidden_states`，找到第一处差异层后再深入该模块继续加打印，逐步缩小。
 
### 3.4 打印方法
 
- 卡数少：直接 `print`
- 卡数多：用 `logger`，能带上卡号信息，更清晰
```python
import logging
logger = logging.getLogger(__name__)
logger.info(f"[rank {torch.distributed.get_rank()}] hidden_states: {hidden_states}")
```
 
## 第 4 步：修复或规避
 
### 4.1 Native 替换
 
定位到可疑代码块但看不出 bug 时，用 **native torch 实现** 替换这段（或参考基线实现/手搓一个）。
 
- 替换后精度恢复 → 定位成功，可以尝试真正的修复
- 还有精度差异 → 继续对比找下一处问题点
### 4.2 单算子测试
 
定位到具体算子后，从整网 dump 输入输出，单独测这个算子，比在整网里调试快得多。
 
**Dump 模板**：
 
```python
# 在调用算子前
if torch.distributed.get_rank() == 0:
    torch.save(q_nope, "/your_dir/q_nope.pt")
    torch.save(k_nope, "/your_dir/k_nope.pt")
    # ... 保存所有输入
    # 算子执行后
    torch.save(attn_output, "/your_dir/output.pt")
    assert False  # 直接退出，或用全局标志位避免重复 save
```
 
**单测判断标准**：
- `torch.allclose(a, b, rtol=1e-2, atol=1e-2)` 通过 → 一般不影响整体精度
- 差异大 → **先检查 shape 和 dtype 是否符合算子要求**（绝大多数算子精度错/报错都是这个原因）
- 数据类型正确仍错 → 对比算子逻辑，找算子接口人
## 其他常见坑
 
- **结果全 0 / 每个 token 都一样**：检查模型输入是否被意外覆盖，检查权重是否被错误加载为全 0
- **模型文件损坏**：比对 `config.json` 是否被改过，用 `md5sum` / `shasum` 对比官方源
## 给新手的一句话总结
 
遇到精度问题别慌，按这个顺序走：**先确认问题真存在 → 关特性找出肇事者 → 找对比基线 → 从 DecoderLayer 前后加打印二分查找 → 定位到算子就 dump 出来单独测**。大部分问题都卡在第 2 步（没有好的基线）和第 3 步（不知道在哪儿加打印），这份指南重点解决这两个。
