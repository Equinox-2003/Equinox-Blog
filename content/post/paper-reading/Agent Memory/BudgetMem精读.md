---
title: "论文精读 | BudgetMem"
description: "让 Agent 学会按问题决定'记忆处理要花多少钱'"
date: 2026-07-22T16:25:22+08:00
lastmod: 2026-07-22T16:25:22+08:00
draft: false

categories:
  - paper-reading
tags:
  - LLM
  - Agent Memory
  - RL

toc: true
math: true
mermaid: true
cover: https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1784821204803_image.png
---

<!--more-->



## 零、写在前面

降低 memory cost 也是一个思路吧，这个工作就是想要降低 token/dollar 的花费，因为不是所有问题都要用深度推理、大model去做检索的。



## 一、标题

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1784821204803_image.png)

>   标题通俗的讲就是：让 Agent 学会按问题决定“记忆处理要花多少钱”

- **Learning（学习）**：预算不是hardcode，而是训练一个轻量 router（路由器）来学。
- **Query-Aware（查询感知）**：不同问题需要的记忆处理深度不同。简单事实不值得调用很多大模型；复杂的多跳问题可能值得花更多钱。
- **Budget-Tier Routing（预算分档路由）**：每个记忆模块有 `LOW / MID / HIGH` 三档计算预算，router 决定当前问题在每个模块使用哪一档。
- **Runtime Agent Memory（运行时 Agent 记忆）**：历史信息不是提前全部总结成固定记忆，而是在用户提问到来后，针对这个问题临时提取一份记忆。



**这个工作研究的是什么“成本”？**

论文主要优化的是：

> **runtime memory extraction cost：为了从历史中提取本次回答所需记忆，要调用多少模型、消耗多少 input/output token、支付多少钱。**

没有关注：

- 数据库占多少 GB；
- 向量索引占多少磁盘；
- 最终回答模型生成答案需要多少 token；
- 一个长期 memory store 如何在线写入、遗忘和跨会话维护。

可以把一次 Agent 回答的成本拆成：

```text
历史存储成本
    + 查询时记忆提取成本  <- BudgetMem 的重点
    + 最终答案生成成本
    + 检索、embedding、服务与网络成本
```



按照 Memory in the Age of AI Agents: A Survey 的分类范式，BudgetMem 主要是一个**动态管理框架**，而不是一种全新的记忆存储形态：

- **Form（形态）**：以 token-level external memory 为主。历史被切成文本 chunks，运行时取出文本，再形成一份 query-specific memory。
- **Function（功能）**：可以支持 factual、episodic 和 semantic memory，但论文的实验重点是对话历史和文档中的事实、人物、时间与主题关系。
- **Dynamics（动态）**：核心在 runtime formation / retrieval，即回答当前问题时动态形成记忆；它不是研究参数记忆，也没有重点研究长期记忆的删除和冲突演化。



## 二、摘要

论文上来先说 许多 Agent Memory 系统默认采用“先处理、以后一直用”的**离线模式**：历史对话先被总结、压缩、建图或索引，用户以后提问时直接检索。这种方式很省事，但它有两个问题：

1. 你不知道未来会问什么，提前总结可能把某个未来问题所需的细节丢掉；
2. 为所有历史、所有问题都用同样强的模型处理，会浪费计算。

BudgetMem 的思路是把大部分原始历史保留到查询时：

```text
用户问题 q
   -> 从原始历史取少量候选 chunks
   -> 按问题决定各记忆模块用 LOW / MID / HIGH
   -> 过滤、抽取实体/时间/主题、总结
   -> 得到本题专用 memory
   -> 回答问题
```

它训练一个小 router，使其学习：

> 对这个问题，过滤模块要不要用高档？时间模块是否值得深度推理？总结模块是否需要大模型？

然后也是点明了本文贡献：

1. **提出模块化 runtime memory pipeline**：把记忆提取拆成 filter、entity、temporal、topic、summary 等模块。
2. **统一定义三种 budget tier 实现方式**：改变模块实现、改变推理深度、改变模型容量。
3. **用强化学习训练共享 router**：根据 query、当前中间状态和模块身份，逐模块选择预算档位，并用答案质量与提取成本共同作为 reward。

论文在 LoCoMo、LongMemEval 和 HotpotQA 上报告：高预算时准确率超过多个记忆基线，预算收紧时仍能形成可控的性能—成本 Pareto 前沿。



## 三、引言

### 3.1 传统记忆系统的问题：build once, use always

设历史对话为 $H$。传统离线 memory pipeline 往往先做：

```text
H -> 摘要 / 抽取 / 建图 / 向量化 -> 固定 memory store
```

以后每个问题都使用这份提前构建好的 store。这种方法有一个隐含前提：**提前做的压缩对未来所有查询都足够好。**

但历史信息的价值依赖于问题。比如一段对话中同时包含：

- 用户的饮食偏好；
- 去年去过哪里；
- 某次争论的具体时间；
- 对未来项目的计划。

如果未来问“用户什么时候去过上海”，需要精确时间；如果问“用户喜欢什么菜”，需要偏好摘要。统一的离线总结很难同时保留二者的最佳粒度。



### 3.2 运行时提取的好处与代价

运行时记忆的做法是把历史先切成较小的原始 chunks，真正有问题到来时再处理：

```text
保留更多原始细节，减少不可逆压缩
                         ↓
                 但查询时要付计算费
```

因此核心问题变成一个典型的决策：

> **本题的准确率收益，是否值得多花这些 token 和 API 钱？**

已有系统常通过固定 top-k、固定模型、固定 CoT 或固定摘要长度控制成本，但这些旋钮通常是全局的，不能回答“哪个模块值得加预算”。



### 3.3 为什么需要模块级预算

一个完整的记忆提取过程可能同时包含：

- 从候选 chunks 中筛选相关内容；
- 找出人物和实体关系；
- 还原时间顺序；
- 判断主题与话题变化；
- 将证据融合为回答所需的摘要。

不同问题对这些模块的需求不同：

- “用户的猫叫什么？”可能只需轻量筛选和实体抽取；
- “用户在三次搬家前后职业变化是什么，哪次变化与某个事件有关？”可能需要时间、实体、主题和跨段总结都提高预算。

BudgetMem 的关键选择是：**不是把整条流水线统一调成 LOW 或 HIGH，而是让 router 对每个模块分别选择。**



### 3.4 why RL？

预算选择是离散的：某模块选 LOW、MID 或 HIGH。并且模块本身可能是规则、BERT 或 LLM，难以对“选了哪个档位”直接反向传播；最终效果要等整条流水线结束、答案生成后才能知道。

这很像一个小型 sequential decision problem（序列决策问题）：

```text
选择 filter 档位
    -> 看到过滤结果
选择 entity 档位
    -> 看到实体结果
选择 summary 档位
    -> 得到最终答案质量与总成本
```

因此论文使用 **PPO（Proximal Policy Optimization）**训练 router。



## 四、相关工作

### 4.1 Agent Memory

论文讨论了几类工作：

- **离线构建型**：MemoryBank、MemoryOS、LightMem 等提前压缩、组织或索引历史；
- **Agentic memory manager**：A-MEM、Mem0 等让 LLM 参与 add、update、delete、no-op 等记忆操作；
- **长上下文管理**：MemGPT / Letta 等管理信息进出上下文；
- **运行时深度利用**：在回答阶段对记忆做更深的规划或处理。

BudgetMem 的区别是，它把“查询时记忆提取的计算量”作为明确的优化目标，并尝试用同一套模块接口比较不同预算实现方式。



### 4.2 LLM 推理成本控制

相关的成本控制方向包括：

- **算法与系统层**：量化、剪枝、early exit、KV cache 优化、快速解码；
- **推理行为层**：直接回答、CoT、自我反思、多次搜索；
- **模型容量层**：小模型、中模型、大模型或不同后端路由。

BudgetMem 将这三类控制轴分别对应到记忆模块：

| BudgetMem 策略             | 改变什么           | 低 / 中 / 高示例                 |
| -------------------------- | ------------------ | -------------------------------- |
| **Implementation tiering** | 模块的实现方式     | 规则/regex -> BERT 类模型 -> LLM |
| **Reasoning tiering**      | 同一模型的推理行为 | 直接推理 -> CoT -> 多步/反思     |
| **Capacity tiering**       | 模块使用的模型大小 | 小模型 -> 中模型 -> 大模型       |

这三个轴的区别很重要：它们都叫“加预算”，但花钱的地方不一样。Implementation 可能改变算法，Reasoning 主要增加 token，Capacity 主要换更大的模型。



### 4.3 与普通自适应 RAG 的区别

自适应 RAG 往往决定“是否检索、检索多少文档、用哪个生成模型”。BudgetMem 的控制点更靠后：**它先用固定 retriever 取候选 chunks，然后控制候选内容如何被进一步加工成 memory。**

因此它研究的是：

```text
检索到候选之后，如何花预算加工候选内容
```

而不是完整解决：是否检索、用什么检索器、如何动态扩充候选池。



## 五、方法

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1784826177737_image.png)

### 5.1 输入：只做切块，不提前做语义记忆

历史 $H$ 先被切成文本块：

$$
C=\{c_1,c_2,\ldots,c_N\}
$$
这一步只是 segmentation（分段）和索引，不做摘要、实体抽取或重写。用户问题 $q$ 通过一个轻量 retriever 找到候选集合：

$$
C_q=R(q,C), \quad C_q\subseteq C
$$
实验默认用 Contriever，候选输入之后统一取 top-5 chunks。这样不同 memory 方法在相同候选上下文规模上比较。

这是一项有意的设计：BudgetMem 把“原始信息保留到运行时”，避免离线压缩提前丢掉未来问题需要的细节；代价是每次查询都可能重复处理历史。



### 5.2 固定的模块流水线

BudgetMem 在实验中使用：

$$
M_{fil}\rightarrow\{M_{ent},M_{tmp},M_{top}\}\rightarrow M_{sum}
$$


#### 5.2.1 Filter module

对已检索的 chunks 进一步过滤、重排或重加权，输出更相关的上下文。



#### 5.2.2 三个并行抽取模块

- **Entity module**：抽取人物、物体、关系和实体中心事实；
- **Temporal module**：抽取时间表达、先后顺序和时间锚点；
- **Topic module**：抽取主题、话题转移和较高层语义线索。

这三个模块并行运行，互相补充。比如问题是“用户为什么后来换工作”，实体模块找人物—工作关系，时间模块排序变化，主题模块帮助区分职业与个人话题。



#### 5.2.3 Summary module

将过滤结果、实体、时间和主题线索融合，输出当前 query 专用的 compact memory (m)：

$$
m=M_{sum}(q,e,t,p)
$$
最后答案模型根据 (q) 和 (m) 回答：

$$
\hat y=f_{ans}(q,m)
$$
需要注意：论文的核心实验是一个**对话/文档记忆提取 pipeline**，不是把每次生成的 (m) 自动永久写回长期 memory store。它主要研究“本题如何从原始历史中临时加工 memory”。



### 5.3 三种 LOW / MID / HIGH 的实现方式

每个模块都遵守同一个输入—输出接口，但内部可以有三个预算档位。

**A. Implementation tiering：换实现**

| 档位 | 典型实现                              | 直觉                             |
| ---- | ------------------------------------- | -------------------------------- |
| LOW  | 规则、regex、spaCy、关键词或稀疏匹配  | 快，但只能处理显式模式           |
| MID  | BERT 类轻量专用模型、embedding 相似度 | 能识别一定语义关系               |
| HIGH | LLM 处理                              | 能做隐式关系、多跳语义和冲突整合 |

例如 Filter 模块：LOW 用词面匹配，MID 用语义表示，HIGH 让 LLM 判断上下文相关性。Entity 模块则从 regex/浅层 NLP，升级到关系抽取模型，再升级到 LLM 关系补全和归纳。



**B. Reasoning tiering：换推理行为**

底层模型大体不变，只改变它思考的深度：

- LOW：直接输出；
- MID：使用 CoT-style 推理；
- HIGH：多步推理或 reflection-style 推理，做迭代检查和全局一致性判断。

例如 Temporal 模块在 LOW 只抓“昨天、去年、2020 年”等显式时间；HIGH 则尝试推断跨段落的隐含先后关系。



**C. Capacity tiering：换模型大小**

模块接口和推理行为保持一致，只换背后的模型：

- LOW：小模型，例如 Llama-3.2-3B / Qwen2.5-7B；
- MID：中等模型，例如 Llama-3.1-8B / QwQ-32B；
- HIGH：大模型，例如 Llama-3.3-70B / Qwen3-Next-80B-A3B。

它回答的是一个很实际的问题：同一个记忆算法，是应该调用一次大模型，还是拆成很多小模型？



### 5.4 Router：谁来决定每个模块的预算

BudgetMem 使用一个共享的 lightweight router。它在每个模块调用前观察状态 $s_k$，然后选择：

$$
a_k\in\{LOW,MID,HIGH\}
$$
状态由三部分组成：

1. 当前用户问题 (q)；
2. 当前模块的输入，也就是前一个阶段的中间结果；
3. 模块描述符，例如“现在正在路由 Temporal Module”。

论文用 `all-mpnet-base-v2` 分别编码三段文本：

-   **问题、模块输入、模块描述符，各得到 768 维向量。**
-   **问题和模块输入先投影到 256 维，模块描述符单独投影到 256 维，拼接成最终 512 维 router state。**

同一个 actor-critic router 被所有模块共享。模块身份通过 descriptor 注入，所以参数可以共享，但决策仍能区分“现在是过滤模块还是时间模块”。



### 5.5 为什么使用强化学习

一次 query 就是一个 RL episode。router 按顺序选择各模块档位，整条 pipeline 执行完后才得到：

- $r_{task}$：答案质量，F1 或 LLM-as-a-judge；
- $r_{cost}$：本次记忆提取有多便宜。

最终 reward 是：

$$
r=r_{task}+\lambda\beta r_{cost}
$$


- $r_{task}\in[0,1]$：越准确越高；
- $r_{cost}$：越省钱越高；
- $lambda$：用户对成本的重视程度；
- $\beta$：让任务 reward 和成本 reward 的数值尺度相匹配。

这里的 $\lambda$ 不是某个模块的选择概率，而是全局的 trade-off preference（权衡偏好）：

- $lambda=0$：只追求性能，近似 performance-first；
- $lambda$ 变大：越来越重视省钱，router 更常选 LOW；
- 代价是性能可能下降。



**成本如何计算**

所有模块的 raw extraction cost 相加：

$$
c_{raw}=\sum_k c(M_k,a_k)
$$
对 LLM 模块，论文根据 input/output token 数乘以对应服务价格；规则模块等非 LLM tier 的成本相对视为可忽略。

为了让不同批次、不同数据集的美元成本能转成稳定 reward，论文使用滑动窗口中的第 5 和第 95 百分位做归一化：

$$
\tilde c=\frac{c_{raw}-Q_5}{Q_{95}-Q_5}
$$

$$
r_{cost}=1-\text{clip}(\tilde c,0,1)
$$

直觉上，最近一段样本里最贵的 5% 得分接近 0，最便宜的 5% 得分接近 1，中间按比例变化。

然后再根据近期 reward 的标准差设置：

$$
\beta=\frac{std(r_{task})}{std(r_{cost})+\epsilon}
$$
避免某一项因为数值波动更大而支配 PPO 更新。



### 5.6 PPO 的直觉

**PPO（Proximal Policy Optimization）**是一种强化学习算法，适合训练“在状态下选择离散动作”的策略。

在 BudgetMem 中：

```text
状态：当前 query + 当前中间结果 + 当前模块名
动作：LOW / MID / HIGH
回报：整条 pipeline 的答案质量 - 成本代价
```

因为一个 query 会连续做多次 tier 选择，论文把这些选择的 log probability 加起来，形成整条路由轨迹的联合概率，再用标准 PPO clipped objective 更新共享 router。训练配置包括 Adam、batch size 32、最多 600 steps、PPO 每次更新 4 个 epoch。



### 5.7 BudgetMem 的一个核心限制：它是软预算，不是硬预算

虽然名字叫 BudgetMem，但论文的主要控制方式是 reward 中的成本权重 (lambda)，并没有给出类似“每个 query 严格不超过 0.5 美元 / 100k token”的硬约束优化。

因此它学到的是：

> 在历史分布上，平均来说如何把性能和成本折中。

而不是保证：

> 每一个 query 都严格在指定预算内。

如果部署需要 SLA（例如每个请求最多 2 秒、最多 0.2 美元），还需要在 router 外再加 hard budget checker，或者把剩余预算作为 state 并做约束 RL。



## 六、实验

### 6.1 数据集与设置

论文使用三个数据集：

| 数据集          | 主要能力           | Train / Val / Test | 平均上下文长度 |
| --------------- | ------------------ | -----------------: | -------------: |
| **LoCoMo**      | 超长期对话记忆     |   1236 / 446 / 304 |   18.0K tokens |
| **LongMemEval** | 长期交互式助手记忆 |      282 / 94 / 94 |  122.1K tokens |
| **HotpotQA**    | 长上下文、多跳问答 |  5250 / 1750 / 128 |   26.0K tokens |

LoCoMo 的 adversarial Category 5 被排除。所有方法使用相同的数据划分、相同候选检索设置和 top-k 预算，尽量保证比较集中在 memory pipeline。

基础答案 / 记忆处理模型包括：

- `LLaMA-3.3-70B-Instruct`；
- `Qwen3-Next-80B-A3B-Instruct`。

论文先用 LLaMA 训练 router，然后直接把同一个 router 测到 Qwen 上而不重新训练；表格中的 `*` 标记表示 Qwen 结果属于这种 transfer evaluation。

对比方法有 ReadAgent、MemoryBank、A-MEM、LangMem、Mem0、MemoryOS 和 LightMem。指标包括 F1、LLM-as-a-judge 分数和美元 cost。



### 6.2 主结果：性能优先设置

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1784828533952_image.png)

**Cost 是论文按模型 API token 价格计算的记忆提取成本，不是整个系统的所有成本。**

这些结果说明：在论文的测试协议下，BudgetMem 不只是省钱，而且在三个任务上同时提升了 F1 / Judge。但需要注意，所有方法的候选 chunks 数量都被控制为 top-5；结果更准确地说明“候选相同的时候，后续 budget-aware processing 更有效”，不能直接等同于所有生产 RAG 配置下都优于基线。



### 6.3 三种 tiering strategy 谁更好

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1784829061319_image.png)

不同策略的性格很清楚：

- **Implementation tiering（IMP）**：从规则升级到 BERT 再升级到 LLM，成本覆盖范围较宽；中等预算下质量提升较快。
- **Reasoning tiering（REA）**：保持模型容量不变，只从 direct 升到 CoT / reflection；质量改善明显，但成本跨度较窄，因为 token overhead 会集中在一个范围。
- **Capacity tiering（CAP）**：换小、中、大模型；高预算时质量上限最好，但模型调用可能更昂贵。

论文在 LoCoMo 上改变 $\lambda$ 画出了性能—成本曲线：

-   放松预算时，Judge 逐渐升高、Cost 逐渐增加，形成较平滑的 frontier。
-   CAP 在高预算区通常质量最好；
-   IMP 适合快速扩大预算覆盖；
-   REA 更像在相对窄的成本区间内做细粒度质量调节。



### 6.4 Router 是否真的学会了“看预算行事”

在 LongMemEval 的 capacity tiering 分析中：

- $\lambda$ 较小时，router 大量选择 MID，偏向质量；
- $\lambda=0.3$ 左右时，LOW 比例上升，但仍保留相当 MID；
- 成本压力更大时，各模块进一步集中到 LOW。

这说明 router 没有简单地永远选同一个档位，而是会随着成本偏好变化调整模块决策。

更有意思的是，不同模块的档位比例不完全相同：Temporal 和 Summary 可能在复杂问题上更值得使用 HIGH，而 Filter 在一些问题上用廉价规则就足够。这正是模块级路由比“一键切换全流程高预算”更有意义的地方。



### 6.5 Reward-scale alignment 消融

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1784829201728_image.png)

论文去掉 $\beta$ 的 reward-scale alignment 后发现：当成本权重 $lambda=0.3$ 时，router 容易塌缩到大多数模块都选择 LOW。虽然成本很低，但 Judge 分数降到最低。

这说明一个很实际的问题：

```text
任务 reward 的数值范围与成本 reward 的数值波动不匹配
       -> PPO 更新被成本项支配
       -> 学会“只省钱”，而不是“用合适的预算解决问题”
```

加入 $\beta$ 后，tier 选择变得渐进，性能—成本曲线更平滑。这是论文中比较关键的训练工程细节。



### 6.6 检索 chunk 数量敏感性

论文还改变初始检索的 raw chunks 数量。chunk 越多，通常：

- 输入变长，处理成本增加；
- 可能提供更多证据；
- 但冗余、无关和噪声也会增加。

在它的设置下，取 `5` 个 chunks 达到较好的平衡；太少证据不足，太多反而可能干扰 LLM。这说明 BudgetMem 只控制后续模块档位，并没有消除初始检索 top-k 这个重要成本旋钮。



### 6.7 延迟分析

在统一的本地 Qwen 部署下，**Implementation tiering 的总延迟随成本压力增加而下降：**

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1784829290618_image.png)

这揭示了成本的主要来源：远程 / 大模型记忆处理的调用，而不是 router 自身。router 的 GPU 时间大约只有 `44–52 ms`，控制器本身很轻。

但是 BudgetMem 的绝对延迟并不总比 offline memory 方法低。Table 2 中：

- MemoryOS offline construction `40657 ms`，总推理 `2842 ms`；
- A-MEM offline construction `26842 ms`，总推理 `1449 ms`；
- LightMem offline construction `9740 ms`，总推理 `1662 ms`；
- BudgetMem 不需要离线构建，但 **performance-first** 版本总推理约 `3.5–3.9 s`。

因此，BudgetMem 是用**不提前付费、查询时按需付费**替代 offline preprocessing。重复查询很多次时，是否划算取决于是否缓存 runtime memory；论文没有完全回答这个 amortization（摊销）问题。



### 6.8 实验结果分析

**优点**

- 比较了多个 memory baseline、多个 backbone 和三个长期记忆数据集；
- 同时报告质量、金钱 cost 与 latency，而不是只报 accuracy；
- 证明了 router 的档位选择随成本权重变化，具有可解释性；
- 有 reward alignment、retrieval size 和 latency 分析。



**仍然需要谨慎的地方**

1. **成本口径不是完整端到端账单**：主要是 memory extraction 的模型调用 token 价格，最终 answer generation、embedding、retriever、服务网络与离线索引成本需要单独计算。
2. **价格依赖供应商**：美元成本随 API provider、缓存、批量折扣和模型价格变化，不是硬件无关的物理量。
3. **硬预算没有验证**：reward 惩罚不等于每个 query 都不超预算。
4. **模块实现存在较强实验工程成分**：规则、BERT 和 LLM 的具体实现质量会明显影响 tier 曲线；三条 tiering 轴并非完全公平的“只改变一个变量”。
5. **运行时重复处理历史**：它保留 raw chunks 避免不可逆信息损失，但同一历史被许多问题反复访问时，成本可能高于一次性的离线压缩。
6. **学习到的是预算策略，不一定是通用记忆策略**：router 可能适应 LoCoMo、LongMemEval 和 HotpotQA 的查询分布；迁移到真实多工具 Agent 时，state 和模块结构都要重新设计。



## 7. 结论和展望

### 7.1 论文贡献

BudgetMem 的核心贡献可以压缩为：

> **把“记忆提取要花多少计算”从固定流水线超参数，变成 query-aware、module-wise 的策略决策。**

它提出的最有价值的抽象是：

```text
一个 memory pipeline
    -> 多个功能模块
    -> 每个模块多个质量/成本档位
    -> 一个共享 router 按 query 动态选择
    -> 用任务收益和成本收益训练选择策略
```

对于你之前关心的“Agent Memory 很烧 token”问题，这篇论文给出了一个清晰答案：**不是所有问题、所有模块都应该使用同样强的记忆处理；预算应该沿着 query 和 pipeline stage 进行分配。**



### 7.2 它与其他 memory cost 方向的区别

| 方向                  | 主要控制对象                 | BudgetMem 的位置                          |
| --------------------- | ---------------------------- | ----------------------------------------- |
| Mem0 / A-MEM          | 写入什么、更新什么、存多少条 | BudgetMem 不重点优化长期 store 的维护     |
| LightMem / 摘要压缩   | 如何提前压缩历史             | BudgetMem 尽量把语义加工推迟到 query time |
| RAG top-k routing     | 取多少文档、调用哪个回答模型 | BudgetMem 主要控制检索后记忆提取模块      |
| MLP Memory / 参数记忆 | 用参数替代外部 datastore     | BudgetMem 仍是外部文本记忆路径            |
| RL-driven memory      | 学什么时候读、写、删         | BudgetMem 学的是每个提取模块选择哪档预算  |



### 7.3 论文的局限与尚未解决的问题

- **无硬预算保证**：未来可以把剩余美元、token 或 latency 作为显式 state，并使用 constrained RL / Lagrangian optimization。
- **没有完整 amortized cost model**：应该比较“离线一次构建 + 多次查询”和“每次 query 重新提取”的长期总成本。
- **缺少缓存与重复查询策略**：如果两个 query 相似，是否复用上一次 (m)，还是重新执行高预算模块？
- **缺少记忆存储更新**：runtime memory 生成后是否进入长期库、如何写回、何时失效，仍是空白。
- **成本与风险没有联合建模**：医疗、金融等高风险问题，即使成本低，也可能必须强制 HIGH 或要求多路验证。
- **router 的跨域泛化有限**：模块身份、tier 实现和 state encoder 都和实验 pipeline 绑定。



### 7.4 启发

如果沿着 Agent Memory Cost 做研究，一些可能的想法：

1. **Budget-aware memory lifecycle**：同时决定“要不要写入长期 memory、写入哪种形式、何时检索、何时总结、何时遗忘”。
2. **硬约束路由**：给定每个请求的 token / 美元 / latency 上限，保证不超预算，并最大化任务质量。
3. **价值感知记忆**：让 router 估计一条记忆的未来复用价值。高复用事实可以提前结构化，低复用内容保留 raw chunk，减少重复提取。
4. **成本—不确定性联合决策**：当 router 不确定是否能正确回答时，主动升级某个模块；简单问题用 LOW，高风险或冲突问题用 HIGH。
5. **跨查询 memory cache**：缓存 query-specific memory，并学习何时复用、何时重新生成，以解决 runtime extraction 的重复付费问题。
6. **真实 Agent 评测**：把 cost 从“每个 QA query 的美元”扩展为整段会话的 token、API、延迟、存储、失败重试和人工纠错总成本。



### 7.5 最终理解

BudgetMem 不是在说“记忆越少越好”，而是在说：

> **记忆处理是一种资源分配问题；真正聪明的 Agent 不只是知道去哪里找信息，还知道这一次搜索值得投入多少计算。**

它为 Agent Memory 引入了一个很实用的研究轴：从“记住多少、记得准不准”，扩展到“为了这一次回答，记忆系统应该花多少钱”。























