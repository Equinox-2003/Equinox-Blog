---
title: "Belief Memory精读"
description: ""
date: 2026-07-09T00:00:08+08:00
lastmod: 2026-07-09T00:00:08+08:00
draft: true

categories:
  - paper-reading
tags:
  - LLM
  - Agent
  - Agent Memory
toc: true
math: true
mermaid: true
cover: https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783526645978_image.png
banner: https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783572736870_image.png
---

<!--more-->



## 零、写在前面

很多 agent memory 工作都着力于怎么去更好地管理记忆，往往忽略了记忆本身是否可靠，这篇工作将 agent 的记忆观测类比成POMDP（Partially Observable Markov Decision Process，部分可观测马尔可夫决策过程），选择维护多个候选记忆以及其置信度，从而提高记忆的可信度。



## 一、标题

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783526645978_image.png)

- **Belief Memory**：不是只存一句“确定事实”，而是存“多个可能结论 + 每个结论的可信度”。
- **Agent Memory**：让 AI Agent 能在多轮、多会话、长期任务中保存和调用历史信息。
- **Partial Observability**：Agent 看到的不是世界真相本身，**而是带噪声、片面的观察**。

**POMDP（Partially Observable Markov Decision Process，部分可观测马尔可夫决策过程）** 是论文的方法基础。简单来说就是：

> 你在一个看不见全貌的房间里做决策，只能通过门缝、声音、别人告诉你的线索来猜房间里到底发生了什么。

AI Agent 也是这样。比如：

- 用户说“我最近不太想吃辣”，这不一定代表用户永远不吃辣。
- 工具 API 连续超时，可能是 API 挂了，也可能只是临时限流。
- 在 ALFWorld 这类环境中，Agent 看到“杯子不在桌上”，不代表杯子不存在，可能只是被放进了柜子。

这篇论文抓住的核心问题是：**Agent 的记忆如果把片面观察直接写成确定结论，就会越来越自信地犯错。**



## 二、摘要

摘要省流就是：**BeliefMem 让 Agent 记住多个候选结论及其置信度，从而在不确定环境中持续修正记忆。**

摘要里作者指出，**很多现有 Agent Memory 方法都会把一次观察压缩成一个确定结论**。比如 Agent 看到 API X 多次 timeout，就写入：

```text
API X failed
```

但真实情况可能有多个解释：

```text
API X failed              0.50
API X rate limiting       0.35
Network issue             0.15
```

这就是 **candidate conclusions（候选结论）**。每个候选结论都有一个 **probability（概率或置信度）**。在论文实现里，这个值更准确地说是 LLM 抽取出来的 evidence strength，也就是“当前观察多支持这个结论”，不一定是严格校准过的贝叶斯概率。

作者认为，确定性记忆会导致 **self-reinforcing error（自我强化错误）**：

1. Agent 观察到片面现象。
2. Memory 写入一个过早的确定结论。
3. Agent 后续行动受这个结论影响。
4. 它不再探索其他可能性。
5. 错误记忆不断被自己的行为强化。

生活类比：

> 你第一次去一家餐厅，服务员态度不好，于是记下“这家店很差”。之后你再也不去，也不会知道那天只是临时缺人。这个结论就没有机会被修正。

BeliefMem 的办法是：**不要只存一个结论，要把多个可能解释都留在记忆里，并随着新证据更新它们的可信度。**



## 三、引言

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783529286661_image.png)

### 3.1 论文要解决的根本问题

引言从 Agent Memory 的长期任务需求讲起。现在的 LLM Agent 越来越多地用于：

- 长期对话助手
- 自动化科研助手
- 工具调用 Agent
- embodied agent，也就是在环境中行动的智能体

这些场景都要求 Agent 记住过去发生过什么。于是出现了很多外部记忆系统，比如 Generative Agents、MemGPT、Mem0、A-MEM、MemoryBank、Zep、MemOS 等。

但作者认为，这些方法虽然存储结构不同，检索策略不同，记忆管理方式不同，却共享一个隐含假设：

> **每条 memory entry 通常只保存一个确定结论。**

这就是论文称为 **deterministic paradigm（确定性范式）** 的东西。



### 3.2 确定性记忆为什么危险？

确定性记忆的问题不只是“可能记错”，而是 **记错之后很难翻身**。

论文用 API timeout 举例：

- Session 1：API X 连续超时。
- 传统 memory 写入：API X failed。
- Session 5：用户让 Agent 搜索论文。
- Agent 读到 memory 后避免使用 API X。
- 因为不再尝试 API X，它永远不知道 API X 其实已经恢复。

这就形成闭环：

```text
片面观察 -> 错误结论 -> 错误行动 -> 缺少反证 -> 错误结论继续存在
```

这也是这篇论文最有研究价值的地方：它把 Agent Memory 的问题从“如何存得更多、检索更准”推进到“如何在不确定环境下维护可修正的信念”。



### 3.3 本文的核心贡献

论文提出 **BeliefMem**，主要贡献可以概括为三点：

- 提出 deterministic memory 会导致 self-reinforcing error，尤其是在部分可观测环境中。
- 将 memory entry 从单一结论改为 **attribute-level belief representation（属性级信念表示）**。
- 在 LoCoMo 和 ALFWorld 上验证，概率化记忆能提升长期对话记忆和 embodied agent 任务表现。

这里的 **attribute（属性）** 可以理解为“Agent 想判断的一件事”。比如：

- API X 的状态
- 用户对辣食的偏好
- 某个物体的位置
- 某个任务环境中的行动规律

每个 attribute 下面可以有多个 **hypothesis（假设或候选结论）**。比如“API X 的状态”下面可以有：

- 永久失败
- 临时限流
- 网络问题



## 四、相关工作

论文把相关工作分成三条线。

### 4.1 Factual and RL-Based Memory

**Factual Memory（事实记忆）** 指保存用户、环境、事件等事实信息的记忆。比如：

```text
用户不喜欢早上开会。
钥匙通常在厨房抽屉里。
API X 上次调用失败。
```

论文提到的代表包括：

- **Generative Agents**：用自然语言 memory stream 记录经历。
- **MemGPT**：把上下文、回忆、外部存储做成类似虚拟内存管理。
- **Mem0**：抽取、更新、合并重要事实，并用向量检索。
- **A-MEM**：把记忆组织成结构化 notes，并建立索引和链接。
- **MemoryBank**：引入遗忘曲线，动态调整记忆强度。
- **Zep**：用 temporal knowledge graph 保存演化信息。
- **MemOS**：把不同类型 memory block 统一管理。

论文也提到 **RL-based Memory（基于强化学习的记忆管理）**，比如 Memory-R1、MEM1、Agentic Memory、MemRL。这些方法想让模型学会何时 add、update、delete memory。

但作者的批评是：

> 这些方法主要改的是“怎么管理记忆”，不是“记忆内部如何表示不确定性”。多数 memory entry 仍然只保留一个结论。



### 4.2 Self-improving Memory

**Self-improving Memory（自我改进记忆）** 指 Agent 从过去经验中总结教训或技能，用来指导未来行动。

代表方法包括：

- **Reflexion**：失败后生成反思，用反思指导下一次尝试。
- **ExpeL**：从多条轨迹中总结可复用经验。
- **Voyager**：在 Minecraft 等环境中自动积累技能库。
- **MemSkill**：学习并演化可迁移的 memory skills。

这类方法的问题是：它们虽然从“事实”升级到“经验”，但仍常常写成确定性教训。

比如：

```text
Avoid API X.
```

这条经验看起来有用，但如果 API X 只是临时限流，这条经验就会误导后续行动。



### 4.3 Belief State under Partial Observability

**Belief State（信念状态）** 是 POMDP 中的核心概念，表示“在看不到真实状态时，我对各种可能真实状态的概率分布”。

人类也经常这样做：

> 朋友迟到了 30 分钟，你不会立刻断言“他不尊重我”。你可能同时保留几个解释：堵车、忘记时间、手机没电、确实不重视。这就是 belief state。

论文认为，现有 Agent Memory 系统还没有充分吸收这个思想。很多系统把观察当真相，把猜测当事实，所以无法优雅处理矛盾和不确定。



## 五、方法

### 5.1 Memory 是 belief state 的近似

论文首先给出一个重要视角：

> 外部记忆可以看成 Agent 对环境 belief state 的一种可计算近似。

也就是说，理想情况下，Agent 应该根据所有历史观察和行动维护一个完整的信念分布。但真实任务太复杂，不可能显式保存所有世界状态，所以我们用 external memory 来压缩历史。

论文中的流程可以简化为：

-   当前观察 $o_t$
    -   Read($M_t$, $o_t$) 得到记忆上下文 $z_t$
    -   Agent 根据 $o_t$ 和 $z_t$ 选择行动 $a_t$
    -   执行动作后得到新观察 $o_{t+1}$
    -   Update($M_t$, $o_t$, $o_{t+1}$) 更新记忆

这里：

- **Mt（memory at time t）**：第 t 步时的外部记忆。
- **Read（读取）**：从记忆中找当前有用的信息。
- **Update（更新）**：把新观察写入或合并到记忆。



### 5.2 只存 argmax 会丢掉不确定性

传统方法通常对一个属性 c 只存一个最可能结论：

$$
M_t = (c, \hat h_{t}(c)) : c \in C_t
$$
用作者前面那个例子的话：

- c 是属性，比如“API X 的状态”。
- $\hat h_t(c)$ 是当前最可能的结论，比如“API X failed”。
- $C_t$ 是当前记忆里保存的属性集合。

这相当于在多个候选结论里只保留分数最高的那个。

问题是：其他候选结论被删掉了。比如“临时限流”“网络问题”不再出现在记忆里，后续 Agent 也就很难主动考虑它们。



### 5.3 Belief-based Memory Formulation

BeliefMem 理想上希望每个属性 c 都保存一个分布：$(c, b_t^c)$

其中 $b_t^c$（属性 c 在时间 t 的信念分布） 表示：

```text
对属性 c 的每个候选结论 h，都保存一个可信度。
```

理想形式类似：

```text
API X status:
  API X failed          0.10
  API X rate limiting   0.80
  Network issue         0.10
```

但是论文也承认，真实环境里的候选空间是开放的，Agent 不可能提前列出所有可能结论。所以 BeliefMem 做了一个近似：

> **只保存被历史观察支持过的候选结论，而不是枚举所有可能结论。**

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783532133365_image.png)



### 5.4 Add：发现新属性或新结论时写入

**Add（新增操作）** 用于新观察支持了一个记忆库里还没有的属性或候选结论。

比如用户第一次说：

```text
我最近在准备 Agent Memory 方向的论文。
```

BeliefMem 可能新增：

```text
attribute: user's research focus
candidate: Agent Memory
probability: 0.85
```

论文中 Add 的概率会被限制在 `[pmin, pmax]`，具体实现是 `[0.7, 0.9]`。这有一个很直观的意义：

> 新观察再强，也不要一上来就写成 100% 确定。

这和人类很像。你第一次听朋友说“我最近喜欢跑步”，你会相信，但不会立刻认为“他一辈子都是跑步爱好者”。



### 5.5 Merge：已有属性收到新证据时合并

**Merge（合并操作）** 用于新观察支持了已有属性下的某个候选结论。

论文使用 **Noisy-OR evidence merge（Noisy-OR 证据合并）** 来更新概率。直觉公式是：

$$
prob_{new} = min(0.99, 1 - (1 - prob_{old}) * (1 - EvidenceStrength))
$$
并且上限截断到 0.99。

这是什么意思？

假设旧概率是 0.70，新证据强度是 0.60，那么：

```text
new_prob = 1 - 0.30 * 0.40 = 0.88
```

类比一下：

> 你原本 70% 相信“API X 是临时限流”，后来又看到一个强证据支持这个解释，于是信心上升到 88%。但不会升到 100%，因为世界总可能还有例外。

论文特别强调：这里的 probability 在实现里是 **confidence score（置信分数）**，不是严格贝叶斯意义上的校准概率。



### 5.6 Contradictory Memory：遇到矛盾证据怎么办？

**Contradictory Memory（矛盾记忆处理）** 是论文里很关键的一点。

当新观察支持同一属性下的矛盾结论时，BeliefMem 会降低原候选结论的置信度，比如降到 0.25，并保留旧版本作为历史版本。

例如：

```text
旧记忆：
API X failed          0.90

新观察：
API X worked after 3.1s

更新后：
API X failed          0.25
API X rate limiting   0.80
```

这里的重点不是简单覆盖，而是让 Agent 能看到：

- 过去为什么这么认为
- 现在为什么变了
- 当前哪个解释更可信

这对长期 Agent 很重要，因为环境和用户偏好经常会变。



### 5.7 Belief-aware Retrieval：检索时不能把不确定性丢掉

只存 belief 还不够。如果检索时又只拿出最强结论，那不确定性还是丢了。

所以 BeliefMem 提出 **Belief-aware Retrieval（信念感知检索）**。它先给每个属性 c 算一个检索分数：

$$
\alpha_t(c) = sim(o_t, c) \times \lambda ^ {\tau_t(c)}
$$
新手可以这样理解：

- **sim(ot, c)**：当前问题和这个属性有多相关。
- **lambda（衰减率）**：越小越偏向近期记忆，越大越愿意保留早期记忆。
- **Delta_t(c)**：这个属性距离上次更新过去了多久。

然后系统选 Top-K 个属性返回，但返回的不是单一结论，而是：

```text
attribute + 该属性下所有候选结论及其概率
```

例如：

```text
API X status:
  API X failed          0.10
  API X rate limiting   0.80
  Network issue         0.10
```

这让 Agent 在决策时能看到备选解释，而不是被一个“看似确定”的 memory 牵着走。



### 5.8 方法整体流程

BeliefMem 可以总结成一个闭环：

```text
新观察
  -> 抽取 attribute 和 candidate conclusion
  -> 如果是新属性，Add
  -> 如果是旧属性，Merge
  -> 如果出现矛盾，降低旧候选置信度并保留历史版本
  -> 检索时返回候选分布
  -> Agent 基于当前观察 + belief memory 行动
  -> 新行动产生新观察，继续更新
```



## 六、实验

### 6.1 实验任务

论文在两个 benchmark 上评估 BeliefMem。

**LoCoMo（Long-term Conversational Memory benchmark，长期对话记忆评测）** 用来测试模型能否在长对话、多 session 中记住并推理历史信息。论文提到其对话平均约 9000 tokens，最多 35 个 sessions。

它包含四类问题：

- **Single-hop（单跳问题）**：从一个片段里找事实。
- **Multi-hop（多跳问题）**：把多个片段的信息组合起来。
- **Temporal（时间推理）**：判断事件先后、持续时间等。
- **Open-domain（开放域问题）**：结合对话历史和外部常识。

指标是 **F1** 和 **BLEU-1**。



**ALFWorld（文本形式的 embodied agent 环境）** 用来测试 Agent 在家庭任务环境中行动的能力，比如找物品、清洁物品、加热物品、放置物品。

指标包括：

- **SR（Success Rate，成功率）**：任务是否完成。
- **#Steps（平均步数）**：成功任务平均用了多少步，越少越好。

ALFWorld 分成 Seen 和 Unseen：

- **Seen**：和训练环境分布相近。
- **Unseen**：房间布局、物体实例等不同，更考察泛化。



### 6.2 对比方法

LoCoMo 中对比了：

- LoCoMo baseline
- ReadAgent
- MemoryBank
- MemGPT
- A-MEM
- Mem0
- BeliefMem

ALFWorld 中还加入：

- No-Memory
- LangMem
- MemoryOS

基础模型方面：

- LoCoMo 使用 GPT-4o-mini 和 GPT-4o。
- ALFWorld 使用 Qwen3-Next-80B-A3B-Instruct。



### 6.3 主结果：LoCoMo

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783533274460_image.png)

在 GPT-4o-mini 上，BeliefMem 的平均表现为：

```text
BeliefMem: 42.38 F1 / 36.30 BLEU
Mem0*:    40.99 F1 / 31.99 BLEU
A-MEM*:   32.42 F1 / 26.46 BLEU
MemGPT:   25.59 F1 / 19.74 BLEU
```

在 GPT-4o 上，BeliefMem 的平均表现为：

```text
BeliefMem: 42.87 F1 / 37.08 BLEU
Mem0:     39.66 F1 / 29.54 BLEU
A-MEM*:   34.45 F1 / 28.45 BLEU
MemGPT:   30.02 F1 / 25.31 BLEU
```

值得注意的是，BeliefMem 在 **Temporal** 类问题上优势明显：

```text
GPT-4o-mini Temporal:
BeliefMem: 51.88 / 45.78
Mem0*:     48.93 / 40.51
A-MEM*:    45.85 / 36.67
```

论文的解释是：BeliefMem 会保留历史版本和时间信息，因此更适合回答“以前是什么情况、后来怎么变了”这类问题。



### 6.4 主结果：ALFWorld

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783533467100_image.png)

在 ALFWorld 上，BeliefMem 的结果很有意思：

```text
BeliefMem*（50% memory corpus）:
Seen SR:   58.57
Unseen SR: 61.19
Avg SR:    59.88

BeliefMem（full memory corpus）:
Seen SR:   63.57
Unseen SR: 53.75
Avg SR:    58.66
```

这里 **BeliefMem*** 使用的是 50% memory corpus，反而在 Unseen 和 Avg 上更好。论文认为这可能说明：

> 过多 in-distribution 记忆可能让 Agent 更依赖见过的轨迹，从而损害泛化。

这是一个很值得研究的现象：**记忆不是越多越好，记忆也会过拟合。**

与其他 baselines 相比，BeliefMem* 的 Avg SR 是 59.88，ReadAgent 是 54.03，Mem0 是 39.81，MemoryBank 是 37.96。



### 6.5 消融实验

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783533572023_image.png)

论文做了四个核心消融：

- **w/o belief-based memory**：去掉概率化信念，只保留确定性结论。
- **w/o belief-aware retrieval**：存多个候选，但检索时不返回概率。
- **w/o Add**：不新增属性。
- **w/o Merge**：不合并新证据。

结果如下：

```text
LoCoMo:
w/o belief-based memory   22.58 F1
w/o belief-aware retrieval 28.50 F1
w/o Add                   14.48 F1
w/o Merge                 20.38 F1
BeliefMem                 42.38 F1

ALFWorld:
w/o belief-based memory   28.71 SR
w/o belief-aware retrieval 51.77 SR
w/o Add                   22.58 SR
w/o Merge                 40.81 SR
BeliefMem                 59.88 SR
```

这说明：

- 只存确定结论会严重掉点。
- 存多个候选但不带概率也会掉点。
- Add 和 Merge 都是核心组件，尤其 Add 去掉后性能崩得很明显。



### 6.6 Belief convergence：信念会不会越来越接近真相？

论文还分析了 **belief convergence（信念收敛）**。

他们在 LoCoMo 多跳任务中选取能映射到单一属性结论的样本，观察随着证据增加，真实结论是否能成为最高置信度候选。

结果是：

```text
BeliefMem Top-1 rate: 87.68%
```

**也就是说，随着证据积累，真实结论在多数情况下会变成最高置信度候选。相比之下，简单用证据出现频率当信心的 baseline 容易被噪声干扰。**



### 6.7 Adversarial correction：错误记忆能不能被修正？

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783533748945_image.png)

论文还做了对抗实验：先往 memory bank 里注入错误结论，再看系统能不能通过后续观察修正。

结果：

```text
BeliefMem correction rate: 60.80%
Deterministic baseline:    33.30%

BeliefMem correction steps: 4.75
Deterministic baseline:    9.45
```

这说明 BeliefMem 不只是“记得更多”，而是更有机会从错误记忆中恢复。



### 6.8 成本分析

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783533766926_image.png)

论文附录报告了 LoCoMo 上每次 generation 的平均 token 消耗：

```text
Mem0:      2115.85
A-MEM:     1755.57
BeliefMem: 1414.11
```

这点比较重要，因为 belief memory 听起来像会更贵，但论文通过限制每个属性最多返回 4 个候选结论，让检索上下文没有无限膨胀。

不过作者也承认，写入和合并阶段仍然有额外计算成本，尤其是需要 LLM 抽取 attribute、candidate conclusion、evidence strength。



## 七、结论和展望

### 7.1 论文结论

论文的结论可以概括为：

> 在部分可观测环境中，Agent Memory 不应该过早把观察压缩成一个确定结论，而应该保留多个候选结论及其置信度，让 Agent 能在后续证据中持续修正自己的记忆。

BeliefMem 的价值不在于提出了一个很复杂的工程系统，而在于它明确指出了 Agent Memory 的一个表示层问题：

-   传统问题：怎么存、怎么检索、怎么更新？
-   本文推进：到底应该存“确定事实”，还是存“带不确定性的信念”？



### 7.2 论文局限

论文自己也列出了一些局限。

**缺少理论保证**

BeliefMem 用 noisy-OR 做近似更新，并不是完整的贝叶斯 posterior。由于候选结论空间是开放的，它没有严格证明概率一定会收敛到真相。

> 它更像一个实用的工程近似，而不是一个有完整数学保证的信念推断系统。

**证据强度来自 LLM 抽取**

Evidence strength 是由 LLM 从观察中抽取的。如果 LLM 自己判断不准，概率更新也会受影响。

这也是后续研究可以切入的地方：能否让 evidence strength 更可校准、更可解释？

**计算开销仍然存在**

BeliefMem 虽然 token 消耗低于 Mem0 和 A-MEM，但 memory writing 和 merging 需要额外抽取、匹配、更新，成本并不为零。



### 7.3 对 Agent Memory 研究的启发

这篇论文对你做 Agent Memory 很有启发，尤其是以下几个方向。

**方向一：Belief Memory + Cognitive Layer**

你之前提到的 **belief system / cognitive layer（信念系统或认知层）** 和这篇论文高度相关。

BeliefMem 目前主要是属性级候选结论概率，但人类的 belief system 还会包含：

- 信念之间的支持关系
- 信念之间的矛盾关系
- 信念的来源可信度
- 信念随时间变化的轨迹
- 不同行动对验证信念的价值

后续可以做：

```text
从 candidate-level belief memory
扩展到 structured belief graph / cognitive belief layer
```

也就是让 Agent 不只知道“哪个结论更可能”，还知道“这些结论之间为什么冲突、哪些证据支持它们、下一步应该如何验证”。



**方向二：Belief-aware Memory Manager**

BeliefMem 的 Add、Merge、Version 还是比较规则化。可以进一步研究：

- 什么时候新增候选？
- 什么时候合并候选？
- 什么时候删除低置信度候选？
- 什么时候主动探索以验证某个候选？

这可以和 RL-driven Memory 结合，让 memory manager 学会以长期任务收益为目标管理 belief。



**方向三：Calibration for Memory Confidence**

论文中的 probability 实际上是 LLM 抽取的 confidence score，不是严格校准概率。

一个很自然的研究问题是：

> Agent Memory 里的置信度如何校准，才能真的反映未来任务中的正确性？

可以考虑：

- 用 verifier 校准 evidence strength。
- 用历史预测正确率校准 memory confidence。
- 用不确定性估计方法区分“模型不知道”和“证据不足”。



**方向四：Belief Memory for Contradictory and Dynamic Environments**

BeliefMem 特别适合处理矛盾和变化：

- 用户偏好变化
- 工具状态变化
- 多 Agent 信息冲突
- 网络搜索结果互相矛盾
- 任务环境状态动态改变

这类场景比静态问答更符合真实 Agent 需求。后续工作可以专门构造更强的 benchmark，而不是只在 LoCoMo 和 ALFWorld 上验证。

