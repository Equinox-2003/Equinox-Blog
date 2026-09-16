---
title: "论文精读 | MemGen"
description: "记忆和推理交叉推进"
date: 2026-07-30T16:47:35+08:00
lastmod: 2026-07-30T16:47:35+08:00
draft: false

categories:
  - paper-reading
tags:
  - LLM
  - Agent
  - Agent Memory
toc: true
math: true
mermaid: true
cover: https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1785500383856_image.png
---

<!--more-->



## 零、写在前面

大佬 Guibin Zhang 的工作，读完之后大为震撼，感觉是调研 agent memory 以来，看到的最 solid 的工作，后续会 follow up。



## 一、标题

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1785500383856_image.png)

>   **来源：ICLR 2026**
>
>   **代码：https://github.com/KANABOON1/MemGen**
>
>   **为自进化 Agent 生成 latent memory**

**MemGen**，Memory Generation，即“**生成记忆**”。论文强调，A**gent 不应只从数据库中检索一段现成文本，还可以根据当前推理状态，动态生成一段专门服务于当前思考的 latent memory。**

**Weaving** 是“编织”。它描述了论文最核心的运行方式：推理不是先检索一次记忆、再一口气生成答案，而是在推理过程中多次穿插：

~~~text
推理一段
  -> 判断现在是否需要记忆
  -> 生成 latent memory
  -> 把 memory 插回当前隐藏状态
  -> 继续推理
~~~

**Generative Latent Memory** 是“生成式潜在记忆”。它不是可直接阅读的文本，也不是从向量库原样取回的历史片段，而是由 memory weaver 生成的 K 个连续向量：
$$
M_t=[m_{t,1},m_{t,2},\ldots,m_{t,K}]
\in \mathbb{R}^{K\times d_{\text{model}}}.
$$
**Self-Evolving Agents** 表示 Agent 能从过去任务和交互经验中提高后续问题解决能力。这里的 self-evolving 重点不是保存用户姓名或偏好，而是把成功轨迹、程序技能和推理模式吸收到 memory weaver 中。

本文的定位便是：**MemGen 冻结基础 LLM，把历史经验训练进一个轻量 memory weaver，并在推理关键位置动态生成 latent tokens 来引导后续推理。**



按照《Memory in the Age of AI Agents: A Survey》的 Forms / Functions / Dynamics 框架，MemGen 需要分两层理解：

| 观察层次               | 分类                                    | 原因                                                         |
| ---------------------- | --------------------------------------- | ------------------------------------------------------------ |
| 推理时生成的 $M_t$     | **Latent Memory**                       | 记忆是 K 个 $d_{\text{model}}$ 维隐藏向量，不是文本 token，也不直接修改 reasoner |
| 历史经验长期保存的位置 | **Parametric Memory**                   | 经验通过 SFT 或 GRPO 学进 memory weaver 的 LoRA 参数         |
| 主要功能               | **Experiential Memory**                 | weaver 从过去成功轨迹中学习可迁移的推理经验                  |
| 推理过程中的作用       | **Working Memory / Procedural Support** | latent memory 在当前任务中支持规划、工具操作、格式和上下文保持 |
| Dynamics               | **按需生成、插入和再生成**              | trigger 在推理过程中决定何时重新构造 memory                  |

因此，最准确的表述不是“纯 latent memory”，而是：

> **MemGen 是一种 parametric-to-latent generative memory：经验长期存进 weaver 参数，运行时再根据当前认知状态生成 latent memory。**

它默认不是 Mem0、MemoryOS 那种可查看、可编辑、可删除的外部长期记忆数据库。



## 二、摘要

### 2.1 问题

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1785502218013_image.png)

**论文认为，Agent Memory 的目标不应局限于维持长对话连贯性，还应让 Agent 从环境交互中吸收经验，逐步提高解决新任务的能力。**

现有方法主要有两类：

1. **Parametric Memory**：通过 SFT、RL 等训练直接修改 Agent 参数。
2. **Retrieval-based Memory**：把轨迹、经验总结、工作流或工具保存在外部数据库，需要时检索并放进 prompt。

作者认为两者分别存在问题：

- Parametric Memory 会侵入基础模型参数，可能造成 catastrophic forgetting。
- Retrieval-based Memory 不修改模型，较容易维护，但通常依赖固定的 context engineering：检索文本、拼进 prompt、再继续推理，记忆与思考的结合仍较粗糙。



### 2.2 MemGen 的两个核心组件

MemGen 包含：

- **Memory Trigger**：监控当前推理隐藏状态，决定现在是 INVOKE 还是 SKIP。
- **Memory Weaver**：收到当前隐藏状态后，生成一段固定长度的 latent token sequence，作为机器原生的 memory。

运行过程可以理解为：

~~~text
Reasoner 正在生成
    -> Trigger 发现一个关键推理节点
    -> 暂停普通生成
    -> Weaver 根据当前上下文生成 K 个 latent tokens
    -> 把它们插入 reasoner 的隐藏状态
    -> Reasoner 在 memory 引导下继续生成
~~~



### 2.3 摘要中的主要实验结论

论文摘要报告：

- 在八个主要 benchmark 上，MemGen 相比 ExpeL、AWM 等外部 memory system 最高提升 38.22%；
- 相比 GRPO 最高提升 13.44%；
- 具有跨领域迁移能力；
- 在没有显式 memory type 监督的情况下，latent memory cluster 被作者解释为 planning、procedural 和 working memory。

需要注意：正文实验设置一共提到九个数据集，其中八个用于主要横向表格，AQuA 主要出现在 continual learning 实验中。因此“八个 benchmark”和“九个数据集”并不完全矛盾。



### 2.4 总结

**MemGen 让 Agent 在推理中按需生成机器可用的 latent memory，而不是只在开头检索一次文本经验。**



## 三、引言

### 3.1 论文关注的不是个性化聊天记忆

论文首先区分了两类 Agent Memory 目标。

**个性化对话 memory** 主要用于：

- 记住用户信息；
- 保持多轮会话一致；
- 在长期对话中维持上下文。

**MemGen 关注的是另一类问题：**

- **让 Agent 内化过去的任务经验；**
- **学会更好的规划和程序技能；**
- **在新任务中复用过去经验；**
- **形成 self-evolving problem-solving capability。**

**因此，MemGen 更接近“技能和推理经验的记忆”，而不是“用户档案记忆”。**



### 3.2 Parametric Memory 的局限

Parametric Memory 通过直接训练 Agent policy 来内化经验。例如：

~~~text
任务轨迹 / 成功示范
    -> SFT、GRPO、DPO 或其他训练
    -> 修改 Agent 的模型参数
~~~

**这种方式的优势是经验已经进入模型内部，推理时不必额外检索。**

问题是：

- **新领域训练可能覆盖旧能力；**
- **一轮轮训练会产生 catastrophic forgetting；**
- **每次新增经验都可能需要重新优化大型 policy；**
- **经验与基础知识混在同一参数空间中，不容易隔离。**



### 3.3 Retrieval-based Memory 的局限

Retrieval-based Memory 将经验保存在外部：

- raw trajectories；
- high-level experience；
- workflow；
- API、skill 或 MCP boxes。

它不会直接破坏基础模型，但常见流程是：

~~~text
接到任务
  -> 根据 query 检索几段经验
  -> 把经验追加到 prompt
  -> Agent 开始推理
~~~

作者认为这种方式存在两个不足：

1. **检索和推理相互分离**：memory 常在任务开始时一次性加入，不能随着思考过程动态变化。
2. **以提取为主而非重构**：系统主要取回现有内容，而不是结合当前问题重新生成更合适的 memory。



### 3.4 为什么引出 Latent Memory？

Latent state 是模型内部的连续向量。它具有几个潜在优势：

- **信息密度高**；
- **不必转成自然语言**；
- **可以直接参与后续神经计算**；
- **能表达很难用短文本完整描述的策略模式**。

但作者认为现有 latent memory 仍没有同时做到：

- **reasoning 与 memory 持续交织；**
- **根据当前认知状态生成 memory；**
- **不修改核心 reasoner；**
- **不局限于 embedding 相似度检索。**

论文由此提出研究问题：

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1785502434034_image.png)

> 如何把 Agent Memory 设计成一种动态认知能力，使记忆能够被重构，并与推理过程持续交织？



### 3.5 MemGen 的回答

MemGen 的设计是：

1. 冻结 core reasoner；
2. 使用 trigger 监控 reasoner 的隐藏状态；
3. 在关键语义边界判断是否调用 memory；
4. 使用 weaver 根据当前状态生成 latent memory；
5. 把 latent memory 插回 reasoner；
6. 继续推理，并允许之后再次调用。

这样，新增经验主要进入 memory weaver，而不是覆盖 reasoner 的基础参数。



## 四、相关工作

### 4.1 LLM 与 Agent Memory

论文将提高 Agent 问题解决能力的 memory 分为三类。

>   因为 Guibin Zhang 是 《Memory in the Age of AI Agents: A Survey》的牵头人，所以这里其实就是那篇 survey 中，Form 维度的分类方式。

#### Parametric Memory

经验被训练进 Agent 参数或外部参数模块，例如 FireAct、AgentLumos 等。

优点是调用自然；缺点是训练侵入性强，可能遗忘通用能力。

#### Retrieval-based Memory

经验被抽象成文本知识、轨迹、技能或工具，在推理时检索，例如 ExpeL、MemoryBank、AWM。

优点是显式、可维护；缺点是依赖检索质量和 prompt 拼接，memory 与 reasoning 的交互通常较粗。

#### Latent Memory

使用隐表示编码和调用经验。MemGen 将自己归入这一类，但强调两个区别：

- memory 在推理过程中动态生成并插入；
- memory 是由当前状态刺激出来的重构结果，而不是只按 embedding similarity 取回。



### 4.2 Latent Computation

论文将相关 latent computation 分为：

1. **Native latent reasoning**：Coconut、CODI、LatentR3、CoLaR 等，让模型在连续空间中推理。
2. **Latent steering**：LaRS、LatentSeek、SoftCoT、Coprocessor 等，用 latent representation 干预或引导生成。

MemGen 更接近第二类：它不把整个 reasoning 都改成 latent，而是在必要位置生成一小段 latent memory 来 steering 冻结的 reasoner。



### 4.3 LLM Decoding

MemGen 的 weaver 根据当前 decoding context 生成额外的 latent tokens，形式上与 speculative decoding 的 drafter 有相似之处。

两者目标不同：

- Speculative decoding 主要追求加速普通 token 生成；
- MemGen 主要追求生成能承载经验的 latent memory。



### 4.4 Reinforcement Learning

MemGen 使用 rule-based RL 训练 trigger，并允许使用 GRPO 训练 weaver。

论文将其与 RLVR、GRPO 和其他 RL Agent 工作联系起来，但强调：

- MemAgent、MEM1 等工作更关注长上下文管理；
- MemGen 关注的是让经验形成可迁移、可动态调用的 latent memory。



## 五、Preliminary

就是基础的 RL 术语在 自回归 llm 语境下的表述罢（

### 5.1 Agent 轨迹记号

设环境为 $\mathcal{E}$，由 LLM 参数 $\theta$ 驱动的 Agent policy 为：

$$
\pi_\theta.
$$
给定任务 $x$，Agent 与环境交互形成高层轨迹：

$$
\tau=(s_0,a_0,s_1,a_1,\ldots,s_T),
$$
其中：

- $s_t$ 是第 $t$ 步环境状态；
- $a_t$ 是 Agent 在该状态下采取的高层 action。

一个 action 本身又是 LLM 自回归生成的 token 序列：

$$
a_t=(z_{t,1},z_{t,2},\ldots,z_{t,L_t}).
$$
第 $j$ 个 token 的生成满足：

$$
z_{t,j}\sim\pi_\theta(\cdot\mid s_t,z_{t,\lt j}). \tag{1}
$$
执行整个 action 后，环境从 $s_t$ 转移到 $s_{t+1}$，最后通过 $R(\tau)$ 评价整条轨迹是否成功。

### 5.2 带 Memory 的统一目标

历史经验记为：

$$
\mathcal{H}=\{(x_i,\tau_i)\}_{i=1}^{N}.
$$
论文希望联合 policy 与 memory system $\mathcal{M}$，提高新任务上的期望 reward：

$$
\max_{\theta,\mathcal{M}}
\mathbb{E}_{x\sim\mathcal{D},\tau\sim\pi_\theta,\mathcal{M}}
[R(\tau)].
\tag{2}
$$
Memory system 生成 $m_t$，并用它影响当前 action：

$$
a_t\sim\pi_\theta(\cdot\mid s_t,m_t).
$$
不同 memory paradigm 的区别，可以统一写成：

$$
m_t=f_{\mathcal{M}}(s_t,\mathcal{H},m_{\lt t}).
\tag{3}
$$



### 5.3 不同调用粒度

论文用 $f_{\mathcal M}$ 的调用时机区分不同方法：

| 方式              | 什么时候调用 memory                                 | 特点                              |
| ----------------- | --------------------------------------------------- | --------------------------------- |
| Task-level memory | 只在 $t=0$ 调用一次                                 | 任务开始前检索经验，之后保持不变  |
| Step-level memory | 每个环境 step 调用                                  | 可以随环境状态更新，但粒度仍较粗  |
| Parametric memory | 经验已编译进 $\theta$                               | 推理时没有显式 memory generation  |
| MemGen            | 在 action 内部的 token / sentence boundary 动态调用 | 可以在同一轮推理中多次重构 memory |

**MemGen 的目标是设计一个更细粒度的 $f_{\mathcal M}$：由系统自己决定，在当前推理的哪个位置重新生成 memory。**



## 六、方法

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1785503982623_image.png)

### 6.1 总体架构

MemGen 包含一个冻结的 reasoner 和两个轻量可训练模块。

| 模块                                | 作用                           | 默认参数化方式                        |
| ----------------------------------- | ------------------------------ | ------------------------------------- |
| Reasoner $\pi_\theta$               | 执行正常推理和动作生成         | 冻结的基础 LLM                        |
| Memory Trigger $T_{\text{trigger}}$ | 判断 INVOKE 或 SKIP            | 独立 LoRA adapter，RL 训练            |
| Memory Weaver $W_{\text{weaver}}$   | 生成 K 个 latent memory tokens | 另一套 LoRA adapter，SFT 或 GRPO 训练 |



### 6.2 Interleaving Memory and Reasoning

Reasoner 在第 $t$ 个环境状态中生成 action：

$$
a_t=(z_{t,1},\ldots,z_{t,L_t}).
$$
生成到第 $j$ 个 token 前，reasoner 已经形成隐藏状态序列：

$$
H_{t, \lt j}=(h_{t,1},\ldots,h_{t,j-1}), \qquad h_{t,k}\in\mathbb{R}^{d_{\text{model}}}.
$$
Trigger 根据这段隐藏状态计算调用概率：

$$
p_j=\sigma(T_{\text{trigger}}(H_{t,\lt j})),
$$
并采样：

$$
d_j\sim\operatorname{Bernoulli}(p_j)
\in\{\text{INVOKE},\text{SKIP}\}.
\tag{4}
$$
如果是 SKIP，reasoner 正常生成：

$$
z_{t,j}\sim\pi_\theta(\cdot\mid s_t,z_{t,\lt j}).
$$
如果是 INVOKE，weaver 生成固定长度 K 的 latent memory：

$$
M_t=[m_{t,1},m_{t,2},\ldots,m_{t,K}]
=W_{\text{weaver}}(H_{t,\lt< j}),
\tag{5}
$$
其中：

$$
M_t\in\mathbb{R}^{K\times d_{\text{model}}}.
$$
随后 $M_t$ 被放到当前隐藏状态之前，reasoner 继续生成：

$$
z_{t,j}\sim
\pi_\theta(\cdot\mid s_t,z_{t,\lt< j},M_t).
\tag{6}
$$
这里的 prepend 不是把一段文本加进 prompt，而是把 K 个连续 latent vectors 加入 reasoner 后续计算使用的隐藏上下文。

>   这方法神了有感觉吗（



### 6.3 Memory Trigger

#### 6.3.1 Trigger 不是每个 token 都运行

如果每生成一个 token 都完整运行 trigger，成本会很高。论文采用 sentence-granularity activation：

- 预先定义 delimiter set $\mathcal D$，例如逗号、句号；
- 只有当前 token 落在 $\mathcal D$ 时，trigger 才真正做判断；
- 其他位置直接把调用概率设为 0。

形式上：

$$
p_j=
\begin{cases}
0, & z_j\notin\mathcal D,\\
T_{\text{trigger}}(H_{t,\lt j}), & z_j\in\mathcal D,
\end{cases}
\qquad
d_j\sim\operatorname{Bernoulli}(p_j).
\tag{7}
$$
这相当于让 Agent 在“一句话说到一个阶段”时决定要不要停下来回忆，而不是每吐出一个字都检查一次。



#### 6.3.2 为什么 Trigger 用 RL？

Trigger 要平衡两个目标：

- memory 调用太少：关键时刻没有获得经验支持；
- memory 调用太多：增加计算，还可能用不合适的 latent memory 干扰推理。

训练 trigger 时：

- reasoner 固定；
- 已训练好的 weaver 固定；
- trigger 在候选位置选择 INVOKE 或 SKIP；
- 整条 trajectory 得到任务 reward。

论文加入 reward-adaptive invocation penalty：

$$
\max_\phi
\mathbb{E}
\left[
R(\tau_i)
-\lambda\sum_{i,j}
\max(0,\tilde d_{i,j}-\bar p)
\right],
\tag{8}
$$
其中 \(\bar p\) 由 batch 中 reward 不低于中位数的高质量 trajectory 计算：

$$
\bar p=
\frac{1}{|\mathcal H_{\text{high}}|}
\sum_{i\in\mathcal H_{\text{high}}}
\frac{1}{|\tau_i|}
\sum_j\tilde d_{i,j}.
\tag{9}
$$
直觉是：

- 如果高 reward 轨迹通常只调用少量 memory，$\bar p$ 较小，额外调用会受到更强惩罚；
- 如果任务确实需要频繁调用，成功轨迹会给出较高 $\bar p$，惩罚相对减弱；
- trigger 因而不是追求最少调用，而是学习“对任务有用的调用频率和位置”。



### 6.4 Memory Weaver

#### 6.4.1 Weaver 保存了什么？

新增经验不会更新 reasoner，而是学习进 weaver 的参数。推理时，weaver 再根据当前 $H_{t,\lt j}$ 把相关经验重构成 $M_t$。

可以用人类类比理解：

~~~text
Weaver 参数：长期积累的经验和技能
当前 hidden states：眼下遇到的问题和思考进度
生成的 M_t：此刻被唤起、重新组织后的工作记忆
~~~

论文把 weaver 实现为附着在同一基础 LLM 上的另一套 LoRA adapter：

$$
M_t=W_{\theta'}(H_{t,\lt j}),
\qquad
M_t\in\mathbb{R}^{K\times d_{\text{model}}}.
$$
主实验中 $K\in\{2,4,8\}$，敏感性分析进一步考察到 32。



#### 6.4.2 Weaver 的统一优化目标

设 reasoner、weaver、trigger 共同产生 trajectory 的过程为：

$$
\Pi_{\theta}^{W_{\theta'},T}(\cdot\mid x).
$$
只优化 weaver 参数 $\theta'$：

$$
\max_{\theta'}
\mathbb{E}_{(x_i,\tau_i)\sim\mathcal H}
\mathbb{E}_{\tau\sim\Pi_{\theta}^{W_{\theta'},T}(\cdot\mid x_i)}
[R(x_i,\tau)].
\tag{10}
$$
梯度经过 frozen reasoner 的计算图回传到 weaver，但 reasoner 的参数 $\theta$ 不更新。



### 6.5 Weaver 如何进行 SFT？

这是整篇论文最容易误解的地方：**latent memory 没有人工标注的正确答案。**

训练数据是高质量 expert trajectories：

$$
\mathcal H=\{(x_i,\tau_i^*)\}_{i=1}^{N}.
$$
在 trigger 激活的位置：

1. Weaver 根据当前 hidden states 生成 $M_{i,t,j}$；
2. **Frozen reasoner 在 $M_{i,t,j}$ 条件下预测 expert trajectory 的下一个 token；**
3. **对 expert token 做普通 negative log-likelihood；**
4. 只更新 weaver LoRA。

损失为：

$$
\mathcal L_{\text{SFT}}(\theta')
=-\mathbb{E}_{(x_i,\tau_i^*)\sim\mathcal H}
\left[
\sum_t\sum_j
\log\pi_\theta
(z_{i,t,j}^*
\mid s_{i,t},z_{i,t,\lt j}^*,M_{i,t,j})
\right].
\tag{11}
$$
其中：

$$
M_{i,t,j}=W_{\theta'}(H_{i,t,\lt j}). \tag{12}
$$
参数更新：

$$
\theta'\leftarrow
\theta'-\eta\nabla_{\theta'}
\mathcal L_{\text{SFT}}.
\tag{13}
$$

> 不是告诉 weaver “你应该生成这几个 latent vectors”，**而是看它生成的 latent memory 能否帮助 frozen reasoner 更准确地复现成功轨迹；**只要后续 token 预测更好，梯度就会反过来塑造 latent memory。



### 6.6 Weaver 如何进行 GRPO？

对于每个任务 $x_i$，当前 MemGen policy 采样一组 trajectory：

$$
\mathcal G_i=\{\tau_{i,1},\ldots,\tau_{i,K_g}\}.
$$
这里用 $K_g$ 表示 rollout group size，避免与 latent memory 长度 K 混淆。

计算组平均 reward：

$$
\bar R(\mathcal G_i)
=\frac{1}{K_g}
\sum_{k=1}^{K_g}R(\tau_{i,k}),
\tag{14}
$$
以及相对 advantage：

$$
A(\tau_{i,k})
=R(\tau_{i,k})-\bar R(\mathcal G_i).
\tag{15}
$$
GRPO objective 让高于组平均 reward 的 trajectory 概率增加，并使用 KL 项限制偏离 reference policy：

$$
J_{\text{GRPO}}(\theta')
= \mathbb{E}
\left[
\frac{1}{K_g}\sum_k
A(\tau_{i,k})
\log\Pi_\theta^{W_{\theta'},T}(\tau_{i,k}\mid x_i)
-\beta\,\mathrm{KL}(\Pi\|\Pi_{\text{ref}})
\right].
\tag{16}
$$
仍然只有 weaver 参数 $\theta'$ 更新，reasoner 保持冻结。



### 6.7 两阶段训练顺序

论文不是一开始同时训练 trigger 和 weaver，而是：

1. **先训练 Weaver**：此时没有已学习的 trigger，在所有 punctuation boundary 或随机选择的 boundary 插入 memory，让 weaver 学会从不同推理位置生成有用 latent memory。
2. **固定 Weaver，再训练 Trigger**：让 trigger 在一个稳定的 memory generator 上学习何时调用。

这种顺序避免了两个模块同时变化导致的 credit assignment 混乱。



### 6.8 关键训练配置

| 项目                    | 设置                                              |
| ----------------------- | ------------------------------------------------- |
| SFT batch size          | 4                                                 |
| SFT learning rate       | $1\times10^{-5}$                                  |
| SFT epochs              | 2                                                 |
| GRPO rollout batch size | 8                                                 |
| GRPO train batch size   | 8                                                 |
| GRPO epochs             | 2                                                 |
| LoRA rank               | 16                                                |
| LoRA alpha              | 32                                                |
| LoRA target modules     | q_proj, v_proj                                    |
| LoRA dropout            | 0.1                                               |
| 优化                    | AdamW、cosine schedule、FlashAttention、DeepSpeed |



### 6.9 与外部 Retrieval Memory 结合

>   到此，标题里面的 self-evolving 才真正完整。

MemGen 默认从 weaver 的 parametric knowledge 生成 latent memory，但也可以接 ExpeL 等外部 memory。

Trigger 激活时，将当前已生成文本解码为 query：

$$
q_{t,j}=\operatorname{Decode}(z_{t,\lt j}). \tag{17}
$$
从外部库检索：

$$
C_t=\mathcal R(q_{t,j};\mathcal M_{\text{ext}}).
\tag{18}
$$
将检索文本编码成 embedding sequence $E_t$，再与当前 hidden states 拼接后交给 weaver：

$$
M_t=W_{\text{weaver}}([H_{t, \lt j};E_t]). \tag{19}
$$
因此外部文本不是直接追加给 reasoner，而是先经过 weaver 压缩、重构成 latent memory。



## 七、实验

>   实验非常完善，很 solid！

### 7.1 研究问题

论文设置四个 RQ：

1. MemGen 能否超过 parametric memory 和 retrieval-based memory？
2. 学到的 memory 能否跨领域泛化，原因是什么？
3. MemGen 能否支持 continual learning 并缓解 catastrophic forgetting？
4. MemGen 是否形成了类似 planning、procedural、working memory 的功能分化？



### 7.2 数据集

论文覆盖五个领域、九个数据集：

| 领域                      | 数据集                |
| ------------------------- | --------------------- |
| Web Search / Knowledge QA | TriviaQA、PopQA       |
| Embodied Action           | ALFWorld              |
| Math Reasoning            | AQuA、GSM8K、MATH     |
| Scientific Reasoning      | GPQA                  |
| Coding                    | KodCode、BigCodeBench |

主要横向结果表使用其中八个；AQuA 主要用于持续学习实验。跨领域泛化实验还额外使用 ScienceWorld 和 FEVER。



### 7.3 Baselines 与 Backbone

Baselines 分为四组：

| 类型                   | 方法                                          |
| ---------------------- | --------------------------------------------- |
| Prompt-based           | Vanilla、CoT                                  |
| Parametric memory      | SFT、GRPO、REINFORCE、REINFORCE++、Agent-FLAN |
| Retrieval-based memory | MemoryBank、ExpeL、AWM                        |
| Latent computation     | SoftCoT、Coprocessor                          |

Backbone 包括：

- Qwen2.5-1.5B；
- SmolLM3-3B；
- Qwen3-8B。

MemGen 有两个主要版本：

- MemGen SFT：weaver 使用 SFT 学习；
- MemGen GRPO：weaver 使用 GRPO 学习。



### 7.4 RQ1：跨领域主结果

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1785505021908_image.png)

#### SmolLM3-3B

| 方法            |  ALFWorld |  TriviaQA |     PopQA |   KodCode | BigCodeBench |      GPQA |     GSM8K |      MATH |
| --------------- | --------: | --------: | --------: | --------: | -----------: | --------: | --------: | --------: |
| Vanilla         |     18.96 |     10.47 |      8.23 |     37.05 |        35.96 |      9.35 |     47.63 |     16.22 |
| SFT             |     32.36 |     55.25 |     37.22 |     59.25 |        40.79 |     19.70 |     63.48 |     45.65 |
| GRPO            |     55.35 |     65.88 |     45.16 |     68.48 |        72.44 |     22.73 |     80.03 |     61.23 |
| ExpeL           |     36.18 |     46.20 |     28.16 |     51.14 |        40.22 |     15.15 |     56.23 |     38.11 |
| **MemGen SFT**  | **50.60** | **68.13** | **42.34** | **62.65** |    **42.99** | **26.75** | **70.42** | **57.44** |
| **MemGen GRPO** | **63.60** | **79.30** | **58.60** | **72.85** |    **74.24** | **25.20** | **83.47** | **63.65** |

值得注意：

- MemGen GRPO 在大多数任务上最佳；
- GPQA 上 MemGen SFT 26.75，高于 MemGen GRPO 25.20，说明 RL 版本并非每项都优于 SFT；
- ALFWorld 上 MemGen SFT 相比 Vanilla 提升 31.64 个百分点，MemGen GRPO 提升 44.64 个百分点。

#### Qwen3-8B

| 方法            |  ALFWorld |  TriviaQA |     PopQA |   KodCode | BigCodeBench |      GPQA |     GSM8K |      MATH |
| --------------- | --------: | --------: | --------: | --------: | -----------: | --------: | --------: | --------: |
| Vanilla         |     58.93 |     52.18 |     34.13 |     49.10 |        33.33 |     38.18 |     89.48 |     79.82 |
| SFT             |     83.59 |     74.55 |     51.12 |     64.75 |        41.33 |     40.33 |     90.76 |     81.35 |
| GRPO            |     85.60 |     76.15 |     58.90 |     73.35 |        70.24 |     39.54 |     92.30 |     83.54 |
| ExpeL           |     78.97 |     65.54 |     40.33 |     57.20 |        34.23 |     35.15 |     86.20 |     77.40 |
| **MemGen SFT**  | **85.82** | **77.22** | **54.65** | **66.15** |        40.35 | **43.23** |     91.25 |     83.30 |
| **MemGen GRPO** | **90.60** | **80.65** | **62.30** | **76.16** |    **75.56** |     40.24 | **93.20** | **88.24** |

论文据此主张 MemGen 在 embodied action、knowledge QA、coding、science 和 math 上都有收益，而不是只适合一个领域。



### 7.5 小模型结果

Qwen2.5-1.5B 上：

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1785505178541_image.png)

这里同样可以看到：GPQA 上 MemGen SFT 略高于 MemGen GRPO，说明不同优化方式会形成不同 memory behavior。



### 7.6 RQ2：跨领域泛化

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1785505222729_image.png)

论文分别在 ALFWorld、TriviaQA、GSM8K、KodCode 上训练，再测试其他领域，并额外加入 ScienceWorld、FEVER。

主要观察包括：

- SFT 和 MemoryBank 往往主要改善训练域；
- 某些 baseline 在训练域外会下降，例如论文报告 FEVER 最多下降 16.2%；
- MemGen 在 KodCode 上训练后，KodCode 从 24.55 提升到 58.16，同时 MATH 从 36.6 提升到 54.2；
- 在 GSM8K 上训练后，对 GPQA 和 KodCode 仍有正迁移。

**论文将这一现象解释为：weaver 学到的不只是任务答案，还包含可迁移的 reasoning pattern。**



### 7.7 Trigger 如何缓解 domain conflict？

论文在 GSM8K 上训练 MemGen，然后分别观察 GSM8K、GPQA、KodCode 推理时的 memory invocation：

- GSM8K 平均调用最多，性能提升也最大；
- GPQA 调用中等；
- KodCode 调用最少，性能提升也较小。

**作者据此认为，trigger 会根据当前任务与已学 memory 的匹配程度调节调用频率：在陌生领域少调用，降低错误 memory 干扰。**

**这个证据说明 invocation frequency 与任务收益相关，但它仍是相关性分析，不能单独证明 trigger 已显式学会 domain identity。**



### 7.8 RQ3：Continual Learning

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1785505292214_image.png)

论文使用 Qwen2.5-1.5B，按以下顺序训练：

~~~text
AQuA -> GPQA -> GSM8K -> KodCode
~~~

每个阶段训练后，在四个任务上统一评测。

最后训练到 KodCode 后：

| 方法           |      AQuA |      GPQA |     GSM8K |   KodCode |
| -------------- | --------: | --------: | --------: | --------: |
| SFT            |     28.61 |      2.53 |     24.14 |     54.10 |
| ExpeL          |     27.14 |      6.23 |     31.44 |     48.35 |
| **MemGen SFT** | **40.34** | **20.09** | **53.72** | **52.95** |

这表明：

- **SFT 对最新任务 KodCode 提升最高，但旧任务损失明显；**
- **MemGen 在最新任务略低于 SFT，却保留了更均衡的旧任务能力；**
- 冻结 reasoner、把新经验隔离到 weaver 中，确实有助于降低对基础能力的直接覆盖。

但应准确使用“缓解 catastrophic forgetting”，而不是“彻底消除”：weaver 自身仍是连续训练的参数模块，也可能发生遗忘。



### 7.9 RQ4：Latent Memory 是否形成层级？

#### Latent memory 是否可读？

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1785505407638_image.png)

论文对每条 K-token memory sequence 求平均向量，再用 t-SNE 可视化，并在高维空间做 K-means 聚类。

不同任务的 memory distribution 有分离趋势，相关领域更接近，例如：

- KodCode 与 BigCodeBench；
- GSM8K 与 MATH。

作者尝试把 latent vectors 强制映射为 vocabulary token，得到的文本大多不可读，但部分 cluster 有重复形式，例如：

- TriviaQA 某 cluster 常出现结尾 SOC；
- GSM8K 某 cluster 常出现 _check 或 _pick。

因此论文称其为 machine-native and human-unreadable memory。



#### 功能干预实验

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1785505420211_image.png)

论文在 TriviaQA 上：

1. 对 latent memory mean embedding 做 K-means，预设 \(N=4\) 个 cluster；
2. 人工把失败轨迹标注为八种 failure mode；
3. 推理时过滤靠近某个 cluster centroid 的 memory sequence；
4. 观察不同 failure mode 是否增加。

作者将 cluster 功能解释为：

| Memory 类型       | 干预证据                                                     |
| ----------------- | ------------------------------------------------------------ |
| Planning Memory   | 移除 Cluster 2 后，planning 与 compositional reasoning failure 增加 |
| Procedural Memory | 移除 Cluster 3 后，tool response、tool parsing、answer formatting error 增加 |
| Working Memory    | 移除 Cluster 1/4 后，task misunderstanding、think-act inconsistency 增加 |

论文也承认 cluster 不是完全独立的，例如移除 working-memory-related cluster 也会影响 planning。



### 7.10 Trigger 消融

Qwen2.5-1.5B 上不同 invocation strategy：

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1785505500315_image.png)

结论是：

- 在语义边界插 memory 优于任意 token 位置随机插入；
- 每个 delimiter 都插入仍不如 learned trigger；
- memory 插入并非越多越好，错误位置会产生 interference。



### 7.11 Weaver 容量消融

论文比较 LoRA weaver 和 full-parameter SFT weaver：

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1785505526210_image.png)

Full SFT 更强，说明 weaver 的参数容量会限制 memory 能力；LoRA 的价值主要是参数效率和隔离性，而不是达到绝对最佳性能。

论文还观察到 latent memory length 从 2 增加到 32 时性能总体提高，说明 K 是一个 memory capacity 参数。但主实验主要使用固定 K，尚未探索根据任务难度自适应生成不同长度。



### 7.12 效率分析

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1785505559856_image.png)

以部分结果为例：

| Backbone / Task        | Vanilla 时间与准确率 | SFT 时间与准确率 | MemGen SFT 时间与准确率 |
| ---------------------- | -------------------- | ---------------- | ----------------------- |
| Qwen2.5-1.5B / KodCode | 11.96s / 24.55       | 2.01s / 55.83    | 2.94s / 58.16           |
| SmolLM3-3B / ALFWorld  | 34.82s / 18.96       | 12.88s / 32.36   | 14.69s / 50.60          |
| Qwen3-8B / ALFWorld    | 55.42s / 58.93       | 19.76s / 83.59   | 20.08s / 85.82          |

MemGen 相比同样经过 SFT 的模型通常多一点 latency，因为要运行 trigger 和 weaver；但相比 Vanilla，完成任务所需轨迹更短，所以总时间仍可能更低。

因此，“MemGen 比 Vanilla 快”不能简单解释为 latent insertion 本身加速了单 token inference。更准确的解释是：

> Memory insertion 有额外计算，但更有效的推理可能减少无效 token、工具调用和错误尝试，使端到端任务时间下降。



### 7.13 与 ExpeL 结合

SmolLM3-3B 上：

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1785505592714_image.png)

这个实验说明 MemGen 与 external memory manager 不是替代关系：

- ExpeL 负责提供可检索文本经验；
- Weaver 负责把检索结果和当前推理状态重构成 compact latent memory；
- Reasoner 使用重构后的 latent memory 继续推理。



## 八、结论

### 8.1 论文结论

论文提出 MemGen，通过：

- RL-trained memory trigger；
- generative memory weaver；
- reasoning-time latent memory insertion；
- frozen core reasoner；

让 Agent 在推理过程中动态生成并调用 memory。

实验显示 MemGen 在多领域任务上取得性能提升，具有一定跨域迁移和 continual learning 能力，并在 post-hoc cluster intervention 中呈现 planning、procedural、working memory 的功能分化。



### 8.2 novelty

MemGen 最有价值的地方不是“又加了一套 LoRA”，而是组合了三个设计：

1. **何时记忆是可学习的**：trigger 在推理中动态选择 invocation position。
2. **记忆内容是生成的**：weaver 根据当前 cognitive state 重构 memory，而不是只返回原始经验。
3. **记忆直接进入 latent computation**：生成结果以 K 个 hidden vectors 影响 frozen reasoner，而不是转回自然语言 prompt。

这三个部分一起构成了 reasoning-memory interleaving。



### 8.3 一些概念边界

#### 它并不是完全非参数化的 Memory

论文把 $M_t$ 称为 latent memory 是合理的，但长期经验实际上保存在 weaver LoRA 参数中。

所以：

~~~text
长期存储介质：Parametric
运行时 memory representation：Latent
调用方式：Generative
~~~

如果只说它“摆脱了 parametric memory”，会忽略 weaver 本身就是一个参数化经验库。它真正避免的是修改 core reasoner，而不是避免所有参数更新。



#### 它不是用户可管理的长期记忆库

MemGen 默认没有：

- memory item ID；
- 显式内容查看；
- 单条事实更新；
- 删除和遗忘接口；
- 来源追溯；
- 时间有效期；
- 冲突 belief 管理。

因此它适合保存“如何做任务”的隐式经验，不适合单独承担“用户现在住在哪里”这种需要可编辑、可审计的事实记忆。



#### Human-like hierarchy 是后验解释

论文的 functional study 有实际干预，而不只是画一张 t-SNE，这一点值得肯定。但“形成了人类式记忆层级”的证据仍有限：

- K-means cluster 数量 $N=4$ 是预设的；
- failure mode 是人工定义和标注的；
- cluster 到 memory type 的对应关系是事后命名；
- cluster 功能并不相互独立；
- 实验主要集中在 TriviaQA。

因此更稳妥的表述是：

> 不同 latent memory cluster 对不同 Agent failure mode 呈现功能差异，作者将这种差异解释为 planning、procedural 和 working memory。



### 8.4 主要局限

1. **不可解释和不可审计**：latent memory 强制解码后仍难以阅读，错误 memory 很难定位和修正。
2. **Weaver 自身仍会遗忘**：冻结 reasoner 缓解了基础能力覆盖，但连续训练同一个 weaver 仍可能产生 adapter-level forgetting。
3. **需要训练数据和任务 reward**：新经验不会自动进入 memory，仍需要 SFT trajectory 或可计算的 RL reward。
4. **两阶段训练增加复杂度**：先训练 weaver、再训练 trigger，训练和部署都比普通 LoRA 多一个控制模块。
5. **固定 memory length**：主实验使用 K 为 2、4、8，尚未学习“该生成多长的 memory”。
6. **Trigger 的候选位置受限**：只在 delimiter 边界判断提高了效率，但可能错过句中真正关键的推理转折。
7. **效率归因不够彻底**：端到端时间下降部分来自输出轨迹变短，论文没有完全拆分 trigger、weaver 和 KV/cache 操作的独立开销。
8. **持续学习规模有限**：只在四个任务序列上验证，尚不足以证明长周期、多阶段 self-evolution。



### 8.5 启发。？

**1、External Memory Manager + Latent Weaver**

让 Mem0、Graph Memory 或 episode database 负责：

- 长期存储；
- 版本管理；
- 更新删除；
- 证据追溯。

让 MemGen weaver 负责：

- 根据当前推理状态选择外部证据；
- 压缩冲突和冗余；
- 生成 task-specific latent working memory。

这样可以同时获得显式 memory 的可管理性和 latent memory 的高密度计算能力。



**2、Adaptive Memory Length**

把固定 K 改为：

~~~text
简单查询 -> 0 或 2 个 latent tokens
一般推理 -> 4 到 8 个
复杂规划 -> 更长 latent sequence
~~~

训练目标同时考虑任务 reward、latency 和 memory length。



**3、可解释 Latent Memory**

可以增加：

- latent-to-text probe；
- concept bottleneck；
- memory attribution；
- counterfactual intervention；
- source trajectory alignment。

目标不是强迫 latent memory 完全变成文本，而是回答“这次 memory 来自哪些经验、影响了哪些 action”。



**4、Belief-aware Trigger**

目前 trigger 主要学习何时调用。进一步可以让它区分：

- 缺少事实；
- 缺少规划；
- 遇到冲突；
- 工具失败；
- 对答案不确定；
- 需要检查格式。

然后调用不同类型的 weaver 或外部 memory source。





















