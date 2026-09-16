---
title: "论文精读 | Delta Mem"
description: "LoRA也能做 Agent Memory？"
date: 2026-07-03T11:07:41+08:00
lastmod: 2026-07-03T11:07:41+08:00
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
cover: https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783090974795_image.png
banner: https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783091139571_image.png
---

<!--more-->

## 零、写在前面

如果按照 [Memory in the Age of AI Agents](https://arxiv.org/abs/2512.13564) 这篇工作的分类方式，δ-Mem 从功能上划分应该属于 **Working/Experiential Memory**，从形式上划分应该属于 **Latent Memory**。作者没有像去年大部分工作那样，加一个外挂的 Memory Manager 然后去做记忆的curd，而是像 Lora 那样，**用一个低维矩阵维护在线记忆**，当新信息来了，它只把“旧记忆预测错的那部分残差”写进去。

## 一、标题

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783090974795_image.png)

> δ-mem: Efficient Online Memory for Large Language Models

关键词：**δ-mem**、**Efficient Online Memory**、**Large Language Models**。

从题目中就能看出来本文做的是一种高效的在线维护Memory的方案，delta 也大概揭示了记忆的维护方式。

## 二、摘要

**δ-mem 用一个极小的在线关联记忆矩阵压缩历史，并把读出的记忆信号转成 attention 修正，从而让冻结 LLM 在不扩展上下文的情况下利用过去信息。**

并且直接提到了本文的三个关键机制：

- **Online State of Associative Memory（在线关联记忆状态，简称 OSAM）**：一个会随输入不断更新的小矩阵。
  
- **Delta-rule Learning（delta 规则学习）**：用“预测误差”来更新记忆。
  
- **Low-rank Attention Corrections（低秩注意力修正）**：把记忆读出的信号变成对 attention 的轻量修正。

论文摘要说，大模型越来越需要在长期助手和 Agent 系统里积累并复用历史信息。单纯扩大 context window 很贵，而且不保证模型真的能用好长上下文。

于是作者提出 δ-mem：

```text
历史信息
  -> 压缩进固定大小的 online memory state
  -> 当前输入读取这个 state
  -> 生成 attention 的低秩修正
  -> 影响模型生成
  -> 新信息再写回 state
```

这和常见的记忆方案差别很大：

- RAG / Mem0：记忆主要以文字形式回到 prompt。
  
- MemGPT：管理上下文窗口，把信息换入换出。
  
- δ-mem：记忆不作为文字进入 prompt，而是作为内部连续信号修正 attention。

论文给出的核心结果包括：

- 使用 **8 × 8 online memory state**，δ-mem 平均分达到 frozen backbone 的 **1.10×**。
  
- 相比最强非 δ-mem 记忆 baseline，达到 **1.15×**。
  
- 在 memory-heavy benchmark 上收益更大：
  
    - **MemoryAgentBench** 达到 **1.31×**。
      
    - **LoCoMo** 达到 **1.20×**。
      
    - **TTL subtask** 从 **26.14** 提升到 **50.50**。
    
- 在 Qwen3-4B-Instruct 上，整体平均分从 **46.79%** 提升到最高 **51.66%**。
  

值得一提的是，这篇论文不是说“8 × 8 矩阵能完整保存所有历史细节”。更准确的理解是：

> 这个很小的状态能保存一部分对后续推理有用的历史信号，并通过 attention 修正把这些信号用起来。

它不是人类式完整回忆，也不是可解释文本记忆，而是一种压缩的、连续的、内部耦合式记忆。

## 三、引言

### 3.1 论文为什么反对“只加长上下文”？

引言从一个现实问题出发：LLM 正在进入长期助手和 Agent 系统。这样的系统不能只回答孤立 prompt，而要在长时间互动中积累、更新和复用历史信息。

最直觉的办法是：把历史都放进 context window。

但论文指出，这只是把记忆问题变成了长上下文处理问题，并没有根本解决记忆问题。

原因主要有两个。

第一，**计算代价高**。标准 attention 对上下文长度通常有很高的计算成本。历史越长，开销越大。

第二，**长上下文不等于有效使用上下文**。模型可能出现 context degradation 或 context rot，也就是上下文越长，模型越容易忽略、混淆或错误利用远处信息。


### 3.2 memory state 与 memory steering

 #### 3.2.1 Memory State：记忆存在哪里？

**Memory State（记忆状态）** 指历史信息以什么形式保存。

比如：

- 文本摘要、文本事实、RAG chunk。
  
- 外部向量库、外部 latent memory。
  
- adapter / LoRA / prefix 等参数。
  
- δ-mem 的在线矩阵状态。
  

#### 3.2.2 Memory Steering：记忆如何影响模型？

**Memory Steering（记忆引导）** 指保存下来的记忆怎样影响当前推理。

比如：

- RAG 把文本拼回 prompt，让模型读。
  
- 外部模块把检索结果编码后融合回来。
  
- 参数记忆通过固定权重改变模型行为。
  
- δ-mem 通过 attention correction 直接影响当前 forward computation。
  

### 3.3 现有方法的不足

论文把已有方法分成三类，并指出各自问题。

**Textual Memory Mechanisms（文本记忆机制）**：把记忆存成文本，再通过 prompt 或 retrieval 放回模型。

代表包括 RAG、MemoryBank、Mem0、MemGPT 类系统。

优点是灵活、可读、可编辑。缺点是：

- 受 context window 限制。
  
- 检索可能有噪声。
  
- 摘要和压缩会丢信息。
  
- 记忆最终还是要消耗 prompt token。
  

**Outside-Channel Memory Mechanisms（外部通道记忆机制）**：把记忆放在模型外部的 latent module 或 memory bank 中，再通过额外通道读回来。

优点是可以不完全依赖文本。缺点是：

- 检索和融合有额外开销。
  
- 外部表示和当前 backbone 可能不对齐。
  
- 系统集成复杂。
  


**Parametric Memory Mechanisms（参数记忆机制）**：把记忆写进 prefix、adapter、LoRA 或模型编辑参数里。

优点是推理时高效，并且和 frozen backbone 兼容。缺点是：

- 多数参数记忆是静态的。
  
- 很难在线适应不断变化的信息。
  
- 更像“改模型习惯”，不太像“随互动演化的记忆状态”。
  

### 3.4 δ-mem 的定位

δ-mem 想结合几类方法的优点：

- 像 parametric memory 一样轻量，不需要长 prompt。
  
- 像 online memory 一样可以随输入动态更新。
  
- 像内部机制一样直接影响 attention，而不是靠外部文本再喂回来。
  

## 四、相关工作

### 4.1 Textual Memory Mechanisms：文本记忆

**Textual Memory Mechanisms（文本记忆机制）** 把历史保存为文本条目、摘要、文档 chunk、记忆日志，再在需要时塞回 prompt。

典型例子：

- **RAG（Retrieval-Augmented Generation，检索增强生成）**：从外部文档或历史记录中检索相关片段，拼到输入里。
  
- **Generative Agents**：把观察写进 memory stream，再检索和反思。
  
- **MemGPT**：像操作系统管理内存一样，在有限 context 和外部长期记忆之间换入换出。
  
- **MemoryBank**：维护连续交互历史和用户记忆。
  
- **Mem0**：从对话中抽取关键事实，进行新增、更新、删除和检索。
  

这类方法很直观，也最接近人类写笔记。

优点：

- 可读性强。
  
- 便于人工检查和修改。
  
- 容易与数据库、向量库、知识图谱结合。
  

缺点：

- 最终还是要占用 token。
  
- 检索错误会把无关信息塞给模型。
  
- 压缩摘要会丢细节。
  
- 记忆是否能被模型正确利用，仍取决于 prompt 和模型长上下文能力。
  

δ-mem 的区别：

> 它不把记忆作为文本重新输入，而是把历史压进连续状态，再把状态读出为 attention 修正信号。

### 4.2 Outside-Channel Memory Mechanisms：外部通道记忆

**Outside-Channel Memory Mechanisms（外部通道记忆机制）** 保存的不是普通文本，而是模型内部表示、latent vector、外部 memory bank 等。

代表工作包括：

- **Memorizing Transformers**：存储过去内部表示，并用 kNN 检索。
  
- **LongMem**：用 frozen backbone 编码记忆，再通过 side network 读取外部 memory bank。
  
- **MLP Memory**：用外部 MLP 模块保存和检索记忆信息。
  

这类方法比文本记忆更“神经网络内部化”，可能减少自然语言摘要带来的信息损失。

但问题是：

- 外部检索仍然有开销。
  
- 外部 memory 和当前 backbone 表示可能不匹配。
  
- 融合模块设计复杂。
  
- 记忆并不一定直接参与 attention 的核心计算。
  

δ-mem 的区别：

> 它没有把 memory 当成一个额外资料源，而是让 memory state 的 readout 直接产生 attention correction，参与当前 forward pass。

### 4.3 Parametric Memory Mechanisms：参数记忆

**Parametric Memory Mechanisms（参数记忆机制）** 把信息编码到额外参数或局部权重修改里。

代表包括：

- **Prefix-Tuning**：学习连续 virtual tokens。
  
- **LoRA**：用低秩矩阵改变模型层的行为。
  
- **ROME / MEMIT**：通过模型编辑写入事实关联。
  
- **Context2LoRA / MemGen**：把上下文或记忆编码进额外参数。
  

这类方法的优点是推理时高效，也适合冻结大模型主体。

缺点是：

- 记忆通常训练后固定。
  
- 在线更新不自然。
  
- 可能改变模型整体行为，而不是只在当前历史条件下生效。
  

δ-mem 和 LoRA 有点像，都用低秩接口影响模型；但关键区别是：

> LoRA 的低秩更新是静态的；δ-mem 的低秩修正来自动态 memory state，同一组参数在不同历史下会产生不同修正。

这就是 δ-mem 的研究定位：**动态的状态记忆 + 轻量参数接口 + attention 内部耦合**。

## 五、方法

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783098667548_image.png)


δ-Mem 的核心想法其实就是把历史信息压缩进一个很小的在线记忆状态 `S`，再用这个状态去修正 Transformer 的 attention 计算。

这和 RAG、Mem0 这类外部文本记忆方法很不一样。RAG 或 Mem0 通常会把历史内容保存成文本、向量或图结构，等需要时再检索出来拼进 prompt。δ-Mem 则更“内部化”：它不把记忆作为文本喂回模型，而是让记忆以连续矩阵的形式参与模型前向计算。

原文把这个矩阵称为 **Online State of Associative Memory，在线关联记忆状态**。

### 5.1 在线记忆状态：用矩阵 S 存 key-value 关联

δ-Mem 维护一个矩阵：
$$
S \in \mathbb{R}^{r \times r} 
$$

论文主实验里 `r = 8`，所以这个记忆状态只有：$8 \times 8$

也就是 64 个数。

**它不是用来逐字保存历史内容的，而是用来保存一种压缩后的“关联记忆”。**

在每个时间步，模型会从当前 hidden state $x_t$ 中投影出三个向量：
$$
q_t^m = \text{L2Norm}(\tanh(W_q^m x_t)) 
$$
$$
 k_t^m = \text{L2Norm}(\tanh(W_k^m x_t))
$$
$$
v_t^m = W_v^m x_t
$$

这三个向量分别承担不同角色：

- $q_t^m$：用来读取记忆，即“我现在想查什么”；
- $k_t^m$：用来写入记忆的 key，即“这条信息的索引”；
- $v_t^m$：用来写入记忆的 value，即“这条信息的内容”。

这里的 \(q, k, v\) 和 Transformer attention 里的 query、key、value 很像，但它们不是标准 attention 的 QKV，而是 δ-Mem 自己投影出来的 **memory QKV**。

### 5.2 读取记忆：当前输入从旧状态里取出历史信号

在写入当前信息之前，δ-Mem 会先从旧记忆状态 $S_{t-1}$ 中读取信息：

$$
r_t = S_{t-1} q_t^m 
$$

这里的 $r_t$ 就是读出来的 memory signal。

注意，这个 $r_t$ 不是一段文本，也不是检索出的 chunk，而是一个连续向量。它代表“当前输入根据过去历史激活出的记忆信号”。

可以类比成人的联想：

> 看到群聊里面开始发vivo50的笑话，便联想到 “今天是周四”、“肯德基疯狂星期四”。我们不是在逐字翻笔记，而是某个线索激活了过去的印象。

δ-Mem 也是类似。当前输入通过 $q_t^m$ 查询记忆矩阵 $S_{t-1}$，得到一个压缩的历史信号 $r_t$。

### 5.3 用记忆修正 attention：低秩 correction

读出 $r_t$ 之后，δ-Mem 并不会把它直接转成文本，而是把它变成对 attention 的修正项。

论文主方法主要修正两个位置：
$$
\Delta q_t = W_q^\Delta r_t
$$
$$
\Delta o_t = W_o^\Delta r_t
$$

也就是：

- 一个修正 attention 的 query；
- 一个修正 attention 的 output。

原始 Transformer 中，当前 token 的 query 是：
$$
q_t^0 = W_Q x_t
$$

δ-Mem 会把记忆修正加上去：
$$
\tilde{q}_t = q_t^0 + \frac{\alpha}{r}\Delta q_t
$$

然后用修正后的 query 做 attention：
$$
 a_t = \text{Attn}(\tilde{q}_t, K_{\leq t}, V_{\leq t}) 
$$

最后，在 attention output 上再加一个记忆修正：

$$
\tilde{y}_t = a_t + \frac{\alpha}{r}\Delta o_t
$$

这也是 δ-Mem 的核心创新点之一：它不是简单加一个 adapter，而是让 adapter 的行为随历史记忆变化。


### 5.4 写入记忆：用 delta-rule 更新状态 S

读完旧记忆、完成 attention 修正之后，δ-Mem 会把当前信息写入记忆状态。

它希望记忆矩阵 S 能学到这样的关联：
$$
S k_t^m \approx v_t^m 
$$


也就是说，之后如果再遇到类似的 key，矩阵 S 应该能回忆出对应的 value。

论文把这个过程解释成一个在线回归问题：

$$
\mathcal{L}_t(S) = \frac{1}{2} \|S k_t - v_t\|^2
$$


这里的意思是：当前状态 $S$ 看到 key $k_t$ 后，预测出的 value 是 $S k_t$，目标 value 是 $v_t$。两者越接近，说明记忆越准确。

如果用 SGD 更新这个 loss，可以得到：
$$
S_t = S_{t-1} + \beta_t (v_t - S_{t-1}k_t)k_t^\top
$$

这个公式看起来复杂，但直觉很简单：

$$
v_t - S_{t-1}k_t 
$$

就是旧记忆的预测误差。

如果旧状态已经能根据 $k_t$ 预测出 $v_t$，说明这条信息已经记住了，就不用大幅更新。反过来，如果预测错了，就把这个“差值”写入状态。

所以 δ-Mem 不是无脑累加新信息，而是只写入旧记忆没掌握好的部分。

这就是 delta-rule 的含义：

> 只更新误差，只写入残差。


### 5.5 遗忘门：不是所有旧记忆都应该永久保留

为了避免记忆状态无限累积噪声，δ-Mem 又引入了一个遗忘门：

$$ S_t = \lambda_t S_{t-1} + \beta_t (v_t - S_{t-1}k_t)k_t^\top $$

其中：

- $\lambda_t$：控制旧记忆保留多少；
- $\beta_t$：控制新信息写入多少。

论文中 $\beta_t$ 由当前 hidden state 计算出来：

$$ \beta_t = \sigma(W_\beta x_t + b) $$
并令：

$$ \lambda_t = 1 - \beta_t $$

这意味着，如果当前信息很重要，写入强度 $\beta_t$ 可以更大；如果当前信息不重要，旧状态就保留得更多。

### 5.6 三种写入粒度：TSW、SSW、MSW

论文还研究了记忆写入的粒度。因为在真实对话和 Agent 轨迹中，每个 token 都写入不一定最合理。一个 token 太细，容易把格式符号、停用词、重复表达也写进记忆。

因此作者设计了三种写入方式。

#### Token-State Write，TSW

**TSW** 是最细粒度的写入方式，每个 token 都更新一次记忆状态：

$$S_t = \text{Update}(S_{t-1}, x_t) $$

它的优点是信息保留细，适合需要捕捉局部变化的任务。缺点是容易被噪声污染。

#### Sequence-State Write，SSW

**SSW** 先把一个 message 或 segment 内的 hidden states 平均：

$$\bar{x}^{(j)} = \frac{1}{|M^{(j)}|}\sum_{t \in M^{(j)}} x_t $$

然后用这个 segment 表示更新一次状态：

$$ S^{(j)} = \text{Update}(S^{(j-1)}, \bar{x}^{(j)}) $$

它相当于“听完一整句话或一整段后再做总结”，比逐 token 写入更平滑，也更抗噪声。

#### Multi-State Write，MSW

**MSW** 不是只维护一个状态矩阵，而是维护多个并行子状态：

$$ S_t = \{S_t^{(1)}, \cdots, S_t^{(N)}\} $$

每个子状态都可以积累不同类型的信息，最后把它们的 readout 拼接起来：

$$ r_t = \text{Concat}(r_t^{(1)}, \cdots, r_t^{(N)}) $$

直觉上，MSW 像是给模型准备多个记忆槽：

- 一个槽记事实；
- 一个槽记偏好；
- 一个槽记任务进度；
- 一个槽记局部事件。

这样可以减少所有信息都挤进一个矩阵造成的互相干扰。


### 5.7 训练目标：SFT loss 训练的是“怎么记、怎么用”

最后，δ-Mem 使用标准的 SFT loss 来训练整个记忆模块。

训练时，历史 context 会先写入 online state，得到 $S_C$。之后模型回答问题时，不再显式重放完整历史，而是让 frozen backbone 在 $S_C$ 的 steering 下生成答案。

训练目标是自回归交叉熵：

$$ \mathcal{L}_{SFT} = -\sum_{j=1}^{|Y|} \log p_{\phi,\theta}(y_j \mid Q, y_{<j}, S_C) $$

这里：

- $\phi$：冻结的 backbone 参数；
- $\theta$：δ-Mem 的可训练参数；
- $Q$：当前问题；
- $Y$：目标回答；
- $S_C$：由历史 context 写入得到的记忆状态。

这说明，delta-rule 负责 state 的在线更新，而 SFT loss 负责训练 δ-Mem 的读写投影、门控和 attention correction 参数。

## 六、实验

### 6.1 实验任务与 benchmark

论文同时评估一般能力和记忆密集任务。

一般能力包括：

- **IFEval（Instruction Following Evaluation，指令遵循评测）**：看模型是否按要求执行指令。
  
- **GPQA-Diamond**：高难度知识推理问答。
  
- **HotpotQA**：多跳问答，需要跨证据推理。
  

记忆密集任务包括：

- **LoCoMo**：长期对话记忆评测，考察 multi-hop、temporal、open-domain、single-hop 等问题。
  
- **MemoryAgentBench**：评估 Agent 在长期交互中对记忆的保留、检索和使用能力。
  

论文还排除了 LoCoMo 的 adversarial question category，与 Mem0 论文采用的处理方式一致。

### 6.2 Baseline 设置

所有主要 baseline 都基于同一个 **Qwen3-4B-Instruct** backbone，便于公平比较。

论文比较了三类记忆机制。

#### Textual Memory Baselines

- **BM25 RAG**：检索相关历史文本并拼回上下文。
  
- **LLMLingua-2**：把长历史压缩成更短 prompt。
  
- **MemoryBank**：维护连续交互历史的文本记忆。
  

#### Parametric Memory Baselines

- **Context2LoRA**：把上下文相关适配编码进 LoRA。
  
- **MemGen**：生成式 latent memory 方法。
  

#### Outside-channel Memory Baseline

- **MLP Memory**：用外部 MLP memory 模块检索和融合信息。
  

这组 baseline 覆盖了当前常见路线：文本记忆、参数记忆、外部模块记忆。δ-mem 的目标是证明：更小、更在线、更贴近 attention 的 state memory 可以更稳定。

### 6.3 Table 1：Qwen3-4B-Instruct 上的主结果

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783103055561_image.png)


原始 Qwen3-4B-Instruct：

- IFEval：81.89
  
- HotpotQA EM / F1：42.35 / 56.00
  
- GPQA-D：39.39
  
- MemoryAgentBench Avg：29.54
  
- LoCoMo Avg：40.79
  
- 总 Avg：46.79
  

δ-mem 三个变体：

- **δ-Mem (SSW)**：总 Avg = **51.44**
  
- **δ-Mem (TSW)**：总 Avg = **51.66**，最高整体平均分
  
- **δ-Mem (MSW)**：总 Avg = **50.74**
  

对比最强非 δ-mem baseline：

- Context2LoRA 总 Avg = **44.90**
  
- δ-Mem (TSW) 总 Avg = **51.66**
  

论文强调，TSW 比 frozen backbone 提升 **+4.87 points**，比 Context2LoRA 提升 **+6.76 points**。

### 6.4 MemoryAgentBench：长期 Agent 记忆收益明显

MemoryAgentBench 上，原始 Qwen3-4B-Instruct Avg 是 **29.54**。

δ-Mem (MSW) 达到 **38.85**，是三种写入方式里最高的。

其中 TTL subtask 从原始 backbone 的 **26.14** 提升到 δ-Mem (SSW) 的 **50.50**。

这说明：

> 对长期交互任务，单纯依赖 frozen backbone 的当前上下文不够。在线状态能把部分历史信号保存下来，并在后续推理中发挥作用。

为什么 MSW 在 MemoryAgentBench 上强？

合理解释是：Agent 记忆里有多种信息混在一起，包括事实、任务目标、用户偏好、阶段进度。多个 state 能减少互相覆盖。


### 6.5 LoCoMo：MSW 在长期对话记忆上最好

LoCoMo 上，原始 Qwen3-4B-Instruct Avg 是 **40.79**。

δ-Mem (MSW) 达到 **49.12**，是 LoCoMo 上最好的 δ-mem 变体，并且在多个子项上领先：

- Multi：42.57
  
- Temporal：39.31
  
- Open：18.12
  
- Single：58.59
  

这与 Mem0 论文形成一个有趣对照：

- Mem0 / Mem0g 是外部文本和图记忆。
  
- δ-mem 是内部在线状态记忆。
  
- 二者都在 LoCoMo 上验证，但机制完全不同。
  

对研究者来说，这说明 LoCoMo 已经成为 Agent Memory 方向很重要的公共评测场，但不同方法在它上面的强项可能不同。

### 6.6 HotpotQA：TSW 提升多跳问答

HotpotQA 上，原始 Qwen3-4B-Instruct：

- EM = 42.35
  
- F1 = 56.00
  

δ-Mem (TSW)：

- EM = **49.41**
  
- F1 = **63.66**
  

这说明 token-level 写入对知识密集、多跳问答有帮助。可能原因是 HotpotQA 的证据粒度比较细，逐 token 或细粒度更新能保存更多局部线索。

### 6.7 Table 2：跨 backbone 的一致性

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783103172192_image.png)

论文还在三个不同 backbone 上测试 δ-mem：

- Qwen3-4B-Instruct
  
- Qwen3-8B
  
- SmolLM3-3B
  

结果显示三者都有提升：

- Qwen3-4B-Instruct：**46.79 → 51.66**
  
- Qwen3-8B：**47.20 → 50.86**
  
- SmolLM3-3B：**26.08 → 36.96**
  

这个结果很有意思：

- 大模型 Qwen3-8B 提升较稳但不夸张，SSW 最好。
  
- 小模型 SmolLM3-3B 提升很大，MSW 最好。
  

可能解释：

> 大模型本身推理能力较强，只需要平滑的片段级记忆辅助；小模型容量更有限，更需要多 state 分担不同信息，避免互相干扰。

### 6.8 Context Recovery：没有显式历史还能恢复多少？

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783103230775_image.png)

论文做了一个很关键的实验：把原始历史上下文移除，只注入压缩后的 memory state，看模型还能不能恢复有用信息。

HotpotQA：

- Overall EM：**0.08% → 6.48%**
  
- Overall F1：**8.27% → 15.20%**
  
- Bridge EM：**0.08% → 3.97%**
  
- Bridge F1：**6.25% → 11.05%**
  

LoCoMo：

- Overall average：**3.49% → 8.05%**
  

这些数值绝对值不高，但意义很重要。

它说明：

> 8 × 8 state 不是完整记忆，但确实携带了一部分可复用历史信号。即使原文不在上下文里，模型也能比完全 no-context 更好。

这也是论文最像“真正记忆”的证据之一。

### 6.9 Head Ablation：记忆应该注入哪里？

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783103417877_image.png)

论文测试了把 δ-mem correction 注入 attention 的不同分支：

- q：query
  
- k：key
  
- v：value
  
- o：output
  
- 以及它们的组合
  

结果：

- 单分支里，output 最好，Avg = **47.05**。
  
- `qo` 表现很强，Avg = **47.97**。
  
- `qkvo` 最高，Avg = **48.05**。
  

但论文默认使用 `qo`，因为 `qkvo` 虽然略高，但额外开销不太划算。

这给后续研究一个启发：

> 记忆不一定要注入所有位置。找到“少量但有效”的控制点，比无脑加模块更重要。

### 6.10 Insertion Depth Ablation：插入哪些层？

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783103416928_image.png)

论文还研究了 δ-mem 插入层数：

- Front 12
  
- Middle 12
  
- Back 12
  
- All Layers
  

结果：

- All Layers 总 Avg = **47.97**，最好。
  
- Middle 12 总 Avg = **46.66**，在部分层设置中最好。
  
- Front 12 和 Back 12 较弱。
  

解释：

- 前层太局部，语义抽象不足。
  
- 后层太晚，记忆信号没有足够深度传播。
  
- 中间层比较平衡。
  
- 全层注入最强，但成本更高。
  

这和很多 mechanistic / adapter 研究的直觉一致：中层通常是语义和任务信息交汇比较丰富的位置。

### 6.11 推理效率和参数量

附录给出了效率和参数量分析。

训练设置：

- 使用 QASPER 的 2,219-sample split 训练 1 epoch。
  
- 最大 backbone training sequence length = 512。
  
- memory write budget = 8192 tokens。
  
- 默认 `r = 8`，`α = 16`。
  
- MSW state 数为 4。
  
- 使用 8 × A800 GPUs、bfloat16、DeepSpeed ZeRO-2。
  

参数量：

- δ-Mem (SSW)：**4.87M**，约 backbone 的 **0.12%**。
  
- δ-Mem (TSW)：**4.87M**，约 **0.12%**。
  
- δ-Mem (MSW)：**19.47M**，约 **0.48%**。
  
- Context2LoRA：**5.90M**，约 **0.15%**。
  
- MemGen：**46.20M**，约 **1.13%**。
  
- MLP Memory：**3078.00M**，约 **76.40%**。
  

效率：

- δ-mem 的 GPU memory usage 接近 Vanilla 和 Context2LoRA。
  
- δ-mem 解码速度比 Vanilla 和 Context2LoRA 慢，因为每步要读写 online state。
  
- 但 δ-mem 比 MemGen 更快、更稳定。
  

这说明 δ-mem 的工程定位不是“完全免费”，而是：

> 用很小的参数和内存开销，换取可观的在线记忆能力。

### 6.12 实验局限


- 8 × 8 state 的可解释性弱，无法像 Mem0 那样直接检查记忆条目。
  
- 论文主要展示 benchmark 分数，还没有充分展示真实多日 Agent 交互中的可控记忆管理。
  
- 记忆写入和遗忘由训练出的机制决定，人工删除、隐私控制、来源追踪都不如文本记忆自然。
  
- 对特别精确的事实回忆，文本/图记忆可能仍有优势。


## 七、结论与展望

### 7.1 论文结论

论文的核心结论是：

> 有效记忆不一定必须来自长上下文、外部检索或大规模微调。一个很小的在线关联记忆状态，只要能动态更新并直接耦合到 attention，也可以显著提升 memory-heavy 任务表现。

δ-mem 的关键贡献在于证明了一个新的方向：

- 记忆可以是 compact online state。
  
- 记忆可以直接参与 forward computation。
  
- 记忆可以在 frozen backbone 上工作。
  
- 记忆可以通过低秩 correction 高效注入。
  

这给 Agent Memory 研究带来了一个新分支：**内部在线状态记忆**。


### 7.2 和 Mem0、MemGPT、RAG 的对比

可以用一张简表理解：

|方法|记忆形式|记忆如何影响模型|优点|局限|
|---|---|---|---|---|
|RAG|文本 chunk|拼回 prompt|简单、可解释|检索噪声、占 token|
|MemGPT|分层文本/外部记忆|上下文换入换出|长任务管理清晰|依赖控制策略|
|Mem0|抽取后的文本/图记忆|检索相关事实给模型|适合生产 Agent，易检查|仍依赖文本检索|
|δ-mem|在线矩阵状态|修正 attention|小、动态、低 token|不可解释、难人工编辑|

所以 δ-mem 不是替代所有 Agent Memory，而是补上了一个以前相对弱的方向：

> 让记忆从“外部资料”变成“模型内部计算的一部分”。



### 7.3 可能的未来工作

#### 方向一：更强的 online state

δ-mem 用 8 × 8 state 证明了小状态有效，但这也只是开始。

可以研究：

- state size 如何自适应变化？
  
- 不同任务是否需要不同 state 维度？
  
- 能否让 state 有结构化分区？
  
- 能否把 state 和显式 memory entry 对齐？
  

#### 方向二：认知式 belief layer

δ-mem 的 state 是连续的，不显式表达信念。如果后续研究能把它和认知科学里的 belief revision 结合，可能形成新方向：

- 连续 state 表示隐式信念。
  
- 文本/图记忆表示显式事实。
  
- belief layer 负责处理矛盾、不确定性、置信度和时间变化。
  

这会比单纯“加记忆库”更像一个自我演化 Agent 的认知层。

#### 方向三：写入粒度自适应

论文固定比较 TSW、SSW、MSW，但真实 Agent 中，不同信息应该用不同粒度写入。

例如：

- 用户偏好：适合 segment-level 写入。
  
- 关键事实：适合精细写入。
  
- 长期任务状态：适合 multi-state。
  
- 噪声闲聊：应该弱写入或不写入。
  

未来可以做 **adaptive writing granularity（自适应写入粒度）**：

> 模型自己判断当前信息是 token 级、message 级、episode 级，还是不值得写。

#### 方向四：和文本/图记忆混合

δ-mem 的短板是不可解释，Mem0 的短板是仍占 token、依赖检索。一个自然方向是混合系统：

 文本/图记忆：保存可解释事实、时间线、用户偏好  
 δ-mem 在线状态：保存连续历史信号、隐式关联和短期动态  
 belief layer：协调冲突、置信度、遗忘和更新

这可能更接近真实 Agent 的长期记忆架构。

#### 方向五：真实 Agent 长期任务评测

论文用了 HotpotQA、LoCoMo、MemoryAgentBench，这已经不错。但未来还需要更真实的 Agent 评测：

- 多天项目协作。
  
- 长期个性化学习助手。
  
- 多 session 编程 Agent。
  
- 任务状态不断变化的研究助理。
  
- 有矛盾、有隐私删除、有错误记忆纠正的交互环境。
  

真正的 Agent Memory 不只是问答分数，而是长期行为是否更稳定、更可信、更少重复犯错。

最后贴一下作者在报告最后提到的未来方向：

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783103883018_image.png)
