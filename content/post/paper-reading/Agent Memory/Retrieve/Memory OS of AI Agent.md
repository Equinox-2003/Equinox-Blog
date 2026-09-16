---
title: "论文精读 | Memory OS of AI Agent"
description: "记忆也能做段页式存储管理"
date: 2026-07-13T12:52:59+08:00
lastmod: 2026-07-13T12:52:59+08:00
draft: false

categories:
  - paper-reading
tags:
  - LLM
  - Agent
  - Agent Memory
  - OS

toc: true
math: true
mermaid: true
cover: https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783918952686_image.png
banner: https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783937272615_image.png
---

<!--more-->



## 零、写在前面

作者的talk：[Talk | 北京邮电大学康佳政：MemoryOS：智能体的记忆操作系统](https://www.bilibili.com/video/BV1RBbUzQEFr/?spm_id_from=333.337.search-card.all.click)

前段时间有做Agent相关内容的群u在准备期末时感慨os的时候发现Agent就像一个OS一样：

-   CPU调度——任务调度
-   内存管理——上下文管理
-   文件系统——长期记忆
-   I/O设备——Tools
-   设备驱动/设备管理器——MCP
-   PCB——AgentState
-   进程通信——Multi-Agent

那么既然OS已经有了相当成熟的设计哲学，那么借鉴 OS Memory 的设计思想，结合人类的记忆习惯，本文这篇工作就显得非常的自然。



## 一、标题

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783918952686_image.png)

标题就点明了本文的理念：**把长期记忆当成一套需要分层存放、动态迁移、按需调度、主动淘汰的资源系统来管理**。



## 二、摘要

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783936699862_image.png)

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783924091061_image.png)

作者从两个痛点出发：

- context window 有限，不能一直塞入全部聊天记录；
- 即使 context 很长，跨 session、跨时间间隔的对话仍容易出现事实不一致、用户偏好遗忘和人格不稳定。

于是论文提出 **MemoryOS**，由四个明确模块组成：

```text
Memory Storage -> Memory Updating -> Memory Retrieval -> Response Generation
```

底层记忆被分成三层：

```text
STM：最近对话
MTM：按话题聚合的历史对话段
LPM：用户与 Agent 的长期 Persona
```

摘要中特别强调两类迁移规则：

- **STM -> MTM**：按 dialogue-chain 的 FIFO 规则迁移；
- **MTM -> LPM**：按 segment heat 和 segmented page 机制迁移。

## 三、引言

### 3.1 Motivation

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783924837104_image.png)

-   **核心问题：**
    -   标准大语言模型受限于固定长度的上下文窗口，导致长期记忆严重匮乏，只依赖RAG方法，检索的精度大大降低，存在事实不一致、个性化不足等问题。
-   **碎片化方案：很多 memory 工作只优化一个环节：**
    -   有的更会组织知识，如 A-Mem；
    -   有的更会检索，如 MemoryBank；
    -   有的重构 context 控制方式，如 MemGPT。
-   **MemoryOS**
    -   借鉴操作系统的内存管理和调度机制，构建段页式三级存储架构及四大核心模块(存储、更新、检索、生成)，提供全链路用户记忆管理方案，让 AI智能体拥有持久记性与深度个性。

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783927498215_image.png)



## 四、相关工作

论文把 LLM Agent Memory 大致分成三类。

### 4.1 Knowledge-organization methods

代表包括 TiM、A-Mem、Grounded Memory。**这类方法关心：如何把对话中的事实、推理过程、关系整理成更好检索的结构。**

- **TiM**：存 reasoning outcomes，而不是只存原始对话。
- **A-Mem**：把记忆组织成带链接的 note network。

它们的优点是记忆内容更有结构；不足是对“新旧信息如何在不同时间尺度迁移”讨论较少。



### 4.2 Retrieval mechanism-oriented methods

代表包括 MemoryBank、Generative Agents、EmotionalRAG。**重点是：从大量历史记录中怎样找回当前最相关的内容。**

- **MemoryBank**：结合语义检索和遗忘曲线。
- **Generative Agents**：维护自然语言 memory stream，并结合重要性和时间等信号检索。

它们提升了 recall，但通常没有一个完整的“从最近对话到长期 persona”的分层状态机。



### 4.3 Architecture-driven methods

代表包括 MemGPT 和 Self-Controlled Memory。

- **MemGPT**：把上下文、recall、archival memory 做成类似虚拟内存的层级，通过显式读写管理上下文。
- **Self-Controlled Memory**：使用双 buffer 和 controller 控制什么时候召回。

MemoryOS 与它们最接近，但它把 OS 类比进一步落在：

```text
dialogue page -> topic segment -> persona entry
FIFO -> STM 迁移
heat score -> MTM 保留与 LPM 升级
```



## 五、方法

### 5.1 overview

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783927498215_image.png)



### 5.2 三级存储

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783934242081_image.png)

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783934798515_image.png)



#### 5.2.1 STM：Short-Term Memory

STM 保存实时对话，基本单元叫 **dialogue page**：

$$
page_i = {Q_i, R_i, T_i}
$$
**分别是用户 query、模型 response、timestamp。**

但作者没有让每一页孤立存在，而是为每页维护 **dialogue chain**：

$$
page_i^{chain} = {Q_i, R_i, T_i, meta_i^{chain}}
$$
LLM 会做两步判断：

1. 新 page 是否与前面 page 语义连续？连续则链接，否则从当前 page 重置一条链；
2. 对链中 page 进行总结，得到 `meta_chain`。

**STM 不只是“最近七轮聊天记录”，还试图知道最近几轮是不是还在说同一件事。**

论文实现中 STM queue 长度为 **7**。新 page 入队；队列满时最旧 page 按 FIFO 迁移至 MTM。



#### 5.2.2 MTM：Mid-Term Memory

它采用 **Segmented Paging**：

```text
Page：一轮 user-query + agent-response
Segment：同一话题下的一组 pages
```

例如：

```text
Segment A：减脂与日常锻炼
  - Page 1：用户想减脂
  - Page 2：用户每周跑步三次
  - Page 3：用户膝盖不适，改为游泳

Segment B：电影偏好
  - Page 1：用户喜欢科幻片
  - Page 2：用户喜欢《Arrival》
```



#### 5.2.3 LPM：Long-term Personal Memory

LPM 存的是不应被轻易遗忘、会影响长期个性化的内容。它分为 User Persona 和 Agent Persona。

**User Persona**：

- User Profile：较稳定属性，如姓名、性别、出生年份；
- User KB：从历史对话抽取的用户事实；
- User Traits：兴趣、习惯、偏好等动态特质。

**Agent Persona**：

- Agent Profile：Agent 固定角色设定；
- Agent Traits：交互中逐渐形成的风格、推荐记录、用户指定的新设置等。

论文为 User Traits 预设了 90 个维度，分为：

-   basic needs and personality
-   AI alignment dimensions
-   content platform interest tags

>   这使它很适合做产品化 user profile，但也带来一个局限：**用户偏好被预定义的 schema 约束，可能不适合开放世界中不断出现的新类型偏好。**



### 5.3 如何更新？

#### 5.3.1 STM-MTM

新 page 是否归入某个 segment，由下式判断：

```text
F_score = cosine_similarity(segment_embedding, page_embedding)
        + Jaccard_similarity(segment_keywords, page_keywords)
```

>   Q：什么是Jaccard_similarity？
>
>   A：两个集合的交集大小，除以它们的并集大小。

其中：

- 第一项看语义接近程度；
- 第二项看关键词是否重合；
- 分数超过阈值，page 就合并进这个 topic segment。



#### 5.3.2 MTM-LPM

MTM segment 有 heat score：

```text
Heat = alpha * N_visit
     + beta  * L_interaction
     + gamma * R_recency

R_recency = exp(-time_since_last_access / lambda)
```

三个含义分别是：

- `N_visit`：被检索越多，说明越常用；
- `L_interaction`：包含 page 越多，说明这一话题承载的互动越丰富；
- `R_recency`：最近访问过，说明当前仍可能相关。

**当 MTM segment 超容量时，最低 heat 的 segment 被淘汰。**

当 segment heat 超过阈值，论文设为 **5**，该 segment 及其 pages 会被 LLM 用来更新：

```text
User KB
User Traits
Agent Traits
```

然后该 segment 的 `L_interaction` 被置零，使 heat 降低，避免它不断重复触发 persona 写入。

User KB 与 Agent Traits 各维护固定长度 **100** 的 FIFO queue。



### 5.4 如何检索/响应

**检索模块：**

给定 query，系统同时从三层取内容。

| 层级  | 检索方式                                                  | 用途                  |
| --- | ----------------------------------------------------- | ------------------- |
| STM | 直接取全部近期 pages                                         | 保留当前会话连续性           |
| MTM | 先取 Top-m segment，再从中取 Top-k page                      | 找到与当前问题有关的历史事件细节    |
| LPM | User KB、Agent Traits 各取 Top-10；稳定 profile/traits 一并使用 | 提供长期个性化和 persona 信息 |
|     |                                                       |                     |

每次 MTM 检索后，系统会更新该 segment 的访问次数和 recency，因此 retrieval 会影响后续保留策略。

**响应模块**

综合检索到的内容和预设的提示词模版，输入底层的LLM 做出回答相应



## 六、实验

### 6.1 设置

论文使用两个数据集：

- **GVD**：15 位虚拟用户与助手持续 10 天的多轮互动，每天至少两个话题。考察个性化、连贯性与长期记忆。
- **LoCoMo**：平均约 300 turns、9K tokens 的长对话，问题包括 Single-hop、Multi-hop、Temporal、Open-domain。

对比方法：TiM、MemoryBank、MemGPT、A-Mem，以及 Full / RAG 类对照。主模型包括 GPT-4o-mini 与 Qwen2.5 系列。

关键配置：

-   STM queue = 7
-   MTM segment maximum length = 200
-   User KB / Agent Traits capacity = 100
-   MTM -> LPM heat threshold = 5
-   MTM retrieved pages top-k = 10



### 6.2 GVD：个性化与连贯性

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783936346693_image.png)

这组结果支持了一个比较直接的结论：分层记忆 + persona 更新在长期个性化聊天中有帮助。



### 6.3 LoCoMo：长对话问答

在 GPT-4o-mini 上，MemoryOS 的平均 rank 在 F1 和 BLEU-1 均为第一。

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783936535808_image.png)

论文报告，相对其选择的 baseline，MemoryOS 在 GPT-4o-mini 上平均提升 **49.11% F1**、**46.18% BLEU-1**。

值得注意的是：

- 原论文报告的 A-Mem 与作者复现的 A-Mem* 有不小差距，说明结果对实现和环境比较敏感；
- 在 GPT-4o-mini 的 Multi-hop F1 上，原论文报告的 A-Mem 为 45.85，高于 MemoryOS 的 41.15。MemoryOS 的“整体最优”不等于每一项绝对最优。



### 6.4 效率

论文在 LoCoMo 上比较了 recalled tokens 与平均 LLM 调用次数：

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783936655090_image.png)

它的效率定位很明确：

- 相比 MemGPT，少取大量 tokens；
- 相比 A-Mem，少做很多 LLM 调用；
- 代价是 token 不如最轻量方法少，但换取更高 F1。



### 6.5 消融、超参数和案例

作者移除了 MTM、LPM、Dialogue Chain 或整个 MemoryOS。

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783936610474_image.png)

结论是：

- **MTM 影响最大**：说明按 topic 管理历史 page 是主要性能来源；
- **LPM 次之**：说明 persona 对长期个性化有效；
- **Dialogue Chain 影响较小**：链式近期上下文有帮助，但不是主要提升来源；
- 去掉整个 MemoryOS，性能显著下降。

超参数分析中，MTM 取回 page 的 `k` 从 5 增到 10 时收益明显；继续提高到 20、30、40 后收益递减，甚至可能引入噪声。因此论文采用 `k = 10`。



## 七、总结

### 7.1 贡献

MemoryOS 给出了一个容易沟通、容易实现的系统分工：

```text
最近发生什么？          STM
某个历史话题里发生过什么？ MTM
这个用户/Agent 长期是谁？  LPM
```

并且让每层都配套有：

```text
写入粒度、迁移规则、淘汰规则、检索方式、生成用途。
```

个人感觉这个论文写的很规整 ，而且让所有设计都能回到同一个 OS memory-management 隐喻。



### 7.2 局限

这篇工作仍有一些值得注意的不足：

- **Heat score 是手工规则**：访问越多的 segment 会更热，**可能形成“被频繁检索所以更容易继续被检索”的自我强化循环；错误但高频的话题也可能被长期保留**。
- **Persona 更新缺少不确定性**：User KB 和 Traits 被确定性地抽取与更新，没有表达“临时偏好、长期偏好、例外情形、相互矛盾偏好”的 confidence。
- **固定 90 维 trait schema 有约束**：容易工程部署，但可能漏掉开放世界的新型用户特征。
- **冲突/遗忘机制较粗**：MTM 用 heat 淘汰、LPM 用固定长度 FIFO，但没有详细讨论语义冲突、错误记忆校正和来源追溯。
- **写入成本仍需更完整报告**：论文的效率表以 retrieval tokens 和每次响应 LLM calls 为主，但 LLM 对 dialogue chain、segment summary、persona extraction 的长期写入成本没有充分拆开。

