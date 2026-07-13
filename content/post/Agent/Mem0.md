---
title: "Mem0框架解读"
description: "爆火的mem0原来只是prompt engineering？"
date: 2026-07-03T11:07:35+08:00
lastmod: 2026-07-03T11:07:35+08:00
draft: false

categories:
  - Agent
tags:
  - LLM
  - Agent Memory

toc: true
math: true
mermaid: true
banner: https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783572616762_image.png
cover: https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783417582876_image.png
---

<!--more-->



## 零、写在前面

过去一年里，mem0的代码仓库 在 github 上获得了6w+ star，因为对外接口简单、适配性较好，被众多开源 Agent 所使用。去年放出来的版本感觉是为了占坑，其实实际使用中会面临token消耗高以及效率较慢的问题，前段时间mem0又发布了mem0-v3，把论文里面提到的 mem0-g 也给删了，做了很多修改，但本身还是 prompt-engineering，个人感觉这种外挂式的memory manager 不是我们理想中的 agent memory。



## 一、技术报告

论文地址：[Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/pdf/2504.19413)



### 1.1 为什么需要 Agent Memory

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783419115869_image.png)

记忆是人类智能的基础。人会记住过去的互动、偏好、关系和经历，然后用这些信息指导后续交流。

>   比如：
>
>   - 朋友上次说他不吃乳制品，你下次约饭会避开奶油餐厅。
>   - 学生上次说自己 Transformer 没学懂，老师下次讲 Agent 时会先补基础。

对 AI Agent 来说也是一样。如果没有长期记忆，它就像一个非常聪明但严重健忘的人：当场聊得很好，换个会话就断片。



### 1.2 mem0 架构

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783419390381_image.png)



agent 会维护用户与llm的对话信息 `messages`，然后 mem0 的流程是这样的：



**Extraction Phase（记忆抽取阶段）**：从新对话中抽取值得保存的候选记忆。

Mem0 每次处理一个新的消息对，例如用户一句、助手一句。为了让抽取更准确，它不只看当前消息，还会看两类上下文：

- **Conversation Summary（对话摘要）**：对整段历史的全局概括。
- **Recent Messages（最近消息窗口）**：最近 `m` 条消息，原论文实验里 `m = 10`。

这相当于人记笔记时既看“整件事的大背景”，也看“刚才具体说了什么”。

>   Q：为什么需要这两者？
>
>   A：
>
>   - 只有当前消息，可能不知道“他说的它”指什么。
>   - 只有全局摘要，可能丢掉细节。
>   - 两者结合，才更容易抽出真正有长期价值的事实。
>

原论文用 GPT-4o-mini 作为抽取模型，把新消息和上下文组合成 prompt，让 LLM 输出一组候选记忆。



**Update Phase（记忆更新阶段）**：判断候选记忆应该如何进入长期记忆库。

这是 Mem0 最关键的设计。它不是简单地“抽到什么就存什么”，而是先用向量检索找到 top `s` 条相似旧记忆，原论文实验里 `s = 10`，**然后让 LLM 根据新旧记忆关系选择四种操作：**

>   本质还是 prompt-engineering

- **ADD（新增）**：旧记忆里没有这条信息，就新增。
- **UPDATE（更新）**：新信息比旧信息更完整，就替换或增强旧记忆。
- **DELETE（删除）**：新信息和旧信息冲突，旧信息应移除。
- **NOOP（不操作）**：信息已经存在，或不值得保存。

### 1.3 mem0-v3 

#### 1.3.1 写入/记忆提取链路

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783420524924_image.png)

>   用户和 Agent 对话结束后，Mem0 **异步**把这轮对话加工成长期记忆；**新版只追加新记忆**，**不让 LLM 自动改写或删除旧记忆**。

##### 1.3.1.1 三类持久化存储

**SQL Database：Facts + Metadata**

这里保存结构化信息，比如：

```
memory text
user_id
agent_id
run_id
created_at
updated_at
metadata
history
recent messages
```

SQLite **主要保存 `history` 和最近 messages**；memory 文本和 metadata 很多时候也会存在 vector store payload 里。



**Vector Database：Embeddings + Similarity**

这里保存 memory embedding，用于语义相似度搜索。

例如：

```
query: What coffee should I avoid?
```

可以通过向量检索找到：

```
User switched from almond milk lattes to oat milk lattes after developing almond sensitivity.
```



**Entity Store：Entities + Relationships**

这里保存实体和 memory 的链接关系。

它不是传统完整知识图谱，但有“图”的味道：

```
Entity -> Memory IDs
Memory -> Related Entities
```

**作用是提高专名、人物、地点、物品相关查询的召回。**



##### 1.3.1.2 写入/记忆提取链路

**1. Store New Memories：对话后异步写记忆**

图最左边：

```
Store New Memories
AFTER RESPONSE (ASYNC)
```

即，**Agent 先正常回复用户，然后 Mem0 在后台处理这轮对话**。

这样做的好处是用户不用等记忆系统慢慢抽取、embedding、入库。比如：

```
用户：我最近因为 almond sensitivity，把 almond milk latte 换成 oat milk latte 了。
助手：了解，我以后会注意这个偏好。
```

助手回复完后，Mem0 才开始后台提取记忆。



**2. Context Lookup：先查相关旧记忆**

```
Context Lookup
FIND RELATED MEMORIES
```

这一步不是给用户回答问题，而是给“记忆抽取器”找背景。

比如系统里已有旧记忆：

```
User likes almond milk latte.
User often drinks coffee before coding.
```

现在用户说自己改喝 oat milk，Mem0 会先查到这些相关旧记忆，让后面的抽取器知道：这是一个“偏好变化”，不是孤立新事实。

这一步对应源码里的：

```
existing_results = vector_store.search(...)
last_messages = db.get_last_messages(...)
```

也就是同时看 **旧长期记忆** 和 **最近几轮上下文**。



**3. Extract Memories：从输入和上下文中抽取新记忆**

```
Extract Memories
FROM INPUT + CONTEXT
ADD ONLY
```

这是 Mem0-v3 的关键变化：**ADD ONLY**。

旧版可能让 LLM 判断：

```
ADD / UPDATE / DELETE / NONE
```

新版主要让 LLM 做一件事：**只抽取值得新增的长期记忆**

例如它不会简单覆盖旧记忆为：

```
User likes oat milk latte.
```

更理想的抽取是：

```
User switched from almond milk lattes to oat milk lattes after developing almond sensitivity.
```

这条记忆更完整，因为它保留了：

-   旧偏好：almond milk latte
-   新偏好：oat milk latte
-   变化原因：almond sensitivity
-   变化关系：switched from A to B

这就是新版 prompt 强调的 **contextually rich memory**，不是只抽原子事实。



**4. Deduplicate + Embed：去重并向量化**

```
Deduplicate + Embed
VECTORIZE NEW MEMORIES
```

抽出来的 memory 不能直接全塞进去，否则会产生大量重复。

比如连续两轮都提到：

```
User prefers oat milk latte.
User switched to oat milk latte.
```

系统会做去重，源码里至少有 hash 去重逻辑。然后把新 memory 送进 embedding model，变成向量。

向量化后的好处是，之后用户即使不说完全相同的话，也能搜到相关记忆：

```
用户问：我现在喝咖啡有什么限制来着？
```

虽然没出现 “oat milk” 或 “almond sensitivity”，语义检索也可能找回那条 memory。



**5. Entity Linking：识别实体并建立链接**

```
Entity Linking
IDENTIFY + LINK ENTITIES
```

这一步是 Mem0-v3 很重要的增强。

它会从 memory 中抽实体，比如：

```
User switched from almond milk lattes to oat milk lattes after developing almond sensitivity.
```

可能抽出：

```
almond milk latte
oat milk latte
almond sensitivity
```

再比如：

```
User celebrated promotion with Elena at Osteria Francescana.
```

可能抽出：

```
Elena
Osteria Francescana
promotion
```

然后 Entity Store 会记录：

```
Elena -> linked_memory_ids: [...]
Osteria Francescana -> linked_memory_ids: [...]
```

之后如果用户问：

```
我和 Elena 去哪家餐厅庆祝过？
```

系统不只靠 embedding，还可以通过实体 `Elena` 找到相关 memory，给检索结果加权。



#### 1.3.2 检索路线：Multi-signal retrieval，多信号检索

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783440290186_image.png)

用户查询来了以后，不只靠向量相似度，而是同时用 **语义相似度、关键词匹配、实体匹配** 三路信号找记忆，最后把三路分数融合，选出 Top-K memories。

**1、Query**

比如用户问：

```text
我上次和 Elena 去哪家餐厅庆祝升职？
```

这个 query 里有几类线索：

-   语义线索：庆祝、餐厅、升职
-   关键词线索：Elena
-   实体线索：Elena 这个人名

如果只靠 embedding，有时能找到；但如果记忆很多、专名很多，单靠 embedding 可能不稳。



**2、Preprocess**

第二步：

```
Preprocess
NORMALIZE KEYWORDS, EXTRACT ENTITIES
```

也就是把 query 拆成几种可检索的信号。

例如：

```
我上次和 Elena 去哪家餐厅庆祝升职？
```

可能处理成：

```
关键词：Elena, 餐厅, 庆祝, 升职
实体：Elena
语义 query：整句话的 embedding
```

在 Mem0 代码里，对应两类处理：

```
lemmatize_for_bm25(query)
extract_entities(query)
embedding_model.embed(query, "search")
```

其中：

-   **lemmatize_for_bm25**：把词形标准化，方便关键词检索。
-   **extract_entities**：抽取人名、地点、机构、专名等实体。
-   **embedding**：把整句话变成向量，用于语义检索。



**3、三路并行检索**

三路检索是并行跑的。

**Semantic Search：语义检索**

```
Semantic Search
VECTOR SIMILARITY SCORING
```

它看的是“意思像不像”。

比如 query：

```
我现在喝咖啡有什么需要避免的吗？
```

即使 memory 里写的是：

```
User switched from almond milk lattes to oat milk lattes after developing almond sensitivity.
```

没有出现“避免”这个词，语义检索也可能找回来，因为它理解“almond sensitivity”和“avoid”有关。

语义检索读的是：

```
Vector Database
EMBEDDINGS + SIMILARITY
```



**Keyword Search：关键词检索**

```
Keyword Search
NORMALIZED TERM MATCHING
```

它看的是“字面上有没有匹配”。

这个对专名特别重要，比如：

```
Osteria Francescana
Elena
Shopify
Agent Memory
```

Embedding 有时会把专名泛化掉，但关键词检索能精确命中。

比如 memory 是：

```
User celebrated promotion with Elena at Osteria Francescana.
```

用户问：

```
Osteria Francescana 是我什么时候提到的？
```

这时 BM25/keyword search 很有用。



**Entity Search：实体检索**

```
Entity Search
ENTITY GRAPH MATCHING
```

这一路不是直接搜全文，而是搜实体索引。

比如系统之前在写入时已经建立了：

```
Elena -> memory_1, memory_7
Osteria Francescana -> memory_1
Shopify -> memory_3, memory_9
```

现在 query 里抽到 `Elena`，系统就可以把和 Elena 相关的 memories 加分。

这一路读的是：

```
Entity Store
ENTITIES + RELATIONSHIPS
```

注意：这里的 “ENTITY GRAPH MATCHING” 不一定代表完整知识图谱推理，更像是实体到 memory 的链接匹配。



**Rank Fusion：分数融合**

三路检索完以后进入：

```
Rank Fusion
COMBINED SCORING, TOP-K SELECTION
```

意思是把三种分数合起来：

```
final_score = semantic_score + keyword_score + entity_boost
```

在 Mem0 OSS 代码里更准确地说，是归一化后融合：

```
semantic_score
+ normalized BM25 score
+ entity boost
然后除以 max_possible_score
```

比如某条 memory：

```
User celebrated promotion with Elena at Osteria Francescana.
```

对于 query：

```
我和 Elena 去哪家餐厅庆祝升职？
```

它可能得到：

```
semantic_score = 0.78
keyword_score  = 0.65
entity_boost   = 0.30
final_score    = 综合后很高
```

另一条 memory：

```
User likes Italian restaurants.
```

可能语义上也有点像，但没有 Elena，也没有升职事件：

```
semantic_score = 0.60
keyword_score  = 0.10
entity_boost   = 0.00
final_score    = 较低
```

所以最终第一条排在前面。



**Results：返回 Top-K Memories**

最后输出：

```
Results
TOP-K MEMORIES
```

也就是最相关的几条长期记忆。

这些 memories 会被放进 Agent 的上下文里，帮助它回答用户。





### 1.4 实验

#### 1.4.1 数据集：LOCOMO

论文使用 **LOCOMO** 数据集。**它专门评估长期对话记忆能力**。

数据集特点：

- 包含 10 段长对话。
- 每段约 600 个 dialogue turns。
- 每段平均约 26000 tokens。
- 对话分布在多个 session 中。
- 每段对话平均约 200 个问题和标准答案。

问题类型包括：

- **Single-hop（单跳问题）**：从一个对话片段里找一个事实。
- **Multi-hop（多跳问题）**：需要综合多个对话片段的信息。
- **Temporal（时间问题）**：需要理解事件顺序、日期、相对时间。
- **Open-domain（开放域问题）**：可能需要结合对话记忆和常识/外部知识。

论文没有评估 adversarial unanswerable questions，因为对应 ground truth 不完整。



#### 1.4.2 评估指标

论文用了两类指标。

第一类是答案质量：

- **F1 Score**：看生成答案和标准答案的词重合程度。
- **BLEU-1**：看 unigram 级别的文本重合。
- **LLM-as-a-Judge (J)**：让另一个 LLM 判断答案是否正确。

作者特别指出，F1 和 BLEU 对事实错误不够敏感。比如标准答案是“Alice born in March”，模型答“Alice born in July”，词面重合很多，但关键事实错了。所以论文更重视 J 指标。

第二类是部署指标：

- **Token Consumption（Token 消耗）**：回答时取出的记忆或 chunk 有多少 token。
- **Search Latency（检索延迟）**：找记忆或找 chunk 花多久。
- **Total Latency（总延迟）**：检索加生成答案总共花多久。

这点对 Agent Memory 很关键，因为真实系统里不能只看准确率。一个系统如果准但慢、贵、刚写入的记忆几个小时后才可用，那就很难用于交互式 Agent。



#### 1.4.3 对比方法

论文比较了六类 baseline：

- 已有 LOCOMO benchmark 方法：LoCoMo、ReadAgent、MemoryBank、MemGPT、A-Mem。
- 开源记忆方案：LangMem。
- 标准 RAG：把整段对话切成不同 chunk size，再检索 top-k chunk。
- Full-context：把完整对话历史直接放进模型上下文。
- OpenAI Memory：使用 ChatGPT 的记忆功能做对比。
- Memory Provider：Zep，基于 temporal knowledge graph 的记忆平台。

这组 baseline 覆盖了目前常见的几种思路：

- “直接看全文”的 full-context。
- “切块检索”的 RAG。
- “长期记忆管理”的 MemGPT / MemoryBank / A-Mem。
- “商业或产品化记忆服务”的 OpenAI Memory / Zep。



#### 1.4.4 不同问题类型上的效果

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783423621727_image.png)



**Single-hop**

Mem0 表现最好：

- Mem0：F1 = 38.72，B1 = 27.13，J = 67.13
- Mem0g：F1 = 38.09，B1 = 26.03，J = 65.71

解释：

> 单跳问题通常只要找一个明确事实，自然语言记忆已经足够。图结构反而可能带来一点额外复杂度。



**Multi-hop**

Mem0 仍然表现最好：

- Mem0：F1 = 28.64，J = 51.15
- Mem0g：F1 = 24.32，J = 47.19

这点有点反直觉。我们可能以为图结构适合多跳推理，但实验中基础 Mem0 更好。论文认为，图记忆可能在多步整合时引入冗余或检索效率问题。

对研究者来说，这是很好的选题信号：

> 图记忆不是天然更强，关键在于图怎么构建、怎么压缩、怎么检索、怎么避免噪声扩散。



**Open-domain**

Zep 最高，Mem0g 接近：

- Zep：J = 76.60
- Mem0g：J = 75.71
- Mem0：J = 72.93

解释：

> 开放域问题可能需要关系组织和外部知识结合，图记忆有优势，但 Zep 在这一项略领先。



**Temporal**

Mem0g 最强：

- Mem0g：F1 = 51.55，J = 58.13
- Mem0：J = 55.51

解释：

> 时间问题需要理解事件顺序、相对时间和关系变化。图结构加 timestamp 更容易表达“过去是什么、现在是什么、哪个事实先发生”。



#### 1.4.5 延迟、Token 和整体质量

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1783441994741_image.png)

关键数字：

- Full-context：平均输入约 26031 tokens，整体 J = 72.90，total p95 = 17.117s。
- Mem0：memory tokens = 1764，整体 J = 66.88，total p95 = 1.440s。
- Mem0g：memory tokens = 3616，整体 J = 68.44，total p95 = 2.590s。
- 最强 RAG 配置整体 J 大约 60.97，低于 Mem0 / Mem0g。
- LangMem 的 search p95 达到 59.82s，交互场景中很重。
- OpenAI 的 total latency 很低，但论文指出其记忆预抽取成本没有计入，并且 benchmark 方法给了它较特殊的上下文处理方式。

最核心的 trade-off 是：

> Full-context 分数略高，但每次都要读完整 26000 tokens；Mem0 / Mem0g 分数接近，并把延迟和 Token 大幅压下来。



#### 1.4.6 Token 成本和构建时间分析

论文还比较了长期记忆库本身的大小：

- Mem0 每段 conversation 平均约 7k tokens。
- Mem0g 平均约 14k tokens。
- Zep 的 memory graph 超过 600k tokens。
- 原始完整对话约 26k tokens。

这组数字非常值得注意：图记忆如果设计不好，可能比原始对话还膨胀很多。论文认为 Zep 的膨胀来自节点上缓存摘要、边上也存事实，导致大量冗余。

**此外，论文观察到 Zep 新增记忆后不能立即稳定检索，过几小时再检索效果更好，说明它可能有较重的异步图构建过程。Mem0g 则声称最坏情况下图构建也在一分钟以内完成。**

对生产 Agent 来说，这很关键：

> 记忆不只是“最终能不能查到”，还要看“刚写进去能不能马上用”。



#### 1.4.7 实验的局限

- **主要评估集中在 LOCOMO，场景仍然是对话记忆，不代表所有 Agent 任务。**
- LLM-as-a-Judge 虽然比 F1 / BLEU 更贴近语义，但仍然可能受评委模型偏差影响。
- **Mem0 / Mem0g 使用 GPT-4o-mini 进行抽取、更新和生成，换模型后效果可能变化。**
- 图记忆在 multi-hop 上不如基础 Mem0，说明图结构的优势还没有被完全释放。
- OpenAI Memory 和 Zep 这类系统的内部实现不完全透明，benchmark 很难做到完全公平。



### 1.5 总结和展望

**长期记忆系统要想进入真实 Agent 应用，必须同时解决准确率、更新一致性、检索效率、延迟和 Token 成本。**

1.  记忆要从原始历史中抽象出来。

2.  记忆必须能更新。

3.  图记忆不是万能药。


可能的方向？：

**1、更聪明的 Memory Consolidation**

现在 Mem0 的整合主要是候选事实和相似旧记忆之间做操作判断。未来可以做得更深：

- **多条碎片记忆合成稳定 belief。**
- **把“临时偏好”和“长期偏好”分开。**
- **对矛盾信息维护多个假设，而不是立刻覆盖。**
- **类睡眠离线 consolidation：在用户不交互时批量整理、压缩、去冲突。**



**2、图记忆的低成本构建与检索**

Mem0g 证明图结构对时间和关系有帮助，但也暴露出延迟和冗余问题。可研究的问题包括：

- 如何决定哪些记忆值得进入图？
- 如何避免节点和边爆炸？
- 如何给边加时间有效期、置信度和来源？
- 如何让自然语言记忆和图记忆协同检索？

这里很适合做 CCF-A 级别的问题，但前提是要有清晰的新机制和扎实 benchmark，而不是简单“加一个 knowledge graph”。



**3、面向冲突信息的记忆更新**

现实世界里信息经常矛盾：

- 用户今天说喜欢 A，明天说不喜欢 A。
- 不同来源对同一事实说法不同。
- Agent 自己过去的结论后来被证明错误。

原论文的 Mem0 用 `DELETE / UPDATE` 处理冲突，即使是 mem0-v3，也只是做了粗粒度的处理。未来可以更像人：

- 保留冲突来源。
- 记录时间和置信度。
- 区分“事实变化”和“过去记错了”。
- 在回答时表达不确定性。

这会把 Agent Memory 从“事实库”推向“信念系统”。



**4、Memory Evaluation 不只看 QA**

LOCOMO 主要用问答评估记忆。未来可以评估更真实的 Agent 能力：

- 长期任务是否少犯重复错误？
- 个性化是否更稳定？
- 多日交互后用户信任是否提升？
- 记忆错误会不会导致危险决策？
- 记忆删除请求是否真的被执行？

这类评估比单纯 QA 更难，但也更接近真实 Agent。



## 二、 代码解读

官方代码仓库：[mem0](https://github.com/mem0ai/mem0)

mem0的仓库非常庞大，最核心的是这几块：

```text
├── mem0/                         # Python SDK 核心
│   ├── memory/
│   │   ├── main.py               # 最核心：Memory / AsyncMemory，add/search/update/delete 都在这里
│   │   ├── base.py               # MemoryBase 抽象接口
│   │   ├── storage.py            # SQLite history + recent messages
│   │   └── utils.py              # 消息解析、JSON 清洗、vision 解析等工具
│   ├── configs/
│   │   ├── base.py               # MemoryConfig / MemoryItem
│   │   └── prompts.py            # 记忆抽取 prompt、旧版 update prompt、procedural memory prompt
│   ├── vector_stores/            # Qdrant、Chroma、PGVector、Redis、Faiss 等向量库适配
│   ├── embeddings/               # OpenAI、Ollama、HuggingFace 等 embedding 适配
│   ├── llms/                     # OpenAI、Anthropic、Gemini、DeepSeek 等 LLM 适配
│   ├── reranker/                 # reranker 可选模块
│   └── utils/
│       ├── factory.py            # LLM / Embedder / VectorStore / Reranker 工厂
│       ├── scoring.py            # semantic + BM25 + entity boost 融合打分
│       ├── lemmatization.py      # BM25 用的词形还原
│       └── entity_extraction.py  # 实体抽取
├── mem0-ts/                      # TypeScript SDK
├── server/                       # 自托管服务端
├── openmemory/                   # OpenMemory 应用层
├── tests/                        # 单测，很多行为可以从这里反推
└── docs/                         # 文档和 changelog
```

当然，我们只学习 mem0 的方法实现的话，优先读下面5个文件：

- mem0/memory/main.py：`Memory` 主类。
- mem0/configs/prompts.py：新版 ADD-only extraction prompt。
- mem0/utils/scoring.py：检索融合打分。
- mem0/utils/factory.py：provider 工厂。
- mem0/memory/storage.py：SQLite 历史和 recent messages。



### 2.1 Memory 类

初始化时，它会创建四类组件：

```python
self.embedding_model = EmbedderFactory.create(...)
self.vector_store = VectorStoreFactory.create(...)
self.llm = LlmFactory.create(...)
self.db = SQLiteManager(...)
```

- **LLM**：负责从对话中抽取可保存的记忆。
- **Embedder**：把记忆文本变成向量。
- **Vector Store**：保存和检索向量。
- **SQLite**：保存操作历史和最近消息。

默认配置来自 MemoryConfig：

```text
vector_store: 默认 qdrant
llm: 默认 openai
embedder: 默认 openai
history_db_path: 默认 ~/.mem0/history.db
reranker: 默认 None
version: 默认 v1.1
```



### 2.2 `add()`：记忆写入流程

#### 2.2.1 外层入口

入口在 Memory.add

它先做输入标准化：

```text
str -> [{"role": "user", "content": "..."}]
dict -> [dict]
list[dict] -> 原样使用
其他类型 -> 报错
```

然后处理几个关键参数：

- `user_id`：用户级记忆。
- `agent_id`：Agent 级记忆。
- `run_id`：某次运行/任务级记忆。
- `metadata`：用户自定义元数据。
- `infer=True`：是否让 LLM 抽取记忆。
- `memory_type="procedural_memory"`：是否写入程序性记忆。

这里有一个非常重要的设计：**Mem0 要求记忆必须有作用域**。也就是说，你不能随便写一条全局记忆，至少要绑定到 `user_id`、`agent_id` 或 `run_id` 之一。

类比：

> 记忆不是丢进一个公共抽屉，而是必须贴标签：这是 Alice 的记忆、某个 Agent 的记忆，还是某次任务的记忆。



#### 2.2.2 infer=False：直接原文入库

如果 `infer=False`，代码不会调用 LLM，而是把每条非 system message 直接写进 vector store。

流程很简单：

```text
message content
  -> embedding_model.embed(content, "add")
  -> _create_memory()
  -> vector_store.insert()
  -> db.add_history(event="ADD")
```

这适合已经有清洗好的 memory 文本，不需要 LLM 再判断。



#### 2.2.3 infer=True：V3 phased batch pipeline

重点在 Memory._add_to_vector_store。

源码把它分成 8 个 phase。



**Phase 0：上下文收集**

```python
session_scope = _build_session_scope(filters)
last_messages = self.db.get_last_messages(session_scope, limit=10)
parsed_messages = parse_messages(messages)
```

这里会拿到同一作用域下最近 10 条消息，用来帮助 LLM 解析代词和上下文。

比如用户说：我昨天又去了那家店。

如果没有最近消息，你不知道“那家店”是哪家。`last_messages` 就是给 LLM 的短期上下文。



**Phase 1：检索相近旧记忆**

```python
query_embedding = self.embedding_model.embed(parsed_messages, "search")
existing_results = self.vector_store.search(... top_k=10 ...)
```

这一步不是为了回答用户，而是为了给记忆抽取器提供“当前相关旧记忆”。

作用有两个：

- 帮助去重：已经记过的不要再记。
- 帮助链接：新记忆如果和旧记忆相关，生成 `linked_memory_ids`。



**Phase 2：一次 LLM 抽取**

```python
system_prompt = ADDITIVE_EXTRACTION_PROMPT
user_prompt = generate_additive_extraction_prompt(...)
response = self.llm.generate_response(... response_format={"type": "json_object"})
```

新版核心 prompt 是 ADDITIVE_EXTRACTION_PROMPT。

它要求 LLM 做的事情不是“判断旧记忆怎么改”，而是：

```text
只做 ADD：
从 user 和 assistant messages 中抽取所有值得长期保存的信息，
生成上下文丰富、自包含、带 attribution 的 memory。
```

输出格式类似：

```json
{
  "memory": [
    {
      "id": "0",
      "text": "User prefers dark mode and Vim keybindings",
      "attributed_to": "user",
      "linked_memory_ids": ["old-memory-uuid"]
    }
  ]
}
```

这就是当前代码和论文版最大的不同：**LLM 不再输出 UPDATE/DELETE 操作，只输出新增 memory。**



**Phase 3：批量 embedding**

```python
mem_embeddings_list = self.embedding_model.embed_batch(mem_texts, "add")
```

把所有抽取出的 memory 文本一次性转成向量。失败时会 fallback 到逐条 embed。

工程意义：

> 单条 embedding 像一个个寄快递，batch embedding 像统一装车发货，成本和延迟更可控。



**Phase 4-5：hash 去重和 payload 构造**

代码对每条 memory 计算：

```python
mem_hash = hashlib.md5(text.encode()).hexdigest()
text_lemmatized = lemmatize_for_bm25(text)
```

然后构造 payload：

```text
data: 原始 memory 文本
text_lemmatized: BM25 检索用文本
hash: 去重 hash
created_at / updated_at
user_id / agent_id / run_id / metadata
attributed_to
```

这里的 `text_lemmatized` 很重要。它不是给向量检索用的，而是给关键词检索/BM25 用的。

比如：

```text
attending / attended / attends -> attend
memories -> memory
```

这样用户搜 “meeting” 时，更容易匹配到不同词形的历史记忆。



**Phase 6：批量写入 vector store 和 history**

```python
self.vector_store.insert(vectors=all_vectors, ids=all_ids, payloads=all_payloads)
self.db.batch_add_history(history_records)
```

vector store 保存“可检索记忆”，SQLite 保存“操作历史”。

二者分工不同：

- vector store：面向 search / get_all。
- SQLite history：面向 history(memory_id)，记录这条记忆经历过 ADD/UPDATE/DELETE。



**Phase 7：实体抽取和 entity linking**

代码会调用 extract_entities_batch：

```python
all_entities = extract_entities_batch(all_texts)
```

然后把实体写进单独的 entity store，并记录：

```text
entity_text -> linked_memory_ids
```

例如：

```text
Entity: Shopify
linked_memory_ids:
  - memory about promotion
  - memory about team switch
```

注意：当前 OSS 已经移除了外部 graph store。过去可能有 Neo4j / Memgraph / Kuzu 这种显式图数据库，现在 OSS 里“图记忆”的工程替代是 **built-in entity linking**。

所以你可以这样理解：

> 当前 Mem0 OSS 不是完整知识图谱，而是“实体索引 + 记忆反向链接”。它有图的味道，但不是传统 Graph RAG 那种显式三元组图。



**Phase 8：保存最近消息并返回**

```python
self.db.save_messages(messages, session_scope)
return [{"id": ..., "memory": ..., "event": "ADD"}]
```

SQLite 只保留每个 session_scope 最近 10 条消息，作为下一次抽取时的短期上下文。



### 2.3 search()：多信号检索流程

入口在 Memory.search，真正核心在 Memory._search_vector_store。

当前检索不是单纯向量相似度，而是融合三类信号：

```text
semantic similarity + BM25 keyword score + entity boost
```

#### 2.3.1 Step 1：query 预处理

```python
query_lemmatized = lemmatize_for_bm25(query)
query_entities = extract_entities(query)
```

同一个 query 会走两条路：

- 原文 query：用于 embedding 语义检索。
- lemmatized query：用于 BM25 关键词检索。
- query_entities：用于实体增强。



#### 2.3.2 Step 2-3：语义检索

```python
embeddings = self.embedding_model.embed(query, "search")
semantic_results = self.vector_store.search(... top_k=internal_limit ...)
```

这里的 `internal_limit = max(limit * 4, 60)`。

意思是：用户要 top_k=20，但系统内部先多取一些候选，再融合排序。

类比：

> 不是直接从简历堆里选 20 人，而是先粗筛 80 人，再用多维评分精排。

#### 2.3.3 Step 4-5：BM25 keyword search

```python
keyword_results = self.vector_store.keyword_search(...)
bm25_scores[mem_id] = normalize_bm25(raw_score, midpoint, steepness)
```

BM25 适合找精确词、专名、编号、标题。比如：

- “Shopify”
- “Osteria Francescana”
- “There Will Be Blood”
- “Agent Memory”

这些东西有时 embedding 会“语义理解过度”，反而关键词更靠谱。

BM25 原始分数不是 0-1，所以 Mem0 用 sigmoid 归一化，逻辑在 [normalize_bm25](D:/gitRepo/mem0/mem0/utils/scoring.py:43)。



#### 2.3.4 Step 6：entity boost

如果 query 里抽到了实体，系统会查 entity store：

```python
entity_boosts = self._compute_entity_boosts(query_entities, filters)
```

核心逻辑在 Memory._compute_entity_boosts：

1. 从 query 抽实体，最多取 8 个。
2. 对实体做 embedding。
3. 在 entity store 搜相似实体。
4. 找到实体后，把该实体 linked 的 memory id 加 boost。

为什么这有用？

假设 query 是：

```text
What happened with Poppy's vet visit?
```

如果某条 memory 明确链接到实体 `Poppy`，即使它和 query 的整体 embedding 不是最高，也应该被加分。



#### 2.3.5 Step 7-8：融合打分

融合公式在 score_and_rank。

直觉上是：

```text
raw_score = semantic_score + bm25_score + entity_boost
final_score = raw_score / max_possible_score
```

`max_possible_score` 会根据启用了哪些信号动态变化：

- semantic only：1.0
- semantic + BM25：2.0
- semantic + entity：1.5
- semantic + BM25 + entity：2.5

其中 entity boost 的权重是 ENTITY_BOOST_WEIGHT = 0.5。

还有一个很重要的细节：**threshold 先卡 semantic score**。也就是说，如果向量相似度太低，即使 BM25 很高，也救不回来。测试里也专门覆盖了这一点。

这说明 Mem0 的检索哲学是：

> 语义相关性是入场券，关键词和实体是加分项。



#### 2.3.6 Optional reranker

如果你配置了 reranker，并且调用：

```python
memory.search(query, filters=..., rerank=True)
```

系统会在融合检索后再 rerank 一次。

reranker 创建逻辑在 RerankerFactory，支持 Cohere、sentence_transformer、LLM reranker、HuggingFace 等。



### 2.4 `update()` 和 `delete()`：现在更像手动维护 API

虽然 `add()` 已经是 ADD-only，但 `Memory` 类仍然提供手动 update/delete。

#### 2.4.1 update

入口在 Memory.update，内部调用 Memory._update_memory。

流程：

```text
读取旧 memory
  -> 如果 data 改了，重新 embedding
  -> 更新 vector store payload
  -> 写 SQLite history: event="UPDATE"
  -> 如果文本变了，清理旧实体链接，再重新抽实体并链接
```

这个 update 是 API 层面的显式操作，不是 `add()` 自动推理出来的 UPDATE。



#### 2.4.2 delete

入口在 Memory.delete，内部调用 Memory._delete_memory。

流程：

```text
读取旧 memory
  -> vector_store.delete(memory_id)
  -> db.add_history(event="DELETE", is_deleted=1)
  -> 从 entity store 里移除这个 memory_id
```

这说明当前系统并不是完全不能改删，而是：

> LLM 自动写入流程默认只 ADD；人工/API 层仍然可以 update/delete。

这个边界一定要分清。



### 2.5 Provider 适配层：为什么 Mem0 能接这么多后端？

Mem0 的工程可扩展性主要来自 factory pattern。

#### 2.5.1 LLM Factory

LlmFactory 根据 provider 名称创建不同 LLM：

```text
openai / anthropic / gemini / deepseek / ollama / vllm / litellm / langchain / ...
```

所以你在 config 里换 provider，本质上就是换一个 `generate_response()` 实现。



#### 2.5.2 Embedder Factory

EmbedderFactory 支持：

```text
openai / ollama / huggingface / azure_openai / gemini / vertexai / together / fastembed / ...
```

对 Mem0 来说，embedder 只需要提供：

```text
embed(text, "add" or "search")
embed_batch(texts, "add" or "search")
```



#### 2.5.3 VectorStore Factory

VectorStoreFactory 支持很多向量库：

```text
qdrant / chroma / pgvector / milvus / pinecone / redis / elasticsearch / faiss / weaviate / ...
```

这些 vector store 都要遵循 VectorStoreBase 的接口：

```text
insert
search
delete
update
get
list
reset
keyword_search
search_batch
```

这里的关键是 `keyword_search()`。如果某个后端不支持 BM25，默认返回 `None`，系统就退化成 semantic + entity 检索。



### 2.6 SQLite 在这里做什么？

SQLite 不是主记忆库，它更像“操作日志 + 最近上下文缓存”。

实现位于 SQLiteManager。

它有两张表：

```text
history:
  memory_id
  old_memory
  new_memory
  event
  created_at
  updated_at
  is_deleted
  actor_id
  role

messages:
  session_scope
  role
  content
  name
  created_at
```

`history` 用来支持：

```python
memory.history(memory_id)
```

`messages` 用来支持下一次 add 时的上下文解析。每个 session_scope 只保留最近 10 条。

这点像人类记忆里的“短期对话缓存”：

> 长期事实进 vector store，最近几句对话进 SQLite messages，用来帮助下一次抽取时理解“它”“那家店”“上次说的项目”指什么。







