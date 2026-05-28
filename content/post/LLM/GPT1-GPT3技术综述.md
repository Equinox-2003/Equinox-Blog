---
title: "GPT1 GPT3技术综述"
description: ""
date: 2026-05-28T22:42:23+08:00
lastmod: 2026-05-28T22:42:23+08:00
draft: true

categories:
  - LLM
tags:
  - LLM
  - 

toc: true
math: true
mermaid: true
---

<!--more-->



## 零、写在前面

之前手搓过GPT-2（[LLMs-From-Scratch](https://equinox.wiki/categories/llms-from-scratch/)），然后一时兴起想梳理一下GPT1~3的技术路线。



## 一、背景：从 Transformer 到自回归语言模型

GPT 系列建立在 Transformer 之上。Transformer 最初由 Vaswani et al. 在 Attention Is All You Need 中提出，其核心是用 self-attention 取代 RNN/CNN 中的序列建模机制，使模型能够并行处理 token，并通过 attention 直接建模长距离依赖。



GPT 使用的是 **decoder-only Transformer**：

- 只保留 Transformer decoder 风格的单向结构；
- 使用 causal mask，当前位置只能看到当前位置及其之前的 token；
- 训练目标是 **autoregressive language modeling**，即根据前文预测下一个 token。

形式上，给定 token 序列 $u = (u_1, u_2, \dots, u_n)$，语言模型最大化：
$$
\sum_i \log P(u_i \mid u_1, \dots, u_{i-1}; \theta)
$$

这个目标非常简单，但 GPT-1 到 GPT-3 的核心发现是：当数据、模型容量和计算规模持续扩大时，单纯的 next-token prediction 可以学到大量可迁移的语言知识、世界知识和任务模式。



## 二、GPT-1：Generative Pre-Training + Supervised Fine-Tuning

### 2.1 标题

>   Improving Language Understanding by Generative Pre-Training

通过大规模预训练来提升语言理解。

论文本身并没有给GPT起名字，大家拿Generative Pre-Training的首字母起了GPT这个名字。好在后续GPT工作的持续跟进，爆火出圈，使得GPT这个名字可以说是家喻户晓了。



### 2.2 核心问题

GPT-1 面对的问题是：当时很多 NLP 任务都依赖任务特定的监督数据和模型结构，迁移能力有限。GPT-1 的目标是先用大规模无标注文本训练一个通用语言表示，再用少量有标注数据微调到具体任务。

这形成了后来非常重要的两阶段范式：

1. **Unsupervised pre-training**：在 BooksCorpus 上训练自回归语言模型。
2. **Supervised fine-tuning**：把下游任务改写成统一的 token 序列输入，再继续训练模型完成分类、问答、自然语言推理等任务。



### 2.3 architecture

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1779980345214_image.png)

GPT-1 使用 12 层 Transformer decoder：

- masked multi-head self-attention；
- position-wise feed-forward network；
- learned positional embeddings；
- Byte Pair Encoding tokenization；
- 单向语言模型目标。

从今天的视角看，GPT-1 的结构并不复杂，真正重要的是它证明了 **decoder-only 生成式预训练也能迁移到语言理解任务**。这点很关键，因为同一时期的很多工作更强调 task-specific architecture，后来的 BERT 则走向 encoder-only masked language modeling 路线。



### 2.4 下游任务适配方式

GPT-1 把不同 NLP 任务统一改写为序列输入。例如：

- 文本分类：`[text] -> label`
- 自然语言推理：`[premise] [delimiter] [hypothesis] -> label`
- 多选问答：把每个候选答案分别拼接到上下文后打分

这种做法的意义是：尽量减少任务特定结构，让同一个预训练模型通过输入格式适配不同任务。



### 2.5 技术贡献与局限

**贡献：**

- 建立了 “pre-training + fine-tuning” 的 GPT 路线；
- 证明生成式语言模型可以服务于语言理解任务；
- 将无标注文本中的知识迁移到多个有监督 benchmark。

**局限：**

- **仍然强依赖下游监督 fine-tuning；**
- **模型和数据规模有限；**
- **对任务的泛化主要来自微调，而不是自然语言 prompt 或上下文学习。**























