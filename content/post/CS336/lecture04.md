---
title: "Lecture04"
description: ""
date: 2026-06-03T17:49:41+08:00
lastmod: 2026-06-03T17:49:41+08:00
draft: true

categories:
  - CS336
tags:
  - LLM

toc: true
math: true
mermaid: true
---

<!--more-->



## 零、写在前面

lecture04主要围绕两个主题：

1. **Attention alternatives：注意力机制的替代方案**
    - 为什么标准 attention 在长上下文下很贵？
    - Linear Attention、Mamba-2、Gated Delta Net、Sparse Attention 等思路如何降低成本？
    - 为什么很多新模型采用 attention + alternative module 的混合架构？

2. **Mixture of Experts，MoE：专家混合模型**
    - MoE 是什么？
    - 为什么它越来越流行？
    - 路由 routing 怎么做？
    - 训练 MoE 有什么困难？
    - DeepSeek MoE v1/v2/v3 等模型做了哪些设计？



## 一、Attention alternatives

### 1.1 为什么需要 Attention alternatives

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1780480817125_image.png)

标准 attention 是：

$$
Attn(Q, K, V) = \rho (QK^T)V
$$
其中：

- `ρ` 通常是 softmax；
- `QK^T` 的形状是 `n × n`；
- 所以计算和存储复杂度与 `n²` 有关。

这在短上下文时还可以接受，但当上下文长度变成：32k、128k、1M tokens时，attention 成本会急剧上升。

所以：**当 context size 变大时，我们如何控制 attention 的成本？**



### 1.2 控制长上下文成本的基础工具

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1780480815374_image.png)

#### 1.2.1 Combine local + global attention

也就是混合局部注意力和全局注意力。

例如：

```text
大多数层用局部窗口 attention
少数层用 full attention
```

>   [Swin Transformer精读](https://equinox.wiki/post/paper-reading/swin-transformer%E7%B2%BE%E8%AF%BB/)中就引入了局部注意力 与 shifted window 注意力
>
>   而且 shifted window 比起 lecture03 提到的sliding window 更加具有访存友好。



#### 1.2.2 Systems engineering

系统工程优化也很重要。

例如：

- FlashAttention；
- KV cache 优化；
- GQA / MQA；
- kernel fusion；
- memory-efficient attention；
- 分布式并行策略。

这些方法不改变 attention 的数学本质，但能显著优化实际运行速度。



#### 1.2.3 更激进的替代方案

讲义接下来关注更激进的方案：

1.  Linear Attention
2.  Mamba-2
3.  Gated Delta Net
4.  Sparse Attention
5.  MoE

这些方法希望不仅靠工程优化，而是改变计算结构本身，获得更大的效率收益。



### 1.3 Linear Attention：线性注意力

#### 1.3.1 如果没有 softmax？

>   Can we do better when ρ is the identity?

如果暂时忽略 softmax，令：

```text
ρ = identity
```

$$
Attn(Q, K, V) = QK^TV = Q (K^TV)
$$

**复杂度变化：**

原始计算：

$$
QK^T: O(n^2 d_k)\\
(QK^T)V: O(n^2 d_v)
$$
总成本大约：

$$
O(n^2 d_k + n^2 d_v)
$$
如果改成：

$$
K^T V
$$
先计算：

$$
K^T \in R^{d_k × n}\\
V \in R^{n × d_v}\\
K^T V \in R^{d_k × d_v}
$$
成本：

$$
O(n d_k d_v)
$$
然后：
$$
Q(K^T V)
$$
成本：

$$
O(n d_k d_v)
$$
总成本：

$$
O(2 n d_k d_v)
$$
这就是**线性注意力**。

因为 softmax 不是线性变换，不能用结合律。

所以 linear attention 的核心是：

> 用某种方式替代或近似 softmax attention，使 attention 可以写成可结合的形式。



#### 1.3.2 Linear Attention 的 recurrent form

讲义接着说，线性注意力不仅训练时可以高效，还天然有一种 **RNN 形式**。

这需要我们从**自回归**的角度来考虑第 `t` 个位置的输出：

定义一个状态矩阵：

$$
S_t = S_{t-1} + k_t v_t^T
$$
然后输出：

$$
y_t = q_t^T S_t
$$
这里：

- `k_t` 是第 `t` 个 token 的 key；	
- `v_t` 是第 `t` 个 token 的 value；
- `q_t` 是第 `t` 个 token 的 query；
- `S_t` 是累积历史信息的状态。



因为 $S_t$ 只依赖 $S_{t-1}$ ，这和 RNN 的状态更新非常像：

$$
hidden_t = f(hidden_{t-1}, input_t)
$$
所以 linear attention 有一种 duality，双重形式：

1. **训练时**可以用并行矩阵形式计算；
2. **推理时**可以用 recurrent 形式逐 token 更新状态。

讲义说：

> This duality allows us to train efficiently using the parallel form and inference efficiently using the serial, linear form.

**这非常关键，因为对推理很有用：**



#### 1.3.3 Linear Attention 下的推理

标准 Transformer 生成时需要 KV cache。

每生成一个新 token，需要 attend 到所有历史 token，这个是 O(n) 的。

如果每一步都越来越长，总体成本会随上下文增长。

而 recurrent linear attention 只需要维护一个固定大小状态：$S_t$。

每一步更新：$S_t = S_{t-1} + k_t v_t^T$

然后：$y_t = q_t^T S_t$

**如果状态大小固定，那么每一步推理成本不随历史长度线性增长，或者至少显著降低对 KV cache 的依赖。**

**这就是很多 attention alternative 的核心吸引力。**

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1780482360456_image.png)



### 1.4 RetNet 与带衰减的线性注意力





















