---
title: "注意力替代方案与MoE"
description: ""
date: 2026-05-28T15:55:49+08:00
lastmod: 2026-05-28T15:55:49+08:00
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

本来没打算听 cs336 的lecture，但发现这课真有东西的，而且assignment2好像有些新东西，先严肃学习一下。

这篇博客主要是对 lecture 04进行总结。

lecture04 主要围绕两个主题：

1.  **Attention alternatives：注意力机制的替代方案**
    - 为什么标准 attention 在长上下文下很贵？
    - Linear Attention、Mamba-2、Gated Delta Net、Sparse Attention 等思路如何降低成本？
    - 为什么很多新模型采用 attention + alternative module 的混合架构？
2.  **Mixture of Experts，MoE：专家混合模型**
    - MoE 是什么？
    - 为什么它越来越流行？
    - 路由 routing 怎么做？
    - 训练 MoE 有什么困难？
    - DeepSeek MoE v1/v2/v3 等模型做了哪些设计？



## 一、Attention alternatives

1.1 



































