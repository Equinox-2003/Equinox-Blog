---
title: "Transformer"
description: ""
date: 2026-04-30T13:08:53+08:00
lastmod: 2026-04-30T13:08:53+08:00
draft: true

categories:
  - MachineLearning
tags:
  - LLM

toc: true
math: true
mermaid: true
---

<!--more-->

## 一、Transformer

### 1.1 Seq2Seq

**Seq2Seq(序列到序列)**模型输入和输出都是一个序列，输入与输出序列长度之间的关系有两种情况。

1.  输入跟输出的长度一样
2.  机器决定输出的长度。

序列到序列模型的常见应用：

![BQACAgUAAyEGAASHRsPbAAETzL9p81aTT9EszbUSwWIqiT0DO-YslQACaygAAncCmFfuLZwmLR9fXTsE.png](https://img.remit.ee/api/file/BQACAgUAAyEGAASHRsPbAAETzL9p81aTT9EszbUSwWIqiT0DO-YslQACaygAAncCmFfuLZwmLR9fXTsE.png)

>   Q：既然把语音识别系统跟机器翻译系统接起来就能达到语音翻译的效果，那么为什么 要做语音翻译？
>
>    A：世界上很多语言是没有文字的，无法做语音识别。因此需要对这些语言做语音翻译， 直接把它翻译成文字。





























