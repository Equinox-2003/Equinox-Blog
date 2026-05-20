---
title: "MoCo精读"
description: ""
date: 2026-05-17T16:11:57+08:00
lastmod: 2026-05-17T16:11:57+08:00
draft: false

categories:
  - paper-reading
tags:
  - Contrastive Learning
  - CV
  - NLP

toc: true
math: true
mermaid: true
---

<!--more-->



## 零、写在前面

MoCo 的核心思想是：把对比学习看成一个“字典查询”问题，用一个队列维护大量负样本，并用一个动量更新的 key encoder 保证这些负样本特征的一致性，从而高效学习无监督视觉表征。

感觉这个 idea 好难想到，可能因为没有了解过对比学习？



## 一、题目和作者

```
Momentum Contrast for Unsupervised Visual Representation Learning
```

这个格式就是：XXX Method for XXX

用动量对比学习的方式来做无监督的视觉表征学习。

**Momentum Contrast**简称**MoCo**。

作者是 **Kaiming He**、Haoqi Fan、Yuxin Wu、Saining Xie、Ross Girshick，来自 Facebook AI Research，发表于 CVPR 2020。这篇论文是视觉自监督学习领域的里程碑之一，对后来的 MoCo v2、SimCLR、BYOL、DINO、MAE、CLIP 等工作都有重要影响。



## 二、摘要

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1779096301874_image.png)

摘要简单介绍了下MoCo。

作者把**对比学习**当作一种**字典查找**。

-   给模型一个 query，例如一张图像的某个增强版本；
-   字典里有很多 key，也就是其他图像或图像增强版本的特征；
-   模型要从字典中找到与 query 匹配的那个 key，让二者尽可能接近；
-   同时把不匹配的 key 推远。

模型学到的是：

>   **同一图像的不同视图应该相似，不同图像应该不相似。**

这个字典是如何构建的？

>   “we build a dynamic dictionary with a queue and a moving-averaged encoder.”

作者用两个机制构建动态字典：

**1、Queue**

队列用于保存大量历史 key。**普通对比学习通常只能使用当前 mini-batch 里的样本作为负样本，负样本数量受 batch size 限制**。MoCo 则维护一个队列：

-   当前 batch 的 key 入队；
-   最旧的 key 出队；
-   队列中保存很多过去 batch 的 key。

这样就能得到一个很大的负样本集合。

**2、Moving-averaged encoder：移动平均编码器**

**队列中保存的是历史 key，但如果编码器变化太快，不同时间产生的 key 可能不一致。**

因此 MoCo 使用一个**移动平均编码器**，也就是前面说的 **momentum encoder**：
$$
\theta_k \leftarrow m\theta_k + (1-m)\theta_q
$$
缓慢地用query的编码器来更新key的编码器。

摘要中写了这么做的好处：**队列 + 移动平均编码器，使得模型可以在线构建一个既大又一致的字典。**

-   **large**：字典中有很多负样本

-   **consistent**：key 表征来自相近的编码器，比较更稳定

-   **on-the-fly**：训练过程中动态构建，不需要预先固定字典

    >   on-the-fly   phr. 立刻、飞快

摘要中也点了一下实验结果：

>   “MoCo provides competitive results under the common linear protocol on ImageNet classification.”

也就是说，把backbone冻住，然后只训练线性分类头做微调，在ImageNet 效果很好。

>   “More importantly, the representations learned by MoCo transfer well to downstream tasks.”

那么说明MoCo可以很好的迁移到下游任务。

**然后也点了一下实验结论**：

>   “MoCo can **outperform** its supervised pre-training counterpart in 7 detection/segmentation tasks...”

MoCo 的无监督预训练，在 7 个检测/分割任务上可以超过对应的 ImageNet 监督预训练。

>   “This suggests that the gap between unsupervised and supervised representation learning has been largely closed...”

作者也提出了观点：**在许多视觉任务中，无监督表征学习和监督表征学习之间的差距已经大大缩小。**

换句话说：

-   过去：监督学习明显强于无监督学习；
-   MoCo 之后：无监督预训练已经可以接近甚至超过监督预训练；
-   尤其是在检测、分割等迁移任务中，MoCo 表现非常强。



## 三、引言

>   视觉无监督学习的关键难点是如何为连续高维图像空间构建有效字典；已有对比学习方法难以同时保证字典“大”和“一致”；MoCo 通过“队列 + 动量编码器”构建一个大规模且稳定的动态字典，从而提升无监督视觉表征学习，并让其在多个下游检测/分割任务中接近甚至超过监督预训练。

### 3.1 问题背景

**NLP 的无监督学习很成功，但 CV 还主要依赖监督预训练。**

BERT、GPT这些模型可以利用大量无标注文本学到强大的语言表示。但在计算机视觉中，当时主流仍然是：、

-   先在 ImageNet 上做有监督预训练；
-   再迁移到检测、分割等下游任务。

作者指出，视觉领域的无监督方法总体上还落后于监督预训练。



### 3.2 为什么 NLP 更容易做无监督学习？

作者给出的解释是：语言和图像的信号空间不同。

语言由一些离散 token 构成，例如：

-   word；
-   sub-word；
-   character；
-   token id。

所以 NLP 很容易构建一个“字典”：

```
the, cat, dog, apple, ...
```

模型可以围绕这些离散 token 做预测、遮盖建模、语言建模等任务。

**CV：连续、高维、没有天然 token**

图像则不同。原始图像是连续高维信号：

-   像素值是连续或近似连续的；
-   图像空间维度极高；
-   图像不像语言那样天然服务于人类通信；
-   图像没有现成的词表或 token 字典。

所以，视觉无监督学习的一个核心难点是：**如何为图像构建合适的字典**。

这就是后文 MoCo 的切入点。



### 3.3 对比学习统一理解为“动态字典查询”

接着，作者回顾了近期一些无监督视觉表征学习方法，指出它们虽然动机不同，但本质上都可以看作在构建动态字典。

这里有几个核心概念：

| 概念                 | 在 MoCo 里的含义                           |
| -------------------- | ------------------------------------------ |
| **dictionary**       | 存放大量 key 表征的集合                    |
| **key**              | 图像或图像块经过 encoder 得到的特征        |
| **query**            | 当前输入图像经过 encoder 得到的特征        |
| **matching key**     | 与 query 来自同一图像/同一语义来源的正样本 |
| **negative keys**    | 与 query 不匹配的其他样本                  |
| **contrastive loss** | 拉近 query 和正样本，推远 query 和负样本   |

直观地说，模型要做的是：

>   给定一个 query，从字典中找到它对应的 key。

例如，对于同一张图片的两个不同裁剪：

>   crop A 编码成 query；
>   crop B 编码成 positive key；
>   其他图片的表示作为 negative keys。

训练目标就是让 query 更接近它自己的 positive key，而远离其他 key。



### 3.4 MoCo 的基本结构

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1779183883964_image.png)

MoCo 的训练流程大致是：

1.  输入一个 query 图像 $x^{query}$；
2.  query encoder 把它编码成 $q$；
3.  字典中有很多 key：$k_0,k_1,k_2,\dots$；
4.  使用 contrastive loss 让 $q$ 匹配正确的 $key$；
5.  字典由一个 queue 维护；
6.  当前 mini-batch 的 $key$ 入队；
7.  最旧的 $key$ 出队；
8.  $key$ 由一个缓慢变化的 momentum encoder 编码。

**queue：负责让字典变大；**
**momentum encoder：负责让字典保持一致。**



### 3.5 好的字典需要 large + consistent

#### 3.5.1 Large dictionary：字典要大

这个很好理解。

因为图像空间是：

-   连续的；
-   高维的；
-   变化丰富的。

如果负样本太少，模型看到的对比对象不够丰富，学到的表示可能不够泛化。

大字典可以更好地采样整个视觉空间，让模型学习更有区分性的特征。



#### 3.5.2 Consistent dictionary：字典要一致

为什么还要一致？

因为字典里的 key 都是 encoder 产生的。如果不同 key 是由差异很大的 encoder 生成的，那么它们可能不在同一个稳定的特征空间里。

这样 query 和 key 的相似度比较就会变得不可靠。

所以作者认为：

-   字典里的 key 最好由相同 encoder 产生；
-   或至少由变化很慢、相近的 encoder 产生。



### 3.6 现有方法的问题：难以同时做到 large 和 consistent

作者指出，以前的对比学习方法通常会在这两点上受限。

**方法一：end-to-end mini-batch 字典**

这类方法只用当前 mini-batch 里的样本作为字典。

**优点：**

-   当前 batch 中的 key 由当前 encoder 产生；
-   所以表示比较一致。

**缺点：**

-   字典大小受 batch size 限制；
-   batch size 又受 GPU 显存限制；
-   很难构建特别大的负样本集合。



**方法二：memory bank**

另一类方法使用 memory bank 存储大量历史样本表示。

**优点：**

-   字典可以很大。

**缺点：**

-   memory bank 中的表示可能是很早之前的 encoder 产生的；
-   encoder 已经变了，但存储的 key 没有同步更新；
-   因此一致性较差。

MoCo 的目标就是在这两类方法之间取长补短。



### 3.7 MoCo的解决方案：queue + momentum encoder

**Queue：让字典足够大**

MoCo 不只使用当前 mini-batch，而是维护一个队列：

-   当前 mini-batch 的 key 加入队列；
-   队列中最旧的 key 被移除；
-   队列大小可以独立设置。

这样一来，字典大小就不再等于 mini-batch size，而可以远大于 batch size。

这解决了 large dictionary 的问题。

**Momentum encoder：让字典保持一致**

但是队列中的 key 来自过去几个 mini-batch。如果 encoder 更新很快，历史 key 和当前 key 的表征空间可能不一致。

因此 MoCo 使用一个缓慢变化的 key encoder：
$$
\theta_k \leftarrow m\theta_k + (1-m)\theta_q
$$


其中：

-   $\theta_q$ 是 query encoder 参数；
-   $\theta_k$ 是 key encoder 参数；
-   $m$ 是 momentum coefficient，通常接近 1。

### 3.8 pretext task：instance discrimination

作者说明：MoCo 是一种构建动态字典的机制，可以搭配不同 pretext task。

本文采用的是比较简单的 **instance discrimination** 任务。

>   **instance discrimination：**如果 query 和 key 是同一张图像的不同增强视图，那么它们匹配；否则不匹配。

这说明 MoCo 的重点不是设计复杂的预训练任务，而是设计一个更好的对比学习机制。



### 3.9 实验目标：不仅看 ImageNet 分类，更看迁移能力

作者强调，无监督学习的主要目标不是只在预训练数据集上表现好，而是学到可以迁移的视觉表征。

所以他们评估了两类任务：

1.  ImageNet linear classification

    冻结 backbone；
    只训练线性分类器；
    测试表征质量。

2.  下游检测/分割任务

    将 MoCo 预训练模型迁移到 PASCAL VOC、COCO 等数据集；

    通过 fine-tuning 看实际迁移效果。

摘要和引言都强调：MoCo 在 7 个检测/分割任务中可以超过 ImageNet 有监督预训练，有些任务优势还比较明显。



## 四、结论

MoCo在一系列视觉的任务和数据集上取得了不错的效果。但是当训练数据从 IN-1M 扩大到 IG-1B 的时候，效果提升不是很大。所以作者认为，可能一个更好的pretext-task 可以解决这一点。

**作者这里提了一个很有预见性的问题：能否用NLP那边的masked-encoding 作为pretext-task来进一步提升呢？**

>   这不就是作者两年后提出的MAE吗（笑哭）

然后作者点题：希望MoCo希望对其他对比学习的任务有帮助。



## 五、相关工作

作者认为，无监督 / 自监督表征学习方法通常可以从两个维度理解：

-   Pretext task（预训练任务 / 代理任务）

    -   它本身不是最终目的，而是为了让模型通过解决这个任务学到有用的表示。

        例如：


        让模型恢复被破坏的图像；
        预测图像块之间的位置关系；
        判断图像是否经过某种旋转；
        预测颜色；
        区分不同图像实例。
    
    这些任务本身并不是我们真正关心的最终任务。真正目标是：**让模型学到可以迁移到分类、检测、分割等下游任务的视觉特征。**

-   Loss function（损失函数）

### 5.1 Loss functions：损失函数相关工作

-   作者首先从损失函数角度回顾已有方法，主要分成三类：
    1.  固定目标的重建 / 分类损失
    2.  对比损失
    3.  对抗损失

**1、固定目标损失：重建或分类**

传统无监督 / 自监督方法经常会定义一个固定目标，然后让模型去预测它。

典型例子包括：
**1）重建输入**

例如自编码器（auto-encoder）：

-   输入一张图像；
-   编码成特征；
-   再从特征重建原图；
-   用 L1 或 L2 loss 衡量重建误差。

**2）预测预定义类别**

例如：

-   预测图像块的相对位置；
-   预测颜色 bin；
-   预测旋转角度。

这类方法通常使用：

-   cross-entropy loss；
-   margin-based loss。

也就是说，模型被要求从预定义类别中选出正确答案。

**2、Contrastive losses：对比损失**

MoCo 主要属于这一类。

对比损失和固定目标损失的关键区别是：
| 类型         | 学习目标                       | 特点                      |
| ------------ | ------------------------------ | ------------------------- |
| 固定目标损失 | 预测固定标签或重建固定像素     | target 预先定义           |
| 对比损失     | 比较样本对在表征空间中的相似性 | target 可在训练中动态变化 |

对比学习不一定要求模型预测某个固定类别，而是要求模型学会：

-   正样本对更相似；
-   负样本对更不相似。

例如在 MoCo 中：

-   同一张图像的两个增强视图是正样本；
-   不同图像的视图是负样本。

这样一来，目标不是固定人工标签，而是由数据本身和当前网络表征动态定义。



### 5.2 对比学习为什么适合无监督表征学习？

对比学习的优势在于，**它可以在没有人工标签的情况下构造监督信号**。

例如给定一张图片 x，做两种数据增强：
$$
x_1 = t_1(x), \quad x_2 = t_2(x)
$$
然后：

-   $x_1$ 和 $x_2$ 应该被拉近；
-   $x_1$ 和其他图像的增强结果应该被推远。

这就形成了一个自监督学习目标。

在 MoCo 的语境中，这个过程被解释为：

-   $x_1$ 编码成 query；
-   $x_2$ 编码成 positive key；
-   其他图像编码成 negative keys；
-   query 需要在字典中找到 positive key。

**3、Adversarial losses：对抗损失**

作者也提到了另一类重要损失：**adversarial loss（对抗损失）**。

对抗损失衡量的是两个概率分布之间的差异。它在无监督生成模型中非常成功，典型代表是 GAN。

GAN 中通常有：

-   generator：生成数据；
-   discriminator：判断数据是真实还是生成的；
-   两者形成对抗训练。

作者指出，对抗方法也被用于表征学习，并且 GAN 和 NCE（Noise-Contrastive Estimation，噪声对比估计）之间存在一定关系。

不过，这篇论文并不主要研究对抗学习，而是把重点放在 contrastive learning 上。



### 5.3 Pretext tasks：代理任务相关工作

接下来，作者从 pretext task 的角度回顾已有方法。

所谓 pretext task，就是人为设计一个任务，让模型通过解决它学到好的视觉表示。

文中列举了几类典型任务。



#### 5.3.1 恢复被破坏的输入

这一类任务要求模型从不完整或被扰动的输入中恢复原始信息。

包括：

-   **denoising auto-encoder**：去噪自编码器；
-   **context auto-encoder**：上下文自编码器；
-   **cross-channel auto-encoder / colorization**：跨通道重建，例如图像上色。

例如图像上色任务：

-   输入灰度图；
-   预测颜色通道；
-   模型为了完成任务，需要理解物体和场景语义。

这类任务的核心逻辑是：

如果模型能恢复缺失信息，说明它学到了一定的结构和语义。



#### 5.3.2 构造伪标签

另一类方法是构造 **pseudo-labels（伪标签）**。

这些标签不是人工标注的，而是由数据本身自动生成。

文中提到的例子包括：	

**1）图像变换相关任务**

例如 exemplar-based task：

-   对同一张图像做多种变换；
-   让模型识别这些变换后的样本属于同一个实例。

**2）图像块排序任务**

例如：

-   给模型打乱的 patch；
-   要求模型恢复正确顺序；
-   或预测 patch 之间的相对位置。

**这类任务迫使模型学习空间结构**。

**3）视频中的追踪或分割**

利用视频天然的时间连续性：

-   跟踪物体；
-   分割运动目标；
-   学习跨帧一致表示。

**4）聚类特征**

例如 DeepCluster 一类方法：

-   先用当前特征对图像聚类；
-   把聚类结果当作伪标签；
-   再训练网络预测这些伪标签。



### 5.4 Contrastive learning 和 pretext task 的关系

作者强调，**contrastive learning 不是某一个具体 pretext task，而是一类 loss / learning framework。**

也就是说，不同的 pretext task 都可以使用对比损失来实现。

例如：

| 方法                               | Pretext task 视角             | Contrastive learning 视角                |
| ---------------------------------- | ----------------------------- | ---------------------------------------- |
| Instance discrimination            | 区分不同图像实例              | 同一实例的增强为正样本，不同实例为负样本 |
| contrastive predictive coding(CPC) | 根据上下文预测未来 / 缺失部分 | 让正确未来 patch 与上下文更相似          |
| contrastive multiview coding(CMC)  | 多视角 / 多通道预测           | 让不同视角或通道的对应表示更相似         |

这说明 MoCo 的贡献不是“发明 instance discrimination”，而是提出一种更通用的机制：**如何为对比学习构建大且一致的动态字典。**



### 5.5 定位

1.  **MoCo 不主要贡献一个新的 pretext task**

2.  **MoCo 主要贡献在 contrastive learning 机制**

3.  **Loss function 和 pretext task 可以解耦**

    MoCo 不是只能用于 instance discrimination。

    论文明确说 MoCo 是一种构建动态字典的机制，可以和多种 pretext task 结合。



## 六、MoCo方法

>   如何用对比学习训练一个无监督视觉表征模型，并且让它拥有足够多、足够稳定的负样本？

### 6.1 Contrastive Learning as Dictionary Look-up

作者把对比学习抽象成一个 dictionary look-up（字典查询） 任务。

给定：

-   一个 query 表征：q
-   一个 key 字典：${k_0,k_1,k_2,\cdots}$
-   其中只有一个 key 是和 query 匹配的正样本，记为 $k_+$
-   其他 key 都是负样本



**query 和 key 是怎么来的？**
$$
q = f_q(x_q)
$$

$$
k = f_k(x_k)
$$

其中：

| 符号  | 含义           |
| ----- | -------------- |
| $x_q$ | query 输入样本 |
| $x_k$ | key 输入样本   |
| $f_q$ | query encoder  |
| $f_k$ | key encoder    |
| $q$   | query 表征     |
| $k$   | key 表征       |

在 MoCo 这篇论文的具体实现里，通常可以理解为：

-   $x_q$：一张图像的一个增强视图；
-   $x_k$：同一张图像的另一个增强视图；
-   这两个视图构成正样本对；
-   其他图像的视图作为负样本。



### 6.2 InfoNCE 

#### 6.2.1 InfoNCE vs NCE

>   关于NCE：[NCE](https://equinox.wiki/post/math/nce/)

在NCE的推导中，我们发现，NCE目标函数的最优解就是使得 模型分布 等于 真实分布，那么为什么还要引入InfoNCE？

**1、目的不同**

**NCE**：可以计算出图片的真实概率。

**表征学习**：不关心概率，更希望将一个图片变成一个好的向量：和正样本接近，负样本远离。



**2、无法定义高维图像的噪声分布 `pn(x)`**

对于一张256 * 256 的图片，我们很难定义噪声数据的分布。



**3、二分类 vs 多分类**

InfoNCE 公式：
$$
 \mathcal{L}_q = - \log \frac{\exp(sim(q, k^+))}{\exp(sim(q, k^+)) + \sum_{i=1}^K \exp(sim(q, k^-_i))} 

$$
Softmax 它不仅要求正样本分数高，还**强制要求正样本分数必须比所有负样本的加和还要高得多**。

**这种机制使得 InfoNCE 提取的特征比 NCE 强大得多。**



**4、最大化“互信息（Mutual Information）”**

InfoNCE 最早是由 Aäron van den Oord（DeepMind的巨佬）在 2018 年的 CPC (Contrastive Predictive Coding) 论文中正式提出的。

他之所以加上 "Info" 这个前缀，是因为他在数学上证明了：**最小化 InfoNCE 损失，实际上是在最大化输入 $x$ 和它的上下文/特征 $c$ 之间的“互信息” $I(x; c)$ 的下界。**

具体的数学结论是：
$$ I(x; c) \ge \log(K) - \mathcal{L}_{\text{InfoNCE}} $$
*(其中 $K$ 是负样本的数量)*

这在理论上提供了一个极其强大的证明：
只要我们使用的负样本数量 $K$ 足够多，且让 InfoNCE Loss 降得足够低，**模型提取出的特征向量就保留了关于原图最多的“本质信息”**，排除了无用的噪声。这就从理论上解释了为什么对比学习学到的特征那么好用。



因此，采用InfoNCE 可以驱动模型学习：

-   拉近正样本对；
-   推远负样本对。



### 6.3 三种对比学习机制的比较

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1779264564040_image.png)

作者对比了三种机制，它们的主要区别在于：

-   key 是怎么保存的；
-   key encoder 是怎么更新的；
-   dictionary 的大小和一致性如何权衡。

#### 6.3.1 End-to-End

这种方法直接用当前 mini-batch 中的样本作为字典。

特点：

-   query encoder 和 key encoder 都通过反向传播更新；
-   当前 batch 内的 key 是用当前 encoder 编码的；
-   因此 key 表征比较一致。

但问题是：

-   dictionary size 受 mini-batch size 限制；
-   mini-batch size 又受 GPU 显存限制；
-   如果想要很多负样本，就需要很大的 batch。

论文指出，end-to-end 方法的字典大小与 mini-batch size 耦合，因此受 GPU memory 限制。



#### 6.3.2 Memory bank 机制

Memory bank 方法会为整个数据集维护一个表征存储库。

特点：

-   可以支持很大的 dictionary；
-   每次训练从 memory bank 中采样负样本；
-   不需要在当前 batch 中包含所有负样本。

但问题是：

-   memory bank 中的表示可能是旧 encoder 计算出来的；
-   当前 encoder 已经更新了，但旧 key 没有同步变化；
-   因此 key 表征可能不一致。

所以 memory bank 的优势是“大”，但劣势是“一致性不足”。



#### 6.3.3 MoCo 机制

-   End-to-End 一致性好，但是容量不够大
-   memory bank 够大但不够一致。

所以MoCo选择折中：

1.  用 queue 维护大量 key；
2.  用 momentum encoder 编码 key，使 key 表征变化更平滑。



### 6.4 核心思想

作者认为，对比学习本质上是在高维连续图像空间上构建一个离散字典。

这个字典是动态的，因为：

-   key 是从数据中随机采样来的；
-   encoder 在训练过程中不断变化；
-   因此 key 的表示也会随时间变化



### 6.5 Dictionary as a Queue

MoCo 引入 queue 之后：

-   当前 mini-batch 的 key 会进入队列；
-   之前若干 mini-batch 的 key 会保留在队列中；
-   最旧的 mini-batch key 会被移除。

这样，dictionary size 不再等于 mini-batch size，而是可以独立设置为一个更大的**超参数**。

MoCo 的队列更新非常简单：

-   当前 mini-batch 经过 key encoder 得到 key；
-   这些 key 被加入队列尾部；
-   队列头部最旧的一批 key 被移除；
-   队列始终保持固定长度。

queue 带来三个直接好处：

1.  **第一，负样本更多**

    字典可以远大于 mini-batch size，因此每个 query 可以和更多负样本对比。

2.  **第二，计算可控**

    不需要像超大 batch 训练那样一次性把所有负样本都放进显存。

3.  **第三，旧 key 会逐渐淘汰**

    最旧的 key 最可能与当前 encoder 不一致，因此定期移除它们反而有助于保持 dictionary 的新鲜度。

queue 虽然解决了 dictionary size 的问题，但也带来一个新问题：

如果 key encoder 变化太快，那么：

-   较早进入队列的 key 来自旧 encoder；
-   新进入队列的 key 来自新 encoder；
-   它们可能不在完全一致的表征空间里；
-   query 和这些 key 的相似度比较就会变得不稳定。

**MoCo解决方式：**

MoCo 使用两个 encoder：

| 编码器 | 作用       | 更新方式     |
| ------ | ---------- | ------------ |
| $f_q$  | 编码 query | 反向传播更新 |
| $f_k$  | 编码 key   | 动量更新     |

key encoder 的参数 $θ_k$ 不直接通过反向传播更新，而是用 query encoder 参数 $θ_q$ 的移动平均来更新：
$$
\theta_k \leftarrow m\theta_k + (1-m)\theta_q
$$
**MoCo 的完整训练流程**

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1779268133214_image.png)



## 七、实验

实验部分主要验证 MoCo 的两个核心主张：

-   大规模字典有助于对比学习；
-   动量编码器能保持字典一致性，从而优于 memory bank 等机制；
-   MoCo 学到的无监督表征不仅在线性分类上有效，也能很好迁移到下游任务。

#### 7.1 预训练数据集设置

论文在两个无标签数据集上做无监督预训练：

| 数据集                   |                           规模 | 特点                                                         |
| ------------------------ | -----------------------------: | ------------------------------------------------------------ |
| **ImageNet-1M / IN-1M**  |                约 128 万张图像 | 类别分布较均衡，图像多为典型物体视角                         |
| **Instagram-1B / IG-1B** | 约 9.4 亿张公开 Instagram 图像 | 更接近真实世界，长尾、不平衡、未充分清洗，包含物体和场景图像 |

这里需要注意：虽然 IN-1M 有类别标签，但 MoCo 预训练阶段**不使用标签**，只使用图像本身。



#### 7.2 训练设置

论文使用 ResNet 作为 backbone，并采用 SGD 优化器。

可以概括如下：

| 设置           |      IN-1M |                           IG-1B |
| -------------- | ---------: | ------------------------------: |
| 优化器         |        SGD |                             SGD |
| weight decay   |     0.0001 |                          0.0001 |
| momentum       |        0.9 |                             0.9 |
| batch size     |        256 |                            1024 |
| GPU 数量       |          8 |                              64 |
| 初始学习率     |       0.03 |                            0.12 |
| 训练时长       | 200 epochs | 1.25M iterations，约 1.4 epochs |
| ResNet-50 时间 | 约 53 小时 |                         约 6 天 |

这说明 MoCo 可以扩展到非常大的无标签数据集，但 IG-1B 训练成本明显更高。



#### 7.3 Linear Classification Protocol

首先，作者使用标准的 **linear classification protocol** 来评估无监督表征质量。

流程是：

1.  在 IN-1M 上进行无监督 MoCo 预训练；
2.  冻结 backbone，不再更新特征提取器；
3.  在 ImageNet 上训练一个监督线性分类器；
4.  使用 ImageNet validation set 的 **1-crop top-1 accuracy** 作为指标。

这个实验的意义是：

如果冻结特征后，只训练一个线性分类器就能取得好结果，说明预训练 backbone 学到的表征本身质量较高。



#### 7.4 消融实验一：对比三种 contrastive loss 机制

![image.png](https://8504cc9c.cloudflare-imgbed-8qo.pages.dev/file/1779269743080_image.png)

论文比较了三种对比学习机制：

| 机制            | 字典来源                 | 优点           | 缺点                             |
| --------------- | ------------------------ | -------------- | -------------------------------- |
| **End-to-end**  | 当前 mini-batch          | key 一致性好   | 字典大小受 batch size 限制       |
| **Memory bank** | 存储历史样本特征         | 字典可以很大   | key 可能来自过旧 encoder，不一致 |
| **MoCo**        | queue + momentum encoder | 字典大且较一致 | 需要维护队列和动量编码器         |

作者控制变量：
三种方法使用相同 pretext task 和相同 InfoNCE loss，只改变 dictionary / key encoder 机制。

实验结论非常关键：

1.  三种机制都受益于更大的 $K$，即更多负样本；
2.  这支持了作者关于 large dictionary 的假设；
3.  end-to-end 方法在 $K$ 小时接近 MoCo，但 $K$ 受 batch size 限制；
4.  memory bank 可以支持大字典，但效果低于 MoCo；
5.  MoCo 在大 $K$ 下表现最好。

根据 Figure 3，ResNet-50 在线性分类下大致表现为：

| 方法        | 最大/典型 $K$ | top-1 accuracy |
| ----------- | ------------: | -------------: |
| End-to-end  |          1024 |          57.3% |
| Memory bank |         65536 |          58.0% |
| **MoCo**    |         65536 |      **60.6%** |

这说明：**仅仅有大字典还不够，字典中的 key 还必须一致**。MoCo 通过 momentum encoder 同时解决了这两个问题。



#### 7.4 消融实验二：动量系数 m 的影响

MoCo 的 key encoder 用动量更新：
$$
\theta_k \leftarrow m\theta_k + (1-m)\theta_q
$$
论文进一步测试不同 m 的影响。

| momentum $m$ |      结果 |
| -----------: | --------: |
|            0 |      fail |
|          0.9 |     55.2% |
|         0.99 |     57.8% |
|        0.999 | **59.0%** |
|       0.9999 |     58.9% |

这个消融实验说明：

-   m=0 时，相当于没有动量，训练 loss 振荡且无法收敛；
-   m=0.9 时，key encoder 变化仍然偏快，效果明显下降；
-   m 在 0.99∼0.9999 区间时表现较好；
-   最佳附近是 m=0.999。

因此，实验强力支持论文的第二个核心假设：

**key encoder 需要缓慢演化，才能让队列中的 dictionary 保持一致。**



#### 7.5 与已有无监督方法比较

论文还将 MoCo 与之前的无监督 / 自监督方法进行比较。

主要结论是：

-   MoCo 使用标准 ResNet-50 就能取得有竞争力的结果；
-   MoCo 不依赖特殊网络结构设计；
-   模型变宽后，性能还能进一步提升；
-   ResNet-50 下 MoCo 达到 60.6% ImageNet linear top-1 accuracy；
-   更宽的 ResNet-50w4× 可以达到更高结果。

这表明 MoCo 的收益主要来自其对比学习机制，而不是依赖复杂 architecture trick。



#### 7.6 下游任务迁移实验

实验部分更重要的结果是迁移学习。论文强调，无监督预训练真正的目标是学到能迁移到下游任务的通用表征，而不仅是 ImageNet 线性分类准确率。

MoCo 被迁移到多个检测和分割任务中，例如：

-   PASCAL VOC object detection；
-   COCO object detection；
-   COCO instance segmentation；
-   keypoint detection；
-   semantic segmentation 等。

论文结论是：MoCo 在多个检测 / 分割任务上可以达到甚至超过 ImageNet supervised pre-training。

从补充实验中也可以看到，MoCo 相比随机初始化有明显提升，并接近监督预训练。

这里的核心含义是：

-   MoCo 无监督预训练比随机初始化更好；
-   MoCo 与 ImageNet 监督预训练接近；
-   在某些检测 / 分割任务上甚至优于监督预训练；
-   这说明 MoCo 学到的表征更具有通用迁移能力。





