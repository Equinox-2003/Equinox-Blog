---
title: "Word Embedding"
description: ""
date: 2026-04-28T15:04:20+08:00
lastmod: 2026-04-28T15:04:20+08:00
draft: false

categories:
  - MachineLearning
tags:
  - Unsuperviesed Learning

toc: true
math: true
mermaid: true
cover: true
---

<!--more-->



## 一、word-Embedding



### 1.1 为什么 one-hot vector 是一个糟糕的选择

之前在学习逻辑回归的时候，采用 one-hot vector 来进行建模，分类。

独热编码非常易于构建，但它却很难去精准表达不同词之间的相似度。

比如我们如果用向量来表示 "queen" 和 "king"，那么我们很可能希望得到 `"queen" - "king" = "woman"`



### 1.2 自监督的word2vec

word2vec 有两个模型：**跳元模型（skip-gram）**和**连续词袋（CBOW）**。

对于在语义上有意义的表示，它们的训练依赖于**条件概率**，条件概率可以被看作使用语料库中一些词来预测另一些单词。由于是**不带标签**的数据，因此跳元模型和连续词袋都是自监督模型。



#### 1.2.1 跳元模型（Skip-Gram）

>   用中心词预测上下文。
>

Skip-Gram假设**一个词可以用来在文本序列中生成其周围的单词**。以文本序列“the”“man”“loves”“his”“son”为例。

假设中心词选“loves”，并将上下文窗口设置成2，

![../_images/skip-gram.svg](https://zh-v2.d2l.ai/_images/skip-gram.svg)

Skip-Gram考虑生成上下文词“the”“man”“him”“son”的条件概率：
$$
P(\textrm{"the"},\textrm{"man"},\textrm{"his"},\textrm{"son"}\mid\textrm{"loves"}).
$$
假设上下文词是在给定中心词的情况下独立生成的（即条件独立性）。在这种情况下，上述条件概率可以重写为：
$$
P(\textrm{"the"}\mid\textrm{"loves"})\cdot P(\textrm{"man"}\mid\textrm{"loves"})\cdot P(\textrm{"his"}\mid\textrm{"loves"})\cdot P(\textrm{"son"}\mid\textrm{"loves"}).
$$
在Skip-Gram中，每个词都有两个 d 维向量表示，用于计算条件概率。

换句话说，索引为 i 的任何词，分别用 $\mathbf{v}_i\in\mathbb{R}^d$和$\mathbf{u}_i\in\mathbb{R}^d$表示其用作**中心词**和**上下文词**时的两个向量。

给定中心词$w_c$（词典中的索引c），生成任何上下文$w_o$（词典中的索引c）的条件概率可以通过向量点积的softmax操作来建模：
$$
P(w_o \mid w_c) = \frac{\text{exp}(\mathbf{u}_o^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)},
$$
假定词表索引集$\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$。给定长度为$T$的文本序列，其中时间步$t$处的词表示为$w^{(t)}$。

假设上下文词是在给定任何中心词的情况下**独立**生成的。

对于上下文窗口为m，Skip-Gram的似然函数是在**给定任何中心词的情况下生成所有上下文词的概率**：
$$
\prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)}),
$$

##### 1.2.1.1 训练

Skip-Gram参数是词表中每个词的中心词向量和上下文词向量。在训练中，我们通过最大化似然函数（**即极大似然估计**）来学习模型参数。这相当于最小化以下损失函数：
$$
- \sum_{t=1}^{T} \sum_{-m \leq j \leq m,\ j \neq 0} \text{log}\, P(w^{(t+j)} \mid w^{(t)}).
$$
当使用随机梯度下降来最小化损失时，在每次迭代中可以随机抽样一个较短的子序列来计算该子序列的（随机）梯度，以更新模型参数。

我们对前面条件概率的公式两边取对数可以得到：
$$
\log P(w_o \mid w_c) =\mathbf{u}_o^\top \mathbf{v}_c - \log\left(\sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)\right).
$$
通过微分，我们可以获得其相对于中心词向量$\mathbf{v}_c$的梯度为
$$
\begin{split}\begin{aligned}\frac{\partial \text{log}\, P(w_o \mid w_c)}{\partial \mathbf{v}_c}&= \mathbf{u}_o - \frac{\sum_{j \in \mathcal{V}} \exp(\mathbf{u}_j^\top \mathbf{v}_c)\mathbf{u}_j}{\sum_{i \in \mathcal{V}} \exp(\mathbf{u}_i^\top \mathbf{v}_c)}\\&= \mathbf{u}_o - \sum_{j \in \mathcal{V}} \left(\frac{\text{exp}(\mathbf{u}_j^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)}\right) \mathbf{u}_j\\&= \mathbf{u}_o - \sum_{j \in \mathcal{V}} P(w_j \mid w_c) \mathbf{u}_j.\end{aligned}\end{split}
$$
从公式可以看出，只要有了以$w_c$为中心词的所有词的条件概率，其他词向量的梯度就可以以相同的方式获得。



#### 1.2.2 连续词袋（CBOW）模型

>   用上下文预测中心词。

连续词袋（CBOW）模型类似于Skip-Gram。与Skip-Gram的主要区别在于，CBOW模型假设中心词是基于其在文本序列中的周围上下文词生成的。例如，在文本序列“the”“man”“loves”“his”“son”中，在“loves”为中心词且上下文窗口为2的情况下，连续词袋模型考虑基于上下文词“the”“man”“him”“son”生成中心词“loves”的条件概率，即：

![../_images/cbow.svg](https://zh-v2.d2l.ai/_images/cbow.svg)

由于CBOW模型中存在多个上下文词，**因此在计算条件概率时对这些上下文词向量进行平均**。具体地说，对于字典中索引$i$的任意词，分别用$\mathbf{v}_i\in\mathbb{R}^d$和$\mathbf{u}_i\in\mathbb{R}^d$表示用作上下文词和中心词的两个向量（符号和 Skip-Gram 中相反）。给定上下文词$w_{o_1}, \ldots, w_{o_{2m}}$，（在词表中索引是$o_1, \ldots, o_{2m}$）生成任意中心词$w_c$（在词表中索引是$c$）的条件概率可以由以下公式建模:
$$
P(w_c \mid w_{o_1}, \ldots, w_{o_{2m}}) = \frac{\text{exp}\left(\frac{1}{2m}\mathbf{u}_c^\top (\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}}) \right)}{ \sum_{i \in \mathcal{V}} \text{exp}\left(\frac{1}{2m}\mathbf{u}_i^\top (\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}}) \right)}.
$$
为了简洁起见，我们设为$\mathcal{W}_o= \{w_{o_1}, \ldots, w_{o_{2m}}\}$和$\bar{\mathbf{v}}_o = \left(\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}} \right)/(2m)$。那么上式可以简化为：
$$
P(w_c \mid \mathcal{W}_o) = \frac{\exp\left(\mathbf{u}_c^\top \bar{\mathbf{v}}_o\right)}{\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)}.
$$
给定长度为$T$的文本序列，其中时间步$t$处的词表示为$w^{(t)}$。对于上下文窗口$m$，CBOW模型的似然函数是在给定其上下文词的情况下生成所有中心词的概率：
$$
\prod_{t=1}^{T}  P(w^{(t)} \mid  w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)}).
$$

##### 1.2.2.1 训练

CBOW模型的最大似然估计等价于最小化以下损失函数：
$$
-\sum_{t=1}^T  \text{log}\, P(w^{(t)} \mid  w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)}).
$$
请注意：
$$
\log\,P(w_c \mid \mathcal{W}_o) = \mathbf{u}_c^\top \bar{\mathbf{v}}_o - \log\,\left(\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)\right).
$$
通过微分，我们可以获得其关于任意上下文词向量的梯度$i = 1, \ldots, 2m$，如下：
$$
\frac{\partial \log\, P(w_c \mid \mathcal{W}_o)}{\partial \mathbf{v}_{o_i}} = \frac{1}{2m} \left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} \frac{\exp(\mathbf{u}_j^\top \bar{\mathbf{v}}_o)\mathbf{u}_j}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \bar{\mathbf{v}}_o)} \right) = \frac{1}{2m}\left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} P(w_j \mid \mathcal{W}_o) \mathbf{u}_j \right).
$$

#### 1.2.3 architecture

上面两种方法都有各自的function，那么只需要构建一个模型，进行训练即可。

比较有趣的是 word2vec 只用了一个单隐藏层的神经网络：

```python
输入层 -> 隐藏层 -> 输出层
```

其中：

-   输入层：一个词的 one-hot 向量
-   隐藏层：词向量层（没有复杂激活）
-   输出层：预测上下文词的概率分布

而我们需要的，正是隐藏层输出的词向量层。

**而隐藏层的权重矩阵是什么形式呢**？
$$
\mathbf{W}\in\mathbb{R}^{V\times d}
$$

-   V = 词表大小
-   d = 向量维度

而输入又是 one-hot编码，所以，输入 乘 权重矩阵就是 **查表取向量**。

**为什么训练快？**

因为它网络很浅：

-   没有很多层
-   没有复杂非线性堆叠
-   输入是 one-hot
-   隐藏层本质上是 embedding lookup

但如果直接 softmax 整个词表，仍然会很慢，特别是词表很大时。

所以 Word2Vec 常用两个优化技巧：

**1、Hierarchical Softmax**

用一棵二叉树表示词表，把原来对整个词表的 softmax 计算，变成沿树路径计算，降低复杂度。

普通 softmax 在预测一个词时，要对整个词表算一遍概率。如果词表有 100 万个词，那么每次预测都要和 100 万个词比较，开销很大。

**Hierarchical Softmax 的思路是：**

-   不直接在所有词里“选一个词”，  
-   而是在一棵二叉树里，从根节点一路做“向左还是向右”的二分类决策，  
-   最后走到某个叶子节点，对应一个词。

那么当树是平衡的时候，我们只需要 **O(log)** 的复杂度即可完成比较。

而这颗树的构建，一种比较常见的做法就是建立**Huffman Tree（哈夫曼树）**，因为高频词会经常预测，而哈夫曼树可以使得频次加权路径最短。

总结一下就是，**Hierarchical Softmax**把传统的对整张词表计算概率，转化成了树上的多次二分类。

**模型在每个分叉点输出往左走还是往右走，最后的叶子节点就是预测输出。**



**2、Negative Sampling**

思想：模型只需要学会谁和谁不相关。

比如：

```
("吃", "苹果")
```

是正样本

随机采样几个负样本：

```
("吃", "汽车")
("吃", "银行")
("吃", "天气")
```

训练时模型只需要学会：

-   “吃”和“苹果”应该接近
-   “吃”和“汽车”不应该接近

这样就不用每次都和整个词表比，大大加快训练。



### 1.4 一个简单的代码示例

>   因为手动实现想要达到特别好的效果的话，需要很多的tips。
>
>   所以还是用工具库来做一个示例。

做一个超简单的情感分类：

-   训练 Word2Vec
-   把句子变成向量
-   训练一个分类器
-   预测新句子

#### 1.4.1 库

```python
from gensim.models import Word2Vec
import numpy as np
from sklearn.linear_model import LogisticRegression
```

#### 1.4.2  数据

```python
# data, 已经分词
sentences = [
    ["质量", "很好", "物流", "很快"],
    ["商品", "不错", "价格", "实惠"],
    ["体验", "很好", "下次", "还会", "购买"],
    ["包装", "破损", "非常", "失望"],
    ["物流", "太慢", "客服", "态度", "差"],
    ["产品", "不好", "质量", "很差"]
]

# label，1: positive, 2: negative
labels = np.array([1, 1, 1, 0, 0, 0])
```

#### 1.4.3 Model

```python
# word2vec model
model = Word2Vec(
    sentences = sentences,
    vector_size=50, # dim of word vector
    window=2, # size of context window
    min_count=1, # min_freq
    workers=1,
    sg=1 # 1 = skip-gram, 0 = cbow
)
```

#### 1.4.4 seq2vec

```python
# 把一句话转成向量：对句子中所有词向量取平均
def sentence_vector(words, model):
    vectors = []
    for word in words:
        if word in model.wv:
            vectors.append(model.wv[word])
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)
```

#### 1.4.5 构建训练特征

```python
X = np.array([sentence_vector(s, model) for s in sentences])
y = labels
```

#### 1.4.6 训练并测试

```python
# classifier
clf = LogisticRegression()
clf.fit(X, y)

# test new sentences
test_sentences = [["物流", "很快", "商品", "不错"], ["质量", "很好", "商品", "不错"], ["物流", "不好", "商品", "很差"]]
for word in test_sentences:
    test_vec = sentence_vector(word, model).reshape(1, -1)
    pred = clf.predict(test_vec)[0]
    print("预测结果：", "正面" if pred == 1 else "负面")
```

**输出：**

```tex
预测结果： 正面
预测结果： 正面
预测结果： 负面
```

































































































